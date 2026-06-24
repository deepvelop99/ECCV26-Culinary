"""Replay-render an MPM episode through the original MPMCuttingSim + MPMRenderer.

Pipeline:
  - subprocess boots a fresh Taichi CUDA context
  - instantiates MPMCuttingSim with the fruit's yaml (same SDFs as collect did)
  - seeds particles exactly like the collect run
  - for each control tick in the recorded trajectory:
        * override sim.knife.y / knife.z_off to the recorded tip_mpm pose
        * step the sim the same number of substeps the collect run used
        * draw with MPMRenderer (scene.particles + knife proxy + board proxy +
          damage coloring + auto camera) into an offscreen Taichi GGUI window
        * save_image(...) each frame
  - ffmpeg concat PNGs -> mp4

This reproduces the Chayoso run.py visual style since it uses the identical
renderer, with our recorded knife motion in place of the default spline.

Usage:
    python render_mpm_episode.py \
        --npz knife_traj.npz \
        --config /data/EEF-Cutting-Simulation/configs/apple.yaml \
        --out mpm.mp4
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml


def _pack(block: dict, with_blade: bool):
    from sdf_utils.mesh_sdf import mesh_to_sdf
    transform = block.get("initial_transform", {})
    kwargs = {}
    if with_blade:
        kwargs["knife_blade"] = block.get("blade", {"axis": "Y", "fraction": 0.5})
    return mesh_to_sdf(
        block["mesh_path"], transform, int(block["sdf_voxel"]), **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="knife trajectory .npz (tip_mpm per tick)")
    ap.add_argument("--config", required=True, help="fruit MPM yaml config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--res", type=int, default=720)
    ap.add_argument("--mpm-substeps-cap", type=int, default=100)
    ap.add_argument("--control-dt", type=float, default=0.05)
    ap.add_argument("--extra-descent", type=float, default=0.0,
                    help="shift MPM knife tip Y down by this amount during the "
                         "cut phase so the blade SDF fully passes through the "
                         "particle cluster (compensates for MS robot's limited "
                         "vertical reach into the board)")
    args = ap.parse_args()

    # Load knife trajectory
    data = np.load(args.npz, allow_pickle=False)
    knife_traj = data["knife"].astype(np.float32)  # (T, 3) in MPM frame
    T = knife_traj.shape[0]
    print(f"[render] knife trajectory T={T}", flush=True)

    # Pre-read cfg so we can pick the matching eef_repo before chdir.
    with open(args.config) as f:
        _cfg_pre = yaml.safe_load(f)

    # EEF repo on path (user cloned fresh copy here). Prefer the candidate
    # whose asset tree matches the cfg yaml's mesh_path style. addfruits-pilot
    # cfgs (Cutting_rubuttal2/Cutting_rebuttal) reference assets/fruit2/*.obj;
    # legacy cfgs (EEF-Cutting-Simulation) reference assets/fruits/*.obj.
    _candidates = [
        "/data/Cutting_rubuttal2",
        "/data/Cutting_rebuttal",
        "/data/mani_skill/EEF-Cutting-Simulation",
        "/data/EEF-Cutting-Simulation",
    ]
    _cfg_mesh_path = str(_cfg_pre.get("cutting_mesh", {}).get("mesh_path", ""))
    _needed_subdir = "assets/fruit2" if "fruit2" in _cfg_mesh_path else "assets/fruits"
    eef_repo = next(
        (c for c in _candidates if os.path.isdir(os.path.join(c, _needed_subdir))),
        _candidates[-1],
    )
    print(f"[render] eef_repo={eef_repo} (cfg expects {_needed_subdir})", flush=True)
    if eef_repo not in sys.path:
        sys.path.insert(0, eef_repo)
    os.chdir(eef_repo)

    # Taichi init (fresh subprocess, no sapien conflict)
    import taichi as ti
    try:
        ti.init(arch=ti.cuda, random_seed=1)
    except Exception:
        ti.init(arch=ti.vulkan)
    print("[render] taichi inited", flush=True)

    # Load MPM config + build SDF packs
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cutting_pack = _pack(cfg["cutting_mesh"], with_blade=False)
    knife_pack = _pack(cfg["knife"], with_blade=True)
    board_pack = _pack(cfg["board"], with_blade=False) if "board" in cfg else None
    cfg["cutting_mesh_pack"] = cutting_pack

    # Build sim with viewer DISABLED (otherwise MPMCuttingSim opens GLFW which
    # fails in the headless pod). Then attach a headless Taichi GGUI window
    # manually so MPMRenderer._draw() has a canvas/scene to render into.
    from mpmcore.sim import MPMCuttingSim
    sim = MPMCuttingSim(
        cfg, cutting_pack, knife_pack, board_pack=board_pack,
        viewer=False,
    )
    # Re-enable the flags MPMRenderer checks
    sim.viewer_enabled = True
    sim.viewer_camera_mode = "auto"
    sim.viewer_lock_on_run = True
    sim.window = ti.ui.Window("MPM Cutting", res=(args.res, args.res),
                              vsync=False, show_window=False)
    sim.canvas = sim.window.get_canvas()
    sim.scene = sim.window.get_scene()
    sim.camera = ti.ui.Camera()

    # Seed particles (same support clamp as bridge)
    top = None
    if sim.board is not None:
        try:
            top = float(sim.board_top_y[None])
        except Exception:
            top = None
    sim.seed_particles_from_mesh(
        cutting_pack, cfg["cutting_mesh"], name="cutting_mesh",
        support_top_y=top)

    # Replace knife.update with no-op (we drive it externally)
    def _external_update(dt):
        sim.knife.sim_time = float(getattr(sim.knife, "sim_time", 0.0)) + float(dt)
    sim.knife.update = _external_update

    # Camera: MPMRenderer auto preset
    sim.camera.position(0.40, 0.35, 0.45)
    sim.camera.lookat(-0.02, 0.08, 0.00)
    sim.camera.up(0, 1, 0)

    # MPM knife drive helpers. The MS robot can only reach down to ~MPM-y=0.09
    # (physical arm/board limit) even though fruits go down to ~y=0.05, so the
    # MPM blade SDF would leave the bottom half uncut. We add `extra_descent`
    # to knife.y but blend it in smoothly based on knife depth — zero above
    # y=blend_hi, fully applied below y=blend_lo.
    blend_hi = 0.20   # tip above this → no extra (approach / retreat)
    blend_lo = 0.11   # tip at or below → full extra (near fruit)

    def drive_knife_to(tip_mpm):
        yfoot = float(getattr(sim, "_knife_yfoot", 0.0))
        y = float(tip_mpm[1])
        t = (blend_hi - y) / max(1e-6, (blend_hi - blend_lo))
        t = max(0.0, min(1.0, t))
        # smoothstep 3t² − 2t³
        s = t * t * (3.0 - 2.0 * t)
        extra = float(args.extra_descent) * s
        sim.knife.y[None] = y - yfoot - extra
        if hasattr(sim.knife, "z_off"):
            sim.knife.z_off[None] = float(tip_mpm[2])

    # MPMRenderer (lazy import to ensure sim attributes exist)
    from render_utils.renderer import MPMRenderer
    renderer = MPMRenderer(sim)

    # Substeps per control tick — same as bridge
    dt_frame = float(sim.dt) * int(sim.substeps)
    n_sub = max(1, int(round(args.control_dt / dt_frame)))
    if args.mpm_substeps_cap > 0:
        n_sub = min(n_sub, args.mpm_substeps_cap)
    print(f"[render] substeps_per_control={n_sub}", flush=True)

    # Cutting_rubuttal2 MPMCuttingSim does not expose `sim_time`; track it
    # ourselves and pass into sim.step(T).
    _mpm_sim_time = 0.0
    _dt_frame = float(sim.dt) * int(sim.substeps)

    tmp_dir = tempfile.mkdtemp(prefix="mpm_render_")
    try:
        for t in range(T):
            drive_knife_to(knife_traj[t])
            for _ in range(n_sub):
                _t_arg = getattr(sim, "sim_time", _mpm_sim_time)
                sim.step(_t_arg)
                _mpm_sim_time += _dt_frame
            # Expanded MPMRenderer._draw — skip window.show() (segfaults in
            # headless) and the GUI overlay (requires visible window).
            try:
                sim.scene.set_camera(sim.camera)
                renderer._setup_scene_lighting()
                renderer._render_particles()
                renderer._render_knife_proxy()
                renderer._render_board_proxy()
                renderer._render_eef_position()
                if getattr(sim, "show_grid", False):
                    renderer._render_grid_visualization()
                sim.canvas.scene(sim.scene)
            except Exception as e:
                print(f"[render] frame {t} draw err: {e}", flush=True)
                continue
            sim.window.save_image(f"{tmp_dir}/f{t:05d}.png")
            if t == 0:
                print(f"[render] frame 0 saved", flush=True)

        # Concat
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = "ffmpeg"
        cmd = [
            ffmpeg, "-y", "-framerate", str(args.fps),
            "-i", f"{tmp_dir}/f%05d.png",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            args.out,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-500:], file=sys.stderr)
            sys.exit(3)
        print(f"[render] wrote {args.out}", flush=True)
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

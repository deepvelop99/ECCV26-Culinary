"""ManiSkill ⇄ MPM co-simulation bridge.

MPM runs as a slave: its knife pose is driven by the ManiSkill panda's knife
link world pose each control step. Both sims log force/velocity for per-step
alignment checks (success criterion per CulinaryCut paper §3.2).

Design:
    - The user's bananacut env uses a fixed knife link `tool_knife`; we read
      `link.pose` and `link.get_net_contact_forces()` each control step.
    - EEF-Cutting-Simulation's `KnifeCollider.update(dt)` advances the knife
      internally. We monkey-patch it with `_external_update(dt)` that
      consumes an externally-supplied target y/z_off (and optional rotation
      slots), skipping the internal Y schedule.
    - MPM dt is ~1e-5 with 6 substeps/frame. Per ManiSkill control tick
      (20Hz → 0.05s), MPM runs ceil(0.05 / (dt*substeps)) ≈ 833 substeps.

This bridge does *not* touch sapien's physics; it only reads poses and writes
into MPM's knife collider fields at control-tick boundaries, interpolating the
knife position linearly within the MPM substep loop.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import math
import sys

import numpy as np

from .frame import (
    FrameCalibration,
    ms_knife_tip_to_mpm,
    mpm_knife_y_from_tip_y,
)


# EEF-Cutting-Simulation is not a pip package. Importing requires it on sys.path.
EEF_REPO = Path("/data/EEF-Cutting-Simulation")
if str(EEF_REPO) not in sys.path:
    sys.path.insert(0, str(EEF_REPO))


def _damage_color(dmg_arr, dmax: float):
    """Map damage scalar to color: tan (healthy) -> red (fully damaged)."""
    import numpy as np
    d = np.clip(np.asarray(dmg_arr, np.float32) / max(1e-6, float(dmax)), 0.0, 1.0)
    # Linear interp from #D9A463 (218,164,99) to #B83330 (184,51,48)
    r = (218 * (1 - d) + 184 * d) / 255.0
    g = (164 * (1 - d) + 51 * d) / 255.0
    b = (99  * (1 - d) + 48 * d) / 255.0
    return np.stack([r, g, b], axis=1)


@dataclass
class BridgeConfig:
    mpm_config_yaml: str                      # e.g. /data/EEF-Cutting-Simulation/configs/banana.yaml
    control_freq_hz: float = 20.0             # ManiSkill control freq
    calibration: FrameCalibration = None      # None = default constants
    headless: bool = True                     # no taichi UI
    mpm_substeps_cap: int = 0                 # 0 = use dt_control/dt_frame; >0 caps it
                                              # (sanity runs: set e.g. 50 to speed up)
    record_particles: bool = False            # store (sub-sampled) particle pos per
                                              # snapshot so save_video() can emit an
                                              # MPM rendering alongside the MS mp4
    particle_subsample: int = 8000            # random subset size per frame
    render_every_n_ticks: int = 1             # snapshot cadence (every tick =
                                              # time-aligned with MS mp4)
    # Tier 2: feed MPM reaction force back to ManiSkill so knife descent becomes
    # material-dependent. When enabled the bridge calls knife_link.add_force_at_point
    # with (-F_mpm_in_ms_frame) every control tick. Use mpm_force_scale to damp
    # spikes (MPM can emit >1000N which, applied to a 0.5kg knife, destabilises
    # ManiSkill physics). scale=1.0 is physically faithful but may be unstable;
    # 0.1–0.3 gives a material-differentiating effect while staying stable.
    apply_mpm_force_to_ms: bool = False
    mpm_force_scale: float = 0.25
    mpm_force_clip: float = 200.0             # N, clamp magnitude before apply
    # Override Chayoso yaml's cutting_speed_mps (default 0.6 m/s, unrealistic
    # for controlled cutting). Applied in setup after sim is loaded.
    knife_speed_mps: float = 0.10
    # MPM-only knife depth offset (m). MS commanded tip is floored above the
    # MS board to avoid rigid-contact lifting, so MS_tip never reaches fruit
    # bottom. We push the MPM knife.y deeper than MS_tip + v_offset by this
    # amount so MPM blade traverses the FULL fruit height. The offset breaks
    # MS↔MPM strict coordinate match by design — MPM-side particle physics
    # is what produces the meaningful cut + force label.
    mpm_extra_descent_y: float = 0.000


class ManiSkillMPMBridge:
    """Couples a ManiSkill BaseEnv with an MPMCuttingSim.

    Usage:
        bridge = ManiSkillMPMBridge(cfg)
        bridge.setup(env)                     # reads knife link pose 0
        for t in range(T):
            action = ...
            obs, r, term, trunc, info = env.step(action)
            telemetry = bridge.step_with_env(env, dt_control=0.05)
            # telemetry: {"ms_tcp_vel", "ms_force", "mpm_vel", "mpm_force", ...}
        bridge.close()
    """

    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.calib = cfg.calibration or FrameCalibration()
        self.sim = None
        self.packs = None
        self._taichi_inited = False
        self._prev_tcp_pose = None
        self._prev_tip_mpm = None
        # Banana pose in the current episode (ManiSkill world).
        self._banana_xy_ms = np.zeros(2, np.float32)
        self._banana_yaw_ms = 0.0
        # Per-episode rendering buffer. List of dicts {parts, knife_mpm}.
        self._render_buf = []
        self._particle_sample_idx = None  # chosen once per episode, stable subset
        self._render_tick_counter = 0

    # ---------- setup ----------
    def _init_taichi(self):
        import taichi as ti
        # Prefer CUDA; fall back through vulkan → cpu.
        try:
            ti.init(arch=ti.cuda)
        except Exception:
            try:
                ti.init(arch=ti.vulkan)
            except Exception:
                ti.init(arch=ti.cpu)
        self._taichi_inited = True

    def _load_mpm(self):
        import yaml
        from sdf_utils.mesh_sdf import mesh_to_sdf
        from mpmcore.sim import MPMCuttingSim

        with open(self.cfg.mpm_config_yaml) as f:
            mpm_cfg = yaml.safe_load(f)

        # Build SDF packs (no cache layer here; extend later if needed).
        def _pack(block, with_blade):
            transform = block.get("initial_transform", {})
            kwargs = {}
            if with_blade:
                kwargs["knife_blade"] = block.get("blade", {"axis": "Y", "fraction": 0.5})
            return mesh_to_sdf(
                block["mesh_path"],
                transform,
                int(block["sdf_voxel"]),
                **kwargs,
            )

        cutting_pack = _pack(mpm_cfg["cutting_mesh"], with_blade=False)
        knife_pack = _pack(mpm_cfg["knife"], with_blade=True)
        board_pack = _pack(mpm_cfg["board"], with_blade=False) if "board" in mpm_cfg else None

        self.packs = dict(cutting=cutting_pack, knife=knife_pack, board=board_pack)
        # run.py injects cutting_mesh_pack into cfg before sim.run() uses it.
        mpm_cfg["cutting_mesh_pack"] = cutting_pack
        self.sim = MPMCuttingSim(
            mpm_cfg, cutting_pack, knife_pack, board_pack=board_pack,
            viewer=False,  # headless: no GLFW / taichi UI
        )

        # Seed cutting-mesh particles above the board (replicates sim.run() prelude).
        top = None
        if self.sim.board is not None:
            try:
                top = float(self.sim.board_top_y[None])
            except Exception:
                top = None
        self.sim.seed_particles_from_mesh(
            cutting_pack, mpm_cfg["cutting_mesh"], name="cutting_mesh",
            support_top_y=top,
        )

        # Do NOT override knife motion — let Chayoso MPM drive the knife
        # natively (adaptive speed, cut_tau decay, etc). We only set the
        # starting position so it lines up with MS's knife-tip height.

    def _override_knife_motion(self):
        """Replace KnifeCollider.update with a no-op; bridge writes y/z_off/quat."""
        knife = self.sim.knife
        self._orig_knife_update = knife.update

        def _external_update(dt):
            # Accumulate sim_time for saw oscillation etc.
            knife.sim_time = float(getattr(knife, "sim_time", 0.0)) + float(dt)
        knife.update = _external_update

    def setup(self, env, *, banana_xy_ms=None, banana_yaw_ms: float = 0.0):
        """Call after env.reset(). Reads the knife link and records banana pose
        so knife motion can be mapped into MPM's banana-local canonical frame.
        """
        if not self._taichi_inited:
            self._init_taichi()
        self._load_mpm()
        u = env.unwrapped
        knife_link = getattr(u, "knife_link", None)
        assert knife_link is not None, "env must expose knife_link (bananacut env does)"
        self._knife_link = knife_link
        self._prev_tcp_pose = None
        self._prev_tip_mpm = None

        # Record banana pose (ManiSkill xy + yaw about z) for banana-local mapping.
        if banana_xy_ms is not None:
            self._banana_xy_ms = np.asarray(banana_xy_ms, np.float32).reshape(2)
        else:
            block = getattr(u, "block_apple", [None])[0]
            if block is not None:
                p = np.asarray(block.pose.p, np.float32).reshape(-1)
                self._banana_xy_ms = p[:2].copy()
        self._banana_yaw_ms = float(banana_yaw_ms)

        # Precompute knife foot offset so we can turn MPM tip_y → knife.y[None].
        self._knife_yfoot = float(self.sim._knife_yfoot) if hasattr(
            self.sim, "_knife_yfoot") else 0.0

        # MS sends position only; MPM owns its own dynamics during cut. Bridge
        # writes knife.y from MS_tip while MS-driven; on contact (F>threshold)
        # MPM takes over its native descent (cut_tau decay etc.) and bridge
        # stashes the MPM velocity for MS to use as commanded velocity (b/c
        # injection). We do NOT override knife.update — MPM uses its native one.
        self._ms_driven_knife = True
        self._in_contact = False
        self._contact_threshold_N = 1.0

        # Align MPM knife.y with ManiSkill's current knife-tip height before
        # the first step (cosmetic — first step_with_env will overwrite anyway).
        try:
            tip_ms_now = u._get_tip_from_eef()[0].detach().cpu().numpy().astype(np.float32)
            start_tip_y_mpm = float(tip_ms_now[2]) + float(self.calib.v_offset)
            self.sim.knife.y[None] = start_tip_y_mpm - self._knife_yfoot
        except Exception as e:
            print(f"[bridge] knife.y init skipped: {e}")

        # knife_speed_mps is unused under (B) but kept for backwards-compat:
        try:
            v = float(self.cfg.knife_speed_mps)
            knife = self.sim.knife
            knife.speed = v
            knife.base_speed = v
            knife.current_speed[None] = v
            knife.min_speed_f[None] = 0.10 * v
            print(f"[bridge] knife speed overridden to {v} m/s")
        except Exception as e:
            print(f"[bridge] knife speed override skipped: {e}")

    # ---------- step ----------
    def step_with_env(self, env, dt_control: float) -> Dict[str, np.ndarray]:
        """Advance MPM to track ManiSkill knife for one control tick.

        Motion mapping:
            1. Read ManiSkill knife TIP world pose (via env helper).
            2. Map TIP into banana-local frame, then into MPM world via
               FrameCalibration (banana_xy cancellation + vertical offset).
            3. Set sim.knife.y[None] = tip_y_mpm - sim._knife_yfoot.
        """
        u = env.unwrapped

        # --- ManiSkill side ---
        # Use env helper for knife tip (computed via URDF hand→knife + STL tip z).
        try:
            tip_ms = u._get_tip_from_eef()[0].detach().cpu().numpy().astype(np.float32)
        except Exception:
            # Fallback: use knife link origin
            knife_pose = self._knife_link.pose
            tip_ms = np.asarray(knife_pose.p, np.float32).reshape(-1, 3)[-1]
        q_ms = np.asarray(self._knife_link.pose.q, np.float32).reshape(-1, 4)[-1]
        tcp_vel = self._tcp_linear_velocity(env, tip_ms, dt_control)
        f_link = self._knife_link.get_net_contact_forces()
        if hasattr(f_link, "cpu"):
            f_link = f_link.cpu().numpy()
        f_link = np.asarray(f_link, np.float32).reshape(-1, 3)[-1]

        # --- ManiSkill → MPM frame ---
        tip_mpm = ms_knife_tip_to_mpm(
            tip_ms=tip_ms,
            banana_pose_ms_xy=self._banana_xy_ms,
            banana_yaw_ms=self._banana_yaw_ms,
            canonical_xz=np.asarray(self.calib.mpm_banana_canonical_xz, np.float32),
            v_offset=self.calib.v_offset,
        )

        # xy: always follow MS (MS owns lateral position).
        if hasattr(self.sim.knife, "z_off"):
            z_canon = float(self.calib.mpm_banana_canonical_xz[1])
            self.sim.knife.z_off[None] = float(tip_mpm[2] - z_canon)
        # y: MS-driven only when _ms_driven_knife (approach + retreat). During
        # cut (post-contact, _ms_driven=False) MPM owns y via native update.
        if self._ms_driven_knife:
            extra = float(getattr(self.cfg, "mpm_extra_descent_y", 0.0))
            self.sim.knife.y[None] = (
                float(tip_ms[2]) + float(self.calib.v_offset)
                - self._knife_yfoot - extra
            )

        # --- Step MPM (MPM drives its own knife.y via KnifeCollider.update) ---
        mpm_substeps = self._mpm_substeps_per_control(dt_control)
        for _ in range(mpm_substeps):
            self.sim.step(self.sim.sim_time)

        # After MPM step, read the MPM-driven knife y so collect can command MS
        # robot to follow it.
        mpm_knife_tip_y = float(self.sim.knife.y[None] + self._knife_yfoot)
        tip_mpm_driven = np.array(
            [tip_mpm[0], mpm_knife_tip_y, tip_mpm[2]], dtype=np.float32)

        # --- Optional rendering snapshot (subsampled particles + knife pose) ---
        if getattr(self.cfg, "record_particles", False):
            cadence = max(1, int(getattr(self.cfg, "render_every_n_ticks", 5)))
            if (len(self._render_buf) == 0) or (self._render_tick_counter % cadence == 0):
                self._snapshot_for_render(tip_mpm_driven)
            self._render_tick_counter += 1

        # --- MPM side ---
        f_mpm = self._read_mpm_force()
        # Contact-triggered phase: pre-contact MS drives MPM via knife.y write
        # (above). On first F>threshold, hand off to MPM-driven cut: MPM uses
        # its native descent (cut_tau decay etc.) for the cut phase, and bridge
        # stashes MPM velocity onto env so collect script can command MS to
        # follow (option b — soft velocity injection).
        if (not self._in_contact) and float(np.linalg.norm(f_mpm)) > self._contact_threshold_N:
            self._in_contact = True
            self._ms_driven_knife = False
            try:
                cur_y = float(self.sim.knife.y[None])
                self.sim.knife.start_y = cur_y
                try:
                    board_top_y_mpm = float(self.sim.board_top_y[None])
                except Exception:
                    board_top_y_mpm = 0.0464
                stop_tip_y = board_top_y_mpm + 0.005  # 5 mm above MPM board
                self.sim.knife.stop_y = stop_tip_y - self._knife_yfoot
                self.sim.knife._down[None] = 1
                self.sim.knife.current_speed[None] = float(self.cfg.knife_speed_mps)
            except Exception as e:
                print(f"[bridge] contact-transition failed: {e}")
        # Stash MPM-side force AND velocity on env so MS-side step / collect
        # script can use them. Force shows in [Step Debugeval]; velocity is
        # used by collect cut-phase to command MS dpos = v_mpm * dt.
        try:
            u._last_mpm_force = np.asarray(f_mpm, np.float32).reshape(-1)[:3]
        except Exception:
            pass
        # Method B: apply MPM reaction force to MS knife_link.
        if getattr(self.cfg, "apply_mpm_force_to_ms", False):
            self._apply_mpm_force_to_ms(f_mpm, tip_ms)
        # Always finite-diff our own tip_mpm track for velocity (ignore sim's
        # _ee_velocity which bleeds saw-oscillation x-component). Use the
        # MPM-DRIVEN knife position so velocity reflects MPM's native descent
        # (with adaptive speed / cut_tau) — not MS scripted motion.
        if self._prev_tip_mpm is not None:
            v_mpm = ((tip_mpm_driven - self._prev_tip_mpm) / float(dt_control)).astype(np.float32)
        else:
            v_mpm = np.zeros(3, np.float32)
        self._prev_tip_mpm = tip_mpm_driven.copy()
        # Stash MPM velocity (in MPM frame, y-up) on env so collect script can
        # use it as commanded velocity for MS during the cut phase.
        try:
            u._last_mpm_velocity = v_mpm.copy()
            u._last_mpm_in_contact = bool(self._in_contact)
        except Exception:
            pass

        return dict(
            ms_tip_pos=tip_ms,
            ms_quat=q_ms,
            ms_tcp_vel=tcp_vel,
            ms_force=f_link,
            mpm_tip_pos=tip_mpm_driven,  # native MPM knife position (driven by MPM)
            mpm_vel=v_mpm,
            mpm_force=f_mpm,
        )

    def _tcp_linear_velocity(self, env, pos_now: np.ndarray, dt: float) -> np.ndarray:
        if self._prev_tcp_pose is None:
            self._prev_tcp_pose = pos_now.copy()
            return np.zeros(3, np.float32)
        v = (pos_now - self._prev_tcp_pose) / dt
        self._prev_tcp_pose = pos_now.copy()
        return v.astype(np.float32)

    def _mpm_substeps_per_control(self, dt_control: float) -> int:
        # self.sim has `dt` and `substeps` per frame; we want total wall time to cover dt_control.
        dt_frame = float(self.sim.dt) * int(self.sim.substeps)
        n = max(1, int(round(dt_control / dt_frame)))
        cap = int(getattr(self.cfg, "mpm_substeps_cap", 0) or 0)
        return min(n, cap) if cap > 0 else n

    def _apply_mpm_force_to_ms(self, f_mpm: np.ndarray, tip_ms: np.ndarray):
        """Inject -F_mpm (in MS frame) as an external wrench at the knife tip.
        Makes ManiSkill robot's descent slow down in hard materials.
        """
        # MPM frame → MS frame axis swap: (x, y_up, z) MPM -> (x, z, y_up) MS.
        # The reaction on the knife is -F (Newton's third law).
        f_ms = np.array([-float(f_mpm[0]),
                         -float(f_mpm[2]),
                         -float(f_mpm[1])], dtype=np.float32)
        mag = float(np.linalg.norm(f_ms))
        clip = float(getattr(self.cfg, "mpm_force_clip", 200.0))
        if mag > clip and mag > 1e-6:
            f_ms *= (clip / mag)
        scale = float(getattr(self.cfg, "mpm_force_scale", 0.25))
        f_ms *= scale
        # knife_link is a ManiSkill PhysxArticulationLinkComponent; it has the
        # same add_force_at_point(force, point, mode) API as a rigid body.
        try:
            rbody = self._knife_link._objs[0] if hasattr(self._knife_link, "_objs") else self._knife_link
            if hasattr(rbody, "add_force_at_point"):
                rbody.add_force_at_point(
                    force=f_ms,
                    point=np.asarray(tip_ms, np.float32),
                    mode="force")
        except Exception as e:
            # Don't crash collection — just log sparsely
            if not hasattr(self, "_force_apply_err_once"):
                print(f"[bridge] apply_force failed: {e}")
                self._force_apply_err_once = True

    def _read_mpm_force(self):
        """MPM reaction force on the knife = integrated grid impulse / sim.dt.
        Velocity is NOT read from sim's tracker (it bleeds saw-oscillation x),
        bridge computes it via finite-diff of tip_mpm instead (see step_with_env).
        """
        sim = self.sim
        f = np.zeros(3, np.float32)
        try:
            imp = sim.knife_impulse_g[None]
            dt = float(sim.dt)
            f = np.array([float(imp[0]) / dt, float(imp[1]) / dt, float(imp[2]) / dt],
                         np.float32)
        except Exception:
            try:
                f = np.asarray(sim._knife_applies_force, np.float32).reshape(-1)[:3]
            except Exception:
                pass
        return f

    # ---------- rendering ----------
    def reset_render_buffer(self):
        self._render_buf = []
        self._particle_sample_idx = None
        self._render_tick_counter = 0

    def begin_render(self, frame_dir: str):
        """No-op kept for API compatibility (matplotlib path writes directly)."""
        pass

    def dump_buffer_npz(self, out_path: str) -> bool:
        """Save knife trajectory (per-tick tip_mpm) to a compact .npz.  The
        subprocess renderer re-runs MPMCuttingSim with this knife motion and
        uses MPMRenderer to produce the demo-style video, so we only need to
        persist the knife poses (few KB) here.
        """
        import os
        buf = self._render_buf
        if not buf:
            return False
        knife = np.stack([f["knife"] for f in buf], axis=0).astype(np.float32)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, knife=knife)
        return True

    def _snapshot_for_render(self, tip_mpm: np.ndarray):
        """Capture subsampled (x, damage) + knife pose once per snapshot."""
        try:
            arr = self.sim.particles.to_numpy()
            x = arr["x"]           # (N, 3)
            D = arr.get("D", None) # (N,) damage scalar if present
        except Exception:
            return
        n = min(int(self.cfg.particle_subsample), int(x.shape[0]))
        if self._particle_sample_idx is None:
            rng = np.random.default_rng(0)
            self._particle_sample_idx = rng.choice(x.shape[0], size=n, replace=False)
        sub_x = x[self._particle_sample_idx].astype(np.float32).copy()
        sub_d = None
        if D is not None:
            sub_d = np.asarray(D[self._particle_sample_idx], np.float32).copy()
        self._render_buf.append({
            "parts": sub_x,
            "dmg": sub_d,
            "knife": tip_mpm.astype(np.float32).copy(),
        })

    def save_video(self, out_path: str, fps: int = 20, res: int = 720) -> bool:
        """Render captured snapshots to mp4 using open3d OffscreenRenderer (EGL
        headless). Particles are drawn as small spheres with damage colouring;
        a small cube marks the knife tip. Much closer to Chayoso's demo.gif
        than the matplotlib fallback, and works without a display server.
        """
        buf = self._render_buf
        if not buf:
            return False
        import os, subprocess, tempfile
        os.environ.setdefault("EGL_PLATFORM", "device")
        try:
            import open3d as o3d
            import open3d.visualization.rendering as rendering
        except Exception as e:
            print(f"[bridge] open3d missing: {e}")
            return False

        # Bounds from first frame for stable camera
        p0 = buf[0]["parts"]
        lo = p0.min(axis=0) - 0.03
        hi = p0.max(axis=0) + 0.03
        center = (lo + hi) / 2.0
        aabb = o3d.geometry.AxisAlignedBoundingBox(lo.tolist(), hi.tolist())

        # Global damage normalization
        dvals = [f.get("dmg") for f in buf if f.get("dmg") is not None]
        dmax = 1.0
        if dvals:
            dmax = max(1e-4,
                       float(np.quantile(np.concatenate(dvals), 0.99)))

        renderer = rendering.OffscreenRenderer(res, res)
        scene = renderer.scene
        scene.set_background([0.05, 0.06, 0.09, 1.0])
        scene.scene.set_sun_light([-0.5, -1.0, -0.5], [1.0, 1.0, 1.0], 70000)
        scene.scene.enable_sun_light(True)
        scene.scene.set_indirect_light_intensity(20000)

        # Camera: looking down at mild angle, roughly like demo.gif
        max_span = float(np.linalg.norm(hi - lo))
        eye = center + np.array([max_span * 1.2, max_span * 1.1, max_span * 1.0], np.float32)
        renderer.setup_camera(55.0, center.tolist(), eye.tolist(), [0, 1, 0])

        # Material for particles (point cloud with per-vertex colour)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.point_size = 6.0

        knife_mat = rendering.MaterialRecord()
        knife_mat.shader = "defaultLit"
        knife_mat.base_color = [0.90, 0.35, 0.17, 1.0]

        tmp_dir = tempfile.mkdtemp(prefix="mpm_render_")
        try:
            for t, fr in enumerate(buf):
                scene.clear_geometry()
                p = fr["parts"]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(p.astype(np.float64))
                d = fr.get("dmg")
                if d is not None:
                    colors = _damage_color(d, dmax)
                else:
                    colors = np.tile([0.85, 0.65, 0.40], (p.shape[0], 1))
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
                scene.add_geometry(f"cloud", pcd, mat)

                # Knife: long thin box from blade tip extending up in +Y (MPM up).
                # Knife mesh Z-extent is 0.476 m; tip at local zmin = -0.246, handle at zmax=+0.23.
                # In world, after URDF rotation, blade points DOWN (-z_ms = +y_mpm towards ground flipped to up).
                k = fr["knife"]
                blade_len = 0.476
                blade_w   = 0.062
                blade_t   = 0.034
                knife_box = o3d.geometry.TriangleMesh.create_box(blade_w, blade_len, blade_t)
                # centre box on knife tip (box origin is a corner), shift so bottom sits at knife tip
                knife_box.translate((float(k[0]) - blade_w / 2,
                                     float(k[1]),           # tip at bottom of box
                                     float(k[2]) - blade_t / 2))
                knife_box.compute_vertex_normals()
                scene.add_geometry("knife", knife_box, knife_mat)

                img = renderer.render_to_image()
                o3d.io.write_image(f"{tmp_dir}/f{t:05d}.png", img)

            # Concat to mp4 via imageio_ffmpeg binary
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg = "ffmpeg"
            cmd = [
                ffmpeg, "-y", "-framerate", str(fps),
                "-i", f"{tmp_dir}/f%05d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                out_path,
            ]
            ret = subprocess.run(cmd, capture_output=True, text=True)
            ok = (ret.returncode == 0)
            if not ok:
                print(f"[bridge] ffmpeg failed:\n{ret.stderr[-500:]}")
            return ok
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def set_apply_force(self, value: bool):
        """Toggle MPM→MS reaction force injection. Turn off during post-cut
        hold/retreat so the elastic spring from embedded MPM particles doesn't
        fight MS retreat under compliant PID."""
        self.cfg.apply_mpm_force_to_ms = bool(value)

    def freeze_knife(self):
        """Stop MPM knife descent (current_speed=0). Caller follows up with
        set_ms_driven(True) for retreat (MS again drives MPM y)."""
        try:
            self.sim.knife.speed = 0.0
            self.sim.knife.base_speed = 0.0
            self.sim.knife.current_speed[None] = 0.0
            self.sim.knife.min_speed_f[None] = 0.0
        except Exception:
            pass

    def set_ms_driven(self, value: bool):
        """Toggle whether the bridge writes knife.y from MS_tip each tick."""
        self._ms_driven_knife = bool(value)

    def close(self):
        self.sim = None

"""ManiSkill ⇄ MPM bridge — addfruits-pilot variant (bridge2).

Same public API as `bridge.py` (BridgeConfig, ManiSkillMPMBridge), but:
  - Loads MPMCuttingSim / mesh_sdf from `/data/Cutting_rebuttal` instead of
    `/data/EEF-Cutting-Simulation`. The Cutting_rebuttal repo carries the
    rebuttal-era MPM core + the `assets/fruit2/` mesh family used by the
    addfruits-pilot (cherry/plum/lemon/kiwi/shine_muscat/tomato/pear/grape/
    golden_strawberry).
  - Resolves the yaml's relative `assets/...` paths against
    `/data/Cutting_rebuttal/` by chdir-ing during pack build.
  - Optional override: `BridgeConfig.eef_repo` to point at a different repo
    if needed (smoke tests).

This is intentionally a near-copy of bridge.py so the two can be diffed
side-by-side. Keep them in sync when the upstream bridge changes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import math
import os
import sys

import numpy as np

from .frame import (
    FrameCalibration,
    ms_knife_tip_to_mpm,
    ms_world_to_mpm_world,
    mpm_knife_y_from_tip_y,
    MS_BOARD_CENTER_XY,
    MPM_BOARD_CENTER_XZ,
)


# Cutting_rebuttal carries the MPM core + assets/fruit2 mesh family used by
# the addfruits-pilot. Importing requires it on sys.path.
DEFAULT_REPO = Path("/data/Cutting_rebuttal")


def _ensure_repo_on_path(repo: Path):
    s = str(repo)
    if s not in sys.path:
        sys.path.insert(0, s)


def _damage_color(dmg_arr, dmax: float):
    """Map damage scalar to color: tan (healthy) -> red (fully damaged)."""
    d = np.clip(np.asarray(dmg_arr, np.float32) / max(1e-6, float(dmax)), 0.0, 1.0)
    r = (218 * (1 - d) + 184 * d) / 255.0
    g = (164 * (1 - d) + 51 * d) / 255.0
    b = (99  * (1 - d) + 48 * d) / 255.0
    return np.stack([r, g, b], axis=1)


@dataclass
class BridgeConfig:
    mpm_config_yaml: str                      # absolute path or relative-to-eef_repo
    control_freq_hz: float = 20.0
    calibration: FrameCalibration = None
    headless: bool = True
    mpm_substeps_cap: int = 0
    record_particles: bool = False
    particle_subsample: int = 8000
    render_every_n_ticks: int = 1
    apply_mpm_force_to_ms: bool = False
    mpm_force_scale: float = 0.25
    mpm_force_clip: float = 200.0
    knife_speed_mps: float = 0.10
    mpm_extra_descent_y: float = 0.000
    eef_repo: str = str(DEFAULT_REPO)         # bridge2-only: which repo to import from
    # Lift the seeded fruit above board_top_y by this much (m). Default 0
    # mirrors stock behaviour. For tiny fruits (cherry/grape/etc) where the
    # MPM knife passes by without registering contact, lift the fruit so the
    # blade SDF actually sweeps through it.
    fruit_y_offset: float = 0.000
    # Override the mesh's initial scale (uniform). None = keep yaml's value.
    # When the ManiSkill side picks a `_variation_scale`, pass it here so
    # the MPM fruit matches the visual fruit size.
    fruit_scale_override: Optional[float] = None
    # Sync MPM fruit world position to MS fruit world position. When True
    # (default) the bridge writes cutting_mesh.initial_transform.translate
    # to the ms-fruit-xy mapped through (MS_BOARD_CENTER_XY → MPM_BOARD_CENTER_XZ).
    # Set False to keep the yaml's static position (legacy behaviour).
    sync_fruit_pose: bool = True


class ManiSkillMPMBridge:
    """Couples a ManiSkill BaseEnv with an MPMCuttingSim.

    bridge2 variant — see module docstring for the differences vs bridge.py.
    """

    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.calib = cfg.calibration or FrameCalibration()
        self.sim = None
        self.packs = None
        self._taichi_inited = False
        self._prev_tcp_pose = None
        self._prev_tip_mpm = None
        self._banana_xy_ms = np.zeros(2, np.float32)
        self._banana_yaw_ms = 0.0
        self._render_buf = []
        self._particle_sample_idx = None
        self._render_tick_counter = 0
        # Cutting_rebuttal MPMCuttingSim does not expose `sim_time`; we
        # accumulate it ourselves and pass into sim.step(T) each substep.
        self._mpm_sim_time = 0.0

    # ---------- setup ----------
    def _init_taichi(self):
        import taichi as ti
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

        repo = Path(self.cfg.eef_repo).resolve()
        assert repo.is_dir(), f"eef_repo not found: {repo}"
        _ensure_repo_on_path(repo)

        # Yaml mesh_paths in Cutting_rebuttal style are relative to repo root
        # (e.g. `assets/fruit2/cherry.obj`). Resolve them by chdir during
        # pack build, then restore.
        from sdf_utils.mesh_sdf import mesh_to_sdf
        from mpmcore.sim import MPMCuttingSim

        with open(self.cfg.mpm_config_yaml) as f:
            mpm_cfg = yaml.safe_load(f)

        prev_cwd = os.getcwd()
        os.chdir(str(repo))
        try:
            # Sync MPM fruit pose + scale to ManiSkill side BEFORE building SDF.
            # mesh_to_sdf reads `initial_transform.{scale,translate}` from the
            # cutting_mesh block, so we mutate the loaded cfg dict in place.
            cm_cfg = mpm_cfg.setdefault("cutting_mesh", {})
            cm_tx = cm_cfg.setdefault("initial_transform", {})
            if self.cfg.fruit_scale_override is not None:
                s = float(self.cfg.fruit_scale_override)
                cm_tx["scale"] = [s, s, s]
            if self.cfg.sync_fruit_pose:
                # Map ms fruit world (x_ms, y_ms) → mpm world (x_mpm, z_mpm).
                # board-center subtraction makes this board-local first.
                x_ms = float(self._banana_xy_ms[0])
                y_ms = float(self._banana_xy_ms[1])
                x_mpm = x_ms - float(MS_BOARD_CENTER_XY[0]) + float(MPM_BOARD_CENTER_XZ[0])
                z_mpm = y_ms - float(MS_BOARD_CENTER_XY[1]) + float(MPM_BOARD_CENTER_XZ[1])
                # Add to the yaml's existing translate (rather than overwrite).
                # The addfruits yamls carry a centroid-alignment translate so
                # the AABB center lands at the mesh origin after scale+rot —
                # without this, fruit2 OBJs (whose origin is offset 3-9cm from
                # the centroid) would have the cut plane miss the fruit middle.
                base_t = list(cm_tx.get("translate", [0.0, 0.0, 0.0]))
                cm_tx["translate"] = [
                    float(x_mpm + base_t[0]),
                    float(base_t[1]),  # keep yaml's y (typ. 0); seed lifts y_min
                    float(z_mpm + base_t[2]),
                ]
                print(f"[bridge2] fruit synced: ms=({x_ms:+.3f},{y_ms:+.3f}) "
                      f"→ mpm=({x_mpm:+.3f}, _, {z_mpm:+.3f})  "
                      f"+ yaml_t=({base_t[0]:+.4f},_,{base_t[2]:+.4f})")

            def _pack(block, with_blade):
                transform = block.get("initial_transform", {})
                kwargs = {}
                if with_blade:
                    kwargs["knife_blade"] = block.get(
                        "blade", {"axis": "Y", "fraction": 0.5})
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
            mpm_cfg["cutting_mesh_pack"] = cutting_pack
            self.sim = MPMCuttingSim(
                mpm_cfg, cutting_pack, knife_pack, board_pack=board_pack,
                viewer=False,
            )
            top = None
            if self.sim.board is not None:
                try:
                    top = float(self.sim.board_top_y[None])
                except Exception:
                    top = None
            # Optional lift: seed the fruit ABOVE the board by fruit_y_offset.
            # Useful for tiny fruits where the knife SDF would otherwise sweep
            # through without registering contact.
            lifted_top = top
            offset = float(getattr(self.cfg, "fruit_y_offset", 0.0))
            if top is not None and offset > 0.0:
                lifted_top = top + offset
                print(f"[bridge2] fruit lifted: support_top_y {top:.4f} → {lifted_top:.4f} (+{offset:.3f})")
            self.sim.seed_particles_from_mesh(
                cutting_pack, mpm_cfg["cutting_mesh"], name="cutting_mesh",
                support_top_y=lifted_top,
            )
        finally:
            os.chdir(prev_cwd)

    def _override_knife_motion(self):
        knife = self.sim.knife
        self._orig_knife_update = knife.update

        def _external_update(dt):
            knife.sim_time = float(getattr(knife, "sim_time", 0.0)) + float(dt)
        knife.update = _external_update

    def setup(self, env, *, banana_xy_ms=None, banana_yaw_ms: float = 0.0):
        if not self._taichi_inited:
            self._init_taichi()
        u = env.unwrapped

        # Resolve banana (= MS fruit) pose FIRST — _load_mpm reads it to
        # write cutting_mesh.initial_transform.translate before SDF build.
        if banana_xy_ms is not None:
            self._banana_xy_ms = np.asarray(banana_xy_ms, np.float32).reshape(2)
        else:
            block = getattr(u, "block_apple", [None])[0]
            if block is not None:
                p = np.asarray(block.pose.p, np.float32).reshape(-1)
                self._banana_xy_ms = p[:2].copy()
        self._banana_yaw_ms = float(banana_yaw_ms)

        self._load_mpm()

        knife_link = getattr(u, "knife_link", None)
        assert knife_link is not None, "env must expose knife_link (bananacut env does)"
        self._knife_link = knife_link
        self._prev_tcp_pose = None
        self._prev_tip_mpm = None

        self._knife_yfoot = float(self.sim._knife_yfoot) if hasattr(
            self.sim, "_knife_yfoot") else 0.0

        self._ms_driven_knife = True
        self._in_contact = False
        self._contact_threshold_N = 1.0

        try:
            tip_ms_now = u._get_tip_from_eef()[0].detach().cpu().numpy().astype(np.float32)
            start_tip_y_mpm = float(tip_ms_now[2]) + float(self.calib.v_offset)
            self.sim.knife.y[None] = start_tip_y_mpm - self._knife_yfoot
        except Exception as e:
            print(f"[bridge2] knife.y init skipped: {e}")

        try:
            v = float(self.cfg.knife_speed_mps)
            knife = self.sim.knife
            knife.speed = v
            knife.base_speed = v
            knife.current_speed[None] = v
            knife.min_speed_f[None] = 0.10 * v
            print(f"[bridge2] knife speed overridden to {v} m/s")
        except Exception as e:
            print(f"[bridge2] knife speed override skipped: {e}")

    # ---------- step ----------
    def step_with_env(self, env, dt_control: float) -> Dict[str, np.ndarray]:
        u = env.unwrapped

        try:
            tip_ms = u._get_tip_from_eef()[0].detach().cpu().numpy().astype(np.float32)
        except Exception:
            knife_pose = self._knife_link.pose
            tip_ms = np.asarray(knife_pose.p, np.float32).reshape(-1, 3)[-1]
        q_ms = np.asarray(self._knife_link.pose.q, np.float32).reshape(-1, 4)[-1]
        tcp_vel = self._tcp_linear_velocity(env, tip_ms, dt_control)
        f_link = self._knife_link.get_net_contact_forces()
        if hasattr(f_link, "cpu"):
            f_link = f_link.cpu().numpy()
        f_link = np.asarray(f_link, np.float32).reshape(-1, 3)[-1]

        # When sync_fruit_pose was applied at setup (mpm fruit seeded at the
        # ms-synced world position), the knife mapping must use the SAME raw
        # frame transform — otherwise the knife lands at the cfg-fixed
        # canonical_xz while the fruit sits at ms-synced xz, missing each
        # other entirely. Use ms_world_to_mpm_world (raw axis swap +
        # board-center offset). When sync is off, fall back to the original
        # banana-local mapping (legacy behaviour).
        if getattr(self.cfg, "sync_fruit_pose", False):
            tip_mpm = ms_world_to_mpm_world(tip_ms)
        else:
            tip_mpm = ms_knife_tip_to_mpm(
                tip_ms=tip_ms,
                banana_pose_ms_xy=self._banana_xy_ms,
                banana_yaw_ms=self._banana_yaw_ms,
                canonical_xz=np.asarray(self.calib.mpm_banana_canonical_xz, np.float32),
                v_offset=self.calib.v_offset,
            )

        if hasattr(self.sim.knife, "z_off"):
            # Under sync_fruit_pose, tip_mpm is already in absolute MPM-world
            # coords; the knife collider's "z_off" is what shifts the SDF in
            # the world frame, so the offset is just tip_mpm[2]. Under the
            # legacy banana-local mapping, the canonical xz[1] is what the
            # collider's zero corresponds to, so we subtract it.
            if getattr(self.cfg, "sync_fruit_pose", False):
                self.sim.knife.z_off[None] = float(tip_mpm[2])
            else:
                z_canon = float(self.calib.mpm_banana_canonical_xz[1])
                self.sim.knife.z_off[None] = float(tip_mpm[2] - z_canon)
        if self._ms_driven_knife:
            extra = float(getattr(self.cfg, "mpm_extra_descent_y", 0.0))
            self.sim.knife.y[None] = (
                float(tip_ms[2]) + float(self.calib.v_offset)
                - self._knife_yfoot - extra
            )

        mpm_substeps = self._mpm_substeps_per_control(dt_control)
        dt_frame = float(self.sim.dt) * int(self.sim.substeps)
        for _ in range(mpm_substeps):
            self.sim.step(self._mpm_sim_time)
            self._mpm_sim_time += dt_frame

        mpm_knife_tip_y = float(self.sim.knife.y[None] + self._knife_yfoot)
        tip_mpm_driven = np.array(
            [tip_mpm[0], mpm_knife_tip_y, tip_mpm[2]], dtype=np.float32)

        if getattr(self.cfg, "record_particles", False):
            cadence = max(1, int(getattr(self.cfg, "render_every_n_ticks", 5)))
            if (len(self._render_buf) == 0) or (self._render_tick_counter % cadence == 0):
                self._snapshot_for_render(tip_mpm_driven)
            self._render_tick_counter += 1

        f_mpm = self._read_mpm_force()
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
                stop_tip_y = board_top_y_mpm + 0.005
                self.sim.knife.stop_y = stop_tip_y - self._knife_yfoot
                self.sim.knife._down[None] = 1
                self.sim.knife.current_speed[None] = float(self.cfg.knife_speed_mps)
            except Exception as e:
                print(f"[bridge2] contact-transition failed: {e}")
        try:
            u._last_mpm_force = np.asarray(f_mpm, np.float32).reshape(-1)[:3]
        except Exception:
            pass
        if getattr(self.cfg, "apply_mpm_force_to_ms", False):
            self._apply_mpm_force_to_ms(f_mpm, tip_ms)
        if self._prev_tip_mpm is not None:
            v_mpm = ((tip_mpm_driven - self._prev_tip_mpm) / float(dt_control)).astype(np.float32)
        else:
            v_mpm = np.zeros(3, np.float32)
        self._prev_tip_mpm = tip_mpm_driven.copy()
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
            mpm_tip_pos=tip_mpm_driven,
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
        dt_frame = float(self.sim.dt) * int(self.sim.substeps)
        n = max(1, int(round(dt_control / dt_frame)))
        cap = int(getattr(self.cfg, "mpm_substeps_cap", 0) or 0)
        return min(n, cap) if cap > 0 else n

    def _apply_mpm_force_to_ms(self, f_mpm: np.ndarray, tip_ms: np.ndarray):
        f_ms = np.array([-float(f_mpm[0]),
                         -float(f_mpm[2]),
                         -float(f_mpm[1])], dtype=np.float32)
        mag = float(np.linalg.norm(f_ms))
        clip = float(getattr(self.cfg, "mpm_force_clip", 200.0))
        if mag > clip and mag > 1e-6:
            f_ms *= (clip / mag)
        scale = float(getattr(self.cfg, "mpm_force_scale", 0.25))
        f_ms *= scale
        try:
            rbody = self._knife_link._objs[0] if hasattr(self._knife_link, "_objs") else self._knife_link
            if hasattr(rbody, "add_force_at_point"):
                rbody.add_force_at_point(
                    force=f_ms,
                    point=np.asarray(tip_ms, np.float32),
                    mode="force")
        except Exception as e:
            if not hasattr(self, "_force_apply_err_once"):
                print(f"[bridge2] apply_force failed: {e}")
                self._force_apply_err_once = True

    def _read_mpm_force(self):
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
        pass

    def dump_buffer_npz(self, out_path: str) -> bool:
        buf = self._render_buf
        if not buf:
            return False
        knife = np.stack([f["knife"] for f in buf], axis=0).astype(np.float32)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, knife=knife)
        return True

    def _snapshot_for_render(self, tip_mpm: np.ndarray):
        try:
            arr = self.sim.particles.to_numpy()
            x = arr["x"]
            D = arr.get("D", None)
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

    def set_apply_force(self, value: bool):
        self.cfg.apply_mpm_force_to_ms = bool(value)

    def freeze_knife(self):
        try:
            self.sim.knife.speed = 0.0
            self.sim.knife.base_speed = 0.0
            self.sim.knife.current_speed[None] = 0.0
            self.sim.knife.min_speed_f[None] = 0.0
        except Exception:
            pass

    def set_ms_driven(self, value: bool):
        self._ms_driven_knife = bool(value)

    def close(self):
        self.sim = None

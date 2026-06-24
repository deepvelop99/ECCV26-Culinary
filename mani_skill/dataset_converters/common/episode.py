"""Common intermediate episode format.

One rollout is serialized to a single .npz under:
  <root>/<task>/<control_mode>/episode_<idx:06d>.npz

Fields are raw (unnormalized), SI units. Each converter (RDT / OpenVLA / Octo)
reads this and writes its own model-specific format.

Force/velocity logs from both ManiSkill and MPM are stored so downstream
analysis can check alignment (success criterion: per-step |F_ms - F_mpm| and
|V_ms - V_mpm| below tolerance, per CulinaryCut paper §3.2).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import json
import numpy as np


CONTROL_MODE_JOINT = "pd_joint_pos"
CONTROL_MODE_EE = "pd_ee_delta_pose"


@dataclass
class Episode:
    task: str
    instruction: str
    control_mode: str          # "pd_joint_pos" | "pd_ee_delta_pose"
    control_freq: int          # Hz
    sim_freq: int              # Hz
    seed: int
    success: bool

    # Per-step arrays, length T
    rgb: np.ndarray            # (T, H, W, 3) uint8, primary camera
    wrist_rgb: Optional[np.ndarray]  # (T, H, W, 3) uint8 or None
    qpos: np.ndarray           # (T, n_q) float32 — full robot qpos incl. gripper
    tcp_pose: np.ndarray       # (T, 7) float32 — xyz + quat (wxyz) world frame
    action: np.ndarray         # (T, A) float32 — raw action fed to env.step
    is_terminal: np.ndarray    # (T,) bool

    # Physics telemetry (per control step). Design note (A+C):
    #   - velocity is used for alignment gating (both sides finite-diff match)
    #   - force label = MPM-only; ManiSkill force kept for debugging/reference
    ms_tcp_vel: np.ndarray     # (T, 3) float32 — ManiSkill TCP linear velocity, world frame
    ms_knife_force: np.ndarray # (T, 3) float32 — ManiSkill net contact force on knife link (DEBUG)
    mpm_knife_vel: np.ndarray  # (T, 3) float32 — MPM knife velocity (MPM world frame)
    mpm_knife_force: np.ndarray# (T, 3) float32 — MPM knife reaction force (LABEL)

    # Tip trajectories (world frames) — useful for reproducing frame mapping
    # and for debugging knife-banana AABB overlap.
    ms_tip_world: np.ndarray   # (T, 3) float32 — knife tip in ManiSkill world (z up)
    mpm_tip_world: np.ndarray  # (T, 3) float32 — knife tip in MPM world (y up)

    # Object variation applied for this episode (CulinaryCut §3.3)
    variation: Dict[str, Any]  # {"object": "banana", "scale": 1.0, "pos_offset": [0,0,0], "yaw": 0.0}

    # MPM sidecar references (paths relative to episode file)
    mpm_particle_dir: Optional[str] = None   # exports_<ep>/frame_*.npz
    mpm_log_path: Optional[str] = None       # JSON log from MPM side

    # Success diagnostics
    alignment_metrics: Dict[str, float] = field(default_factory=dict)
    # e.g. {"max_force_err": 0.8, "max_vel_err": 0.03,
    #       "force_tol": 1.5, "vel_tol": 0.05, "passed": True}

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "task": self.task,
            "instruction": self.instruction,
            "control_mode": self.control_mode,
            "control_freq": self.control_freq,
            "sim_freq": self.sim_freq,
            "seed": self.seed,
            "success": bool(self.success),
            "variation": self.variation,
            "mpm_particle_dir": self.mpm_particle_dir,
            "mpm_log_path": self.mpm_log_path,
            "alignment_metrics": self.alignment_metrics,
        }
        arrs = dict(
            rgb=self.rgb,
            qpos=self.qpos.astype(np.float32),
            tcp_pose=self.tcp_pose.astype(np.float32),
            action=self.action.astype(np.float32),
            is_terminal=self.is_terminal.astype(bool),
            ms_tcp_vel=self.ms_tcp_vel.astype(np.float32),
            ms_knife_force=self.ms_knife_force.astype(np.float32),
            mpm_knife_vel=self.mpm_knife_vel.astype(np.float32),
            mpm_knife_force=self.mpm_knife_force.astype(np.float32),
            ms_tip_world=self.ms_tip_world.astype(np.float32),
            mpm_tip_world=self.mpm_tip_world.astype(np.float32),
            meta_json=np.array(json.dumps(meta)),
        )
        if self.wrist_rgb is not None:
            arrs["wrist_rgb"] = self.wrist_rgb
        np.savez_compressed(path, **arrs)

    @staticmethod
    def load(path: Path) -> "Episode":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        T = z["qpos"].shape[0]
        zeros_T3 = np.zeros((T, 3), np.float32)
        return Episode(
            task=meta["task"],
            instruction=meta["instruction"],
            control_mode=meta["control_mode"],
            control_freq=meta["control_freq"],
            sim_freq=meta["sim_freq"],
            seed=meta["seed"],
            success=meta["success"],
            rgb=z["rgb"],
            wrist_rgb=z["wrist_rgb"] if "wrist_rgb" in z.files else None,
            qpos=z["qpos"],
            tcp_pose=z["tcp_pose"],
            action=z["action"],
            is_terminal=z["is_terminal"],
            ms_tcp_vel=z["ms_tcp_vel"] if "ms_tcp_vel" in z.files else zeros_T3,
            ms_knife_force=z["ms_knife_force"] if "ms_knife_force" in z.files else zeros_T3,
            mpm_knife_vel=z["mpm_knife_vel"] if "mpm_knife_vel" in z.files else zeros_T3,
            mpm_knife_force=z["mpm_knife_force"] if "mpm_knife_force" in z.files else zeros_T3,
            ms_tip_world=z["ms_tip_world"] if "ms_tip_world" in z.files else zeros_T3,
            mpm_tip_world=z["mpm_tip_world"] if "mpm_tip_world" in z.files else zeros_T3,
            variation=meta.get("variation", {}),
            mpm_particle_dir=meta.get("mpm_particle_dir"),
            mpm_log_path=meta.get("mpm_log_path"),
            alignment_metrics=meta.get("alignment_metrics", {}),
        )


def episode_path(root: Path, task: str, control_mode: str, idx: int) -> Path:
    return Path(root) / task / control_mode / f"episode_{idx:06d}.npz"


def iter_episode_paths(root: Path, task: str, control_mode: str):
    d = Path(root) / task / control_mode
    if not d.exists():
        return
    for p in sorted(d.glob("episode_*.npz")):
        yield p

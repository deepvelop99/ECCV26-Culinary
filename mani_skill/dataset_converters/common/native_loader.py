"""Loader for the native ManiSkill + MPM co-sim episode layout.

Produced by `collect_rollouts_mpm.py`:
    <root>/<task>/auto_<seed>/
        trajectory.h5       - env_states + obs/agent/qpos + actions + rewards
        trajectory.json     - episode meta (seed, success, source)
        <N>.mp4             - ManiSkill camera video
        alignment.json      - per-step F/V + tip trajectories + variation

This module provides an `EpisodeBundle` that lazily yields per-step fields
from these four files so each model-specific converter can consume them
uniformly without knowing the on-disk layout.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator, Dict, Any
import json
import numpy as np


@dataclass
class EpisodeBundle:
    root: Path                  # auto_<seed>/ directory
    task: str
    seed: int
    instruction: str
    success: bool
    variation: Dict[str, Any]
    alignment: Dict[str, Any]   # alignment stats (passed, vel_err, contact_frames, ...)
    control_mode: str

    # Raw arrays (lazy; caller triggers by property)
    _h5_path: Path
    _mp4_path: Optional[Path]
    _align_path: Path

    # Cached loads
    _qpos: Optional[np.ndarray] = None
    _qvel: Optional[np.ndarray] = None
    _actions: Optional[np.ndarray] = None
    _rewards: Optional[np.ndarray] = None
    _art_state: Optional[np.ndarray] = None      # pkour articulation state
    _block_state: Optional[np.ndarray] = None    # block_apple_0 actor state
    _per_step: Optional[Dict[str, np.ndarray]] = None
    _frames: Optional[np.ndarray] = None

    # ---------- factories ----------
    @staticmethod
    def from_dir(auto_dir: Path, task: str) -> "EpisodeBundle":
        auto_dir = Path(auto_dir)
        traj_json = json.load(open(auto_dir / "trajectory.json"))
        meta = traj_json["episodes"][0]
        align = json.load(open(auto_dir / "alignment.json"))
        mp4 = next(iter(auto_dir.glob("*.mp4")), None)
        return EpisodeBundle(
            root=auto_dir,
            task=task,
            seed=int(meta.get("episode_seed", align.get("seed", 0))),
            instruction=align.get("instruction", ""),
            success=bool(meta.get("success", False)),
            variation=align.get("variation", {}),
            alignment=align.get("alignment", {}),
            control_mode=meta.get("control_mode", align.get("control_mode", "")),
            _h5_path=auto_dir / "trajectory.h5",
            _mp4_path=mp4,
            _align_path=auto_dir / "alignment.json",
        )

    # ---------- h5 accessors ----------
    def _load_h5(self):
        import h5py
        with h5py.File(self._h5_path, "r") as f:
            g = f["traj_0"]
            self._qpos = g["obs"]["agent"]["qpos"][:]
            self._qvel = g["obs"]["agent"]["qvel"][:]
            self._actions = g["actions"][:]
            self._rewards = g["rewards"][:]
            self._art_state = g["env_states"]["articulations"]["pkour"][:]
            self._block_state = g["env_states"]["actors"]["block_apple_0"][:]

    @property
    def qpos(self) -> np.ndarray:
        if self._qpos is None:
            self._load_h5()
        return self._qpos

    @property
    def qvel(self) -> np.ndarray:
        if self._qvel is None:
            self._load_h5()
        return self._qvel

    @property
    def actions(self) -> np.ndarray:
        if self._actions is None:
            self._load_h5()
        return self._actions

    @property
    def rewards(self) -> np.ndarray:
        if self._rewards is None:
            self._load_h5()
        return self._rewards

    @property
    def block_state(self) -> np.ndarray:
        if self._block_state is None:
            self._load_h5()
        return self._block_state  # (T+1, 13) [pose(7)+linvel(3)+angvel(3)]

    @property
    def T(self) -> int:
        """Step count (action length, obs has T+1)."""
        return int(self.actions.shape[0])

    # ---------- mp4 frames ----------
    def load_frames(self) -> np.ndarray:
        """Returns (T_video, H, W, 3) uint8 array. Calls cached after first load."""
        if self._frames is not None:
            return self._frames
        assert self._mp4_path is not None, f"no mp4 in {self.root}"
        import imageio
        r = imageio.get_reader(str(self._mp4_path))
        frames = []
        for f in r:
            frames.append(np.asarray(f, np.uint8))
        r.close()
        self._frames = np.stack(frames, axis=0)
        return self._frames

    # ---------- alignment sidecar ----------
    @property
    def per_step(self) -> Dict[str, np.ndarray]:
        if self._per_step is None:
            raw = json.load(open(self._align_path))["per_step"]
            self._per_step = {k: np.asarray(v, np.float32) for k, v in raw.items()}
        return self._per_step


def iter_bundles(root: Path, task: str) -> Iterator[EpisodeBundle]:
    """Yield bundles from <root>/<task>/auto_*/"""
    task_dir = Path(root) / task
    if not task_dir.exists():
        return
    for d in sorted(task_dir.glob("auto_*")):
        if (d / "trajectory.h5").exists():
            yield EpisodeBundle.from_dir(d, task)

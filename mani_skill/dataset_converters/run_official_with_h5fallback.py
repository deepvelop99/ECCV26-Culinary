"""Run to_openvla / to_octo / to_rdt converters but read frames from h5 base_camera
when the mp4 sidecar is absent (combined dataset case)."""
import sys, importlib, h5py, numpy as np, json
from pathlib import Path

# 1) Patch EpisodeBundle.load_frames to fall back to h5 base_camera/rgb
from dataset_converters.common import native_loader as nl

# Patch _load_h5 to skip missing optional groups (block_apple_0 / pkour) found in
# some auto_*/trajectory.h5 files in the combined dataset.
_orig_load_h5 = nl.EpisodeBundle._load_h5
def _load_h5_safe(self):
    import h5py
    with h5py.File(self._h5_path, "r") as f:
        g = f["traj_0"]
        self._qpos = g["obs"]["agent"]["qpos"][:]
        self._qvel = g["obs"]["agent"]["qvel"][:]
        self._actions = g["actions"][:]
        self._rewards = g["rewards"][:]
        try:
            self._art_state = g["env_states"]["articulations"]["pkour"][:]
        except (KeyError, OSError):
            self._art_state = None
        try:
            self._block_state = g["env_states"]["actors"]["block_apple_0"][:]
        except (KeyError, OSError):
            actors = g["env_states"]["actors"]
            cand = [k for k in actors.keys() if k.startswith("block_")]
            self._block_state = actors[cand[0]][:] if cand else None
nl.EpisodeBundle._load_h5 = _load_h5_safe

_orig = nl.EpisodeBundle.load_frames
def load_frames_h5fallback(self):
    if self._frames is not None:
        return self._frames
    if self._mp4_path is not None and Path(self._mp4_path).exists():
        return _orig(self)
    with h5py.File(self._h5_path, "r") as f:
        rgb = f["traj_0/obs/sensor_data/base_camera/rgb"][:]
    self._frames = np.asarray(rgb, np.uint8)
    return self._frames
nl.EpisodeBundle.load_frames = load_frames_h5fallback

# 2) Patch from_dir to inject a synthetic trajectory.json when missing (a couple of eps lack it)
_orig_from_dir = nl.EpisodeBundle.from_dir
@staticmethod
def from_dir_safe(auto_dir, task):
    auto_dir = Path(auto_dir)
    tj = auto_dir / "trajectory.json"
    if not tj.exists():
        # Synthesize minimal traj_json from alignment.json + h5
        ali_path = auto_dir / "alignment.json"
        ali = json.load(open(ali_path)) if ali_path.exists() else {}
        with h5py.File(auto_dir / "trajectory.h5", "r") as f:
            success = bool(f["traj_0/success"][-1]) if "traj_0/success" in f else True
        synthetic = {"episodes": [{
            "episode_seed": ali.get("seed", 0),
            "success": success,
            "control_mode": ali.get("control_mode", "pd_ee_delta_pose"),
        }]}
        tj.write_text(json.dumps(synthetic))
    return _orig_from_dir(auto_dir, task)
nl.EpisodeBundle.from_dir = from_dir_safe

# 3) Re-import the converter modules so they pick up the patched loader
target = sys.argv[1]  # "openvla" / "octo" / "rdt"
in_root = sys.argv[2]
out_root = sys.argv[3]
task = sys.argv[4]
extra = sys.argv[5:]  # e.g. --include-failed --no-frames

if target == "openvla":
    sys.argv = ["build_openvla_rlds.py", "--in", in_root, "--out", out_root, "--task", task, *extra]
    from dataset_converters.to_openvla import build_openvla_rlds as m
elif target == "octo":
    sys.argv = ["build_octo_rlds.py", "--in", in_root, "--out", out_root, "--task", task, *extra]
    from dataset_converters.to_octo import build_octo_rlds as m
elif target == "rdt":
    sys.argv = ["build_rdt_hdf5.py", "--in", in_root, "--out", out_root, "--task", task, *extra]
    from dataset_converters.to_rdt import build_rdt_hdf5 as m
else:
    raise SystemExit(f"unknown target {target}")

m.main()

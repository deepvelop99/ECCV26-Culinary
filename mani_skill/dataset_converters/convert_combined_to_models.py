"""Convert /data/datasets/maniskill_mpm_combined/ to per-model dataset variants.

Outputs:
  /data/datasets/cuttingvla_openvla_5hz/{fruit}/auto_*/trajectory.h5
  /data/datasets/cuttingvla_octo_5hz/{fruit}/auto_*/trajectory.h5
  /data/datasets/cuttingvla_rdt_20hz/{fruit}/auto_*/trajectory.h5

Differences (per model):

OpenVLA (5Hz, single image, accumulated actions):
  observations:
    image:        (T_5, 224, 224, 3) uint8     ← base_camera resized
    state:        (T_5, 8)           float32   ← qpos[7] + gripper
  actions:        (T_5-1, 7)         float32   ← delta pose accumulated over 4 ticks
  language_instruction: bytes        ← "Cut the {fruit} ..."
  rewards:        (T_5-1,)           float32

Octo (5Hz, primary+wrist images, action chunks):
  observations:
    image_primary:(T_5, 256, 256, 3) uint8     ← base_camera
    image_wrist:  (T_5, 128, 128, 3) uint8     ← hand_camera resized (or zeros if absent)
    proprio:      (T_5, 8)           float32   ← qpos[7] + gripper
  actions:        (T_5-1, H=4, 7)    float32   ← action chunks (next 4 deltas)
  language_instruction: bytes
  rewards:        (T_5-1,)           float32

RDT-1B (native 20Hz, image 384, single arm padded to 14):
  observations:
    image:        (T, 384, 384, 3)   uint8     ← base_camera upscaled
    state:        (T, 14)            float32   ← qpos[7]+grip padded right (right arm)
  actions:        (T-1, H=64, 14)    float32   ← chunks of 64 (last padded)
  language_instruction: bytes
"""
from __future__ import annotations
import argparse, glob, os, sys, json, multiprocessing as mp
from pathlib import Path
import h5py
import numpy as np
try:
    import cv2
except ImportError:
    os.system(f"{sys.executable} -m pip install --quiet opencv-python-headless")
    import cv2

FRUITS = ["apple", "banana", "cucumber", "melon", "orange", "peach", "strawberry"]
INSTR = {f: f"Cut the {f} in the middle with the knife and return the robot arm back to its original position." for f in FRUITS}
SRC_ROOT = Path("/data/datasets/maniskill_mpm_combined/bananacut")
DST_OPENVLA = Path("/data/datasets/cuttingvla_openvla_5hz")
DST_OCTO    = Path("/data/datasets/cuttingvla_octo_5hz")
DST_RDT     = Path("/data/datasets/cuttingvla_rdt_20hz")


def load_ep(d: Path):
    """Read combined h5; return (rgb_base, qpos, qvel, actions, rewards, hand_rgb_or_none).

    Actions denormalized from ManiSkill [-1,1] convention to physical units:
    delta_pose (translation m, rotation rad) × 0.1 (ManiSkill action_scale).
    Gripper component preserved as-is.
    """
    with h5py.File(d / "trajectory.h5", "r") as f:
        t0 = f["traj_0"]
        rgb = t0["obs/sensor_data/base_camera/rgb"][:]
        qpos = t0["obs/agent/qpos"][:]
        qvel = t0["obs/agent/qvel"][:]
        actions = t0["actions"][:].astype(np.float32)
        # Denormalize ManiSkill normalized action [-1,1] → meter/radian.
        # pd_ee_delta_pose: 6-DoF delta (×0.1), gripper unchanged.
        if actions.shape[1] >= 7:
            actions[:, :6] = actions[:, :6] * 0.1
        else:
            actions = actions * 0.1
        rewards = t0["rewards"][:]
        hand = None
        if "obs/sensor_data/hand_camera/rgb" in t0:
            hand = t0["obs/sensor_data/hand_camera/rgb"][:]
    return rgb, qpos, qvel, actions, rewards, hand


def subsample_5hz(rgb, qpos, qvel, actions, rewards, hand=None):
    """20Hz → 5Hz by stride 4. Actions accumulated over 4 ticks."""
    s = 4
    rgb_s  = rgb[::s]
    qpos_s = qpos[::s]
    qvel_s = qvel[::s]
    hand_s = hand[::s] if hand is not None else None
    # Accumulate actions over 4 consecutive ticks → length T_5 - 1
    T_5 = rgb_s.shape[0]
    act_acc = []
    for i in range(T_5 - 1):
        a = actions[i*s:(i+1)*s]  # (≤4, 7)
        if a.shape[0] == 0:
            act_acc.append(np.zeros(actions.shape[1], dtype=actions.dtype))
        else:
            # delta-pose components sum; gripper take last
            summed = a[:, :6].sum(axis=0) if a.shape[1] >= 7 else a.sum(axis=0)
            grip = a[-1, 6:7] if a.shape[1] >= 7 else np.zeros(0, dtype=actions.dtype)
            act_acc.append(np.concatenate([summed, grip]))
    actions_5 = np.stack(act_acc) if act_acc else np.zeros((0, actions.shape[1]), dtype=actions.dtype)
    rewards_5 = np.array([rewards[i*s:(i+1)*s].sum() for i in range(T_5 - 1)], dtype=rewards.dtype)
    return rgb_s, qpos_s, qvel_s, actions_5, rewards_5, hand_s


def write_openvla(dst: Path, rgb, qpos, actions, rewards, instr: str):
    """OpenVLA-friendly h5: image 224×224, state[8]=qpos[:7]+grip, action[7]."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    T = rgb.shape[0]
    img = np.stack([cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA) for f in rgb])
    state = np.concatenate([qpos[:, :7], qpos[:, 7:8]], axis=1).astype(np.float32)  # 8
    with h5py.File(dst, "w") as f:
        g = f.create_group("traj_0")
        og = g.create_group("observations")
        og.create_dataset("image", data=img, compression="gzip", compression_opts=4)
        og.create_dataset("state", data=state)
        g.create_dataset("actions", data=actions.astype(np.float32))
        g.create_dataset("rewards", data=rewards.astype(np.float32))
        g.attrs["language_instruction"] = instr


def write_octo(dst: Path, rgb, hand, qpos, actions, rewards, instr: str, H=4):
    """Octo: image_primary 256, image_wrist 128, action chunks."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    T = rgb.shape[0]
    if hand is None:
        wrist = np.zeros((T, 128, 128, 3), dtype=np.uint8)
    else:
        wrist = np.stack([cv2.resize(f, (128, 128), interpolation=cv2.INTER_AREA) for f in hand])
    proprio = np.concatenate([qpos[:, :7], qpos[:, 7:8]], axis=1).astype(np.float32)
    # Action chunks of H (next H actions); last few padded
    action_chunks = np.zeros((max(0, T - 1), H, actions.shape[1]), dtype=np.float32)
    for i in range(T - 1):
        end = min(i + H, T - 1)
        chunk = actions[i:end]
        action_chunks[i, :chunk.shape[0]] = chunk
    with h5py.File(dst, "w") as f:
        g = f.create_group("traj_0")
        og = g.create_group("observations")
        og.create_dataset("image_primary", data=rgb, compression="gzip", compression_opts=4)
        og.create_dataset("image_wrist",   data=wrist, compression="gzip", compression_opts=4)
        og.create_dataset("proprio",       data=proprio)
        g.create_dataset("actions", data=action_chunks)
        g.create_dataset("rewards", data=rewards.astype(np.float32))
        g.attrs["language_instruction"] = instr


def write_rdt(dst: Path, rgb, qpos, actions, rewards, instr: str, H=64):
    """RDT-1B: image 384, state 14 (single arm padded), action chunks H=64."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    T = rgb.shape[0]
    img = np.stack([cv2.resize(f, (384, 384), interpolation=cv2.INTER_LINEAR) for f in rgb])
    # state: single-arm 8 (qpos[:7] + gripper) padded to 14 (right-arm-only convention)
    state14 = np.zeros((T, 14), dtype=np.float32)
    state14[:, :8] = np.concatenate([qpos[:, :7], qpos[:, 7:8]], axis=1)
    # action: pad each 7-DoF action to 14 (right arm), then chunk
    actions14 = np.zeros((actions.shape[0], 14), dtype=np.float32)
    actions14[:, :actions.shape[1]] = actions
    chunks = np.zeros((max(0, T - 1), H, 14), dtype=np.float32)
    for i in range(T - 1):
        end = min(i + H, T - 1)
        chunks[i, :end - i] = actions14[i:end]
    with h5py.File(dst, "w") as f:
        g = f.create_group("traj_0")
        og = g.create_group("observations")
        og.create_dataset("image", data=img, compression="gzip", compression_opts=4)
        og.create_dataset("state", data=state14)
        g.create_dataset("actions", data=chunks)
        g.create_dataset("rewards", data=rewards.astype(np.float32))
        g.attrs["language_instruction"] = instr


def process_one(args_t):
    fruit, src_dir = args_t
    src = Path(src_dir)
    name = src.name  # auto_xxx
    instr = INSTR[fruit]
    try:
        rgb, qpos, qvel, actions, rewards, hand = load_ep(src)
    except Exception as e:
        return f"LOAD_FAIL {fruit}/{name}: {e}"
    rgb_5, qpos_5, qvel_5, act_5, rew_5, hand_5 = subsample_5hz(rgb, qpos, qvel, actions, rewards, hand)
    try:
        write_openvla(DST_OPENVLA / fruit / name / "trajectory.h5",
                      rgb_5, qpos_5, act_5, rew_5, instr)
        write_octo(DST_OCTO / fruit / name / "trajectory.h5",
                   rgb_5, hand_5, qpos_5, act_5, rew_5, instr)
        write_rdt(DST_RDT / fruit / name / "trajectory.h5",
                  rgb, qpos, actions, rewards, instr)
    except Exception as e:
        return f"WRITE_FAIL {fruit}/{name}: {e}"
    return f"OK {fruit}/{name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-per-fruit", type=int, default=0,
                    help="Cap eps per fruit (0=all)")
    args = ap.parse_args()

    tasks = []
    for fruit in FRUITS:
        dirs = sorted(glob.glob(str(SRC_ROOT / fruit / "bananacut" / "auto_*/")))
        if args.limit_per_fruit > 0:
            dirs = dirs[:args.limit_per_fruit]
        tasks.extend([(fruit, d) for d in dirs])
    print(f"{len(tasks)} eps to convert  ({len(FRUITS)} fruits)", flush=True)

    ok = 0
    fail = 0
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for i, msg in enumerate(pool.imap_unordered(process_one, tasks)):
                if msg.startswith("OK"): ok += 1
                else:
                    fail += 1
                    print(msg, flush=True)
                if (i + 1) % 100 == 0:
                    print(f"progress {i+1}/{len(tasks)}", flush=True)
    else:
        for i, t in enumerate(tasks):
            msg = process_one(t)
            if msg.startswith("OK"): ok += 1
            else:
                fail += 1
                print(msg, flush=True)

    print(f"\nDONE: ok={ok} fail={fail}")


if __name__ == "__main__":
    main()

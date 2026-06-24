"""Convert HDF5 cuttingvla_{openvla,octo}_5hz → TFDS RLDS TFRecord shards.

OpenVLA expects RLDS:
    observation: {image_0: uint8 (224,224,3), state: float32 (8,)}
    action: float32 (7,)
    reward: float32, discount: float32
    is_first/is_last/is_terminal: bool
    language_instruction: string

Octo similar but:
    observation: {image_primary: (256,256,3), image_wrist: (128,128,3), proprio: (8,)}
    action: float32 (4, 7)   # action chunk

Outputs:
    /data/datasets/cuttingvla_openvla_rlds/openvla_cutting/1.0.0/*.tfrecord-...
    /data/datasets/cuttingvla_octo_rlds/octo_cutting/1.0.0/*.tfrecord-...
plus features.json + dataset_info.json + dataset_statistics.json
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

FRUITS = ["apple", "banana", "cucumber", "melon", "orange", "peach", "strawberry"]
INSTR = {f: f"Cut the {f} in the middle with the knife and return the robot arm back to its original position." for f in FRUITS}


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


# ----- OpenVLA builder -----
class OpenVLACutting(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    SRC_ROOT = "/data/datasets/cuttingvla_openvla_5hz"

    def _info(self):
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image_0": tfds.features.Image(shape=(224, 224, 3), dtype=tf.uint8),
                        "state": tfds.features.Tensor(shape=(8,), dtype=tf.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                    "discount": tfds.features.Scalar(dtype=tf.float32),
                    "reward": tfds.features.Scalar(dtype=tf.float32),
                    "is_first": tfds.features.Scalar(dtype=tf.bool),
                    "is_last": tfds.features.Scalar(dtype=tf.bool),
                    "is_terminal": tfds.features.Scalar(dtype=tf.bool),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "fruit": tfds.features.Text(),
                    "file_path": tfds.features.Text(),
                }),
            })
        )

    def _split_generators(self, dl_manager):
        return {"train": self._gen_examples()}

    def _gen_examples(self):
        for fruit in FRUITS:
            paths = sorted(glob.glob(f"{self.SRC_ROOT}/{fruit}/auto_*/trajectory.h5"))
            for p in paths:
                key = f"{fruit}/{Path(p).parent.name}"
                ep = self._load(p, fruit)
                if ep is not None:
                    yield key, ep

    def _load(self, path, fruit):
        try:
            with h5py.File(path, "r") as f:
                t = f["traj_0"]
                img = t["observations/image"][:]              # (T_5, 224, 224, 3)
                state = t["observations/state"][:].astype(np.float32)  # (T_5, 8)
                actions = t["actions"][:].astype(np.float32)  # (T_5-1, 7)
                rewards = t["rewards"][:].astype(np.float32)  # (T_5-1,)
                instr = INSTR[fruit]
        except Exception as e:
            print(f"skip {path}: {e}")
            return None
        T = img.shape[0]
        steps = []
        for i in range(T):
            is_first = (i == 0)
            is_last  = (i == T - 1)
            a = actions[i] if i < T - 1 else np.zeros(7, np.float32)
            r = float(rewards[i]) if i < T - 1 else 0.0
            steps.append({
                "observation": {"image_0": img[i], "state": state[i]},
                "action": a,
                "discount": np.float32(1.0),
                "reward":   np.float32(r),
                "is_first": is_first,
                "is_last":  is_last,
                "is_terminal": is_last,
                "language_instruction": instr,
            })
        return {"steps": steps,
                "episode_metadata": {"fruit": fruit, "file_path": path}}


# ----- Octo builder -----
class OctoCutting(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    SRC_ROOT = "/data/datasets/cuttingvla_octo_5hz"
    H = 4  # action chunk

    def _info(self):
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image_primary": tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8),
                        "image_wrist":   tfds.features.Image(shape=(128, 128, 3), dtype=tf.uint8),
                        "proprio":       tfds.features.Tensor(shape=(8,), dtype=tf.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(self.H, 7), dtype=tf.float32),
                    "discount": tfds.features.Scalar(dtype=tf.float32),
                    "reward":   tfds.features.Scalar(dtype=tf.float32),
                    "is_first": tfds.features.Scalar(dtype=tf.bool),
                    "is_last":  tfds.features.Scalar(dtype=tf.bool),
                    "is_terminal": tfds.features.Scalar(dtype=tf.bool),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "fruit": tfds.features.Text(),
                    "file_path": tfds.features.Text(),
                }),
            })
        )

    def _split_generators(self, dl_manager):
        return {"train": self._gen_examples()}

    def _gen_examples(self):
        for fruit in FRUITS:
            paths = sorted(glob.glob(f"{self.SRC_ROOT}/{fruit}/auto_*/trajectory.h5"))
            for p in paths:
                key = f"{fruit}/{Path(p).parent.name}"
                ep = self._load(p, fruit)
                if ep is not None:
                    yield key, ep

    def _load(self, path, fruit):
        try:
            with h5py.File(path, "r") as f:
                t = f["traj_0"]
                primary = t["observations/image_primary"][:]
                wrist   = t["observations/image_wrist"][:]
                proprio = t["observations/proprio"][:].astype(np.float32)
                action_chunks = t["actions"][:].astype(np.float32)  # (T_5-1, H, 7)
                rewards = t["rewards"][:].astype(np.float32)
        except Exception as e:
            print(f"skip {path}: {e}")
            return None
        T = primary.shape[0]
        H = self.H
        steps = []
        for i in range(T):
            is_first = (i == 0); is_last = (i == T - 1)
            a = action_chunks[i] if i < T - 1 else np.zeros((H, 7), np.float32)
            r = float(rewards[i]) if i < T - 1 else 0.0
            steps.append({
                "observation": {
                    "image_primary": primary[i],
                    "image_wrist":   wrist[i],
                    "proprio":       proprio[i],
                },
                "action": a,
                "discount": np.float32(1.0),
                "reward":   np.float32(r),
                "is_first": is_first,
                "is_last":  is_last,
                "is_terminal": is_last,
                "language_instruction": INSTR[fruit],
            })
        return {"steps": steps,
                "episode_metadata": {"fruit": fruit, "file_path": path}}


def build_one(builder_cls, out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    builder = builder_cls(data_dir=str(out_root))
    builder.download_and_prepare()
    print(f"\n{builder.info.full_name} → {out_root}/{builder.info.full_name}")
    print(f"size: {builder.info.dataset_size}")


def write_stats(model_root: Path, src_root: Path, action_dim: int, action_key: str = "actions"):
    """Compute action mean/std/q01/q99 across training set."""
    arrs = []
    for fruit in FRUITS:
        for p in glob.glob(f"{src_root}/{fruit}/auto_*/trajectory.h5"):
            try:
                with h5py.File(p, "r") as f:
                    a = f[f"traj_0/{action_key}"][:].astype(np.float32)
                    if a.ndim == 3:  # action chunks (T, H, D) → flatten
                        a = a.reshape(-1, a.shape[-1])
                    arrs.append(a)
            except Exception:
                pass
    if not arrs:
        return
    A = np.concatenate(arrs, axis=0)
    stats = {
        "mean": A.mean(0).tolist(),
        "std":  A.std(0).tolist(),
        "q01":  np.quantile(A, 0.01, axis=0).tolist(),
        "q99":  np.quantile(A, 0.99, axis=0).tolist(),
        "min":  A.min(0).tolist(),
        "max":  A.max(0).tolist(),
        "num_actions": int(A.shape[0]),
    }
    with open(model_root / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"stats → {model_root}/dataset_statistics.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["openvla", "octo", "both"], default="both")
    args = ap.parse_args()
    if args.target in ("openvla", "both"):
        out = Path("/data/datasets/cuttingvla_openvla_rlds")
        build_one(OpenVLACutting, out)
        write_stats(out, Path(OpenVLACutting.SRC_ROOT), 7, "actions")
    if args.target in ("octo", "both"):
        out = Path("/data/datasets/cuttingvla_octo_rlds")
        build_one(OctoCutting, out)
        write_stats(out, Path(OctoCutting.SRC_ROOT), 7, "actions")


if __name__ == "__main__":
    main()

"""Repack step-level tfrecord shards into episode-level TFDS RLDS format.

Input  (our converters' output):
    <in_dir>/<fruit>/<orig_prefix>-train.tfrecord-XXXXX-of-YYYYY
    <in_dir>/<fruit>/dataset_statistics.json
    each tf.train.Example = one STEP, with is_first/is_last markers

Output (TFDS-RLDS layout):
    <out_dir>/<dataset_name>/<version>/
        features.json
        dataset_info.json
        <dataset_name>-train.tfrecord-XXXXX-of-YYYYY
    each tf.train.Example = one EPISODE, with sequence-typed step fields

Usage:
    python repack_to_rlds_episode.py --target octo  --fruit banana
    python repack_to_rlds_episode.py --target openvla --fruit apple
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

VERSION = "1.0.0"


# ======================================================================
# Per-target schemas (must match what our build_*_rlds.py wrote at step level)
# ======================================================================

def octo_features():
    """Octo episode-level RLDS — matches per-step features:
    image_primary 256, image_wrist 128, proprio[9], action[7], timestep, language."""
    return tfds.features.FeaturesDict({
        "steps": tfds.features.Dataset({
            "observation": tfds.features.FeaturesDict({
                "image_primary": tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8),
                "image_wrist":   tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8),
                "proprio":       tfds.features.Tensor(shape=(9,), dtype=tf.float32),
                "timestep":      tfds.features.Scalar(dtype=tf.int64),
            }),
            "action": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
            "is_first": tfds.features.Scalar(dtype=tf.bool),
            "is_last":  tfds.features.Scalar(dtype=tf.bool),
            "is_terminal": tfds.features.Scalar(dtype=tf.bool),
            "language_instruction": tfds.features.Text(),
        }),
        "episode_metadata": tfds.features.FeaturesDict({
            "episode_id": tfds.features.Scalar(dtype=tf.int64),
        }),
    })


def openvla_features():
    """OpenVLA episode-level RLDS — matches per-step features:
    image 224, state[9], action[7], discount, reward, is_first/last/terminal."""
    return tfds.features.FeaturesDict({
        "steps": tfds.features.Dataset({
            "observation": tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=(224, 224, 3), dtype=tf.uint8),
                "state": tfds.features.Tensor(shape=(9,), dtype=tf.float32),
            }),
            "action": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
            "discount": tfds.features.Scalar(dtype=tf.float32),
            "reward":   tfds.features.Scalar(dtype=tf.float32),
            "is_first": tfds.features.Scalar(dtype=tf.bool),
            "is_last":  tfds.features.Scalar(dtype=tf.bool),
            "is_terminal": tfds.features.Scalar(dtype=tf.bool),
            "language_instruction": tfds.features.Text(),
        }),
        "episode_metadata": tfds.features.FeaturesDict({
            "episode_id": tfds.features.Scalar(dtype=tf.int64),
        }),
    })


# ======================================================================
# Step-level proto parsers
# ======================================================================

OCTO_STEP_PROTO = {
    "observation/image_primary": tf.io.FixedLenFeature([], tf.string),
    "observation/image_wrist":   tf.io.FixedLenFeature([], tf.string),
    "observation/proprio":       tf.io.FixedLenFeature([9], tf.float32),
    "observation/timestep":      tf.io.FixedLenFeature([1], tf.int64),
    "action":                    tf.io.FixedLenFeature([7], tf.float32),
    "is_first":                  tf.io.FixedLenFeature([], tf.int64),
    "is_last":                   tf.io.FixedLenFeature([], tf.int64),
    "is_terminal":               tf.io.FixedLenFeature([], tf.int64),
    "language_instruction":      tf.io.FixedLenFeature([], tf.string),
}

OPENVLA_STEP_PROTO = {
    "observation/image":   tf.io.FixedLenFeature([], tf.string),
    "observation/state":   tf.io.FixedLenFeature([9], tf.float32),
    "action":              tf.io.FixedLenFeature([7], tf.float32),
    "discount":            tf.io.FixedLenFeature([1], tf.float32),
    "reward":              tf.io.FixedLenFeature([1], tf.float32),
    "is_first":            tf.io.FixedLenFeature([], tf.int64),
    "is_last":             tf.io.FixedLenFeature([], tf.int64),
    "is_terminal":         tf.io.FixedLenFeature([], tf.int64),
    "language_instruction": tf.io.FixedLenFeature([], tf.string),
}


def parse_step(target, raw):
    proto = OCTO_STEP_PROTO if target == "octo" else OPENVLA_STEP_PROTO
    p = tf.io.parse_single_example(raw, proto)
    out = {
        "is_first": bool(p["is_first"].numpy()),
        "is_last":  bool(p["is_last"].numpy()),
        "is_terminal": bool(p["is_terminal"].numpy()),
        "action": p["action"].numpy().astype(np.float32),
        "language_instruction": p["language_instruction"].numpy().decode("utf-8"),
    }
    if target == "octo":
        out["observation"] = {
            "image_primary": tf.io.decode_image(p["observation/image_primary"]).numpy(),
            "image_wrist":   tf.io.decode_image(p["observation/image_wrist"]).numpy(),
            "proprio":       p["observation/proprio"].numpy().astype(np.float32),
            "timestep":      int(p["observation/timestep"].numpy()[0]),
        }
    else:
        img = tf.io.decode_image(p["observation/image"])
        # openvla feature spec expects 224x224; raw frames are 256x256
        img = tf.image.resize(img, [224, 224], method="bilinear")
        img = tf.cast(img, tf.uint8).numpy()
        out["observation"] = {
            "image": img,
            "state": p["observation/state"].numpy().astype(np.float32),
        }
        out["discount"] = float(p["discount"].numpy()[0])
        out["reward"]   = float(p["reward"].numpy()[0])
    return out


def iter_episodes(target, in_glob):
    """Stream episodes from all step-level shards using is_first/is_last markers."""
    files = sorted(glob.glob(in_glob))
    ds = tf.data.TFRecordDataset(files)
    cur_steps = []
    eid = 0
    for raw in ds:
        s = parse_step(target, raw)
        if s["is_first"] and cur_steps:
            yield eid, cur_steps
            eid += 1
            cur_steps = []
        cur_steps.append(s)
        if s["is_last"]:
            yield eid, cur_steps
            eid += 1
            cur_steps = []
    if cur_steps:
        yield eid, cur_steps


# ======================================================================
# TFDS Builder shim — write metadata once with a single example, then
# stream-write the rest as plain tfrecord shards using the same Example proto
# ======================================================================

def make_builder_class(dataset_name: str, features, target: str, in_glob: str):
    """Dynamically create a tfds GeneratorBasedBuilder subclass bound to our shards."""
    class _Builder(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "Initial release"}
        name = dataset_name

        def _info(self):
            return tfds.core.DatasetInfo(
                builder=self,
                description=f"CuttingVLA per-fruit RLDS dataset ({dataset_name}).",
                features=features,
                supervised_keys=None,
            )

        def _split_generators(self, dl_manager):
            return {"train": self._generate_examples()}

        def _generate_examples(self):
            for eid, steps in iter_episodes(target, in_glob):
                yield eid, {
                    "steps": steps,
                    "episode_metadata": {"episode_id": eid},
                }
    _Builder.__name__ = f"Builder_{dataset_name}"
    return _Builder


def repack(target: str, fruit: str, in_root: Path, out_root: Path, dataset_name: str):
    in_glob = str(in_root / fruit / "*.tfrecord-*")
    features = octo_features() if target == "octo" else openvla_features()

    out_dir = out_root / dataset_name / VERSION
    # NOTE: do NOT pre-create out_dir — tfds will mistake it for an already-prepared
    # dataset and skip the download_and_prepare call silently. Let tfds make it.

    BuilderCls = make_builder_class(dataset_name, features, target, in_glob)
    builder = BuilderCls(data_dir=str(out_root))
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(),
        file_format="tfrecord",
    )
    # Copy the source dataset_statistics.json into the new dir so trainers can
    # find action quantiles without recomputing.
    src_stats = in_root / fruit / "dataset_statistics.json"
    if src_stats.exists() and out_dir.exists():
        (out_dir / "dataset_statistics.json").write_text(src_stats.read_text())
    n_shards = len(list(out_dir.glob("*.tfrecord-*"))) if out_dir.exists() else 0
    print(f"[done] {dataset_name} → {out_dir}  ({n_shards} shards)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["octo", "openvla"], required=True)
    ap.add_argument("--fruit", required=True,
                    help="apple / banana / orange / strawberry")
    ap.add_argument("--in-root", default=None,
                    help="defaults to /data/datasets/cuttingvla_{target}_rlds")
    ap.add_argument("--out-root", default=None,
                    help="defaults to /data/datasets/cuttingvla_{target}_tfds")
    args = ap.parse_args()

    in_root  = Path(args.in_root  or f"/data/datasets/cuttingvla_{args.target}_rlds")
    out_root = Path(args.out_root or f"/data/datasets/cuttingvla_{args.target}_tfds")
    dataset_name = f"cuttingvla_{args.fruit}"
    repack(args.target, args.fruit, in_root, out_root, dataset_name)


if __name__ == "__main__":
    main()

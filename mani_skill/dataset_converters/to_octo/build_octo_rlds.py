"""Octo RLDS converter (native h5 edition).

Per-step features:
    observation/image_primary   uint8 jpeg
    observation/image_wrist     uint8 jpeg (zeros — no wrist cam in bananacut env)
    observation/proprio         (9,) float32 qpos
    observation/timestep        int64
    action                      (A,) float32 (raw; normalized in loader via stats)
    language_instruction        bytes
    is_first / is_last / is_terminal

Emits `dataset_statistics.json` with action + proprio mean/std/min/max/p01/p99
(Octo's standard normalization source).

Usage:
    PYTHONPATH=/data:/data/mani_skill python \
        dataset_converters/to_octo/build_octo_rlds.py \
        --in /data/datasets/maniskill_mpm_pilot \
        --out /data/datasets/octo/maniskill_bananacut \
        --task bananacut
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_converters.common import iter_bundles


def _bytes(v): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
def _floats(v): return tf.train.Feature(float_list=tf.train.FloatList(value=v))
def _ints(v): return tf.train.Feature(int64_list=tf.train.Int64List(value=v))


def encode_step(img_primary, img_wrist, proprio, action, instruction,
                is_first, is_last, is_terminal, t):
    feats = {
        "observation/image_primary": _bytes(tf.io.encode_jpeg(img_primary).numpy()),
        "observation/image_wrist": _bytes(tf.io.encode_jpeg(img_wrist).numpy()),
        "observation/proprio": _floats(proprio.astype(np.float32).tolist()),
        "observation/timestep": _ints([int(t)]),
        "action": _floats(action.astype(np.float32).tolist()),
        "language_instruction": _bytes(instruction.encode("utf-8")),
        "is_first": _ints([int(is_first)]),
        "is_last": _ints([int(is_last)]),
        "is_terminal": _ints([int(is_terminal)]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feats))


def compute_stats(arr: np.ndarray) -> dict:
    return dict(
        mean=arr.mean(axis=0).tolist(),
        std=(arr.std(axis=0) + 1e-8).tolist(),
        min=arr.min(axis=0).tolist(),
        max=arr.max(axis=0).tolist(),
        p01=np.quantile(arr, 0.01, axis=0).tolist(),
        p99=np.quantile(arr, 0.99, axis=0).tolist(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--shard-size", type=int, default=64)
    ap.add_argument("--include-failed", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    bundles = []
    for b in iter_bundles(args.inp, args.task):
        passed = b.alignment.get("passed", False)
        if not args.include_failed and not (b.success and passed):
            continue
        bundles.append(b)
    assert bundles, f"no bundles under {args.inp}/{args.task}"
    print(f"[octo] {len(bundles)} bundles kept")

    # Stats
    all_act = np.concatenate([b.actions for b in bundles], axis=0)
    all_prop = np.concatenate([b.qpos[:-1] for b in bundles], axis=0)  # align with actions length
    stats = {
        "num_transitions": int(all_act.shape[0]),
        "num_trajectories": len(bundles),
        "action": compute_stats(all_act),
        "proprio": compute_stats(all_prop),
    }
    (args.out / "dataset_statistics.json").write_text(json.dumps(stats, indent=2))

    # Shards
    n_shards = (len(bundles) + args.shard_size - 1) // args.shard_size
    for s in range(n_shards):
        shard_bundles = bundles[s * args.shard_size:(s + 1) * args.shard_size]
        shard_file = args.out / f"octo_maniskill_{args.task}-train.tfrecord-{s:05d}-of-{n_shards:05d}"
        with tf.io.TFRecordWriter(str(shard_file)) as w:
            for b in shard_bundles:
                frames = b.load_frames()
                qpos = b.qpos
                actions = b.actions
                T = actions.shape[0]
                H, W = frames.shape[1], frames.shape[2]
                wrist = np.zeros((H, W, 3), np.uint8)  # no wrist camera
                for t in range(T):
                    ex = encode_step(
                        img_primary=frames[min(t, len(frames) - 1)],
                        img_wrist=wrist,
                        proprio=qpos[t],
                        action=actions[t],
                        instruction=b.instruction,
                        is_first=(t == 0),
                        is_last=(t == T - 1),
                        is_terminal=(t == T - 1 and b.success),
                        t=t,
                    )
                    w.write(ex.SerializeToString())
        for b in shard_bundles:
            b._frames = None; b._qpos = None; b._qvel = None
            b._actions = None; b._rewards = None
            b._art_state = None; b._block_state = None
        print(f"[octo] shard {s + 1}/{n_shards}: {shard_file.name}")

    print(f"[done] stats → {args.out / 'dataset_statistics.json'}")


if __name__ == "__main__":
    main()

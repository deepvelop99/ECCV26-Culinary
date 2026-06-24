"""OpenVLA RLDS converter (native h5 edition).

Reads `auto_<seed>/` from collect_rollouts_mpm and emits RLDS-style TFRecords
for OpenVLA finetuning.

Per-step features:
    observation/image       uint8 jpeg
    observation/state       (9,) float32   — qpos (panda 7 arm + 2 gripper)
    action                  (A,) float32   — same dims as saved (pd_ee_delta_pos: 4)
    discount                float32
    reward                  float32
    is_first / is_last / is_terminal  int64
    language_instruction    bytes

Plus `dataset_statistics.json` computed over all included episodes using
q01/q99 (OpenVLA's normalization).

Usage:
    PYTHONPATH=/data:/data/mani_skill python \
        dataset_converters/to_openvla/build_openvla_rlds.py \
        --in /data/datasets/maniskill_mpm_pilot \
        --out /data/datasets/openvla/maniskill_bananacut \
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


def encode_step(img, state, action, instruction,
                is_first, is_last, is_terminal):
    feats = {
        "observation/image": _bytes(tf.io.encode_jpeg(img).numpy()),
        "observation/state": _floats(state.astype(np.float32).tolist()),
        "action": _floats(action.astype(np.float32).tolist()),
        "discount": _floats([1.0]),
        "reward": _floats([float(is_terminal)]),
        "is_first": _ints([int(is_first)]),
        "is_last": _ints([int(is_last)]),
        "is_terminal": _ints([int(is_terminal)]),
        "language_instruction": _bytes(instruction.encode("utf-8")),
    }
    return tf.train.Example(features=tf.train.Features(feature=feats))


def compute_stats(arr: np.ndarray) -> dict:
    return {
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mask": [True] * (arr.shape[1] - 1) + [False],
    }


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
    print(f"[openvla] {len(bundles)} bundles kept")

    # Pass 1: stats (actions only; OpenVLA normalizes actions with q01/q99)
    all_actions = np.concatenate([b.actions for b in bundles], axis=0)
    stats = compute_stats(all_actions)
    (args.out / "dataset_statistics.json").write_text(json.dumps(stats, indent=2))

    # Pass 2: shards
    n_shards = (len(bundles) + args.shard_size - 1) // args.shard_size
    for s in range(n_shards):
        shard_bundles = bundles[s * args.shard_size:(s + 1) * args.shard_size]
        shard_file = args.out / f"maniskill_{args.task}-train.tfrecord-{s:05d}-of-{n_shards:05d}"
        with tf.io.TFRecordWriter(str(shard_file)) as w:
            for b in shard_bundles:
                frames = b.load_frames()  # (T_video, H, W, 3)
                qpos = b.qpos      # (T+1, 9)
                actions = b.actions  # (T, A)
                T = actions.shape[0]
                for t in range(T):
                    img = frames[min(t, len(frames) - 1)]
                    state = qpos[t]           # (9,)
                    action = actions[t]       # (A,)
                    ex = encode_step(
                        img=img,
                        state=state,
                        action=action,
                        instruction=b.instruction,
                        is_first=(t == 0),
                        is_last=(t == T - 1),
                        is_terminal=(t == T - 1 and b.success),
                    )
                    w.write(ex.SerializeToString())
        for b in shard_bundles:
            b._frames = None; b._qpos = None; b._qvel = None
            b._actions = None; b._rewards = None
            b._art_state = None; b._block_state = None
        print(f"[openvla] shard {s + 1}/{n_shards}: {shard_file.name}")

    print(f"[done] stats → {args.out / 'dataset_statistics.json'}")


if __name__ == "__main__":
    main()

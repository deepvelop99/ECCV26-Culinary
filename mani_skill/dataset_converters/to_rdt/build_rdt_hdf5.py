"""Build RDT-compatible HDF5 + PNG frames from the native co-sim layout.

RDT's `data/hdf5_maniskill_dataset.py` (thu-ml/RoboticsDiffusionTransformer)
expects:
    <out>/<task>/motionplanning/
        data.h5
            traj_0/obs/agent/qpos       (T+1, n_q)
            traj_0/actions              (T, A)
            traj_1/...
        <proc_idx>/<ep_idx>/1.png, 2.png, ...

with proc_idx = i // 100, ep_idx = i % 100 for episode i.

This script reads auto_<seed>/ produced by collect_rollouts_mpm.py and merges
into RDT's expected layout. Only success-passing episodes are included by
default.

Usage:
    PYTHONPATH=/data:/data/mani_skill python \
        dataset_converters/to_rdt/build_rdt_hdf5.py \
        --in /data/datasets/maniskill_mpm_pilot \
        --out /data/datasets/rdt-ft-data/demo_1k \
        --task bananacut
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from dataset_converters.common import iter_bundles, EpisodeBundle


def write_frames(bundle: EpisodeBundle, out_dir: Path, proc: int, ep: int):
    d = out_dir / str(proc) / str(ep)
    d.mkdir(parents=True, exist_ok=True)
    frames = bundle.load_frames()
    for t, img in enumerate(frames):
        Image.fromarray(img).save(d / f"{t + 1}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--eps-per-proc", type=int, default=100,
                    help="RDT loader uses proc_idx = i // N, ep_idx = i % N")
    ap.add_argument("--include-failed", action="store_true",
                    help="also include episodes where success==False / alignment not passed")
    ap.add_argument("--no-frames", action="store_true",
                    help="skip PNG frame writing (h5 only, smaller dataset)")
    ap.add_argument("--embed-frames", action="store_true",
                    help="embed image frames inside data.h5 under traj_<i>/obs/image (uint8 HxWx3)")
    args = ap.parse_args()

    task_out = args.out / args.task / "motionplanning"
    task_out.mkdir(parents=True, exist_ok=True)
    h5_path = task_out / "data.h5"

    with h5py.File(h5_path, "w") as h5:
        kept = 0
        for bundle in iter_bundles(args.inp, args.task):
            passed = bundle.alignment.get("passed", False)
            if not args.include_failed and not (bundle.success and passed):
                print(f"[skip] {bundle.root.name}  success={bundle.success}  align.passed={passed}")
                continue

            traj_group_name = f"traj_{kept}"
            g = h5.create_group(traj_group_name)
            og = g.create_group("obs").create_group("agent")
            og.create_dataset("qpos", data=bundle.qpos.astype(np.float32),
                              compression="gzip")
            og.create_dataset("qvel", data=bundle.qvel.astype(np.float32),
                              compression="gzip")
            g.create_dataset("actions", data=bundle.actions.astype(np.float32),
                             compression="gzip")
            g.attrs["instruction"] = bundle.instruction
            g.attrs["seed"] = bundle.seed
            g.attrs["source_auto_dir"] = str(bundle.root.name)
            g.attrs["control_mode"] = bundle.control_mode

            if args.embed_frames:
                frames = bundle.load_frames()
                og_obs = g["obs"]
                og_obs.create_dataset("image", data=frames,
                                      compression="gzip", compression_opts=4,
                                      chunks=(1, frames.shape[1], frames.shape[2], 3))
            elif not args.no_frames:
                proc = kept // args.eps_per_proc
                inner = kept % args.eps_per_proc
                write_frames(bundle, task_out, proc, inner)

            print(f"[rdt] merged {traj_group_name}  T={bundle.qpos.shape[0]}  "
                  f"← {bundle.root.name}")
            bundle._frames = None; bundle._qpos = None; bundle._qvel = None
            bundle._actions = None; bundle._rewards = None
            bundle._art_state = None; bundle._block_state = None
            kept += 1

    print(f"[done] {kept} episodes → {h5_path}")


if __name__ == "__main__":
    main()

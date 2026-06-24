"""Measure MPM particle-centroid canonical for each of the 7 fruits.

For each fruit's MPM yaml, loads MPMCuttingSim (headless), seeds particles
above the board, lets them settle for a few frames, then records the mean
particle xyz position (in MPM world frame).

Output: /data/datasets/fruit_canonical.json
    {
      "banana": {"canonical_xz": [-0.0419, 0.0041], "canonical_y": 0.0776},
      "apple":  {...}, ...
    }

Usage:
    PYTHONPATH=/data:/data/mani_skill TI_ARCH=cuda \
      /workspace/envs/maniskill/bin/python \
      /data/mani_skill/dataset_converters/mpm/measure_fruit_canonicals.py
"""
from __future__ import annotations
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

FRUITS = ["banana", "apple", "cucumber", "melon", "orange", "peach", "strawberry"]
# New fruits added for the addfruits-pilot. Measured separately so the legacy
# fruit_canonical.json (used in production) is not silently overwritten.
# NOTE: cherry/plum/lemon/kiwi/shine_muscat/tomato already measured and
# registered in frame.py. pear yaml was retuned — only re-measure it here.
NEW_FRUITS = ["pear"]
CONFIG_ROOT = Path("/data/EEF-Cutting-Simulation/configs")
OUT_PATH = Path("/data/datasets/fruit_canonical.json")
OUT_PATH_NEW = Path("/data/datasets/fruit_canonical_addfruits.json")

# Must be imported after taichi init
SETTLE_STEPS = 50


def _init_taichi():
    import taichi as ti
    try:
        ti.init(arch=ti.cuda, device_memory_fraction=0.5)
    except Exception:
        ti.init(arch=ti.cpu)


def _measure_one(fruit: str) -> dict:
    from sdf_utils.mesh_sdf import mesh_to_sdf
    from mpmcore.sim import MPMCuttingSim

    yaml_path = CONFIG_ROOT / f"{fruit}.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    def _pack(block, with_blade):
        transform = block.get("initial_transform", {})
        kwargs = {}
        if with_blade:
            kwargs["knife_blade"] = block.get("blade", {"axis": "Y", "fraction": 0.5})
        return mesh_to_sdf(
            block["mesh_path"],
            transform,
            int(block["sdf_voxel"]),
            **kwargs,
        )

    cutting_pack = _pack(cfg["cutting_mesh"], with_blade=False)
    knife_pack = _pack(cfg["knife"], with_blade=True)
    board_pack = _pack(cfg["board"], with_blade=False) if "board" in cfg else None

    cfg["cutting_mesh_pack"] = cutting_pack
    sim = MPMCuttingSim(
        cfg, cutting_pack, knife_pack, board_pack=board_pack, viewer=False,
    )

    # Get board support top (y)
    top = None
    if sim.board is not None:
        try:
            top = float(sim.board_top_y[None])
        except Exception:
            top = None

    sim.seed_particles_from_mesh(
        cutting_pack, cfg["cutting_mesh"], name="cutting_mesh",
        support_top_y=top,
    )

    # Settle: step a few substeps so particles rest on board
    for _ in range(SETTLE_STEPS):
        sim.step(sim.sim_time)

    # Read particle positions, use AABB midpoint (matches legacy measure_frame.py)
    arr = sim.particles.to_numpy()
    x = arr["x"]  # (N, 3) in MPM frame (x, y=up, z)
    lb = x.min(axis=0)
    ub = x.max(axis=0)
    mid = (lb + ub) / 2.0
    print(f"[{fruit}] N={x.shape[0]} aabb=[{lb.round(4).tolist()}, {ub.round(4).tolist()}] mid={mid.round(4).tolist()}  board_top_y={top}")
    return {
        "canonical_xz": [float(mid[0]), float(mid[2])],
        "canonical_y": float(mid[1]),
        "aabb_lb": lb.tolist(),
        "aabb_ub": ub.tolist(),
        "n_particles": int(x.shape[0]),
        "board_top_y": float(top) if top is not None else None,
    }


def _run_single_fruit_subprocess(fruit: str, timeout: int = 1200) -> dict:
    """Launch a fresh Python subprocess per fruit so Taichi state is isolated.
    Returns parsed JSON result dict (or {'error': str})."""
    import subprocess
    env = os.environ.copy()
    env["TI_ARCH"] = "cuda"
    script = __file__
    try:
        proc = subprocess.run(
            ["/workspace/envs/maniskill/bin/python", script, "--fruit", fruit],
            cwd="/data/EEF-Cutting-Simulation",
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout}s"}
    # Parse the last line which should be JSON
    last = proc.stdout.strip().splitlines()
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        return {"error": f"exit {proc.returncode}"}
    for line in reversed(last):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return {"error": "no JSON output"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", help="(internal) measure just this fruit and print JSON")
    ap.add_argument("--new-only", action="store_true",
                    help="measure only NEW_FRUITS (addfruits-pilot) and write to OUT_PATH_NEW")
    args = ap.parse_args()

    if args.fruit:
        # Child: measure one fruit, print JSON, exit
        _init_taichi()
        r = _measure_one(args.fruit)
        print(json.dumps(r))
        return

    fruits = NEW_FRUITS if args.new_only else FRUITS
    out = OUT_PATH_NEW if args.new_only else OUT_PATH
    # Per-fruit timeout: pear's mesh + SDF settle is ~3-4× heavier than the
    # rest, so give it more headroom to avoid wasting earlier results.
    HEAVY_FRUITS = {"pear"}

    # Parent: spawn one child per fruit. Always dump partial results so a
    # late failure does not throw away the earlier successful measurements.
    results = {}
    out.parent.mkdir(parents=True, exist_ok=True)
    for fruit in fruits:
        print(f"\n==== measuring {fruit} ====")
        try:
            timeout = 3600 if fruit in HEAVY_FRUITS else 1200
            results[fruit] = _run_single_fruit_subprocess(fruit, timeout=timeout)
        except Exception as e:
            print(f"[parent] {fruit} crashed: {e}")
            traceback.print_exc()
            results[fruit] = {"error": f"parent exception: {e}"}
        # Persist after EVERY fruit so we never lose progress.
        out.write_text(json.dumps(results, indent=2))
        print(f"[partial] wrote {out} ({len(results)}/{len(fruits)} done)")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()

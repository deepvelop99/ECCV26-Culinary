"""Measure MPM particle AABB canonical for addfruits not yet in FRUIT_CANONICAL.

Targets: grape, golden_strawberry, pear
Uses /data/Cutting_rubuttal2 (bridge3 MPM core + assets).

Usage (inside pod with GPU):
    cd /data/Cutting_rubuttal2
    PYTHONPATH=/data:/data/mani_skill python \
        /data/mani_skill/dataset_converters/mpm/measure_addfruits_canonicals.py
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path("/data/Cutting_rubuttal2")
FRUITS = ["grape", "golden_strawberry", "pear"]
SETTLE_STEPS = 200   # pear is heavy — give extra settle time
OUT = Path("/data/datasets/fruit_canonical_addfruits_b3.json")


def _init_taichi():
    import taichi as ti
    try:
        ti.init(arch=ti.cuda)
        print("[taichi] cuda ok")
    except Exception as e:
        print(f"[taichi] cuda failed ({e}), falling back to cpu")
        ti.init(arch=ti.cpu)


def _measure_one(fruit: str) -> dict:
    from sdf_utils.mesh_sdf import mesh_to_sdf
    from mpmcore.sim import MPMCuttingSim

    yaml_path = REPO / "configs" / "fruits" / f"{fruit}.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    prev_cwd = os.getcwd()
    os.chdir(str(REPO))
    try:
        def _pack(block, with_blade):
            transform = block.get("initial_transform", {})
            kw = {}
            if with_blade:
                kw["knife_blade"] = block.get("blade", {"axis": "Y", "fraction": 0.5})
            return mesh_to_sdf(block["mesh_path"], transform, int(block["sdf_voxel"]), **kw)

        cutting_pack = _pack(cfg["cutting_mesh"], with_blade=False)
        knife_pack   = _pack(cfg["knife"],         with_blade=True)
        board_pack   = _pack(cfg["board"],         with_blade=False) if "board" in cfg else None

        cfg["cutting_mesh_pack"] = cutting_pack
        sim = MPMCuttingSim(cfg, cutting_pack, knife_pack, board_pack=board_pack, viewer=False)

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

        t = 0.0
        dt_frame = float(sim.dt) * int(sim.substeps)
        for _ in range(SETTLE_STEPS):
            sim.step(t)
            t += dt_frame

        arr = sim.particles.to_numpy()
        x = arr["x"]   # (N, 3): x=MPM-x, y=up, z=MPM-z
        lb  = x.min(axis=0)
        ub  = x.max(axis=0)
        mid = (lb + ub) / 2.0
        print(f"[{fruit}] N={x.shape[0]}  "
              f"aabb_lb={lb.round(5).tolist()}  aabb_ub={ub.round(5).tolist()}  "
              f"mid={mid.round(5).tolist()}  board_top_y={top}")
        return {
            "canonical_xz": [round(float(mid[0]), 5), round(float(mid[2]), 5)],
            "canonical_y":   round(float(mid[1]), 5),
            "aabb_lb": lb.tolist(),
            "aabb_ub": ub.tolist(),
            "n_particles": int(x.shape[0]),
            "board_top_y": float(top) if top is not None else None,
        }
    finally:
        os.chdir(prev_cwd)


def main():
    sys.path.insert(0, str(REPO))
    _init_taichi()

    results = {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for fruit in FRUITS:
        print(f"\n{'='*40}")
        print(f"  measuring: {fruit}")
        print(f"{'='*40}")
        try:
            results[fruit] = _measure_one(fruit)
            print(f"[ok] {fruit}: xz={results[fruit]['canonical_xz']}  y={results[fruit]['canonical_y']}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[fruit] = {"error": str(e)}
            print(f"[FAIL] {fruit}: {e}")
        OUT.write_text(json.dumps(results, indent=2))
        print(f"[saved] {OUT}")

    print("\n\n=== RESULTS ===")
    for fruit, r in results.items():
        if "error" in r:
            print(f"  {fruit}: ERROR — {r['error']}")
        else:
            print(f'    "{fruit}":  {{"xz": {r["canonical_xz"]}, "y": {r["canonical_y"]}}},')

    print(f"\nFull results: {OUT}")


if __name__ == "__main__":
    main()

"""Object variation sampler.

Per CulinaryCut paper §3.3 / Fig 2d:
    - object_scale:     uniform in [0.8, 1.2]
    - object_pos_xy:    uniform in [-0.05, 0.05] m (offset from base spawn)
    - object_yaw:       uniform in [-pi, pi]
    - object_category:  one of 7 food meshes (banana, apple, cucumber, melon,
                        orange, peach, strawberry)

A `Variation` dict is:
    {"object": "banana",
     "mesh_path": "/data/mani_skill/assets/fruits/banana.obj",
     "scale": 1.02,
     "pos_offset": [0.01, -0.02, 0.0],
     "yaw": 1.3}
and is passed to env.reset(options={"variation": ...}).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


FRUITS_DIR = Path("/data/mani_skill/assets/fruits")

OBJECT_CATEGORIES: List[str] = [
    "banana", "apple", "cucumber", "melon", "orange", "peach", "strawberry",
]


def mesh_path(category: str) -> Path:
    p = FRUITS_DIR / f"{category}.obj"
    if not p.exists():
        raise FileNotFoundError(f"fruit mesh not found for '{category}': {p}")
    return p


def sample_variation(
    rng: np.random.Generator,
    *,
    categories: Optional[List[str]] = None,
    fixed_object: Optional[str] = None,
    scale_range: tuple = (0.8, 1.2),
    pos_x_range: tuple = (-0.02, 0.05),
    pos_y_range: tuple = (-0.05, 0.05),
    yaw_range: tuple = (-np.pi, np.pi),
) -> Dict[str, Any]:
    if fixed_object is not None:
        obj = fixed_object
    else:
        obj = rng.choice(categories or OBJECT_CATEGORIES)
    return {
        "object": str(obj),
        "mesh_path": str(mesh_path(obj)),
        "scale": float(rng.uniform(*scale_range)),
        "pos_offset": [
            float(rng.uniform(*pos_x_range)),
            float(rng.uniform(*pos_y_range)),
            0.0,
        ],
        "yaw": float(rng.uniform(*yaw_range)),
    }


def serialize(v: Dict[str, Any]) -> Dict[str, Any]:
    """Plain-json-safe snapshot."""
    return {
        "object": v["object"],
        "mesh_path": v["mesh_path"],
        "scale": float(v["scale"]),
        "pos_offset": list(map(float, v["pos_offset"])),
        "yaw": float(v["yaw"]),
    }

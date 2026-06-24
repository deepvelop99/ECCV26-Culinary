"""Render OBJ + mtl + texture properly via trimesh + pyrender (loads mtl chain
automatically, unlike Open3D which needs explicit albedo_img override)."""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from pathlib import Path
import numpy as np

import trimesh
import pyrender
from PIL import Image


SRC = Path("/data/Cutting_rebuttal/assets/fruits")
OUT = SRC / "_previews"
OUT.mkdir(exist_ok=True)


def render_fruit(name, size=640):
    obj = SRC / f"{name}.obj"
    if not obj.exists():
        print(f"skip {name}: no obj")
        return None
    # trimesh loads mtl + texture chain automatically
    mesh_or_scene = trimesh.load(str(obj), force='scene')

    scene = pyrender.Scene(bg_color=[0.95, 0.95, 0.95, 1.0],
                          ambient_light=[0.4, 0.4, 0.4])
    if isinstance(mesh_or_scene, trimesh.Scene):
        for n, geom in mesh_or_scene.geometry.items():
            pmesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
            scene.add(pmesh)
    else:
        pmesh = pyrender.Mesh.from_trimesh(mesh_or_scene, smooth=False)
        scene.add(pmesh)

    # camera + light
    bbox = mesh_or_scene.bounds if hasattr(mesh_or_scene, "bounds") else mesh_or_scene.extents
    if isinstance(bbox, np.ndarray) and bbox.shape == (2, 3):
        center = (bbox[0] + bbox[1]) / 2.0
        extent = float(np.linalg.norm(bbox[1] - bbox[0]))
    else:
        center = np.array([0.0, 0.0, 0.0])
        extent = 0.2
    eye = center + np.array([extent * 1.5, extent * 0.7, extent * 1.5])
    # Look-at matrix
    forward = (center - eye); forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    up2 = np.cross(right, forward)
    cam_pose = np.eye(4)
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = up2
    cam_pose[:3, 2] = -forward
    cam_pose[:3, 3] = eye
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4)
    scene.add(cam, pose=cam_pose)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0)
    scene.add(light, pose=cam_pose)

    r = pyrender.OffscreenRenderer(size, size)
    color, _ = r.render(scene)
    r.delete()
    out_path = OUT / f"{name}.png"
    Image.fromarray(color).save(out_path)
    print(f"{name}: {out_path}")
    return out_path


def main():
    fruits = ["apple", "banana", "orange", "peach", "strawberry",
              "avocado", "grape", "kiwi", "lemon", "mango", "pear",
              "persimmon", "pineapple_slice", "tomato", "watermelon_slice"]
    for f in fruits:
        try:
            render_fruit(f)
        except Exception as e:
            print(f"{f}: ERR {e}")


if __name__ == "__main__":
    main()

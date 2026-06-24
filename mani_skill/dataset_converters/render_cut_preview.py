"""Render cut pieces (left + right) at same pose with current ±CUT_GAP — shows
how separation looks per fruit."""
import os
os.environ.setdefault("EGL_PLATFORM", "device")
from pathlib import Path
import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat
import open3d as o3d
import open3d.visualization.rendering as rendering


CUT_PIECES = Path("/data/mani_skill/assets/fruits/update")
DST = Path("/data/Cutting_rebuttal/assets/fruits/_previews")
DST.mkdir(exist_ok=True)
CUT_GAP = 0.015


def fruit_rotation(fruit):
    if fruit in ("apple", "orange"):
        # mesh-X → world-Y
        q = euler2quat(np.pi / 2, 0, -np.pi / 2)
    else:
        # mesh-Z → world-Y
        q = euler2quat(np.pi / 2, 0, np.pi)
    return quat2mat(q)


def main():
    fruits = ["apple", "banana", "cucumber", "melon", "peach", "strawberry", "orange"]
    renderer = rendering.OffscreenRenderer(800, 600)
    for fruit in fruits:
        L_path = CUT_PIECES / fruit / f"{fruit}_left.obj"
        R_path = CUT_PIECES / fruit / f"{fruit}_right.obj"
        if not L_path.exists() or not R_path.exists():
            print(f"skip {fruit}: missing pieces")
            continue
        L = o3d.io.read_triangle_mesh(str(L_path), True)
        R = o3d.io.read_triangle_mesh(str(R_path), True)
        L.compute_vertex_normals()
        R.compute_vertex_normals()

        rot = fruit_rotation(fruit)
        # Apply lay-down rotation
        L.rotate(rot, center=(0, 0, 0))
        R.rotate(rot, center=(0, 0, 0))
        # Apply ±CUT_GAP separation in world-Y
        L.translate((0, +CUT_GAP, 0))
        R.translate((0, -CUT_GAP, 0))

        scene = renderer.scene
        scene.clear_geometry()
        scene.set_background([0.95, 0.95, 0.95, 1.0])
        scene.scene.set_sun_light([-0.5, -1.0, -0.5], [1.0, 1.0, 1.0], 75000)
        scene.scene.enable_sun_light(True)

        matL = rendering.MaterialRecord()
        matL.shader = "defaultLit"
        matL.base_color = [0.85, 0.40, 0.40, 1.0]  # red
        matR = rendering.MaterialRecord()
        matR.shader = "defaultLit"
        matR.base_color = [0.40, 0.55, 0.85, 1.0]  # blue
        scene.add_geometry("L", L, matL)
        scene.add_geometry("R", R, matR)

        # Camera looking from +x at the cut pair
        bbox = scene.bounding_box
        center = bbox.get_center()
        extent = float(np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound()))
        eye = center + np.array([extent * 1.5, extent * 0.6, 0.0])
        renderer.setup_camera(45.0, center.tolist(), eye.tolist(), [0, 0, 1])

        out = DST / f"cut_{fruit}.png"
        img = renderer.render_to_image()
        o3d.io.write_image(str(out), img)
        # gap measure
        Lc = np.asarray(L.vertices).mean(axis=0)
        Rc = np.asarray(R.vertices).mean(axis=0)
        print(f"{fruit}: L_center={np.round(Lc,3)}  R_center={np.round(Rc,3)}  "
              f"gap_Y={Lc[1]-Rc[1]:+.4f} m  → {out.name}")


if __name__ == "__main__":
    main()

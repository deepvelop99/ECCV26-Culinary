"""Render OBJ + texture preview to PNG (offscreen GPU)."""
import os
os.environ.setdefault("EGL_PLATFORM", "device")
from pathlib import Path
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


DST = Path("/data/Cutting_rebuttal/assets/fruits")
OUT = Path("/data/Cutting_rebuttal/assets/fruits/_previews")
OUT.mkdir(exist_ok=True)


def main():
    fruits = ["avocado", "grape", "kiwi", "lemon", "mango", "pear", "persimmon",
              "pineapple_slice", "tomato", "watermelon_slice",
              "apple", "banana", "orange", "peach", "strawberry"]

    renderer = rendering.OffscreenRenderer(640, 640)
    for fruit in fruits:
        obj_path = DST / f"{fruit}.obj"
        tex_path = DST / f"{fruit}_diffuse.jpg"
        if not obj_path.exists():
            print(f"skip {fruit}: no obj")
            continue
        # read_triangle_model loads OBJ + MTL chain (texture, normals, etc.)
        try:
            model = o3d.io.read_triangle_model(str(obj_path))
        except Exception as e:
            print(f"skip {fruit}: read_triangle_model err {e}")
            continue
        if not model.meshes:
            print(f"skip {fruit}: empty model")
            continue

        scene = renderer.scene
        scene.clear_geometry()
        scene.set_background([1.0, 1.0, 1.0, 1.0])
        scene.scene.set_sun_light([-0.5, -1.0, -0.5], [1.0, 1.0, 1.0], 75000)
        scene.scene.enable_sun_light(True)
        scene.scene.set_indirect_light_intensity(30000)
        scene.add_model(fruit, model)
        # Get mesh for camera bbox
        mesh = model.meshes[0].mesh
        mesh.compute_vertex_normals()

        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = float(np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound()))
        eye = center + np.array([extent * 1.5, extent * 1.0, extent * 1.5])
        renderer.setup_camera(45.0, center.tolist(), eye.tolist(), [0, 1, 0])

        out_path = OUT / f"{fruit}_render.png"
        img = renderer.render_to_image()
        o3d.io.write_image(str(out_path), img)
        print(f"{fruit}: {out_path}")


if __name__ == "__main__":
    main()

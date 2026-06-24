"""
bananacut_bgvar — bananacut with per-bg_idx visual+physical variation of the
table and cutting board, including a no-board mode where the fruit sits and
the cut surface lives on the table top directly.

Pattern (8 bgs with board, 7 without; ratio 8:7):
  has_board=True  → bgs 1, 3, 5, 7, 9, 11, 13, 15 — visual + collision board
                    rebuilt with per-bg thickness; cut_surface_z = thickness.
  has_board=False → bgs 2, 4, 6, 8, 10, 12, 14 — parent's board collision
                    pushed below ground; cut_surface_z = 0.0 (table top).

Per-bg cut surface (`cut_surface_z`) and tip clearance (`cut_tip_clearance`)
are exposed on the env so the collect script + bridge4 can read them and
adjust the floor / cut target Z dynamically. This replaces the previous
hardcoded `MS_BOARD_TOP_Z=0.020` assumption.

Render cameras: single base-view camera at 2048×2048 by default; set
`BGVAR_RENDER_RES` env var to override (e.g. "1024" for smaller files).
"""
import os
import os.path as osp
from typing import Optional, Sequence, Tuple

import numpy as np
import sapien
import sapien.render
import transforms3d
import trimesh

from mani_skill.envs.tasks.knife.bananacut import TableTopCuttingEnv
from mani_skill.utils.registration import register_env


# Lab floor z in TableSceneBuilder world coords (table top at z=0).
LAB_FLOOR_Z = -0.9196429
TABLE_TOP_Z = 0.0


def _asset_root() -> str:
    return osp.join(
        os.environ.get("MS_ASSET_DIR", os.path.expanduser("~/.maniskill")),
        "data", "scene_datasets",
    )


# -------- Per-bg variations --------
# Tables: auto-scaled so top sits at z=0 and bottom (legs) at z≈LAB_FLOOR_Z,
# unless `scale` is given explicitly. `wd_boost` (default 1.0) multiplies the
# horizontal extents AFTER auto-scale, so small tables can be made wider so the
# robot base (-0.615 m in x) sits comfortably on top.
TABLE_VARIATIONS: Sequence[Optional[dict]] = [
    None,  # bg_idx 0
    # 1
    {"glb": "ai2thor/ai2thor-hab/assets/objects/Dining_Table_201_1.glb",
     "yaw_deg": 0.0, "xy": (-0.12, 0.0)},
    # 2
    {"glb": "ai2thor/ai2thor-hab/assets/objects/Dining_Table_204_1.glb",
     "yaw_deg": 0.0, "xy": (-0.12, 0.0)},
    # 3
    {"glb": "ai2thor/ai2thor-hab/assets/objects/Dining_Table_208_1.glb",
     "yaw_deg": 0.0, "xy": (-0.12, 0.0)},
    # 4
    {"glb": "ai2thor/ai2thor-hab/assets/objects/Dining_Table_216_1.glb",
     "yaw_deg": 0.0, "xy": (-0.12, 0.0), "wd_boost": 1.3},
    # 5
    {"glb": "ai2thor/ai2thor-hab/assets/objects/Dining_Table_221_1.glb",
     "yaw_deg": 0.0, "xy": (-0.12, 0.0)},
    # 6
    {"glb": "ai2thor/ai2thor-hab/assets/objects/Dining_Table_223_1.glb",
     "yaw_deg": 0.0, "xy": (-0.12, 0.0)},
    # 7
    {"glb": "ai2thor/ai2thor-hab/assets/objects/Dining_Table_227_1.glb",
     "yaw_deg": 0.0, "xy": (-0.12, 0.0)},
    # 8
    {"glb": "ai2thor/ai2thor-hab/assets/objects/Dining_Table_230_1.glb",
     "yaw_deg": 0.0, "xy": (-0.12, 0.0)},
    # 9
    {"glb": "ai2thor/ai2thor-hab/assets/objects/Desk_204_1.glb",
     "yaw_deg": 0.0, "xy": (-0.12, 0.0)},
    # 10
    {"glb": "replica_cad_dataset/objects/frl_apartment_table_01.glb",
     "yaw_deg": 0.0, "xy": (-0.12, 0.0)},
    # 11
    {"glb": "replica_cad_dataset/objects/frl_apartment_table_02.glb",
     "yaw_deg": 0.0, "xy": (-0.12, 0.0)},
    # 12
    {"glb": "replica_cad_dataset/objects/frl_apartment_table_04.glb",
     "yaw_deg": 0.0, "xy": (-0.12, 0.0), "wd_boost": 1.3},
    # 13
    {"glb": "ai2thor/ai2thor-hab/assets/objects/Coffee_Table_201_1.glb",
     "yaw_deg": 0.0, "xy": (-0.12, 0.0), "wd_boost": 1.3},
    # 14
    {"glb": "ai2thor/ai2thor-hab/assets/objects/Dining_Table_204_1.glb",
     "yaw_deg": 90.0, "xy": (-0.12, 0.0)},
    # 15
    {"glb": "ai2thor/ai2thor-hab/assets/objects/Desk_204_1.glb",
     "yaw_deg": 90.0, "xy": (-0.12, 0.0)},
]
assert len(TABLE_VARIATIONS) == 16

# Boards: None = no-board (fruit cut directly on the table). Otherwise a dict
# with `thickness` (m) — the visual top + collision top + cut_surface_z all
# resolve to this thickness above the table top (z=0). Sizes are 1.3× the
# original pilot to give a more substantial cutting-board feel.
BOARD_VARIATIONS: Sequence[Optional[dict]] = [
    None,  # bg_idx 0 (passthrough)
    # bg 1 — has board: ReplicaCAD chopping board glb (light wood, scaled 1.3×)
    {"type": "glb",
     "path": "replica_cad_dataset/objects/frl_apartment_choppingboard_02.glb",
     "yaw_deg": 0.0, "scale": 1.3, "color": None,
     "thickness": 0.018},
    # bg 2 — no board
    None,
    # bg 3 — walnut rectangle (dark)
    {"type": "proc_box", "size_xy": (0.47, 0.60), "thickness": 0.024,
     "color": [0.30, 0.18, 0.08, 1.0],
     "thickness_alias": True},
    # bg 4 — no board
    None,
    # bg 5 — bamboo strip
    {"type": "proc_box", "size_xy": (0.44, 0.65), "thickness": 0.022,
     "color": [0.85, 0.70, 0.45, 1.0]},
    # bg 6 — no board
    None,
    # bg 7 — black slate (thin)
    {"type": "proc_box", "size_xy": (0.44, 0.62), "thickness": 0.014,
     "color": [0.10, 0.10, 0.12, 1.0]},
    # bg 8 — no board
    None,
    # bg 9 — round wood disk
    {"type": "proc_cyl", "radius": 0.30, "thickness": 0.020,
     "color": [0.65, 0.45, 0.25, 1.0]},
    # bg 10 — no board
    None,
    # bg 11 — walnut thick square-ish
    {"type": "proc_box", "size_xy": (0.50, 0.60), "thickness": 0.026,
     "color": [0.25, 0.13, 0.07, 1.0]},
    # bg 12 — no board
    None,
    # bg 13 — large oval (round disk wider)
    {"type": "proc_cyl", "radius": 0.34, "thickness": 0.016,
     "color": [0.75, 0.55, 0.30, 1.0]},
    # bg 14 — no board
    None,
    # bg 15 — ReplicaCAD chopping board honey-tinted, scaled 1.5×
    {"type": "glb",
     "path": "replica_cad_dataset/objects/frl_apartment_choppingboard_02.glb",
     "yaw_deg": 0.0, "scale": 1.5, "color": [0.78, 0.60, 0.35, 1.0],
     "thickness": 0.021},
]
assert len(BOARD_VARIATIONS) == 16
# Sanity: 8 with-board, 7 no-board.
assert sum(1 for b in BOARD_VARIATIONS[1:] if b is None) == 7
assert sum(1 for b in BOARD_VARIATIONS[1:] if b is not None) == 8


# Cutting-board collision footprint (always the same XY footprint regardless of
# visual shape — the visual may be smaller (e.g., a round disk visual with a
# rectangular collision underneath). Matches the original BOARD_SIZE half-XY.
_COLL_HALF_XY: Tuple[float, float] = (0.35, 0.55)
_BOARD_CENTER_XY: Tuple[float, float] = (-0.10, 0.0)


def _y_up_to_z_up_quat(yaw_deg: float) -> np.ndarray:
    qx = transforms3d.quaternions.axangle2quat(np.array([1, 0, 0]), np.deg2rad(90))
    qz = transforms3d.quaternions.axangle2quat(np.array([0, 0, 1]), np.deg2rad(yaw_deg))
    return transforms3d.quaternions.qmult(qz, qx)


def _rotated_z_bounds(path: str, q_wxyz: np.ndarray, scale: float = 1.0
                     ) -> Tuple[float, float]:
    m = trimesh.load(path, force='mesh')
    T = np.eye(4)
    T[:3, :3] = transforms3d.quaternions.quat2mat(q_wxyz) * float(scale)
    m2 = m.copy()
    m2.apply_transform(T)
    return float(m2.bounds[1, 2]), float(m2.bounds[0, 2])


def _hide_actor_visuals(actor) -> None:
    if actor is None:
        return
    objs = getattr(actor, "_objs", None)
    if not objs:
        return
    for ent in objs:
        try:
            rb = ent.find_component_by_type(sapien.render.RenderBodyComponent)
        except Exception:
            rb = None
        if rb is None:
            continue
        try:
            rb.visibility = 0
        except Exception:
            pass


def _move_actor_far(actor, z: float = -100.0) -> None:
    """Move a static actor below ground so its collision/visual is out of the
    way. Visual is also hidden as a safety belt."""
    if actor is None:
        return
    try:
        actor.set_pose(sapien.Pose(p=[0, 0, float(z)]))
    except Exception:
        pass
    _hide_actor_visuals(actor)


def _build_visual_table(scene, abs_path: str, world_xy: Tuple[float, float],
                       yaw_deg: float, scale_override: Optional[float],
                       wd_boost: float, name: str):
    """Visual-only static actor for a Y-up table glb. Auto-scales (unless
    scale_override is given) so top sits at z=0 and bottom at z≈LAB_FLOOR_Z.
    `wd_boost` is a non-uniform horizontal multiplier applied after auto-scale
    (height stays matched to floor)."""
    q = _y_up_to_z_up_quat(yaw_deg)
    z_top_unit, z_bot_unit = _rotated_z_bounds(abs_path, q, scale=1.0)
    if scale_override is not None:
        scale = float(scale_override)
    else:
        height = z_top_unit - z_bot_unit
        scale = (TABLE_TOP_Z - LAB_FLOOR_Z) / height if height > 1e-6 else 1.0
    z_top = z_top_unit * scale
    p_z = TABLE_TOP_Z - z_top
    p = [float(world_xy[0]), float(world_xy[1]), float(p_z)]
    s = float(scale)
    boost = float(wd_boost)
    # Non-uniform: width/depth × wd_boost, height × s.
    builder = scene.create_actor_builder()
    builder.add_visual_from_file(abs_path, scale=(s * boost, s * boost, s))
    builder.initial_pose = sapien.Pose(p=p, q=q.tolist())
    return builder.build_static(name=name), s, boost


def _build_visual_board_yup(scene, abs_path: str, world_xy: Tuple[float, float],
                           world_top_z: float, yaw_deg: float, scale: float,
                           color: Optional[Sequence[float]], name: str):
    q = _y_up_to_z_up_quat(yaw_deg)
    z_top, _ = _rotated_z_bounds(abs_path, q, scale=scale)
    p = [float(world_xy[0]), float(world_xy[1]), float(world_top_z - z_top)]
    builder = scene.create_actor_builder()
    if color is not None:
        mat = sapien.render.RenderMaterial(base_color=list(color))
        builder.add_visual_from_file(abs_path, scale=(scale,) * 3, material=mat)
    else:
        builder.add_visual_from_file(abs_path, scale=(scale,) * 3)
    builder.initial_pose = sapien.Pose(p=p, q=q.tolist())
    return builder.build_static(name=name)


def _build_visual_proc_box(scene, world_xy: Tuple[float, float],
                           world_top_z: float, size_xy: Tuple[float, float],
                           thickness: float, color: Sequence[float],
                           name: str):
    half = (float(size_xy[0]) / 2, float(size_xy[1]) / 2, float(thickness) / 2)
    builder = scene.create_actor_builder()
    mat = sapien.render.RenderMaterial(base_color=list(color))
    builder.add_box_visual(half_size=half, material=mat)
    p = [float(world_xy[0]), float(world_xy[1]),
         float(world_top_z - thickness / 2)]
    builder.initial_pose = sapien.Pose(p=p)
    return builder.build_static(name=name)


def _build_visual_proc_cyl(scene, world_xy: Tuple[float, float],
                           world_top_z: float, radius: float,
                           thickness: float, color: Sequence[float],
                           name: str):
    q = transforms3d.quaternions.axangle2quat(np.array([0, 1, 0]),
                                              np.deg2rad(90))
    builder = scene.create_actor_builder()
    mat = sapien.render.RenderMaterial(base_color=list(color))
    builder.add_cylinder_visual(radius=float(radius),
                                half_length=float(thickness) / 2,
                                material=mat,
                                pose=sapien.Pose(q=q.tolist()))
    p = [float(world_xy[0]), float(world_xy[1]),
         float(world_top_z - thickness / 2)]
    builder.initial_pose = sapien.Pose(p=p)
    return builder.build_static(name=name)


def _build_collision_board(scene, world_xy: Tuple[float, float],
                          half_xy: Tuple[float, float], thickness: float,
                          name: str):
    """Build a static actor with a box collision matching the visual board's
    thickness. Top at z = thickness, bottom at z=0."""
    half = (float(half_xy[0]), float(half_xy[1]), float(thickness) / 2)
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=half)
    builder.initial_pose = sapien.Pose(
        p=[float(world_xy[0]), float(world_xy[1]), float(thickness) / 2])
    return builder.build_static(name=name)


@register_env("bananacut_bgvar", max_episode_steps=1000)
class TableTopCuttingBGVarEnv(TableTopCuttingEnv):
    """bananacut + per-bg_idx visual & physical variation for table and board.

    Exposes `cut_surface_z` (m) and `cut_tip_clearance` (m) on the env so the
    collect script + bridge4 can compute the per-episode floor dynamically:

      cut_surface_z       = thickness (with board) or 0.0 (no board)
      cut_tip_clearance   = 0.005 (5 mm) — knife tip stops this far above
                            the cut surface.

    Single render camera (base view), 2048×2048 by default. Set
    BGVAR_RENDER_RES to override.
    """

    # Default exposed values; overridden in _load_scene per bg_idx.
    cut_surface_z: float = 0.020   # parent default: board top
    cut_tip_clearance: float = 0.005

    @property
    def _default_human_render_camera_configs(self):
        from mani_skill.sensors.camera import CameraConfig
        from mani_skill.utils import sapien_utils
        try:
            res = int(os.environ.get("BGVAR_RENDER_RES", "2048"))
        except ValueError:
            res = 2048
        return [
            CameraConfig(
                uid="render_camera",
                pose=sapien_utils.look_at(eye=[0.5, -0.5, 0.8],
                                          target=[0, 0, 0]),
                width=res, height=res, fov=1.2, near=0.01, far=100,
            ),
        ]

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        bg_idx = int(getattr(self, "_bg_idx", 0) or 0)

        # Defaults (will be overridden below if bg_idx in [1..15]).
        self.cut_surface_z = 0.020
        self.cut_tip_clearance = 0.005

        if bg_idx < 1 or bg_idx >= len(TABLE_VARIATIONS):
            return  # bg_idx 0 keeps default appearance & physics

        # Always: hide parent's lab table visual + push parent's board out of
        # the way (we will rebuild the board collision per-bg below).
        _hide_actor_visuals(getattr(self.table_scene, "table", None))
        _move_actor_far(getattr(self, "board", None), z=-100.0)

        root = _asset_root()

        # ---- Table visual (always present) ----
        tcfg = TABLE_VARIATIONS[bg_idx]
        try:
            tpath = osp.join(root, tcfg["glb"])
            self._bgvar_table, _scale, _wd = _build_visual_table(
                self.scene, tpath,
                world_xy=tcfg.get("xy", (-0.12, 0.0)),
                yaw_deg=float(tcfg.get("yaw_deg", 0.0)),
                scale_override=tcfg.get("scale"),
                wd_boost=float(tcfg.get("wd_boost", 1.0)),
                name=f"bgvar_table_{bg_idx}",
            )
            print(f"[bgvar] table glb={osp.basename(tpath)} bg_idx={bg_idx} "
                  f"yaw={tcfg.get('yaw_deg', 0.0)} scale={_scale:.3f} "
                  f"wd_boost={_wd:.2f}")
        except Exception as e:
            print(f"[bgvar] custom table build failed (bg_idx={bg_idx}): {e}")
            self._bgvar_table = None

        # ---- Board (optional) ----
        bcfg = BOARD_VARIATIONS[bg_idx]
        if bcfg is None:
            # No board: fruit will land directly on the table top (z=0).
            self.cut_surface_z = 0.0
            self._bgvar_board = None
            self._bgvar_board_collision = None
            print(f"[bgvar] bg_idx={bg_idx} has_board=False "
                  f"cut_surface_z={self.cut_surface_z:.3f}")
        else:
            t = bcfg["type"]
            # thickness controls cut_surface_z. For glb boards we trust the
            # explicit `thickness` field rather than measuring (glb origins may
            # not center on geometry).
            thickness = float(bcfg.get("thickness", 0.020))
            self.cut_surface_z = thickness

            # Rebuild the cutting-board collision box at the right thickness so
            # the knife rests on the actual visual top, not on parent's 0.020
            # collision. Footprint matches original BOARD_SIZE.
            try:
                self._bgvar_board_collision = _build_collision_board(
                    self.scene,
                    world_xy=_BOARD_CENTER_XY,
                    half_xy=_COLL_HALF_XY,
                    thickness=thickness,
                    name=f"bgvar_board_collision_{bg_idx}",
                )
            except Exception as e:
                print(f"[bgvar] board collision build failed (bg_idx={bg_idx}): {e}")
                self._bgvar_board_collision = None

            # Build visual board on top of the collision.
            try:
                if t == "glb":
                    bpath = osp.join(root, bcfg["path"])
                    self._bgvar_board = _build_visual_board_yup(
                        self.scene, bpath,
                        world_xy=_BOARD_CENTER_XY,
                        world_top_z=thickness,
                        yaw_deg=float(bcfg.get("yaw_deg", 0.0)),
                        scale=float(bcfg.get("scale", 1.0)),
                        color=bcfg.get("color"),
                        name=f"bgvar_board_glb_{bg_idx}",
                    )
                elif t == "proc_box":
                    self._bgvar_board = _build_visual_proc_box(
                        self.scene,
                        world_xy=_BOARD_CENTER_XY,
                        world_top_z=thickness,
                        size_xy=tuple(bcfg["size_xy"]),
                        thickness=thickness,
                        color=bcfg["color"],
                        name=f"bgvar_board_box_{bg_idx}",
                    )
                elif t == "proc_cyl":
                    self._bgvar_board = _build_visual_proc_cyl(
                        self.scene,
                        world_xy=_BOARD_CENTER_XY,
                        world_top_z=thickness,
                        radius=float(bcfg["radius"]),
                        thickness=thickness,
                        color=bcfg["color"],
                        name=f"bgvar_board_cyl_{bg_idx}",
                    )
                else:
                    raise ValueError(f"unknown board type {t}")
                print(f"[bgvar] bg_idx={bg_idx} has_board=True type={t} "
                      f"thickness={thickness:.3f} cut_surface_z={self.cut_surface_z:.3f}")
            except Exception as e:
                print(f"[bgvar] custom board build failed (bg_idx={bg_idx}): {e}")
                self._bgvar_board = None

        # Sync parent's `board_center[2]` so its cut-detection / fruit-placement
        # logic uses the new cut surface. Parent computes:
        #   _board_anchor = board_center[2] + BOARD_SIZE[2]
        # We want this to equal cut_surface_z, so:
        self.board_center[2] = float(self.cut_surface_z) - float(self.BOARD_SIZE[2])

from typing import Dict
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat, euler2mat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig

import trimesh  # STL 로드용
import os
from typing import Any, Dict, Union
import numpy as np
import sapien
import torch

# ==========================
# 설정 / 경로 / 파라미터
# ==========================
KNIFE_STL_PATH = "/data/mani_skill/assets/robots/panda/franka_description/meshes/visual/knifev2.stl"

# Cut-piece extra separation gap (world-Y). All fruits are laid down such that
# their natural cut-split axis maps to world-Y (see block_rot_quat below).
_CUT_GAP = 0.025  # legacy fruits (banana/apple/etc): per-piece offset.
# New fruits use a larger gap because their world scale (~1-3 cm fruit) makes
# 25 mm separation visually ambiguous — bump to 50 mm so cut is unmistakable.
_CUT_GAP_NEW = 0.050
# Per-fruit override: lemon and pear are long enough that a 5 cm offset still
# leaves the two halves visually overlapping at the end-frame. Bump to 80 mm.
_CUT_GAP_NEW_PER_FRUIT = {
    "lemon": 0.080,
    "pear":  0.080,
}

# Fruits that use identity rotation (mesh-axis = world-axis) AND whose
# _left/_right cut meshes have origin at fruit base (not cut plane). Used to
# branch sign + cut-plane offset logic in the cut-piece swap. Keep in sync
# with block_rot_quat assignment.
_NEW_FRUITS = {"cherry", "grape", "shine_muscat", "golden_strawberry",
               "plum", "pear", "lemon", "kiwi", "tomato"}

# URDF hand→knife 고정 변환
HAND_TO_KNIFE_XYZ = (0.1, 0.0, 0.1)
HAND_TO_KNIFE_RPY = (1.5708, -1.5708, -1.5708)

# 접촉 여유(칼팁 기준)
CONTACT_Z_MARGIN  = 0.006   # z 마진(6mm)
CONTACT_XY_MARGIN = 0.02    # x,y 마진(2cm)

USE_FORCE_CONTACT = False
KNIFE_LINK_NAME   = "tool_knife"
CONTACT_F_THR     = 0.5     # N


@register_env("bananacut", max_episode_steps=1000)
class TableTopCuttingEnv(BaseEnv):
    
    _sample_video_link = None
    KNIFE_TIP_LOCAL_Z_DEFAULT = -0.20

    BOARD_THICKNESS = 0.02
    BOARD_SIZE = np.array([0.35, 0.55, BOARD_THICKNESS / 2])

    BLOCK_HALF_SIZE = np.array([0.1, 0.1, 0.1])
    KERF = 0.004
    CUT_DEPTH_MARGIN = 0.003

    SUPPORTED_REWARD_MODES = ["none"]
    SUPPORTED_ROBOTS = ["pk", "pkv","pkours"]

    def __init__(self, *args, robot_uids="pkv", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

        self._prev_tcp_z = None
        self._contact = None
        self._contact_prev = None
        self._contact_enter_frame = None
        self._contact_exit_frame = None
        self._contact_frame = None

        self._knife_local_tip_z = self.KNIFE_TIP_LOCAL_Z_DEFAULT
        self._printed_tip_once = False
        self._already_cut = None
        self._ever_success = None
    # ---------- 기본 설정 ----------
    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=100,
            control_freq=20,
            scene_config=SceneConfig(
                contact_offset=0.005,
                solver_position_iterations=8,
                solver_velocity_iterations=1,
            ),
        )

    @property
    def _default_sensor_configs(self):
        # base_camera at 256x256 (VLA-standard input resolution; OpenVLA/Octo/RDT
        # accept 224 or 256 px). Stored in h5 when obs_mode="rgb".
        cams = [
            CameraConfig(
                uid="base_camera",
                pose=sapien_utils.look_at(
                    eye=[0.5, -0.5, 0.8],
                    target=[0, 0, 0]
                ),
                width=256,
                height=256,
                fov=1.2,
                near=0.01,
                far=100,
            ),
        ]
        # Optional 4K review camera, gated on env var PILOT_HIRES=1 so prod
        # collection h5 size isn't bloated. Adds ~1.5GB/ep.
        if os.environ.get("PILOT_HIRES", "0") == "1":
            cams.append(CameraConfig(
                uid="review_camera",
                pose=sapien_utils.look_at(
                    eye=[1.0, -1.0, 0.7],
                    target=[0, 0, 0.3]
                ),
                width=256,
                height=256,
                fov=1.0,
                near=0.01,
                far=100,
            ))
        return cams
    # @property
    # def _default_human_render_camera_configs(self):
    #     if self.robot_uids == "fetch":
    #         room_camera_pose = sapien_utils.look_at([2.5, -2.5, 3], [0.0, 0.0, 0])
    #         room_camera_config = CameraConfig(
    #             "render_camera",
    #             room_camera_pose,
    #             512,
    #             512,
    #             1,
    #             0.01,
    #             100,
    #         )
    #         robot_camera_pose = sapien_utils.look_at([2, 0, 1], [0, 0, -1])
    #         robot_camera_config = CameraConfig(
    #             "robot_render_camera",
    #             robot_camera_pose,
    #             512,
    #             512,
    #             1.5,
    #             0.01,
    #             100,
    #             mount=self.agent.torso_lift_link,
    #         )
    #         return [room_camera_config, robot_camera_config]

    #     if self.robot_uids == "panda":
    #         pose = sapien_utils.look_at([0.4, 0.4, 0.8], [0.0, 0.0, 0.4])
    #     else:
    #         pose = sapien_utils.look_at([0, 10, -3], [0, 0, 0])
    #     return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
    # @property
    # def _default_human_render_camera_configs(self):
    #     pose = sapien_utils.look_at(eye=[0.5, 0.8, 0.35], target=[-0.1, 0, 0.05])
    #     return CameraConfig(
    #         "render_camera",
    #         pose=pose,
    #         width=1280,
    #         height=960,
    #         fov=1.2,
    #         near=0.01,
    #         far=100,
    #     )
    
    @property
    def _default_human_render_camera_configs(self):
        # Two render cams for the saved video grid:
        #  - "render_camera": tight close-up on cutting (existing).
        #  - "wide_camera":   far-back high view so apartment backdrop fills frame.
        return [
            CameraConfig(
                uid="render_camera",
                pose=sapien_utils.look_at(
                    eye=[0.5, -0.5, 0.8],
                    target=[0, 0, 0]
                ),
                width=256,
                height=256,
                fov=1.2,
                near=0.01,
                far=100,
            ),
            CameraConfig(
                uid="wide_camera",
                # Wide-shot looking at cutting setup from (+x, -y, +z) toward
                # origin; view direction (-x, +y, -z) so apartment placed at
                # offset (-6, +10) appears as distant backdrop in frame.
                pose=sapien_utils.look_at(
                    eye=[1.5, -1.5, 1.0],
                    target=[-0.5, +0.5, 0.4]
                ),
                width=512,
                height=384,
                fov=1.2,
                near=0.05,
                far=30,
            ),
        ]

    # ---------- 로딩 ----------
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build()

        # Background variation: pick a "scene preset" per episode so RGB has
        # visual diversity for VLA training. Color triples = (r,g,b) in [0,1].
        variation = (options or {}).get("variation") or {}
        scene_idx = int(variation.get("scene_idx", 0))
        SCENE_PRESETS = [
            # (board_color,           ground_color,           ambient,         sun_dir)
            ([0.95, 0.92, 0.85],      [0.85, 0.85, 0.85],     [0.30, 0.30, 0.30], [-0.5, -1.0, -0.5]),
            ([0.55, 0.35, 0.20],      [0.40, 0.45, 0.35],     [0.35, 0.30, 0.25], [+0.3, -1.0, -0.4]),
            ([0.92, 0.92, 0.95],      [0.20, 0.22, 0.25],     [0.25, 0.27, 0.30], [-0.4, -1.0, +0.3]),
            ([0.30, 0.45, 0.60],      [0.80, 0.78, 0.72],     [0.30, 0.32, 0.35], [+0.5, -1.0, -0.2]),
            ([0.85, 0.55, 0.45],      [0.60, 0.55, 0.50],     [0.35, 0.32, 0.28], [-0.2, -1.0, +0.5]),
        ]
        preset = SCENE_PRESETS[scene_idx % len(SCENE_PRESETS)]
        board_rgb, ground_rgb, ambient_rgb, sun_dir = preset
        # Board color is unified to white across all variations (per request).
        board_rgb = [1.0, 1.0, 1.0]
        self._scene_idx = scene_idx
        self._scene_preset = preset

        # Re-color the table-scene ground if accessible (TableSceneBuilder
        # exposes self.ground after build). Best-effort — skip on failure.
        try:
            ground_mat = sapien.render.RenderMaterial(base_color=ground_rgb + [1.0])
            for visual in self.table_scene.ground.find_component_by_type(
                sapien.render.RenderBodyComponent).render_shapes:
                visual.material = ground_mat
        except Exception:
            pass

        # board with per-scene color
        board_mat = sapien.render.RenderMaterial(base_color=board_rgb + [1.0])
        board_builder = self.scene.create_actor_builder()
        board_builder.add_box_visual(half_size=self.BOARD_SIZE.tolist(), material=board_mat)
        board_builder.add_box_collision(half_size=self.BOARD_SIZE.tolist())
        board_builder.initial_pose = sapien.Pose(p=[-0.1, 0, self.BOARD_SIZE[2]])
        self.board = board_builder.build_static(name="board")

        # Optional ReplicaCAD apartment-stage backdrop. bg_idx in {0..5}:
        #   0 = no apartment (default lab look)
        #   1..5 = ReplicaCAD stages 0..4 (see _RCAD_STAGES below)
        # Stage glb is loaded as a static visual+nonconvex actor offset away
        # from the cutting volume so Franka/board/fruit poses stay fixed.
        bg_idx = int(variation.get("bg_idx", 0))
        self._bg_idx = bg_idx
        if bg_idx > 0:
            # Use the original TableSceneBuilder lab table (2.42m × 1.21m) —
            # same desk as the pre-bg pilot. Apt's largest open room (5.7m
            # diameter, 2.85m wall clearance) fits the lab table comfortably.
            self._load_replica_stage(bg_idx - 1)
            # Hide TableSceneBuilder's checkered ground plane (z-fights with
            # the apt's textured floor at z=-0.9196). Move it far below scene.
            try:
                self.table_scene.ground.set_pose(sapien.Pose(p=[0, 0, -100]))
                print("[bg] lab-ground moved to z=-100 (hidden)")
            except Exception as e:
                print(f"[bg] hide ground failed: {e}")

        # Variation support: read options["variation"] for mesh_path / scale.
        # yaw goes into _initialize_episode (affects pose only, not scene build).
        # If absent, fall back to banana defaults.
        variation = (options or {}).get("variation") or {}
        mesh_path = variation.get("mesh_path",
                                  "/data/mani_skill/assets/fruits/banana.obj")
        scale = float(variation.get("scale", 1.0))
        # Record on env so post-cut swap logic can reuse (or extend to use per-fruit cut pieces).
        self._variation_mesh_path = mesh_path
        self._variation_scale = scale
        self._variation_object = variation.get("object", "banana")

        from pathlib import Path as _P
        _mesh_path = _P(mesh_path).expanduser()
        _mesh = trimesh.load(str(_mesh_path), force='mesh')
        self._mesh_centroid_local = np.asarray(_mesh.centroid, dtype=np.float32) * float(scale)
        # Cache mesh vertices (scaled) for accurate per-rotation zmin computation
        # in _initialize_episode. After rotation by init_quat we need the world
        # zmin of the rotated mesh to lift the fruit so its lowest point is
        # 1cm above the board top — not half-buried inside it.
        self._mesh_verts_local = np.asarray(_mesh.vertices, dtype=np.float32) * float(scale)

        # apple (visual only, no collision!)
        self.block_apple = []

        for env_idx in range(self.num_envs):
            from pathlib import Path
            p = Path(mesh_path).expanduser()
            assert p.exists(), f"OBJ not found: {p}"
            obj_builder = self.scene.create_actor_builder()
            obj_builder.set_scene_idxs([env_idx])
            # Dynamic actor w/ near-zero collision (1e-1000 underflows to 0).
            # build_kinematic hung during Taichi JIT + RecordEpisode's state
            # recording; dynamic path keeps co-sim stable. Known minor artifact:
            # the fruit may nudge up <1 cm in Z on t=0 as physics resolves the
            # tiny collision against the board - acceptable for Tier 1.
            obj_builder.add_sphere_collision(radius=1e-1000)
            obj_builder.add_visual_from_file(str(p), scale=[scale, scale, scale])
            obj_builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 1.0])
            # Static — knife/board contact won't drive the fruit through the
            # board. We still call set_pose() in _initialize_episode to place
            # the fruit; static actors honor explicit pose updates.
            self.block_apple.append(obj_builder.build_static(name=f"block_apple_{env_idx}"))

        # buffers
        self._ensure_buffers()
        self._prev_tcp_z = torch.zeros(self.num_envs, dtype=torch.float32)
        self._contact = torch.zeros(self.num_envs, dtype=torch.bool)
        self._contact_prev = torch.zeros(self.num_envs, dtype=torch.bool)
        
        self.board_center = np.array([-0.1, 0.0, self.BOARD_SIZE[2]])
        self.block_center0 = self.board_center + np.array([0.0, 0.0, self.BLOCK_HALF_SIZE[2]])
        self.block_center1 = self.board_center
        # Per-fruit lay-down rotation, chosen so cut-piece split axis
        # (apple/orange cut along mesh-X, others along mesh-Z) always maps to
        # world-Y — which is the axis the knife actually cuts through.
        fruit = getattr(self, "_variation_object", "banana")
        # New fruits split along different mesh axes — pick a lay-down rotation
        # that puts each fruit's split axis onto world-Y (the knife's cut axis).
        # Identity (no rotation) puts cherry-style mesh-Z splits onto world-Z
        # (vertical) which makes cuts visually go top↔bottom — the previous bug.
        _NEW_MESH_Z = {"cherry", "plum", "shine_muscat", "pear",
                       "grape", "golden_strawberry"}
        _NEW_MESH_X = {"kiwi", "tomato"}
        _NEW_MESH_Y = {"lemon"}
        if fruit in _NEW_MESH_Z:
            # mesh-Z → world-Y (same family as banana). Lay fruit on its side.
            self.block_rot_quat = euler2quat(np.pi/2, 0, 0)
        elif fruit in _NEW_MESH_X:
            # mesh-X → world-Y (same family as apple/orange).
            self.block_rot_quat = euler2quat(np.pi/2, 0, -np.pi/2)
        elif fruit in _NEW_MESH_Y:
            # mesh-Y → world-Y already. Identity is fine.
            self.block_rot_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        elif fruit in ("apple", "orange"):
            # mesh-X → world-Y (cut axis aligned with knife), mesh-Y → world-Z (height)
            self.block_rot_quat = euler2quat(np.pi/2, 0, -np.pi/2)
        else:
            # banana / cucumber / melon / peach / strawberry: mesh-Z → world-Y
            self.block_rot_quat = euler2quat(np.pi/2, 0, np.pi)
        self.update_block_rot_quat = self.block_rot_quat  # cut pieces keep same rotation
        # self.block_rot_quat = euler2quat(0, 0, np.pi/2) # Chang young HB

        try:
            ambient = self._scene_preset[2]
            sun_dir = self._scene_preset[3]
            self.scene.set_ambient_light(ambient)
            self.scene.add_directional_light(sun_dir, [1, 1, 1], shadow=False)
        except Exception:
            self.scene.set_ambient_light([0.4, 0.4, 0.4])
            self.scene.add_directional_light([0.5, 1, -1], [1, 1, 1], shadow=False)

        try:
            mesh = trimesh.load(KNIFE_STL_PATH, force='mesh')
            zmin, zmax = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
            self._knife_local_tip_z = zmin
            print(f"[knife] STL ok. zmin={zmin:.4f}, zmax={zmax:.4f} → tip=zmin")
        except Exception as e:
            print(f"[knife] STL load failed: {e}")
            self._knife_local_tip_z = self.KNIFE_TIP_LOCAL_Z_DEFAULT
            print(f"[knife] tip z fallback = {self._knife_local_tip_z:.3f} m")

        try:
            self.knife_link = sapien_utils.get_obj_by_name(self.agent.robot.get_links(), KNIFE_LINK_NAME)
        except Exception:
            self.knife_link = None

    # ---------- ReplicaCAD apartment-stage backdrop ----------
    # Stage glb measured AABB (Y-up source frame): x[-2.66,+4.60], y[-0.08,+3.06],
    # z[-4.76,+8.17] — ~7m × 3m × 13m apartment, floor at y≈0.
    # After R_x(+90deg): floor at world z≈0. With offset (0,0,0) the cutting
    # setup at x∈[-0.62,0], y∈[-0.5,0.5], z>0 sits inside the apartment, and
    # the back walls (world x≈-2.66, y≈+4.76) appear behind the cut from the
    # camera's POV (eye=(0.5,-0.5,0.8) → target=(0,0,0)).
    # We additionally rotate -90deg about world Z so the long apartment axis
    # aligns with camera's depth axis → more wall visible behind cutting setup.
    # Canonical R_x(+90deg) only — same as ReplicaCADSceneBuilder. World-Z = up.
    # Apartment AABB after rotation: x[-2.66,+4.60], y[-8.17,+4.76], z[-0.08,+3.06].
    # TableSceneBuilder places its table-top at world z=0 with the lab "floor"
    # at z=-0.9196 (table_height). We offset the apartment by z=-0.9196 so the
    # apartment floor coincides with the lab floor — i.e., the lab table becomes
    # a "desk" standing on the apartment floor, with Franka mounted on top.
    # XY offset places the apartment so the cutting setup (world origin) lands
    # in a navmesh-confirmed open room area, avoiding wall intersections.
    # Free-point analysis from apt_0..5 navmesh: most-open xy in apt frame is
    # roughly at (-0.75, +0.9..3.0); we shift by -1× that so it maps to (0,0).
    # Different per stage to give visual variation (different rooms shown).
    # Each entry: (relative_path_from_scene_datasets, world_offset_xyz, yaw_deg).
    # World offset puts the cutting setup at the room center identified by
    # raycast analysis. z offset is just _LZ (= -table_height) so apt floor
    # at source y=0 lands at world z=-0.92, matching the lab floor level. Lab
    # table legs (z=-0.92→0) extend up from there and stay fully visible.
    # Canonical bg env list: bg_idx 1..15. bg_idx 0 = default lab (no apt).
    # Diverse mix across 6 visual style clusters: ReplicaCAD modern, iTHOR
    # cartoon (kitchen/LR/bedroom), ArchitecTHOR photoreal, ProcTHOR procedural.
    # World offset puts the cutting setup at the room center identified by
    # raycast analysis; z offset = _LZ so apt floor matches lab floor.
    _LZ = -0.9196429
    _BG_STAGES = [
        # 1..3: ReplicaCAD modern minimalist (frl + 2 staging variants)
        ("replica_cad_dataset/stages/frl_apartment_stage.glb",                        (-1.42, +1.96, _LZ), 0.0),
        ("replica_cad_dataset/stages/Stage_v3_sc0_staging.glb",                       (-1.44, +2.07, _LZ), 0.0),
        ("replica_cad_dataset/stages/Stage_v3_sc2_staging.glb",                       (-1.44, +2.07, _LZ), 0.0),
        # 4..5: iTHOR kitchens (FloorPlan 1-30) — cartoon style
        ("ai2thor/ai2thor-hab/assets/stages/iTHOR/FloorPlan1_physics.glb",            (+0.33, -1.20, _LZ), 0.0),
        ("ai2thor/ai2thor-hab/assets/stages/iTHOR/FloorPlan15_physics.glb",           (-1.49, +1.51, _LZ), 0.0),
        # 6..7: iTHOR living rooms (200s) — cartoon LR
        ("ai2thor/ai2thor-hab/assets/stages/iTHOR/FloorPlan210_physics.glb",          (-3.35, +2.31, _LZ), 0.0),
        ("ai2thor/ai2thor-hab/assets/stages/iTHOR/FloorPlan220_physics.glb",          (-3.24, +1.45, _LZ), 0.0),
        # 8..9: iTHOR bedrooms (300s) — cartoon BR (NEW visual cluster)
        ("ai2thor/ai2thor-hab/assets/stages/iTHOR/FloorPlan301_physics.glb",          (+0.80, +0.10, _LZ), 0.0),
        ("ai2thor/ai2thor-hab/assets/stages/iTHOR/FloorPlan325_physics.glb",          (+2.05, -0.41, _LZ), 0.0),
        # 10..11: ArchitecTHOR — photoreal detailed apartments
        ("ai2thor/ai2thor-hab/assets/stages/ArchitecTHOR/ArchitecTHOR-Val-00.glb",    (-2.06, +0.74, _LZ), 0.0),
        ("ai2thor/ai2thor-hab/assets/stages/ArchitecTHOR/ArchitecTHOR-Val-02.glb",    (+1.94, +2.46, _LZ), 0.0),
        # 12..15: ProcTHOR — procedurally generated houses (very diverse layouts)
        ("ai2thor/ai2thor-hab/assets/stages/ProcTHOR/3/ProcTHOR-Train-1500.glb",      (-5.98, +7.88, _LZ), 0.0),
        ("ai2thor/ai2thor-hab/assets/stages/ProcTHOR/9/ProcTHOR-Train-7000.glb",      (-4.98, +3.14, _LZ), 0.0),
        ("ai2thor/ai2thor-hab/assets/stages/ProcTHOR/c/ProcTHOR-Val-0.glb",           (-7.62, +4.11, _LZ), 0.0),
        ("ai2thor/ai2thor-hab/assets/stages/ProcTHOR/1/ProcTHOR-Test-549.glb",        (-2.48, +6.33, _LZ), 0.0),
    ]
    assert len(_BG_STAGES) == 15, f"expected 15 bg envs, got {len(_BG_STAGES)}"
    # Backwards-compatible alias used by _load_replica_stage
    _RCAD_STAGES = _BG_STAGES

    def _load_replica_stage(self, apt_idx: int):
        import os
        import transforms3d
        asset_dir = os.environ.get("MS_ASSET_DIR", os.path.expanduser("~/.maniskill"))
        rel, offset, yaw_deg = self._BG_STAGES[apt_idx % len(self._BG_STAGES)]
        glb = os.path.join(asset_dir, "data/scene_datasets", rel)
        if not os.path.exists(glb):
            print(f"[bg] stage glb missing: {glb}, skipping")
            return
        # ReplicaCAD is Y-up; SAPIEN is Z-up — rotate +90deg about X first,
        # then yaw_deg about world Z to orient the apartment for the camera.
        qx = transforms3d.quaternions.axangle2quat(np.array([1, 0, 0]), np.deg2rad(90))
        qz = transforms3d.quaternions.axangle2quat(np.array([0, 0, 1]), np.deg2rad(yaw_deg))
        q = transforms3d.quaternions.qmult(qz, qx)
        pose = sapien.Pose(p=list(offset), q=q)
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(glb)
        # Visual-only — apartment geometry must NOT collide with the cutting
        # volume (Franka/board/fruit). Robot-scene contact from the apartment
        # walls/floor would corrupt the scripted cut trajectory.
        builder.initial_pose = pose
        self._bg_actor = builder.build_static(name=f"bg_replica_apt_{apt_idx}")
        print(f"[bg] loaded {os.path.basename(glb)} at offset={offset} yaw={yaw_deg}deg")

    # ---------- 에피소드 초기화 ----------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            qpos = self.agent.keyframes["rest"].qpos
            self.agent.robot.set_qpos(qpos)
        assert hasattr(self.agent, "tcp") and (self.agent.tcp is not None)
        self._ensure_buffers()

        self.table_scene.initialize(env_idx)

        # Variation: yaw is applied here as a z-axis rotation on top of
        # block_rot_quat (which lays the fruit on its side on the board).
        variation = (options or {}).get("variation") or {}
        yaw = float(variation.get("yaw", 0.0))

        for e in env_idx.tolist():
            base_quat = self.block_rot_quat
            if yaw != 0.0:
                yaw_quat = euler2quat(0.0, 0.0, yaw)
                # Compose: first base lay-down, then yaw about world z.
                # transforms3d quaternions: (w, x, y, z); use standard multiply.
                from transforms3d.quaternions import qmult
                init_quat = qmult(yaw_quat, base_quat)
            else:
                init_quat = base_quat

            # --- 위치 랜덤 + variation pos_offset ---
            pos_offset = np.asarray(variation.get("pos_offset", [0.0, 0.0, 0.0]),
                                    dtype=float)
            rand_x = self.board_center[0] + np.random.uniform(0.05, 0.15) + pos_offset[0]
            rand_y = self.board_center[1] + np.random.uniform(-0.15, 0.15) + pos_offset[1]
            # Compute rotated-mesh world zmin so we can place the fruit ~1cm
            # above the board top (not buried). _mesh_verts_local are scaled
            # vertices in mesh frame. After applying init_quat, world z = the
            # rotated z-component. We then offset rand_z so min(world z) = board+0.01.
            _verts = getattr(self, "_mesh_verts_local", None)
            _GAP = 0.001  # 1 mm clearance to avoid SAPIEN render z-fight (touching look)
            # board_center.z is the BOARD CENTER (= half-thickness above world 0).
            # Fruit min world z must clear the BOARD TOP (= center + half-thickness),
            # otherwise large meshes (cherry/etc) sink ~1 cm into the board.
            # Apply this fix only for additional fruits — legacy fruits were
            # tuned around the old (buggy) base, so don't disturb them.
            _board_anchor = self.board_center[2]
            _fruit_for_anchor = getattr(self, "_variation_object", "banana")
            if _fruit_for_anchor in _NEW_FRUITS:
                _board_anchor = _board_anchor + float(self.BOARD_SIZE[2])
            if _verts is not None and len(_verts) > 0:
                from transforms3d.quaternions import rotate_vector as _rot_v
                # Rotate every vertex to world; find min z. Vectorize via mat.
                from transforms3d.quaternions import quat2mat as _q2m
                R = _q2m(init_quat).astype(np.float32)
                # world coords (origin-anchored): (R @ v.T).T
                world_z = _verts @ R[2]  # i-th row dot with R's z-row
                z_min_rot = float(world_z.min())
                rand_z = _board_anchor + _GAP - z_min_rot
            else:
                rand_z = _board_anchor + _GAP

            # Expose rotated mesh-centroid XY offset so external callers
            # (motion planner in collect script) can aim at the *visual* center
            # of the fruit rather than block.pose.p (which == mesh origin).
            from transforms3d.quaternions import rotate_vector
            ctr_world = rotate_vector(self._mesh_centroid_local, init_quat)
            self._visual_centroid_xy = np.array(
                [float(ctr_world[0]), float(ctr_world[1])], dtype=np.float32)

            self.block_apple[e].set_pose(
                Pose.create_from_pq([rand_x, rand_y, rand_z], init_quat)
            )

            # contact 상태 초기화
            self._contact[e] = False
            self._contact_prev[e] = False

        # TCP 관련 상태 리셋
        self._prev_tcp_z = self.agent.tcp.pose.p[:, 2].clone().cpu()
        self._contact_enter_frame[:] = False
        self._contact_exit_frame[:] = False
        self._contact_frame[:] = False



    # ---------- pose utils ----------
    def _pose_to_T(self, p: torch.Tensor, q: torch.Tensor):
        N = p.shape[0]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.empty((N, 3, 3), dtype=p.dtype, device=p.device)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)
        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - x * w)
        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)

        T = torch.eye(4, dtype=p.dtype, device=p.device).unsqueeze(0).repeat(N, 1, 1)
        T[:, :3, :3] = R
        T[:, :3, 3] = p
        return T

    def _T_const_from_rpy_xyz(self, rpy, xyz, device, dtype=torch.float32):
        rx, ry, rz = rpy
        tx, ty, tz = xyz
        R = torch.tensor(euler2mat(rx, ry, rz), dtype=dtype, device=device)
        T = torch.eye(4, dtype=dtype, device=device)
        T[:3, :3] = R
        T[:3, 3] = torch.tensor([tx, ty, tz], dtype=dtype, device=device)
        return T

    def _get_tip_from_eef(self):
        tcp_p = self.agent.tcp.pose.p
        tcp_q = self.agent.tcp.pose.q
        device = tcp_p.device
        dtype = tcp_p.dtype

        T_world_hand = self._pose_to_T(tcp_p, tcp_q)
        T_hand_knife = self._T_const_from_rpy_xyz(HAND_TO_KNIFE_RPY, HAND_TO_KNIFE_XYZ, device, dtype=dtype)

        tip_local = torch.tensor([0.0, 0.0, self._knife_local_tip_z, 1.0], dtype=dtype, device=device)
        T_world_knife = torch.einsum("nij,jk->nik", T_world_hand, T_hand_knife)
        tip_world = torch.einsum("nij,j->ni", T_world_knife, tip_local)
        return tip_world[:, :3]

    # ---------- 시뮬 스텝 이후 ----------
   
    def _after_control_step(self):
        self._ensure_buffers()
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

        # 칼 끝 + 손잡이 쪽 좌표
        tip_p = self.agent.tcp.pose.p        # (N,3)
        contact = torch.zeros(self.num_envs, dtype=torch.bool, device=tip_p.device)

        # Per-fruit cut pieces; chosen from variation-specified object.
        # Falls back to banana if the attribute wasn't set (legacy path).
        _fruit = getattr(self, "_variation_object", "banana")
        cut_paths = [
            f"/data/mani_skill/assets/fruits/update/{_fruit}/{_fruit}_left.obj",
            f"/data/mani_skill/assets/fruits/update/{_fruit}/{_fruit}_right.obj",
        ]

        for e in range(self.num_envs):
            # tip이 보드에 닿았는지 확인
            obj_actor = self.block_apple[e][0] if isinstance(self.block_apple[e], list) else self.block_apple[e]
            
            # AABB min/max 계산 (Trimesh나 사전 정의된 half 기반)
            aabb_min, aabb_max = self._get_actor_aabb(obj_actor)
            aabb_min = torch.tensor(aabb_min, dtype=torch.float32, device=tip_p.device)
            aabb_max = torch.tensor(aabb_max, dtype=torch.float32, device=tip_p.device)

            # === 오브젝트 중심 & half 계산 ===
            obj_pose = obj_actor.pose
            pose_p = np.array(obj_pose.p, dtype=float).reshape(-1)
            # print(f"obj_pose:{obj_pose}")
            # obj_p = torch.tensor(obj_pose.p, dtype=torch.float32, device=tip_p.device)  # 중심좌표
            obj_p = obj_pose.p.clone().detach().to(dtype=torch.float32, device=tip_p.device)
            half = (aabb_max - aabb_min) / 2.0                                         # 절반 크기
            print(f"half:{half}")
            # === contact 판정: tip이 obj_pose 중심 ± half 범위 내 ===
            inside_tip = (
                (tip_p[e, 1]- 0.02 > (pose_p[1] - half[2])) & (tip_p[e, 1]- 0.02 < (pose_p[1] + half[2])) &
                (tip_p[e, 2] > (pose_p[2] - half[1])) & (tip_p[e, 2] < (pose_p[2] + half[1]))
            )
            contact[e] = inside_tip
            # Don't rely on _get_tip_from_eef — its matrix math gives unstable
            # results across robot configurations. Use TCP directly + an
            # empirical xy threshold large enough to cover the URDF blade
            # offset (~0.25 m mounted horizontally).
            tcp_p_e = self.agent.tcp.pose.p
            tcp_x = float(tcp_p_e[e, 0].item())
            tcp_y = float(tcp_p_e[e, 1].item())
            tcp_z = float(tcp_p_e[e, 2].item())
            board_z = self.board_center[2]
            # TCP is "in cut zone" when descended deep enough.
            reached_board = tcp_z <= board_z + 0.20
            # Compare TCP to fruit visual centroid; threshold 0.30 m to cover
            # both centroid offset and URDF blade-mount offset (knife mounted
            # ~25 cm to +x of TCP, so commanded TCP is offset accordingly).
            ctr = getattr(self, "_visual_centroid_xy", np.zeros(2, np.float32))
            obj_xy_np = (pose_p[:2].cpu().numpy() if hasattr(pose_p, "cpu") else np.asarray(pose_p[:2]))
            obj_xy = obj_xy_np + np.asarray(ctr, np.float32)
            tip_xy_err_x = abs(tcp_x - float(obj_xy[0]))
            tip_xy_err_y = abs(tcp_y - float(obj_xy[1]))
            near_fruit_xy = (tip_xy_err_x < 0.35) and (tip_xy_err_y < 0.10)
            object_center_y, object_center_z = pose_p[1], pose_p[2]
            one_target_y  = object_center_y

            # 보드에 닿았고 + xy도 fruit 중심 근처 + 아직 안 잘렸을 때만 swap
            if reached_board and near_fruit_xy and (not self._already_cut[e]):
                current_pose = self.block_apple[e].pose

                # 기존 actor 제거
                old_obj = self.block_apple[e]
                if isinstance(old_obj, list):
                    for piece in old_obj:
                        self.scene.remove_actor(piece)
                else:
                    self.scene.remove_actor(old_obj)
                    self.scene.update_render()

                # 새 잘린 조각 불러오기
                new_pieces = []
                # SAPIEN's OBJ loader does not auto-apply map_Kd from the MTL
                # for these cut pieces (pre-cut works, post-cut doesn't). Build
                # the diffuse-texture material explicitly and bind it to every
                # visual record after add_visual_from_file.
                from pathlib import Path
                # Override single-material cut pieces (banana etc.) with an
                # explicit texture so SAPIEN's OBJ loader doesn't drop map_Kd.
                # MULTI-material cut pieces (e.g. cherry: skin+cut_flesh) are
                # left alone — SAPIEN loads their MTL natively and the override
                # would collapse them to a single material.
                if not hasattr(self, "_cut_tex_cache"):
                    self._cut_tex_cache = {}
                if _fruit not in self._cut_tex_cache:
                    _mtl_dir = Path(f"/data/mani_skill/assets/fruits/update/{_fruit}")
                    # Find the cut piece MTL to count materials.
                    _mtl_files = list(_mtl_dir.glob("*.mtl")) if _mtl_dir.exists() else []
                    _n_mat = 0
                    if _mtl_files:
                        try:
                            with open(_mtl_files[0]) as fh:
                                _n_mat = sum(1 for L in fh if L.startswith("newmtl "))
                        except Exception:
                            _n_mat = 0
                    _tex_path = _mtl_dir / f"{_fruit}_diffuse.jpg"
                    if _n_mat == 1 and _tex_path.exists():
                        try:
                            _tex = sapien.render.RenderTexture2D(
                                filename=str(_tex_path), mipmap_levels=4)
                            _mat = sapien.render.RenderMaterial()
                            _mat.set_base_color([1.0, 1.0, 1.0, 1.0])
                            _mat.base_color_texture = _tex
                            self._cut_tex_cache[_fruit] = (_tex, _mat)
                            print(f"[cut] cached single-mat override for {_fruit}")
                        except Exception as _ex:
                            print(f"[cut] tex load fail {_tex_path}: {_ex}")
                            self._cut_tex_cache[_fruit] = (None, None)
                    else:
                        # Multi-material or no MTL — let SAPIEN auto-load MTL.
                        print(f"[cut] {_fruit}: {_n_mat} materials in MTL — no override")
                        self._cut_tex_cache[_fruit] = (None, None)
                _diffuse_tex, _diffuse_mat = self._cut_tex_cache[_fruit]
                for cut_path in cut_paths:
                    p = Path(cut_path).expanduser()
                    assert p.exists(), f"OBJ not found: {p}"

                    builder = self.scene.create_actor_builder()
                    _scale = getattr(self, "_variation_scale", 1.0)
                    if _diffuse_mat is not None:
                        builder.add_visual_from_file(
                            str(p), scale=[_scale, _scale, _scale],
                            material=_diffuse_mat)
                    else:
                        builder.add_visual_from_file(str(p), scale=[_scale, _scale, _scale])
                    builder.add_sphere_collision(radius=1e-100)

                    cut_actor = builder.build_static(
                        name=f"block_apple_cut_{os.path.basename(cut_path)}_{e}"
                    )

                    # Keep banana's CURRENT z (it settles ~1cm above board_center
                    # after physics warmup). Do NOT override with board_z or
                    # pieces snap ~1cm down visually at cut.
                    # Also separate the two halves by 1.5cm each (3cm gap) so
                    # the cut is visually obvious at 256x256 — useful for VLA
                    # training where "cut" must be readable from image.
                    pose_p = np.array(current_pose.p, dtype=float).reshape(-1)
                    pose_q = np.array(current_pose.q, dtype=float).reshape(-1)
                    # Per-fruit cut-axis (mesh-frame, identified empirically by
                    # comparing _left vs _right OBJ centroids). Maps to world
                    # via init_quat — but since most new fruits use identity
                    # rotation, mesh axis ≈ world axis.
                    _CUT_AXIS = {
                        "cherry": 2, "plum": 2, "shine_muscat": 2, "pear": 2,
                        "lemon": 1, "kiwi": 0, "tomato": 0, "grape": 2,
                        "golden_strawberry": 2,
                    }
                    _ax = _CUT_AXIS.get(_fruit, 1)  # default world-Y (banana, apple, etc.)
                    # Legacy fruits: mesh origin == cut plane, rely on
                    # block_rot_quat to align mesh-axis to world-_ax. Original
                    # sign (left += , right −=) separates them.
                    #
                    # NEW fruits: mesh origin is at the fruit BASE (not cut
                    # plane). We do two corrections in the MESH frame and then
                    # rotate by the fruit's pose quat to land in world frame:
                    #   (a) shift cut-plane to fruit-pose origin (-_plane*_scale)
                    #   (b) ±_CUT_GAP_NEW separation along the mesh split axis
                    # Mesh split axis is auto-detected from L_max ≈ R_min.
                    if _fruit in _NEW_FRUITS:
                        # Anchor each piece's CENTROID at the fruit pose, then
                        # push centroids ±_CUT_GAP_NEW along the mesh split
                        # axis. The piece visual extent grows from each piece's
                        # own centroid, so this keeps both pieces' bodies
                        # spread evenly around the knife (symmetric body
                        # placement). Cache split axis + each piece's centroid.
                        if not hasattr(self, "_cut_axis_cache"):
                            self._cut_axis_cache = {}
                        if _fruit not in self._cut_axis_cache:
                            import trimesh as _tm
                            _L = _tm.load(cut_paths[0], force='mesh')
                            _R = _tm.load(cut_paths[1], force='mesh')
                            _diffs = np.abs(_R.vertices.min(0) - _L.vertices.max(0))
                            _mesh_ax = int(np.argmin(_diffs))
                            _ctr_L = _L.vertices.mean(0).astype(float)
                            _ctr_R = _R.vertices.mean(0).astype(float)
                            self._cut_axis_cache[_fruit] = (
                                _mesh_ax, _ctr_L, _ctr_R)
                            print(f"[cut] {_fruit}: mesh_axis={_mesh_ax}, "
                                  f"L_centroid={_ctr_L.round(3).tolist()}, "
                                  f"R_centroid={_ctr_R.round(3).tolist()}")
                        _mesh_ax, _ctr_L, _ctr_R = self._cut_axis_cache[_fruit]
                        _is_left = "left" in cut_path.lower()
                        _ctr = _ctr_L if _is_left else _ctr_R
                        _sign = -1.0 if _is_left else +1.0
                        # Shift piece so its OWN centroid lands at fruit pose,
                        # then push ±gap along split axis. Per-fruit override
                        # for long fruits (lemon/pear) where 5 cm leaves halves
                        # overlapping; otherwise use the default 5 cm.
                        _gap = _CUT_GAP_NEW_PER_FRUIT.get(_fruit, _CUT_GAP_NEW)
                        _mesh_off = -_ctr * _scale
                        _mesh_off[_mesh_ax] += _sign * _gap
                        from transforms3d.quaternions import rotate_vector as _rv
                        _world_off = _rv(_mesh_off, np.asarray(pose_q, dtype=float))
                        pose_p = pose_p + _world_off
                    else:
                        # Legacy: keep original sign (banana/apple/etc).
                        if "left" in cut_path.lower():
                            pose_p[_ax] += _CUT_GAP
                        elif "right" in cut_path.lower():
                            pose_p[_ax] -= _CUT_GAP
                    # Per-fruit visual tweaks at cut update: small uniform
                    # offsets applied to BOTH halves so the swap doesn't
                    # visually jump. Tuned empirically from pilot mp4s.
                    #   - small/round fruits (cherry/shine_muscat/grape/
                    #     golden_strawberry) drift slightly to the left when
                    #     the pre-cut single-piece is replaced by two halves;
                    #     a small +X warp re-centers them visually.
                    #   - kiwi's cut pieces sit slightly below the board top
                    #     because the pre-cut block had settled on the board
                    #     while the two-piece composite has a different AABB
                    #     bottom; a small +Z lift hides the sink.
                    # Compensate for single→two-piece visual-center shift.
                    # Pre-cut: the SINGLE block_apple actor renders the OBJ
                    # with its origin at pose_p, so the visible center of the
                    # fruit is at pose_p + (mesh.centroid * scale, rotated to
                    # world). Post-cut: each half's centroid is anchored at
                    # pose_p (per the _ctr_L/_ctr_R logic above), which leaves
                    # them visually shifted from where the original fruit was
                    # by the SINGLE-mesh centroid offset. Add it back here so
                    # the swap doesn't appear to teleport the fruit.
                    if _fruit in _NEW_FRUITS:
                        from transforms3d.quaternions import rotate_vector as _rv2
                        ctr_world = _rv2(
                            np.asarray(self._mesh_centroid_local, dtype=float),
                            np.asarray(pose_q, dtype=float),
                        )
                        pose_p += np.asarray(ctr_world, dtype=float)
                    cut_actor.set_pose(Pose.create_from_pq(pose_p, pose_q))

                    new_pieces.append(cut_actor)

                # 교체
                self.block_apple[e] = new_pieces
                self._already_cut[e] = True

        self._contact_prev = self._contact.clone()
        self._contact = contact.clone()
        self._contact_frame = contact.detach().cpu().numpy()
        self._prev_tcp_z = tip_p[:, 2].detach().cpu()



        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()


    def evaluate(self):
        result = {}
        try:
            # TCP (EEF) pose 직접 읽기
            tcp_pose = self.agent.tcp.pose
            eef_pos = np.asarray(tcp_pose.p).reshape(1, 3)   # (1, 3)
            eef_quat = np.asarray(tcp_pose.q).reshape(1, 4)  # (1, 4)
        except Exception as e:
            print(f"[Evaluate] tcp pose fetch failed: {e}")
            return {"success": torch.tensor([False], dtype=torch.bool)}

        # ---------- 조건별 판정 ----------
        # 1) 컷 완료 여부
        cut_done = bool(self._already_cut.any())
        result["cut_done"] = cut_done

        # 2) orientation 안정성
        if not hasattr(self, "_eef_q_init"):
            self._eef_q_init = eef_quat.copy()

        def quat_angle_error(q0, q):
            dot = np.abs(np.sum(q0 * q, axis=-1))
            dot = np.clip(dot, -1.0, 1.0)
            return np.degrees(2 * np.arccos(dot))

        quat_err = quat_angle_error(self._eef_q_init, eef_quat)
        stable_orientation = bool((quat_err < 25.0).all())
        result["stable_orientation"] = stable_orientation

        # 3) XY 위치 (중심선 근처) — use actual fruit pose, not board center,
        # and tighten threshold to fruit-size scale (was 20 cm, way too loose).
        eef_xy = eef_pos[:, :2]
        try:
            actor = self.block_apple[0][0] if isinstance(self.block_apple[0], list) else self.block_apple[0]
            obj_xy = np.asarray(actor.pose.p, np.float32).reshape(-1)[:2]
        except Exception:
            obj_xy = np.asarray(self.block_center1[:2], np.float32)
        # Compare BLADE TIP (not TCP) — TCP is offset by HAND_TO_KNIFE_XYZ so
        # raw TCP comparison hides the actual cut location.
        try:
            tip_xy = self._get_tip_from_eef()[0].detach().cpu().numpy()[:2]
        except Exception:
            tip_xy = eef_xy[0]
        dist_xy = np.abs(np.asarray(tip_xy, np.float32) - obj_xy)
        # blade tip is ~6-8cm offset from fruit center due to URDF knife mount;
        # 12 cm threshold accommodates that while still catching truly off-axis
        # cuts (was 0.2 m which was way too loose, 0.05 was too tight).
        near_center_line = bool(dist_xy[0] < 0.12 and dist_xy[1] < 0.12)
        result["near_center_line"] = near_center_line

        # ---------- 최종 성공 조건 ----------
        success_flag = cut_done and stable_orientation and near_center_line
        if success_flag:
            self._ever_success[:] = True
        final_success = bool(self._ever_success.any())

        result["success"] = torch.tensor([final_success], dtype=torch.bool)

        print(
            f"[Evaluate] cut_done={cut_done} "
            f"| ori_ok={stable_orientation} (max_err={quat_err.max():.2f}°) "
            f"| centerline_ok={near_center_line} "
            f"(tip_xy_err=[{dist_xy[0]:.3f},{dist_xy[1]:.3f}] m) "
            f"-> Success={final_success}"
        )

        return result




    def _get_obs_extra(self, info: Dict):
        extra = dict(
            contact=self._contact_frame.astype(np.bool_),
      
        )
        try:
            agent_state = self.agent.get_state()
            for k in ["eef_pos", "eef_quat", "eef_lin_vel", "eef_ang_vel", "tcp_pose"]:
                if k in agent_state:
                    extra[k] = np.asarray(agent_state[k])
        except Exception:
            pass
        return extra
        
    def get_state_dict(self) -> Dict:
        # 원래 부모(BaseEnv)의 get_state_dict 호출 (flatten 아님!)
        state = super().get_state_dict()

        # contact 추가
        task = state.get("task", {})
        task["contact"]       = np.asarray(self._contact_frame, dtype=np.bool_)

        state["task"] = task

        return state


    def _ensure_buffers(self):
        # ✅ num을 항상 int로 강제
        raw_num = getattr(self, "num_envs", 1)
        num = int(raw_num) if not isinstance(raw_num, int) else raw_num

        if not isinstance(getattr(self, "_contact", None), torch.Tensor) or self._contact.numel() != num:
            self._contact = torch.zeros(num, dtype=torch.bool)
        if not isinstance(getattr(self, "_contact_prev", None), torch.Tensor) or self._contact_prev.numel() != num:
            self._contact_prev = torch.zeros(num, dtype=torch.bool)
        if not isinstance(getattr(self, "_prev_tcp_z", None), torch.Tensor) or self._prev_tcp_z.numel() != num:
            self._prev_tcp_z = torch.zeros(num, dtype=torch.float32)

        # ✅ 배열 재초기화 방지: 존재하면 shape만 맞춰주고 값은 유지
        if not isinstance(getattr(self, "_contact_enter_frame", None), np.ndarray) or self._contact_enter_frame.shape != (num,):
            old = getattr(self, "_contact_enter_frame", None)
            self._contact_enter_frame = np.zeros(num, dtype=bool)
            if isinstance(old, np.ndarray):
                self._contact_enter_frame[:min(num, old.size)] = old[:min(num, old.size)]

        if not isinstance(getattr(self, "_contact_exit_frame", None), np.ndarray) or self._contact_exit_frame.shape != (num,):
            old = getattr(self, "_contact_exit_frame", None)
            self._contact_exit_frame = np.zeros(num, dtype=bool)
            if isinstance(old, np.ndarray):
                self._contact_exit_frame[:min(num, old.size)] = old[:min(num, old.size)]

        if not isinstance(getattr(self, "_contact_frame", None), np.ndarray) or self._contact_frame.shape != (num,):
            old = getattr(self, "_contact_frame", None)
            self._contact_frame = np.zeros(num, dtype=bool)
            if isinstance(old, np.ndarray):
                self._contact_frame[:min(num, old.size)] = old[:min(num, old.size)]

        if not isinstance(getattr(self, "_already_cut", None), np.ndarray) or self._already_cut.shape != (num,):
            # ✅ 여기서도 기존 값 보존
            old = getattr(self, "_already_cut", None)
            self._already_cut = np.zeros(num, dtype=bool)
            if isinstance(old, np.ndarray):
                self._already_cut[:min(num, old.size)] = old[:min(num, old.size)]
        if not isinstance(getattr(self, "_ever_success", None), np.ndarray) or self._ever_success.shape != (num,):
            self._ever_success = np.zeros(num, dtype=bool)
        if not isinstance(getattr(self, "_contact_frame", None), np.ndarray) or self._contact_frame.shape != (num,):
            old = getattr(self, "_contact_frame", None)
            self._contact_frame = np.zeros(num, dtype=bool)
            if isinstance(old, np.ndarray):
                self._contact_frame[:min(num, old.size)] = old[:min(num, old.size)]
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        try:
            eval_result = self.evaluate()
            print("[EvalResult]", eval_result)
            if isinstance(eval_result, dict):
                info.update(eval_result)  # success 외의 reached_board 등도 전부 포함
        except Exception as e:
            print(f"[TableTopCuttingEnv.step] evaluate merge failed: {e}")
        # Per-step diagnostics: MS-side contact force on knife + TCP linear vel.
        # MPM-side force is logged by the bridge separately (alignment.json).
        # TCP vel: finite-diff, since Link wrapper has no velocity getter.
        try:
            tcp_p = np.asarray(self.agent.tcp.pose.p.cpu()
                               if hasattr(self.agent.tcp.pose.p, "cpu")
                               else self.agent.tcp.pose.p, np.float32).reshape(-1)[:3]
            dt = 1.0 / float(getattr(self, "control_freq", 20))
            prev = getattr(self, "_dbg_prev_tcp_p", None)
            v_np = (tcp_p - prev) / dt if prev is not None else np.zeros(3, np.float32)
            self._dbg_prev_tcp_p = tcp_p.copy()
            v_mag = float(np.linalg.norm(v_np))
            if self.knife_link is not None:
                f = self.knife_link.get_net_contact_forces()
                f_np = f.cpu().numpy().reshape(-1)[:3] if hasattr(f, "cpu") else np.asarray(f).reshape(-1)[:3]
                f_mag = float(np.linalg.norm(f_np))
            else:
                f_mag, f_np = 0.0, np.zeros(3)
            # MPM-side cutting force (stashed by bridge — 1 tick lag). This is
            # the physically meaningful label; F_ms is sapien rigid contact
            # only and ≈0 because MS-fruit is a visual proxy.
            fmpm = getattr(self, "_last_mpm_force", None)
            if fmpm is not None:
                fmpm_np = np.asarray(fmpm, np.float32).reshape(-1)[:3]
                fmpm_mag = float(np.linalg.norm(fmpm_np))
                fmpm_str = f" |F_mpm|={fmpm_mag:6.2f}N"
            else:
                fmpm_str = ""
            fv_str = (f"|F_ms|={f_mag:6.2f}N (Fz={float(f_np[2]):+6.2f})"
                      f"{fmpm_str}"
                      f" |v_tcp|={v_mag:5.3f}m/s (vz={float(v_np[2]):+5.3f})")
        except Exception as e:
            fv_str = f"force/vel read err: {e}"
        print(f"[Step Debugeval] reward={reward}, {fv_str}, info={info}")
        return obs, reward, terminated, truncated, info
    
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # --- 기본 상태 ---
        tip_p = self._get_tip_from_eef()  # (N, 3)
        tip_z = tip_p[:, 2]
        board_z = torch.tensor(self.board_center[2], device=tip_z.device)
        init_tip_z = getattr(self, "_init_tip_z", None)

        if init_tip_z is None:
            # 첫 프레임에서 초기 높이 기록
            self._init_tip_z = tip_z.clone().detach()
            init_tip_z = self._init_tip_z
        else:
            init_tip_z = self._init_tip_z.to(tip_z.device)

        # --- Phase 1: 절삭 깊이 보상 ---
        cut_depth = torch.clamp(board_z - tip_z, min=0.0)
        cut_progress = 1 - torch.tanh(10 * (board_z - tip_z))
        cut_progress = torch.clamp(cut_progress, 0.0, 1.0)

        # --- Phase 2: 복귀 보상 (칼이 다시 올라가는 경우) ---
        rising = (tip_z > self._prev_tcp_z.to(tip_z.device))
        return_reward = rising.float() * torch.clamp(cut_depth, 0, 0.05) * 10  # 올라오면서 깊은 곳이면 보상 큼
        return_reward = torch.clamp(return_reward, 0.0, 1.0)

        # --- Phase 3: 완전 들어올림 보상 ---
        lifted_enough = tip_z > (init_tip_z + 0.03)  # 3cm 이상 위로 들어올렸을 때
        lift_reward = lifted_enough.float() * 5.0

        # --- Contact 보상 (실제 접촉 중이면 약간 추가) ---
        contact_reward = torch.tensor([
            1.0 if c else 0.0 for c in self._contact_frame
        ], dtype=torch.float32, device=tip_z.device)

        # --- Orientation 안정성 (칼이 너무 기울면 감점) ---
        tcp_q = self.agent.tcp.pose.q
        if not hasattr(self, "_eef_q_init"):
            self._eef_q_init = tcp_q.clone().detach().cpu()
        dot = torch.abs((self._eef_q_init.to(tcp_q.device) * tcp_q).sum(dim=-1))
        dot = torch.clamp(dot, -1.0, 1.0)
        quat_err = 2 * torch.acos(dot)
        stable_orientation_reward = 1 - torch.tanh(0.5 * quat_err)

        # --- 최종 종합 ---
        reward = (
            1.0 * cut_progress +
            0.5 * return_reward +
            0.3 * contact_reward +
            0.5 * stable_orientation_reward +
            lift_reward
        )

        # info로 세부 로그 기록
        info.update({
            "r_cut_progress": cut_progress.detach().cpu().numpy(),
            "r_return": return_reward.detach().cpu().numpy(),
            "r_contact": contact_reward.detach().cpu().numpy(),
            "r_orientation": stable_orientation_reward.detach().cpu().numpy(),
            "r_lift": lift_reward.detach().cpu().numpy(),
        })

        return reward
    def _get_actor_aabb(self, actor):
        """
        주어진 actor의 mesh 파일을 직접 읽어서 실제 크기(x, y, z)를 계산하고
        object pose 중심을 기준으로 world-space 가상 AABB 생성.
        """
        # --- mesh 파일 경로 추정 (visual 파일에서 첫 번째 obj 경로 가져오기) ---
        vis_shapes = actor.get_visual_shapes() if hasattr(actor, "get_visual_shapes") else []
        mesh_path = None
        if vis_shapes:
            rs = vis_shapes[0].get_render_shape()
            if rs and hasattr(rs, "mesh") and hasattr(rs.mesh, "filename"):
                mesh_path = rs.mesh.filename
        # fallback: 기본 banana.obj
        if mesh_path is None or not os.path.exists(mesh_path):
            mesh_path = "/data/mani_skill/assets/fruits/banana.obj"

        # --- trimesh 로드로 실제 mesh 크기 계산 ---
   
        mesh = trimesh.load_mesh(mesh_path, force='mesh')
        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        size = bounds[1] - bounds[0]  # (3,)
        half = size / 2.0
        # except Exception as e:
        #     print(f"[WARN] Failed to load mesh bounds for {mesh_path}: {e}")
        #     half = np.array([0.1, 0.1, 0.1], dtype=np.float32)  # fallback

        # --- actor.pose 중심 기준 가상 AABB 생성 ---
        pose_p = np.array(actor.pose.p, dtype=np.float32).reshape(3)
        min_bound = pose_p - half
        max_bound = pose_p + half

        # --- 디버그 출력 ---
        # print(f"[DEBUG AABB] {os.path.basename(mesh_path)} | size={size} | min={min_bound}, max={max_bound}")

        return min_bound, max_bound    




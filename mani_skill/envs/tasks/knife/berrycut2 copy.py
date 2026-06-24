from typing import Dict, Any
import numpy as np
import torch
import sapien
import os
import random
import trimesh

from transforms3d.euler import euler2quat, euler2mat
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder  # optional
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig

# ===========================================================
# CONFIG
# ===========================================================
KNIFE_STL_PATH = "/home/por1329/2025cvpr/ManiSkill/mani_skill/assets/robots/panda/franka_description/meshes/visual/knifev2.stl"
HAND_TO_KNIFE_XYZ = (0.1, 0.0, 0.1)
HAND_TO_KNIFE_RPY = (1.5708, -1.5708, -1.5708)
CONTACT_Z_MARGIN  = 0.006
CONTACT_XY_MARGIN = 0.02
KNIFE_LINK_NAME   = "tool_knife"

@register_env("Robocutv1", max_episode_steps=1000)
class RoboCasaCuttingEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]
    SUPPORTED_ROBOTS = ["pk", "pkv", "pkours"]

    def __init__(self, *args, robot_uids="pkv", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

        self._knife_local_tip_z = -0.20
        self._contact_frame = None
        self._already_cut = None
        self._ever_success = None

    # ===========================================================
    # 기본 설정
    # ===========================================================
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
        pose = sapien_utils.look_at(eye=[2.8, -2.5, 1.2], target=[0.0, 0, 0.3])
        return [
            CameraConfig("base_camera", pose, 512, 512, 1.2, 0.01, 100)
        ]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            "render_camera",
            sapien_utils.look_at(eye=[3.0, -3.0, 2.0], target=[0.0, 0, 0.5]),
            1024, 1024, 0.8, 0.01, 100
        )

    # ===========================================================
    # 로드
    # ===========================================================


    def _load_agent(self, options: dict):
        # 90도 회전 (yaw = +π/2)
        quat = euler2quat(0, np.pi / 2, np.pi / 2)  # roll=0, pitch=0, yaw=90°
        # 월드 좌표 기준 이동
        pos = [0.4, 0.4, 0.1]
        # Pose(p, q)
        base_pose = sapien.Pose(p=pos, q=quat)
        # 로봇 로드
        super()._load_agent(options, base_pose)

    def _load_scene(self, options: dict):
        # ✅ RoboCasa 주방 배경 사용
        self.scene_builder = RoboCasaSceneBuilder(self)
        self.scene_builder.build()

        # 보드 위 절단용 테이블 설정
        board_mat = sapien.render.RenderMaterial(base_color=[0.95, 0.92, 0.85, 1.0])
        board_builder = self.scene.create_actor_builder()
        board_builder.add_box_visual(half_size=[0.35, 0.55, 0.01], material=board_mat)
        board_builder.add_box_collision(half_size=[0.35, 0.55, 0.01])
        board_builder.initial_pose = sapien.Pose(p=[-0.1, 0, 0.01])
        self.board = board_builder.build_static(name="board")
        self._already_cut = np.zeros(self.num_envs, dtype=bool)
        self._ever_success = np.zeros(self.num_envs, dtype=bool)
        # 과일 로드
        fruit_path = "./mani_skill/assets/fruits/peach.obj";
        self.block_fruit = []
        for env_idx in range(self.num_envs):
            from pathlib import Path
            p = Path(fruit_path).expanduser()
            assert p.exists(), f"OBJ not found: {p}"

            obj_builder = self.scene.create_actor_builder()
            obj_builder.set_scene_idxs([env_idx])
            obj_builder.add_visual_from_file(str(p), scale=[1, 1, 1])
            obj_builder.add_sphere_collision(radius=1e-6)
            self.block_fruit.append(obj_builder.build(name=f"fruit_{env_idx}"))

        # 조명 세팅
        self.scene.set_ambient_light([0.4, 0.4, 0.4])
        self.scene.add_directional_light([0.5, 1, -1], [1, 1, 1], shadow=False)

        # STL 로드 (칼 끝 위치)
        try:
            mesh = trimesh.load(KNIFE_STL_PATH, force='mesh')
            zmin = float(mesh.bounds[0, 2])
            self._knife_local_tip_z = zmin
            print(f"[knife] loaded tip z={zmin:.4f}")
        except Exception as e:
            print(f"[knife] STL load failed: {e}")

    # ===========================================================
    # 초기화
    # ===========================================================
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.scene_builder.initialize(env_idx)

        # 로봇 pose 재설정 (여기서 하면 scene_builder가 덮어쓰지 못함)
        quat = euler2quat(0, 0, np.pi / 2)
        pos = [0.8, 0.8, 0.1]
        self.agent.robot.set_pose(sapien.Pose(p=pos, q=quat))

        # 과일 배치 등 나머지 초기화
        for e in env_idx.tolist():
            rand_x = -0.1 + np.random.uniform(0.05, 0.15)
            rand_y = np.random.uniform(-0.15, 0.15)
            self.block_fruit[e].set_pose(
                Pose.create_from_pq([rand_x, rand_y, 0.05], euler2quat(np.pi/2, 0, np.pi))
            )

    # ===========================================================
    # 평가 및 리워드
    # ===========================================================
    def evaluate(self):
     
        result = {"cut_done": bool(self._already_cut.any())}
        result["success"] = torch.tensor([result["cut_done"]], dtype=torch.bool)
        return result

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, dtype=torch.float32)


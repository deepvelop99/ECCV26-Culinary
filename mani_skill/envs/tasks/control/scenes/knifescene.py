# mani_skill/envs/knife_env.py

import sapien
from sapien import Pose
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.agents.franka_with_knife import FrankaWithKnife

@register_env("KnifeManipulation-v1", max_episode_steps=200)
class KnifeManipulationEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["franka_with_knife"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="franka_with_knife", **kwargs)

    def _load_agent(self, options: dict):
        self.agent = FrankaWithKnife(self.scene, self._sim_config)
        self.agent.load()
        self.agent.set_pose(Pose([0, 0, 0]))

    def _load_scene(self, options: dict):
        
        sapien_utils.create_ground(self.scene)

      
        table_builder = self.scene.create_actor_builder()
        table_builder.add_box_collision(half_size=[0.4, 0.4, 0.05])
        table_builder.add_box_visual(half_size=[0.4, 0.4, 0.05], color=[0.7, 0.5, 0.3])
        table = table_builder.build_static(name="table")
        table.set_pose(Pose([0.5, 0, 0.05]))

      
        cube_builder = self.scene.create_actor_builder()
        cube_builder.add_box_collision(half_size=[0.05, 0.05, 0.05])
        cube_builder.add_box_visual(half_size=[0.05, 0.05, 0.05], color=[0.1, 0.5, 0.8])
        cube = cube_builder.build(name="cube")
        cube.set_pose(Pose([0.5, 0, 0.15]))

        self.scene_objects = {"table": table, "cube": cube}

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.8, 0, 0.6], [0.5, 0, 0.2])
        #pose = sapien_utils.look_at([0.5, -0., 0.9],  [0.5, 0, 0.2])

        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 3, 0.01, 10)]

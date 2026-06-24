# This file defines a custom robot agent for the ManiSkill simulation environment.
# The agent is a Panda robot equipped with a knife end-effector.

from copy import deepcopy
import numpy as np
import sapien
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from typing import Dict
# We also import the custom PDJointPosVelControllerConfig2 and PDJointPosVelController2
# for the new control modes.
from .controllers.pd_ee_pv import PDJointPosVelControllerConfig2, PDJointPosVelController2


# The @register_agent() decorator makes this agent available for use in ManiSkill environments.
@register_agent()
class PandaKnifev2(BaseAgent):
    # Unique identifier for this agent.
    uid = "pkhs"

    # Path to the robot's URDF file, which defines its physical properties.
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_knife_hs.urdf"
    
    urdf_config = dict(
        _materials=dict(
            knife_mat=dict(static_friction=0.1, dynamic_friction=0.1, restitution=0.5),
        ),
        link=dict(
            tool_knife=dict(material="knife_mat", patch_radius=0.05, min_patch_radius=0.01),
        ),
    )
    # Defines the robot's initial configuration (rest pose) using a Keyframe object.
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                0.0,
                np.pi / 8,
                0,
                -np.pi * 5 / 8,
                0,
                np.pi * 3 / 4,
                np.pi / 4,
                0.04,
                0.04,
            ], dtype=np.float32),
            pose=sapien.Pose(),
        )
    )

    # Lists the names of the arm joints.
    arm_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    
    # Lists the names of the gripper joints.
    gripper_joint_names = [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    # Defines the name of the end-effector link (Tool Center Point).
    ee_link_name = "panda_hand_tcp"

    # Defines PID control parameters for the arm and gripper.
    arm_stiffness = 1e10
    arm_damping = 1e1
    arm_force_limit = 1000000000000

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100
  
    @property
    def _controller_configs(self):
        # This property defines and returns a dictionary of all available controllers.

        # ---------------- Arm Controllers ----------------
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None, upper=None,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1, upper=0.1,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        
        # New PDJointPosVelController configurations.
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig2(
            self.arm_joint_names,
            lower=None, upper=None,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
            vel_lower=-1.0, vel_upper=1.0,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig2(
            self.arm_joint_names,
            lower=-0.1, upper=0.1,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
            vel_lower=-1.0, vel_upper=1.0,
        )

        # ---------------- Gripper Controller ----------------
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01, upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"panda_finger_joint2": {"joint": "panda_finger_joint1"}},
        )

        # Dictionary of all available controller combinations.
        controller_configs = dict(
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos),
            pd_joint_pos_vel=dict(arm=arm_pd_joint_pos_vel, gripper=gripper_pd_joint_pos),
            pd_joint_delta_pos_vel=dict(arm=arm_pd_joint_delta_pos_vel, gripper=gripper_pd_joint_pos),
        )

        # Return a deep copy to prevent users from modifying the original configs.
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        # This method is called after the robot is initialized in the scene.
        # It gets the specific links needed for future reference.
        self.finger1_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "panda_leftfinger")
        self.finger2_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "panda_rightfinger")
        self.finger1pad_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "panda_leftfinger_pad")
        self.finger2pad_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "panda_rightfinger_pad")
        
        # Get the TCP link, with fallback options if not found.
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
        if self.tcp is None:
            candidates = ["tool_tcp", "panda_hand_tcp", "panda_hand", "panda_link7"]
            names = [l.get_name() for l in self.robot.get_links()]
            for cand in candidates:
                self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), cand)
                if self.tcp is not None:
                    print(f"[PandaKnife] EE '{self.ee_link_name}' not found -> using '{cand}'")
                    self.ee_link_name = cand
                    break
        assert self.tcp is not None, (
            f"[PandaKnife] EE link '{self.ee_link_name}' not found. "
            f"Available links: {names}"
        )

    def get_state(self) -> Dict[str, np.ndarray]:
        # Overrides the BaseAgent's get_state method to include EEF pose and velocity.
        state = super().get_state()
        try:
            controller: CombinedController = self.controller
            arm_controller: PDJointPosVelController2 = controller.controllers["arm"]
            eef_state = arm_controller.get_eef_state(eef_name=self.ee_link_name)
            state.update(eef_state)
        except (KeyError, AttributeError):
            pass
        return state

    def is_static(self, threshold: float = 0.2):
        # Checks if the arm joints are static (velocity below a threshold).
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        # Returns the 3D position of the TCP.
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        # Returns the full pose (position and orientation) of the TCP.
        return self.tcp.pose
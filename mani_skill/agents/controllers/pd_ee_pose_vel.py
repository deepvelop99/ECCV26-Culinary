from dataclasses import dataclass
from typing import Sequence, Union
import numpy as np
import torch
from gymnasium import spaces

from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.types import DriveMode
from mani_skill.agents.controllers.base_controller import BaseController, ControllerConfig
from mani_skill.agents.controllers.utils.kinematics import Kinematics
import sapien.core as sapien


class PDEEPosVelController(BaseController):
    """
    Hybrid EE controller: IK 기반 위치 + velocity 제어
    Action = [Δx, Δy, Δz, vx, vy, vz, wx, wy, wz] (총 9차원)
      - Δpos: 목표 위치 오프셋 (IK 기반)
      - vx, vy, vz: 선속도
      - wx, wy, wz: 각속도
    """
    config: "PDEEPosVelControllerConfig"

    def _initialize_joints(self):
        super()._initialize_joints()
        self.ee_link = sapien_utils.get_obj_by_name(
            self.articulation.get_links(), self.config.ee_link
        )
        assert self.ee_link is not None, f"EE link {self.config.ee_link} not found!"
        self.kinematics = Kinematics(
            self.config.urdf_path, self.config.ee_link,
            self.articulation, self.active_joint_indices,
        )

    def _initialize_action_space(self):
        low = np.float32(np.hstack([
            np.broadcast_to(self.config.pos_lower, 3),
            np.broadcast_to(self.config.linear_lower, 3),
            np.broadcast_to(self.config.angular_lower, 3),
        ]))
        high = np.float32(np.hstack([
            np.broadcast_to(self.config.pos_upper, 3),
            np.broadcast_to(self.config.linear_upper, 3),
            np.broadcast_to(self.config.angular_upper, 3),
        ]))
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)

        # === 1) 위치 오프셋 기반 IK ===
        dpos = action[:3]
        current_pose = self.ee_link.get_pose()
        target_pos = np.asarray(current_pose.p) + dpos
        target_pose = sapien.Pose(p=target_pos, q=current_pose.q)

        qpos_target = self.kinematics.compute_ik(
            pose=target_pose,
            q0=self.articulation.get_qpos(),
            is_delta_pose=True,
            current_pose=current_pose
        )

        if qpos_target is not None:
            self.articulation.set_drive_targets(qpos_target)

        # === 2) 속도 제어 ===
        v, w = action[3:6], action[6:9]
        target_twist = np.concatenate([v, w])  # (6,)

        J = self.kinematics.compute_jacobian()
        JJt = J @ J.T
        lam = self.config.damping_factor
        J_pinv = J.T @ np.linalg.inv(JJt + (lam**2) * np.eye(J.shape[0]))
        qvel = J_pinv @ target_twist

        if isinstance(self.articulation.get_qpos(), torch.Tensor):
            qvel = torch.as_tensor(qvel, dtype=torch.float32, device=self.device)

        self.articulation.set_joint_drive_velocity_targets(
            qvel, self.joints, self.active_joint_indices
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(ee={self.config.ee_link}, dof={len(self.joints)})"


@dataclass
class PDEEPosVelControllerConfig(ControllerConfig):
    pos_lower: Union[float, Sequence[float]] = -0.05
    pos_upper: Union[float, Sequence[float]] = 0.05
    linear_lower: Union[float, Sequence[float]] = -0.2
    linear_upper: Union[float, Sequence[float]] = 0.2
    angular_lower: Union[float, Sequence[float]] = -0.5
    angular_upper: Union[float, Sequence[float]] = 0.5
    damping: Union[float, Sequence[float]] = 1.0
    force_limit: Union[float, Sequence[float]] = 100.0
    ee_link: str = None
    urdf_path: str = None
    damping_factor: float = 0.05
    drive_mode: Union[Sequence[DriveMode], DriveMode] = "force"

    controller_cls = PDEEPosVelController

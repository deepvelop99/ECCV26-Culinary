from dataclasses import dataclass
from typing import Sequence, Union, Dict

import numpy as np
import torch
from gymnasium import spaces

from .pd_joint_pos import PDJointPosController, PDJointPosControllerConfig


class PDJointPosVelController2(PDJointPosController):
    config: "PDJointPosVelControllerConfig"
    _target_qvel = None
    sets_target_qpos = True
    sets_target_qvel = True

    def _initialize_action_space(self):
        joint_limits = self._get_joint_limits()
        pos_low, pos_high = joint_limits[:, 0], joint_limits[:, 1]
        vel_low = np.broadcast_to(self.config.vel_lower, pos_low.shape)
        vel_high = np.broadcast_to(self.config.vel_upper, pos_high.shape)
        low = np.float32(np.hstack([pos_low, vel_low]))
        high = np.float32(np.hstack([pos_high, vel_high]))
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        super().reset()
        if self._target_qvel is None:
            self._target_qvel = self.qvel.clone()
        else:
            self._target_qvel[self.scene._reset_mask] = torch.zeros_like(
                self._target_qpos[self.scene._reset_mask], device=self.device
            )

    def set_drive_velocity_targets(self, targets):
        self.articulation.set_joint_drive_velocity_targets(
            targets, self.joints, self.active_joint_indices
        )

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)
        nq = len(action[0]) // 2

        self._step = 0
        self._start_qpos = self.qpos

        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos = self._target_qpos + action[:, :nq]
            else:
                self._target_qpos = self._start_qpos + action[:, :nq]
        else:
            self._target_qpos = torch.broadcast_to(
                action[:, :nq], self._start_qpos.shape
            )

        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

        self._target_qvel = action[:, nq:]
        self.set_drive_velocity_targets(self._target_qvel)

    # ---------- 추가 부분 ----------
    def get_eef_state(self, eef_name: str = "panda_hand_tcp") -> Dict[str, np.ndarray]:
        """
        Return EEF pose and velocity.
        eef_name: articulation link name of the EEF (default: 'panda_hand_tcp')
        """
        link = self.articulation.links_map[eef_name]

        pose = link.pose
        pos = pose.p
        quat = pose.q

        lin_vel = link.get_linear_velocity()
        ang_vel = link.get_angular_velocity()

        return {
            "eef_pos": pos.cpu().numpy(),
            "eef_quat": quat.cpu().numpy(),
            "eef_lin_vel": lin_vel.cpu().numpy(),
            "eef_ang_vel": ang_vel.cpu().numpy(),
        }

    def get_state(self) -> dict:
        state = super().get_state()
        try:
            eef_state = self.get_eef_state()
            state.update(eef_state)
        except KeyError:
            pass
        return state
    # -----------------------------


@dataclass
class PDJointPosVelControllerConfig2(PDJointPosControllerConfig):
    """
    Joint Position + Velocity PD Controller Config
    """
    controller_cls = PDJointPosVelController2

    # --- 기존 PDJointPosControllerConfig 인자 ---
    lower: Union[None, float, Sequence[float]] = None
    upper: Union[None, float, Sequence[float]] = None
    stiffness: Union[float, Sequence[float]] = 1e9
    damping: Union[float, Sequence[float]] = 1e9
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    use_delta: bool = False
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    drive_mode: Union[Sequence[str], str] = "force"

    # --- 추가된 속도 범위 인자 ---
    vel_lower: Union[float, Sequence[float]] = -1.0
    vel_upper: Union[float, Sequence[float]] = 1.0

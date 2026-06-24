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
KNIFE_STL_PATH = "/home/por1329/2025cvpr/ManiSkill/mani_skill/assets/robots/panda/franka_description/meshes/visual/knifev2.stl"

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
        pose = sapien_utils.look_at(eye=[0.35, 0, 0.9], target=[-0.1, 0, 0.05])
        return [
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
        
    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
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
        )

    # ---------- 로딩 ----------
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build()
        
        # board
        board_mat = sapien.render.RenderMaterial(base_color=[0.95, 0.92, 0.85, 1.0])
        board_builder = self.scene.create_actor_builder()
        board_builder.add_box_visual(half_size=self.BOARD_SIZE.tolist(), material=board_mat)
        board_builder.add_box_collision(half_size=self.BOARD_SIZE.tolist())
        board_builder.initial_pose = sapien.Pose(p=[-0.1, 0, self.BOARD_SIZE[2]])
        self.board = board_builder.build_static(name="board")

        # apple (visual only, no collision!)
        self.block_apple = []
        
        for env_idx in range(self.num_envs):
            from pathlib import Path
            path = "./mani_skill/assets/fruits/banana.obj" # banana
            # path = "/home/por1329/maniskill/ManiSkill/mani_skill/assets/fruits/apple.obj" # apple
            # path = "/home/por1329/maniskill/ManiSkill/mani_skill/assets/fruits/cucumber.obj" # cucumber
            # path = "/home/por1329/maniskill/ManiSkill/mani_skill/assets/fruits/melon.obj" # melon
            # path = "/home/por1329/maniskill/ManiSkill/mani_skill/assets/fruits/orange.obj" # orange
            # path = "/home/por1329/maniskill/ManiSkill/mani_skill/assets/fruits/peach.obj" # peach
            # path = "/home/por1329/maniskill/ManiSkill/mani_skill/assets/fruits/strawberry.obj" # strawberry                           
            # path = "/home/por1329/maniskill/10868_birthday-cake_v3.obj" # hbd
            p = Path(path).expanduser()
            assert p.exists(), f"OBJ not found: {p}"
            obj_builder = self.scene.create_actor_builder()
            obj_builder.set_scene_idxs([env_idx])
            # obj_builder.add_visual_from_file(str(p), scale=[0.02, 0.02, 0.02])
            obj_builder.add_sphere_collision(radius=1e-1000)
            # obj_builder.add_nonconvex_collision_from_file(str(p), scale=[1, 1, 1])
            obj_builder.add_visual_from_file(str(p), scale=[1, 1, 1])
            # ⛔ no collision
            self.block_apple.append(obj_builder.build(name=f"block_apple_{env_idx}"))
            
        # buffers
        self._ensure_buffers()
        self._prev_tcp_z = torch.zeros(self.num_envs, dtype=torch.float32)
        self._contact = torch.zeros(self.num_envs, dtype=torch.bool)
        self._contact_prev = torch.zeros(self.num_envs, dtype=torch.bool)
        
        self.board_center = np.array([-0.1, 0.0, self.BOARD_SIZE[2]])
        self.block_center0 = self.board_center + np.array([0.0, 0.0, self.BLOCK_HALF_SIZE[2]])
        self.block_center1 = self.board_center
        self.block_rot_quat = euler2quat(np.pi/2, 0, np.pi)
        self.update_block_rot_quat = euler2quat(np.pi/2, 0, np.pi) # apple update rotation
        # self.block_rot_quat = euler2quat(0, 0, np.pi/2) # Chang young HB

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

    # ---------- 에피소드 초기화 ----------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            qpos = self.agent.keyframes["rest"].qpos
            self.agent.robot.set_qpos(qpos)
        assert hasattr(self.agent, "tcp") and (self.agent.tcp is not None)
        self._ensure_buffers()

        self.table_scene.initialize(env_idx)

        for e in env_idx.tolist():
            # 현재 block 초기 orientation(quat) 가져오기
            init_quat = self.block_rot_quat  

            # --- 위치만 랜덤화 ---
            rand_x = self.board_center[0] + np.random.uniform(0.05, 0.15)
            rand_y = self.board_center[1] + np.random.uniform(-0.15, 0.15)
            rand_z = self.board_center[2]  # 보드 위

            # 초기 quat 그대로 유지
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
        tip_world = T_world_knife @ tip_local
        return tip_world[:, :3]

    # ---------- 시뮬 스텝 이후 ----------
   
    def _after_control_step(self):
        self._ensure_buffers()
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

        # 칼 끝 + 손잡이 쪽 좌표
        tip_p  = self._get_tip_from_eef()        # (N,3)
        base_p = self.agent.tcp.pose.p           # (N,3)

        tip_p = self._get_tip_from_eef()
        contact = torch.zeros(self.num_envs, dtype=torch.bool, device=tip_p.device)

        # 잘린 mesh 경로 (banana 예시, apple/cucumber 등 교체 가능)
        cut_paths = [
            "./mani_skill/assets/fruits/update/banana/banana_left.obj",
            "./mani_skill/assets/fruits/update/banana/banana_right.obj",
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

            tip_z = tip_p[e, 2].item()
            board_z = self.board_center[2]
            reached_board = tip_z <= board_z + 0.1   # 작은 margin 허용

            # 보드에 닿았고, 아직 잘리지 않은 경우만 실행
            if reached_board and (not self._already_cut[e]):
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
                for cut_path in cut_paths:
                    from pathlib import Path
                    p = Path(cut_path).expanduser()
                    assert p.exists(), f"OBJ not found: {p}"

                    builder = self.scene.create_actor_builder()
                    builder.add_visual_from_file(str(p), scale=[1, 1, 1])
                    builder.add_sphere_collision(radius=1e-100)

                    cut_actor = builder.build_static(
                        name=f"block_apple_cut_{os.path.basename(cut_path)}_{e}"
                    )

                    # z는 보드 높이에 flush, xy/quat은 기존 유지
                    pose_p = np.array(current_pose.p, dtype=float).reshape(-1)
                    pose_q = np.array(current_pose.q, dtype=float).reshape(-1)
                    pose_p[2] = board_z
                    if "left" in cut_path.lower():
                        pose_p[1] += 0.001
                        
                    elif "right" in cut_path.lower():
                        pose_p[1] -= 0.001
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

        # 3) XY 위치 (중심선 근처)
        eef_xy = eef_pos[:, :2]
        obj_center_xy = self.block_center1[:2]
        dist = np.abs(eef_xy - obj_center_xy)
        near_center_line = bool((dist[:, 0] < 0.3).all())
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
            f"(max_x_err={dist[:,0].max():.3f} m) "
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
        print(f"[Step Debugeval] reward={reward}, info={info}")
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
            mesh_path = "./mani_skill/assets/fruits/banana.obj"

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




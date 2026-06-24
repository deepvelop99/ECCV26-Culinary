from typing import Dict, Any, Union
import numpy as np
import sapien
import torch
import os
import trimesh
from transforms3d.euler import euler2quat, euler2mat
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from pathlib import Path

# ==========================
# 설정 / 경로 / 파라미터
# ==========================
KNIFE_STL_PATH = "/home/por1329/2025cvpr/ManiSkill/mani_skill/assets/robots/panda/franka_description/meshes/visual/knifev2.stl"
HAND_TO_KNIFE_XYZ = (0.1, 0.0, 0.1)
HAND_TO_KNIFE_RPY = (1.5708, -1.5708, -1.5708)
KNIFE_LINK_NAME = "tool_knife"

@register_env("cucumber3cut", max_episode_steps=1000)
class TableTopCuttingEnv(BaseEnv):
    _sample_video_link = None
    KNIFE_TIP_LOCAL_Z_DEFAULT = -0.20
    BOARD_THICKNESS = 0.02
    BOARD_SIZE = np.array([0.35, 0.55, BOARD_THICKNESS / 2])
    BLOCK_HALF_SIZE = np.array([0.1, 0.1, 0.1])

    SUPPORTED_REWARD_MODES = ["none"]
    SUPPORTED_ROBOTS = ["pk", "pkv", "pkours"]

    def __init__(self, *args, robot_uids="pkv", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self._knife_local_tip_z = self.KNIFE_TIP_LOCAL_Z_DEFAULT
        self._cut_count = None
        self._contact = None
        self._contact_prev = None
        self._contact_frame = None
        self._prev_tcp_z = None
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
        pose = sapien_utils.look_at(eye=[0.5, -0.5, 0.8], target=[0, 0, 0])
        return [CameraConfig("base_camera", pose=pose, width=256, height=256, fov=1.2, near=0.01, far=100)]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            uid="render_camera",
            pose = sapien_utils.look_at(eye=[0.5, -0.5, 0.8], target=[0, 0, 0]),
            # width=256, height=256, far=25, fov=0.63,
            width=256, height=256, fov=1.2, near=0.01, far=100
        )

    # ---------- 로딩 ----------
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build()

        # Board
        board_mat = sapien.render.RenderMaterial(base_color=[0.95, 0.92, 0.85, 1.0])
        board_builder = self.scene.create_actor_builder()
        board_builder.add_box_visual(half_size=self.BOARD_SIZE.tolist(), material=board_mat)
        board_builder.add_box_collision(half_size=self.BOARD_SIZE.tolist())
        board_builder.initial_pose = sapien.Pose(p=[-0.1, 0, self.BOARD_SIZE[2]])
        self.board = board_builder.build_static(name="board")

        # Object (cucumber)
        self.block_apple = []
        for env_idx in range(self.num_envs):
            path = "./mani_skill/assets/fruits/cucumber.obj"
            p = Path(path).expanduser()
            assert p.exists(), f"OBJ not found: {p}"
            builder = self.scene.create_actor_builder()
            builder.set_scene_idxs([env_idx])
            builder.add_visual_from_file(str(p), scale=[1, 1, 1])
            builder.add_sphere_collision(radius=1e-10)
            self.block_apple.append(builder.build(name=f"block_cucumber_{env_idx}"))

        self.board_center = np.array([-0.1, 0.0, self.BOARD_SIZE[2]])
        self.block_center0 = self.board_center + np.array([0.0, 0.0, self.BLOCK_HALF_SIZE[2]])
        self.block_rot_quat = euler2quat(np.pi / 2, 0, np.pi)
        self.scene.set_ambient_light([0.4, 0.4, 0.4])
        self.scene.add_directional_light([0.5, 1, -1], [1, 1, 1], shadow=False)
        self._ensure_buffers()

    # ---------- 에피소드 초기화 ----------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            qpos = self.agent.keyframes["rest"].qpos
            self.agent.robot.set_qpos(qpos)
        self.table_scene.initialize(env_idx)
        self._ensure_buffers()
        for e in env_idx.tolist():
            quat = self.block_rot_quat
            rand_x = self.board_center[0] + np.random.uniform(0.05, 0.15)
            rand_y = self.board_center[1] + np.random.uniform(-0.15, 0.15)

            rand_z = self.board_center[2]
            self.block_apple[e].set_pose(Pose.create_from_pq([rand_x, rand_y, rand_z], quat))
            self._cut_count[e] = 0
        self._prev_tcp_z = self.agent.tcp.pose.p[:, 2].clone().cpu()

    # ---------- Pose Utils ----------
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
        R = torch.tensor(euler2mat(*rpy), dtype=dtype, device=device)
        T = torch.eye(4, dtype=dtype, device=device)
        T[:3, :3] = R
        T[:3, 3] = torch.tensor(xyz, dtype=dtype, device=device)
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

    # ---------- collider 기반 AABB 계산 ----------
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
        # fallback: 기본 cucumber.obj
        if mesh_path is None or not os.path.exists(mesh_path):
            mesh_path = "./mani_skill/assets/fruits/cucumber.obj"

        # --- trimesh 로드로 실제 mesh 크기 계산 ---
   
        mesh = trimesh.load_mesh(mesh_path, force='mesh')
        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        size = bounds[1] - bounds[0]  # (3,)
        half = size / 3.0
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




    # ---------- 여러번 컷팅 ----------
    def _after_control_step(self):
        """여러 단계 컷팅 + collider AABB 기반 contact"""
        self._ensure_buffers()
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

        tip_p = self._get_tip_from_eef()
        contact = torch.zeros(self.num_envs, dtype=torch.bool, device=tip_p.device)
   
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

            # ===== 나머지 절삭 로직 =====
            tip_y = tip_p[e, 1].item() - 0.02
            tip_z = tip_p[e, 2].item()
            board_z = self.board_center[2]
            cut_stage = int(self._cut_count[e])
            reached_board = tip_z <= board_z + 0.25
            #print(contact[e])
            # tip_y = tip_p[e, 1].item() -0.02
            # tip_z = tip_p[e, 2].item()
            # board_z = self.board_center[2]
            # cut_stage = int(self._cut_count[e])
            # reached_board = tip_z <= board_z + 1e-1

            # === 디버그 출력 ===
            print(f"[CONTACT] env={e} | tip={tip_p[e].tolist()} | obj_p={pose_p.tolist()} | half={half.tolist()} | inside={inside_tip}")

            # ✅ AABB에서 오브젝트 크기 계산
            # object_width1 = float((aabb_max[0] - aabb_min[0]).item())
            # print(object_width1)
            # object_width = float((aabb_max[1] - aabb_min[1]).item())
            # print(object_width)
            object_width = float((aabb_max[2] - aabb_min[2]).item())
            #print(object_width3)
            obj_pose = obj_actor.pose
            pose_p = np.array(obj_pose.p, dtype=float).reshape(-1)
            object_center_y, object_center_z = pose_p[1], pose_p[2]

            # ✅ object width의 1/4 지점 기준으로 타겟 설정
            left_target_y = object_center_y - object_width / 4.0 - 0.005
            right_target_y = object_center_y + object_width / 4.0  + 0.0001
            center_target_y = object_center_y

            # === 상태 출력 ===
            print(
                f"[STATE] env={e} | stage={cut_stage} | tip_x={tip_y:.3f} | tip_z={tip_z:.3f} "
                f"| left_target={left_target_y:.3f} | center_target={center_target_y:.3f} | right_target={right_target_y:.3f} "
                f"| reached_board={reached_board}"
            )

            # ===== 1차 절삭 =====
            if cut_stage == 0 and abs(tip_y - left_target_y) <= 0.1 and reached_board :
                print(f"[CUT STAGE 1] env={e} | tip_y={tip_y:.3f} | target={left_target_y:.3f} | z={tip_z:.3f}")
                self._replace_with_single_mesh(
                    e,
                    "./mani_skill/assets/fruits/update/cucumber/multicut/3cut/cucumber_1cut_final.obj",
                    object_center_z,
                )
                self._cut_count[e] = 1

            # ===== 2차 절삭 =====
            elif cut_stage == 1 and abs(tip_y - center_target_y) <= 0.03 and reached_board:
                print(f"[CUT STAGE 2] env={e} | tip_x={tip_y:.3f} | target={center_target_y:.3f} | z={tip_z:.3f}")
                self._replace_with_single_mesh(
                    e,
                    "./mani_skill/assets/fruits/update/cucumber/multicut/3cut/cucumber_2cut_final.obj",
                    object_center_z,
                )
                self._cut_count[e] = 2

            # ===== 3차 절삭 =====
            elif cut_stage == 2 and abs(tip_y - right_target_y) <= 0.04 and reached_board:
                print(f"[CUT STAGE 3] env={e} | tip_x={tip_y:.3f} | target={right_target_y:.3f} | z={tip_z:.3f}")
                self._replace_with_single_mesh(
                    e,
                    "./mani_skill/assets/fruits/update/cucumber/multicut/3cut/cucumber_3cut_final.obj",
                    object_center_z,
                )
                self._cut_count[e] = 3

        self._contact_prev = self._contact.clone()
        self._contact = contact.clone()
        self._contact_frame = contact.detach().cpu().numpy()
        self._prev_tcp_z = tip_p[:, 2].detach().cpu()

        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
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


    # ---------- Mesh 교체 ----------
    def _replace_with_single_mesh(self, e, mesh_path: str, ref_z: float):
        print("mesh update")
        p = Path(mesh_path).expanduser()
        if not p.exists():
            print(f"[WARN] Mesh not found: {p}")
            return
        current_pose = (
            self.block_apple[e][0].pose if isinstance(self.block_apple[e], list)
            else self.block_apple[e].pose
        )
        if isinstance(self.block_apple[e], list):
            for piece in self.block_apple[e]:
                self.scene.remove_actor(piece)
        else:
            self.scene.remove_actor(self.block_apple[e])
        self.scene.update_render()

        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(str(p), scale=[1, 1, 1])
        builder.add_sphere_collision(radius=1e-10)
        new_actor = builder.build_static(name=f"block_cut_{os.path.basename(p)}_{e}")

        pose_p = np.array(current_pose.p, dtype=float).reshape(3)
        pose_q = np.array(current_pose.q, dtype=float).reshape(4)
        pose_p[2] = ref_z
        new_actor.set_pose(Pose.create_from_pq(pose_p, pose_q))
        self.block_apple[e] = new_actor

    # ---------- 평가 ----------
    def evaluate(self):
        result = {}
        try:
            cut_arr = np.array(self._cut_count)
            cut_done = bool((cut_arr >= 3).any())
        except Exception as e:
            print(f"[Evaluate] cut_count check failed: {e}")
            cut_done = False
        result["cut_done"] = cut_done
        result["success"] = torch.tensor([cut_done], dtype=torch.bool)
        print(f"[Evaluate] cut_count={self._cut_count.tolist()} -> cut_done={cut_done}")
        return result

    # ---------- 버퍼 ----------
    def _ensure_buffers(self):
        raw_num = getattr(self, "num_envs", 1)
        num = int(raw_num) if not isinstance(raw_num, int) else raw_num
        def _keep_old(name, dtype=bool):
            old = getattr(self, name, None)
            new = np.zeros(num, dtype=dtype)
            if isinstance(old, np.ndarray):
                new[:min(num, old.size)] = old[:min(num, old.size)]
            setattr(self, name, new)
        self._contact = torch.zeros(num, dtype=torch.bool)
        self._contact_prev = torch.zeros(num, dtype=torch.bool)
        self._prev_tcp_z = torch.zeros(num, dtype=torch.float32)
        _keep_old("_ever_success", dtype=bool)
        old_cut = getattr(self, "_cut_count", None)
        new_cut = np.zeros(num, dtype=int)
        if isinstance(old_cut, np.ndarray):
            new_cut[:min(num, old_cut.size)] = old_cut[:min(num, old_cut.size)]
        self._cut_count = new_cut
        if not isinstance(getattr(self, "_contact_frame", None), np.ndarray) or self._contact_frame.shape != (num,):
            old = getattr(self, "_contact_frame", None)
            self._contact_frame = np.zeros(num, dtype=bool)
            if isinstance(old, np.ndarray):
                self._contact_frame[:min(num, old.size)] = old[:min(num, old.size)]
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
            mesh_path = "./mani_skill/assets/fruits/cucumber.obj"

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


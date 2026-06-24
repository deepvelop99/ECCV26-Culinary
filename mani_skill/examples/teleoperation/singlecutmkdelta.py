import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from typing import Annotated
import tyro
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers.recordori import RecordEpisode
import os, random

OPEN = 1.0
CLOSED = -1.0

# ================================================================
# Argument 정의
# ================================================================
@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "bananacut"
    obs_mode: str = "state_dict"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "pkour"
    record_dir: str = "middlecutdelta"
    save_video: bool = False

def parse_args() -> Args:
    return tyro.cli(Args)

# ================================================================
# Helper: EEF 선형 이동 (pd_ee_delta_pos, 위치 오차만)
# ================================================================
def move_to_pose(env: BaseEnv, start_pose_w: sapien.Pose, target_pose_w: sapien.Pose,
                 steps=100, kp=7.0, clip=0.03, gripper_delta=0.0):
    """
    - control_mode: pd_ee_delta_pos (위치 3D만 제어)
    - 매 스텝에서 루트 프레임 기준 dpos = (p_target_base - p_current_base) * kp 를 클립해서 action으로 전달
    - 일부 환경은 arm(3) + gripper(1) → 총 4차원 action을 요구할 수 있으므로 런타임에 자동 적응
    """
    start_p = np.array(start_pose_w.p, dtype=np.float32).reshape(-1)
    end_p   = np.array(target_pose_w.p, dtype=np.float32).reshape(-1)

    # action 차원 파악 (3 또는 4가 일반적)
    action_dim = int(np.prod(env.action_space.shape))
    has_gripper = action_dim >= 4  # 보수적으로 4이상이면 그리퍼 차원이 있다고 가정

    for i in range(steps):
        alpha = i / max(steps - 1, 1)
        curr_p_w = (1 - alpha) * start_p + alpha * end_p  # 이번 스텝 목표(월드)

        # 루트 프레임으로 변환
        base_pose = env.agent.robot.get_links()[0].pose  # root_link (base)
        # 현재 EEF (base 프레임)
        ee_base = (base_pose.inv() * env.agent.tcp.pose).p
        ee_base = np.array(ee_base, dtype=np.float32).reshape(-1)
        # 이번 스텝 목표 (base 프레임)
        target_base = (base_pose.inv() * sapien.Pose(p=curr_p_w)).p
        target_base = np.array(target_base, dtype=np.float32).reshape(-1)

        # Δ 위치 (root 프레임) + 게인/클립
        dpos = (target_base - ee_base) * kp
        dpos = np.clip(dpos, -clip, clip).astype(np.float32)

        if has_gripper:
            action = np.concatenate([dpos, np.array([gripper_delta], dtype=np.float32)], axis=0)
        else:
            action = dpos

        env.step(action)

# ================================================================
# Main 실행
# ================================================================
def main(args: Args):
    # -----------------------------
    # 경로 및 seed 설정
    # -----------------------------
    seed = random.randint(0, 4096)
    output_dir = f"/data2/{args.record_dir}/{args.env_id}/auto_{seed}/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Trajectory will be saved to: {output_dir}")

    # -----------------------------
    # 환경 생성 (회전 없는 위치 델타 컨트롤러)
    # -----------------------------
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_ee_delta_pos",  # ✅ 위치만 델타 제어
        render_mode="rgb_array",
        reward_mode="none",
        enable_shadow=False,
        robot_uids=args.robot_uid,
    )

    # (선택) 컨트롤러 설정 강제 (버전/환경에 따라 있을 수도/없을 수도 있음)
    if hasattr(env.agent, "controller_configs") and "arm" in env.agent.controller_configs:
        cfg = env.agent.controller_configs["arm"]
        if hasattr(cfg, "use_delta"):
            cfg.use_delta = True                   # 델타 모드 보장
        if hasattr(cfg, "frame"):
            cfg.frame = "root_translation"         # 위치는 루트 프레임
        # 정규화 끄고 SI 단위로 쓰고 싶다면:
        if hasattr(cfg, "normalize_action"):
            cfg.normalize_action = False           # 가능하면 SI 단위 직접 사용

    # -----------------------------
    # RecordEpisode 래퍼
    # -----------------------------
    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=args.save_video,
        info_on_video=False,
        record_env_state=True,
        source_type="scripted_teleop",
        source_desc="auto cutting (delta-pos only)",
    )

    # -----------------------------
    # 환경 초기화 및 EEF 안정화
    # -----------------------------
    obs, _ = env.reset(seed=seed, options=dict(save_trajectory=True))

    # 로봇 자세를 rest pose로 강제 초기화 (가능할 때)
    if hasattr(env.agent, "keyframes") and "rest" in env.agent.keyframes:
        env.agent.robot.set_qpos(env.agent.keyframes["rest"].qpos)
        env.agent.robot.set_qvel(np.zeros_like(env.agent.robot.get_qvel()))
        # action 차원에 맞춰 0 벡터 한 번 step
        action_dim = int(np.prod(env.action_space.shape))
        env.step(np.zeros(action_dim, dtype=np.float32))
        env.render()

    # ✅ 현재 EEF pose (world 기준)
    init_pose_world = env.agent.tcp.pose
    print(f"[INIT(world)] EEF pose: {init_pose_world.p}")

    # -----------------------------
    # Object pose 가져오기 (world)
    # -----------------------------
    obj_actor = getattr(env, "block_apple", [None])[0]
    assert obj_actor is not None, "환경에 object가 없습니다."
    obj_pos = np.array(obj_actor.pose.p, dtype=np.float32).reshape(-1)
    print(f"[INFO] Object center (world): {obj_pos}")

    # -----------------------------
    # Target Pose 정의 (world 좌표계)
    # -----------------------------
    approach_randx = random.uniform(-0.35, -0.1)   # 접근 X 오프셋
    randz = random.uniform(0.3, 0.5)               # 접근 높이
    randomspeed = random.randint(10, 100)          # 절삭 단계 스텝 수

    approach_pose = sapien.Pose(p=obj_pos + np.array([approach_randx, 0.0, randz], dtype=np.float32))
    cut_pose      = sapien.Pose(p=obj_pos + np.array([approach_randx, 0.0, 0.1], dtype=np.float32))
    retreat_pose  = sapien.Pose(p=obj_pos + np.array([approach_randx, 0.0, randz], dtype=np.float32))

    # -----------------------------
    # 자동 절삭 시퀀스 (월드 타깃 → 루트 프레임 오차만 전달)
    # -----------------------------
    print("[Stage 1] Approach")
    move_to_pose(env, init_pose_world, approach_pose, steps=100, kp=7.0, clip=0.03, gripper_delta=0.0)

    print("[Stage 2] Cut Down")
    move_to_pose(env, approach_pose,    cut_pose,    steps=randomspeed, kp=7.0, clip=0.03, gripper_delta=0.0)

    print("[Stage 3] Retreat")
    move_to_pose(env, cut_pose,         retreat_pose,steps=100, kp=7.0, clip=0.03, gripper_delta=0.0)

    print("[DONE] Cutting trajectory executed successfully.")

    # -----------------------------
    # 파일 저장
    # -----------------------------
    h5_file_path = env._h5_file.filename
    json_file_path = env._json_path
    env.close()

    print(f"[SAVED] Trajectory saved → {h5_file_path}")
    print(f"[META] JSON metadata → {json_file_path}")

    if args.save_video:
        print("[INFO] Video saving handled by RecordEpisode.]")
    print(f"[INIT(world)] Final EEF pose: {init_pose_world.p}")

if __name__ == "__main__":
    main(parse_args())

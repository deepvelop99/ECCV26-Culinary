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
    record_dir: str = "sawcutfinal"
    save_video: bool = False


def parse_args() -> Args:
    return tyro.cli(Args)


# ================================================================
# Helper: EEF 선형 이동
# ================================================================
def move_to_pose(env: BaseEnv, start_pose: sapien.Pose, target_pose: sapien.Pose,
                 steps=100, gripper=CLOSED, sawcut=False, amplitude=0.02, frequency=3.0):
    """
    EEF 선형 이동 (7D action: pos + rot + gripper)
    sawcut=True → 절삭 중 x축 방향 진동 추가
    """
    start_p = np.array(start_pose.p, dtype=np.float32).reshape(3)
    end_p = np.array(target_pose.p, dtype=np.float32).reshape(3)
    fixed_rot = np.array([np.pi, 0.0, 0.0], dtype=np.float32)  # 칼날이 아래로

    for i in range(steps):
        alpha = i / (steps - 1)
        pos = (1 - alpha) * start_p + alpha * end_p

        if sawcut:
            # ✅ x축만 sin 진동으로 보정
            pos[0] += amplitude * np.sin(2 * np.pi * frequency * alpha)

        action = np.concatenate([pos, fixed_rot, np.array([gripper])], axis=0).astype(np.float32)
        env.step(action)

    return


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
    # 환경 생성 (headless)
    # -----------------------------
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_ee_pose",  # ✅ base frame 기준
        render_mode="rgb_array",
        reward_mode="none",
        enable_shadow=False,
        robot_uids=args.robot_uid,
    )

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
        source_desc="auto cutting linear pose trajectory",
    )

    # -----------------------------
    # 환경 초기화 및 EEF 안정화
    # -----------------------------
    obs, _ = env.reset(seed=seed, options=dict(save_trajectory=True))

    # 로봇 자세를 rest pose로 강제 초기화
    if hasattr(env.agent, "keyframes") and "rest" in env.agent.keyframes:
        env.agent.robot.set_qpos(env.agent.keyframes["rest"].qpos)
        env.agent.robot.set_qvel(np.zeros_like(env.agent.robot.get_qvel()))
        env.step(np.zeros(7, dtype=np.float32))  # 한 프레임 물리 업데이트
        env.render()

    # ✅ 현재 EEF pose (world 기준)
    init_pose_world = env.agent.tcp.pose
    print(f"[INIT(world)] EEF pose: {init_pose_world.p}")

    # ✅ base_link pose 가져와 변환 (world → base frame)
    base_pose = env.agent.robot.get_links()[0].pose  # root_link 기준
    init_pose_base = base_pose.inv() * init_pose_world
    print(f"[INIT(base)] EEF pose(base frame): {init_pose_base.p}")

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
    #approach_randx = random.uniform(-0.1, 0.35)   # 접근 시 X 오프셋 (-0.1~0.2)
    #randz = random.uniform(0.3, 0.5)             # 접근 높이 (0.3~0.5)
    approach_randx =-0.2
    randz = 0.25
    retreat_randx = random.uniform(-0.25, -0.15) # 복귀 시 X 오프셋
    randomspeed =200
    #randomspeed = random.randint(10, 100)        # 절삭 단계 속도 (40~200 스텝)
    approach_pose = sapien.Pose(p=obj_pos + np.array([approach_randx, 0.0, randz], dtype=np.float32))
    cut_pose = sapien.Pose(p=obj_pos + np.array([approach_randx, 0.0, 0.15], dtype=np.float32))
    retreat_pose = sapien.Pose(p=obj_pos + np.array([approach_randx, 0.0,randz ], dtype=np.float32))

    # ✅ world → base 변환
    approach_pose_base = base_pose.inv() * approach_pose
    cut_pose_base = base_pose.inv() * cut_pose
    retreat_pose_base = base_pose.inv() * retreat_pose

    # -----------------------------
    # 자동 절삭 시퀀스
    # -----------------------------
# -----------------------------
# 자동 절삭 시퀀스
# -----------------------------
    print("[Stage 1] Approach")
    move_to_pose(
        env,
        init_pose_base,
        approach_pose_base,
        steps=80,
        gripper=CLOSED,
        sawcut=False,        # 직선 접근
    )
    val = random.uniform(0.01, 0.04)
    print("[Stage 2] Saw-Cut Down")
    move_to_pose(
        env,
        approach_pose_base,
        cut_pose_base,
        steps=randomspeed,
        gripper=CLOSED,
        sawcut=True,         # ✅ 진동 활성화
        amplitude=val,      # 진폭 (m) — 좌우 흔들림 폭
        frequency=20,       # 주파수 — 왕복 횟수
    )

    print("[Stage 3] Retreat")
    move_to_pose(
        env,
        cut_pose_base,
        retreat_pose_base,
        steps=80,
        gripper=CLOSED,
        sawcut=False,        # 복귀는 직선
    )

    print("[DONE] Cutting trajectory executed successfully.")

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
        print("[INFO] Video saving handled by RecordEpisode.")
    print(f"[INIT(base)] Final EEF base pose: {init_pose_base.p}")


if __name__ == "__main__":
    main(parse_args())

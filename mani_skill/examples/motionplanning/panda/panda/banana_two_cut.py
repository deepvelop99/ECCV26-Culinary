import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from typing import Annotated
import tyro
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers.recordori import RecordEpisode
import os, random
import torch

OPEN = 1.0
CLOSED = -1.0


# ================================================================
# Argument 정의
# ================================================================
@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "banana2cut"
    obs_mode: str = "state_dict"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "pkour"
    record_dir: str = "testtesttesttesttesttest"
    save_video: bool = True
    vis: bool = False

def parse_args() -> Args:
    return tyro.cli(Args)

def move_to_pose(env: BaseEnv, start_pose: sapien.Pose, target_pose: sapien.Pose, steps=100, gripper=CLOSED):
    """EEF 선형 이동 (7D action: pos + rot + gripper)"""
    start_p = np.array(start_pose.p, dtype=np.float32).reshape(3)
    end_p = np.array(target_pose.p, dtype=np.float32).reshape(3)

    fixed_rot = np.array([np.pi, 0.0, 0.0], dtype=np.float32)

    for i in range(steps):
        alpha = i / (steps - 1)
        pos = (1 - alpha) * start_p + alpha * end_p
        action = np.concatenate([pos, fixed_rot, np.array([gripper])], axis=0).astype(np.float32)
        env.step(action)
    return


# ================================================================
# Main 실행
# ================================================================
def main(args: Args):
    seed = random.randint(0, 4096)
    output_dir = f"{args.record_dir}/{args.env_id}/auto_{seed}/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Trajectory will be saved to: {output_dir}")

    # -----------------------------
    # 환경 생성 (headless)
    # -----------------------------
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_ee_pose",  # ✅ base frame 기준
        render_mode="human" if args.vis else "rgb_array",  # ✅ 시각화 모드 적용
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

    obs, _ = env.reset(seed=seed, options=dict(save_trajectory=True))
    if hasattr(env.agent, "keyframes") and "rest" in env.agent.keyframes:
        env.agent.robot.set_qpos(env.agent.keyframes["rest"].qpos)
        env.agent.robot.set_qvel(np.zeros_like(env.agent.robot.get_qvel()))
        env.step(np.zeros(7, dtype=np.float32))
        env.render()

        if hasattr(env.agent, "controller"):
            env.agent.controller._target_pose = env.agent.tcp.pose

    init_pose_world = env.agent.tcp.pose
    base_pose = env.agent.robot.get_links()[0].pose  # root_link 기준
    init_pose_base = base_pose.inv() * init_pose_world

    obj_actor = getattr(env, "block_apple", [None])[0]
    assert obj_actor is not None, "환경에 object가 없습니다."
    
    tip_p = env.unwrapped._get_tip_from_eef()
    
    aabb_min, aabb_max = env.unwrapped._get_actor_aabb(obj_actor)
    aabb_min = torch.tensor(aabb_min, dtype=torch.float32, device=tip_p.device)
    aabb_max = torch.tensor(aabb_max, dtype=torch.float32, device=tip_p.device)
    object_width = float((aabb_max[2] - aabb_min[2]).item())

    obj_pose = obj_actor.pose
    pose_p = np.array(obj_pose.p, dtype=float).reshape(-1)
    object_center_y, object_center_z = pose_p[1], pose_p[2]

    # === y축 절삭 오프셋 자동 계산 ===
    num_cuts = 3
    object_width = float((aabb_max[2] - aabb_min[2]).item())
    dy = object_width / 4.0

    y_offsets = [-dy + 0.01, dy + 0.01]  # 상대 오프셋들
    
    # rand_x_variation = random.uniform(-0.35, -0.1)   # 접근 시 X 오프셋 (-0.1~0.2)
    # rand_approach_z = random.uniform(0.3, 0.5)             # 접근 높이 (0.3~0.5)
    # randomspeed = random.randint(10, 100)
    rand_x_variation = -0.25   # 접근 X 오프셋
    rand_approach_z = 0.5              # 접근 높이
    randomspeed = 30          # 절삭 단계 스텝 수
    for i, y_off in enumerate(y_offsets):
        target_y = object_center_y + y_off
        print(f"[Cutting {i+1}] Target y={target_y:.3f}")

        approach_pose = sapien.Pose(p=pose_p + np.array([rand_x_variation, y_off, rand_approach_z], dtype=np.float32))
        cut_pose      = sapien.Pose(p=pose_p + np.array([rand_x_variation, y_off, 0.15], dtype=np.float32))
        retreat_pose  = sapien.Pose(p=pose_p + np.array([rand_x_variation, y_off, rand_approach_z], dtype=np.float32))

        approach_pose_base = base_pose.inv() * approach_pose
        cut_pose_base = base_pose.inv() * cut_pose
        retreat_pose_base = base_pose.inv() * retreat_pose

        move_to_pose(env, init_pose_base, approach_pose_base, steps=100, gripper=CLOSED)
        move_to_pose(env, approach_pose_base, cut_pose_base, steps=30, gripper=CLOSED)
        move_to_pose(env, cut_pose_base, retreat_pose_base, steps=100, gripper=CLOSED)
        init_pose_base = retreat_pose_base

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

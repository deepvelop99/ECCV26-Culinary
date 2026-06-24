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
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "bananacut"
    obs_mode: str = "state_dict"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "pkour"
    record_dir: str = "demotmp"
    save_video: bool = True
    vis: bool = False

def parse_args() -> Args:
    return tyro.cli(Args)

def move_to_pose(env: BaseEnv, start_pose_w: sapien.Pose, target_pose_w: sapien.Pose,
                 steps=100, kp=7.0, clip=0.03, gripper_delta=0.0):
    """pd_ee_delta_pos 모드용: base frame 기준으로 실제 z축 하강"""
    base_pose = env.agent.robot.get_links()[0].pose  # root_link 기준

    start_p_base = (base_pose.inv() * start_pose_w).p
    target_p_base = (base_pose.inv() * target_pose_w).p

    start_p_base = np.array(start_p_base, dtype=np.float32).reshape(3)
    target_p_base = np.array(target_p_base, dtype=np.float32).reshape(3)

    for i in range(steps):
        # 현재 EEF 위치 (base frame)
        ee_base = (base_pose.inv() * env.agent.tcp.pose).p
        ee_base = np.array(ee_base, dtype=np.float32).reshape(3)

        # 선형 보간된 목표 위치 (base frame)
        alpha = i / max(steps - 1, 1)
        interp_target = (1 - alpha) * start_p_base + alpha * target_p_base

        # delta 계산
        dpos = (interp_target - ee_base) * kp
        dpos = np.clip(dpos, -clip, clip).astype(np.float32)

        # gripper 포함 여부 자동 감지
        action_dim = int(np.prod(env.action_space.shape))
        if action_dim >= 4:
            action = np.concatenate([dpos, np.array([gripper_delta], dtype=np.float32)], axis=0)
        else:
            action = dpos

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
        control_mode="pd_ee_delta_pos",  # ✅ base frame 기준
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
    
    if hasattr(env.agent, "controller_configs") and "arm" in env.agent.controller_configs:
        cfg = env.agent.controller_configs["arm"]
        if hasattr(cfg, "use_delta"):
            cfg.use_delta = True                   # 델타 모드 보장
        if hasattr(cfg, "frame"):
            cfg.frame = "root_translation"         # 위치는 루트 프레임
        # 정규화 끄고 SI 단위로 쓰고 싶다면:
        if hasattr(cfg, "normalize_action"):
            cfg.normalize_action = False           # 가능하면 SI 단위 직접 사용

    obs, _ = env.reset(seed=seed, options=dict(save_trajectory=True))
    if hasattr(env.agent, "keyframes") and "rest" in env.agent.keyframes:
        env.agent.robot.set_qpos(env.agent.keyframes["rest"].qpos)
        env.agent.robot.set_qvel(np.zeros_like(env.agent.robot.get_qvel()))
        action_dim = int(np.prod(env.action_space.shape))
        env.step(np.zeros(action_dim, dtype=np.float32))
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

    y_offsets = [-dy + 0.0265, 0, dy - 0.016]  # 상대 오프셋들

    rand_x_variation = -0.2
    # rand_x_variation = random.uniform(-0.35, -0.1)   # 접근 시 X 오프셋 (-0.1~0.2)
    rand_approach_z = random.uniform(0.3, 0.5)             # 접근 높이 (0.3~0.5)
    randomspeed = random.randint(10, 100)
    
    for i, y_off in enumerate(y_offsets):
        target_y = object_center_y + y_off
        print(f"[Cutting {i+1}] Target y={target_y:.3f}")

        approach_pose = sapien.Pose(p=pose_p + np.array([rand_x_variation, y_off, rand_approach_z], dtype=np.float32))
        cut_pose      = sapien.Pose(p=pose_p + np.array([rand_x_variation, y_off, 0.11], dtype=np.float32))
        retreat_pose  = sapien.Pose(p=pose_p + np.array([rand_x_variation, y_off, rand_approach_z], dtype=np.float32))

        approach_pose_base = base_pose.inv() * approach_pose
        cut_pose_base = base_pose.inv() * cut_pose
        retreat_pose_base = base_pose.inv() * retreat_pose

        # move_to_pose(env, init_pose_base, approach_pose_base, steps=50, gripper=CLOSED)
        # move_to_pose(env, approach_pose_base, cut_pose_base, steps=50, gripper=CLOSED)
        # move_to_pose(env, cut_pose_base, retreat_pose_base, steps=50, gripper=CLOSED)
        move_to_pose(env, init_pose_world, approach_pose, steps=100, kp=10.0, clip=0.1, gripper_delta=0.0)
        move_to_pose(env, approach_pose,    cut_pose,    steps=randomspeed, kp=25.0, clip=0.1, gripper_delta=0.0)
        move_to_pose(env, cut_pose,         retreat_pose,steps=100, kp=10.0, clip=0.1, gripper_delta=0.0)

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

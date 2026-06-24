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
    record_dir: str = "sawcutdelta"
    save_video: bool = True

def parse_args() -> Args:
    return tyro.cli(Args)

# ================================================================
# Helper: 델타-포지션 제어용 선형 이동(+옵션: saw-cut)
# ================================================================
def move_to_pose(
    env: BaseEnv,
    start_pose_w: sapien.Pose,
    target_pose_w: sapien.Pose,
    steps=100,
    kp=7.0,
    clip=0.03,
    sawcut=False,
    amplitude=0.02,    # x축 고정 델타 (m)
    frequency=20.0,    # 전체 구간에서 진동 주기(사인 파형의 주기 수)
    gripper_delta=0.0,
    preserve_z_error=True,  # True면 z는 목표로 계속 내려감, False면 z도 0
):
    """
    control_mode='pd_ee_delta_pos' 전제.
    sawcut=True이면 x축 델타를 매 step '고정 진폭'으로 강제로 넣는다(사각/사인 선택 가능).
    y=0, z는 preserve_z_error=True면 정상 오차 제어, False면 0.
    """
    start_p = np.asarray(start_pose_w.p, dtype=np.float32).reshape(-1)
    end_p   = np.asarray(target_pose_w.p, dtype=np.float32).reshape(-1)

    action_dim = int(np.prod(env.action_space.shape))
    has_gripper = action_dim >= 4

    for i in range(steps):
        alpha = i / max(steps - 1, 1)

        # 월드 목표 위치(선형 보간)
        target_p_w = (1 - alpha) * start_p + alpha * end_p

        # 루트 프레임으로 변환
        base_pose = env.agent.robot.get_links()[0].pose
        ee_base_p     = (base_pose.inv() * env.agent.tcp.pose).p
        target_base_p = (base_pose.inv() * sapien.Pose(p=target_p_w)).p
        ee_base_p     = np.asarray(ee_base_p,     dtype=np.float32).reshape(-1)
        target_base_p = np.asarray(target_base_p, dtype=np.float32).reshape(-1)

        # 기본 dpos: 오차 기반
        dpos = (target_base_p - ee_base_p) * kp

        if sawcut:
            # 1) x축은 '고정 진폭'으로 강제 (사인파의 부호만 사용 → 크기는 항상 amplitude)
            sign = 1.0 if np.sin(2 * np.pi * frequency * alpha) >= 0.0 else -1.0
            dpos[0] = sign * amplitude

            # 2) y축은 0으로 고정
            dpos[1] = 0.0

            # 3) z축은 선택: 목표로 내려가게 유지 or 0으로 고정
            if not preserve_z_error:
                dpos[2] = 0.0

        # 포화
        dpos = np.clip(dpos, -clip, clip).astype(np.float32)

        # 액션 구성
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

    # (권장) 컨트롤러 설정 강제 — 가능한 경우에만
    if hasattr(env.agent, "controller_configs") and "arm" in env.agent.controller_configs:
        cfg = env.agent.controller_configs["arm"]
        if hasattr(cfg, "use_delta"):
            cfg.use_delta = True                   # 델타 모드 확실히
        if hasattr(cfg, "frame"):
            cfg.frame = "root_translation"         # Δ는 root 프레임 기준
        if hasattr(cfg, "normalize_action"):
            # 가능하면 SI 단위(m)를 그대로 쓰기 위해 정규화 해제
            cfg.normalize_action = False

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
        source_desc="auto cutting (delta-pos with saw oscillation)",
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

    # ✅ 현재 EEF pose (world)
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
    # saw-cut을 위해 x를 살짝 뒤에 배치하고 아래로 내렸다가 복귀
    approach_randx = -0.2
    randz = 0.25
    randomspeed = 200  # 절삭 구간 step

    approach_pose = sapien.Pose(p=obj_pos + np.array([approach_randx, 0.0, randz], dtype=np.float32))
    cut_pose      = sapien.Pose(p=obj_pos + np.array([approach_randx, 0.0, 0.15], dtype=np.float32))
    retreat_pose  = sapien.Pose(p=obj_pos + np.array([approach_randx, 0.0, randz], dtype=np.float32))

    # -----------------------------
    # 자동 절삭 시퀀스 (월드 타깃 → root Δ만 전달)
    # -----------------------------
    print("[Stage 1] Approach")
    move_to_pose(
        env,
        init_pose_world,
        approach_pose,
        steps=80,
        kp=7.0,
        clip=0.03,
        sawcut=False,   # 직선 접근
        gripper_delta=0.0,
    )

    print("[Stage 2] Saw-Cut Down")
    #val = random.uniform(0.0, 0.04)  # 진동 진폭
    val =0.2
    move_to_pose(
        env,
        approach_pose,
        cut_pose,
        steps=randomspeed,
        kp=7.0,
        clip=1,
        sawcut=True,           # ✅ 진동 활성화
        amplitude=val,         # 진폭 (m)
        frequency=50,          # 주파수 (사인)
        gripper_delta=0.0,
    )

    print("[Stage 3] Retreat")
    move_to_pose(
        env,
        cut_pose,
        retreat_pose,
        steps=80,
        kp=7.0,
        clip=0.03,
        sawcut=False,          # 복귀는 직선
        gripper_delta=0.0,
    )

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

"""Collect bananacut rollouts in both control modes.

For each seed, rolls out the scripted motion-planning cutting trajectory once
per control_mode and saves to the common intermediate Episode format.

Usage:
    PYTHONPATH=/data LD_LIBRARY_PATH=/workspace/envs/maniskill/lib:$LD_LIBRARY_PATH \
      /workspace/envs/maniskill/bin/python dataset_converters/collect_rollouts.py \
      --task bananacut --num-episodes 50 --out /data/datasets/maniskill_intermediate
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import sapien.core as sapien

import mani_skill  # noqa: F401  # registers envs
from mani_skill.envs.sapien_env import BaseEnv

from dataset_converters.common import (
    Episode,
    CONTROL_MODE_JOINT,
    CONTROL_MODE_EE,
    episode_path,
)


INSTRUCTIONS_PATH = Path(__file__).parent / "common" / "instructions.json"


def load_instruction(task: str) -> str:
    with open(INSTRUCTIONS_PATH) as f:
        return json.load(f)[task]


def _move_ee(env, start_w: sapien.Pose, target_w: sapien.Pose, *, steps: int,
             kp: float, clip: float, control_mode: str, gripper: float,
             record):
    """Drive EE linearly from start_w -> target_w in world frame.
    Works for both pd_ee_delta_pos / pd_ee_delta_pose (we keep rot delta = 0)
    and pd_joint_pos (we IK-ish through controller's internal solver by
    feeding the env's built-in pose targets)."""
    base_pose = env.agent.robot.get_links()[0].pose
    start_p = np.asarray((base_pose.inv() * start_w).p, dtype=np.float32).reshape(3)
    target_p = np.asarray((base_pose.inv() * target_w).p, dtype=np.float32).reshape(3)

    for i in range(steps):
        ee_p = np.asarray((base_pose.inv() * env.agent.tcp.pose).p,
                          dtype=np.float32).reshape(3)
        alpha = i / max(steps - 1, 1)
        interp = (1 - alpha) * start_p + alpha * target_p
        dpos = np.clip((interp - ee_p) * kp, -clip, clip).astype(np.float32)

        if control_mode == CONTROL_MODE_EE:
            # [dx, dy, dz, drx, dry, drz, gripper]
            action = np.concatenate([dpos, np.zeros(3, np.float32),
                                     np.array([gripper], np.float32)])
        elif control_mode == CONTROL_MODE_JOINT:
            # For joint-pos control, we cannot directly command EE delta.
            # We use the current qpos as the target plus a small joint delta
            # from the Jacobian pseudoinverse so the script is consistent
            # across both modes without needing an external IK.
            # Simpler alternative taken here: snap target to current + 0
            # so the robot holds, and rely on EE-mode collection for the
            # cutting motion. The joint-mode collection path uses
            # `_replay_joint_from_ee` instead (see main()).
            raise RuntimeError("_move_ee should not be called in joint mode; "
                               "use _replay_joint_from_ee.")
        else:
            raise ValueError(control_mode)

        record(action)
        env.step(action)


def _rollout_ee(env, obj_pos: np.ndarray, *, init_pose_w: sapien.Pose,
                approach_x: float, approach_z: float, cut_speed: int,
                control_mode: str, record):
    approach = sapien.Pose(p=obj_pos + np.array([approach_x, 0.0, approach_z], np.float32))
    cut = sapien.Pose(p=obj_pos + np.array([approach_x, 0.0, 0.025], np.float32))
    retreat = sapien.Pose(p=obj_pos + np.array([approach_x, 0.0, approach_z], np.float32))

    _move_ee(env, init_pose_w, approach, steps=100, kp=7.0, clip=0.03,
             control_mode=control_mode, gripper=0.0, record=record)
    _move_ee(env, approach, cut, steps=cut_speed, kp=8.0, clip=0.1,
             control_mode=control_mode, gripper=0.0, record=record)
    _move_ee(env, cut, retreat, steps=100, kp=10.0, clip=0.1,
             control_mode=control_mode, gripper=0.0, record=record)


def _rollout_joint_replay(env, qpos_ref: np.ndarray, *, record):
    """Replay a reference qpos trajectory under pd_joint_pos.

    qpos_ref has shape (T, n_q). We feed the arm qpos (first 7) + gripper
    action (from the reference, here approximated as open=1.0) as the action.
    """
    T = qpos_ref.shape[0]
    for t in range(T):
        arm = qpos_ref[t, :7].astype(np.float32)
        gripper = np.array([1.0], np.float32)  # bananacut keeps gripper open
        action = np.concatenate([arm, gripper], axis=0)
        record(action)
        env.step(action)


def collect_one(task: str, seed: int, control_mode: str, out_root: Path,
                idx: int, *, obs_mode="rgb", image_size=224) -> Optional[Episode]:
    instruction = load_instruction(task)

    env = gym.make(
        task,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode="rgb_array",
        reward_mode="none",
        enable_shadow=False,
        robot_uids="pkour",
        sensor_configs=dict(width=image_size, height=image_size),
    )
    obs, info = env.reset(seed=seed, options=dict(save_trajectory=False))
    u: BaseEnv = env.unwrapped

    # Hold rest pose
    if hasattr(u.agent, "keyframes") and "rest" in u.agent.keyframes:
        u.agent.robot.set_qpos(u.agent.keyframes["rest"].qpos)
        u.agent.robot.set_qvel(np.zeros_like(u.agent.robot.get_qvel()))

    init_pose_w = u.agent.tcp.pose
    obj_actor = getattr(u, "block_apple", [None])[0]
    if obj_actor is None:
        env.close()
        raise RuntimeError(f"{task} has no `block_apple` actor; adjust for this task")
    obj_pos = np.asarray(obj_actor.pose.p, dtype=np.float32).reshape(-1)

    rgb_buf, qpos_buf, tcp_buf, action_buf, term_buf = [], [], [], [], []

    def snapshot_obs():
        # Primary camera rgb
        cam = u.get_obs()["sensor_data"]
        cam_name = next(iter(cam.keys()))
        img = np.asarray(cam[cam_name]["rgb"])
        if img.ndim == 4:
            img = img[0]
        rgb_buf.append(img.astype(np.uint8))
        qpos_buf.append(np.asarray(u.agent.robot.get_qpos(), np.float32).reshape(-1))
        p = np.asarray(u.agent.tcp.pose.p, np.float32).reshape(3)
        q = np.asarray(u.agent.tcp.pose.q, np.float32).reshape(4)
        tcp_buf.append(np.concatenate([p, q]))

    def record(action):
        snapshot_obs()
        action_buf.append(np.asarray(action, np.float32).reshape(-1))
        term_buf.append(False)

    # --- actual rollout ---
    if control_mode == CONTROL_MODE_EE:
        _rollout_ee(env, obj_pos, init_pose_w=init_pose_w,
                    approach_x=-0.25, approach_z=0.5, cut_speed=30,
                    control_mode=control_mode, record=record)
    elif control_mode == CONTROL_MODE_JOINT:
        # For joint-pos collection, we first run an EE rollout in a side env
        # to get a qpos reference, then replay on the real env in joint mode.
        ee_env = gym.make(
            task, obs_mode="state", control_mode=CONTROL_MODE_EE,
            render_mode="rgb_array", reward_mode="none",
            enable_shadow=False, robot_uids="pkour",
        )
        ee_env.reset(seed=seed, options=dict(save_trajectory=False))
        ee_u: BaseEnv = ee_env.unwrapped
        if hasattr(ee_u.agent, "keyframes") and "rest" in ee_u.agent.keyframes:
            ee_u.agent.robot.set_qpos(ee_u.agent.keyframes["rest"].qpos)
        ee_init = ee_u.agent.tcp.pose
        ee_obj = np.asarray(getattr(ee_u, "block_apple", [None])[0].pose.p,
                            np.float32).reshape(-1)
        qpos_ref = []

        def rec_ee(_action):
            qpos_ref.append(np.asarray(ee_u.agent.robot.get_qpos(),
                                       np.float32).reshape(-1))

        _rollout_ee(ee_env, ee_obj, init_pose_w=ee_init,
                    approach_x=-0.25, approach_z=0.5, cut_speed=30,
                    control_mode=CONTROL_MODE_EE, record=rec_ee)
        ee_env.close()
        qpos_ref = np.stack(qpos_ref, axis=0)
        _rollout_joint_replay(env, qpos_ref, record=record)
    else:
        raise ValueError(control_mode)

    term_buf[-1] = True

    success = bool(info.get("success", [False])[0]) if isinstance(
        info.get("success", False), (list, np.ndarray)) else bool(info.get("success", False))

    ep = Episode(
        task=task,
        instruction=instruction,
        control_mode=control_mode,
        control_freq=int(getattr(u, "control_freq", 20)),
        sim_freq=int(getattr(u, "sim_freq", 100)),
        seed=seed,
        success=success,
        rgb=np.stack(rgb_buf, axis=0),
        wrist_rgb=None,
        qpos=np.stack(qpos_buf, axis=0),
        tcp_pose=np.stack(tcp_buf, axis=0),
        action=np.stack(action_buf, axis=0),
        is_terminal=np.asarray(term_buf, dtype=bool),
    )
    env.close()
    ep.save(episode_path(out_root, task, control_mode, idx))
    return ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="bananacut")
    ap.add_argument("--num-episodes", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--modes", nargs="+",
                    default=[CONTROL_MODE_EE, CONTROL_MODE_JOINT])
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    for i in range(args.num_episodes):
        seed = args.seed_base + i
        for mode in args.modes:
            print(f"[collect] task={args.task} mode={mode} seed={seed} idx={i}")
            collect_one(args.task, seed, mode, args.out, i,
                        image_size=args.image_size)
    print("[done]")


if __name__ == "__main__":
    main()

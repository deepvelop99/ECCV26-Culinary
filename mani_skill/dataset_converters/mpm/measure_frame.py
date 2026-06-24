"""One-shot frame-calibration measurement script.

Run on a GPU node (CUDA + Vulkan). Prints and saves the following so we can
lock in the FrameCalibration constants for the real collect run:

    [ManiSkill]
      board top z (world) from env
      block_apple initial pose (p, q) per env
      knife link pose at reset
      knife tip world xyz (env._get_tip_from_eef)
      tcp pose
      agent.tcp.linear_velocity at rest
    [MPM]
      sim.board_top_y[None]  (runtime, after board SDF init)
      sim._knife_yfoot       (runtime, from knife SDF)
      KnifeCollider world_low_y()   (tip world y at reset)
      sim.dx, sim.dt, sim.substeps  (grid/time)
      bounds_min/max (from yaml)
    [cross-check]
      predicted MPM tip y from (tip z_ms + V_OFFSET - knife_yfoot)
         vs actual sim.knife.y[None] after reset
      tip_ms -> tip_mpm via ms_knife_tip_to_mpm, compare to MPM canonical
    [dynamic]
      A small scripted Z-descent (10 cm in 0.5s under pd_ee_delta_pose).
      Every control step log:
        tip_z_ms, tip_y_mpm_predicted, knife_net_contact_force (z_ms),
        mpm eef force (if available), tcp_vel (z_ms), mpm knife_speed.

Usage (GPU node):
    export LD_LIBRARY_PATH=/workspace/envs/maniskill/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/data
    /workspace/envs/maniskill/bin/python dataset_converters/mpm/measure_frame.py \
        --mpm-config /data/EEF-Cutting-Simulation/configs/banana.yaml \
        --out /data/datasets/frame_calibration/run0.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _np(x):
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, np.float32).tolist()


def _read_ms_static(env) -> dict:
    u = env.unwrapped
    out = {}

    # Board
    board = getattr(u, "board", None)
    if board is not None:
        out["board_pose_p"] = _np(board.pose.p)
        out["board_pose_q"] = _np(board.pose.q)
    out["BOARD_SIZE_half"] = list(map(float, u.BOARD_SIZE.tolist()))
    out["ms_board_top_z_predicted"] = 2.0 * float(u.BOARD_SIZE[2])

    # Block (banana proxy)
    block_apple = getattr(u, "block_apple", [])
    if len(block_apple) > 0:
        b = block_apple[0]
        out["block_pose_p"] = _np(b.pose.p)
        out["block_pose_q"] = _np(b.pose.q)

    # Agent / knife link / tcp
    try:
        out["tcp_pose_p"] = _np(u.agent.tcp.pose.p)
        out["tcp_pose_q"] = _np(u.agent.tcp.pose.q)
    except Exception as e:
        out["tcp_read_err"] = str(e)
    try:
        out["knife_link_pose_p"] = _np(u.knife_link.pose.p)
        out["knife_link_pose_q"] = _np(u.knife_link.pose.q)
    except Exception as e:
        out["knife_link_err"] = str(e)
    try:
        out["knife_tip_ms"] = _np(u._get_tip_from_eef()[0])
    except Exception as e:
        out["knife_tip_err"] = str(e)
    return out


def _read_mpm_static(bridge, yaml_cfg) -> dict:
    sim = bridge.sim
    out = {
        "yaml_bounds_min": list(map(float, yaml_cfg["world"]["bounds_min"])),
        "yaml_bounds_max": list(map(float, yaml_cfg["world"]["bounds_max"])),
        "yaml_grid_resolution": int(yaml_cfg["world"]["grid_resolution"]),
        "yaml_dt": float(yaml_cfg["world"]["dt"]),
        "yaml_substeps": int(yaml_cfg["world"]["substeps"]),
        "yaml_knife_start_y": float(yaml_cfg["knife"]["motion"]["knife_start_y"]),
        "yaml_knife_stop_y": float(yaml_cfg["knife"]["motion"]["knife_stop_y"]),
    }
    try:
        out["sim_dx"] = float(sim.dx_s[None])
    except Exception:
        pass
    try:
        out["sim_dt"] = float(sim.dt)
        out["sim_substeps"] = int(sim.substeps)
    except Exception:
        pass
    try:
        out["sim_board_top_y"] = float(sim.board_top_y[None])
    except Exception:
        pass
    try:
        out["sim_knife_yfoot"] = float(sim._knife_yfoot)
    except Exception:
        pass
    try:
        out["sim_knife_y_init"] = float(sim.knife.y[None])
        out["sim_knife_world_low_y_init"] = float(sim.knife.world_low_y())
    except Exception as e:
        out["sim_knife_read_err"] = str(e)
    # Banana particle AABB in MPM world frame (post-SDF sampling)
    try:
        x = sim.particles.x.to_numpy()
        lb = x.min(axis=0).tolist()
        ub = x.max(axis=0).tolist()
        mid = [(lb[i] + ub[i]) / 2 for i in range(3)]
        out["sim_particles_aabb_lb"] = list(map(float, lb))
        out["sim_particles_aabb_ub"] = list(map(float, ub))
        out["sim_particles_centroid"] = list(map(float, mid))
        out["sim_particles_n"] = int(x.shape[0])
    except Exception as e:
        out["sim_particles_err"] = str(e)
    try:
        out["cutting_pack_origin"] = _np(sim.cfg["cutting_mesh"].get("initial_transform", {}).get("translate", [0, 0, 0]))
    except Exception:
        pass
    return out


def _cross_check(ms: dict, mpm: dict, calib) -> dict:
    """Predict MPM tip y from MS tip z, compare with actual MPM init."""
    out = {}
    if "knife_tip_ms" in ms and "sim_knife_yfoot" in mpm:
        tip_z = ms["knife_tip_ms"][2]
        v_off = calib["v_offset"]
        pred_y = tip_z + v_off
        out["predicted_tip_y_mpm"] = float(pred_y)
        out["knife_y_None_target"] = float(pred_y - mpm["sim_knife_yfoot"])
    out["delta_board_top"] = {
        "ms": ms.get("ms_board_top_z_predicted"),
        "mpm": mpm.get("sim_board_top_y"),
        "diff_vs_calib_v_offset":
            (mpm.get("sim_board_top_y", 0.0) - ms.get("ms_board_top_z_predicted", 0.0))
            if (mpm.get("sim_board_top_y") is not None and ms.get("ms_board_top_z_predicted") is not None) else None,
        "calib_v_offset": calib["v_offset"],
    }
    return out


def _scripted_descent(env, bridge, *, steps=30, descent_m=0.10, control_dt=0.05) -> list:
    """Nudge arm down along z by directly stepping qpos targets (pd_joint_pos).

    We compute the full arm qpos target each step by biasing joint 3 and 5
    slightly to lower the TCP; this is a pragmatic hack for a calibration
    run (not a real cut trajectory). The goal is only to produce per-step
    knife Z motion so we can compare MS tip_z vs MPM tip_y over time.
    """
    import numpy as np
    u = env.unwrapped
    log = []
    try:
        rest = u.agent.keyframes["rest"].qpos
        if hasattr(rest, "cpu"):
            rest = rest.cpu().numpy()
        qpos0 = np.asarray(rest, np.float32).reshape(-1)
    except Exception:
        qpos0 = np.asarray(u.agent.robot.get_qpos().cpu().numpy(), np.float32).reshape(-1)

    action_dim = int(np.prod(env.action_space.shape))
    # For pd_joint_pos the action is the target joint positions (arm + gripper),
    # typically arm joints are the first 7 dims with gripper as last.
    arm_dim = min(7, action_dim - 1) if action_dim >= 8 else min(7, action_dim)

    for t in range(steps):
        alpha = (t + 1) / steps
        # bias j3 (elbow) downward and j5 (wrist) upward for a net TCP descent
        bias = np.zeros(action_dim, np.float32)
        if arm_dim >= 6:
            bias[3] = +0.3 * alpha  # elbow flex (depends on sign convention)
            bias[5] = -0.3 * alpha
        target = qpos0[:action_dim].astype(np.float32) + bias
        obs, r, term, trunc, info = env.step(target)
        try:
            tele = bridge.step_with_env(env, dt_control=control_dt)
        except Exception as e:
            log.append({"t": t, "bridge_err": str(e)})
            continue
        rec = {
            "t": t,
            "tip_ms": _np(tele["ms_tip_pos"]),
            "tip_mpm": _np(tele["mpm_tip_pos"]),
            "tcp_vel_ms": _np(tele["ms_tcp_vel"]),
            "force_ms": _np(tele["ms_force"]),
            "force_mpm": _np(tele["mpm_force"]),
            "vel_mpm": _np(tele["mpm_vel"]),
        }
        log.append(rec)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpm-config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", default="bananacut")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--descent-steps", type=int, default=30)
    ap.add_argument("--descent-m", type=float, default=0.10)
    ap.add_argument("--skip-descent", action="store_true")
    ap.add_argument("--control-mode", default="pd_joint_pos")
    ap.add_argument("--robot-uid", default="pkv")
    args = ap.parse_args()

    import gymnasium as gym
    import mani_skill  # noqa: F401
    from dataset_converters.mpm import (
        ManiSkillMPMBridge, BridgeConfig, FrameCalibration,
    )
    import yaml

    calib = FrameCalibration()
    env = gym.make(
        args.task,
        obs_mode="none",
        control_mode=args.control_mode,
        render_mode="rgb_array",
        reward_mode="none",
        enable_shadow=False,
        robot_uids=args.robot_uid,
    )
    env.reset(seed=args.seed)

    ms_static = _read_ms_static(env)
    print("[ManiSkill static]", json.dumps(ms_static, indent=2))

    # Build bridge with MPM
    cfg = BridgeConfig(mpm_config_yaml=str(args.mpm_config), calibration=calib)
    bridge = ManiSkillMPMBridge(cfg)
    block_p = np.asarray(ms_static.get("block_pose_p", [[0, 0, 0]]), np.float32).reshape(-1)
    banana_xy = block_p[:2]
    bridge.setup(env, banana_xy_ms=banana_xy, banana_yaw_ms=0.0)

    with open(args.mpm_config) as f:
        yaml_cfg = yaml.safe_load(f)
    mpm_static = _read_mpm_static(bridge, yaml_cfg)
    print("[MPM static]", json.dumps(mpm_static, indent=2))

    cross = _cross_check(ms_static, mpm_static, calib.to_dict())
    print("[cross-check]", json.dumps(cross, indent=2))

    descent = None
    if not args.skip_descent:
        try:
            descent = _scripted_descent(
                env, bridge,
                steps=args.descent_steps,
                descent_m=args.descent_m,
                control_dt=1.0 / 20.0,
            )
            print(f"[descent] {len(descent)} steps logged")
        except Exception as e:
            print(f"[descent] failed: {e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "calib_defaults": calib.to_dict(),
        "ms_static": ms_static,
        "mpm_static": mpm_static,
        "cross_check": cross,
        "descent_log": descent,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[saved] {args.out}")

    bridge.close()
    env.close()


if __name__ == "__main__":
    main()

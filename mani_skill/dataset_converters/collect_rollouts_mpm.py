"""ManiSkill + MPM co-sim rollout collection — native-h5 edition.

Redesigned to:
  • Use ManiSkill's native `RecordEpisode` wrapper → per-episode
    `trajectory.h5` + `.mp4` + `trajectory.json` (same layout RDT reads).
  • Use the proven scripted 3-phase motion from
    examples/motionplanning/panda/panda/bananacut.py (`_move_to_pose` with
    `pd_ee_delta_pos`) — no mplib planner.
  • Drive MPM through the bridge after each `env.step` (no extra physics).
  • Write a sidecar `alignment.json` per episode with F/V telemetry, the
    variation metadata, and derived stats (for filtering / analysis).

Robot: `pkour` (pandaours.py) — supports pd_ee_delta_pos/pose out of the box.

Output layout (per run):
    <out>/<task>/auto_<seed>/
        trajectory.h5         (RDT-compatible)
        trajectory.json
        <video>.mp4
        alignment.json        (F/V + variation + frame markers)

Usage (GPU pod):
    export LD_LIBRARY_PATH=/workspace/envs/maniskill/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/data:/data/mani_skill
    cd /data/EEF-Cutting-Simulation
    /workspace/envs/maniskill/bin/python \
        /data/mani_skill/dataset_converters/collect_rollouts_mpm.py \
            --task bananacut --num-episodes 3 \
            --out /data/datasets/maniskill_mpm \
            --mpm-config /data/EEF-Cutting-Simulation/configs/banana.yaml
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------------- #
# Scripted motion (direct copy of examples/motionplanning/panda/panda/bananacut.py,
# tuned minimally for our use case).
# ------------------------------------------------------------------------- #
def move_to_pose(env, start_pose_w, target_pose_w, *,
                 steps=100, kp=7.0, clip=0.03, gripper_delta=0.0,
                 on_step=None):
    """Drive pd_ee_delta_pos arm toward target by linear interpolation of
    base-frame positions, feeding position-delta (+1 grip) actions."""
    import sapien
    u = env.unwrapped
    base_pose = u.agent.robot.get_links()[0].pose

    start_p = np.asarray((base_pose.inv() * start_pose_w).p,
                         dtype=np.float32).reshape(3)
    target_p = np.asarray((base_pose.inv() * target_pose_w).p,
                          dtype=np.float32).reshape(3)

    for i in range(steps):
        ee_p = np.asarray((base_pose.inv() * u.agent.tcp.pose).p,
                          dtype=np.float32).reshape(3)
        alpha = i / max(steps - 1, 1)
        interp = (1 - alpha) * start_p + alpha * target_p
        dpos = np.clip((interp - ee_p) * kp, -clip, clip).astype(np.float32)

        action_dim = int(np.prod(env.action_space.shape))
        if action_dim >= 7:
            # pd_ee_delta_pose: (dx, dy, dz, drx, dry, drz, gripper)
            drot = np.zeros(3, np.float32)
            action = np.concatenate([dpos, drot, np.array([gripper_delta], np.float32)])
        elif action_dim >= 4:
            action = np.concatenate([dpos, np.array([gripper_delta], np.float32)])
        else:
            action = dpos

        env.step(action)
        if on_step is not None:
            on_step()


def hold_pose(env, bridge, *, n_ticks=20, gripper_delta=0.0, on_step=None):
    """Keep TCP where it currently is for n_ticks control ticks.

    Used between cut and retreat — lets MPM material finish damaging and
    makes the cut motion look natural (thrust → dwell → retract).
    """
    import sapien
    u = env.unwrapped
    base_pose = u.agent.robot.get_links()[0].pose
    cur_p_base = np.asarray(
        (base_pose.inv() * u.agent.tcp.pose).p, np.float32).reshape(3)
    for _ in range(n_ticks):
        ee_p = np.asarray(
            (base_pose.inv() * u.agent.tcp.pose).p, np.float32).reshape(3)
        dpos = np.clip((cur_p_base - ee_p) * 10.0, -0.02, 0.02).astype(np.float32)
        action_dim = int(np.prod(env.action_space.shape))
        if action_dim >= 7:
            drot = np.zeros(3, np.float32)
            action = np.concatenate(
                [dpos, drot, np.array([gripper_delta], np.float32)])
        elif action_dim >= 4:
            action = np.concatenate(
                [dpos, np.array([gripper_delta], np.float32)])
        else:
            action = dpos
        env.step(action)
        if on_step is not None:
            on_step()


def hold_at_pose(env, target_pose_w, *, n_ticks=20, gripper_delta=0.0,
                 on_step=None, kp=10.0, clip=0.02):
    """Hold TCP at a SPECIFIC target pose (world frame) for n_ticks.

    Used to actively settle MS to the MPM cut-bottom before flipping to
    MS-driven mode for retreat — avoids the upward MPM teleport that
    happens when MS lags MPM at end of cut.
    """
    import sapien
    u = env.unwrapped
    base_pose = u.agent.robot.get_links()[0].pose
    target_p_base = np.asarray((base_pose.inv() * target_pose_w).p,
                               dtype=np.float32).reshape(3)
    for _ in range(n_ticks):
        ee_p = np.asarray((base_pose.inv() * u.agent.tcp.pose).p,
                          dtype=np.float32).reshape(3)
        dpos = np.clip((target_p_base - ee_p) * kp, -clip, clip).astype(np.float32)
        action_dim = int(np.prod(env.action_space.shape))
        if action_dim >= 7:
            drot = np.zeros(3, np.float32)
            action = np.concatenate(
                [dpos, drot, np.array([gripper_delta], np.float32)])
        elif action_dim >= 4:
            action = np.concatenate(
                [dpos, np.array([gripper_delta], np.float32)])
        else:
            action = dpos
        env.step(action)
        if on_step is not None:
            on_step()


def cut_velocity_descent(env, bridge, *, control_dt, v_pre_contact,
                         max_ticks, min_tip_z, tcp_to_tip_z,
                         gripper_delta=0.0, on_step=None,
                         stop_tolerance=0.005, hard_tcp_floor_z=None):
    """Cut phase with (b) soft velocity injection.

    Pre-contact: MS commands constant descent at v_pre_contact (m/s).
    On contact: bridge auto-flips to MPM-driven (knife.y advanced by MPM).
    Bridge stashes MPM velocity on env each tick; MS reads it and commands
    dpos = v_mpm_in_ms_frame * dt (soft velocity injection through PID).
    Loop exits when MPM knife reaches its stop_y (tip just above MPM board)
    or max_ticks elapsed.
    """
    import sapien
    u = env.unwrapped
    yfoot = float(getattr(bridge, "_knife_yfoot", 0.0))
    v_off = float(bridge.calib.v_offset)

    # Hard absolute TCP floor: even if tcp_to_tip_z is mis-measured, never
    # let the commanded TCP drop below this. Empirically the EE was driving
    # ~3 cm BELOW the board (hand_camera z = -0.007 vs board top 0.020) when
    # the relative-floor was the only guard, so we add an absolute backstop.
    # The wrist/blade extend below TCP. With knife mounted via
    # HAND_TO_KNIFE_(XYZ,RPY), tip is ~10 cm BELOW TCP (=hand link) when the
    # hand z-axis points world-up at cut time. So TCP must stay >= board+10cm
    # for tip to clear the board. 0.13 absolute → tip ≈ board+30mm: small
    # fruits like cherry (~25mm tall) get a real cut without board contact.
    # Caller may override (e.g. legacy fruits keep no extra floor since their
    # cut tuning already worked).
    HARD_TCP_FLOOR_Z = (hard_tcp_floor_z if hard_tcp_floor_z is not None
                        else MS_BOARD_TOP_Z + 0.110)  # = 0.130 by default

    # pd_ee_delta_pose action range is [-1,1] mapped to [pos_lower,pos_upper]=
    # [-0.1, 0.1]. So action = (commanded delta in m) / 0.1.
    ACTION_SCALE = 0.1
    # Sanity-fallback for tcp_to_tip_z (NaN / wildly off measurement would
    # silently disable the floor). Knife mount → tip is at hand local
    # (+0.30, 0, +0.10); under cut-time orientation, tip world z ≈ TCP_z, so
    # tcp_to_tip_z is expected to be ~ -0.10 to 0. Anything outside [-0.5, 0.1]
    # is suspicious; clamp it.
    if not np.isfinite(tcp_to_tip_z) or tcp_to_tip_z > 0.1 or tcp_to_tip_z < -0.5:
        print(f"[cut] suspicious tcp_to_tip_z={tcp_to_tip_z}; clamping to -0.10")
        tcp_to_tip_z = -0.10
    for tick in range(max_ticks):
        in_contact = bool(getattr(u, "_last_mpm_in_contact", False))
        if not in_contact:
            v_z_ms = -float(v_pre_contact)
        else:
            v_mpm = getattr(u, "_last_mpm_velocity", None)
            if v_mpm is None:
                v_z_ms = -float(v_pre_contact)
            else:
                # MPM y-up → MS z-up. v_mpm[1] is vertical (= MS z).
                v_z_ms = float(v_mpm[1])
        cur_tcp_p = np.asarray(u.agent.tcp.pose.p, np.float32).reshape(-1)
        target_tcp_z = cur_tcp_p[2] + v_z_ms * float(control_dt)
        # Floor: don't command MS knife tip below safe z. Combine the
        # tip-relative floor with the hard absolute backstop.
        floor_tcp_z = max(min_tip_z + (-tcp_to_tip_z), HARD_TCP_FLOOR_Z)
        # NaN guard — float NaN compares False everywhere so max() can return
        # NaN and silently disable the clamp. Force a finite floor.
        if not np.isfinite(floor_tcp_z):
            print(f"[cut tick {tick}] floor_tcp_z is NaN — forcing to {HARD_TCP_FLOOR_Z}")
            floor_tcp_z = HARD_TCP_FLOOR_Z
        target_tcp_z = max(target_tcp_z, floor_tcp_z)
        # Diagnostic print: log floor activity at the start and every 30 ticks.
        if tick < 3 or tick % 30 == 0:
            print(f"[cut tick {tick}] cur_z={cur_tcp_p[2]:+.4f} "
                  f"v_z={v_z_ms:+.4f} target_z={target_tcp_z:+.4f} "
                  f"floor={floor_tcp_z:+.4f} in_contact={in_contact}")
        delta_z = target_tcp_z - cur_tcp_p[2]
        action_z = float(np.clip(delta_z / ACTION_SCALE, -1.0, 1.0))
        dpos = np.array([0.0, 0.0, action_z], np.float32)
        action_dim = int(np.prod(env.action_space.shape))
        if action_dim >= 7:
            drot = np.zeros(3, np.float32)
            action = np.concatenate([dpos, drot, np.array([gripper_delta], np.float32)])
        elif action_dim >= 4:
            action = np.concatenate([dpos, np.array([gripper_delta], np.float32)])
        else:
            action = dpos
        env.step(action)
        if on_step is not None:
            on_step()
        # Stop when MPM knife reaches its stop position
        if in_contact:
            stop_y_tip_mpm = float(bridge.sim.knife.stop_y) + yfoot
            mpm_knife_tip_y = float(bridge.sim.knife.y[None]) + yfoot
            if mpm_knife_tip_y <= stop_y_tip_mpm + stop_tolerance:
                break


def follow_mpm_descent(env, bridge, *, xy_fixed, tcp_to_tip_z_offset,
                       v_commanded=0.1, control_dt=0.05,
                       max_ticks=200, kp=10.0, clip=0.1, gripper_delta=0.0,
                       on_step=None, stop_tolerance=0.005):
    """Contact-triggered cut phase.

    Pre-contact: MS commands knife to descend at v_commanded (constant).
    On contact: bridge auto-flips to MPM-driven; MS follows MPM knife.y.
    Stops when MPM reaches stop_y OR max_ticks.
    """
    import sapien
    u = env.unwrapped
    v_off = float(bridge.calib.v_offset)
    yfoot = float(getattr(bridge, "_knife_yfoot", 0.0))
    base_pose = u.agent.robot.get_links()[0].pose

    for tick in range(max_ticks):
        if not bridge._in_contact:
            # Pre-contact: MS dictates a descent at commanded velocity.
            cur_tcp_z = float(np.asarray(u.agent.tcp.pose.p, np.float32).reshape(-1)[2])
            tcp_z_target_ms = cur_tcp_z - float(v_commanded) * float(control_dt)
        else:
            # Post-contact: track MPM knife (its physics now dictates).
            mpm_knife_tip_y = float(bridge.sim.knife.y[None]) + yfoot
            tip_z_target_ms = mpm_knife_tip_y - v_off
            # Floor: keep MS blade tip ≥ MS_BOARD_TOP + clearance to avoid
            # sapien rigid contact with the board (which would push the knife
            # back up — visible "lifting" near the bottom).
            tip_z_target_ms = max(tip_z_target_ms, MS_TIP_FLOOR_Z)
            tcp_z_target_ms = tip_z_target_ms + float(tcp_to_tip_z_offset)

        # Current TCP in base frame, target in base frame
        ee_p = np.asarray((base_pose.inv() * u.agent.tcp.pose).p,
                          dtype=np.float32).reshape(3)
        target_p = np.asarray((base_pose.inv() * sapien.Pose(
            p=[float(xy_fixed[0]), float(xy_fixed[1]), tcp_z_target_ms])
        ).p, dtype=np.float32).reshape(3)

        dpos = np.clip((target_p - ee_p) * kp, -clip, clip).astype(np.float32)
        action_dim = int(np.prod(env.action_space.shape))
        if action_dim >= 7:
            drot = np.zeros(3, np.float32)
            action = np.concatenate([dpos, drot, np.array([gripper_delta], np.float32)])
        elif action_dim >= 4:
            action = np.concatenate([dpos, np.array([gripper_delta], np.float32)])
        else:
            action = dpos

        env.step(action)
        if on_step is not None:
            on_step()

        # Stop when MPM knife has reached its stop_y (only meaningful in contact)
        if bridge._in_contact:
            stop_y_tip_mpm = float(bridge.sim.knife.stop_y) + yfoot
            mpm_knife_tip_y = float(bridge.sim.knife.y[None]) + yfoot
            if mpm_knife_tip_y <= stop_y_tip_mpm + stop_tolerance:
                break


# ------------------------------------------------------------------------- #
# Variation sampling — delegate to common.variations
# ------------------------------------------------------------------------- #
# Per-fruit scripted motion + variation overrides.
# - scale_range: cap large fruits so knife doesn't collide with oversized object
# - approach_z / cut_z: tuned per fruit. cut_z is the TCP-target Z relative to
#   the fruit's mesh-origin Z (= board_center.z). It must be low enough that the
#   MPM knife's blade zone (extends ~0.24 m from its tip upward along +Y in MPM
#   frame, i.e. upward in world) sweeps through the *entire* fruit height, not
#   only the top portion. Fruit heights (world-Z after lay-down rotation):
#       banana 0.06, cucumber 0.10, apple 0.10, orange 0.12, peach 0.12,
#       strawberry 0.11, melon 0.18.
#   Rule of thumb: cut_z ≈ -(fruit_height - 0.01) so blade passes through the
#   whole fruit. We keep slightly above board bottom (-0.02) to avoid collision
#   with the physical board.
# mpm_extra_descent pushes the MPM knife Y below the MS-tracked tip during the
# cut phase of rendering. The MS robot bottoms out at MS-z ≈ 0.065 due to
# physical arm/board constraints (cut_z tuning has no effect past that), so
# the MPM blade SDF would only touch the top ~1–2 cm of the fruit. Extra
# descent = (MS-tip-min-mpm-y) − (fruit bottom MPM-y). Tuned per fruit from
# measured particle AABB (y_bot) + a small margin below board.
# MS-side board top z = BOARD_THICKNESS = 0.020 (see bananacut.py). Floor MS
# commanded tip safely above board to avoid the rigid-contact "lifting" that
# happens when the knife is commanded close to the board. MS visual stops
# near fruit top — that is fine because cut depth is achieved in MPM via
# bridge.cfg.mpm_extra_descent_y (knife.y in MPM is shifted lower than
# MS_tip would naturally map to, so MPM blade reaches fruit bottom).
MS_BOARD_TOP_Z = 0.020
MS_TIP_FLOOR_Z = MS_BOARD_TOP_Z + 0.015  # = 0.035 (1.5 cm above board)

FRUIT_CONFIG = {
    # mpm_extra_descent: applied ONLY in the render subprocess (mpm.mp4) to push
    # the visualized knife down through the fruit body — the bridge data keeps
    # MS↔MPM coordinates in strict v_offset relation (npz is honest). Knife in
    # video stops just above MPM board (board_top_y_mpm ≈ 0.046, so 0.040–0.045
    # extra puts visualized knife at ~0.005 m above board). Force label in
    # alignment.json reflects MPM physics at the data-honest knife position.
    # mpm_extra_descent now 0 — MPM has its own native cut that drives knife
    # to board_top + 5 mm, no render-side push needed (was causing rendered
    # knife to plunge below the board).
    "banana":     {"scale_range": (0.8, 1.2),  "approach_z": 0.50, "cut_z": 0.025, "mpm_extra_descent": 0.0},
    "cucumber":   {"scale_range": (0.8, 1.2),  "approach_z": 0.50, "cut_z": 0.025, "mpm_extra_descent": 0.0},
    "apple":      {"scale_range": (0.8, 1.2),  "approach_z": 0.50, "cut_z": 0.025, "mpm_extra_descent": 0.0},
    "orange":     {"scale_range": (0.8, 1.2),  "approach_z": 0.50, "cut_z": 0.025, "mpm_extra_descent": 0.0},
    "peach":      {"scale_range": (0.8, 1.2),  "approach_z": 0.50, "cut_z": 0.025, "mpm_extra_descent": 0.0},
    "strawberry": {"scale_range": (0.8, 1.2),  "approach_z": 0.50, "cut_z": 0.025, "mpm_extra_descent": 0.0},
    "melon":      {"scale_range": (0.6, 1.0),  "approach_z": 0.55, "cut_z": 0.050, "mpm_extra_descent": 0.0},
    # New (additional_fruits) — mesh sizes vary widely so scale_range absolutizes
    # to ~5-10 cm physical fruit dimension on the board (visible from review_camera).
    # Bumped ~1.3× from previous values per pilot review — fruits looked too
    # small at 256×256 RGB. Keeps per-fruit relative sizing intact.
    "cherry":            {"scale_range": (0.007, 0.013),  "approach_z": 0.50, "cut_z": 0.025, "mpm_extra_descent": 0.0},
    "grape":             {"scale_range": (0.012, 0.020),  "approach_z": 0.50, "cut_z": 0.025, "mpm_extra_descent": 0.0},
    "shine_muscat":      {"scale_range": (0.025, 0.035),  "approach_z": 0.50, "cut_z": 0.025, "mpm_extra_descent": 0.0},
    # golden_strawberry excluded from production data collection (user request).
    # plum, tomato — mesh is already in meter scale (~5-8cm), aligned with
    # banana/apple/peach existing convention so use uniform (0.8, 1.2) scale.
    "plum":              {"scale_range": (1.5,   2.2),    "approach_z": 0.50, "cut_z": 0.025, "mpm_extra_descent": 0.0},
    "tomato":            {"scale_range": (1.2,   1.8),    "approach_z": 0.50, "cut_z": 0.025, "mpm_extra_descent": 0.0},
    # pear/lemon/kiwi — mesh in non-meter units (1-9m raw), per-fruit calibrated
    "pear":              {"scale_range": (0.080, 0.130),  "approach_z": 0.50, "cut_z": 0.025, "mpm_extra_descent": 0.0},
    "lemon":             {"scale_range": (0.018, 0.029),  "approach_z": 0.50, "cut_z": 0.025, "mpm_extra_descent": 0.0},
    "kiwi":              {"scale_range": (0.010, 0.018),  "approach_z": 0.50, "cut_z": 0.025, "mpm_extra_descent": 0.0},
}


def _sample_variation(rng, args):
    from dataset_converters.common import sample_variation, OBJECT_CATEGORIES
    kwargs = {}
    if args.fixed_object:
        kwargs["fixed_object"] = args.fixed_object
    else:
        kwargs["categories"] = OBJECT_CATEGORIES
    # Apply per-fruit scale_range unless user overrode
    fruit_cfg = FRUIT_CONFIG.get(args.fixed_object or "banana", {})
    if args.no_scale:
        kwargs["scale_range"] = (1.0, 1.0)
    elif "scale_range" in fruit_cfg:
        kwargs["scale_range"] = fruit_cfg["scale_range"]
    if args.no_pos:
        kwargs["pos_x_range"] = (0.0, 0.0)
        kwargs["pos_y_range"] = (0.0, 0.0)
    # Tier 1: yaw always forced off for stable scripted cut. Tier 2 will re-enable.
    kwargs["yaw_range"] = (0.0, 0.0)
    v = sample_variation(rng, **kwargs)
    # Background variation: cycle scene presets (5 in env). Random per-ep.
    v["scene_idx"] = int(rng.integers(0, 5))
    # Apartment-stage backdrop: 0=default lab, 1..5=ReplicaCAD apt 0..4.
    # Gated on --bg-scenes; default off so existing collection pipeline is unchanged.
    if getattr(args, "bg_scenes", False):
        # 0=default lab, 1..15=apartment backdrops (1 ReplicaCAD + 14 AI2THOR).
        v["bg_idx"] = int(rng.integers(0, 16))
    if getattr(args, "bg_only_idx", -1) >= 0:
        # Force a single bg_idx for every episode (used to build per-env splits
        # of 60 eps each once an env passes pilot review).
        v["bg_idx"] = int(args.bg_only_idx)
    return v


# ------------------------------------------------------------------------- #
# Core episode
# ------------------------------------------------------------------------- #
def _collect_one(args, seed, idx, variation, rng):
    import gymnasium as gym
    import sapien
    import mani_skill  # noqa: F401
    from mani_skill.utils.wrappers.recordori import RecordEpisode
    from dataset_converters.mpm import (
        ManiSkillMPMBridge, BridgeConfig, FrameCalibration, compute_alignment,
    )
    # bridge2/bridge3/bridge5 — addfruits-pilot only. Same public API as
    # ManiSkillMPMBridge. bridge2 → /data/Cutting_rebuttal, bridge3 / bridge5
    # → /data/Cutting_rubuttal2. Select with env var MPM_BRIDGE_VARIANT.
    # bridge5 is the stable snapshot of bridge3 after the v8 verification
    # round (runtime canonical + cut-phase-only coupling); bridge3 stays as
    # the live dev copy.
    _BR_VARIANT = os.environ.get("MPM_BRIDGE_VARIANT", "bridge5").lower()
    if _BR_VARIANT == "bridge5":
        from dataset_converters.mpm.additional_fruit_bridge5 import (
            ManiSkillMPMBridge as ManiSkillMPMBridge2,
            BridgeConfig as BridgeConfig2,
        )
        _BR_CONFIGS_ROOT = "/data/Cutting_rubuttal2/configs/fruits"
    elif _BR_VARIANT == "bridge4":
        # bridge4: bridge3 + env-driven cut_surface_z (bgvar variant). The
        # bridge auto-reads env.cut_surface_z at setup time and overrides
        # FrameCalibration.v_offset accordingly. MPM physics / material
        # properties are unchanged from bridge3.
        from dataset_converters.mpm.additional_fruit_bridge4 import (
            ManiSkillMPMBridge as ManiSkillMPMBridge2,
            BridgeConfig as BridgeConfig2,
        )
        _BR_CONFIGS_ROOT = "/data/Cutting_rubuttal2/configs/fruits"
    elif _BR_VARIANT == "bridge3":
        from dataset_converters.mpm.additional_fruit_bridge3 import (
            ManiSkillMPMBridge as ManiSkillMPMBridge2,
            BridgeConfig as BridgeConfig2,
        )
        _BR_CONFIGS_ROOT = "/data/Cutting_rubuttal2/configs/fruits"
    else:
        from dataset_converters.mpm.additional_fruit_bridge2 import (
            ManiSkillMPMBridge as ManiSkillMPMBridge2,
            BridgeConfig as BridgeConfig2,
        )
        _BR_CONFIGS_ROOT = "/data/Cutting_rebuttal/configs/fruits"
    print(f"[collect] using {_BR_VARIANT} (cfg root: {_BR_CONFIGS_ROOT})")

    output_dir = Path(args.out) / args.task / f"auto_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # obs_mode="rgb" stores per-frame base_camera RGB (256x256 uint8) inside the
    # h5 trajectory — required for VLA training (OpenVLA, Octo, RDT).
    obs_mode = getattr(args, "obs_mode", "rgb")
    env = gym.make(
        args.task,
        obs_mode=obs_mode,
        control_mode=args.control_mode,
        render_mode="rgb_array",
        reward_mode="none",
        enable_shadow=False,
        robot_uids=args.robot_uid,
    )
    save_video = bool(getattr(args, "save_video", False))
    mpm_only = bool(getattr(args, "mpm_only", False))
    save_trajectory = not mpm_only
    if mpm_only:
        save_video = False  # mpm-only: skip MS mp4 entirely
    env = RecordEpisode(
        env,
        output_dir=str(output_dir),
        trajectory_name="trajectory",
        save_trajectory=save_trajectory,
        save_video=save_video,
        info_on_video=False,
        record_env_state=True,
        source_type="scripted_teleop_mpm",
        source_desc=f"bananacut scripted 3-phase with MPM co-sim; variation={variation}",
    )

    obs, info = env.reset(
        seed=seed,
        options=dict(
            save_trajectory=True,
            reconfigure=True,
            # Pass variation so env applies scale/yaw/mesh_path in _load_scene /
            # _initialize_episode. pos_offset is added on top of env's own
            # rand_x/rand_y.
            variation=variation,
        ),
    )
    u = env.unwrapped

    # Per-bg dynamic cut surface (bananacut_bgvar / bananacut_bgvar2 expose
    # this; vanilla bananacut falls back to the legacy 0.020 board top).
    cut_surface_z_dyn = float(getattr(u, "cut_surface_z", MS_BOARD_TOP_Z))
    cut_tip_clearance_dyn = float(getattr(u, "cut_tip_clearance", 0.015))
    print(f"[collect] cut_surface_z={cut_surface_z_dyn:.4f}  "
          f"cut_tip_clearance={cut_tip_clearance_dyn:.4f}")

    # ---- banana pose for bridge (banana-local frame canceling) ----
    block = getattr(u, "block_apple", [None])[0]
    assert block is not None, "env has no block_apple"
    banana_p = np.asarray(block.pose.p.cpu().numpy(), np.float32).reshape(-1)
    # Rotated-mesh-centroid XY — use for BOTH knife motion target (MS) and
    # bridge mapping origin (MPM). This aligns MS visual center with MPM
    # particle centroid so the knife actually passes through the fruit middle.
    centroid_xy = np.asarray(
        getattr(u, "_visual_centroid_xy", np.zeros(2, dtype=np.float32)),
        dtype=np.float32)
    banana_yaw = float(variation.get("yaw", 0.0))
    banana_xy_visual = banana_p[:2] + centroid_xy

    # ---- bridge ----
    # addfruits-pilot fruits → bridge2 (loads from /data/Cutting_rebuttal,
    # uses assets/fruit2 mesh family). Legacy fruits → original bridge.
    _ADDFRUITS_USE_BRIDGE2 = {"cherry", "plum", "lemon", "kiwi", "shine_muscat",
                              "tomato", "pear", "grape"}
    _fruit_for_bridge = variation.get("object", "banana")
    if _fruit_for_bridge in _ADDFRUITS_USE_BRIDGE2:
        _br2_cfg_path = f"{_BR_CONFIGS_ROOT}/{_fruit_for_bridge}.yaml"
        # Per-fruit lift: tiny fruits (cherry/grape/etc) need the seed lifted
        # above the board so the knife SDF sweeps through them, not above.
        # MPM knife min-y ≈ 0.056 (board+10mm). To cut the WHOLE fruit (not
        # just its top), fruit min-y must be ABOVE the knife min so the blade
        # sweeps from above the fruit down through it. Fruit natural min-y is
        # ≈ 0.049 (board+~3mm), so lift each fruit by ~10mm to put its bottom
        # above the knife min while keeping it well below the knife max (0.51).
        _LIFT_PER_FRUIT = {
            "cherry": 0.000, "grape": 0.000, "shine_muscat": 0.000,
            "kiwi": 0.000, "plum": 0.000,
            "lemon": 0.000, "tomato": 0.000, "pear": 0.000,
        }
        _lift = _LIFT_PER_FRUIT.get(_fruit_for_bridge, 0.0)
        # mpm_extra_descent_y must be 0 for all fruits.
        # Non-zero values push world_low_y() below board_tolerance (board_top_y-0.01≈0.036),
        # causing sim.step() to call force_up() every substep — the knife bounces back up
        # below the board surface where no fruit particles exist, preventing any contact.
        _extra_desc = 0.0
        _ms_scale = variation.get("scale", None)
        bridge = ManiSkillMPMBridge2(BridgeConfig2(
            mpm_config_yaml=_br2_cfg_path,
            calibration=FrameCalibration.for_fruit(_fruit_for_bridge),
            mpm_substeps_cap=int(args.mpm_substeps_cap),
            record_particles=True,
            apply_mpm_force_to_ms=bool(args.apply_mpm_force),
            mpm_force_scale=float(args.mpm_force_scale),
            mpm_force_clip=float(args.mpm_force_clip),
            knife_speed_mps=float(args.knife_speed_mps),
            fruit_y_offset=_lift,
            fruit_scale_override=float(_ms_scale) if _ms_scale is not None else None,
            mpm_extra_descent_y=_extra_desc,
            # User intent: do NOT couple ms/mpm fruit world positions. mpm
            # fruit stays at the cfg-canonical position (FRUIT_CANONICAL +
            # banana-local frame mapping). Only scale is shared (above).
            sync_fruit_pose=False,
        ))
        print(f"[{_BR_VARIANT}] using {_br2_cfg_path} "
              f"(lift={_lift}, scale={_ms_scale}, extra_desc={_extra_desc})")
    else:
        bridge = ManiSkillMPMBridge(BridgeConfig(
            mpm_config_yaml=str(args.mpm_config),
            calibration=FrameCalibration.for_fruit(_fruit_for_bridge),
            mpm_substeps_cap=int(args.mpm_substeps_cap),
            record_particles=True,
            apply_mpm_force_to_ms=bool(args.apply_mpm_force),
            mpm_force_scale=float(args.mpm_force_scale),
            mpm_force_clip=float(args.mpm_force_clip),
            knife_speed_mps=float(args.knife_speed_mps),
        ))
    bridge.setup(env, banana_xy_ms=banana_xy_visual, banana_yaw_ms=banana_yaw)
    bridge.reset_render_buffer()
    # Always begin recording — particles go into mpm_render.npz.
    bridge.begin_render(str(output_dir / "_mpm_frames"))

    # ---- scripted 3-phase targets (world frame) ----
    # Per-fruit override of approach_z / cut_z unless user passed explicit CLI vals.
    fruit_cfg = FRUIT_CONFIG.get(variation.get("object", "banana"), {})
    # Knife aims at VISUAL center (pose.p + rotated centroid XY). Bridge keeps
    # raw pose.p for MPM (see above).
    obj_pos = banana_p.astype(np.float32).copy()
    obj_pos[:2] += centroid_xy
    approach_x = args.approach_x
    approach_z = fruit_cfg.get("approach_z", args.approach_z)
    cut_z = fruit_cfg.get("cut_z", args.cut_z)
    cut_steps = args.cut_steps

    # NOTE: previous attempts to add tip→TCP xy correction (knife-mount has
    # ~30 cm offset) pushed TCP outside Panda workspace and broke joints.
    # Original approach: TCP at fruit_x + approach_x (-0.25 default) — the
    # blade BODY still covers the fruit even though the tip ends up a few cm
    # to the side, so the cut succeeds. Centerline-tolerance handles the
    # offset on the eval side.
    init_pose_w = u.agent.tcp.pose
    # Per-bg cut surface — bananacut_bgvar exposes `cut_surface_z` (varies
    # 0.0–0.026 m depending on board presence/thickness); other envs fall back
    # to the global MS_BOARD_TOP_Z constant. `cut_tip_clearance` is the safety
    # gap between the knife tip floor and the cut surface (default 0.015 m for
    # legacy envs, 0.005 m for bgvar).
    _cut_surface_z = float(getattr(u, "cut_surface_z", MS_BOARD_TOP_Z))
    _cut_clearance = float(getattr(u, "cut_tip_clearance",
                                   MS_TIP_FLOOR_Z - MS_BOARD_TOP_Z))
    _ms_tip_floor_z = _cut_surface_z + _cut_clearance
    print(f"[ep {idx}] cut_surface_z={_cut_surface_z:.4f} m  "
          f"clearance={_cut_clearance:.4f} m  tip_floor={_ms_tip_floor_z:.4f} m")
    # cut_z is now ABSOLUTE relative to board_top (not fruit-center). New
    # fruits with lifted obj_z would otherwise put cut_pose.z above board by
    # fruit_center_z + cut_z → blade misses board, or commanded TCP-z makes
    # the blade tip drive into the board (joint distortion). Anchor to
    # _cut_surface_z + cut_z regardless of fruit position.
    approach_pose = sapien.Pose(
        p=obj_pos + np.array([approach_x, 0.0, approach_z], np.float32))
    cut_pose = sapien.Pose(
        p=np.array([obj_pos[0] + approach_x, obj_pos[1],
                    _cut_surface_z + cut_z], np.float32))
    retreat_pose = sapien.Pose(
        p=obj_pos + np.array([approach_x, 0.0, approach_z], np.float32))

    # ---- telemetry accumulators ----
    telemetry = dict(
        t=[], tip_ms=[], tip_mpm=[],
        v_ms=[], v_mpm=[], f_ms=[], f_mpm=[],
    )
    control_dt = 1.0 / float(getattr(u, "control_freq", 20))

    def _on_step():
        tele = bridge.step_with_env(env, dt_control=control_dt)
        telemetry["t"].append(len(telemetry["t"]))
        telemetry["tip_ms"].append(tele["ms_tip_pos"].tolist())
        telemetry["tip_mpm"].append(tele["mpm_tip_pos"].tolist())
        telemetry["v_ms"].append(tele["ms_tcp_vel"].tolist())
        telemetry["v_mpm"].append(tele["mpm_vel"].tolist())
        telemetry["f_ms"].append(tele["ms_force"].tolist())
        telemetry["f_mpm"].append(tele["mpm_force"].tolist())

    phase_bounds = {}

    print(f"[ep {idx}] approach  obj={obj_pos.tolist()}  approach_target={approach_pose.p}")
    phase_bounds["approach_start"] = len(telemetry["t"])
    # MPM↔MS coupling is OFF during approach. The TCP rotates and translates
    # laterally on the way to approach_pose; if the bridge kept writing
    # knife.y / knife.z_off from those frames the MPM knife would wiggle and
    # may sweep through the seeded fruit before the cut even starts.
    # Coupling is re-enabled at cut_start below (pure-Z descent).
    bridge.set_ms_driven(False)
    move_to_pose(env, init_pose_w, approach_pose, steps=100,
                 kp=7.0, clip=0.03, gripper_delta=0.0, on_step=_on_step)

    # Cut phase (b — soft velocity injection): bridge auto-flips to MPM-driven
    # on first contact. From that point MPM owns knife.y via its native cut
    # physics; bridge stashes MPM velocity onto env._last_mpm_velocity each
    # tick. MS commands its TCP delta = v_mpm * dt so it tracks MPM's velocity.
    # Pre-contact, MS commands a fixed descent at v_commanded (knife_speed_mps).
    _tip_ms_now = u._get_tip_from_eef()[0].detach().cpu().numpy().astype(np.float32)
    _tcp_ms_now = np.asarray(u.agent.tcp.pose.p, np.float32).reshape(-1)
    tcp_to_tip_z = float(_tip_ms_now[2] - _tcp_ms_now[2])  # typically negative
    if not np.isfinite(tcp_to_tip_z) or tcp_to_tip_z > 0.1 or tcp_to_tip_z < -0.5:
        print(f"[ep {idx}] suspicious tcp_to_tip_z={tcp_to_tip_z:+.3f}; clamping to -0.10")
        tcp_to_tip_z = -0.10
    print(f"[ep {idx}] tcp_to_tip_z={tcp_to_tip_z:+.3f} → updating bridge")
    bridge.cfg.tcp_to_tip_z = tcp_to_tip_z
    # NOTE: TCP xy offset for blade-tip alignment is now baked into
    # approach_pose / cut_pose (above), so no extra "over_fruit_pose" step.
    phase_bounds["cut_start"] = len(telemetry["t"])
    # Re-enable MPM↔MS coupling: cut phase is a pure-Z descent so this is the
    # only phase where tracking the MS TCP into MPM knife.y/z_off is sound.
    bridge.set_ms_driven(True)
    # Apply the conservative absolute floor (board+11cm → tip ≈ board+3cm)
    # ONLY for the addfruits-pilot fruits whose cut tuning needs the wrist
    # protection. Legacy fruits (banana/apple/etc) keep the original
    # tip-relative floor only — no behaviour change for production data.
    # Per-fruit override: tall fruits (tomato ~15cm, pear ~10cm) need a lower
    # floor so the blade reaches the bottom of the fruit. Smaller fruits keep
    # the conservative +110mm floor.
    _ADDFRUITS = {"cherry", "plum", "lemon", "kiwi", "shine_muscat",
                  "tomato", "pear", "grape"}
    _HARD_FLOOR_OFFSET_PER_FRUIT = {
        # offset above board_top_z (0.02). Knife tip = TCP + tcp_to_tip_z (-0.10).
        # → world_low_y_mpm = TCP - 0.075. Lower offset = deeper cut.
        # pear is OK with the 0.110 default (user-validated). Tomato ~15cm tall
        # needs a deeper cut to reach the bottom.
        "tomato":  0.090,   # tall → tip ≈ board+1cm
    }
    _fruit_name = variation.get("object", "banana")
    if _fruit_name in _ADDFRUITS:
        _floor_off = _HARD_FLOOR_OFFSET_PER_FRUIT.get(_fruit_name, 0.110)
        # Anchor to per-bg cut surface (bgvar) or default board top — keeps
        # the same +110mm wrist clearance regardless of board presence.
        _hard_floor = _cut_surface_z + _floor_off
    else:
        _hard_floor = float("-inf")
    cut_velocity_descent(env, bridge, control_dt=control_dt,
                         v_pre_contact=float(args.knife_speed_mps),
                         max_ticks=max(args.cut_steps, 200),
                         min_tip_z=_ms_tip_floor_z, tcp_to_tip_z=tcp_to_tip_z,
                         on_step=_on_step,
                         hard_tcp_floor_z=_hard_floor)

    # Freeze MPM knife: stop native descent and DISABLE MS coupling. During
    # hold + retreat the TCP translates upward / laterally, but the MPM knife
    # should stay frozen at its post-cut position so the cut state is preserved
    # for the rendered video. Force injection also off (embedded-particle
    # elastic spring would fight MS retreat).
    bridge.freeze_knife()
    bridge.set_ms_driven(False)
    bridge.set_apply_force(False)
    print(f"[ep {idx}] hold at cut bottom (20 ticks ≈ 1s)")
    phase_bounds["hold_start"] = len(telemetry["t"])
    cur_p = np.asarray(u.agent.tcp.pose.p, np.float32).reshape(-1)
    cur_q = np.asarray(u.agent.tcp.pose.q, np.float32).reshape(-1)
    hold_target = sapien.Pose(p=cur_p.tolist(), q=cur_q.tolist())
    hold_at_pose(env, hold_target, n_ticks=20, gripper_delta=0.0,
                 on_step=_on_step)

    # Retreat: keep MPM knife frozen (set_ms_driven=False is already in effect
    # from hold). The MS arm pulls the visual knife up but MPM stays at the
    # cut-bottom state — this preserves the visible cut in the MPM video.
    print(f"[ep {idx}] retreat  retreat_target={retreat_pose.p}")
    phase_bounds["retreat_start"] = len(telemetry["t"])
    cur_p = np.asarray(u.agent.tcp.pose.p, np.float32).reshape(-1)
    cur_q = np.asarray(u.agent.tcp.pose.q, np.float32).reshape(-1)
    current_tcp_pose = sapien.Pose(p=cur_p.tolist(), q=cur_q.tolist())
    move_to_pose(env, current_tcp_pose, retreat_pose, steps=100,
                 kp=10.0, clip=0.1, gripper_delta=0.0, on_step=_on_step)
    phase_bounds["end"] = len(telemetry["t"])

    # Close env (flushes trajectory.h5 + .mp4 into output_dir).
    env.close()

    # Post-process: inject obs/agent/qpos into trajectory.h5 so RDT loader
    # (which reads `traj_X/obs/agent/qpos`) works without re-collecting.
    # For panda-like 9-DOF articulation, the 31-dim state layout is:
    #   [root_pose(7), root_linvel(3), root_angvel(3), qpos(9), qvel(9)]
    # so qpos = state[:, 13:22].
    if bool(getattr(args, "mpm_only", False)):
        # MPM-only mode: no h5 saved, skip qpos injection.
        pass
    else:
        try:
            import h5py
            h5_path = output_dir / "trajectory.h5"
            with h5py.File(h5_path, "a") as h5:
                for traj in list(h5.keys()):
                    if not traj.startswith("traj_"):
                        continue
                    art = h5[traj]["env_states"]["articulations"]["pkour"][:]
                    qpos = art[:, 13:22].astype(np.float32)
                    qvel = art[:, 22:31].astype(np.float32)
                    agent_grp = h5[traj]["obs"].require_group("agent")
                    if "qpos" in agent_grp:
                        del agent_grp["qpos"]
                    if "qvel" in agent_grp:
                        del agent_grp["qvel"]
                    agent_grp.create_dataset("qpos", data=qpos, compression="gzip")
                    agent_grp.create_dataset("qvel", data=qvel, compression="gzip")
            print(f"[ep {idx}] injected obs/agent/qpos ({qpos.shape}) into {h5_path.name}")
        except Exception as e:
            print(f"[ep {idx}] qpos injection failed: {e}")

    # ---- alignment + sidecar ----
    arr = lambda k: np.asarray(telemetry[k], np.float32)
    align = compute_alignment(
        ms_force=arr("f_ms"), ms_vel=arr("v_ms"),
        mpm_force=arr("f_mpm"), mpm_vel=arr("v_mpm"),
    )
    sidecar = {
        "task": args.task,
        "seed": int(seed),
        "variation": variation,
        "instruction": json.load(open(
            "/data/mani_skill/dataset_converters/common/instructions.json"
        ))[args.task],
        "control_mode": args.control_mode,
        "robot_uid": args.robot_uid,
        "phase_bounds": phase_bounds,
        "alignment": align,
        "per_step": {
            # Kept compact (per-step lists of 3-vecs).
            "tip_ms": telemetry["tip_ms"],
            "tip_mpm": telemetry["tip_mpm"],
            "v_ms": telemetry["v_ms"],
            "v_mpm": telemetry["v_mpm"],
            "f_ms": telemetry["f_ms"],    # debug only
            "f_mpm": telemetry["f_mpm"],  # label
        },
        "design": "A+C — velocity-gated success, MPM force is the label.",
    }
    (output_dir / "alignment.json").write_text(json.dumps(sidecar, indent=2))

    # Always dump MPM particle buffer to npz (lightweight; needed for force /
    # contact analysis labels). mp4 rendering is gated on --render-mpm-mp4.
    npz_path = output_dir / "mpm_render.npz"
    try:
        bridge.dump_buffer_npz(str(npz_path))
    except Exception as e:
        print(f"[ep {idx}] mpm npz dump failed: {e}")
    # Optional: render MPM cutting-demo view + time-aligned side-by-side grid
    if getattr(args, "render_mpm_mp4", False) or getattr(args, "render_mpm", False):
        mpm_mp4 = output_dir / "mpm.mp4"
        grid_mp4 = output_dir / "grid.mp4"
        try:
            # Dump buffer + render via isolated subprocess (Taichi GGUI headless).
            import subprocess as _sp
            ok_dump = True
            ok = False
            if ok_dump:
                extra_descent = float(
                    FRUIT_CONFIG.get(variation.get("object", "banana"), {})
                                .get("mpm_extra_descent", 0.0))
                sub = _sp.run(
                    ["/workspace/envs/maniskill/bin/python",
                     "/data/mani_skill/dataset_converters/mpm/render_mpm_episode.py",
                     "--npz", str(npz_path),
                     "--config", str(args.mpm_config),
                     "--out", str(mpm_mp4),
                     "--fps", "20",
                     "--extra-descent", f"{extra_descent:.4f}"],
                    capture_output=True, text=True, timeout=1200,
                )
                if sub.returncode == 0:
                    ok = mpm_mp4.exists()
                    print(f"[ep {idx}] render subprocess: {sub.stdout.strip().splitlines()[-1] if sub.stdout.strip() else 'ok'}")
                else:
                    print(f"[ep {idx}] render subprocess FAILED (rc={sub.returncode}):\n{sub.stderr[-400:]}")
            print(f"[ep {idx}] mpm.mp4 {'saved' if ok else 'skipped'}")
            if ok:
                # Build side-by-side grid (MS RGB | MPM cutting view) at 20 fps.
                import subprocess
                try:
                    import imageio_ffmpeg
                    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception:
                    ffmpeg = "ffmpeg"
                ms_mp4 = output_dir / "0.mp4"
                if ms_mp4.exists():
                    # MS mp4 is one frame per env.step (20 Hz) but RecordEpisode
                    # labels it 30 fps. Re-declare input framerate as 20 fps so
                    # that MS tick N and MPM tick N line up exactly on hstack.
                    cmd = [
                        ffmpeg, "-y",
                        "-r", "20", "-i", str(ms_mp4),
                        "-i", str(mpm_mp4),
                        "-filter_complex",
                        "[0:v]scale=-2:720[l];[1:v]scale=-2:720[r];[l][r]hstack",
                        "-r", "20",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(grid_mp4),
                    ]
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    if r.returncode == 0:
                        print(f"[ep {idx}] grid.mp4 saved")
                    else:
                        print(f"[ep {idx}] grid.mp4 ffmpeg err: {r.stderr[-300:]}")
        except Exception as e:
            print(f"[ep {idx}] mpm.mp4 FAILED: {e}")

    print(f"[ep {idx}] T={len(telemetry['t'])}  "
          f"vel_passed={align['passed']}  "
          f"max_V_err={align['max_vel_err']:.4g}m/s  "
          f"F_mpm_peak={align['mpm_force_peak']:.1f}N (label)  "
          f"F_ms_peak={align['ms_force_peak']:.1f}N (dbg)  "
          f"mpm_contact={align['mpm_contact_frames']}fr  "
          f"→ {output_dir}")


# ------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="bananacut")
    ap.add_argument("--num-episodes", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mpm-config", required=True)

    ap.add_argument("--robot-uid", default="pkour",
                    help="must support pd_ee_delta_pos (pkour does; pkv does not)")
    ap.add_argument("--control-mode", default="pd_ee_delta_pose",
                    help="Stiff PID (k=2e4): MS tracks commanded TCP precisely. "
                         "Compliant variant was tried but compliance effect of "
                         "Method B force injection turned out to be sub-mm under "
                         "reasonable stiffnesses, while compliant tracking lag "
                         "made cut commands unreachable. F_mpm is now used as a "
                         "force LABEL (logged in alignment.json) rather than a "
                         "feedback that bends the trajectory.")

    # Scripted motion params (defaults from examples/motionplanning/.../bananacut.py)
    ap.add_argument("--approach-x", type=float, default=-0.25)
    ap.add_argument("--approach-z", type=float, default=0.50)
    ap.add_argument("--cut-z", type=float, default=0.025)
    ap.add_argument("--cut-steps", type=int, default=30)

    # Variation
    ap.add_argument("--fixed-object", default="banana")
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--no-pos", action="store_true")
    ap.add_argument("--no-yaw", action="store_true")

    # Bridge perf knob
    ap.add_argument("--mpm-substeps-cap", type=int, default=0)
    # Optional MPM particle-view video (written as mpm.mp4 next to 0.mp4).
    # Note: mpm_render.npz is ALWAYS saved (records particles) even without this.
    ap.add_argument("--render-mpm-mp4", action="store_true",
                    help="also render mpm.mp4 + grid.mp4 (slow). NPZ always saved.")
    # backwards-compat alias
    ap.add_argument("--render-mpm", action="store_true", dest="render_mpm",
                    help="(legacy alias for --render-mpm-mp4)")
    # Per-frame env observation mode. "rgb" stores 256x256 RGB in trajectory.h5
    # (required for VLA training); "none" matches the legacy lab-pilot pipeline.
    ap.add_argument("--obs-mode", default="rgb", choices=["none", "rgb", "rgbd", "state"])
    # Whether to also save the standard 0.mp4 RGB video alongside the h5.
    ap.add_argument("--save-video", action="store_true",
                    help="also write 0.mp4 (the env's primary RGB video).")
    # MPM-only verification mode: no h5, no MS mp4 — only mpm.mp4 + alignment.json.
    # Use this to visually verify MPM cuts the fruit center and aligns per-tick
    # with ManiSkill before committing to a full RDT-format collection.
    ap.add_argument("--mpm-only", action="store_true",
                    help="skip h5 + MS mp4; only render MPM mp4 (verification mode)")
    # Tier 2: enable MPM → MS force feedback (bidirectional coupling)
    ap.add_argument("--apply-mpm-force", action="store_true")
    ap.add_argument("--mpm-force-scale", type=float, default=0.25)
    ap.add_argument("--mpm-force-clip", type=float, default=200.0)
    # Initial commanded knife descent speed (Chayoso yaml default 0.6 m/s is
    # too fast — we override to a realistic cutting speed). Grid-search this
    # across fruits: [0.05, 0.10, 0.15, 0.20, 0.30] m/s.
    ap.add_argument("--knife-speed-mps", type=float, default=0.10,
                    help="commanded knife descent speed (overrides yaml)")
    # Apartment-stage backdrop variation (ReplicaCAD). When on, each episode
    # picks bg_idx in {0..5} (0=default lab, 1..5=ReplicaCAD apt 0..4).
    ap.add_argument("--bg-scenes", action="store_true",
                    help="enable apartment-stage backdrop variation (random across 0..15)")
    ap.add_argument("--bg-only-idx", type=int, default=-1,
                    help="force every episode to use this single bg_idx (overrides --bg-scenes randomness)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed_base)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for i in range(args.num_episodes):
        seed = args.seed_base + i
        variation = _sample_variation(rng, args)
        print(f"[collect] ep={i} seed={seed} variation={variation}")
        _collect_one(args, seed, i, variation, rng)

    print("[done]")


if __name__ == "__main__":
    main()

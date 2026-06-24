"""Replay a saved Episode by injecting qpos into the env and rendering each
step with ManiSkill's human_render_camera, then writing the result to mp4.

Needs GPU/Vulkan (sapien RenderSystem). Run inside the same pod where
collect_rollouts_mpm.py was run. No MPM is instantiated here; this is a
pure ManiSkill visual replay.

Usage:
    /workspace/envs/maniskill/bin/python \
        dataset_converters/viz/replay_to_mp4.py \
        --ep /data/datasets/maniskill_mpm/bananacut/pd_joint_pos/episode_000001.npz \
        --out /data/datasets/maniskill_mpm/bananacut/pd_joint_pos/episode_000001_replay.mp4
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

from dataset_converters.common import Episode


def _annotate(img: np.ndarray, texts: list) -> np.ndarray:
    pil = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()
    y = 4
    for t in texts:
        draw.text((4, y), t, fill=(255, 255, 0), font=font,
                  stroke_width=1, stroke_fill=(0, 0, 0))
        y += 18
    return np.asarray(pil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--robot-uid", default="pkv")
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()

    ep = Episode.load(args.ep)
    T = ep.qpos.shape[0]
    out = args.out or args.ep.with_name(args.ep.stem + "_replay.mp4")

    import gymnasium as gym
    import mani_skill  # noqa: F401

    env = gym.make(
        ep.task,
        obs_mode="none",
        control_mode=ep.control_mode,
        render_mode="rgb_array",
        reward_mode="none",
        enable_shadow=False,
        robot_uids=args.robot_uid,
    )
    obs, info = env.reset(seed=ep.seed, options=dict(reconfigure=True))
    u = env.unwrapped

    writer = imageio.get_writer(out, fps=args.fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")

    import torch
    print(f"[replay] {T} frames → {out}")
    for t in range(T):
        # Inject saved qpos directly (bypass controller; pure visual replay).
        q = torch.as_tensor(ep.qpos[t], dtype=torch.float32, device=u.device)
        u.agent.robot.set_qpos(q.unsqueeze(0) if q.ndim == 1 else q)
        if u.gpu_sim_enabled:
            u.scene._gpu_apply_all()
            u.scene.px.gpu_update_articulation_kinematics()
            u.scene._gpu_fetch_all()
        img = u.render()
        if hasattr(img, "cpu"):
            img = img.cpu().numpy()
        img = np.asarray(img, np.uint8)
        if img.ndim == 4:
            img = img[0]
        if not args.no_overlay:
            f_mpm = float(np.linalg.norm(ep.mpm_knife_force[t]))
            f_ms = float(np.linalg.norm(ep.ms_knife_force[t]))
            texts = [
                f"t={t:03d}/{T-1}  task={ep.task}",
                f"tcp_z={float(ep.tcp_pose[t,2]):+.3f}  tip_mpm_y={float(ep.mpm_tip_world[t,1]):+.3f}",
                f"v_ms.z={float(ep.ms_tcp_vel[t,2]):+.3f}  v_mpm.y={float(ep.mpm_knife_vel[t,1]):+.3f}",
                f"|F_mpm|={f_mpm:6.1f}N  |F_ms|={f_ms:6.1f}N",
                f"success={ep.success}",
            ]
            img = _annotate(img, texts)
        writer.append_data(img)
    writer.close()
    env.close()
    print(f"[done] {out}  ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

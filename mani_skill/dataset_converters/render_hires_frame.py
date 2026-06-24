"""Re-render one frame of an existing episode at higher resolution.

Reads alignment.json + trajectory.h5, recreates env with the same seed + variation
but with high-res human-render cameras, sets env_state to the chosen frame, and
saves the wide-cam and render-cam images as PNGs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=Path, required=True,
                    help="episode dir, e.g. /data/datasets/.../bananacut/auto_13")
    ap.add_argument("--frame", type=int, default=130)
    ap.add_argument("--render-cam-size", type=int, default=1024)
    ap.add_argument("--wide-cam-w", type=int, default=1920)
    ap.add_argument("--wide-cam-h", type=int, default=1080)
    args = ap.parse_args()

    ep_dir = args.ep
    align = json.loads((ep_dir / "alignment.json").read_text())
    seed = align["seed"]
    task = align["task"]
    robot_uid = align["robot_uid"]
    control_mode = align["control_mode"]
    variation = align["variation"]
    print(f"[hires] task={task} seed={seed} robot={robot_uid} frame={args.frame}")

    import gymnasium as gym
    import mani_skill  # noqa: F401

    env = gym.make(
        task,
        obs_mode="none",
        control_mode=control_mode,
        render_mode="rgb_array",
        reward_mode="none",
        enable_shadow=True,
        robot_uids=robot_uid,
        human_render_camera_configs={
            "render_camera": {"width": args.render_cam_size, "height": args.render_cam_size},
            "wide_camera": {"width": args.wide_cam_w, "height": args.wide_cam_h},
        },
    )
    env.reset(seed=seed, options=dict(reconfigure=True, variation=variation))
    u = env.unwrapped

    # Load env_state at the requested frame
    with h5py.File(ep_dir / "trajectory.h5", "r") as f:
        traj = f["traj_0"]
        actors = traj["env_states"]["actors"]
        arts = traj["env_states"]["articulations"]
        state = {
            "actors": {k: np.asarray(actors[k][args.frame])[None] for k in actors.keys()},
            "articulations": {k: np.asarray(arts[k][args.frame])[None] for k in arts.keys()},
        }
    # Convert numpy → torch on env's device
    def to_t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=u.device)
    state = {top: {k: to_t(v) for k, v in d.items()} for top, d in state.items()}
    u.set_state_dict(state)

    # Use ManiSkill's standard human-render path (handles capture/cleanup
    # internally; calling cam.capture() ourselves caused vk::DeviceLost).
    for obj in u._hidden_objects:
        obj.show_visual()
    u.scene.update_render(update_sensors=False, update_human_render_cameras=True)
    images = u.scene.get_human_render_camera_images()
    print(f"[hires] cameras: {list(images.keys())}")

    out_dir = ep_dir
    saved = []
    for uid, rgb_t in images.items():
        rgb = rgb_t.cpu().numpy() if hasattr(rgb_t, "cpu") else np.asarray(rgb_t)
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                   if rgb.max() <= 1.5
                   else np.clip(rgb, 0, 255).astype(np.uint8))
        rgb = rgb[..., :3]
        out = out_dir / f"frame{args.frame}_{uid}_hires.png"
        Image.fromarray(rgb).save(out)
        saved.append(out)
        print(f"  {uid}: {rgb.shape} -> {out}")

    env.close()
    print("[done]")
    for s in saved:
        print(s)


if __name__ == "__main__":
    main()

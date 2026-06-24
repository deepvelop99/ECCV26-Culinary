"""Render an mp4 from a saved Episode.npz RGB buffer + an info overlay.

Each frame shows:
    tcp_z (ManiSkill), tip_mpm_y (MPM), vel z/y, force MPM/MS magnitudes.

Usage:
    /workspace/envs/maniskill/bin/python \
        dataset_converters/viz/render_video.py \
        --ep /data/datasets/maniskill_mpm/bananacut/pd_joint_pos/episode_000000.npz \
        --out /data/datasets/maniskill_mpm/bananacut/pd_joint_pos/episode_000000.mp4 \
        --fps 20
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

from dataset_converters.common import Episode


def _annotate(img: np.ndarray, texts: list, upscale: int = 4) -> np.ndarray:
    pil = Image.fromarray(img)
    W, H = pil.size
    pil = pil.resize((W * upscale, H * upscale), Image.NEAREST)
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
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()

    ep = Episode.load(args.ep)
    out = args.out or args.ep.with_suffix(".mp4")
    T = ep.rgb.shape[0]

    writer = imageio.get_writer(out, fps=args.fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    print(f"[render] {T} frames @ {args.fps}fps → {out}")
    for t in range(T):
        frame = ep.rgb[t]
        if not args.no_overlay:
            tcp_z = float(ep.tcp_pose[t, 2])
            tip_mpm_y = float(ep.mpm_tip_world[t, 1]) if ep.mpm_tip_world.any() else float("nan")
            v_ms_z = float(ep.ms_tcp_vel[t, 2])
            v_mpm_y = float(ep.mpm_knife_vel[t, 1])
            f_mpm = float(np.linalg.norm(ep.mpm_knife_force[t]))
            f_ms = float(np.linalg.norm(ep.ms_knife_force[t]))
            texts = [
                f"t={t:03d}/{T-1}  task={ep.task}",
                f"tcp_z={tcp_z:+.3f}  tip_mpm_y={tip_mpm_y:+.3f}",
                f"v_ms.z={v_ms_z:+.3f}  v_mpm.y={v_mpm_y:+.3f}",
                f"|F_mpm|={f_mpm:6.1f}N (label)  |F_ms|={f_ms:6.1f}N (dbg)",
                f"success={ep.success}",
            ]
            frame = _annotate(frame, texts, upscale=4)
        writer.append_data(frame)
    writer.close()
    print(f"[done] {out}  ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

"""Re-render mpm.mp4 + grid.mp4 for already-collected episodes with the new
mpm_extra_descent=0 setting (knife no longer plunges below the board in the
rendered MPM view).

Reuses the existing per-episode `mpm_render.npz` (knife trajectory) and `0.mp4`
(MS render). Calls `render_mpm_episode.py --extra-descent 0` then hstacks with
`0.mp4` to produce a new `grid.mp4`. Existing trajectory.h5 / alignment.json
are untouched.

Usage (single-process):
    python rerender_mpm.py --root /data/datasets/maniskill_mpm_full

Usage (sharded by --shard-of-n for parallel kube jobs):
    python rerender_mpm.py --root ... --shard 0 --of 10
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional


CONFIGS_ROOT = Path("/data/EEF-Cutting-Simulation/configs")
RENDER_SCRIPT = Path("/data/mani_skill/dataset_converters/mpm/render_mpm_episode.py")
PYTHON = Path("/workspace/envs/maniskill/bin/python")


def find_episodes(root: Path):
    """Return list of (auto_dir, fruit) tuples."""
    out = []
    for align in sorted(root.glob("*/bananacut/auto_*/alignment.json")):
        d = json.loads(align.read_text())
        fruit = d.get("variation", {}).get("object", "banana")
        out.append((align.parent, fruit))
    return out


def rerender_one(ep_dir: Path, fruit: str, *, extra_descent: float = 0.0,
                 timeout_sec: int = 600) -> Optional[str]:
    npz = ep_dir / "mpm_render.npz"
    mpm_mp4 = ep_dir / "mpm.mp4"
    grid_mp4 = ep_dir / "grid.mp4"
    ms_mp4 = ep_dir / "0.mp4"
    cfg = CONFIGS_ROOT / f"{fruit}.yaml"
    if not npz.exists():
        return f"missing npz: {ep_dir}"
    if not cfg.exists():
        return f"missing config: {cfg}"

    # 1) Re-render mpm.mp4
    sub = subprocess.run(
        [str(PYTHON), str(RENDER_SCRIPT),
         "--npz", str(npz),
         "--config", str(cfg),
         "--out", str(mpm_mp4),
         "--fps", "20",
         "--extra-descent", f"{extra_descent:.4f}"],
        capture_output=True, text=True, timeout=timeout_sec,
    )
    if sub.returncode != 0:
        return f"render rc={sub.returncode}: {sub.stderr[-200:]}"

    # 2) Rebuild grid.mp4 (hstack 0.mp4 | mpm.mp4 at 20fps)
    if ms_mp4.exists() and mpm_mp4.exists():
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = "ffmpeg"
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
        if r.returncode != 0:
            return f"grid ffmpeg rc={r.returncode}: {r.stderr[-200:]}"

    return None  # success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--extra-descent", type=float, default=0.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, only re-render this many episodes (debug)")
    args = ap.parse_args()

    eps = find_episodes(args.root)
    eps = [e for i, e in enumerate(eps) if i % args.of == args.shard]
    if args.limit > 0:
        eps = eps[:args.limit]

    print(f"[rerender] shard {args.shard}/{args.of}  total {len(eps)} eps")
    n_ok = 0
    n_err = 0
    for i, (ep_dir, fruit) in enumerate(eps):
        err = rerender_one(ep_dir, fruit, extra_descent=args.extra_descent)
        if err is None:
            n_ok += 1
            if i % 25 == 0:
                print(f"  [{i+1}/{len(eps)}] {fruit}/{ep_dir.name}: ok")
        else:
            n_err += 1
            print(f"  [{i+1}/{len(eps)}] {fruit}/{ep_dir.name}: ERR {err}")
    print(f"[rerender] done — ok={n_ok}  err={n_err}")


if __name__ == "__main__":
    main()

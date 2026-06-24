"""Coordinate transform between ManiSkill and MPM world frames.

Measured constants for the bananacut env ↔ banana.yaml MPM config:

ManiSkill frame:                        MPM frame:
    up axis = z                             up axis = y
    board top z = 0.02                      board top y = 0.045
    board center (x,y) = (-0.1, 0)          board center (x,z) = (0, 0)

Axis relabel: (x, y, z)_ms → (x, z, y)_mpm  (keep right-handedness)
Vertical offset: y_mpm = z_ms + V_OFFSET, V_OFFSET = 0.045 - 0.02 = 0.025
Horizontal offset: x_mpm = x_ms + 0.1, z_mpm = y_ms.

Banana variation handling (option α from design):
    knife → banana-local frame in ManiSkill, then place into MPM at the
    canonical banana center. This cancels out banana xy/yaw offset so the
    MPM knife always lands on the fixed MPM banana.

Scale variation is left to an SDF bucket cache (handled by the collect script),
not by this frame module.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# Constants measured from:
#   /data/mani_skill/envs/tasks/knife/bananacut.py    (BOARD_SIZE, board_center)
#   /data/EEF-Cutting-Simulation/configs/banana.yaml  (bounds, knife_start_y)
#   runtime reading of sim.board_top_y at /data/datasets/frame_calibration/run0.json
# Values below are the post-GPU-measurement calibrated ones.
MS_BOARD_TOP_Z = 0.02
MPM_BOARD_TOP_Y = 0.04642368108034134  # measured from MPM runtime (board SDF)
V_OFFSET = MPM_BOARD_TOP_Y - MS_BOARD_TOP_Z  # 0.02642...

MS_BOARD_CENTER_XY = np.array([-0.1, 0.0], dtype=np.float32)
MPM_BOARD_CENTER_XZ = np.array([0.0, 0.0], dtype=np.float32)

# Canonical MPM particle-AABB midpoint per fruit (measured at runtime after
# SDF seed + board-support clamp). See measure_fruit_canonicals.py — the
# midpoint is the natural cut-target since particles are pushed up to the
# board top after seeding.
FRUIT_CANONICAL = {
    "banana":     {"xz": (-0.04190, 0.00414), "y": 0.07762},
    "apple":      {"xz": (-0.03848, -0.00059), "y": 0.09884},
    "cucumber":   {"xz": (-0.02758, 0.01120), "y": 0.09613},
    "melon":      {"xz": (-0.03988, -0.00016), "y": 0.13771},
    "orange":     {"xz": (-0.03014, -0.00027), "y": 0.10870},
    "peach":      {"xz": (-0.02310, 0.00992), "y": 0.11179},
    "strawberry": {"xz": (-0.00665, 0.00081), "y": 0.09114},
    # additional_fruits — measured via measure_fruit_canonicals.py --new-only
    # (May 6 2026). pear pending (settle timeout in 1200s; needs longer run).
    "cherry":            {"xz": (-0.00150, 0.01220), "y": 0.05770},
    "plum":              {"xz": ( 0.00230, 0.04530), "y": 0.09800},
    "lemon":             {"xz": ( 0.00030, 0.02670), "y": 0.08330},
    "kiwi":              {"xz": ( 0.00390, 0.02000), "y": 0.07190},
    "shine_muscat":      {"xz": ( 0.00050, 0.02280), "y": 0.07130},
    "tomato":            {"xz": (-0.00560,-0.00050), "y": 0.12070},
    # Measured 2026-05-09 via measure_addfruits_canonicals.py (bridge3/Cutting_rubuttal2).
    "grape":             {"xz": ( 0.00025, 0.01199), "y": 0.05880},
    "golden_strawberry": {"xz": (-0.00059, 0.01897), "y": 0.06757},
    "pear":              {"xz": (-0.00080, 0.09160), "y": 0.09973},
}
# Legacy aliases (default to banana for backward-compat callers)
MPM_BANANA_CANONICAL_XZ = np.array(FRUIT_CANONICAL["banana"]["xz"], dtype=np.float32)
MPM_BANANA_CANONICAL_Y = float(FRUIT_CANONICAL["banana"]["y"])


def canonical_for_fruit(fruit: str):
    """Return (canonical_xz: np.ndarray(2,), canonical_y: float) for the given
    fruit. Falls back to banana if unknown."""
    d = FRUIT_CANONICAL.get(fruit, FRUIT_CANONICAL["banana"])
    return np.asarray(d["xz"], dtype=np.float32), float(d["y"])


@dataclass
class FrameCalibration:
    """Calibration snapshot, serialize with each episode for reproducibility.

    The *_canonical_* fields are per-fruit (see FRUIT_CANONICAL). Construct via
    `FrameCalibration.for_fruit("apple")` to pick the correct values; default
    no-arg construction keeps the banana values for backward compat.
    """
    v_offset: float = V_OFFSET
    ms_board_center_xy: tuple = (float(MS_BOARD_CENTER_XY[0]),
                                 float(MS_BOARD_CENTER_XY[1]))
    mpm_board_center_xz: tuple = (float(MPM_BOARD_CENTER_XZ[0]),
                                  float(MPM_BOARD_CENTER_XZ[1]))
    mpm_banana_canonical_xz: tuple = (float(MPM_BANANA_CANONICAL_XZ[0]),
                                      float(MPM_BANANA_CANONICAL_XZ[1]))
    mpm_banana_canonical_y: float = float(MPM_BANANA_CANONICAL_Y)
    fruit: str = "banana"

    @classmethod
    def for_fruit(cls, fruit: str) -> "FrameCalibration":
        xz, y = canonical_for_fruit(fruit)
        return cls(
            mpm_banana_canonical_xz=(float(xz[0]), float(xz[1])),
            mpm_banana_canonical_y=float(y),
            fruit=fruit,
        )

    def to_dict(self):
        return dict(
            v_offset=self.v_offset,
            ms_board_center_xy=list(self.ms_board_center_xy),
            mpm_board_center_xz=list(self.mpm_board_center_xz),
            mpm_banana_canonical_xz=list(self.mpm_banana_canonical_xz),
            mpm_banana_canonical_y=self.mpm_banana_canonical_y,
            fruit=self.fruit,
        )


def ms_world_to_mpm_world(p_ms: np.ndarray) -> np.ndarray:
    """Raw frame swap with vertical & horizontal offsets, *without* banana-local
    remapping. Useful for coarse debug / verification."""
    p_ms = np.asarray(p_ms, np.float32).reshape(-1)
    x_mpm = p_ms[0] - MS_BOARD_CENTER_XY[0] + MPM_BOARD_CENTER_XZ[0]
    y_mpm = p_ms[2] + V_OFFSET
    z_mpm = p_ms[1] - MS_BOARD_CENTER_XY[1] + MPM_BOARD_CENTER_XZ[1]
    return np.array([x_mpm, y_mpm, z_mpm], np.float32)


def ms_knife_tip_to_mpm(
    tip_ms: np.ndarray,
    banana_pose_ms_xy: np.ndarray,
    banana_yaw_ms: float = 0.0,
    *,
    canonical_xz: np.ndarray = MPM_BANANA_CANONICAL_XZ,
    v_offset: float = V_OFFSET,
) -> np.ndarray:
    """Map ManiSkill knife tip → MPM knife tip, via banana-local frame.

    tip_ms             : (3,) ManiSkill world xyz (z up)
    banana_pose_ms_xy  : (2,) ManiSkill banana (x, y) world
    banana_yaw_ms      : rotation of banana about z_ms (applied here as 2D yaw
                         around banana center, then un-rotated when placing
                         into MPM's canonical unrotated banana).

    Returns (3,) MPM world xyz (y up), positioned so the knife-banana relative
    pose is preserved w.r.t. the MPM banana's canonical (unrotated) pose.
    """
    tip_ms = np.asarray(tip_ms, np.float32).reshape(3)
    banana_xy = np.asarray(banana_pose_ms_xy, np.float32).reshape(2)

    # (1) knife xy in banana-local frame (ManiSkill, z-up)
    d_xy = tip_ms[:2] - banana_xy
    if banana_yaw_ms != 0.0:
        c, s = np.cos(-banana_yaw_ms), np.sin(-banana_yaw_ms)
        R = np.array([[c, -s], [s, c]], np.float32)
        d_xy = R @ d_xy

    # (2) axis swap to MPM frame (x stays x, y_ms -> z_mpm)
    local_mpm_xz = np.array([d_xy[0], d_xy[1]], np.float32)

    # (3) add canonical MPM banana xz
    mpm_xz = canonical_xz + local_mpm_xz

    # (4) vertical
    y_mpm = tip_ms[2] + v_offset
    return np.array([mpm_xz[0], y_mpm, mpm_xz[1]], np.float32)


def mpm_knife_y_from_tip_y(tip_y_mpm: float, knife_yfoot: float) -> float:
    """Convert desired MPM knife tip world-y into the `knife.y[None]` value
    used as the SDF origin's dynamic y (see KnifeCollider._sample_grid).
    """
    return float(tip_y_mpm) - float(knife_yfoot)

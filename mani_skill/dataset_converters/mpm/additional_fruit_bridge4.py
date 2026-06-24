"""ManiSkill ⇄ MPM bridge — bgvar variant (bridge4).

Identical to bridge3, with one addition: the FrameCalibration's `v_offset`
(MS-z → MPM-y conversion) is overridden at setup time to track the env's
per-bg `cut_surface_z`. This lets bananacut_bgvar (where the MS cut surface
varies between table-top z=0 and a thicker board top up to z≈0.026) keep
the MS↔MPM mapping aligned without per-bg manual calibration.

When ms_cut_surface_z is None and env has no `cut_surface_z` attribute,
bridge4 behaves identically to bridge3 (v_offset = 0.0264, MS_BOARD_TOP_Z=0.020).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from .additional_fruit_bridge3 import (
    BridgeConfig as _Bridge3Config,
    ManiSkillMPMBridge as _Bridge3,
    DEFAULT_REPO,  # noqa: F401  — re-export for callers using bridge3 surface
)
from .frame import MPM_BOARD_TOP_Y


@dataclass
class BridgeConfig(_Bridge3Config):
    """bridge3 config + ms_cut_surface_z override.

    If `ms_cut_surface_z` is None, bridge4 falls back to reading
    `env.cut_surface_z` at setup time. If neither is available the bridge
    keeps the bridge3 default v_offset.
    """
    ms_cut_surface_z: Optional[float] = None


class ManiSkillMPMBridge(_Bridge3):
    """bridge3 + env-driven v_offset adjustment for bananacut_bgvar."""

    def setup(self, env, *, banana_xy_ms=None, banana_yaw_ms: float = 0.0):
        ms_z = getattr(self.cfg, "ms_cut_surface_z", None)
        if ms_z is None:
            ms_z = getattr(env.unwrapped, "cut_surface_z", None)
        if ms_z is not None:
            old_v = float(self.calib.v_offset)
            new_v = float(MPM_BOARD_TOP_Y) - float(ms_z)
            self.calib.v_offset = new_v
            print(f"[bridge4] cut_surface_z={float(ms_z):.4f} m  "
                  f"v_offset {old_v:.4f} → {new_v:.4f}")
        else:
            print(f"[bridge4] no cut_surface_z available — using bridge3 default "
                  f"v_offset={float(self.calib.v_offset):.4f}")
        # bgvar fix: small fruits (cherry/grape/shine_muscat) need MPM fruit
        # to be seeded at the MS fruit's randomized world position so the
        # MS-mapped MPM knife reliably contacts the MPM fruit. Without this
        # the MPM fruit stays at the yaml canonical (~origin) while MS fruit
        # moves up to 15 cm away → MPM knife misses.
        if not bool(self.cfg.sync_fruit_pose):
            print("[bridge4] forcing sync_fruit_pose=True (bgvar small-fruit fix)")
            self.cfg.sync_fruit_pose = True
        super().setup(env, banana_xy_ms=banana_xy_ms, banana_yaw_ms=banana_yaw_ms)

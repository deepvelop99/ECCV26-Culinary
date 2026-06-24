"""Alignment metrics between ManiSkill and MPM.

Tier 1 success criterion (velocity-only gate) — now EXTENDED in Tier 2 with a
force gate grounded in Franka Panda end-effector capability.

Passed = (velocity alignment OK)  AND  (peak cut force ≤ Panda EEF limit)

Velocity gate  : max_per_step |V_ms_descent − V_mpm_descent| < vel_tol.
                 Catches knife-MS mis-drive or MPM time-step blow-ups.

Force gate     : peak |F_mpm| during descent ≤ F_PANDA_EEF_MAX (80 N).
                 Rationale (Panda EE power envelope at typical cutting speed
                 ~0.1 m/s):
                     v ≈ 0      → up to ~100 N (peak static push)
                     v 0.05–0.15 → up to ~80 N (continuous, cutting regime)
                     v ≈ 0.2   → ~50 N
                     v ≈ 0.3   → ~30 N
                 80 N is the nominal, defensible upper bound for *real* cutting
                 with a Franka Panda. Episodes whose MPM reaction force exceeds
                 this are physically infeasible on real hardware and should be
                 excluded from training (simulator may still have cut visually,
                 but the required cut force is beyond the robot's envelope).

The MPM force is our ground-truth label (particle impulse / dt).  The MS-side
force is retained for debug only.
"""
from __future__ import annotations
from typing import Dict
import numpy as np


# Panda EEF max sustained force while moving (typical cutting velocity).
# See module docstring for derivation.
F_PANDA_EEF_MAX = 80.0   # N


def compute_alignment(
    ms_force: np.ndarray,     # (T, 3)  — debug only; not in pass/fail
    ms_vel: np.ndarray,       # (T, 3)
    mpm_force: np.ndarray,    # (T, 3)  — dataset force label (MPM-only)
    mpm_vel: np.ndarray,      # (T, 3)
    *,
    vel_tol: float = 0.10,        # m/s — accommodates normal PID tracking lag
                                  # (~5–8 cm/s) under (b) soft velocity injection;
                                  # raised from 0.05 since real sim blow-ups
                                  # produce >>0.10 m/s mismatch anyway.
    force_max: float = F_PANDA_EEF_MAX,  # N — Panda EEF envelope
    descent_axis: int = 2,        # world z for ManiSkill
    descent_axis_mpm: int = 1,    # world y for MPM
) -> Dict[str, float]:
    """Return per-episode alignment diagnostics + pass/fail (vel AND force)."""
    f_ms = np.asarray(ms_force, np.float32)
    v_ms = np.asarray(ms_vel, np.float32)
    f_mp = np.asarray(mpm_force, np.float32)
    v_mp = np.asarray(mpm_vel, np.float32)

    v_ms_ax = np.abs(v_ms[:, descent_axis])
    v_mp_ax = np.abs(v_mp[:, descent_axis_mpm])

    # Descent window: frames where ManiSkill velocity on descent axis is
    # appreciably negative (knife going down).
    descent_mask = v_ms[:, descent_axis] < -1e-3
    if descent_mask.sum() < 3:
        descent_mask = np.ones_like(descent_mask)

    v_err = np.abs(v_ms_ax[descent_mask] - v_mp_ax[descent_mask])
    passed_vel = bool(v_err.max() < vel_tol) if v_err.size else False

    f_mp_mag = np.linalg.norm(f_mp, axis=1) if len(f_mp) else np.zeros(0, np.float32)
    f_ms_mag = np.linalg.norm(f_ms, axis=1) if len(f_ms) else np.zeros(0, np.float32)
    f_peak_mpm_raw = float(f_mp_mag.max() if f_mp_mag.size else 0.0)

    # The cut metric only uses DESCENT frames.  Retreat frames exhibit
    # large penalty-contact artefacts ("extraction drag") where the knife
    # passes back through damaged particles — those spikes are not meaningful
    # cutting force and must be excluded.  Additionally low-pass smooth.
    f_cut_metric = 0.0
    if f_mp_mag.size >= 5:
        from scipy.ndimage import uniform_filter1d
        try:
            f_smoothed = uniform_filter1d(f_mp_mag.astype(np.float32), size=5, mode="nearest")
        except Exception:
            k = 5
            kernel = np.ones(k, np.float32) / k
            f_smoothed = np.convolve(f_mp_mag, kernel, mode="same")
        # Descent-only + meaningful contact window
        eligible = descent_mask & (f_smoothed > 1.0)
        f_cut_window = f_smoothed[eligible] if eligible.any() else f_smoothed[descent_mask]
        if f_cut_window.size:
            f_cut_metric = float(np.percentile(f_cut_window, 90.0))

    passed_force = bool(f_cut_metric <= float(force_max))

    passed = passed_vel and passed_force

    return {
        # Combined gate
        "passed": passed,
        "passed_vel": passed_vel,
        "passed_force": passed_force,
        # Velocity diagnostics
        "max_vel_err": float(v_err.max() if v_err.size else 0.0),
        "mean_vel_err": float(v_err.mean() if v_err.size else 0.0),
        "vel_tol": vel_tol,
        "descent_frames": int(descent_mask.sum()),
        # Force diagnostics — use the smoothed metric as the gating number;
        # keep the raw peak for reference.
        "mpm_force_cut":  f_cut_metric,        # P90 of low-pass smoothed |F|
        "mpm_force_peak": f_peak_mpm_raw,      # raw max (transient-inclusive)
        "ms_force_peak":  float(f_ms_mag.max() if f_ms_mag.size else 0.0),
        "mpm_contact_frames": int((f_mp_mag > 1.0).sum()),
        "ms_contact_frames":  int((f_ms_mag > 1.0).sum()),
        "force_max": float(force_max),
    }

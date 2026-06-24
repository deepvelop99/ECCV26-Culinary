
import taichi as ti
from .elastic import deviatoric, frob_norm

@ti.func
def j2_return_mapping(stress_trial: ti.types.matrix(3,3, ti.f32),
                      mu_e: ti.f32, alpha_old: ti.f32,
                      sigma_y0: ti.f32, H_iso: ti.f32,
                      eta_vp: ti.f32):
    """Classic J2 radial return with optional simple Perzyna viscoplasticity."""
    dev, p = deviatoric(stress_trial)
    dev_norm = frob_norm(dev) + 1e-6
    sigma_eq = ti.sqrt(1.5) * dev_norm                     # von Mises equiv. stress
    sigma_y  = sigma_y0 + H_iso * alpha_old                # isotropic hardening

    stress_new = stress_trial
    alpha_inc = ti.f32(0.0)

    if sigma_eq > sigma_y:
        excess = sigma_eq - sigma_y
        if eta_vp > 0.0:
            excess = excess / (1.0 + eta_vp)

        dgamma = excess / (3.0 * mu_e + H_iso)             # plastic multiplier
        r = ti.max(0.0, 1.0 - 3.0 * mu_e * dgamma / sigma_eq)
        dev_new = r * dev
        alpha_inc = ti.sqrt(2.0/3.0) * dgamma
        I = ti.Matrix.identity(ti.f32, 3)
        stress_new = dev_new + p * I                       # preserve hydrostatic

    return stress_new, alpha_inc

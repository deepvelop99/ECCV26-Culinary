
"""Small continuum‑mechanics helpers used inside Taichi kernels."""


import taichi as ti

@ti.func
def deviatoric(A: ti.types.matrix(3, 3, ti.f32)):
    """Return deviatoric part and hydrostatic pressure (trace/3)."""
    tr = A[0, 0] + A[1, 1] + A[2, 2]
    I = ti.Matrix.identity(ti.f32, 3)
    return A - (tr / 3.0) * I, tr / 3.0

@ti.func
def frob_norm(A: ti.types.matrix(3, 3, ti.f32)) -> ti.f32:
    """Frobenius norm of a 3x3 matrix."""
    s = ti.f32(0.0)
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            s += A[i, j] * A[i, j]
    return ti.sqrt(s)

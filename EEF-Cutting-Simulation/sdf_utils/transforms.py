
"""
Small transformation utilities:
- Euler XYZ (intrinsic) to rotation matrix
- Frame convention adapters (X_up / Y_up / Z_up)
- Compose/invert 4x4 SE(3) transforms

Pure helpers are memoized to avoid recomputing identical results.
"""


import numpy as np
from functools import lru_cache
from math import radians, sin, cos

@lru_cache(maxsize=256)

def euler_xyz_to_mat(rx, ry, rz):
    """Convert Euler angles (degrees) to a 3×3 rotation matrix (XYZ intrinsic convention).
    Memoized because identical angles are common across configs.
    """
    # Convert degrees to radians
    rx, ry, rz = radians(rx), radians(ry), radians(rz)
    
    # Compute trigonometric values
    cx, sx = cos(rx), sin(rx)
    cy, sy = cos(ry), sin(ry)
    cz, sz = cos(rz), sin(rz)
    
    # Build rotation matrices for each axis
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    
    # XYZ intrinsic convention: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R

@lru_cache(maxsize=8)

def frame_up_to_world(frame: str):
    """Convert an asset's up-axis to world Y-up (right-handed). Memoized."""
    frame = frame.upper()
    
    # World frame: right-handed, Y-up, Z forward, X right
    if frame == "Y_UP":
        return np.eye(3, dtype=np.float32)
    
    if frame == "Z_UP":
        # Z-up to Y-up: rotate -90° about X
        Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        return Rx
    
    if frame == "X_UP":
        # X-up to Y-up: rotate +90° about Z then +90° about X
        Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        return Rx @ Rz
    
    raise ValueError(f"Unsupported frame '{frame}'. Use Y_UP, Z_UP, or X_UP.")

def compose_transform(scale, R, t):
    """Compose a 4x4 transformation matrix from scale, rotation, and translation."""
    S = np.diag([scale[0], scale[1], scale[2]]).astype(np.float32)
    A = (R @ S).astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = A
    T[:3, 3] = np.array(t, dtype=np.float32)
    return T

def parse_transform(block: dict):
    """Parse transformation parameters from configuration block."""
    frame = block.get("frame", "Y_UP")
    Rf = frame_up_to_world(frame)
    
    euler = block.get("euler_deg_xyz", [0, 0, 0])
    Re = euler_xyz_to_mat(euler[0], euler[1], euler[2])
    R = (Rf @ Re).astype(np.float32)
    
    s = np.array(block.get("scale", [1, 1, 1]), dtype=np.float32)
    t = np.array(block.get("translate", [0, 0, 0]), dtype=np.float32)
    
    return compose_transform(s, R, t)

def invert_transform(T4):
    """Invert a 4x4 transformation matrix."""
    R = T4[:3, :3]
    t = T4[:3, 3]
    Rinv = R.T
    tinv = -Rinv @ t
    
    Out = np.eye(4, dtype=np.float32)
    Out[:3, :3] = Rinv
    Out[:3, 3] = tinv
    return Out

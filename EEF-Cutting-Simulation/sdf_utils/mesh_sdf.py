
"""
Mesh → SDF sampling utilities.
- Loads meshes/scenes via trimesh.
- Applies asset→world transforms (scale/rotate/translate).
- Samples a dense signed distance field on a regular grid.
- Optionally builds a "blade mask" for a knife asset.
"""


import numpy as np
import trimesh
from .transforms import parse_transform, invert_transform

def load_mesh(path:str):
    m = trimesh.load(path, force='mesh')
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(m.dump(concatenate=True))
    return m

def _compute_local_bounds(vertices_local: np.ndarray):
    vmin = vertices_local.min(axis=0).astype(np.float32)
    vmax = vertices_local.max(axis=0).astype(np.float32)
    return vmin, vmax

def _build_blade_mask(X, Y, Z, T_world_to_asset, local_bounds, axis, fraction):
    """
    Create a [Nz,Ny,Nx] float32 mask in [0,1] marking 'blade region' in asset-local coordinates.
    axis: 'X'|'Y'|'Z' (asset local)
    fraction: lower portion along axis considered blade (e.g., 0.5 => lower half)
    """
    # Map grid centers (world) -> asset local
    P = np.c_[X.reshape(-1), Y.reshape(-1), Z.reshape(-1), np.ones((X.size,), np.float32)]
    Pl = (T_world_to_asset @ P.T).T[:, :3]  # (N,3)
    lb, ub = local_bounds
    # Normalize coord to [0,1] along chosen axis
    ax = {'X':0,'Y':1,'Z':2}[axis.upper()]
    t = (Pl[:, ax] - lb[ax]) / max(1e-6, (ub[ax]-lb[ax]))
    w = (t <= float(fraction)).astype(np.float32)
    return w.reshape(X.shape).transpose(2,0,1).astype(np.float32)  # [Nz,Ny,Nx]


# NOTE: Use voxel_size_m to set absolute SDF resolution.
#       If omitted, 'voxel' specifies the number of cells along the longest AABB edge.
def mesh_to_sdf(
    mesh_path: str,
    transform_block: dict,
    voxel: int = None,                 # Legacy: sample longest edge with voxel count
    padding: float = 0.01,
    voxel_size_m: float = None,        # New option: absolute voxel size (meters). Takes priority over voxel
    flip_mode: str = "auto",           # "auto" | "on" | "off"  (SDF sign correction)
    repair: bool = False,              # Attempt simple watertight repair
    knife_blade: dict = None           # Optional: {"axis":"Y","fraction":0.5}
):
    """
    Returns:
      dict {
        "sdf": float32 [Nz, Ny, Nx],      # SDF: inside<0, outside>0
        "origin": float32[3],             # world-space min corner
        "voxel_size": float32,            # cell size (isotropic)
        "T_asset_to_world": float32[4,4], # Applied transformation
        "grid_shape": (Nz, Ny, Nx),       # Convenience
        "blade_mask": float32 [Nz,Ny,Nx]  # Optional, 1=blade, 0=not blade (knife only)
      }
    """
    # 1) Load mesh (+Scene merge) and optional repair
    m_asset = load_mesh(mesh_path)
    if repair:
        try:
            trimesh.repair.fix_normals(m_asset)
            trimesh.repair.fill_holes(m_asset)
            m_asset.remove_degenerate_faces()
            m_asset.remove_duplicate_faces()
        except Exception as e:
            print(f"[mesh_to_sdf] repair warning: {e}")

    # 2) Apply asset->world transformation
    T = parse_transform(transform_block)
    v_asset_h = np.c_[m_asset.vertices, np.ones((len(m_asset.vertices), 1), dtype=np.float32)]
    v_world = (T @ v_asset_h.T).T[:, :3]
    m_world = trimesh.Trimesh(vertices=v_world, faces=m_asset.faces, process=False)

    # 3) Determine bounds (+padding) and voxel size
    bounds = np.array([m_world.bounds[0] - padding, m_world.bounds[1] + padding], dtype=np.float32)
    size = bounds[1] - bounds[0]
    maxlen = float(np.max(size))

    if voxel_size_m is not None:
        vsize = float(voxel_size_m)
    else:
        if voxel is None or voxel <= 0:
            raise ValueError("Either voxel_size_m must be set or voxel (int>0) must be provided.")
        vsize = maxlen / float(voxel)

    Nx = max(int(np.floor(size[0] / vsize)) + 1, 2)
    Ny = max(int(np.floor(size[1] / vsize)) + 1, 2)
    Nz = max(int(np.floor(size[2] / vsize)) + 1, 2)

    # 4) Sample points — origin + k * vsize (optimized)
    xs = bounds[0, 0] + vsize * np.arange(Nx, dtype=np.float32)
    ys = bounds[0, 1] + vsize * np.arange(Ny, dtype=np.float32)
    zs = bounds[0, 2] + vsize * np.arange(Nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='xy')  # (Ny, Nx, Nz)
    
    # Optimized point generation using stack instead of c_
    pts = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)
    
    # 5) SDF sample (trimesh: inside<0, outside>0) - optimized
    d = trimesh.proximity.signed_distance(m_world, pts)
    sdf = d.reshape(Ny, Nx, Nz).transpose(2, 0, 1).astype(np.float32)  # [Nz,Ny,Nx]
    origin = bounds[0].astype(np.float32)

    # 6) Auto-correct sign to inside<0
    if flip_mode not in ("auto", "on", "off"):
        raise ValueError("flip_mode must be 'auto'|'on'|'off'")
    neg_ratio = float((sdf < 0).mean())
    do_flip = (flip_mode == "on") or (flip_mode == "auto" and neg_ratio > 0.5)
    if do_flip:
        print(f"[mesh_to_sdf] outside-negative detected or forced (neg_ratio={neg_ratio:.3f}) → flip sign")
        sdf = -sdf

    pack = {
        "sdf": sdf,
        "origin": origin,
        "voxel_size": np.float32(vsize),
        "T_asset_to_world": T.astype(np.float32),
        "grid_shape": (int(Nz), int(Ny), int(Nx)),
    }

    # 7) Optional blade mask (knife only)

    # 7.5) Always store asset-local bounds for downstream EE planning
    try:
        lb_all, ub_all = _compute_local_bounds(np.asarray(m_asset.vertices, np.float32))
        pack["asset_bounds_local_lb"] = lb_all.astype(np.float32)
        pack["asset_bounds_local_ub"] = ub_all.astype(np.float32)
    except Exception as e:
        print(f"[mesh_to_sdf] bounds metadata warning: {e}")
    if knife_blade is not None and isinstance(knife_blade, dict):
        axis = str(knife_blade.get("axis", "Y")).upper()
        fraction = float(knife_blade.get("fraction", 0.5))
        if axis not in ("X","Y","Z"):
            raise ValueError("knife_blade.axis must be one of X|Y|Z")
        Tl = invert_transform(T)  # world -> asset(local)
        # Compute local bounds from asset (untransformed) vertices
        lb, ub = _compute_local_bounds(np.asarray(m_asset.vertices, np.float32))
        blade_mask = _build_blade_mask(X, Y, Z, Tl, (lb, ub), axis, fraction)
        pack["blade_mask"] = blade_mask
        pack["blade_axis"] = axis
        pack["blade_fraction"] = np.float32(fraction)

    return pack

def save_sdf_npz(npz_path, sdf_pack:dict):
    np.savez_compressed(npz_path, **sdf_pack)

def load_sdf_npz(npz_path):
    pack = np.load(npz_path)
    return {k: pack[k] for k in pack.files}

def trilinear_sample(sdf_grid, origin, voxel_size, xyz):
    """
    Optimized trilinear sampling for SDF grid.
    sdf_grid [Nz,Ny,Nx]
    xyz [..., 3] in world coords
    returns sdf [...]
    """
    g = sdf_grid
    o = origin
    v = voxel_size
    
    # Optimized coordinate transformation
    q = (xyz - o) / v
    xi, yi, zi = q[...,0], q[...,1], q[...,2]
    
    # Floor and ceiling indices
    x0 = np.floor(xi).astype(np.int32)
    y0 = np.floor(yi).astype(np.int32)
    z0 = np.floor(zi).astype(np.int32)
    x1, y1, z1 = x0+1, y0+1, z0+1
    
    # Optimized clamping function
    def clamp(a, lo, hi): 
        return np.minimum(np.maximum(a, lo), hi)
    
    # Grid dimensions
    Nx, Ny, Nz = g.shape[2], g.shape[1], g.shape[0]
    
    # Clamp indices to grid bounds
    x0 = clamp(x0, 0, Nx-1); x1 = clamp(x1, 0, Nx-1)
    y0 = clamp(y0, 0, Ny-1); y1 = clamp(y1, 0, Ny-1)
    z0 = clamp(z0, 0, Nz-1); z1 = clamp(z1, 0, Nz-1)
    
    # Interpolation factors
    xd = xi - x0; yd = yi - y0; zd = zi - z0
    
    # Precompute inverse factors for optimization
    xd_inv = 1.0 - xd; yd_inv = 1.0 - yd; zd_inv = 1.0 - zd
    
    # Sample grid values - optimized memory access
    c000 = g[z0, y0, x0]; c100 = g[z0, y0, x1]
    c010 = g[z0, y1, x0]; c110 = g[z0, y1, x1]
    c001 = g[z1, y0, x0]; c101 = g[z1, y0, x1]
    c011 = g[z1, y1, x0]; c111 = g[z1, y1, x1]
    
    # Optimized trilinear interpolation
    c00 = c000*xd_inv + c100*xd
    c01 = c001*xd_inv + c101*xd
    c10 = c010*xd_inv + c110*xd
    c11 = c011*xd_inv + c111*xd
    c0 = c00*yd_inv + c10*yd
    c1 = c01*yd_inv + c11*yd
    c = c0*zd_inv + c1*zd
    
    return c

def finite_diff_grad(sdf_grid, origin, voxel_size, xyz, eps=None):
    if eps is None:
        eps = 0.5 * float(voxel_size)
    xyz = np.asarray(xyz, dtype=np.float32)
    e = np.array([eps,0,0], np.float32)
    dx = (trilinear_sample(sdf_grid, origin, voxel_size, xyz+e) - trilinear_sample(sdf_grid, origin, voxel_size, xyz-e)) / (2*eps)
    e = np.array([0,eps,0], np.float32)
    dy = (trilinear_sample(sdf_grid, origin, voxel_size, xyz+e) - trilinear_sample(sdf_grid, origin, voxel_size, xyz-e)) / (2*eps)
    e = np.array([0,0,eps], np.float32)
    dz = (trilinear_sample(sdf_grid, origin, voxel_size, xyz+e) - trilinear_sample(sdf_grid, origin, voxel_size, xyz-e)) / (2*eps)
    g = np.array([dx,dy,dz], np.float32)
    n = np.linalg.norm(g)+1e-6
    return g / n

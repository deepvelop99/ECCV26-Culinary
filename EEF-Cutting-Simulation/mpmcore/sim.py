
"""
MLS‑MPM cutting core (physics + interactions).
Responsibilities:
  - World/grid initialization
  - Particle seeding from SDF
  - P2G/G2P transfers with corotated elasticity and optional J2 plasticity
  - Knife/board contact and cutting logic
  - Multicut scheduler and EEF tracking
"""


import math
import json
import os
import numpy as np
import taichi as ti
from functools import lru_cache

from mpmcore.colliders import KnifeCollider, BoardCollider, MultiCollider
from physics.plastic import j2_return_mapping
from render_utils.renderer import should_force_full_recolor, MPMRenderer
from io_utils.output_manager import OutputManager, FPSExporter, JSONLogger

def _pick_arch():
    """Select the best available Taichi backend in order of preference."""
    # Try CUDA first (best performance for most cases)
    try:
        if getattr(ti, "cuda", False) and ti.cuda:
            return ti.cuda
    except Exception:
        pass
    
    # Try Vulkan as fallback
    try:
        if getattr(ti, "vulkan", False) and ti.vulkan:
            return ti.vulkan
    except Exception:
        pass
    
    # Fallback to CPU
    return ti.cpu

_TI_INIT = False

def _ensure_ti():
    """Ensure Taichi is initialized with the best available backend."""
    global _TI_INIT
    if not _TI_INIT:
        ti.init(arch=_pick_arch())
        _TI_INIT = True

@ti.dataclass
class Particle:
    x: ti.types.vector(3, ti.f32)
    v: ti.types.vector(3, ti.f32)
    F: ti.types.matrix(3,3, ti.f32)
    C: ti.types.matrix(3,3, ti.f32)
    Jp: ti.f32
    alpha: ti.f32
    D:  ti.f32
    mass: ti.f32
    vol:  ti.f32
    mu0:  ti.f32
    la0:  ti.f32

@lru_cache(maxsize=256)
def _center_out_order(n: int):
    """Return indices 0..n-1 reordered center-out. Handles even/odd n correctly."""
    if n <= 0:
        return []
    
    order = []
    
    if n % 2 == 1:
        # Odd n: start from center
        center = n // 2
        order.append(center)
        left, right = center - 1, center + 1
    else:
        # Even n: start from two center elements
        left, right = n // 2 - 1, n // 2
    
    # Expand outward from center
    while left >= 0 or right < n:
        if right < n:
            order.append(right)
            right += 1
        if left >= 0:
            order.append(left)
            left -= 1
    
    return order

@ti.data_oriented
class MPMCuttingSim:
    @ti.func
    def bspline_quadratic_weights(self, fx: ti.types.vector(3, ti.f32)):
        """Quadratic B‑spline weights {wx, wy, wz} for a particle's fractional offset fx in grid space."""
        wx = ti.Vector([ti.f32(0.5)*(ti.f32(1.5)-fx[0])**2,
                        ti.f32(0.75)-(fx[0]-ti.f32(1.0))**2,
                        ti.f32(0.5)*(fx[0]-ti.f32(0.5))**2])
        wy = ti.Vector([ti.f32(0.5)*(ti.f32(1.5)-fx[1])**2,
                        ti.f32(0.75)-(fx[1]-ti.f32(1.0))**2,
                        ti.f32(0.5)*(fx[1]-ti.f32(0.5))**2])
        wz = ti.Vector([ti.f32(0.5)*(ti.f32(1.5)-fx[2])**2,
                        ti.f32(0.75)-(fx[2]-ti.f32(1.0))**2,
                        ti.f32(0.5)*(fx[2]-ti.f32(0.5))**2])
        return wx, wy, wz




    """
    Single-responsibility core: MLS-MPM physics + cutting interactions.
    Colliders are injected (DIP). Knife cuts (optionally limited to blade mask); board collides only.
    """
    def __init__(self, cfg, cutting_mesh_pack, knife_pack, board_pack=None,
                 viewer=True, viewer_radius_scale=1.0,
                 viewer_camera_mode="manual", viewer_initial_pose=None, viewer_lock_on_run=True):
        _ensure_ti()
        self.cfg = cfg
        # Physics clock (advances by dt≈2e-5 s)
        self.sim_time = 0.0
        # Visual clock for saw‐cut (advances by ~1/60 s per frame)
        self.saw_time = 0.0

        # ----- world/grid -----
        wcfg = cfg["world"]
        self.n_grid   = int(wcfg["grid_resolution"])
        self.substeps = int(wcfg["substeps"])
        self.dt       = float(wcfg["dt"])
        self.output_fps = float(wcfg.get("output_fps", 24.0))

        self.bounds_min = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.bounds_max = ti.Vector.field(3, dtype=ti.f32, shape=())
        bmin_py = np.array(wcfg["bounds_min"], dtype=np.float32)
        bmax_py = np.array(wcfg["bounds_max"], dtype=np.float32)
        self.bounds_min[None] = ti.Vector(bmin_py.tolist())
        self.bounds_max[None] = ti.Vector(bmax_py.tolist())

        dx_scalar = float(np.max(bmax_py - bmin_py)) / float(self.n_grid)
        self.dx_s     = ti.field(dtype=ti.f32, shape=())
        self.inv_dx_s = ti.field(dtype=ti.f32, shape=())
        self.dx_s[None]     = dx_scalar
        self.inv_dx_s[None] = 1.0 / dx_scalar
        
        # Resolution normalization (reference: grid 48)
        self.dx_ref_s = ti.field(dtype=ti.f32, shape=())
        dx_ref = float(np.max(bmax_py - bmin_py)) / float(48)
        self.dx_ref_s[None] = dx_ref
        self.dx_ratio_s = ti.field(dtype=ti.f32, shape=())
        self.dx_ratio_s[None] = self.dx_s[None] / self.dx_ref_s[None]
        # Resolution scaling control (positive exponent => stronger at higher grid resolution)
        self.reso_exp_f = ti.field(dtype=ti.f32, shape=())
        self.reso_exp_f[None] = float(wcfg.get("reso_exponent", 0.5))  # default: 0.5 (gentle strengthening at higher N)
        
        self._dx_py = dx_scalar
        self._bmin_py = bmin_py
        self._bmax_py = bmax_py

        g = wcfg.get("gravity", [0.0, -9.81, 0.0])
        self.gravity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.gravity[None] = ti.Vector(list(map(float, g)))

        # ----- material (cutting mesh) -----
        bc = cfg["cutting_mesh"]
        ecfg = bc.get("elasticity", {})
        E  = float(ecfg.get("youngs_modulus", bc.get("youngs_modulus", bc.get("E", 3e5))))
        nu = float(ecfg.get("poisson_ratio",  bc.get("poisson_ratio",  bc.get("nu", 0.45))))
        self.mu0 = E / (2*(1+nu))
        self.la0 = E * nu / ((1+nu)*(1-2*nu) + 1e-6)  # Improved numerical precision
        self.grid_damping     = min(float(bc.get("grid_damping_ratio", bc.get("damping", 0.03))), 0.2)
        self.particle_damping = min(float(bc.get("particle_damping", 0.03)), 0.2)
        self.mesh_restitution = float(bc.get("restitution", 0.01))  # Mesh restitution coefficient

        # stability: CFL with improved safety margin
        rho_cfl = float(bc.get("density", 400.0))
        c_wave  = math.sqrt(max(1e-6, (self.la0 + 2.0*self.mu0) / rho_cfl))
        dt_cfl  = 0.2 * self._dx_py / max(1e-6, c_wave)  # Reduced from 0.3 to 0.2 for better stability
        if self.dt > dt_cfl:
            self.dt = float(dt_cfl)
            print(f"Warning: dt reduced to {self.dt:.2e} for CFL stability (c_wave={c_wave:.2f})")

        n = self.n_grid
        self.grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(n,n,n))
        self.grid_m = ti.field(dtype=ti.f32, shape=(n,n,n))
        self.particles = None
        self.pcount = ti.field(dtype=ti.i32, shape=())
        self.max_particles = 600000

        # ----- colliders (injected) -----
        # Keep packs for rendering / planning
        self.knife_pack = knife_pack
        self.board_pack = board_pack

        # Knife collider + precomputed base voxel centers for rendering
        knife_sdf = knife_pack["sdf"]
        voxel     = float(knife_pack["voxel_size"])
        origin    = np.array(knife_pack["origin"], dtype=np.float32)

        neg_idx = np.argwhere(knife_sdf < 0.0).astype(np.int32)  # (M,3) = (z,y,x)
        self._knife_neg_idx = neg_idx
        self._knife_voxel   = voxel
        self._knife_origin  = origin

        stride = int(cfg.get("viewer", {}).get("knife_render_stride", 4))
        base = (self._knife_neg_idx[::max(1,stride)][:, [2,1,0]].astype(np.float32) + 0.5) * voxel + origin[None, :]
        self._knife_base_xyz = base  # (M,3)

        ymin_idx = int(neg_idx[:, 1].min())
        self._knife_yfoot = float((ymin_idx + 0.5) * voxel)

        motion = cfg["knife"]["motion"]
        # ── Saw-cut configuration ─────────────────────────────────────────
        cut_mode = motion.get("cut_mode", "cut")
        saw_cfg = motion.get("saw", {})
        saw_amplitude = float(saw_cfg.get("amplitude_m", 0.0))
        saw_frequency = float(saw_cfg.get("frequency_hz", 0.0))
        saw_axis = saw_cfg.get("axis", "x").lower()

        start_y_tip = float(motion.get("knife_start_y", 0.30)) - self._knife_yfoot
        stop_y_tip  = float(motion.get("knife_stop_y",  0.00)) - self._knife_yfoot



        blade_mask = knife_pack.get("blade_mask", None)
        
        # Determine board_shape: use board's shape if available, otherwise dummy shape
        board_shape = (1, 1, 1)  # Default dummy shape
        if board_pack is not None:
            board_shape = board_pack["sdf"].shape
        
        knife = KnifeCollider(
            sdf_grid=knife_sdf, origin=knife_pack["origin"], voxel_size=voxel,
            start_y=start_y_tip, stop_y=stop_y_tip,
            speed=float(motion.get("cutting_speed_mps", 0.6)),
            return_speed=float(motion.get("return_speed_mps", 0.6)),
            z_offset=float(motion.get("z_offset", 0.0)),
            friction=float(cfg["knife"].get("friction", 0.3)),
            blade_mask_grid=blade_mask,
            mesh_restitution=float(bc.get("restitution", 0.01)),
            board_shape=board_shape,
            # saw-cut parameters:
            cut_mode=cut_mode,
            saw_amplitude=saw_amplitude,
            saw_frequency=saw_frequency,
            saw_axis=saw_axis,
        )
        self.knife = knife
        # ---- speed‑resistance & floor (from YAML; defaults keep current behavior) ----
        kcfg = cfg.get("knife", {})
        sres = kcfg.get("speed_resistance", {})
        c_scale = float(sres.get("c_norm_scale", 0.35))                # c_norm = c_scale * dx/dt
        self.knife.c_norm_f[None] = c_scale * float(self.dx_s[None]) / max(1e-6, float(self.dt))
        
        # ---- material-dependent k2 scaling (no YAML key expansion) ----
        k2_base = float(sres.get("k_quad_per_s", 4.0))   # [1/s] baseline from YAML
        # current material properties
        E_cur = float(ecfg.get("youngs_modulus", 3.0e5)) # Pa
        sigma_y_cur = float(self.cfg.get("plasticity", {}).get("yield_stress_kpa", 0.0)) * 1e3  # Pa
        # soft-food friendly baselines (tunable constants)
        E_ref = 3.0e5     # Pa
        sigma_ref = 5.0e3 # Pa (5 kPa)
        # exponents (kept in code to avoid YAML expansion)
        alpha_E = 0.5
        beta_sig = 1.0
        # guard against zeros
        E_ratio = max(E_cur, 1e-6) / E_ref
        sig_ratio = max(sigma_y_cur, 1.0) / sigma_ref
        mat_gain = (E_ratio ** alpha_E) * (sig_ratio ** beta_sig)
        k2_eff = k2_base * mat_gain
        # apply to knife resistance model
        self.knife.res_kq_per_s = float(k2_eff)          # KnifeCollider.update() uses this k2
        
        # ---- keep speed floor as-is (from YAML) ----
        floor_ratio = float(kcfg.get("min_speed_floor_ratio", 0.25))
        self.knife.min_speed_f[None] = floor_ratio * float(self.knife.base_speed)
        
        # ---- stash telemetry for JSON/debug ----
        self._k2_base = float(k2_base)
        self._k2_eff = float(k2_eff)
        self._k2_mat_gain = float(mat_gain)
        self._E_cur = float(E_cur)
        self._sigma_y_cur = float(sigma_y_cur)
        self._E_ref = float(E_ref)
        self._sigma_ref = float(sigma_ref)

        self.board = None
        if board_pack is not None:
            bcfg = cfg.get("board", {})
            origin_b = np.array(board_pack["origin"], dtype=np.float32)
            render_offset_y = float(board_pack.get("render_offset_y", 0.0))
            origin_b[1] += render_offset_y  # render offset baked into physics
            print(f"[info] Board origin before offset: {board_pack['origin'][1]:.6f}")
            print(f"[info] Board render offset Y: {render_offset_y:.6f}")
            print(f"[info] Board origin after offset: {origin_b[1]:.6f}")
            self.board = BoardCollider(
                sdf_grid=board_pack["sdf"], origin=origin_b, voxel_size=float(board_pack["voxel_size"]),
                friction=float(bcfg.get("friction", 0.5)),
                restitution=float(bcfg.get("restitution", 0.0)),
            )
            print(f"[info] Board initialized successfully")
        else:
            # Create dummy board for collision functions (no collision)
            dummy_sdf = np.full((1, 1, 1), 1e6, dtype=np.float32)  # Always positive (no collision)
            self.board = BoardCollider(
                sdf_grid=dummy_sdf, origin=[0, 0, 0], voxel_size=1.0,
                friction=0.0, restitution=0.0
            )
            print(f"[INFO] Dummy board created (no collision)")
        self.colliders = MultiCollider(knife=self.knife, board=self.board) if self.board is not None else MultiCollider(knife=self.knife)
        
        # Call attach_board only when using board for physics collision
        # Do not call when only enabling board for rendering (keep has_board_flag=0)
        if board_pack is not None:
            try:
                self.knife.attach_board(self.board)
                print(f"[knife] Board physics attached successfully")
            except Exception as e:
                print(f"[knife] attach_board failed: {e}")
        else:
            print(f"[knife] Board physics disabled (render only)")

        # ----- viewer -----
        self.viewer_enabled = bool(viewer)
        self.viewer_radius_scale = float(viewer_radius_scale)
        self.viewer_camera_mode = str(viewer_camera_mode).lower()
        self.viewer_lock_on_run = bool(viewer_lock_on_run)
        viewer_cfg = cfg.get("viewer", {})
        self.particle_render_radius = float(viewer_cfg.get("particle_radius", 0.0015))
        self.show_board = bool(viewer_cfg.get("show_board", False))
        self.show_grid = bool(viewer_cfg.get("show_grid", False))
        self.viewer_initial_pose = viewer_initial_pose or {
            "eye":    [0.30, 0.22, 0.30],
            "lookat": [0.00, 0.10, 0.00],
            "up":     [0.00, 1.00, 0.00],
            "fov":    35.0,
        }
        self.color_update_every = int(viewer_cfg.get("color_update_every", 4))
        self._last_color_update_frame = -1
        self._force_full_recolor = 1  # recolor on first frame
        self._last_z_off = float(self.knife.z_off[None])
        self._dwell_completed = False  # Track if dwell is completed for current cut


        if self.viewer_enabled:
            self.renderer = MPMRenderer(self)
            self.renderer._init_viewer()
            self.renderer._apply_initial_camera()

        # ----- Multi-cut planning (Z-sweep) -----
        cutcfg = cfg.get("cutting", {})
        # Resolution exponent controlling how tip forces scale with grid resolution
        self.tip_reso_exp_f = ti.field(dtype=ti.f32, shape=()); self.tip_reso_exp_f[None] = float(cutcfg.get("tip_reso_exponent", float(self.reso_exp_f[None])))
        self.num_cuts      = int(cutcfg.get("num_cuts", cutcfg.get("slices", 1)))
        self.z_margin_ratio= float(cutcfg.get("z_margin_ratio", 0.05))
        self.cut_order     = str(cutcfg.get("cut_order", "left-to-right")).lower()
        self.dwell_frames  = int(cutcfg.get("dwell_frames", 6))
        #self.multi_cutting_enabled = (self.num_cuts is not None and self.num_cuts >= 1)
        self.multi_cutting_enabled = True
        self._multi_init_done = False
        self._dwell_counter = 0
        self.current_cut_idx = 0
        self.cutting_complete = False
        self._slice_hit_bottom = False  # Have we seen the blade hit the board?
        self.z_positions = None
        self._z_offsets = None
        try:
            if cutting_mesh_pack is not None and self.multi_cutting_enabled:
                b_sdf = np.asarray(cutting_mesh_pack["sdf"], np.float32)
                b_org = np.asarray(cutting_mesh_pack["origin"], np.float32)
                b_vox = float(cutting_mesh_pack["voxel_size"])
                solid = (b_sdf < 0.0)
                if np.any(solid):
                    zs = np.where(solid)[0]
                    zmin_idx = int(zs.min()); zmax_idx = int(zs.max())
                    zmin = float(b_org[2] + (zmin_idx + 0.5) * b_vox)
                    zmax = float(b_org[2] + (zmax_idx + 0.5) * b_vox)
                    rng = max(1e-4, (zmax - zmin))
                    zmin_i = zmin + self.z_margin_ratio * rng
                    zmax_i = zmax - self.z_margin_ratio * rng
                    zmin_i, zmax_i = min(zmin_i, zmax_i), max(zmin_i, zmax_i)
                    if self.num_cuts <= 1:
                        z_world = [0.0]
                    else:
                        z_world = list(np.linspace(zmin_i, zmax_i, int(self.num_cuts), dtype=np.float32))

                    # --- order ---
                    if self.cut_order == "center-out":
                        order = _center_out_order(len(z_world))
                        z_world = [z_world[j] for j in order]
                    elif self.cut_order in ("left-to-right","ltr"):
                        z_world = list(reversed(z_world))
                    elif self.cut_order in ("right-to-left","rtl"):
                        pass
                    else:
                        print(f"[WARN] Unknown cut_order '{self.cut_order}', using left-to-right")

                    self.z_positions = z_world

                    # Offsets relative to knife center Z
                    if self._knife_base_xyz is not None:
                        k_center_z = float(np.mean(self._knife_base_xyz[:,2]))
                    else:
                        k_center_z = float(self._knife_origin[2])
                    self._z_offsets = [float(z - k_center_z) for z in self.z_positions]

                    print(f"[PLAN] Multi-cut: N={self.num_cuts}, Z-world [{zmin:.4f}, {zmax:.4f}] → positions={list(map(lambda x: round(x,4), self.z_positions))}")
                    print(f"[PLAN] Knife center Z: {k_center_z:.4f} m")
                    print(f"[PLAN] Z offsets: {list(map(lambda x: round(x,4), self._z_offsets))}")
        except Exception as e:
            print(f"[WARN] Multi-cut plan failed: {e}")

        self._frame = 0
        
        # ----- EEF tracking variables -----
        self._ee_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._ee_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._ee_prev_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._fps = 0.0
        self._knife_applies_force = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # FPS tracking
        import time
        self._fps_start_time = time.time()
        self._fps_frame_count = 0

        # --- substep accumulators for fixed-FPS averaging (blade-only for EEF) ---
        self._J_blade_accum = np.zeros(3, np.float32)
        self._J_handle_accum = np.zeros(3, np.float32)
        self._J_board_accum = np.zeros(3, np.float32)
        self._accum_time = 0.0

        # ----- end-effector (EE) base from knife pack -----
        self._ee_available = False
        try:
            kp = knife_pack
            Tkw = np.asarray(kp.get("T_asset_to_world"), np.float32)
            lb = np.asarray(kp.get("asset_bounds_local_lb"), np.float32)
            ub = np.asarray(kp.get("asset_bounds_local_ub"), np.float32)
            axis = str(kp.get("blade_axis", "Y")).upper() if "blade_axis" in kp else str(cfg["knife"].get("blade", {}).get("axis", "Y")).upper()
            frac = float(kp.get("blade_fraction", cfg["knife"].get("blade", {}).get("fraction", 0.5)))
            axis_to_idx = {"X":0,"Y":1,"Z":2}
            a = axis_to_idx.get(axis, 1)
            # Local EE: at the blade boundary plane, centered in the other axes
            p_local = np.array([0.0,0.0,0.0], np.float32)
            for d in range(3):
                if d == a:
                    p_local[d] = lb[d] + frac * max(1e-6, (ub[d] - lb[d]))
                else:
                    p_local[d] = 0.5 * (lb[d] + ub[d])
            # World base (before runtime Y/Z animation)
            p_h = np.array([p_local[0], p_local[1], p_local[2], 1.0], np.float32)
            p_world0 = (Tkw @ p_h)[:3]
            self._ee_base_world = p_world0.astype(np.float32)
            # Axis direction in world (unit)
            Rkw = Tkw[:3,:3]
            e_axis = np.zeros(3, np.float32); e_axis[a] = 1.0
            n_world = Rkw @ e_axis
            n_world = n_world / (np.linalg.norm(n_world) + 1e-6)
            self._ee_axis_world = n_world.astype(np.float32)
            self._ee_axis_name = axis
            self._ee_fraction = frac
            self._ee_available = True
        except Exception as e:
            print(f"[EE] unavailable: {e}")

        # ----- contact impulse accumulators (grid space) -----
        self.knife_impulse_g = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.board_impulse_g = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.knife_impulse_blade_g = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.knife_impulse_handle_g = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.knife_impulse_g[None] = ti.Vector([0.0, 0.0, 0.0])
        self.board_impulse_g[None] = ti.Vector([0.0, 0.0, 0.0])
        self.knife_impulse_blade_g[None] = ti.Vector([0.0, 0.0, 0.0])
        self.knife_impulse_handle_g[None] = ti.Vector([0.0, 0.0, 0.0])
        
        # ----- collision detection -----
        self.knife_board_collision_detected = ti.field(dtype=ti.i32, shape=())
        self.knife_mesh_collision_detected = ti.field(dtype=ti.i32, shape=())
        self.knife_board_collision_detected[None] = 0
        self.knife_mesh_collision_detected[None] = 0
        self.knife_particle_collision_logged = False  # Track if knife-particle collision was already logged
        


        # ----- logging (EE + force) -----
        # Use unified output logging config
        out_cfg = cfg.get("output", {})
        log_cfg = out_cfg.get("logging", {})
        # Check both output.enabled and output.logging.enabled
        self.log_enabled = bool(out_cfg.get("enabled", True)) and bool(log_cfg.get("enabled", False))
        self.log_dir = str(log_cfg.get("out_dir", "logs"))
        
        # ----- unified output (logging + export) -----
        # If not specified, fall back to world.output_fps
        out_fps = float(out_cfg.get("fps", self.output_fps))
        log_cfg2 = log_cfg  # Use the same config
        exp_cfg2 = out_cfg.get("export", {})

        self._out_manager = None
        if bool(out_cfg.get("enabled", True)):
            # Prepare logger
            logger = None
            if bool(log_cfg2.get("enabled", False)) or self.log_enabled:
                # prefer unified config, else fall back to legacy ee_force_json
                log_dir = str(log_cfg2.get("out_dir", self.log_dir if self.log_enabled else "logs"))
                logger = JSONLogger(out_dir=log_dir)
            # Prepare exporter
            exporter = None
            if bool(exp_cfg2.get("enabled", False)) or bool(cfg.get("export", {}).get("enabled", False)):
                exp_dir = str(exp_cfg2.get("out_dir", cfg.get("export", {}).get("out_dir", "exports")))
                exp_fmt = str(exp_cfg2.get("format", cfg.get("export", {}).get("format", "npz")))
                exporter = FPSExporter(out_dir=exp_dir, fmt=exp_fmt, with_color=True)
            self._out_manager = OutputManager(fps=out_fps, exporter=exporter, logger=logger)

        # Normalization scales for logging (resolution-independent)
        self.dx_units = self.dx_s[None]
        self.dt_units = self.dt
        self.E_units  = float(self.mu0*2.0*(1.0+self.la0/(self.la0+2.0*self.mu0)+1e-6))  # crude back-out; we will store E as cfg too
        self._E_nominal = float(cfg["cutting_mesh"].get("elasticity", {}).get("youngs_modulus", cfg["cutting_mesh"].get("youngs_modulus", cfg["cutting_mesh"].get("E", 3e5))))
        if self._E_nominal > 0.0:
            self.E_units = float(self._E_nominal)
        if self.log_enabled:
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"[LOG] EE/force JSON logging enabled → dir='{self.log_dir}'")

        # ----- parameters -----
        self.pushout_eps = ti.field(dtype=ti.f32, shape=()); self.pushout_eps[None] = 0.02 * self._dx_py
        self.board_top_y = ti.field(dtype=ti.f32, shape=()); self.board_top_y[None] = 0.0
        self.mesh_restitution_f = ti.field(dtype=ti.f32, shape=()); self.mesh_restitution_f[None] = self.mesh_restitution

        dmg = cfg.get("damage", {})
        # Resolution-aware bands: prefer explicit meters or voxel-multiples; else scale old meters by dx_ratio
        band_m = dmg.get("damage_band_m", None)
        band_vox = dmg.get("damage_band_vox", None)
        if band_m is not None:
            damage_band_m = float(band_m)
        elif band_vox is not None:
            damage_band_m = float(band_vox) * float(self.dx_s[None])
        else:
            damage_band_m = float(dmg.get("damage_band", 0.0012)) * float(self.dx_ratio_s[None])
        # Apply resolution-based gain so cutting band grows as grid resolution increases
        damage_band_m = float(damage_band_m) * (float(self.dx_ref_s[None]) / max(1e-12, float(self.dx_s[None]))) ** float(self.reso_exp_f[None])
        self.damage_v_threshold_f = ti.field(dtype=ti.f32, shape=()); 
        # Prefer dimensionless velocity threshold v_hat = v * dt / dx (CFL-like)
        v_hat = float(dmg.get("damage_v_hat", 0.30))  # default tuned at 48
        self.damage_v_hat_f = ti.field(dtype=ti.f32, shape=()); self.damage_v_hat_f[None] = v_hat
        self.damage_v_threshold_f[None] = float(self.dx_s[None]) * v_hat / max(1e-6, float(self.dt))
        self.damage_band_f        = ti.field(dtype=ti.f32, shape=()); self.damage_band_f[None] = float(damage_band_m)
        self.damage_visualization = bool(dmg.get("damage_visualization", True))

        pcfg = cfg.get("plasticity", {})
        
        # --- Force-gated cutting: min normalized contact ĉ threshold ---
        # Auto from material if YAML not provided: grows with toughness (σy) and stiffness (E)
        try:
            E_cur = float(ecfg.get("youngs_modulus", E))
        except Exception:
            E_cur = E
        sigma_y_cur = float(pcfg.get("yield_stress_kpa", 0.0)) * 1e3  # Pa
        c_auto = 0.09 * (max(1e-6, sigma_y_cur) / 5e3)**0.6 * (max(1e-6, E_cur) / 3e5)**0.2
        c_auto = max(0.05, min(0.60, c_auto))  # clamp to [0.05, 0.60]
        self.c_hat_cut_min_f = ti.field(dtype=ti.f32, shape=())
        self.c_hat_cut_min_f[None] = float(dmg.get("c_hat_cut_min", c_auto))

        self.plastic_enabled = ti.field(dtype=ti.i32, shape=()); self.plastic_enabled[None] = 1 if pcfg.get("enabled", False) else 0
        self.plastic_model   = (pcfg.get("model", "off") or "off").lower()
        self.sigma_y0_f = ti.field(dtype=ti.f32, shape=()); self.sigma_y0_f[None] = float(pcfg.get("yield_stress_kpa", 0.0)) * 1e3
        self.H_iso_f    = ti.field(dtype=ti.f32, shape=()); self.H_iso_f[None]    = float(pcfg.get("hardening_kpa", 0.0)) * 1e3
        self.eta_vp_f   = ti.field(dtype=ti.f32, shape=()); self.eta_vp_f[None]   = float(pcfg.get("viscoplastic_gamma", 0.0))

        # Cutting/tipping control (symmetric by sign(dknife) to avoid bias)
        cutcfg = cfg.get("cutting", {})
        self.tip_force_gain_f = ti.field(dtype=ti.f32, shape=()); self.tip_force_gain_f[None] = float(cutcfg.get("tip_force_gain", 2.0))
        self.tip_band_f       = ti.field(dtype=ti.f32, shape=()); self.tip_band_f[None]       = float(cutcfg.get("tip_band", float(self.damage_band_f[None]) * 1.5))
        self.no_stiffness_drop = ti.field(dtype=ti.i32, shape=()); self.no_stiffness_drop[None] = 1 if bool(cutcfg.get("no_stiffness_drop", True)) else 0
        # Post-cut tipping controls
        self.tip_post_gain_f = ti.field(dtype=ti.f32, shape=()); self.tip_post_gain_f[None] = float(cutcfg.get("tip_post_gain", 0.0))
        self.tip_post_mul_f  = ti.field(dtype=ti.f32, shape=()); self.tip_post_mul_f[None]  = float(cutcfg.get("tip_post_band_mul", 4.0))
        self.tip_post_above_only = ti.field(dtype=ti.i32, shape=()); self.tip_post_above_only[None] = 1 if bool(cutcfg.get("tip_post_above_board_only", True)) else 0

        # ----- stability monitoring -----
        stcfg = cfg.get("stability", {})
        self.stab_report_every = int(stcfg.get("report_every", 0))
        self.stab_J_lo = float(stcfg.get("J_warn_min", 0.2))
        self.stab_J_hi = float(stcfg.get("J_warn_max", 5.0))
        self.max_speed_f = ti.field(dtype=ti.f32, shape=()); self.max_speed_f[None] = 0.0
        self.min_J_f     = ti.field(dtype=ti.f32, shape=()); self.min_J_f[None]     = 1e9
        self.max_J_f     = ti.field(dtype=ti.f32, shape=()); self.max_J_f[None]     = -1e9
        self.speed_cap_f = ti.field(dtype=ti.f32, shape=()); self.speed_cap_f[None] = float(stcfg.get("max_speed_cap", 4.0))  
        # Resolution-aware grid speed cap factor (v_max_grid = factor * dx/dt)
        self.grid_speed_cap_factor_f = ti.field(dtype=ti.f32, shape=()); self.grid_speed_cap_factor_f[None] = float(stcfg.get("grid_speed_cap_factor", 0.45))  

        self.J_clamp_lo_f = ti.field(dtype=ti.f32, shape=()); self.J_clamp_lo_f[None] = float(stcfg.get("J_clamp_min", 0.3))
        self.J_clamp_hi_f = ti.field(dtype=ti.f32, shape=()); self.J_clamp_hi_f[None] = float(stcfg.get("J_clamp_max", 3.0))

        # cache board top world y
        if self.board is not None:
            self.board_top_y[None] = float(self.board.solid_ymax_world)

        # Store board existence for Taichi kernels
        self.has_board = ti.field(dtype=ti.i32, shape=())
        self.has_board[None] = 1 if (self.board is not None) else 0

        # AABB fields for GPU color filtering
        self.kmin_f = ti.Vector.field(3, ti.f32, shape=())
        self.kmax_f = ti.Vector.field(3, ti.f32, shape=())

    # ---------- viewer ----------



    # ---------- helpers ----------
    @ti.func
    def world_to_grid(self, x: ti.types.vector(3, ti.f32)):
        return (x - self.bounds_min[None]) / self.dx_s[None]

    @ti.func
    def grid_to_world(self, I: ti.types.vector(3, ti.i32)):
        return self.bounds_min[None] + (I.cast(ti.f32) + 0.5) * self.dx_s[None]

    @ti.kernel
    def _reset_contact_accumulators(self):
        """Reset per-substep contact impulse accumulators (grid-space)."""
        self.knife_impulse_g[None] = ti.Vector([0.0, 0.0, 0.0])
        self.board_impulse_g[None] = ti.Vector([0.0, 0.0, 0.0])
        self.knife_impulse_blade_g[None] = ti.Vector([0.0, 0.0, 0.0])
        self.knife_impulse_handle_g[None] = ti.Vector([0.0, 0.0, 0.0])



    def knife_world_aabb(self, margin: float = 0.02):
        """Current world AABB of the knife, expanded with margin."""
        if self._knife_neg_idx is None:
            return None, None
        y_anim = float(self.knife.y[None])
        z_off = float(self.knife.z_off[None])
        idx = self._knife_neg_idx[:, [2,1,0]].astype(np.float32) + 0.5
        xyz = idx * self._knife_voxel + self._knife_origin[None, :]
        xyz[:,1] += (y_anim - float(self._knife_origin[1]))
        xyz[:,2] += z_off
        min_coords = np.min(xyz, axis=0) - margin
        max_coords = np.max(xyz, axis=0) + margin
        return min_coords, max_coords

    # ---------- transfers ----------
    @ti.kernel
    def clear_grid(self):
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.Vector([ti.f32(0.0), ti.f32(0.0), ti.f32(0.0)])
            self.grid_m[I] = 0.0

    @ti.kernel
    def p2g(self):
        ti.loop_config(block_dim=256)  # better occupancy on GPU
        dx, inv_dx = self.dx_s[None], self.inv_dx_s[None]
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0.0
            self.grid_v[I] = ti.Vector([ti.f32(0.0), ti.f32(0.0), ti.f32(0.0)])

        for p in range(self.pcount[None]):
            part = self.particles[p]
            basef = self.world_to_grid(part.x) - 0.5
            base  = ti.cast(ti.floor(basef), ti.i32)
            base  = ti.max(ti.Vector([1,1,1]), ti.min(ti.Vector([self.n_grid-3]*3), base))
            fx = basef - base.cast(ti.f32)

            wx, wy, wz = self.bspline_quadratic_weights(fx)

            # Corotated elasticity (trial stress) - optimized
            mu = part.mu0 if self.no_stiffness_drop[None] == 1 else part.mu0 * (1.0 - part.D)
            la = part.la0 if self.no_stiffness_drop[None] == 1 else part.la0 * (1.0 - part.D)
            R, S = ti.polar_decompose(part.F)
            J = ti.max(ti.Matrix.determinant(part.F), 1e-6)  # Improved numerical precision
            J_inv = 1.0 / J  # Precompute inverse
            FinvT = part.F.inverse().transpose()
            P = 2.0 * mu * (part.F - R) + la * (J - 1.0) * J * FinvT
            stress_trial = J_inv * (P @ part.F.transpose())  # Use precomputed inverse

            # J2 plastic projection (optional)
            if self.plastic_enabled[None] == 1 and self.plastic_model == "j2":
                stress_new, alpha_inc = j2_return_mapping(stress_trial, mu, part.alpha, self.sigma_y0_f[None], self.H_iso_f[None], self.eta_vp_f[None])
                stress_trial = stress_new
                part.alpha += alpha_inc

            vol = part.vol; mass = part.mass
            # Precompute common values for optimization
            dt_4_inv_dx2 = self.dt * 4.0 * inv_dx * inv_dx
            affine = (-dt_4_inv_dx2 * vol * stress_trial) + mass * part.C

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        off = ti.Vector([i,j,k])
                        dpos = (off.cast(ti.f32) - fx) * dx
                        weight = wx[i] * wy[j] * wz[k]
                        weight_mass = weight * mass
                        idx = base + off
                        self.grid_v[idx] += weight * (mass * (part.v + part.C @ dpos) + affine @ dpos)
                        self.grid_m[idx] += weight_mass

            self.particles[p] = part

        # grid update + collider response
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m > 0:
                v = self.grid_v[I] / m
                pos = self.grid_to_world(I)
                # Apply domain boundaries first (not counted as contact)
                if pos[1] <= (self.bounds_min[None][1] + 0.5*self.dx_s[None]) and v[1] < 0: v[1] = 0.0
                if pos[0] <= (self.bounds_min[None][0] + 0.5*self.dx_s[None]) and v[0] < 0: v[0] = 0.0
                if pos[0] >= (self.bounds_max[None][0] - 0.5*self.dx_s[None]) and v[0] > 0: v[0] = 0.0
                if pos[2] <= (self.bounds_min[None][2] + 0.5*self.dx_s[None]) and v[2] < 0: v[2] = 0.0
                if pos[2] >= (self.bounds_max[None][2] - 0.5*self.dx_s[None]) and v[2] > 0: v[2] = 0.0
                v_before_contact = v

                                # Contact culling by AABB to avoid unnecessary SDF sampling
                pad = 2.0 * self.dx_s[None]
                in_knife = (pos[0] >= self.kmin_f[None][0]-pad) and (pos[0] <= self.kmax_f[None][0]+pad) and \
                           (pos[1] >= self.kmin_f[None][1]-pad) and (pos[1] <= self.kmax_f[None][1]+pad) and \
                           (pos[2] >= self.kmin_f[None][2]-pad) and (pos[2] <= self.kmax_f[None][2]+pad)
                # Additional cheap board band test (Y-only)
                in_board_band = (self.has_board[None] == 1) and (pos[1] <= self.board_top_y[None] + pad)
                contact = ti.i32(0)
                if in_knife:
                    contact = self.knife.classify_at(pos)
                elif in_board_band:
                    if ti.static(self.board != None):
                        db_here = self.board.sample(pos)
                        # Scale-normalized epsilon for board collision detection
                        eps_b = ti.f32(0.25) * self.board.voxel[None]
                        if db_here < eps_b:
                            contact = 2
                # Handle contacts
                if contact == 1:
                    # Mesh contact by the knife
                    v = self.knife.grid_respond(pos, v)
                    dv = v - v_before_contact
                    ti.atomic_add(self.knife_impulse_g[None][0], m * dv[0])
                    ti.atomic_add(self.knife_impulse_g[None][1], m * dv[1])
                    ti.atomic_add(self.knife_impulse_g[None][2], m * dv[2])
                    # Split accumulators: blade vs handle (for EEF force averaging)
                    if self.knife.is_blade(pos) == 1:
                        ti.atomic_add(self.knife_impulse_blade_g[None][0], m * dv[0])
                        ti.atomic_add(self.knife_impulse_blade_g[None][1], m * dv[1])
                        ti.atomic_add(self.knife_impulse_blade_g[None][2], m * dv[2])
                    else:
                        ti.atomic_add(self.knife_impulse_handle_g[None][0], m * dv[0])
                        ti.atomic_add(self.knife_impulse_handle_g[None][1], m * dv[1])
                        ti.atomic_add(self.knife_impulse_handle_g[None][2], m * dv[2])
                    ti.atomic_add(self.knife_mesh_collision_detected[None], 1)
                elif contact == 2:
                    # Board contact
                    v = self.knife.grid_respond(pos, v)
                    dv = v - v_before_contact
                    ti.atomic_add(self.board_impulse_g[None][0], m * dv[0])
                    ti.atomic_add(self.board_impulse_g[None][1], m * dv[1])
                    ti.atomic_add(self.board_impulse_g[None][2], m * dv[2])
                    ti.atomic_add(self.knife_board_collision_detected[None], 1)

                # Apply damping after contact
                v *= (1.0 - self.grid_damping)

                # Grid speed cap (resolution-aware): v_max_grid = factor * dx / dt
                v_cap_grid = self.grid_speed_cap_factor_f[None] * (self.dx_s[None] / self.dt)
                v_len = v.norm()
                if v_len > v_cap_grid:
                    v = (v_cap_grid / (v_len + 1e-12)) * v
                self.grid_v[I] = v

    @ti.kernel
    def g2p(self):
        ti.loop_config(block_dim=256)  # better occupancy on GPU
        dx, inv_dx = self.dx_s[None], self.inv_dx_s[None]
        for p in range(self.pcount[None]):
            part = self.particles[p]
            basef = self.world_to_grid(part.x) - 0.5
            base  = ti.cast(ti.floor(basef), ti.i32)
            base  = ti.max(ti.Vector([1,1,1]), ti.min(ti.Vector([self.n_grid-3]*3), base))
            fx = basef - base.cast(ti.f32)

            wx, wy, wz = self.bspline_quadratic_weights(fx)

            new_v = ti.Vector([ti.f32(0.0), ti.f32(0.0), ti.f32(0.0)])
            new_C = ti.Matrix.zero(ti.f32, 3, 3)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        dpos = (ti.Vector([i,j,k]).cast(ti.f32) - fx) * dx
                        weight = wx[i] * wy[j] * wz[k]
                        idx = base + ti.Vector([i,j,k])
                        vg = self.grid_v[idx]
                        new_v += weight * vg
                        new_C += 4.0*inv_dx*weight * vg.outer_product(dpos)

            # Clamp C to avoid blow-up
            for a in ti.static(range(3)):
                for b in ti.static(range(3)):
                    new_C[a,b] = ti.max(-30.0, ti.min(30.0, new_C[a,b]))

            # Symmetric tipping impulse (anti-bias via sign(dknife))
            if self.tip_force_gain_f[None] > 0.0:
                dknife_local = self.knife.sample(part.x)
                if ti.abs(dknife_local) < self.tip_band_f[None] and self.knife._down[None] == 1 and self.knife.is_blade(part.x) == 1:
                    n_tip = self.knife.normal(part.x)
                    # Project preferred global direction (ey) onto tangent plane for vertical cutting
                    ey = ti.Vector([ti.f32(0.0), ti.f32(1.0), ti.f32(0.0)])  # y-axis directions
                    t_raw = ey - (ey.dot(n_tip)) * n_tip
                    t_len = t_raw.norm()
                    t_planar = ti.Vector([ti.f32(1.0), ti.f32(0.0), ti.f32(0.0)])  # fallback
                    if t_len > 1e-6:
                        t_planar = t_raw / t_len
                    else:
                        ex = ti.Vector([ti.f32(1.0), ti.f32(0.0), ti.f32(0.0)])
                        t2 = ex - (ex.dot(n_tip)) * n_tip
                        t_planar = t2 / (t2.norm() + 1e-6)
                    # Symmetric by side sign (based on signed SDF)
                    side = ti.f32(1.0)
                    if dknife_local < 0.0:
                        side = -1.0
                    w_tip = 1.0 - ti.abs(dknife_local) / self.tip_band_f[None]
                    # Scale tip force relative to dx for consistent behavior
                    tip_force_scaled = self.tip_force_gain_f[None] * self.dx_ref_s[None] * (float(self.dx_ref_s[None]) / max(1e-12, float(self.dx_s[None]))) ** float(self.tip_reso_exp_f[None])
                    dv_tip = self.dt * tip_force_scaled * w_tip * side * t_planar
                    cap = 0.10 * (self.dx_s[None] / self.dt)
                    if dv_tip.norm() > cap:
                        dv_tip = (cap / (dv_tip.norm() + 1e-12)) * dv_tip
                    new_v += dv_tip

            # Post-cut tipping (knife rising): push sideways; symmetric by side sign
            if self.tip_post_gain_f[None] > 0.0 and self.knife._down[None] == 0:
                apply_ok = 1
                if self.tip_post_above_only[None] == 1 and self.has_board[None] == 1:
                    if not (part.x[1] > self.board_top_y[None] + self.pushout_eps[None]):
                        apply_ok = 0
                if apply_ok == 1 and self.knife.is_blade(part.x) == 1:
                    dknife_local = self.knife.sample(part.x)
                    n_tip = self.knife.normal(part.x)
                    # Project onto tangent plane for vertical cutting
                    ey = ti.Vector([ti.f32(0.0), ti.f32(1.0), ti.f32(0.0)])   # Y-axis direction
                    t_raw = ey - (ey.dot(n_tip)) * n_tip                      # project onto tangent plane
                    t_len = t_raw.norm()
                    t_planar = ti.Vector([ti.f32(1.0), ti.f32(0.0), ti.f32(0.0)])  # fallback
                    if t_len > 1e-6:
                        t_planar = t_raw / t_len
                    else:
                        ex = ti.Vector([ti.f32(1.0), ti.f32(0.0), ti.f32(0.0)])    # fallback if ez ≈ n_tip
                        t2 = ex - (ex.dot(n_tip)) * n_tip
                        t_planar = t2 / (t2.norm() + 1e-6)
                    side = 1.0
                    if dknife_local < 0.0: side = -1.0
                    post_band = self.tip_band_f[None] * self.tip_post_mul_f[None]
                    w_post = ti.min(1.0, ti.max(0.0, 1.0 - ti.abs(dknife_local) / post_band))
                    
                    # Scale post-cut tip force relative to dx for consistent behavior
                    tip_post_force_scaled = self.tip_post_gain_f[None] * self.dx_ref_s[None] * (float(self.dx_ref_s[None]) / max(1e-12, float(self.dx_s[None]))) ** float(self.tip_reso_exp_f[None])
                    dv_tip = self.dt * tip_post_force_scaled * w_post * side * t_planar
                    cap = 0.10 * (self.dx_s[None] / self.dt)  # Reduced cap for Z-only movement
                    if dv_tip.norm() > cap:
                        dv_tip = (cap / (dv_tip.norm() + 1e-12)) * dv_tip
                    new_v += dv_tip

            # gravity
            new_v += self.dt * self.gravity[None]

            part.v = (1.0 - self.particle_damping) * new_v
            # Apply additional velocity damping to reduce bouncing
            part.v *= 0.98  # 2% additional damping per timestep
            # speed cap
            # Enhanced speed cap with gradual limiting
            v2 = part.v.dot(part.v)
            vmax2 = self.speed_cap_f[None] * self.speed_cap_f[None]
            if v2 > vmax2:
                # Gradual speed reduction for better stability
                v_norm = ti.sqrt(v2 + 1e-6)
                scale = self.speed_cap_f[None] / v_norm
                # Apply stronger reduction to minimize bouncing
                part.v *= 0.7 * scale + 0.3  # 70% of cap + 30% of current (more aggressive)
            part.C = new_C

            # deformation gradient update + improved J clamp
            part.F = (ti.Matrix.identity(ti.f32,3) + self.dt * part.C) @ part.F
            Jcur = ti.Matrix.determinant(part.F)
            # More conservative J clamping for better stability
            J_min = ti.max(0.1, self.J_clamp_lo_f[None])  # Prevent extreme compression
            J_max = ti.min(5.0, self.J_clamp_hi_f[None])  # Prevent extreme expansion
            if (Jcur < J_min) or (Jcur > J_max):
                s = ti.pow(ti.max(1e-4, 1.0 / Jcur), ti.f32(1.0/3.0))
                part.F = s * part.F
                # Update J after clamping
                Jcur = ti.Matrix.determinant(part.F)

            part.x += self.dt * part.v

            # Particle-level projection and collision
            contact = self.knife.classify_at(part.x)
            if contact == 1:
                d = self.knife.sample(part.x)
                n = self.knife.normal(part.x)
                vknife = self.knife.velocity_at(part.x)

                # Relative velocity
                v_rel = part.v - vknife
                vn_rel = v_rel.dot(n)
                vt_rel = v_rel - vn_rel * n

                # Small stiction guard: treat tiny negatives as separation to avoid sticking
                v_eps = ti.f32(0.03) * (self.dx_s[None] / self.dt)  # 0.02~0.05 recommended

                # Apply response ONLY when approaching (vn_rel < -v_eps) and near the surface
                eps_k = ti.f32(0.25) * self.knife.voxel[None]
                if (ti.abs(d) <= eps_k) and (vn_rel < -v_eps):
                    # Normal restitution (use same coefficient as grid_respond)
                    vn_new = - self.mesh_restitution_f[None] * vn_rel

                    # Coulomb friction (slip clamp) on tangential RELATIVE velocity
                    mu = self.knife.friction_f[None]
                    if self.knife._down[None] == 0:
                        mu *= 0.10  # rising stroke: drastically lower friction to avoid adhesion
                    
                    # Initialize vt_new 
                    vt_new = vt_rel
                    vt_len = vt_rel.norm()
                    if vt_len > 1e-6:
                        slip = ti.min(mu, vt_len / (ti.abs(vn_rel) + 1e-4))
                        vt_new = (1.0 - slip) * vt_rel

                    # Compose back
                    part.v = vknife + vn_new * n + vt_new
                # else: separating or far — do NOT force normal to follow the knife; leave part.v as is.

                # Position correction: ONLY if truly inside (d < 0)
                if d < 0.0:
                    eps_clear = ti.f32(0.0)  # or tiny clearance: 0.002 * self.dx_s[None]
                    part.x -= (d - eps_clear) * n
                # DO NOT push out when d >= 0 (outside): avoids lifting particles during rise

                # Cutting allowed only by the blade on down-stroke + require sufficient contact (ĉ_now)
                c_hat_now = ti.min(ti.f32(1.0), self.knife.contact_accum_f[None] / (self.knife.c_norm_f[None] + 1e-6))
                if (self.knife.is_blade(part.x) == 1) \
                   and (ti.abs(self.knife.sample(part.x)) < self.damage_band_f[None]) \
                   and (vn_rel < -self.damage_v_threshold_f[None]) \
                   and (self.knife._down[None] == 1) \
                   and (c_hat_now >= self.c_hat_cut_min_f[None]):
                    push_distance = self.damage_band_f[None] - ti.abs(self.knife.sample(part.x))
                    normalized_push = push_distance / self.damage_band_f[None]
                    push_direction = 1.0 if self.knife.sample(part.x) > 0.0 else -1.0
                    displacement = push_direction * normalized_push * self.damage_band_f[None] * 0.25  # Reduced normal correction
                    part.x += n * displacement

                    if self.no_stiffness_drop[None] == 0:
                        damage_increment = 0.05 * normalized_push
                        part.D = ti.min(1.0, part.D + damage_increment)
                        strength_reduction = 1.0 - 0.9 * normalized_push
                        part.mu0 *= strength_reduction
                        part.la0 *= strength_reduction

            elif contact == 2:
                # Board collision at particle-level
                if ti.static(self.board != None):
                    n = self.board.normal(part.x)
                    vn = part.v.dot(n)
                    vt = part.v - vn*n
                    if vn < 0.0:
                        vn_new = - self.board.restitution_f[None] * vn
                        vt *= (1.0 - self.board.friction_f[None])
                        part.v = vt + vn_new * n
                    eps = self.pushout_eps[None]
                    part.x -= (self.board.sample(part.x) - eps) * n

            # Clamp to domain
            for d in ti.static(range(3)):
                if part.x[d] < self.bounds_min[None][d]:
                    part.x[d] = self.bounds_min[None][d]; part.v[d] = 0
                if part.x[d] > self.bounds_max[None][d]:
                    part.x[d] = self.bounds_max[None][d]; part.v[d] = 0

            self.particles[p] = part

    # ---------- seeding ----------
    def seed_particles_from_mesh(self, pack, material_cfg, name="mesh", support_top_y: float=None):
        sdf = pack["sdf"]; origin = pack["origin"]; vox = float(pack["voxel_size"])
        spacing = float(material_cfg.get("particle_spacing", 0.0033))
        density = float(material_cfg.get("density", 400.0))
        Nz, Ny, Nx = sdf.shape
        xs = np.arange(Nx, dtype=np.float32) * vox + origin[0]
        ys = np.arange(Ny, dtype=np.float32) * vox + origin[1]
        zs = np.arange(Nz, dtype=np.float32) * vox + origin[2]
        step = max(1, int(round(spacing / vox)))
        pts = []
        for z in range(0, Nz, step):
            for y in range(0, Ny, step):
                row = sdf[z, y, :]
                for x in range(0, Nx, step):
                    if row[x] < 0.0:
                        pts.append([xs[x], ys[y], zs[z]])
        pts = np.array(pts, np.float32)
        if len(pts) == 0:
            raise RuntimeError(f"No {name} particles; check mesh/SDF.")

        volp = float(spacing**3); massp = float(density * volp)
        num = len(pts)
        self.pcount[None] = num
        ecfg = material_cfg.get("elasticity", {})
        E  = float(ecfg.get("youngs_modulus", material_cfg.get("youngs_modulus", material_cfg.get("E", 3e5))))
        nu = float(ecfg.get("poisson_ratio",  material_cfg.get("poisson_ratio",  material_cfg.get("nu", 0.45))))
        mu0 = E / (2*(1+nu)); la0 = E * nu / ((1+nu)*(1-2*nu) + 1e-4)

        arr = np.zeros(num, dtype=[
            ('x',np.float32,3),('v',np.float32,3),
            ('F',np.float32,(3,3)),('C',np.float32,(3,3)),
            ('Jp',np.float32),('alpha',np.float32),('D',np.float32),
            ('mass',np.float32),('vol',np.float32),
            ('mu0',np.float32),('la0',np.float32)
        ])
        arr['x'] = pts; arr['v'] = 0.0
        I = np.eye(3, dtype=np.float32)
        arr['F'] = np.broadcast_to(I, (num, 3, 3))
        arr['C'] = 0.0; arr['Jp'] = 1.0; arr['alpha'] = 0.0; arr['D']  = 0.0
        arr['mass'] = massp; arr['vol']  = volp; arr['mu0']  = mu0; arr['la0']  = la0


        dx = float(self._dx_py); margin = max(1.0*dx, 0.5*spacing)
        bmin = self._bmin_py; bmax = self._bmax_py
        min_y = float(arr['x'][:,1].min())
        support = bmin[1]
        if support_top_y is not None:
            support = max(support, float(support_top_y))
        # place cutting mesh so that min gap = 0.5*dx (≤ dx)
        target_gap = 0.5 * dx
        need_up = (support + target_gap) - min_y
        if need_up > -1e-6:
            arr['x'][:,1] += max(0.0, need_up)
        # clamp tiny negative penetration
        arr['x'][:,1] = np.maximum(arr['x'][:,1], support + 0.01*dx)
        arr['x'][:,0] = np.clip(arr['x'][:,0], bmin[0] + margin, bmax[0] - margin)
        arr['x'][:,1] = np.clip(arr['x'][:,1], bmin[1] + margin, bmax[1] - margin)
        arr['x'][:,2] = np.clip(arr['x'][:,2], bmin[2] + margin, bmax[2] - margin)

        self.particles = Particle.field(shape=(num,))
        self.particles.from_numpy(arr)

        if self.viewer_enabled:
            self.p_color = ti.Vector.field(3, dtype=ti.f32, shape=(num,))
            for i in range(num): self.p_color[i] = ti.Vector([1.0, 1.0, 0.0])

    # ---------- step/run ----------
    def step(self, T):
        # Update knife AABB for this physics step (used for contact culling & coloring)
        try:
            kmin, kmax = self.knife_world_aabb(margin=2.0 * float(self.dx_s[None]))
            if kmin is not None and kmax is not None:
                self.kmin_f[None] = ti.Vector(kmin.tolist())
                self.kmax_f[None] = ti.Vector(kmax.tolist())
        except Exception:
            pass

        self._reset_contact_accumulators()
        # Bounce knife off the board if it reaches the top (with more tolerance for vertical cutting)
        if (self.board is not None):
            knife_low = float(self.knife.world_low_y())
            # Increased tolerance to allow deeper cutting
            board_tolerance = float(self.board_top_y[None]) - 0.01  # Allow 1cm deeper cutting
            if knife_low <= board_tolerance + float(self.pushout_eps[None]):
                self.knife.force_up()


        # ─── Visual saw-cut timer (once per drawn frame) ─────────
        # Advance at ≈60 Hz so you actually *see* the oscillation
        slow_factor = 3.0  # 1 = reg speed, 5 = 5x slower, 10 = 10x slower
        self.saw_time += (1.0 / 60.0) / slow_factor
        # Push this into the collider for its sine calculation
        self.knife.saw_time = self.saw_time
        # Advance global sim time
        self.sim_time += self.dt
        # Feed it into the KnifeCollider
        self.knife.sim_time = self.sim_time

        self.knife.update(self.dt)

        # Multi-cut scheduler
        if self.multi_cutting_enabled and (not self.cutting_complete):
            if not self._multi_init_done and (self._z_offsets is not None) and (len(self._z_offsets) > 0):
                try:
                    self.knife.z_off[None] = float(self._z_offsets[0])
                    self._multi_init_done = True
                except Exception as e:
                    print(f"[WARN] Cannot set initial z_off: {e}")

            # Detect top arrival (knife rising to start_y) - simplified and more robust
            at_top = False
            try:
                knife_down = int(self.knife._down[None])
                knife_y = float(self.knife.y[None])
                knife_start_y = float(self.knife.start_y)
                # More lenient top detection with larger tolerance
                at_top = (knife_down == 0) and (knife_y >= knife_start_y - 0.005)  # 5mm tolerance
            except Exception:
                at_top = False

            # Simplified multicut logic - single path
            if at_top and self._multi_init_done and not self.cutting_complete:
                if self._dwell_counter < self.dwell_frames:
                    self._dwell_counter += 1
                else:
                    # Dwell completed, move to next cut
                    self._dwell_counter = 0
                    self.current_cut_idx += 1
                    
                    if (self._z_offsets is not None) and (self.current_cut_idx < len(self._z_offsets)):
                        old = float(self.knife.z_off[None])
                        new = float(self._z_offsets[self.current_cut_idx])
                        self.knife.z_off[None] = new
                        # Reset knife speed to initial speed for new cut
                        self.knife.current_speed[None] = float(self.knife.base_speed)
                        self.knife._down[None] = 1  # trigger next down-stroke
                        print(f"[MULTICUT] Cut {self.current_cut_idx}/{len(self._z_offsets)}: Z={new:.4f}")
                        # Force full recolor if Z jumped notably
                        if should_force_full_recolor(old, new, threshold=0.5 * float(self.damage_band_f[None])):
                            self._force_full_recolor = 1
                    else:
                        self.cutting_complete = True
                        self.knife._down[None] = 0
                        print(f"[MULTICUT] All cuts completed!")

        # Recolor trigger when Z changes outside scheduler (e.g., manual)
        z_now = float(self.knife.z_off[None])
        if should_force_full_recolor(self._last_z_off, z_now, threshold=0.5 * float(self.damage_band_f[None])):
            self._force_full_recolor = 1
        self._last_z_off = z_now

        # self.clear_grid() removed: p2g zeros grid internally
        self.p2g()
        self.g2p()
        
        # --- accumulate substep contact impulses for FPS-averaged EEF force ---
        imp_b = self.knife_impulse_blade_g[None]
        imp_h = self.knife_impulse_handle_g[None]
        imp_board = self.board_impulse_g[None]
        Jb = np.array([float(imp_b[0]), float(imp_b[1]), float(imp_b[2])], np.float32)
        Jh = np.array([float(imp_h[0]), float(imp_h[1]), float(imp_h[2])], np.float32)
        Jboard = np.array([float(imp_board[0]), float(imp_board[1]), float(imp_board[2])], np.float32)
        
        self._J_blade_accum += Jb
        self._J_handle_accum += Jh
        self._J_board_accum += Jboard
        self._accum_time += self.dt


        # Reset particle collision counter for next frame
        self.knife_board_collision_detected[None] = 0
        self.knife_mesh_collision_detected[None] = 0

        # Update EEF tracking
        self._update_eef_tracking(T)
        
        self._frame += 1
        # Unified output manager (export + logging) at fixed FPS
        if self._out_manager is not None:
            def _make_rec(sim, sim_time, frame_idx):
                ee = sim.get_end_effector_state(sim_time)
                if ee is None:
                    return None
                # ---- FPS‑frame averaged force from substep accumulators ----
                Jb = sim._J_blade_accum
                Jh = sim._J_handle_accum
                J_sum = Jb + Jh
                dt_acc = max(1e-6, sim._accum_time)
                F = (J_sum / dt_acc).astype(np.float32)
                F_norm = float(np.linalg.norm(F) + 1e-6)
                F_normalized = F / F_norm
                # Consume accumulators (start fresh for next frame window)
                sim._J_blade_accum[:] = 0.0
                sim._J_handle_accum[:] = 0.0
                sim._J_board_accum[:]  = 0.0
                sim._accum_time = 0.0
                # Axis mapping for JSON (Y↔Z swap for ManiSkill)
                axis_map = str(sim.cfg.get("output", {}).get("logging", {}).get("axis_map", "maniskill")).lower()
                if axis_map == "maniskill":
                    f_out = {"x": float(F[0]), "y": float(F[2]), "z": float(F[1]), "norm": F_norm}
                    f_out_n = {"x": float(F_normalized[0]), "y": float(F_normalized[2]), "z": float(F_normalized[1])}
                else:
                    f_out = {"x": float(F[0]), "y": float(F[1]), "z": float(F[2]), "norm": F_norm}
                    f_out_n = {"x": float(F_normalized[0]), "y": float(F_normalized[1]), "z": float(F_normalized[2])}
                # Normalized scalars (for context)
                dx, dt, E = float(sim.dx_units), float(sim.dt_units), float(sim.E_units)
                vcap = dx / max(1e-6, dt)
                return {
                    "frame": int(frame_idx),
                    "timestep": int(sim._frame),
                    "time": float(sim_time),
                    "ee_pos_m": ee["pos"],
                    "ee_normal": ee["normal"],
                    "knife": {"y_anim": ee["y_anim"], "z_offset": ee["z_offset"]},
                    "force_world_N": f_out,
                    "force_normalized": f_out_n,
                    "scales": {"dx": dx, "dt": dt, "E": E, "vcap_dx_over_dt": vcap},
                    "telemetry": {
                        "material": {"E_Pa": sim._E_cur, "sigma_y_Pa": sim._sigma_y_cur,
                                     "E_ref_Pa": sim._E_ref, "sigma_ref_Pa": sim._sigma_ref},
                        "knife_resistance": {
                            "k2_base_per_s": sim._k2_base,
                            "k2_eff_per_s": sim._k2_eff,
                            "mat_gain": sim._k2_mat_gain,
                            "c_hat": float(sim.knife.last_c_hat_f[None]),
                            "c_hat_current": float(min(1.0, sim.knife.contact_accum_f[None] / max(1e-6, sim.knife.c_norm_f[None])))
                        }
                    }
                }
            self._out_manager.on_step(self, T, _make_rec)


    def get_end_effector_state(self, sim_time: float):
        """Return current EE world pose (position+normal) and knife animation scalars."""
        if not self._ee_available:
            return None
        # runtime translation relative to SDF pack origin
        y_anim = float(self.knife.y[None])
        z_off  = float(self.knife.z_off[None])
        dy = y_anim - float(self._knife_origin[1])
        pos = self._ee_base_world.copy()
        pos[1] += dy
        pos[2] += z_off
        n = self._ee_axis_world.copy()
        # Normalize just in case
        n = n / (np.linalg.norm(n) + 1e-6)
        return {
            "pos": pos.tolist(),
            "normal": n.tolist(),
            "y_anim": y_anim,
            "z_offset": z_off,
            "axis": self._ee_axis_name,
            "fraction": float(self._ee_fraction),
            "time": float(sim_time)
        }

    def _calculate_total_mass(self) -> float:
        """Calculate accurate total mass from particle properties."""
        if self.pcount[None] == 0:
            return 0.0
        
        # Get mass from first particle (all particles have same mass)
        first_particle_mass = float(self.particles[0].mass)
        return first_particle_mass * self.pcount[None]

    def _update_eef_tracking(self, sim_time: float):
        """Update EEF position, velocity, and force tracking for GUI display."""
        if not self._ee_available:
            return
            
        # Get current EEF state
        ee_state = self.get_end_effector_state(sim_time)
        if ee_state is None:
            return
            
        # Update position - ensure numpy arrays for vector operations
        self._ee_prev_position = self._ee_position.copy()

        base_pos = np.array(ee_state["pos"], dtype=np.float32)
        self._ee_position = base_pos.copy()

        # Apply saw cut offset in GUI
        if self.knife.cut_mode == "saw_cut":
            amp = float(self.knife.saw_amplitude[None])
            freq = float(self.knife.saw_frequency[None])
            phase = 2 * np.pi * freq * self.sim_time
            disp = amp * np.sin(phase)

            if self.knife.saw_axis == "x":
                self._ee_position[0] += disp
            else:
                self._ee_position[2] += disp

        # Calculate velocity (position difference / time step)
        if self._frame > 0:  # Skip first frame
            self._ee_velocity = (self._ee_position - self._ee_prev_position) / self.dt
        
        # Update force information
        imp = self.knife_impulse_g[None]
        self._knife_applies_force = np.array([
            float(imp[0] / self.dt),
            float(imp[1] / self.dt), 
            float(imp[2] / self.dt)
        ], dtype=np.float32)
        # Update FPS calculation
        import time
        self._fps_frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self._fps_start_time
        if elapsed_time >= 1.0:  # Update FPS every second
            self._fps = self._fps_frame_count / elapsed_time
            self._fps_frame_count = 0
            self._fps_start_time = current_time



    def run(self, *, run_sim=True, max_frames=50_000):  # CHANGE MAX_FRAMES!
        # Seed cutting mesh above board
        top = None
        if self.board is not None:
            top = float(self.board_top_y[None])
        self.seed_particles_from_mesh(
            self.cfg["cutting_mesh_pack"], self.cfg["cutting_mesh"], name="cutting_mesh", support_top_y=top
        )

        # Initialize multi-cut starting z-offset if enabled
        if self.multi_cutting_enabled and (self._z_offsets is not None) and len(self._z_offsets) > 0:
            initial_z_offset = float(self._z_offsets[0])
            self.knife.z_off[None] = initial_z_offset
            # Set recolor flag if you use color band visualization elsewhere
            if hasattr(self, "_force_full_recolor"):
                self._force_full_recolor = 1

        # Viewer ON, simulation OFF (preview only)
        if self.viewer_enabled and not run_sim:
            while self.window.running:
                self.renderer._draw()
            # Finish output manager (flush remaining logs)
            if self._out_manager is not None:
                self._out_manager.finish()
            return

        # Viewer OFF, simulation ON (headless)
        if not self.viewer_enabled and run_sim:
            T, steps_per_frame, frame = 0.0, self.substeps, 0
            while frame < max_frames:
                for _ in range(steps_per_frame):
                    self.step(T)
                    T += self.dt
                frame += 1
            # Finish output manager (flush remaining logs)
            if self._out_manager is not None:
                self._out_manager.finish()
            return

        # Viewer ON, simulation ON
        T, steps_per_frame, frame = 0.0, self.substeps, 0
        while self.window.running and frame < max_frames:
            for _ in range(steps_per_frame):
                self.step(T)
                T += self.dt
            frame += 1
            self.renderer._draw()

        # Finish output manager (flush remaining logs)
        if self._out_manager is not None:
            self._out_manager.finish()



"""
SDF‑based collider primitives for the cutting demo.
- BaseCollider: sampling, normals, optional mask, and basic motion hooks
- KnifeCollider: animated knife with blade mask and contact classification
- BoardCollider: static chopping board
"""


from __future__ import annotations
import numpy as np
import taichi as ti

@ti.data_oriented
class BaseCollider:
    """
    Base interface for SDF-based colliders.
    - Provides SDF samples and normals in world space.
    - Provides local velocity of the collider at a world-space point (for friction/impact).
    - Updates internal animation parameters each step (default: static).
    - Exposes approximate world-space Y-range of the solid region.
    """
    def __init__(self, sdf_grid: np.ndarray, origin, voxel_size: float, friction: float=0.3, restitution: float=0.0, mask_grid: np.ndarray=None):
        # Numpy handles (CPU-side)
        self._sdf_np = np.asarray(sdf_grid, np.float32)
        self._origin_py = np.asarray(origin, np.float32).copy()
        self._voxel_py  = float(voxel_size)

        # Taichi fields (GPU-side)
        self.sdf = ti.field(dtype=ti.f32, shape=self._sdf_np.shape)
        self.sdf.from_numpy(self._sdf_np)
        self.origin = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.voxel  = ti.field(dtype=ti.f32, shape=())
        self.origin[None] = ti.Vector(self._origin_py.tolist())
        self.voxel[None]  = self._voxel_py
        self.friction_f   = ti.field(dtype=ti.f32, shape=()); self.friction_f[None] = float(friction)
        self.restitution_f= ti.field(dtype=ti.f32, shape=()); self.restitution_f[None] = float(restitution)

        # Optional mask grid (e.g., knife blade mask)
        self._has_mask = int(mask_grid is not None)
        if self._has_mask:
            self._mask_np = np.asarray(mask_grid, np.float32)
            if self._mask_np.shape != self._sdf_np.shape:
                raise ValueError("mask_grid shape must match sdf_grid shape")
            self.mask = ti.field(dtype=ti.f32, shape=self._mask_np.shape)
            self.mask.from_numpy(self._mask_np)
        else:
            self._mask_np = None
            self.mask = None

        # Precompute solid Y-index range for CPU-side logic (Nz, Ny, Nx) -> axis=1 is Y
        solid = self._sdf_np < 0.0
        if np.any(solid):
            ys = np.where(solid)[1]
            self._yidx_min = int(ys.min())
            self._yidx_max = int(ys.max())
        else:
            self._yidx_min = 0
            self._yidx_max = -1  # empty

    # World-space Y of a given Y-index (cell center)
    def _y_index_to_world(self, y_idx: int) -> float:
        return float(self.origin[None][1] + (y_idx + 0.5) * self._voxel_py)

    @property
    def solid_ymin_world(self) -> float:
        if self._yidx_max < self._yidx_min:  # empty
            return float(self.origin[None][1])
        return self._y_index_to_world(self._yidx_min)

    @property
    def solid_ymax_world(self) -> float:
        if self._yidx_max < self._yidx_min:
            return float(self.origin[None][1])
        return self._y_index_to_world(self._yidx_max)

    # --------- sampling ---------
    @ti.func
    def _sample_grid(self, grid, p, y_anim, z_off) -> ti.f32:
        """Trilinear sample 'grid' at world point p using dynamic Y/Z offsets."""
        o = self.origin[None]
        v = self.voxel[None]
        # Optimize: precompute offset vector
        offset = ti.Vector([0.0, 0.0, z_off])
        o_dyn = ti.Vector([o[0], y_anim, o[2]])
        q = (p - o_dyn - offset) / v
        xi, yi, zi = q[0], q[1], q[2]
        x0 = ti.cast(ti.floor(xi), ti.i32); x1 = x0+1
        y0 = ti.cast(ti.floor(yi), ti.i32); y1 = y0+1
        z0 = ti.cast(ti.floor(zi), ti.i32); z1 = z0+1
        Nx = ti.static(grid.shape[2]); Ny = ti.static(grid.shape[1]); Nz = ti.static(grid.shape[0])
        x0 = ti.max(0, ti.min(Nx-1, x0)); x1 = ti.max(0, ti.min(Nx-1, x1))
        y0 = ti.max(0, ti.min(Ny-1, y0)); y1 = ti.max(0, ti.min(Ny-1, y1))
        z0 = ti.max(0, ti.min(Nz-1, z0)); z1 = ti.max(0, ti.min(Nz-1, z1))
        xd = xi - ti.cast(x0, ti.f32); yd = yi - ti.cast(y0, ti.f32); zd = zi - ti.cast(z0, ti.f32)
        # Optimize: precompute interpolation weights
        xd_inv = 1.0 - xd; yd_inv = 1.0 - yd; zd_inv = 1.0 - zd
        c000=grid[z0,y0,x0]; c100=grid[z0,y0,x1]
        c010=grid[z0,y1,x0]; c110=grid[z0,y1,x1]
        c001=grid[z1,y0,x0]; c101=grid[z1,y0,x1]
        c011=grid[z1,y1,x0]; c111=grid[z1,y1,x1]
        c00=c000*xd_inv+c100*xd; c01=c001*xd_inv+c101*xd
        c10=c010*xd_inv+c110*xd; c11=c011*xd_inv+c111*xd
        c0=c00*yd_inv+c10*yd;    c1=c01*yd_inv+c11*yd
        return c0*zd_inv+c1*zd

    @ti.func
    def sample(self, p):
        """SDF sample at world point p (negative = inside). Override in subclasses to add motion."""
        return self._sample_grid(self.sdf, p, self.origin[None][1], 0.0)

    @ti.func
    def sample_mask(self, p):
        """Sample optional mask at world point p (1=blade, 0=not blade)."""
        # Set in __init__ as self._has_mask = 0/1 → safe for ti.static
        if ti.static(self._has_mask == 0):
            return 1.0
        # Default collider has no motion; subclasses can override with animation state
        return self._sample_grid(self.mask, p, self.origin[None][1], 0.0)

    @ti.func
    def normal(self, p):
        """SDF gradient-based normal (outward)."""
        h = 0.75 * self.voxel[None]  # Smoother normal for high resolution
        sx = self.sample(p + ti.Vector([ti.f32(h), ti.f32(0.0), ti.f32(0.0)])) - self.sample(p - ti.Vector([ti.f32(h), ti.f32(0.0), ti.f32(0.0)]))
        sy = self.sample(p + ti.Vector([ti.f32(0.0), ti.f32(h), ti.f32(0.0)])) - self.sample(p - ti.Vector([ti.f32(0.0), ti.f32(h), ti.f32(0.0)]))
        sz = self.sample(p + ti.Vector([ti.f32(0.0), ti.f32(0.0), ti.f32(h)])) - self.sample(p - ti.Vector([ti.f32(0.0), ti.f32(0.0), ti.f32(h)]))
        n = ti.Vector([ti.f32(sx), ti.f32(sy), ti.f32(sz)])
        return n / (n.norm() + 1e-8)

    @ti.func
    def velocity_at(self, p):
        """Rigid velocity at world point (default static)."""
        return ti.Vector([ti.f32(0.0), ti.f32(0.0), ti.f32(0.0)])

    def update(self, dt: float):
        """Advance internal animation; default no-op."""
        pass


@ti.data_oriented
class KnifeCollider(BaseCollider):
    """
    Moving knife with vertical Y motion and optional Z offsets.
    Optional blade mask limits cutting to the blade region while handle only collides.
    """
    def __init__(self, sdf_grid, origin, voxel_size, start_y, stop_y,
                 speed, return_speed, z_offset=0.0, friction=0.3,
                 blade_mask_grid=None, mesh_restitution=0.01, board_shape=None,
                 # new saw params
                 cut_mode: str="cut",
                 saw_amplitude: float=0.0,
                 saw_frequency: float=0.0,
                 saw_axis: str="x",
                 sim_time: float = 0.0):
        super().__init__(sdf_grid, origin, voxel_size, friction, 0.0, mask_grid=blade_mask_grid)
        self.y = ti.field(dtype=ti.f32, shape=()); self.y[None] = float(start_y)
        self.start_y = float(start_y); self.stop_y = float(stop_y)
        self.speed = float(speed);     self.return_speed = float(return_speed)
        # store original Z offset
        self._z_init = float(z_offset)
        self.z_off = ti.field(dtype=ti.f32, shape=()); self.z_off[None] = float(z_offset)

        # X offset for saw oscillation
        self.x_off = ti.field(dtype=ti.f32, shape=())
        self.x_off[None] = 0.0

        # cut mode and saw parameters
        self.sim_time = 0.0
        self.saw_time = 0.0
        self.cut_mode = cut_mode
        self.saw_axis = saw_axis.lower()  # "x" or "z"
        self.saw_amplitude = ti.field(dtype=ti.f32, shape=())
        self.saw_amplitude[None] = float(saw_amplitude)
        self.saw_frequency = ti.field(dtype=ti.f32, shape=())
        self.saw_frequency[None] = float(saw_frequency)

        self._down = ti.field(dtype=ti.i32, shape=()); self._down[None] = 1
        self._knife_yidx_min = getattr(self, "_yidx_min", 0)
        
        # Board collision tracking fields
        self.has_board_flag = ti.field(dtype=ti.i32, shape=()); self.has_board_flag[None] = 0
        self._hit_mesh = ti.field(dtype=ti.i32, shape=()); self._hit_mesh[None] = 0
        self._hit_board = ti.field(dtype=ti.i32, shape=()); self._hit_board[None] = 0
        self._board = None
        
        # Mesh restitution for cutting simulation
        self.mesh_restitution_f = ti.field(dtype=ti.f32, shape=()); self.mesh_restitution_f[None] = float(mesh_restitution)
        
        # Smooth speed control (tunable)
        self.base_speed = float(speed)
        self.current_speed = ti.field(dtype=ti.f32, shape=()); self.current_speed[None] = float(speed)
        # Minimum speed floor (fraction of base speed). Lower default to allow real deceleration.
        self.min_speed_f = ti.field(dtype=ti.f32, shape=()); self.min_speed_f[None] = 0.10 * float(speed)  # 10% floor (will be overridden from YAML)
        # Apply restitution to cut_tau: lower restitution = slower decay (less speed reduction)
        base_tau = 1.0  # [s] base decay time constant (increased for slower speed changes)
        restitution_factor = 1.0 + (1.0 - float(mesh_restitution))  # 1.0 to 2.0 range (inverted)
        self.cut_tau_f = ti.field(dtype=ti.f32, shape=()); self.cut_tau_f[None] = base_tau * restitution_factor
        self.rec_tau_f = ti.field(dtype=ti.f32, shape=()); self.rec_tau_f[None] = 0.6   # [s] recovery time constant
        self.c_norm_f = ti.field(dtype=ti.f32, shape=()); self.c_norm_f[None] = 1.0    # [m/s] normalize contact metric (will be set from YAML)
        
        # Per-substep contact metric accumulator (atomic-updated in grid_respond)
        self.contact_accum_f = ti.field(dtype=ti.f32, shape=()); self.contact_accum_f[None] = 0.0
        # Telemetry: last normalized contact (ĉ in [0,1])
        self.last_c_hat_f = ti.field(dtype=ti.f32, shape=()); self.last_c_hat_f[None] = 0.0
        
        # ---- [Additional] Board replication fields ----
        if board_shape is None:
            board_shape = (1, 1, 1)  # No collision dummy

        Nz, Ny, Nx = int(board_shape[0]), int(board_shape[1]), int(board_shape[2])
        self.board_sdf = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx))
        self.board_origin = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.board_voxel  = ti.field(dtype=ti.f32, shape=())
        self.board_friction_f    = ti.field(dtype=ti.f32, shape=())
        self.board_restitution_f = ti.field(dtype=ti.f32, shape=())

        # Default: no collision dummy board
        self.board_sdf.fill(1e6)
        self.board_origin[None] = ti.Vector([0.0, 0.0, 0.0])
        self.board_voxel[None]  = 1.0
        self.board_friction_f[None]    = 0.0
        self.board_restitution_f[None] = 0.0
        

    def world_low_y(self) -> float:
        """Lowest voxel center height = y_anim + (ymin_idx+0.5)*voxel"""
        return float(self.y[None]) + (self._knife_yidx_min + 0.5) * float(self._voxel_py)

    def force_up(self):
        """Force the knife to start moving upward from the next update."""
        self._down[None] = 0

    def update(self, dt: float):
        """Ping-pong between start_y and stop_y with smooth speed control."""
        t = float(self.sim_time)  # you'll need to store sim_time each frame
        # Ping-pong Y motion (existing)
        if self._down[None] == 1:
            y_new = self.y[None] - dt * self.current_speed[None]
            if y_new <= self.stop_y:
                y_new = self.stop_y; self._down[None] = 0
        else:
            y_new = self.y[None] + dt * self.return_speed
            if y_new >= self.start_y:
                y_new = self.start_y; self._down[None] = 1
        self.y[None] = y_new
        
        # Physical speed control (Quadratic): du/dt = -k2 * c_hat * u^2,  u = s/s0
        c = float(self.contact_accum_f[None]); self.contact_accum_f[None] = 0.0
        c_hat = min(1.0, c / max(1e-4, float(self.c_norm_f[None])))  # ∈[0,1]
        self.last_c_hat_f[None] = c_hat  # Store for telemetry
        s    = float(self.current_speed[None]); s0 = float(self.base_speed)
        s_min= float(self.min_speed_f[None])
        k2   = float(getattr(self, "res_kq_per_s", 4.0))  # [1/s] - injected from YAML in sim.py
        if int(self._down[None]) == 1 and c_hat > 1e-4:
            # Semi‑implicit update: u_next = u / (1 + k2 * c_hat * u * dt)
            u = max(1e-6, s / max(1e-6, s0))
            s = s / (1.0 + (k2 * c_hat * u) * dt)
        # Recovery toward base speed when rising or no contact: s' = (s0 - s)/tau
        if int(self._down[None]) == 0 or c_hat < 1e-4:
            tau = float(self.rec_tau_f[None])  # [s]
            s += (s0 - s) * (dt / max(1e-4, tau))
        self.current_speed[None] = max(s_min, min(s, s0))

        # Apply saw-cut oscillation
        if self.cut_mode == "saw_cut":
            # compute sinusoidal displacement
            w = 2.0 * 3.14159265 * self.saw_frequency[None]
            disp = self.saw_amplitude[None] * ti.sin(w * self.saw_time)

            if self.saw_axis == "x":
                self.x_off[None] = disp
            else:
                self.x_off[None] = 0.0
                self.z_off[None] = disp
        else:
            # plain cut: no oscillation in X, but respect whatever
            # z_off the scheduler just wrote in for this slice.
            self.x_off[None] = 0.0
            # NOTE: DO NOT reset self.z_off[None] here!

    @ti.func
    def sample(self, p):
        """SDF sample with animated X, Y, and Z offsets."""
        # shift the query point forward/back by self.x_off
        p_shift = p + ti.Vector([self.x_off[None], 0.0, 0.0])
        return self._sample_grid(self.sdf, p_shift, self.y[None], self.z_off[None])

    @ti.func
    def sample_mask(self, p):
        """Mask sample with animated Y and Z offsets (1=blade, 0=not blade)."""
        if ti.static(self.mask == None):
            return 1.0
        p_shift = p + ti.Vector([self.x_off[None], 0.0, 0.0])
        return self._sample_grid(self.mask, p_shift, self.y[None], self.z_off[None])

    @ti.func
    def is_blade(self, p):
        """Return 1 if inside the blade region; 0 otherwise."""
        return 1 if self.sample_mask(p) > 0.5 else 0

    @ti.func
    def normal(self, p):
        h = 0.75 * self.voxel[None]  # Smoother normal for high resolution
        sx = self.sample(p + ti.Vector([h, 0.0, 0.0])) - self.sample(p - ti.Vector([h, 0.0, 0.0]))
        sy = self.sample(p + ti.Vector([0.0, h, 0.0])) - self.sample(p - ti.Vector([0.0, h, 0.0]))
        sz = self.sample(p + ti.Vector([0.0, 0.0, h])) - self.sample(p - ti.Vector([0.0, 0.0, h]))
        n = ti.Vector([sx, sy, sz])
        return n / (n.norm() + 1e-8)

    @ti.func
    def velocity_at(self, p):
        # Knife moves along -Y / +Y (use current_speed for accurate collision response)
        vy = -self.current_speed[None] if self._down[None]==1 else self.return_speed
        return ti.Vector([ti.f32(0.0), ti.f32(vy), ti.f32(0.0)])

    def attach_board(self, board):
        """
        Call this only when you want to use board for physics collision.
        Do not call this if you only want board rendering.
        """
        assert tuple(self.board_sdf.shape) == tuple(board.sdf.shape), \
            f"board_shape mismatch: expected {self.board_sdf.shape}, got {board.sdf.shape}. " \
            f"Pass board.sdf.shape as board_shape when creating KnifeCollider."

        # Copy values (kernel only accesses copied fields)
        self.board_sdf.copy_from(board.sdf)
        self.board_origin[None] = board.origin[None]
        self.board_voxel[None]  = board.voxel[None]
        self.board_friction_f[None]    = board.friction_f[None]
        self.board_restitution_f[None] = board.restitution_f[None]

        self.has_board_flag[None] = 1
        self._board = board  # For debugging only (kernel must not use this)

    @ti.kernel
    def reset_contact_counters(self):
        self._hit_mesh[None] = 0
        self._hit_board[None] = 0

    @ti.func
    def _board_sample(self, p):
        return self._sample_grid(self.board_sdf, p, self.board_origin[None][1], 0.0)

    @ti.func
    def _board_normal(self, p):
        h = 0.75 * self.board_voxel[None]
        sx = self._board_sample(p + ti.Vector([h, 0.0, 0.0])) - self._board_sample(p - ti.Vector([h, 0.0, 0.0]))
        sy = self._board_sample(p + ti.Vector([0.0, h, 0.0])) - self._board_sample(p - ti.Vector([0.0, h, 0.0]))
        sz = self._board_sample(p + ti.Vector([0.0, 0.0, h])) - self._board_sample(p - ti.Vector([0.0, 0.0, h]))
        n = ti.Vector([sx, sy, sz])
        return n / (n.norm() + 1e-8)

    @ti.func
    def classify_at(self, p) -> ti.i32:
        """Return 1 if knife-vs-mesh, 2 if knife-vs-board, 0 if no collision at point p."""
        dknife = self.sample(p)
        eps_k = ti.f32(0.25) * self.voxel[None]

        # Board path: only access copied fields
        db = 1e9
        eps_b = ti.f32(0.0)
        if self.has_board_flag[None] == 1:
            db = self._board_sample(p)
            eps_b = ti.f32(0.25) * self.board_voxel[None]

        result = 0
        knife_in = dknife < eps_k
        board_in = (eps_b > 0.0) and (db < eps_b)

        # Prefer knife when both are in contact and knife is closer
        if knife_in and (not board_in or (dknife < db)):
            result = 1
        elif board_in:
            result = 2

        return result

    @ti.func
    def grid_respond(self, pos, v):
        """
        Collision response for grid velocity at 'pos'.
        Applies reflection/friction only when approaching (energy safe).
        """
        which = self.classify_at(pos)
        if which == 1:
            # Knife ↔ Mesh (use knife's restitution/friction)
            n = self.normal(pos)
            vknife = self.velocity_at(pos)
            v_rel = v - vknife
            vn_rel = v_rel.dot(n)
            if vn_rel < 0.0:
                # Accumulate contact metric (normal approach speed), blade-only
                if self.is_blade(pos) == 1:
                    ti.atomic_add(self.contact_accum_f[None], -vn_rel)  # unit: m/s
                
                # Reflect normal component with mesh restitution (energy loss)
                vn_rel_new = - self.mesh_restitution_f[None] * vn_rel
                vt_rel = v_rel - vn_rel * n
                # Coulomb friction with slip clamp
                vt_len = vt_rel.norm()
                if vt_len > 1e-6:
                    slip = ti.min(self.friction_f[None], vt_len / (ti.abs(vn_rel) + 1e-4))
                    v_rel = vn_rel_new * n + (1.0 - slip) * vt_rel
                else:
                    v_rel = vn_rel_new * n
                v = v_rel + vknife
                # Apply additional damping to reduce bouncing
                v *= 0.95  # 5% velocity reduction per collision
                
        elif which == 2 and self.has_board_flag[None] == 1:
            # Knife ↔ Board (use board's restitution/friction)
            n = self._board_normal(pos)
            vn = v.dot(n)
            if vn < 0.0:
                # Apply restitution (energy loss) and friction
                vn_new = - self.board_restitution_f[None] * vn
                vt = v - vn * n
                vt *= (1.0 - self.board_friction_f[None])
                v = vt + vn_new * n
                # Apply additional damping to reduce bouncing
                v *= 0.95  # 5% velocity reduction per collision
        return v


@ti.data_oriented
class BoardCollider(BaseCollider):
    """Static chopping board. Only collides, does not cut."""
    def __init__(self, sdf_grid, origin, voxel_size, friction=0.5, restitution=0.0):
        super().__init__(sdf_grid, origin, voxel_size, friction, restitution)
        # --- environment hooks & counters ---
        self.has_board_flag = ti.field(dtype=ti.i32, shape=()); self.has_board_flag[None] = 0
        self._hit_mesh = ti.field(dtype=ti.i32, shape=()); self._hit_mesh[None] = 0
        self._hit_board = ti.field(dtype=ti.i32, shape=()); self._hit_board[None] = 0

    @ti.func
    def velocity_at(self, p):
        return ti.Vector([0.0, 0.0, 0.0])

    def update(self, dt: float):
        pass


class MultiCollider:
    """Thin container that keeps named colliders and returns the most penetrating one at a point."""
    def __init__(self, **named):
        self.named = named

    def get(self, name: str):
        return self.named.get(name)

    @property
    def all(self):
        return list(self.named.values())



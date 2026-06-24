"""
Rendering utilities for the MLS‑MPM cutting simulation.
- Sets up Taichi UI (window, camera, lighting)
- Renders particles, knife/board proxies, optional grid, and overlay HUD
"""
import numpy as np
import taichi as ti

def should_force_full_recolor(prev_z_off: float, new_z_off: float, threshold: float = 0.01) -> bool:
    """Return True when |Δz| is large enough to warrant a full recolor pass."""
    return abs(float(new_z_off) - float(prev_z_off)) >= float(threshold)

@ti.data_oriented
class MPMRenderer:
    """Renderer class for MPM cutting simulation."""
    def __init__(self, sim):
        self.sim = sim
        self._board_proxy_cache = None
        self._grid_proxy_cache = None

    def _pick_scene(self, window):
        if hasattr(window, "get_scene"):
            return window.get_scene()
        return ti.ui.Scene()

    def _init_viewer(self):
        self.sim.window = ti.ui.Window("MLS‑MPM Cutting", res=(1280, 800), vsync=False)
        self.sim.canvas = self.sim.window.get_canvas()
        self.sim.scene  = self._pick_scene(self.sim.window)
        try:
            self.sim.camera = self.sim.window.get_camera()
        except Exception:
            self.sim.camera = ti.ui.Camera()

    def _apply_initial_camera(self):
        mode = self.sim.viewer_camera_mode
        camera_presets = {
            "top":   ([0.0, 0.7, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
            "front": ([1.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 1.0, 0.0]),
            "auto":  ([0.4, 0.3, 0.4], [0.0, 0.1, 0.0], [0.0, 1.0, 0.0])
        }
        if mode in camera_presets:
            eye, look, up = camera_presets[mode]
        else:
            pose = self.sim.viewer_initial_pose
            eye  = pose.get("eye",    [0.3, 0.22, 0.3])
            look = pose.get("lookat", [0.0, 0.1, 0.0])
            up   = pose.get("up",     [0.0, 1.0, 0.0])
        try:
            self.sim.camera.position(*map(float, eye))
            self.sim.camera.lookat(*map(float, look))
            self.sim.camera.up(*map(float, up))
            self.sim.camera.fov(float(self.sim.viewer_initial_pose.get("fov", 35.0)))
        except Exception:
            pass

    @ti.func
    def _get_cutting_color(self, cut_intensity: ti.f32):
        """Yellow → Orange → Red → Dark Red."""
        red = 1.0
        green = ti.max(0.0, 1.0 - cut_intensity)
        blue = ti.max(0.0, 0.2 * cut_intensity - 0.1)
        return ti.Vector([red, green, blue])

    @ti.kernel
    def _recolor_full_kernel(self, band_eps: ti.f32):
        for p in range(self.sim.pcount[None]):
            X = self.sim.particles[p].x
            if self.sim.knife.is_blade(X) == 1 and ti.abs(self.sim.knife.sample(X)) < band_eps:
                nd = ti.abs(self.sim.knife.sample(X)) / band_eps
                self.sim.p_color[p] = self._get_cutting_color(1.0 - nd)
            else:
                self.sim.p_color[p] = ti.Vector([1.0, 1.0, 0.0])   # yellow
                # self.sim.p_color[p] = ti.Vector([1.0, 0.0, 0.0])   # red
                # self.sim.p_color[p] = ti.Vector([0.0, 0.2, 0.0])   # dark green
                # self.sim.p_color[p] = ti.Vector([1.0, 0.6, 0.4])   # peach color
                # self.sim.p_color[p] = ti.Vector([0.9, 1.0, 0.85])  # light green
                # self.sim.p_color[p] = ti.Vector([1.0, 0.55, 0.0])  # orange

    def _update_cut_colors(self, band_eps: float):
        self._recolor_full_kernel(float(band_eps))
        self.sim._force_full_recolor = 0

    def _setup_scene_lighting(self):
        self.sim.scene.ambient_light((0.45, 0.45, 0.45))
        self.sim.scene.point_light(pos=(0.6, 0.8, 0.6), color=(1, 1, 1))

    def _handle_camera_input(self):
        if not self.sim.viewer_lock_on_run:
            self.sim.camera.track_user_inputs(self.sim.window, movement_speed=0.02, hold_key=ti.ui.RMB)

    def _render_particles(self):
        radius = self.sim.particle_render_radius * self.sim.viewer_radius_scale
        if self.sim.p_color is not None and self.sim.damage_visualization:
            if self.sim._frame % self.sim.color_update_every == 0:
                self._update_cut_colors(float(self.sim.damage_band_f[None]))
                self.sim._last_color_update_frame = self.sim._frame
            self.sim.scene.particles(self.sim.particles.x, radius=radius, per_vertex_color=self.sim.p_color)
        else:
            self.sim.scene.particles(self.sim.particles.x, radius=radius, color=(1.0, 1.0, 0.0))

    def _render_eef_position(self):
        if not self.sim._ee_available:
            return
        ee_state = self.sim.get_end_effector_state(0.0)
        if ee_state is not None:
            eef_pos = np.array(ee_state["pos"], dtype=np.float32)
            self.sim.scene.particles(eef_pos.reshape(1,3), radius=self.sim.particle_render_radius * 2.0, color=(0.0, 0.0, 1.0))

    def _render_knife_proxy(self):
        if self.sim._knife_base_xyz is None:
            return
        y_anim = float(self.sim.knife.y[None])
        z_off  = float(self.sim.knife.z_off[None])
        x_off = float(self.sim.knife.x_off[None])  # ← NEW

        knife_origin_y = float(self.sim.knife.origin[None][1])
        dy = y_anim - knife_origin_y

        base = self.sim._knife_base_xyz
        xyz = np.empty_like(base)
        xyz[:, 0] = base[:, 0] + x_off      # ← APPLY X OFFSET
        xyz[:, 1] = base[:, 1] + dy
        xyz[:, 2] = base[:, 2] + z_off
        self.sim.scene.particles(xyz, radius=self.sim.particle_render_radius * 0.8, color=(0.8, 0.8, 0.8))

    def _render_board_proxy(self):
        if self.sim.board is None or not self.sim.show_board:
            return
        if self._board_proxy_cache is None:
            board_sdf = self.sim.board._sdf_np
            board_origin = self.sim.board.origin[None].to_numpy()
            board_voxel  = self.sim.board._voxel_py
            Nz, Ny, Nx = board_sdf.shape
            zs = np.arange(0, Nz, 4, dtype=np.int32)
            ys = np.arange(0, Ny, 4, dtype=np.int32)
            xs = np.arange(0, Nx, 4, dtype=np.int32)
            sdf_sub = board_sdf[np.ix_(zs, ys, xs)]
            surf = np.abs(sdf_sub) < (0.5 * board_voxel)
            if np.any(surf):
                Z, Y, X = np.meshgrid(zs, ys, xs, indexing='ij')
                pts = np.column_stack([
                    (X[surf] + 0.5) * board_voxel + board_origin[0],
                    (Y[surf] + 0.5) * board_voxel + board_origin[1],
                    (Z[surf] + 0.5) * board_voxel + board_origin[2],
                ]).astype(np.float32)
            else:
                pts = np.empty((0,3), np.float32)
            self._board_proxy_cache = pts
        if self._board_proxy_cache.shape[0] > 0:
            self.sim.scene.particles(self._board_proxy_cache, radius=self.sim.particle_render_radius * 0.8, color=(0.6, 0.4, 0.2))

    def _render_grid_visualization(self):
        if not self.sim.show_grid:
            return
        if self._grid_proxy_cache is None:
            n = int(self.sim.n_grid)
            dx = float(self.sim.dx_s[None])
            bmin = self.sim.bounds_min[None].to_numpy()
            k = max(1, n // 16)
            xs = np.arange(0, n, k, dtype=np.int32)
            ys = np.arange(0, n, k, dtype=np.int32)
            zs = np.arange(0, n, k, dtype=np.int32)
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
            pts = np.column_stack([
                bmin[0] + (X.flatten() + 0.5) * dx,
                bmin[1] + (Y.flatten() + 0.5) * dx,
                bmin[2] + (Z.flatten() + 0.5) * dx,
            ]).astype(np.float32)
            self._grid_proxy_cache = pts
        self.sim.scene.particles(self._grid_proxy_cache, radius=self.sim.particle_render_radius * 0.5, color=(1.0, 0.0, 0.0))

    def _draw_info_overlay(self):
        if not self.sim.viewer_enabled or self.sim.window is None:
            return
        try:
            gui = self.sim.window.get_gui()
            with gui.sub_window("Simulation Info", 0.58, 0.58, 0.4, 0.4):
                gui.text("=== PERFORMANCE ===")
                gui.text(f"FPS: {self.sim._fps:.1f}")
                gui.text(f"Frame: {self.sim._frame}")

                gui.text("")
                gui.text("=== END EFFECTOR ===")
                gui.text(f"Position: [{self.sim._ee_position[0]:.3f}, {self.sim._ee_position[1]:.3f}, {self.sim._ee_position[2]:.3f}] m")
                gui.text(f"Velocity: [{self.sim._ee_velocity[0]:.3f}, {self.sim._ee_velocity[1]:.3f}, {self.sim._ee_velocity[2]:.3f}] m/s")
                gui.text(f"Speed: {np.linalg.norm(self.sim._ee_velocity):.3f} m/s")

                gui.text("")
                gui.text("=== FORCES ===")
                gui.text(f"Knife Applies X [ManiSkills X]: {self.sim._knife_applies_force[0]:.3f} N")
                gui.text(f"Knife Applies Y [ManiSkills Z]: {self.sim._knife_applies_force[1]:.3f} N")
                gui.text(f"Knife Applies Z [ManiSkills Y]: {self.sim._knife_applies_force[2]:.3f} N")
                gui.text(f"Force Magnitude: {np.linalg.norm(self.sim._knife_applies_force):.3f} N")

                gui.text("")
                gui.text("=== SIMULATION STATUS ===")
                gui.text("Status: Running")
                gui.text("Physics: MPM + CPIC")
        except Exception as e:
            print(f"GUI Error: {e}")

    def _draw(self):
        self._handle_camera_input()
        self.sim.scene.set_camera(self.sim.camera)
        self._setup_scene_lighting()
        self._render_particles()
        self._render_knife_proxy()
        self._render_board_proxy()
        self._render_eef_position()
        self._render_grid_visualization()
        self._draw_info_overlay()
        self.sim.canvas.scene(self.sim.scene)
        self.sim.window.show()

    def render(self):
        self._draw()

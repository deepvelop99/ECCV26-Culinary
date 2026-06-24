import os, json, numpy as np


class FPSSampler:
    """Fixed cadence scheduler based on wall simulation time; independent of dt/substeps."""

    def __init__(self, fps: float):
        self.set_fps(fps)
        self.reset()

    def set_fps(self, fps: float):
        self.fps = float(max(1e-4, fps))
        self.dt_out = 1.0 / self.fps

    def reset(self):
        self._next_t = 0.0
        self._frame = 0

    @property
    def frame(self): return self._frame

    def due(self, t: float):
        return (t + 1e-10) >= self._next_t

    def step_until_now(self, t: float):
        """Advance schedule to time t, yielding the list of (out_time, frame_idx) to emit now."""
        outs = []
        while t + 1e-10 >= self._next_t:
            outs.append((self._next_t, self._frame))
            self._frame += 1
            self._next_t += self.dt_out
        return outs


class FPSExporter:
    """Export particle states at a fixed FPS cadence (NPZ or ASCII PLY)."""

    def __init__(self, out_dir="exports_banana_saw_cut", fmt="npz", with_color=True):  # CHANGE EXPORTS FOR EACH FRUIT
        self.out_dir = str(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.fmt = str(fmt).lower()
        self.with_color = bool(with_color)

    def _save_npz(self, sim, sim_time: float, frame_idx: int):
        x = sim.particles.x.to_numpy()
        v = sim.particles.v.to_numpy()
        out = {"x": x, "v": v, "time": float(sim_time)}
        if self.with_color and hasattr(sim, "p_color") and sim.p_color is not None:
            out["color"] = sim.p_color.to_numpy()
        path = os.path.join(self.out_dir, f"frame_{frame_idx:06d}.npz")
        np.savez_compressed(path, **out)
        return path

    def _save_ply(self, sim, sim_time: float, frame_idx: int):
        x = sim.particles.x.to_numpy()
        c = None
        if self.with_color and hasattr(sim, "p_color") and sim.p_color is not None:
            c = np.clip(sim.p_color.to_numpy() * 255.0, 0, 255).astype(np.uint8)
        path = os.path.join(self.out_dir, f"frame_{frame_idx:06d}.ply")

        n = x.shape[0]
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            if c is not None:
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")

            # Optimized: use numpy string formatting for better performance
            if c is None:
                # Vectorized string formatting for positions only
                coords_str = np.array([f"{x[i, 0]:.7f} {x[i, 1]:.7f} {x[i, 2]:.7f}" for i in range(n)])
                f.write("\n".join(coords_str) + "\n")
            else:
                # Vectorized string formatting for positions and colors
                coords_str = np.array(
                    [f"{x[i, 0]:.7f} {x[i, 1]:.7f} {x[i, 2]:.7f} {int(c[i, 0])} {int(c[i, 1])} {int(c[i, 2])}" for i in
                     range(n)])
                f.write("\n".join(coords_str) + "\n")
        return path

    def export_once(self, sim, sim_time: float, frame_idx: int):
        if self.fmt == "npz": return self._save_npz(sim, sim_time, frame_idx)
        if self.fmt == "ply": return self._save_ply(sim, sim_time, frame_idx)
        return self._save_npz(sim, sim_time, frame_idx)  # fallback


class JSONLogger:
    """
    Optimized JSON logger that writes normalized physical quantities:
      - Forces in SI (N) AND normalized forms:
          * v_hat = |v| * dt / dx (dimensionless, CFL-like)
          * Fn_hat = |Fn| / (E * dx^2) (dimensionless, coarse proxy for contact-normal stress normalized by local stiffness scale)
      - EE pose in meters (SI)
      - Knife animation params
    """

    def __init__(self, out_dir="logs", fruit_name="banana_saw_cut"):  # CHANGE FRUIT NAME
        self.out_dir = str(out_dir)
        self.fruit_name = str(fruit_name)
        os.makedirs(self.out_dir, exist_ok=True)
        # Pre-allocate string buffer for more efficient writing
        self._string_buf = []

    def log_once(self, record: dict):
        # Pretty format with indentation for better readability
        self._string_buf.append(json.dumps(record, indent=2, separators=(',', ': ')))

    def flush(self, final: bool = False):
        if len(self._string_buf) == 0: return

        # Use fruit name in the output file
        file_prefix = f"{self.fruit_name}_" if self.fruit_name else ""
        path = os.path.join(self.out_dir, f"{file_prefix}ee_force_fps.json")

        # Pretty formatted JSON array with proper indentation
        with open(path, "w", encoding="utf-8") as f:
            f.write("[\n")
            for i, json_str in enumerate(self._string_buf):
                if i > 0:
                    f.write(",\n")
                # Add proper indentation for each record
                lines = json_str.split('\n')
                for j, line in enumerate(lines):
                    if j == 0:
                        f.write("  " + line)
                    else:
                        f.write("\n  " + line)
            f.write("\n]")

        print(f"[LOG] wrote {len(self._string_buf)} frames → {path}")
        self._string_buf.clear()


class OutputManager:
    """
    One manager to orchestrate logging and export **on the same FPS schedule**.
    """

    def __init__(self, fps: float, exporter: FPSExporter = None, logger: JSONLogger = None):
        self.sampler = FPSSampler(fps=fps)
        self.exporter = exporter
        self.logger = logger

    def reset(self):
        self.sampler.reset()

    def on_step(self, sim, sim_time: float, make_record_callable):
        """Call each substep; emits both export and log at scheduled times."""
        outs = self.sampler.step_until_now(sim_time)
        written = []
        for t_out, fidx in outs:
            if self.exporter is not None:
                written.append(self.exporter.export_once(sim, t_out, fidx))
            if self.logger is not None:
                rec = make_record_callable(sim, t_out, fidx)
                if rec is not None: self.logger.log_once(rec)
        return written

    def finish(self):
        if self.logger is not None: self.logger.flush(final=True)
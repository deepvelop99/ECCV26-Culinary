"""
Entry point for the MLS‑MPM cutting demo.
- Loads YAML config, builds/loads SDF packs (with NPZ cache), and launches the simulation.
- See README.md for usage examples and CLI options.
"""


#!/usr/bin/env python3
import sys, os, argparse, yaml, json, hashlib, numpy as np
from sdf_utils.mesh_sdf import mesh_to_sdf
from mpmcore.sim import MPMCuttingSim

def _sdf_cache_key(mesh_path: str, transform_block: dict, voxel: int, extra=None) -> str:
    st = os.stat(mesh_path)
    payload = {
        "mesh_path": os.path.abspath(mesh_path),
        "mesh_mtime": int(st.st_mtime),
        "mesh_size": int(st.st_size),
        "voxel": int(voxel),
        "cache_version": 6,  # ⬅ bump version to force rebuild of broken caches
        "extra": extra or {}
    }
    j = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(j.encode("utf-8")).hexdigest()[:16]

def _sdf_cache_path(cache_dir: str, name: str, key: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{name.lower()}.{key}.npz")


def _save_pack_npz(path: str, pack: dict):
    """
    Save a minimal NPZ pack to disk.
    - Only numpy arrays and scalars are serialized (numpy scalars coerced to 0‑D arrays).
    """
    out = {}
    for k, v in pack.items():
        if isinstance(v, np.ndarray):
            out[k] = v
        elif np.isscalar(v):
            # Ensure 0-D array for numpy scalars and Python scalars alike
            out[k] = np.array(v)
    np.savez_compressed(path, **out)


def _load_pack_npz(path: str) -> dict:
    """
    Load NPZ and normalize common scalar fields.
    Also validates presence of mandatory keys and emits actionable errors if the cache is incomplete/corrupt.
    """
    data = np.load(path, allow_pickle=False)
    out = {k: data[k] for k in data.files}

    # Normalize common scalar fields to float
    for key in ("voxel_size", "blade_fraction", "render_offset_y"):
        if key in out:
            out[key] = float(np.array(out[key]).reshape(()))

    # Normalize small vectors (if present)
    for key in ("origin", "asset_bounds_local_lb", "asset_bounds_local_ub"):
        if key in out:
            out[key] = np.asarray(out[key], dtype=np.float32)

    # Hard requirement checks (fail fast with guidance)
    if "voxel_size" not in out:
        raise ValueError(
            f"[cache] Missing 'voxel_size' in NPZ '{path}'. "
            f"Please rebuild SDF (use --rebuild-sdf) or delete the cache file."
        )
    if "sdf" not in out or "origin" not in out:
        raise ValueError(
            f"[cache] Corrupt NPZ '{path}' (missing 'sdf' or 'origin'). "
            f"Please rebuild SDF (use --rebuild-sdf)."
        )
    return out


def _build_or_load_sdf(name: str, mesh_cfg: dict, cache_dir: str, use_cache: bool, force_rebuild: bool):
    """
    Either load SDF from cache or rebuild via mesh_to_sdf().
    Caches are keyed on mesh content and sampling settings.
    """
    mesh_path = mesh_cfg["mesh_path"]
    transform = mesh_cfg.get("initial_transform", {})
    voxel     = int(mesh_cfg["sdf_voxel"])
    extra = {}
    if name.lower() == "knife":
        kb = mesh_cfg.get("blade", {"axis": "Y","fraction": 0.5})
        extra["blade"] = kb
    key       = _sdf_cache_key(mesh_path, transform, voxel, extra=extra)
    cpath     = _sdf_cache_path(cache_dir, name, key)
    if use_cache and (not force_rebuild) and os.path.exists(cpath):
        print(f"[cache] {name}: load SDF ← {cpath}")
        return _load_pack_npz(cpath)

    if name.lower() == "knife":
        pack = mesh_to_sdf(mesh_path, transform, voxel, knife_blade=mesh_cfg.get("blade", {"axis":"Y","fraction":0.5}))
    else:
        pack = mesh_to_sdf(mesh_path, transform, voxel)

    if use_cache:
        _save_pack_npz(cpath, pack)
        print(f"[cache] {name}: saved → {cpath}")
    return pack

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/example_cutting.yaml", help="YAML config (fallback to ./example_cutting.yaml if not found)")
    ap.add_argument("--preview", action="store_true", help="sim OFF, seeding/viewer only")
    ap.add_argument("--run-sim", action="store_true", help="sim ON")
    ap.add_argument("--headless", action="store_true", help="viewer OFF, only simulation")
    ap.add_argument("--radius-scale", type=float, default=1.0, help="render radius scale")
    ap.add_argument("--no-sdf-cache", action="store_true", help="disable SDF cache")
    ap.add_argument("--rebuild-sdf", action="store_true", help="force SDF rebuild")
    ap.add_argument("--sdf-cache-dir", type=str, default="cache_sdf", help="SDF cache folder")
    ap.add_argument("--viewer-camera", type=str, choices=["top", "front", "auto", "manual"], default=None,
                    help="override viewer.camera_mode")
    ap.add_argument("--lock-camera", action="store_true", help="lock camera to initial pose (overrides YAML)")
    ap.add_argument("--unlock-camera", action="store_true", help="do not lock camera even if YAML requests")
    ap.add_argument("--damage-viz", action="store_true", help="enable damage visualization (overrides YAML)")
    ap.add_argument("--no-damage-viz", action="store_true", help="disable damage visualization (overrides YAML)")
    ap.add_argument("--output-fps", type=float, default=None, help="unified FPS for logging & export")
    ap.add_argument("--export-format", type=str, default=None, help="export format: npz|ply")
    ap.add_argument("--export-dir", type=str, default=None, help="export folder")
    ap.add_argument("--log-dir", type=str, default=None, help="logging folder")
    ap.add_argument("--log-chunk", type=int, default=None, help="logging chunk size")
    ap.add_argument("--show-board", action="store_true", help="show board rendering")
    ap.add_argument("--show-grid", action="store_true", help="show grid visualization as red particles")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        alt = "example_cutting.yaml"
        if os.path.exists(alt):
            print(f"[INFO] Config not found at '{cfg_path}', falling back to '{alt}'")
            cfg_path = alt
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    use_cache = (not args.no_sdf_cache)
    force_rebuild = args.rebuild_sdf
    cache_dir = args.sdf_cache_dir

    # CLI damage viz override
    if args.damage_viz:
        cfg.setdefault("damage", {})["damage_visualization"] = True
        print("[INFO] Damage visualization enabled via CLI")
    elif args.no_damage_viz:
        cfg.setdefault("damage", {})["damage_visualization"] = False
        print("[INFO] Damage visualization disabled via CLI")

    # CLI board rendering override
    enable_board = bool(args.show_board)
    if args.show_board:
        cfg.setdefault("viewer", {})["show_board"] = True
        print("[INFO] Board rendering enabled via CLI")
    else:
        cfg.setdefault("viewer", {})["show_board"] = False
        print("[INFO] Board rendering disabled (no board)")
    
    # CLI grid visualization override
    if args.show_grid:
        cfg.setdefault("viewer", {})["show_grid"] = True
        print("[INFO] Grid visualization enabled via CLI")

    # Unified output overrides
    if args.output_fps is not None or args.export_format is not None or args.export_dir is not None or args.log_dir is not None or args.log_chunk is not None:
        cfg.setdefault("output", {})["enabled"] = True
        if args.output_fps is not None:
            cfg["output"]["fps"] = float(args.output_fps)
        # export nested
        if "export" not in cfg.setdefault("output", {}): cfg["output"]["export"] = {}
        if args.export_format is not None:
            cfg["output"]["export"]["format"] = str(args.export_format)
        if args.export_dir is not None:
            cfg["output"]["export"]["out_dir"] = str(args.export_dir)
        # logging nested
        if "logging" not in cfg.setdefault("output", {}): cfg["output"]["logging"] = {}
        if args.log_dir is not None:
            cfg["output"]["logging"]["out_dir"] = str(args.log_dir)
        if args.log_chunk is not None:
            cfg["output"]["logging"]["chunk_size"] = int(args.log_chunk)

    cutting_mesh_pack = _build_or_load_sdf("CuttingMesh", cfg["cutting_mesh"], cache_dir, use_cache, force_rebuild)
    knife_pack  = _build_or_load_sdf("Knife",  cfg["knife"],  cache_dir, use_cache, force_rebuild)
    
    # Print grid and mesh x-coordinate lengths
    world_bounds_min = cfg["world"]["bounds_min"]
    world_bounds_max = cfg["world"]["bounds_max"]
    grid_x_length = world_bounds_max[0] - world_bounds_min[0]
    print(f"[INFO] Grid x-coordinate length: {grid_x_length:.6f}")
    
    if cutting_mesh_pack is not None:
        cutting_mesh_origin = cutting_mesh_pack["origin"]
        cutting_mesh_voxel = cutting_mesh_pack["voxel_size"]
        cutting_mesh_sdf = cutting_mesh_pack["sdf"]
        mesh_x_voxels = cutting_mesh_sdf.shape[0]
        mesh_x_length = mesh_x_voxels * cutting_mesh_voxel
        print(f"[INFO] Cutting mesh x-coordinate length: {mesh_x_length:.6f} (voxels: {mesh_x_voxels}, voxel_size: {cutting_mesh_voxel:.6f})")

    board_pack = None
    if enable_board and ("board" in cfg):
        board_pack = _build_or_load_sdf("Board", cfg["board"], cache_dir, use_cache, force_rebuild)
        # Auto-place board under cutting mesh with small gap
        if board_pack is not None and cutting_mesh_pack is not None:
            cutting_mesh_origin = cutting_mesh_pack["origin"]; cutting_mesh_voxel = cutting_mesh_pack["voxel_size"]; cutting_mesh_sdf = cutting_mesh_pack["sdf"]
            bsolid = cutting_mesh_sdf < 0.0
            if np.any(bsolid):
                ys = np.where(bsolid)[1]
                cutting_mesh_bottom = cutting_mesh_origin[1] + (int(ys.min()) + 0.5) * cutting_mesh_voxel
                print(f"[info] Cutting mesh bottom: {cutting_mesh_bottom:.6f}")

                board_origin = board_pack["origin"]; board_voxel = board_pack["voxel_size"]; board_sdf = board_pack["sdf"]
                bsolid = board_sdf < 0.0
                if np.any(bsolid):
                    ys_b = np.where(bsolid)[1]
                    y_min_b = int(ys_b.min()); y_max_b = int(ys_b.max())
                    board_top = board_origin[1] + (y_max_b + 0.5) * board_voxel
                    print(f"[info] Board top (before offset): {board_top:.6f}")

                    world_bounds_min = cfg["world"]["bounds_min"]
                    world_bounds_max = cfg["world"]["bounds_max"]
                    grid_resolution = cfg["world"]["grid_resolution"]
                    dx = (world_bounds_max[1] - world_bounds_min[1]) / grid_resolution
                    gap = 1 * dx  # Reduced gap for closer contact

                    # Target board bottom (to touch floor particles)
                    floor_height = world_bounds_min[1] + 0.05 * dx  # Floor particle height (increased from 0.01 to 0.05)
                    board_bottom = board_origin[1] + (y_min_b + 0.5) * board_voxel
                    board_floor_gap = float(cfg.get("board", {}).get("floor_gap", 0.1)) * dx  # Configurable gap between board and floor
                    # Use gap to adjust board position relative to cutting mesh
                    target_board_bottom = floor_height + board_floor_gap + gap
                    offset = target_board_bottom - board_bottom
                    board_pack["render_offset_y"] = float(offset)
                    print(f"[info] Board render offset Y = {offset:.6f} (gap={gap:.6f})")
                    print(f"[info] Floor height: {floor_height:.6f}, Board bottom: {board_bottom:.6f}")
                    print(f"[info] Board top (after offset): {board_top + offset:.6f}")
                    banana_board_gap = (board_top + offset) - cutting_mesh_bottom
                    print(f"[info] Banana-Board gap: {banana_board_gap:.6f}")

    cfg["cutting_mesh_pack"] = cutting_mesh_pack

    viewer = (not args.headless)
    vcfg = cfg.get("viewer", {})
    cfg_cam_mode = str(vcfg.get("camera_mode", "top")).lower()
    initial_pose = vcfg.get("initial_camera", {
        "eye":    [0.40, 0.55, 0.40],
        "lookat": [0.00, 0.10, 0.00],
        "up":     [0.00, 0.0, -1.0],
        "fov":    35.0,
    })
    lock_on_run = bool(vcfg.get("lock_on_run", True))
    if args.viewer_camera is not None:
        cfg_cam_mode = args.viewer_camera
    if args.lock_camera:   lock_on_run = True
    if args.unlock_camera: lock_on_run = False
    if args.run_sim and (not args.unlock_camera):
        lock_on_run = True

    sim = MPMCuttingSim(cfg, cutting_mesh_pack, knife_pack, board_pack,
                        viewer=viewer,
                        viewer_radius_scale=args.radius_scale,
                        viewer_camera_mode=cfg_cam_mode,
                        viewer_initial_pose=initial_pose,
                        viewer_lock_on_run=lock_on_run)

    if args.preview and not args.run_sim:
        sim.run(run_sim=False)
        return
    if args.run_sim:
        sim.run(run_sim=True)
        return
    print("\n[hint] specify '--preview' or '--run-sim'")

if __name__ == "__main__":
    main()

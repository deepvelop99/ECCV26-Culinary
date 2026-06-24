# CulinaryCut: A Physics-aware VLA Benchmark for Food Cutting

**ECCV 2026** | [Project Page](https://deepvelop99.github.io/iprf.github.io/) | [Paper](https://drive.google.com/file/d/10vp2SWByAJGr-Ccrw68c2hTepsMr5di9/view?usp=sharing) | [Google Drive](https://drive.google.com/drive/folders/1SwtYeItEToyYlJ2Sn1m8qqSIndDWxIHN?usp=sharing)

Dataset generation pipeline for **CulinaryCut** — a robot food-cutting benchmark that couples Taichi MLS-MPM physics simulation with ManiSkill v3 robot environments to produce physically-grounded trajectory datasets.

---

## Repository Structure

```
EEF-Cutting-Simulation/     # Taichi MLS-MPM physics engine (fruit deformation & cutting forces)
│   ├── run.py              # Standalone MPM entry point
│   ├── mpmcore/            # MPM solver (P2G/G2P, elasticity, plasticity, cutting)
│   ├── configs/            # Per-fruit material configs (banana.yaml, apple.yaml, ...)
│   └── assets/             # Fruit & knife 3D meshes

mani_skill/                 # ManiSkill v3 fork with knife-cutting environments
│   ├── envs/tasks/knife/   # 40+ cutting tasks (per-fruit × cut count)
│   ├── dataset_converters/ # Data collection + RDT / Octo / OpenVLA conversion
│   └── assets/             # Fruit meshes and textures
```

---

## Generating Data

### Setup

```bash
export LD_LIBRARY_PATH=/workspace/envs/maniskill/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/data:/data/mani_skill
export TI_ARCH=cuda
```

### Basic collection

```bash
cd /data/EEF-Cutting-Simulation

python /data/mani_skill/dataset_converters/collect_rollouts_mpm.py \
    --task bananacut \
    --fixed-object banana \
    --mpm-config configs/banana.yaml \
    --out /data/datasets/maniskill_mpm \
    --num-episodes 100 \
    --seed-base 0
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | `bananacut` | Task name — `<fruit>cut`, `<fruit>cut2`, `<fruit>cut_bgvar`, ... |
| `--fixed-object` | `banana` | Fruit type |
| `--mpm-config` | required | Path to fruit YAML (`EEF-Cutting-Simulation/configs/<fruit>.yaml`) |
| `--out` | required | Output directory |
| `--num-episodes` | `3` | Episodes to collect |
| `--seed-base` | `0` | RNG seed for first episode; episode `i` uses `seed_base + i` |
| `--obs-mode` | `rgb` | `rgb` \| `rgbd` \| `state` \| `none` |
| `--bg-scenes` | off | Enable apartment backdrop variation (16 scenes) |
| `--bg-only-idx` | `-1` | Pin all episodes to one background index |
| `--save-video` | off | Write `0.mp4` alongside the HDF5 |

### Supported fruits

`banana` · `apple` · `orange` · `melon` · `peach` · `strawberry` · `cucumber` · `cherry` · `grape` · `lemon` · `kiwi` · `plum` · `tomato` · `pear` · `shine_muscat`

---

## Output Format

Each episode produces:

```
<out>/<task>/auto_<seed>/
├── trajectory.h5       # Robot trajectories + observations
├── trajectory.json     # Seed, variation params, task config
└── alignment.json      # Per-step MPM force/velocity telemetry
```

**`trajectory.h5` structure:**

```
traj_0/
├── observations/sensor_data/base_camera/rgb   (T, 256, 256, 3) uint8
├── observations/agent/qpos                    (T, 9)  float32   # 7-DoF arm + 2 gripper
├── observations/agent/qvel                    (T, 9)  float32
├── actions                                    (T-1, 7) float32  # EE delta pose + gripper
└── rewards                                    (T-1,)  float32
```

---

## Scene & Seed

The seed controls all per-episode randomization:

| Component | Range |
|---|---|
| Fruit scale | Per-fruit (e.g. banana 0.8–1.2×) |
| Position offset | ±2–5 cm XY |
| Yaw rotation | 0–360° |
| Lab scene preset | 5 variants (lighting, board color) |
| Background (`--bg-scenes`) | 0 = lab; 1–15 = apartment scenes |

For reproducible single-background runs:

```bash
python mani_skill/dataset_converters/collect_rollouts_mpm.py \
    --task bananacut_bgvar --fixed-object grape \
    --mpm-config EEF-Cutting-Simulation/configs/grape.yaml \
    --out /data/datasets/grape_bg3 \
    --num-episodes 200 --seed-base 1000 --bg-only-idx 3
```

---

## Dataset Conversion

```bash
# → RDT
python mani_skill/dataset_converters/to_rdt/convert.py --input <raw> --output <out>

# → Octo
python mani_skill/dataset_converters/to_octo/build_rlds.py --input <raw> --output <out>

# → OpenVLA
python mani_skill/dataset_converters/to_openvla/build_rlds.py --input <raw> --output <out>
```

---

## Citation

```bibtex
@inproceedings{culinarycut2026,
  title     = {CulinaryCut: A Physics-aware Vision-Language-Action Benchmark for Food Cutting via Material Point Method},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

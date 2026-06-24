# (ECCV 2026) CulinaryCut: A Physics-aware Vision-Language-Action Benchmark for Food Cutting via Material Point Method

Official Implementation of ECCV 2026 paper: "CulinaryCut: A Physics-aware Vision-Language-Action Benchmark for Food Cutting via Material Point Method"

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://deepvelop99.github.io/iprf.github.io/)
[![Accepted](https://img.shields.io/badge/Status-Accepted-green)](https://drive.google.com/file/d/10vp2SWByAJGr-Ccrw68c2hTepsMr5di9/view?usp=sharing)
[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-blue)](https://eccv.ecva.net/)

---

## Overview

CulinaryCut is a physics-aware robot learning benchmark for food-cutting tasks. It couples a **Taichi MLS-MPM continuum mechanics simulator** (EEF-Cutting-Simulation) with **ManiSkill v3** robot environments to generate diverse, physically-grounded trajectory datasets for training Vision-Language-Action (VLA) models.

```
EEF-Cutting-Simulation/   ← Taichi MLS-MPM physics (fruit deformation, cutting forces)
mani_skill/               ← ManiSkill v3 fork: robot envs + scripted rollout collection
RoboticsDiffusionTransformer/  ← RDT-1B training code + dataset conversion pipelines
Cutting_rebuttal/         ← Rebuttal experiments (15 fruits)
Cutting_rubuttal2/        ← Rebuttal experiments v2 (updated bridges)
```

---

## Repository Structure

### `EEF-Cutting-Simulation/`
Standalone Taichi MLS-MPM physics engine for knife-cutting simulation.

```
EEF-Cutting-Simulation/
├── run.py                  # Main entry point
├── mpmcore/sim.py          # MPM core (P2G/G2P, elasticity, plasticity, cutting logic)
├── mpmcore/colliders.py    # SDF-based knife and board colliders
├── physics/                # Elastic/plastic material models
├── sdf_utils/              # Mesh → SDF conversion
├── configs/                # Per-fruit YAML configs (material properties)
└── assets/                 # Fruit 3D meshes (.obj/.mtl)
```

### `mani_skill/`
ManiSkill v3 fork with cutting task environments and dataset generation tools.

```
mani_skill/
├── envs/tasks/knife/       # 40+ cutting task environments (per-fruit × cut count)
├── agents/robots/          # Robot definitions (panda, xarm6, etc.)
├── dataset_converters/     # Data collection + format conversion scripts
│   ├── collect_rollouts_mpm.py   # Main co-simulation data collector
│   ├── to_rdt/             # → RDT HDF5 format
│   ├── to_octo/            # → Octo RLDS format
│   └── to_openvla/         # → OpenVLA RLDS format
└── evaluation/             # Evaluation harness
```

---

## Dataset Generation

### Prerequisites

```bash
# Environment setup
conda activate maniskill  # or your env with ManiSkill v3 + Taichi installed

export LD_LIBRARY_PATH=/workspace/envs/maniskill/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/data:/data/mani_skill
export TI_ARCH=cuda        # Taichi backend: cuda | vulkan | cpu
```

### Step 1: Run the MPM + ManiSkill Co-Simulation

The main collection script drives a scripted Panda arm through a fruit-cutting task, coupling ManiSkill physics with the MPM solver each step:

```bash
cd /data/EEF-Cutting-Simulation

python /data/mani_skill/dataset_converters/collect_rollouts_mpm.py \
    --task bananacut \
    --fixed-object banana \
    --mpm-config /data/EEF-Cutting-Simulation/configs/banana.yaml \
    --out /data/datasets/maniskill_mpm \
    --num-episodes 100 \
    --seed-base 0
```

#### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | `bananacut` | ManiSkill task name (see supported tasks below) |
| `--fixed-object` | `banana` | Fruit type to use |
| `--mpm-config` | required | Path to fruit YAML (material properties) |
| `--out` | required | Output root directory |
| `--num-episodes` | `3` | Number of episodes to collect |
| `--seed-base` | `0` | RNG seed for first episode; episode `i` uses `seed_base + i` |
| `--obs-mode` | `rgb` | Observation type: `rgb` \| `rgbd` \| `state` \| `none` |
| `--knife-speed-mps` | `0.10` | Knife descent speed (m/s) |
| `--bg-scenes` | off | Enable apartment-stage backdrop variation (bg_idx 0–15) |
| `--bg-only-idx` | `-1` | Pin all episodes to a single background index |
| `--save-video` | off | Write `0.mp4` alongside the HDF5 |
| `--render-mpm-mp4` | off | Write MPM particle visualization `mpm.mp4` |
| `--apply-mpm-force` | off | Bidirectional force feedback (MPM → robot) |
| `--mpm-force-scale` | `0.25` | Force feedback scale factor |

#### Motion Control Arguments

| Argument | Default | Description |
|---|---|---|
| `--robot-uid` | `pkour` | Robot definition (must support `pd_ee_delta_pos`) |
| `--control-mode` | `pd_ee_delta_pose` | ManiSkill control mode |
| `--approach-x` | `-0.25` | Knife approach X offset (m) |
| `--approach-z` | `0.50` | Knife approach Z height (m) |
| `--cut-z` | `0.025` | Target Z at bottom of cut stroke (m) |
| `--cut-steps` | `30` | Number of steps for the cut phase |

#### Variation Arguments

| Argument | Default | Description |
|---|---|---|
| `--no-scale` | off | Disable random fruit scale variation |
| `--no-pos` | off | Disable random fruit position offset |
| `--no-yaw` | off | Disable random fruit yaw rotation |

---

### Step 2: Batch Collection (Parallel)

For large-scale collection across multiple fruits/backgrounds, use the parallel launcher:

```bash
python /data/mani_skill/dataset_converters/launch_parallel_collect.py \
    --fruits banana apple orange cucumber peach \
    --num-episodes 500 \
    --out /data/datasets/maniskill_mpm_large \
    --seed-base 0
```

For background variation collection across all 15 apartment scenes:

```bash
bash /data/mani_skill/dataset_converters/run_bgvar_pilot.sh
```

---

## Supported Fruits & Tasks

### Fruit Types (14 varieties)

| Fruit | Config | Notes |
|---|---|---|
| banana | `configs/banana.yaml` | Reference fruit; tuned baseline |
| apple | `configs/apple.yaml` | High stiffness (6.5 MPa Young's modulus) |
| orange | `configs/orange.yaml` | |
| melon | `configs/melon.yaml` | Large scale |
| peach | `configs/peach.yaml` | |
| strawberry | `configs/strawberry.yaml` | |
| cucumber | `configs/cucumber.yaml` | Elongated geometry |
| cherry | `configs/cherry.yaml` | Very small (~25 mm) |
| grape | `configs/grape.yaml` | Small, spherical |
| lemon | `configs/lemon.yaml` | |
| kiwi | `configs/kiwi.yaml` | |
| plum | `configs/plum.yaml` | |
| tomato | `configs/tomato.yaml` | Low stiffness (7 kPa) |
| pear | `configs/pear.yaml` | |
| shine_muscat | `configs/shine_muscat.yaml` | |

### Task Naming Convention

Tasks follow the pattern `<fruit>cut[_<variant>]`:

```
bananacut              # 1-cut, lab scene
bananacut_bgvar        # 1-cut, background variation enabled
applecut               # 1-cut
bananacut2             # 2-cut
bananacut3             # 3-cut
...
```

---

## Scene & Seed Configuration

### Seed-based Reproducibility

Each episode deterministically seeds all variation with:
```
episode_seed = seed_base + episode_index
rng = np.random.default_rng(seed)
```

The seed controls: fruit scale, position offset, yaw rotation, scene index, and background index.

### Scene Variation Components

| Component | Range | Description |
|---|---|---|
| `scale` | Per-fruit (e.g. banana: 0.8–1.2×) | Fruit size randomization |
| `pos_offset_x` | [-0.02, 0.05] m | Fruit X position jitter |
| `pos_offset_y` | [-0.05, 0.05] m | Fruit Y position jitter |
| `yaw` | [0°, 360°] | Fruit rotation around Z-axis |
| `scene_idx` | 0–4 | Lab preset (board/ground colors + lighting) |
| `bg_idx` | 0–15 | 0 = lab; 1–15 = apartment backgrounds |

### Background Scene Index

| `bg_idx` | Scene | Type |
|---|---|---|
| 0 | Lab tabletop | Default |
| 1, 3, 5, 7, 9, 11, 13, 15 | Apartment (with cutting board) | ReplicaCAD / AI2THOR |
| 2, 4, 6, 8, 10, 12, 14 | Apartment (bare table) | ArchitecTHOR / ProcTHOR |

> **Note**: Small fruits (cherry, grape, shine_muscat) are recommended for `has_board` backgrounds (odd indices 1–15) so the raised surface provides a reliable cutting plane.

### Example: Reproducible Single-Background Collection

```bash
# Collect 200 grape episodes all on background 3 (apartment with board), starting from seed 1000
python mani_skill/dataset_converters/collect_rollouts_mpm.py \
    --task bananacut_bgvar \
    --fixed-object grape \
    --mpm-config Cutting_rubuttal2/configs/fruits/grape.yaml \
    --out /data/datasets/grape_bg3 \
    --num-episodes 200 \
    --seed-base 1000 \
    --bg-only-idx 3
```

---

## Output Data Structure

### Per-Episode Output Layout

```
<out>/<task>/auto_<seed>/
├── trajectory.h5       # Robot trajectories + observations (RDT-compatible)
├── trajectory.json     # Episode metadata (seed, variation params, task config)
├── alignment.json      # F/V telemetry, contact markers, MPM physics log
└── 0.mp4               # RGB video (if --save-video)
```

### HDF5 Format (`trajectory.h5`)

```
traj_0/
├── observations/
│   ├── sensor_data/
│   │   ├── base_camera/rgb        (T, 256, 256, 3)  uint8   — primary RGB
│   │   └── hand_camera/rgb        (T, 256, 256, 3)  uint8   — wrist RGB (optional)
│   └── agent/
│       ├── qpos                   (T, 9)  float32   — joint positions (7-DoF + 2 gripper)
│       └── qvel                   (T, 9)  float32   — joint velocities
├── actions                        (T-1, 7) float32  — EE delta pose + gripper
├── rewards                        (T-1,)   float32
└── env_states/
    └── articulations/pkour        (T+1, 31) float32
```

### `trajectory.json` (Episode Metadata)

```json
{
  "task": "bananacut",
  "seed": 42,
  "variation": {
    "object": "banana",
    "scale": 1.05,
    "pos_offset": [0.01, -0.02],
    "yaw": 0.0,
    "scene_idx": 2,
    "bg_idx": 0
  },
  "num_steps": 180,
  "success": true
}
```

### `alignment.json` (Physics Telemetry)

Per-step log of:
- Knife force (N) from MPM contact
- Knife velocity (m/s)
- Cut progress markers (entry, mid, exit frames)
- MPM particle stats (active count, max stress)

---

## MPM Physics Configuration

Per-fruit YAML files define material properties for the MPM solver:

```yaml
# configs/banana.yaml (example structure)
fruit:
  name: banana
  mesh: assets/Banana.obj

material:
  E: 50000          # Young's modulus (Pa) — banana ~50 kPa
  nu: 0.45          # Poisson's ratio
  yield_stress: 800 # J2 yield stress (Pa)
  rho: 900          # Density (kg/m³)

simulation:
  voxel_size: 0.004
  n_particles: 8000
  substeps: 40
  cut_mode: general   # general | saw_cut
```

### Material Properties Across Fruits

| Fruit | Young's Modulus | Notes |
|---|---|---|
| Tomato | ~7 kPa | Softest |
| Banana | ~50 kPa | Reference |
| Peach | ~100 kPa | |
| Kiwi | ~500 kPa | |
| Apple | ~6.5 MPa | Stiffest |

Spans **3 orders of magnitude** in stiffness, validating the simulator across physically diverse materials.

---

## Dataset Conversion

Convert collected HDF5 trajectories to formats for different VLA models:

```bash
# → RDT HDF5
python mani_skill/dataset_converters/to_rdt/convert.py \
    --input /data/datasets/maniskill_mpm \
    --output /data/datasets/cuttingvla_rdt_hdf5

# → Octo RLDS
python mani_skill/dataset_converters/to_octo/build_rlds.py \
    --input /data/datasets/maniskill_mpm \
    --output /data/datasets/cuttingvla_octo_rlds

# → OpenVLA RLDS
python mani_skill/dataset_converters/to_openvla/build_rlds.py \
    --input /data/datasets/maniskill_mpm \
    --output /data/datasets/cuttingvla_openvla_rlds
```

---

## Standalone MPM Simulation

To run the physics simulator without the robot environment (verification / visualization):

```bash
cd EEF-Cutting-Simulation

python run.py \
    --config configs/banana.yaml \
    --render          # open Taichi UI viewer
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

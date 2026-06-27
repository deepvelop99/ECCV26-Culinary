# CulinaryCut: A Physics-Aware VLA Benchmark for Food Cutting

**ECCV 2026** &nbsp;|&nbsp; [Project Page](https://deepvelop99.github.io/culinary.github.io/) &nbsp;|&nbsp; [arXiv](https://arxiv.org/abs/2601.06451) &nbsp;|&nbsp; [Paper](https://drive.google.com/file/d/1MD6uJXdBlgY86GfR1rZ3JY5KIk2vcO0l/view?usp=sharing) &nbsp;|&nbsp; [Dataset](https://drive.google.com/drive/folders/1SwtYeItEToyYlJ2Sn1m8qqSIndDWxIHN?usp=sharing)

Dataset generation pipeline for **CulinaryCut** — coupling Taichi MLS-MPM physics simulation with ManiSkill v3 to produce physically-grounded robot food-cutting trajectories.

---

## Repository Structure

```
EEF-Cutting-Simulation/     # Taichi MLS-MPM physics engine
│   ├── run.py              # Standalone MPM entry point
│   ├── mpmcore/            # MPM solver (elasticity, plasticity, cutting)
│   ├── configs/            # Per-fruit material configs (*.yaml)
│   └── assets/             # Fruit & knife meshes

mani_skill/                 # ManiSkill v3 with knife-cutting environments
│   ├── envs/tasks/knife/   # Cutting tasks (per-fruit × cut count)
│   ├── dataset_converters/ # Collection + RDT / Octo / OpenVLA conversion
│   └── assets/             # Fruit meshes and textures
```

---

## Setup

```bash
export LD_LIBRARY_PATH=/workspace/envs/maniskill/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/data:/data/mani_skill
export TI_ARCH=cuda
```

---

## Data Collection

```bash
python /data/mani_skill/dataset_converters/collect_rollouts_mpm.py \
    --task bananacut \
    --fixed-object banana \
    --mpm-config EEF-Cutting-Simulation/configs/banana.yaml \
    --out /data/datasets/maniskill_mpm \
    --num-episodes 100
```

**Key arguments**

| Argument | Description |
|---|---|
| `--task` | `<fruit>cut`, `<fruit>cut2`, `<fruit>cut_bgvar`, ... |
| `--fixed-object` | Fruit type (see supported list below) |
| `--mpm-config` | Path to fruit YAML |
| `--out` | Output directory |
| `--num-episodes` | Episodes to collect (default: `3`) |
| `--seed-base` | RNG seed for first episode (default: `0`) |
| `--bg-scenes` | Enable apartment backdrop variation |
| `--save-video` | Write `0.mp4` alongside the HDF5 |

**Supported fruits**

`banana` · `apple` · `orange` · `melon` · `peach` · `strawberry` · `cucumber` · `cherry` · `grape` · `lemon` · `kiwi` · `plum` · `tomato` · `pear` · `shine_muscat`

---

## Output Format

```
<out>/<task>/auto_<seed>/
├── trajectory.h5       # Observations + actions
├── trajectory.json     # Task config & variation params
└── alignment.json      # Per-step MPM force/velocity telemetry
```

---

## Dataset Conversion

```bash
# RDT
python mani_skill/dataset_converters/to_rdt/convert.py --input <raw> --output <out>

# Octo
python mani_skill/dataset_converters/to_octo/build_rlds.py --input <raw> --output <out>

# OpenVLA
python mani_skill/dataset_converters/to_openvla/build_rlds.py --input <raw> --output <out>
```

---

## Citation

```bibtex
@inproceedings{koh2026culinarycut,
  title     = {CulinaryCut: A Physics-Aware Benchmark for
               Robotic Food Cutting in Deformable Simulation},
  author    = {Koh, Hyunseo and Song, Chang-Yong and Choi, Youngjae
               and Viveiros, Misa and Hyde, David and Kim, Heewon},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

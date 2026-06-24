# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ManiSkill v3 is a robotic manipulation simulation framework built on SAPIEN (PhysX backend). It provides a Gymnasium-compatible API for robotics research with support for both CPU and GPU-parallelized simulation across 100+ environments and 20+ robot types.

## Running Tasks

```bash
# Run a task with random actions (main demo entry point)
python -m mani_skill.examples.demo_random_action -e PushCube-v1 -n 4 -b gpu

# Key flags for demo_random_action
#   -e / --env-id         environment ID (default: PushCube-v1)
#   -n / --num-envs       parallel environments (default: 4)
#   -b / --sim-backend    'auto', 'cpu', 'gpu'
#   -o / --obs-mode       'none', 'state', 'rgb', 'rgbd', 'depth', 'point_cloud'
#   -c / --control-mode   e.g. 'pd_ee_pose', 'pd_joint_pos'
#   --render-mode         'human' (GUI), 'rgb_array', 'none'

# Run robot demonstration
python -m mani_skill.examples.demo_robot

# Run the EEF cutting simulation (Taichi-based MLS-MPM)
python EEF-Cutting-Simulation/run.py
```

## Environment Variables

- `MS_ASSET_DIR` — override asset/demo storage location (default: `~/.maniskill/`)
  - Assets stored at `$MS_ASSET_DIR/data/`
  - Demos stored at `$MS_ASSET_DIR/demos/`

## Architecture

### Core Classes

- **`envs/sapien_env.py` — `BaseEnv`**: Root class for all environments (inherits `gym.Env`). Manages the simulation loop, reset logic (reconfiguration → episode initialization), observation collection, and rendering. All tasks subclass this.

- **`envs/scene.py` — `ManiSkillScene`**: Manages the collection of SAPIEN sub-scenes that back GPU-parallel simulation. Handles actor/articulation creation across parallel environments and coordinates GPU/CPU memory.

- **`agents/base_agent.py` — `BaseAgent`**: Base for all robots. Wraps SAPIEN articulations with controller logic, sensor mounting, and action/observation interfaces.

### Registration System

Both environments and agents use decorator-based registration:

```python
from mani_skill.utils.registration import register_env

@register_env("MyTask-v1", max_episode_steps=200)
class MyTask(BaseEnv):
    ...
```

After importing the file, `gym.make("MyTask-v1")` works.

### Creating Custom Environments

Copy `envs/template.py` as a starting point. The annotated minimal example is `envs/tasks/tabletop/push_cube.py`. Required override methods:

| Method | Purpose |
|---|---|
| `_load_scene(options)` | Load actors/articulations into the scene (called once on reconfigure) |
| `_initialize_episode(env_idx, options)` | Randomize poses/goals per reset (batched) |
| `evaluate(obs)` | Return dict with `"success"` and/or `"fail"` bool tensors of shape `(num_envs,)` |
| `compute_dense_reward(obs, action, info)` | Return reward tensor of shape `(num_envs,)` |
| `compute_normalized_dense_reward(...)` | Same but normalized to [0, 1] |

Optional but important:
- `_default_sim_config` — tune `GPUMemoryConfig` for your task's parallel scale
- `_default_sensor_configs` — define observation cameras via `CameraConfig`
- `_default_human_render_camera_configs` — cameras for `env.render()`
- `get_state_dict` / `set_state_dict` — for trajectory replay (include non-sim state like goal positions)

### Task Directory Structure

```
envs/tasks/
├── tabletop/       # table-top manipulation tasks
├── control/        # locomotion/control tasks
├── dexterity/      # dexterous hand tasks
├── digital_twins/  # real-world replica environments (ReplicaCAD, AI2THOR)
├── humanoid/       # humanoid robot tasks
├── knife/          # cutting tasks
├── mobile_manipulation/
└── quadruped/
```

### Robots (`agents/robots/`)

22+ robots including: `panda`, `fetch`, `xarm6`, `unitree_g1`, `unitree_h1`, `unitree_go`, `googlerobot`, `allegro_hand`, `inspire_hand`, `dclaw`, `anymal`, `stompy`, `widowx`, `so100`, `lerobot`, `koch`.

### Controllers (`agents/controllers/`)

Common control modes: `pd_joint_pos`, `pd_joint_vel`, `pd_ee_pose`, `pd_ee_delta_pos`, `pd_ee_delta_pose`, base velocity controllers for mobile robots.

### GPU vs CPU Simulation

- GPU simulation: all parallel envs share a single SAPIEN scene subdivided into sub-scenes; all tensors are batched
- CPU simulation: each env may reconfigure independently (supports asset swapping mid-training)
- Code must be written in **batched form** (operate on tensors of shape `(num_envs, ...)`) regardless of backend
- `env_idx` parameter in `_initialize_episode` indicates which sub-envs are being reset (supports partial resets)

### Scene Builders (`utils/scene_builder/`)

Pre-built scene configurations for common setups: `TableSceneBuilder`, `ReplicaCADSceneBuilder`, `AI2THORSceneBuilder`. Use these in `_load_scene` to avoid manual scene construction.

### Asset System (`utils/building/`)

- `actors` module — helpers for loading primitive shapes and URDF/MJCF assets
- Assets are registered with groups and downloaded on demand to `MS_ASSET_DIR`

### Wrappers (`utils/wrappers/`)

- `RecordEpisode` — records video and trajectory data
- Standard vectorization wrappers for RL training pipelines

### Dataset Converters (`dataset_converters/`)

Converters for training robot learning models: RDT, OpenVLA, Octo. Used to transform collected trajectories into model-specific training datasets.

### EEF Cutting Simulation (`EEF-Cutting-Simulation/`)

Separate Taichi-based MLS-MPM physics module for knife-cutting tasks. Has its own entry point (`run.py`) and is independent of the main SAPIEN simulation loop.

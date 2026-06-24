# dataset_converters

One scripted rollout source → three model-specific datasets (RDT / OpenVLA / Octo).

## Alignment & force policy (ratified A+C)

- **Pass gate = velocity alignment only.** At each control tick, MPM knife
  velocity (finite-diff of bridge-mapped tip) must match ManiSkill TCP z
  velocity within `vel_tol` (0.05 m/s) on descent frames. Empirically
  converges to `corr ≈ 1.0` under the current bridge.
- **Force label = MPM-only.** `mpm_knife_force` (integrated grid impulse per
  `sim.dt`) is the physically-grounded force label used for training.
  `ms_knife_force` is kept in the Episode for debugging (e.g. spotting
  banana/board confusion) but should *not* be used as a supervised label:
  bananacut's banana has near-zero rigid collision, so ManiSkill's knife
  force reports board contact, not banana cutting.

## Pipeline

```
collect_rollouts.py                ──▶  intermediate .npz (per episode)
   (pd_ee_delta_pose + pd_joint_pos)      common/episode.py

intermediate ──▶ to_rdt/build_rdt_hdf5.py        (joint-mode episodes only)
intermediate ──▶ to_openvla/build_openvla_rlds.py (EE-mode episodes)
intermediate ──▶ to_octo/build_octo_rlds.py       (EE-mode episodes)
```

## Why two control modes

- RDT (`eval_rdt_maniskill.py`) rolls out under `pd_joint_pos` — action = 7 joint targets + gripper.
- OpenVLA (`eval_openvla.py`) and Octo (`eval_octo.py`) roll out under `pd_ee_delta_pose` — action = 3 pos delta + 3 rot delta + 1 gripper.

Collecting under each mode separately avoids replaying an action that wasn't actually executed, which would create a train/eval mismatch.

## Hz

Collected at ManiSkill default (sim_freq=100, control_freq=20). The same 20Hz data is reused for all three models; each framework's config records the Hz:

- RDT: `configs/dataset_control_freq.json`
- OpenVLA: implicit in RLDS step spec
- Octo: `dataset_statistics.json` (this script emits it)

## Commands

Assume the env set up earlier:

```bash
export LD_LIBRARY_PATH=/workspace/envs/maniskill/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/data
PY=/workspace/envs/maniskill/bin/python

# 1. Collect (needs GPU/Vulkan host — submit as job)
$PY dataset_converters/collect_rollouts.py \
    --task bananacut --num-episodes 50 \
    --out /data/datasets/maniskill_intermediate

# 2a. RDT
$PY dataset_converters/to_rdt/build_rdt_hdf5.py \
    --in /data/datasets/maniskill_intermediate \
    --out /data/datasets/rdt-ft-data/demo_1k \
    --task bananacut

# 2b. OpenVLA
$PY dataset_converters/to_openvla/build_openvla_rlds.py \
    --in /data/datasets/maniskill_intermediate \
    --out /data/datasets/openvla/maniskill_bananacut \
    --task bananacut

# 2c. Octo
$PY dataset_converters/to_octo/build_octo_rlds.py \
    --in /data/datasets/maniskill_intermediate \
    --out /data/datasets/octo/maniskill_bananacut \
    --task bananacut
```

## Extending to new tasks

1. Add instruction to `common/instructions.json`.
2. If the task uses a non-`block_apple` target actor, edit `collect_rollouts.collect_one` (the `getattr(u, "block_apple", ...)` line).
3. Run the three converters with `--task <new-task>`.

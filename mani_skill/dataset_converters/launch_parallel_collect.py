"""Parallel collection launcher.

Splits (objects × episodes) into kube jobs and `kubectl apply`s them.
Each job handles a single (object, seed_range) slice so that N jobs run
in parallel on available H200s.

Usage:
    python launch_parallel_collect.py \
        --objects banana apple cucumber melon orange peach strawberry \
        --episodes-per-object 500 \
        --shards-per-object 10 \
        --out-root /data/datasets/maniskill_mpm_full \
        --dry-run
    # inspect printed YAMLs, then drop --dry-run

Outputs end up at <out-root>/<object_task_id>/auto_<seed>/... uniformly.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_OBJECTS = [
    "banana", "apple", "cucumber", "melon", "orange", "peach", "strawberry"
]

# Per-object MPM config (assumed present in EEF repo).
MPM_CONFIG = {
    "banana":     "/data/EEF-Cutting-Simulation/configs/banana.yaml",
    "apple":      "/data/EEF-Cutting-Simulation/configs/apple.yaml",
    "cucumber":   "/data/EEF-Cutting-Simulation/configs/cucumber.yaml",
    "melon":      "/data/EEF-Cutting-Simulation/configs/melon.yaml",
    "orange":     "/data/EEF-Cutting-Simulation/configs/orange.yaml",
    "peach":      "/data/EEF-Cutting-Simulation/configs/peach.yaml",
    "strawberry": "/data/EEF-Cutting-Simulation/configs/strawberry.yaml",
}

# Currently the env is registered as "bananacut" but accepts any mesh_path via
# variation. We reuse "bananacut" as the env id for all fruits; the resulting
# auto_<seed>/ dir hierarchy is grouped by --out-root/<object>/ externally.
ENV_ID = "bananacut"

JOB_TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: p-test2
  labels:
    management.mlx.navercorp.com/project: test2
    management.mlx.navercorp.com/workspace: realitylab
    mlx.navercorp.com/zone: private-h200-realitylab-0
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        management.mlx.navercorp.com/project: test2
        management.mlx.navercorp.com/workspace: realitylab
        mlx.navercorp.com/zone: private-h200-realitylab-0
      annotations:
        mlx.navercorp.com/zone: private-h200-realitylab-0
    spec:
      restartPolicy: Never
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: kubernetes.io/hostname
                    operator: NotIn
                    values:
                      - h200-03-w-78dc
                      - h200-03-w-7271
      containers:
        - name: pytorch
          image: mlx-public.kr.ncr.ntruss.com/mlx/notebook/kubeflow-jupyter:2.7.0
          imagePullPolicy: IfNotPresent
          command: ["/bin/bash", "-lc"]
          args:
            - |
              set -euo pipefail
              nvidia-smi --query-gpu=index,name --format=csv,noheader || true
              export LD_LIBRARY_PATH=/workspace/envs/maniskill/lib:${{LD_LIBRARY_PATH:-}}
              export PYTHONPATH=/data:/data/mani_skill:${{PYTHONPATH:-}}
              export PYTHONUNBUFFERED=1
              export TI_ARCH=cuda
              mkdir -p {out_dir}
              cd /data/EEF-Cutting-Simulation
              /workspace/envs/maniskill/bin/python \\
                /data/mani_skill/dataset_converters/collect_rollouts_mpm.py \\
                  --task {env_id} \\
                  --num-episodes {num_eps} \\
                  --seed-base {seed_base} \\
                  --out {out_dir} \\
                  --mpm-config {mpm_config} \\
                  --robot-uid pkour \\
                  --control-mode pd_ee_delta_pose \\
                  --fixed-object {fruit} \\
                  --mpm-substeps-cap 0 \\
                  --render-mpm \\
                  --apply-mpm-force \\
                  --knife-speed-mps 0.30 \\
                2>&1 | tee {out_dir}/collect_{shard_id}.log
          resources:
            requests:
              cpu: "8"
              memory: "64Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "16"
              memory: "128Gi"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: workspace
              mountPath: /workspace
            - name: data
              mountPath: /data
            - name: dshm
              mountPath: /dev/shm
      volumes:
        - name: workspace
          persistentVolumeClaim:
            claimName: hyunsuh-workspace-1
        - name: data
          persistentVolumeClaim:
            claimName: data2
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 16Gi
"""


def plan(objects, episodes_per_object, shards_per_object, out_root):
    """Return list of shard specs: (fruit, seed_base, num_eps, shard_id, out_dir)."""
    out_root = Path(out_root)
    per_shard = math.ceil(episodes_per_object / shards_per_object)
    specs = []
    for fruit in objects:
        out_dir = out_root / fruit
        for s in range(shards_per_object):
            seed_base = s * per_shard
            num_eps = min(per_shard, episodes_per_object - s * per_shard)
            if num_eps <= 0:
                continue
            specs.append({
                "fruit": fruit,
                "seed_base": seed_base,
                "num_eps": num_eps,
                "shard_id": s,
                "out_dir": str(out_dir),
                "mpm_config": MPM_CONFIG[fruit],
                "env_id": ENV_ID,
                "job_name": f"por1329-collect-{fruit}-s{s:02d}",
            })
    return specs


def render(spec):
    return JOB_TEMPLATE.format(**spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", default=DEFAULT_OBJECTS)
    ap.add_argument("--episodes-per-object", type=int, default=500)
    ap.add_argument("--shards-per-object", type=int, default=10)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan + rendered YAMLs but don't kubectl apply")
    ap.add_argument("--manifests-dir", type=Path,
                    default=Path("/tmp/collect_manifests"),
                    help="where to write rendered job YAMLs")
    ap.add_argument("--max-concurrent", type=int, default=0,
                    help="if >0, stagger kubectl apply so at most this many jobs run concurrently")
    args = ap.parse_args()

    specs = plan(args.objects, args.episodes_per_object,
                 args.shards_per_object, args.out_root)
    print(f"[plan] {len(specs)} jobs: {args.objects} × {args.shards_per_object} shards")
    total_eps = sum(s["num_eps"] for s in specs)
    print(f"[plan] total episodes: {total_eps}  "
          f"({args.episodes_per_object} per object × {len(args.objects)} objects)")

    args.manifests_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        yaml = render(spec)
        path = args.manifests_dir / f"{spec['job_name']}.yaml"
        path.write_text(yaml)

    print(f"[plan] wrote {len(specs)} YAMLs → {args.manifests_dir}")

    if args.dry_run:
        print("[dry-run] Not submitting. Inspect above manifests then re-run without --dry-run.")
        return

    # Submit jobs. max_concurrent=0 = fire all at once (cluster scheduler queues).
    submitted = 0
    for spec in specs:
        path = args.manifests_dir / f"{spec['job_name']}.yaml"
        ret = subprocess.run(["kubectl", "apply", "-f", str(path)],
                             capture_output=True, text=True)
        if ret.returncode == 0:
            submitted += 1
            print(f"[submit] {spec['job_name']}  ({spec['num_eps']} eps, seed_base={spec['seed_base']})")
        else:
            print(f"[ERROR] {spec['job_name']}: {ret.stderr.strip()}")
    print(f"[done] submitted {submitted}/{len(specs)} jobs")


if __name__ == "__main__":
    main()

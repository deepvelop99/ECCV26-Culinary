"""Phase-A grid search: per-(fruit, knife_speed) smoke sweep.

Fires one kube job per (fruit, speed) combo, each with N episodes. Output
goes under /data/datasets/maniskill_mpm_gridsearch/<fruit>/v<speed>/bananacut/.

Usage:
    python launch_grid_search.py \
        --objects banana apple cucumber melon orange peach strawberry \
        --speeds 0.05 0.10 0.15 0.20 0.30 \
        --episodes 5
"""
from __future__ import annotations
import argparse
import os
import subprocess
from pathlib import Path


DEFAULT_OBJECTS = [
    "banana", "apple", "cucumber", "melon", "orange", "peach", "strawberry"
]
DEFAULT_SPEEDS = [0.05, 0.10, 0.15, 0.20, 0.30]

MPM_CONFIG = {
    "banana":     "/data/EEF-Cutting-Simulation/configs/banana.yaml",
    "apple":      "/data/EEF-Cutting-Simulation/configs/apple.yaml",
    "cucumber":   "/data/EEF-Cutting-Simulation/configs/cucumber.yaml",
    "melon":      "/data/EEF-Cutting-Simulation/configs/melon.yaml",
    "orange":     "/data/EEF-Cutting-Simulation/configs/orange.yaml",
    "peach":      "/data/EEF-Cutting-Simulation/configs/peach.yaml",
    "strawberry": "/data/EEF-Cutting-Simulation/configs/strawberry.yaml",
}


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
                  --task bananacut \\
                  --num-episodes {num_eps} \\
                  --seed-base 0 \\
                  --out {out_dir} \\
                  --mpm-config {mpm_config} \\
                  --robot-uid pkour \\
                  --control-mode pd_ee_delta_pose \\
                  --apply-mpm-force \\
                  --fixed-object {fruit} \\
                  --mpm-substeps-cap 0 \\
                  --render-mpm \\
                  --knife-speed-mps {speed} \\
                2>&1 | tee {out_dir}/collect.log
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


def job_name_for(fruit, speed):
    tag = f"{int(round(speed * 100)):03d}"  # "005", "010", …
    return f"por1329-grid-{fruit}-v{tag}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", default=DEFAULT_OBJECTS)
    ap.add_argument("--speeds", nargs="+", type=float, default=DEFAULT_SPEEDS)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out-root", type=Path,
                    default=Path("/data/datasets/maniskill_mpm_gridsearch"))
    ap.add_argument("--manifests-dir", type=Path,
                    default=Path("/tmp/collect_manifests_grid"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.manifests_dir.mkdir(parents=True, exist_ok=True)

    specs = []
    for fruit in args.objects:
        for v in args.speeds:
            tag = f"v{int(round(v*100)):03d}"
            out_dir = args.out_root / fruit / tag
            specs.append(dict(
                fruit=fruit,
                speed=v,
                num_eps=args.episodes,
                out_dir=str(out_dir),
                mpm_config=MPM_CONFIG[fruit],
                job_name=job_name_for(fruit, v),
            ))

    print(f"[plan] {len(specs)} grid jobs ({len(args.objects)} fruits × {len(args.speeds)} speeds × {args.episodes} eps)")

    for spec in specs:
        yaml_text = JOB_TEMPLATE.format(**spec)
        path = args.manifests_dir / f"{spec['job_name']}.yaml"
        path.write_text(yaml_text)

    if args.dry_run:
        print(f"[dry-run] YAMLs in {args.manifests_dir}")
        return

    submitted = 0
    for spec in specs:
        path = args.manifests_dir / f"{spec['job_name']}.yaml"
        ret = subprocess.run(["kubectl", "apply", "-f", str(path)],
                             capture_output=True, text=True)
        if ret.returncode == 0:
            submitted += 1
            print(f"[submit] {spec['job_name']}  v={spec['speed']}")
        else:
            print(f"[ERROR] {spec['job_name']}: {ret.stderr.strip()[:200]}")
    print(f"[done] submitted {submitted}/{len(specs)}")


if __name__ == "__main__":
    main()

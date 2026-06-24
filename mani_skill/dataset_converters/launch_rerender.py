"""Launch sharded re-render of mpm.mp4 + grid.mp4 across 10 kube pods."""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path


JOB_TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: por1329-rerender-{shard:02d}
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
              export LD_LIBRARY_PATH=/workspace/envs/maniskill/lib:${{LD_LIBRARY_PATH:-}}
              export PYTHONPATH=/data:/data/mani_skill:${{PYTHONPATH:-}}
              export PYTHONUNBUFFERED=1
              export TI_ARCH=cuda
              cd /data/EEF-Cutting-Simulation
              /workspace/envs/maniskill/bin/python \\
                /data/mani_skill/dataset_converters/rerender_mpm.py \\
                  --root {root} \\
                  --extra-descent 0 \\
                  --shard {shard} --of {n_shards} \\
                2>&1 | tee /tmp/rerender_{shard:02d}.log
          resources:
            requests:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "16"
              memory: "64Gi"
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
            sizeLimit: 8Gi
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/datasets/maniskill_mpm_full")
    ap.add_argument("--n-shards", type=int, default=10)
    ap.add_argument("--manifests-dir", type=Path,
                    default=Path("/tmp/rerender_manifests"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.manifests_dir.mkdir(parents=True, exist_ok=True)
    submitted = 0
    for s in range(args.n_shards):
        yaml_text = JOB_TEMPLATE.format(
            shard=s, n_shards=args.n_shards, root=args.root)
        path = args.manifests_dir / f"rerender_{s:02d}.yaml"
        path.write_text(yaml_text)
        if args.dry_run:
            continue
        ret = subprocess.run(["kubectl", "apply", "-f", str(path)],
                             capture_output=True, text=True)
        if ret.returncode == 0:
            submitted += 1
            print(f"[submit] por1329-rerender-{s:02d}")
        else:
            print(f"[ERROR] shard {s}: {ret.stderr.strip()[:200]}")
    if not args.dry_run:
        print(f"[done] submitted {submitted}/{args.n_shards}")


if __name__ == "__main__":
    main()

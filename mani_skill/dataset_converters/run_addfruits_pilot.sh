#!/usr/bin/env bash
# Addfruits pilot collection — bridge3 (Cutting_rubuttal2)
# Usage: bash run_addfruits_pilot.sh [fruit1 fruit2 ...]
# Default: runs all addfruits with 5 episodes each.

set -euo pipefail

PYTHON=/workspace/envs/maniskill/bin/python
SCRIPT=/data/mani_skill/dataset_converters/collect_rollouts_mpm.py
CFG_ROOT=/data/Cutting_rubuttal2/configs/fruits
OUT_ROOT=/data/datasets/maniskill_mpm_addfruits_pilot_v7
N_EPS=5
SEED_BASE=9000

export LD_LIBRARY_PATH=/workspace/envs/maniskill/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/data:/data/mani_skill:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export MPM_BRIDGE_VARIANT=bridge3

# All addfruits that use bridge3
ALL_FRUITS=(cherry grape kiwi lemon plum shine_muscat tomato pear golden_strawberry)

# Allow override from CLI args
FRUITS=("${@:-${ALL_FRUITS[@]}}")

cd /data/Cutting_rubuttal2

for FRUIT in "${FRUITS[@]}"; do
    CFG="${CFG_ROOT}/${FRUIT}.yaml"
    if [[ ! -f "$CFG" ]]; then
        echo "[skip] No config for ${FRUIT}: ${CFG}"
        continue
    fi
    OUT_DIR="${OUT_ROOT}/${FRUIT}"
    mkdir -p "$OUT_DIR"
    echo "========================================"
    echo "[pilot] fruit=${FRUIT}  eps=${N_EPS}  out=${OUT_DIR}"
    echo "========================================"
    $PYTHON $SCRIPT \
        --task bananacut \
        --fixed-object "$FRUIT" \
        --num-episodes "$N_EPS" \
        --seed-base "$SEED_BASE" \
        --out "$OUT_ROOT" \
        --mpm-config "$CFG" \
        --robot-uid pkour \
        --control-mode pd_ee_delta_pose \
        --knife-speed-mps 0.10 \
        --obs-mode rgb \
        --save-video \
        2>&1 | tee "${OUT_DIR}/pilot.log"
    echo "[done] ${FRUIT}"
done

echo "========================================"
echo "All done. Results at ${OUT_ROOT}"

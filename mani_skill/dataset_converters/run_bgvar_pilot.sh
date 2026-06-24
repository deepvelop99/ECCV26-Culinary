#!/usr/bin/env bash
# bgvar pilot collection — 1 episode per (bg_idx, fruit) pair across all 15
# backgrounds, cycling 9 addfruits. Uses bananacut_bgvar (table+board visual
# variation per bg_idx) and bridge3 MPM physics. High-res render only — no
# mpm.mp4, no grid.mp4, and mpm_render.npz is removed post-ep.
#
# Output: /data/datasets/maniskill_mpm_bgvar_pilot/bg<N>_<fruit>/
#
# Run inside the pilot pod (env vars expected: LD_LIBRARY_PATH, PYTHONPATH,
# MS_ASSET_DIR set externally; this script sets the bgvar-specific ones).

set -uo pipefail

PYTHON=/workspace/envs/maniskill/bin/python
SCRIPT=/data/mani_skill/dataset_converters/collect_rollouts_mpm.py
CFG_ROOT=/data/Cutting_rubuttal2/configs/fruits
OUT_ROOT=/data/datasets/maniskill_mpm_bgvar_pilot_smallfix
SEED_BASE=51000

export PYTHONUNBUFFERED=1
export TI_ARCH=cuda
export MPM_BRIDGE_VARIANT=bridge4
# bananacut_bgvar reads BGVAR_RENDER_RES for high-res render+wide cameras.
# 1024 = good visual quality, ~4-8 MB per ep. Set to 2048 for press-quality.
export BGVAR_RENDER_RES="${BGVAR_RENDER_RES:-2048}"

# Per-bg explicit fruit assignment. Small fruits (cherry/grape/shine_muscat)
# are pinned to has_board bgs (1,3,5,7,9,11,13,15) so the knife always has a
# raised cut surface — small fruits on bare table miss too often. Big fruits
# (lemon/kiwi/plum/tomato/pear) take the no_board bgs (2,4,6,8,10,12,14) plus
# any leftover board bgs. golden_strawberry excluded.
declare -A BG_FRUIT
BG_FRUIT[1]=cherry        # board
BG_FRUIT[2]=plum          # no board
BG_FRUIT[3]=grape         # board
BG_FRUIT[4]=tomato        # no board
BG_FRUIT[5]=shine_muscat  # board
BG_FRUIT[6]=pear          # no board
BG_FRUIT[7]=cherry        # board
BG_FRUIT[8]=lemon         # no board
BG_FRUIT[9]=grape         # board
BG_FRUIT[10]=kiwi         # no board
BG_FRUIT[11]=shine_muscat # board
BG_FRUIT[12]=plum         # no board
BG_FRUIT[13]=lemon        # board
BG_FRUIT[14]=tomato       # no board
BG_FRUIT[15]=kiwi         # board

cd /data/Cutting_rubuttal2

for BG in $(seq 1 15); do
  FRUIT="${BG_FRUIT[$BG]}"
  CFG="${CFG_ROOT}/${FRUIT}.yaml"
  if [[ ! -f "$CFG" ]]; then
    echo "[skip] No config for ${FRUIT}: ${CFG}"
    continue
  fi
  OUT_DIR="${OUT_ROOT}/bg${BG}_${FRUIT}"
  mkdir -p "$OUT_DIR"
  echo "================================================================"
  echo "[bgvar pilot] bg=${BG}  fruit=${FRUIT}  res=${BGVAR_RENDER_RES}  out=${OUT_DIR}"
  echo "================================================================"
  $PYTHON $SCRIPT \
    --task bananacut_bgvar \
    --fixed-object "$FRUIT" \
    --num-episodes 1 \
    --seed-base "$(( SEED_BASE + BG * 100 ))" \
    --out "$OUT_DIR" \
    --mpm-config "$CFG" \
    --robot-uid pkour \
    --control-mode pd_ee_delta_pose \
    --knife-speed-mps 0.10 \
    --obs-mode rgb \
    --save-video \
    --bg-only-idx "$BG" \
    2>&1 | tee "${OUT_DIR}/collect.log"

  # Drop the mpm particle dump (user opted out of MPM artifacts). mp4 files
  # produced by --render-mpm-mp4 / grid are already absent since that flag
  # isn't passed.
  find "$OUT_DIR" -name "mpm_render.npz" -delete 2>/dev/null || true
  echo "[done] bg${BG}_${FRUIT}"
done

echo "================================================================"
echo "All bgvar pilot eps complete. Results at ${OUT_ROOT}"
ls "${OUT_ROOT}"

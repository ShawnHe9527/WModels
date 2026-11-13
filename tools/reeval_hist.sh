#!/usr/bin/env bash
# 2026-11-13：用多指标标准复查历史实验（64 trial，200/400 ms）
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mplcache
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
PY=/home/Hesj/envs/worm_sim/bin/python
SC=/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json
BASE=runs/2026-11-13-reeval
LOG="$BASE/reeval.log"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

batch() {  # $1=outname, rest = name=ckpt
  local OUT=$1; shift
  say "batch $OUT: $*"
  timeout 3600 env -u DISPLAY "$PY" -u -m tools.eval_gate \
    --data datasets/1s/va_s2 --t0 4000 --ks 2000,8000,16000 --n-trials 64 --batch 16 \
    --fit-data datasets/1s/tr_s1 --fit-n 500 --scaler "$SC" \
    "$@" --out "$BASE/$OUT.json" 2>&1 | tee -a "$LOG" | tail -8
}

batch b1 \
  --model AB1=r6_ab_AB/last.pt --model AB2=r6_ab_s2/last.pt --model AB3=r6_ab_s3/last.pt \
  --model A1=r6_ab_A/last.pt --model A2=r6_a_s2/last.pt --model A3=r6_a_s3/last.pt \
  --model FW2_1=r6_fw2/last.pt --model FW2_2=r6_fw2_s2/last.pt --model FW2_3=r6_fw2_s3/last.pt

batch b2 \
  --model OV1=r6_ov/last.pt --model OV2=r6_ov_s2/last.pt --model OV3=r6_ov_s3/last.pt \
  --model E1_4=r6_e1/last.pt --model E1_2=r6_e1_s2/last.pt \
  --model E1S1=r6_e1strong/last.pt --model E1S2=r6_e1s_s2/last.pt --model E1S3=r6_e1s_s3/last.pt

batch b3 \
  --model OVL1=r6_ovlong/last.pt --model OVL2=r6_ovlong_s2/last.pt --model OVL3=r6_ovlong_s3/last.pt \
  --model PB_4=r6_pinball/last.pt --model PB2_4=r6_pinball2/last.pt \
  --model VH_4=r6_vhead/last.pt --model VH_2=r6_vhead_s2/last.pt \
  --model UH_2=r6_uhist/last.pt

batch b4 \
  --model E0_4=r6_e0/last.pt --model EO_4=r6_eventout/last.pt --model EOT_4=r6_eventout_thr/last.pt \
  --model MH_4=r6_mh/last.pt --model DZ32=r6_dz32/last.pt --model FT=r6_finetune/last.pt --model S2D=r6_2s/last.pt

say "=== done ==="

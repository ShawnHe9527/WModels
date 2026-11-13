#!/usr/bin/env bash
# 2026-11-13：128-trial 复查（长 OV 三 seed、E1 四 seed）
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mplcache
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY=/home/Hesj/envs/worm_sim/bin/python
SC=/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json
BASE=runs/2026-11-13-reeval128
mkdir -p "$BASE"; LOG="$BASE/reeval.log"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
ev(){  # $1=tag $2=ckpt
  say "eval $1"
  timeout 2400 env -u DISPLAY "$PY" -u -m tools.eval_gate \
    --data datasets/1s/va_s2 --t0 4000 --ks 2000,8000,16000 --n-trials 128 --batch 16 \
    --fit-data datasets/1s/tr_s1 --fit-n 500 --scaler "$SC" \
    --model "$1=$2" --out "$BASE/eval_$1.json" 2>&1 | tail -4 | tee -a "$LOG"
}
ev OVL1 r6_ovlong/last.pt
ev OVL2 r6_ovlong_s2/last.pt
ev OVL3 r6_ovlong_s3/last.pt
ev E1_s2 r6_e1_s2/last.pt
ev E1_s3 r6_e1_s3/last.pt
ev E1_s5 r6_e1_s5/last.pt
say "=== done ==="

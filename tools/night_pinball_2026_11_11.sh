#!/usr/bin/env bash
# 2026-11-11：pinball 分位数损失（q=0.85）seed4，FW2 基座
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/mplcache
mkdir -p /tmp/mplcache
PY=/home/Hesj/envs/worm_sim/bin/python
SCALER=/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json
BASE=runs/2026-11-11-night-pinball
LOG="$BASE/night.log"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

R=$(bash tools/new_run.sh r6_pinball)
say "开始 r6_pinball → $R"
bash tools/guard_run.sh "$R" "$R/stdout.log" -- \
  env -u DISPLAY "$PY" -u -m world.learn.train2 t2 \
  --data datasets/1s/tr_s1 --val datasets/1s/va_s2 --run r6_pinball --tier t2 \
  --init t1v4_dz16/best.pt --scaler "$SCALER" \
  --steps 9000 --batch 24 --length 128 --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
  --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 64 --cl-horizon 512 \
  --cl-chunk 64 --cl-lags 64,256,512 --cl-weight 1.0 --cl-weight-start 0.1 \
  --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.1 \
  --event-weight 10 --event-tau 0.2 --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
  --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
  --sigma-gate 0.008 --rss-cap-mb 32000 --mem-floor-mb 16000 \
  --save-every 500 --mem-mb 16000 --overshoot-lags 32,64,96 --overshoot-weight 0.1 \
  --factor-weights "1,1,1,1,1,1,1,2,2" --pinball-q 0.85 --pinball-weight 0.1 \
  --seed 4
say "训练结束 rc=$?"

timeout 2400 env -u DISPLAY "$PY" -u -m tools.eval_gate \
  --data datasets/1s/va_s2 --t0 4000 --ks 2000,8000,16000 --n-trials 128 --batch 16 \
  --fit-data datasets/1s/tr_s1 --fit-n 300 --scaler "$SCALER" \
  --model PB=r6_pinball/last.pt --out "$BASE/eval_PB.json" 2>&1 | tee -a "$LOG" | tail -5
say "=== 全部结束 ==="

#!/usr/bin/env bash
# 2026-11-07 D：Ca 事件头（辅助损失）+ 评测 + Ca 事件指标
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/mplcache
mkdir -p /tmp/mplcache
PY=/home/Hesj/envs/worm_sim/bin/python
SCALER=/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json
BASE=runs/2026-11-07-night-d
mkdir -p "$BASE"
LOG="$BASE/night.log"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "=== D: 事件头（delta=800, weight=0.1）==="
RUN=$(bash tools/new_run.sh r6_eventhead)
echo "RUN=$RUN" | tee -a "$LOG"
bash tools/guard_run.sh "$RUN" "$RUN/stdout.log" -- \
  env -u DISPLAY "$PY" -u -m world.learn.train2 t2 \
  --data datasets/1s/tr_s1 --val datasets/1s/va_s2 \
  --run r6_eventhead --tier t2 --init t1v4_dz16/best.pt --scaler "$SCALER" \
  --steps 9000 --batch 24 --length 128 --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
  --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 64 --cl-horizon 512 \
  --cl-chunk 64 --cl-lags 64,256,512 --cl-weight 1.0 --cl-weight-start 0.1 \
  --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.1 \
  --event-weight 10 --event-tau 0.2 --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
  --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
  --sigma-gate 0.008 --rss-cap-mb 32000 --mem-floor-mb 16000 \
  --save-every 500 --mem-mb 16000 --seed 1 \
  --event-head --event-head-delta 800 --event-head-weight 0.1
say "训练结束 rc=$?"

say "=== 评测（1s）==="
timeout 2400 env -u DISPLAY "$PY" -u -m tools.eval_gate \
  --data datasets/1s/va_s2 --t0 4000 --ks 2000,8000,16000 --n-trials 128 --batch 16 \
  --fit-data datasets/1s/tr_s1 --fit-n 500 --scaler "$SCALER" \
  --model EH=r6_eventhead/last.pt --out "$BASE/eval_EH.json" 2>&1 | tee -a "$LOG" | tail -5

say "=== Ca 事件指标 ==="
timeout 3600 env -u DISPLAY "$PY" -u -m tools.ca_event_metrics \
  --runs r6_eventhead r6_ab_AB --data datasets/1s/va_s2 --k 8000 --t0 4000 \
  --n-trials 999 --out "$BASE/ca_metrics.json" 2>&1 | tee -a "$LOG" | tail -10
say "=== 全部结束 ==="

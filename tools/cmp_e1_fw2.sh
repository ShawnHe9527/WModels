#!/usr/bin/env bash
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mplcache
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
PY=/home/Hesj/envs/worm_sim/bin/python
BASE=runs/2026-11-13-cmp-e1
mkdir -p "$BASE"
LOG="$BASE/cmp.log"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
say "=== regime bias ==="
for m in r6_fw2_s4 r6_e1 r6_fw2_s2 r6_e1_s2; do
  say "regime_bias $m"
  timeout 1200 env -u DISPLAY "$PY" -u -m tools.regime_bias "$m/last.pt" 2>&1 | tee -a "$LOG" | tail -4
done
say "=== peak bias ==="
for m in r6_fw2_s4 r6_e1 r6_fw2_s2 r6_e1_s2; do
  say "bias_check $m"
  timeout 1200 env -u DISPLAY "$PY" -u -m tools.bias_check "$m/last.pt" "$BASE/bias_$m.json" 2>&1 | tee -a "$LOG" | tail -4
done
say "=== Ca F1 ==="
timeout 3600 env -u DISPLAY "$PY" -u -m tools.ca_event_metrics \
  --runs r6_fw2_s4 r6_e1 r6_fw2_s2 r6_e1_s2 --data datasets/1s/va_s2 --k 8000 --t0 4000 \
  --n-trials 999 --out "$BASE/ca_metrics.json" 2>&1 | tee -a "$LOG" | tail -10
say "=== done ==="

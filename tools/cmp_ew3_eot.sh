#!/usr/bin/env bash
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mplcache
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY=/home/Hesj/envs/worm_sim/bin/python
BASE=runs/2026-11-14-ew3-eot-peak
mkdir -p "$BASE"; LOG="$BASE/cmp.log"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
for m in r6_ew3 r6_ew3_s2 r6_eot_s2; do
  say "bias $m"
  timeout 1200 env -u DISPLAY "$PY" -u -m tools.bias_check "$m/last.pt" "$BASE/bias_$m.json" 2>&1 | tee -a "$LOG" | tail -4
done
say "=== ca ==="
timeout 3600 env -u DISPLAY "$PY" -u -m tools.ca_event_metrics \
  --runs r6_ew3 r6_ew3_s2 r6_eot_s2 --data datasets/1s/va_s2 --k 8000 --t0 4000 \
  --n-trials 999 --out "$BASE/ca_metrics.json" 2>&1 | tee -a "$LOG" | tail -8
say "=== done ==="

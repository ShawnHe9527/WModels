#!/usr/bin/env bash
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
PY=/home/Hesj/envs/worm_sim/bin/python
BASE=runs/2026-10-31-hybrid-seeds
mkdir -p "$BASE"
for m in S1:r6_c_1s_oldsc/last.pt S2:r6_c_1s_oldsc_s2/last.pt S3:r6_c_1s_oldsc_s3/last.pt; do
  tag=${m%%:*}; ck=${m#*:}
  echo "[$(date -Is)] $tag $ck" | tee -a "$BASE/log"
  timeout 1800 env -u DISPLAY $PY -u -m tools.hybrid_rollout \
    --model "$ck" --k 8000 --n 128 --lams 0,0.25,0.5,1.0 \
    --checkpoints 512,2048,8000 --out "$BASE/${tag}.json" 2>&1 | tail -6 | tee -a "$BASE/log"
done
echo "[$(date -Is)] DONE" | tee -a "$BASE/log"

#!/usr/bin/env bash
# 固化当前最优配置（A+B）：horizon 512, lags 64/256/512, TBPTT 64。
# 用途：作为"金标准"复现脚本；默认不自动运行（需手动 bash 运行）。
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
PY=/home/Hesj/envs/worm_sim/bin/python
SCALER=/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json
NAME=${1:-r6_ab_best}
SEED=${2:-1}
RUN=$(bash tools/new_run.sh "$NAME")
echo "RUN=$RUN"
bash tools/guard_run.sh "$RUN" "$RUN/stdout.log" -- \
  env -u DISPLAY "$PY" -u -m world.learn.train2 t2 \
  --data datasets/1s/tr_s1 --val datasets/1s/va_s2 \
  --run "$NAME" --tier t2 --init t1v4_dz16/best.pt --scaler "$SCALER" \
  --steps 9000 --batch 24 --length 128 --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
  --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 64 --cl-horizon 512 \
  --cl-chunk 64 --cl-lags 64,256,512 --cl-weight 1.0 --cl-weight-start 0.1 \
  --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.1 \
  --event-weight 10 --event-tau 0.2 --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
  --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
  --sigma-gate 0.008 --rss-cap-mb 32000 --mem-floor-mb 16000 \
  --save-every 500 --mem-mb 16000 --seed "$SEED"

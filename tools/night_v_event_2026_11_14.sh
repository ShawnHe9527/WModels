#!/usr/bin/env bash
# 2026-11-14：B = event_weight=0 + V 事件输出（thr 0.5），seed4、seed2
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/mplcache
mkdir -p /tmp/mplcache
PY=/home/Hesj/envs/worm_sim/bin/python
SC=/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json
BASE=runs/2026-11-14-night-vev
mkdir -p "$BASE"; LOG="$BASE/night.log"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
train(){  # $1=name $2=seed
  local NAME=$1 SEED=$2 R
  R=$(bash tools/new_run.sh "$NAME")
  say "训练 $NAME seed=$SEED → $R"
  bash tools/guard_run.sh "$R" "$R/stdout.log" -- \
    env -u DISPLAY "$PY" -u -m world.learn.train2 t2 \
    --data datasets/1s/tr_s1 --val datasets/1s/va_s2 --run "$NAME" --tier t2 \
    --init t1v4_dz16/best.pt --scaler "$SC" \
    --steps 9000 --batch 24 --length 128 --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
    --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 64 --cl-horizon 512 \
    --cl-chunk 64 --cl-lags 64,256,512 --cl-weight 1.0 --cl-weight-start 0.1 \
    --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.1 \
    --event-weight 0 --event-tau 0.2 --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
    --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
    --sigma-gate 0.008 --rss-cap-mb 32000 --mem-floor-mb 16000 \
    --save-every 500 --mem-mb 16000 --overshoot-lags 32,64,96 --overshoot-weight 0.1 \
    --factor-weights "1,1,1,1,1,1,1,2,2" \
    --event-out --event-out-factor V --event-out-thr 0.5 --event-out-weight 0.1 --seed "$SEED"
  say "训练结束 $NAME rc=$?"
}
ev(){  # $1=tag $2=ckpt
  timeout 2400 env -u DISPLAY "$PY" -u -m tools.eval_gate \
    --data datasets/1s/va_s2 --t0 4000 --ks 2000,8000,16000 --n-trials 128 --batch 16 \
    --fit-data datasets/1s/tr_s1 --fit-n 500 --scaler "$SC" \
    --model "$1=$2" --out "$BASE/eval_$1.json" 2>&1 | tee -a "$LOG" | tail -5
}
say "=== VEV seed4 ==="
train r6_vev 4; ev VEV_s4 r6_vev/last.pt
timeout 1200 env -u DISPLAY "$PY" -u -m tools.bias_check r6_vev/last.pt "$BASE/bias_VEV_s4.json" 2>&1 | tee -a "$LOG" | tail -3
say "=== VEV seed2 ==="
train r6_vev_s2 2; ev VEV_s2 r6_vev_s2/last.pt
timeout 1200 env -u DISPLAY "$PY" -u -m tools.bias_check r6_vev_s2/last.pt "$BASE/bias_VEV_s2.json" 2>&1 | tee -a "$LOG" | tail -3
say "=== done ==="

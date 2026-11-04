#!/usr/bin/env bash
# 2026-11-04 夜间 A：加强版导数损失（weight 30 / tau 0.01）seed 2、3 复现 + Ca 指标
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/mplcache
mkdir -p /tmp/mplcache
PY=/home/Hesj/envs/worm_sim/bin/python
SCALER=/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json
BASE=runs/2026-11-04-night-e1strong
mkdir -p "$BASE"
LOG="$BASE/night.log"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

train2() {  # $1=name $2=seed
    local NAME=$1 SEED=$2 RUNDIR
    RUNDIR=$(bash tools/new_run.sh "$NAME")
    say "训练 $NAME (E1-strong seed=$SEED) → $RUNDIR"
    bash tools/guard_run.sh "$RUNDIR" "$RUNDIR/stdout.log" -- \
        env -u DISPLAY "$PY" -u -m world.learn.train2 t2 \
        --data datasets/1s/tr_s1 --val datasets/1s/va_s2 \
        --run "$NAME" --tier t2 --init t1v4_dz16/best.pt --scaler "$SCALER" \
        --steps 9000 --batch 24 --length 128 --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
        --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 64 --cl-horizon 512 \
        --cl-chunk 64 --cl-lags 64,256,512 --cl-weight 1.0 --cl-weight-start 0.1 \
        --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.1 \
        --event-weight 10 --event-tau 0.2 --event-v-weight 30 --event-v-tau 0.01 \
        --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
        --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
        --sigma-gate 0.008 --rss-cap-mb 32000 --mem-floor-mb 16000 \
        --save-every 500 --mem-mb 16000 --seed "$SEED"
    say "训练结束 $NAME rc=$?"
}

ev() {  # $1=tag $2=ckpt
    local TAG=$1 CK=$2
    say "评测 $TAG"
    timeout 2400 env -u DISPLAY "$PY" -u -m tools.eval_gate \
        --data datasets/1s/va_s2 --t0 4000 --ks 2000,8000,16000 --n-trials 128 --batch 16 \
        --fit-data datasets/1s/tr_s1 --fit-n 500 --scaler "$SCALER" \
        --model "$TAG=$CK" --out "$BASE/eval_$TAG.json" 2>&1 | tee -a "$LOG" | tail -5
}

say "=== E1-strong seed 2 ==="
train2 r6_e1s_s2 2
ev E1S_s2 r6_e1s_s2/last.pt
say "=== E1-strong seed 3 ==="
train2 r6_e1s_s3 3
ev E1S_s3 r6_e1s_s3/last.pt
say "=== Ca 指标 ==="
timeout 3600 env -u DISPLAY "$PY" -u -m tools.ca_event_metrics \
    --runs r6_e1strong r6_e1s_s2 r6_e1s_s3 r6_ab_AB \
    --data datasets/1s/va_s2 --k 8000 --t0 4000 --n-trials 999 \
    --out "$BASE/ca_metrics.json" 2>&1 | tee -a "$LOG" | tail -12
say "=== 全部结束 ==="

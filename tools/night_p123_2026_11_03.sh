#!/usr/bin/env bash
# 2026-11-03 夜间 P1–P3：
#   P1 horizon 2048（lags 64/256/512/1024/2048）
#   P2 E1 事件感知损失（V 事件加权，A+B 配置）
#   P3 E2 快慢双头 decoder（A+B + fast_head）
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/mplcache
mkdir -p /tmp/mplcache
PY=/home/Hesj/envs/worm_sim/bin/python
SCALER=/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json
BASE=runs/2026-11-03-night-p123
mkdir -p "$BASE"
LOG="$BASE/night.log"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

train2() {  # $1=name $2=seed $3=horizon $4=lags $5=tag [$6=extra]
    local NAME=$1 SEED=$2 H=$3 LAGS=$4 TAG=$5 EXTRA=${6:-} RUNDIR
    RUNDIR=$(bash tools/new_run.sh "$NAME")
    say "训练 $NAME ($TAG: horizon=$H lags=$LAGS seed=$SEED extra=$EXTRA) → $RUNDIR"
    bash tools/guard_run.sh "$RUNDIR" "$RUNDIR/stdout.log" -- \
        env -u DISPLAY "$PY" -u -m world.learn.train2 t2 \
        --data datasets/1s/tr_s1 --val datasets/1s/va_s2 \
        --run "$NAME" --tier t2 --init t1v4_dz16/best.pt --scaler "$SCALER" \
        --steps 9000 --batch 24 --length 128 --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
        --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 64 --cl-horizon "$H" \
        --cl-chunk 64 --cl-lags "$LAGS" --cl-weight 1.0 --cl-weight-start 0.1 \
        --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.1 \
        --event-weight 10 --event-tau 0.2 --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
        --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
        --sigma-gate 0.008 --rss-cap-mb 32000 --mem-floor-mb 16000 \
        --save-every 500 --mem-mb 16000 --seed "$SEED" $EXTRA
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

say "=== P1: horizon 2048 ==="
train2 r6_h2048 1 2048 64,256,512,1024,2048 P1_h2048
ev H2048 r6_h2048/last.pt

say "=== P2: E1 事件感知损失（V 加权）==="
train2 r6_e1_vevent 1 512 64,256,512 P2_E1 "--event-v-weight 10 --event-v-tau 0.02"
ev E1 r6_e1_vevent/last.pt

say "=== P3: E2 快慢双头 ==="
train2 r6_e2_fast 1 512 64,256,512 P3_E2 "--fast-head"
ev E2 r6_e2_fast/last.pt

say "=== 全部结束 ==="

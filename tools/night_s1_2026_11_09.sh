#!/usr/bin/env bash
# 2026-11-09 跨夜：长滞后 OV、多视界头、事件输出 seed2（FW2 基座）
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/mplcache
mkdir -p /tmp/mplcache
PY=/home/Hesj/envs/worm_sim/bin/python
SCALER=/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json
BASE=runs/2026-11-09-night-s1
mkdir -p "$BASE"
LOG="$BASE/night.log"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

train2() {  # $1=name $2=seed $3=tag $4=extra
    local NAME=$1 SEED=$2 TAG=$3 EXTRA=$4 RUNDIR
    RUNDIR=$(bash tools/new_run.sh "$NAME")
    say "训练 $NAME ($TAG seed=$SEED) → $RUNDIR"
    bash tools/guard_run.sh "$RUNDIR" "$RUNDIR/stdout.log" -- \
        env -u DISPLAY "$PY" -u -m world.learn.train2 t2 \
        --data datasets/1s/tr_s1 --val datasets/1s/va_s2 --run "$NAME" --tier t2 \
        --init t1v4_dz16/best.pt --scaler "$SCALER" \
        --steps 9000 --batch 24 --length 128 --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
        --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 64 --cl-horizon 512 \
        --cl-chunk 64 --cl-lags 64,256,512 --cl-weight 1.0 --cl-weight-start 0.1 \
        --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.1 \
        --event-weight 10 --event-tau 0.2 --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
        --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
        --sigma-gate 0.008 --rss-cap-mb 32000 --mem-floor-mb 16000 \
        --save-every 500 --mem-mb 16000 --overshoot-lags 32,64,96 --overshoot-weight 0.1 \
        --factor-weights "1,1,1,1,1,1,1,2,2" --seed "$SEED" $EXTRA
    say "训练结束 $NAME rc=$?"
}
ev() {  # $1=tag $2=ckpt [$3=data]
    local TAG=$1 CK=$2 DATA=${3:-datasets/1s/va_s2} OUTK="2000,8000,16000"
    [ "$DATA" = "datasets/2s/te_s4_2s" ] && OUTK="2000,8000,32000"
    say "评测 $TAG ($DATA)"
    timeout 2400 env -u DISPLAY "$PY" -u -m tools.eval_gate \
        --data "$DATA" --t0 4000 --ks "$OUTK" --n-trials 64 --batch 16 \
        --fit-data datasets/1s/tr_s1 --fit-n 300 --scaler "$SCALER" \
        --model "$TAG=$CK" --out "$BASE/eval_${TAG}.json" 2>&1 | tee -a "$LOG" | tail -5
}

say "=== 1) 长滞后 OV（128/256）==="
train2 r6_ovlong 1 "OV-long" "--overshoot-long-lags 128,256 --overshoot-long-every 10 --overshoot-long-weight 0.1"
ev OVLONG r6_ovlong/last.pt
ev OVLONG_2s r6_ovlong/last.pt datasets/2s/te_s4_2s

say "=== 2) 多视界头（64/256）==="
train2 r6_mh 1 "MH" "--mh-lags 64,256 --mh-every 10 --mh-weight 0.1"
ev MH r6_mh/last.pt
ev MH_2s r6_mh/last.pt datasets/2s/te_s4_2s

say "=== 3) 事件输出 seed 2 ==="
train2 r6_eventout_s2 2 "EO-s2" "--event-out --event-out-weight 0.1"
ev EO_s2 r6_eventout_s2/last.pt

say "=== 全部结束 ==="

#!/usr/bin/env bash
# 2026-11-10 跨夜：Decoder 端到端微调（seed4）+ 2s 数据训练（seed2）
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/mplcache
mkdir -p /tmp/mplcache
PY=/home/Hesj/envs/worm_sim/bin/python
SCALER=/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json
BASE=runs/2026-11-10-night-ft2s
mkdir -p "$BASE"
LOG="$BASE/night.log"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

train2() {  # $1=name $2=seed $3=data $4=val $5=tag $6=extra
    local NAME=$1 SEED=$2 DATA=$3 VAL=$4 TAG=$5 EXTRA=${6:-} RUNDIR
    RUNDIR=$(bash tools/new_run.sh "$NAME")
    say "训练 $NAME ($TAG seed=$SEED data=$DATA) → $RUNDIR"
    bash tools/guard_run.sh "$RUNDIR" "$RUNDIR/stdout.log" -- \
        env -u DISPLAY "$PY" -u -m world.learn.train2 t2 \
        --data "$DATA" --val "$VAL" --run "$NAME" --tier t2 \
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
ev() {  # $1=tag $2=ckpt $3=data $4=ks
    local TAG=$1 CK=$2 DATA=$3 KS=$4
    say "评测 $TAG ($DATA)"
    timeout 2400 env -u DISPLAY "$PY" -u -m tools.eval_gate \
        --data "$DATA" --t0 4000 --ks "$KS" --n-trials 64 --batch 16 \
        --fit-data datasets/1s/tr_s1 --fit-n 300 --scaler "$SCALER" \
        --model "$TAG=$CK" --out "$BASE/eval_${TAG}.json" 2>&1 | tee -a "$LOG" | tail -5
}

say "=== 1) Decoder 端到端微调（seed4）==="
train2 r6_finetune 4 datasets/1s/tr_s1 datasets/1s/va_s2 "FT" "--unfreeze-dec --dec-lr 2e-4"
ev FT r6_finetune/last.pt datasets/1s/va_s2 2000,8000,16000
ev FT_2s r6_finetune/last.pt datasets/2s/te_s4_2s 2000,8000,32000

say "=== 2) 2s 数据训练（seed2）==="
train2 r6_2s 2 datasets/2s/tr_s1_2s datasets/2s/va_s2_2s "2S"
ev S2 r6_2s/last.pt datasets/1s/va_s2 2000,8000,16000
ev S2_2s r6_2s/last.pt datasets/2s/te_s4_2s 2000,8000,32000

say "=== 全部结束 ==="

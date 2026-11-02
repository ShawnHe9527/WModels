#!/usr/bin/env bash
# 2026-11-02 夜间总包（1-5 + 延长评测 400/800 ms）：
#   1 H1024 seed2   2 H1024 seed3   3 H1024 三 seed 盲测
#   4 H1024-dense seed1   5 P3a+H1024 seed1
#   延长评测：k=2000/8000/16000/32000（50/200/400/800 ms）
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/mplcache
mkdir -p /tmp/mplcache
PY=/home/Hesj/envs/worm_sim/bin/python
SCALER=/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json
BASE=runs/2026-11-02-night-all
mkdir -p "$BASE"
LOG="$BASE/night.log"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

train2() {  # $1=name $2=seed $3=horizon $4=lags $5=chunk $6=tag [$7=extra]
    local NAME=$1 SEED=$2 H=$3 LAGS=$4 CH=$5 TAG=$6 EXTRA=${7:-} RUNDIR
    RUNDIR=$(bash tools/new_run.sh "$NAME")
    say "训练 $NAME ($TAG: horizon=$H lags=$LAGS chunk=$CH seed=$SEED extra=$EXTRA) → $RUNDIR"
    bash tools/guard_run.sh "$RUNDIR" "$RUNDIR/stdout.log" -- \
        env -u DISPLAY "$PY" -u -m world.learn.train2 t2 \
        --data datasets/1s/tr_s1 --val datasets/1s/va_s2 \
        --run "$NAME" --tier t2 --init t1v4_dz16/best.pt --scaler "$SCALER" \
        --steps 9000 --batch 24 --length 128 --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
        --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 64 --cl-horizon "$H" \
        --cl-chunk "$CH" --cl-lags "$LAGS" --cl-weight 1.0 --cl-weight-start 0.1 \
        --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.1 \
        --event-weight 10 --event-tau 0.2 --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
        --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
        --sigma-gate 0.008 --rss-cap-mb 32000 --mem-floor-mb 16000 \
        --save-every 500 --mem-mb 16000 --seed "$SEED" $EXTRA
    say "训练结束 $NAME rc=$?"
}

ev() {  # $1=tag $2=ckpt $3=ks $4=data(optional)
    local TAG=$1 CK=$2 KS=$3 DATA=${4:-datasets/1s/va_s2}
    say "评测 $TAG ($DATA, ks=$KS)"
    timeout 2400 env -u DISPLAY "$PY" -u -m tools.eval_gate \
        --data "$DATA" --t0 4000 --ks "$KS" --n-trials 128 --batch 16 \
        --fit-data datasets/1s/tr_s1 --fit-n 500 --scaler "$SCALER" \
        --model "$TAG=$CK" --out "$BASE/eval_$TAG.json" 2>&1 | tee -a "$LOG" | tail -6
}

KS4="2000,8000,16000,32000"

say "=== 1) H1024 seed 2 ==="
train2 r6_b_h1024_s2 2 1024 64,256,512,1024 64 H1024_s2
ev H1024_s2 r6_b_h1024_s2/last.pt "$KS4"

say "=== 2) H1024 seed 3 ==="
train2 r6_b_h1024_s3 3 1024 64,256,512,1024 64 H1024_s3
ev H1024_s3 r6_b_h1024_s3/last.pt "$KS4"

say "=== 3) H1024 三 seed 延长评测 + 盲测 ==="
ev H1024_s1 r6_h1024/last.pt "$KS4"
ev AB_s2 r6_ab_s2/last.pt "$KS4"
ev S1base r6_c_1s_oldsc/last.pt "$KS4"
ev blind_H1024_s1 r6_h1024/last.pt "2000,8000" datasets/2s/te_s4_2s
ev blind_H1024_s2 r6_b_h1024_s2/last.pt "2000,8000" datasets/2s/te_s4_2s
ev blind_H1024_s3 r6_b_h1024_s3/last.pt "2000,8000" datasets/2s/te_s4_2s

say "=== 4) H1024-dense seed 1 ==="
train2 r6_b_dense 1 1024 128,256,384,512,640,768,896,1024 64 DENSE
ev DENSE r6_b_dense/last.pt "2000,8000,16000"

say "=== 5) P3a + H1024 seed 1 ==="
train2 r6_b_p3a 1 1024 64,256,512,1024 64 P3A "--slow-gate"
ev P3A r6_b_p3a/last.pt "2000,8000,16000"

say "=== 全部结束 ==="

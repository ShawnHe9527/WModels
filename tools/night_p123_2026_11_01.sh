#!/usr/bin/env bash
# 2026-11-01 夜间 P1+P3+P2：
#   P1 盲测 te_s4_2s（A+B 三 seed + 基线）
#   P3 horizon=1024（lags 64/256/512/1024，chunk 64）
#   P2 C：全 BPTT（horizon 512，chunk 0）
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/mplcache
mkdir -p /tmp/mplcache
PY=/home/Hesj/envs/worm_sim/bin/python
SCALER=/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json
BASE=runs/2026-11-01-night-p123
mkdir -p "$BASE"
LOG="$BASE/night.log"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

ev2s() {  # $1=tag $2=ckpt
    local TAG=$1 CK=$2
    say "盲测(2s) $TAG"
    timeout 1800 env -u DISPLAY "$PY" -u -m tools.eval_gate \
        --data datasets/2s/te_s4_2s --t0 4000 --ks 2000,8000 --n-trials 128 --batch 16 \
        --fit-data datasets/1s/tr_s1 --fit-n 500 --scaler "$SCALER" \
        --model "$TAG=$CK" --out "$BASE/blind_$TAG.json" 2>&1 | tee -a "$LOG" | tail -4
}

evv() {  # $1=tag $2=ckpt
    local TAG=$1 CK=$2
    say "验证(1s) $TAG"
    timeout 1800 env -u DISPLAY "$PY" -u -m tools.eval_gate \
        --data datasets/1s/va_s2 --t0 4000 --ks 2000,8000 --n-trials 128 --batch 16 \
        --fit-data datasets/1s/tr_s1 --fit-n 500 --scaler "$SCALER" \
        --model "$TAG=$CK" --out "$BASE/eval_$TAG.json" 2>&1 | tee -a "$LOG" | tail -4
}

train2() {  # $1=name $2=seed $3=horizon $4=lags $5=chunk $6=tag
    local NAME=$1 SEED=$2 H=$3 LAGS=$4 CH=$5 TAG=$6 RUNDIR
    RUNDIR=$(bash tools/new_run.sh "$NAME")
    say "训练 $NAME ($TAG: horizon=$H lags=$LAGS chunk=$CH seed=$SEED) → $RUNDIR"
    cat > "$RUNDIR/config.yaml" <<YAML
experiment: $NAME
purpose: "$TAG: horizon=$H, lags=$LAGS, chunk=$CH"
host: a6000middle
date: 2026-11-01
git_head: $(git rev-parse --short=7 HEAD)
data: {train: datasets/1s/tr_s1, val: datasets/1s/va_s2, scaler: $SCALER}
model: {d_z: 16, hidden: 192, n_layers: 2, init: t1v4_dz16/best.pt}
optim: {steps: 9000, batch: 24, length: 128, lr: 1.0e-3,
        stage_b_lr: 2.0e-4, stage_b_grad_clip: 0.3}
curriculum: {cl_stage_a_steps: 2000, cl_start: 4, cl_horizon: $H,
             cl_chunk: $CH, cl_lags: "$LAGS", cl_weight: 1.0,
             cl_weight_start: 0.1, anchor_weight: 0.1, anchor_every: 4,
             anchor_cl: 0.1}
sampling: {event_weight: 10, event_tau: 0.2, regime_weights: "0.25,0.35,0.40", hot_frac: 0.5}
seed: $SEED
guards: {sigma_gate: 0.008, rss_cap_mb: 32000, mem_floor_mb: 16000, save_every: 500}
YAML
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
        --save-every 500 --mem-mb 16000 --seed "$SEED"
    say "训练结束 $NAME rc=$?"
}

say "=== P1：盲测 te_s4_2s ==="
ev2s AB_s1 r6_ab_AB/last.pt
ev2s AB_s2 r6_ab_s2/last.pt
ev2s AB_s3 r6_ab_s3/last.pt
ev2s S1base r6_c_1s_oldsc/last.pt

say "=== P3：horizon=1024 ==="
train2 r6_h1024 1 1024 64,256,512,1024 64 H1024
evv H1024 r6_h1024/last.pt

say "=== P2：C 全 BPTT（chunk=0）==="
train2 r6_Cfull 1 512 64,256,512 0 Cfull
evv Cfull r6_Cfull/last.pt

say "=== 全部结束 ==="

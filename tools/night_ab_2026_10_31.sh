#!/usr/bin/env bash
# 2026-10-31 夜间：A（长视界、仅末端损失）与 A+B（多滞后损失）
# 一次启动、顺序执行、结果写 runs/2026-10-31-night-ab/
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/mplcache
mkdir -p /tmp/mplcache
PY=/home/Hesj/envs/worm_sim/bin/python
SCALER=/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json
BASE=runs/2026-10-31-night-ab
mkdir -p "$BASE"
LOG="$BASE/night.log"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

train2() {  # $1=name $2=seed $3=horizon $4=lags $5=chunk
    local NAME=$1 SEED=$2 H=$3 LAGS=$4 CH=$5 RUNDIR
    RUNDIR=$(bash tools/new_run.sh "$NAME")
    say "训练 $NAME (horizon=$H lags=$LAGS chunk=$CH) → $RUNDIR"
    cat > "$RUNDIR/config.yaml" <<YAML
experiment: $NAME
purpose: "A/A+B: long-horizon TBPTT receding-horizon (horizon=$H, lags=$LAGS, chunk=$CH)"
host: a6000middle
date: 2026-10-31
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

ev() {  # $1=tag $2=ckpt
    local TAG=$1 CK=$2
    say "评测 $TAG"
    timeout 1800 env -u DISPLAY "$PY" -u -m tools.eval_gate \
        --data datasets/1s/va_s2 --t0 4000 --ks 2000,8000 --n-trials 128 --batch 16 \
        --fit-data datasets/1s/tr_s1 --fit-n 500 --scaler "$SCALER" \
        --model "$TAG=$CK" --out "$BASE/eval_$TAG.json" 2>&1 | tee -a "$LOG" | tail -4
}

say "=== A / A+B 夜间实验开始 ==="
train2 r6_ab_A 1 512 512 64
ev A r6_ab_A/last.pt
train2 r6_ab_AB 1 512 64,256,512 64
ev AB r6_ab_AB/last.pt
say "=== 夜间实验结束 ==="

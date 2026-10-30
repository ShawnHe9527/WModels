#!/usr/bin/env bash
# 2026-10-30 F2b：S2 原配方 + 快照（闭环选模验证）；复评 S3 最优快照。
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/mplcache
mkdir -p /tmp/mplcache
PY=/home/Hesj/envs/worm_sim/bin/python
SCALER=/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json
BASE=runs/2026-10-30-night2
mkdir -p "$BASE"
LOG="$BASE/night2.log"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

train2() {  # $1=name $2=seed $3=stage_b_lr $4=snap_every
    local NAME=$1 SEED=$2 LR=$3 SNAP=$4 RUNDIR
    RUNDIR=$(bash tools/new_run.sh "$NAME")
    say "开始训练 $NAME → $RUNDIR (seed=$SEED stage_b_lr=$LR snap=$SNAP)"
    cat > "$RUNDIR/config.yaml" <<YAML
experiment: $NAME
purpose: "F2b: closed-loop checkpoint selection evidence"
host: a6000middle
date: 2026-10-30
git_head: $(git rev-parse --short=7 HEAD)
data: {train: datasets/1s/tr_s1, val: datasets/1s/va_s2, scaler: $SCALER}
model: {d_z: 16, hidden: 192, n_layers: 2, init: t1v4_dz16/best.pt}
optim: {steps: 9000, batch: 24, length: 128, lr: 1.0e-3,
        stage_b_lr: $LR, stage_b_grad_clip: 0.3}
curriculum: {cl_stage_a_steps: 2000, cl_start: 4, cl_steps: 64,
             cl_weight: 1.0, cl_weight_start: 0.1, anchor_weight: 0.1,
             anchor_every: 4, anchor_cl: 0.1}
sampling: {event_weight: 10, event_tau: 0.2, regime_weights: "0.25,0.35,0.40", hot_frac: 0.5}
seed: $SEED
snap_every: $SNAP
guards: {sigma_gate: 0.008, rss_cap_mb: 32000, mem_floor_mb: 16000, save_every: 500}
YAML
    bash tools/guard_run.sh "$RUNDIR" "$RUNDIR/stdout.log" -- \
        env -u DISPLAY "$PY" -u -m world.learn.train2 t2 \
        --data datasets/1s/tr_s1 --val datasets/1s/va_s2 \
        --run "$NAME" --tier t2 --init t1v4_dz16/best.pt --scaler "$SCALER" \
        --steps 9000 --batch 24 --length 128 --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
        --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 64 --cl-weight 1.0 --cl-weight-start 0.1 \
        --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.1 \
        --event-weight 10 --event-tau 0.2 --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
        --stage-b-lr "$LR" --stage-b-grad-clip 0.3 \
        --sigma-gate 0.008 --rss-cap-mb 32000 --mem-floor-mb 16000 \
        --save-every 500 --mem-mb 16000 --snap-every "$SNAP" --seed "$SEED"
    say "结束训练 $NAME rc=$?"
}

ev() {  # $1=model_name $2=ckpt $3=tag $4=trials
    local M=$1 C=$2 TAG=$3 N=${4:-64}
    say "评测 $TAG ($M)"
    timeout 1200 env -u DISPLAY "$PY" -u -m tools.eval_gate \
        --data datasets/1s/va_s2 --t0 4000 --ks 8000 --n-trials "$N" --batch 16 \
        --fit-data datasets/1s/tr_s1 --fit-n 200 --scaler "$SCALER" \
        --model "$M=$C" --out "$BASE/eval_${TAG}.json" 2>&1 | tee -a "$LOG"
}

say "=== F2b 开始 ==="
train2 r6_c_1s_s2_snap 2 2e-4 1000
for st in 02000 04000 06000 08000; do
    ev "S2s$st" "r6_c_1s_s2_snap/snap_${st}.pt" "s2_snap_$st" 64
done
ev S3snap8000 r6_c_1s_s3_snap/snap_08000.pt s3_snap_08000_128 128
say "=== F2b 结束 ==="

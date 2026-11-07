#!/usr/bin/env bash
# 2026-11-07 容量筛查 + 逐因子权重（OV 4/5 之后跑）
#   1) T1 d_z=32（新 encoder/decoder）
#   2) T2 d_z=32（A+B + overshoot）
#   3) T2 d_z=16 逐因子权重（V/cai ×2）
set -u
cd /home/Hesj/WModel || exit 1
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/mplcache
mkdir -p /tmp/mplcache
PY=/home/Hesj/envs/worm_sim/bin/python
SCALER=/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json
BASE=runs/2026-11-07-night-cap
mkdir -p "$BASE"
LOG="$BASE/night.log"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

run() {  # $1=name $2=stdout-run-dir $3... cmd
    local NAME=$1 RUNDIR=$2; shift 2
    say "开始 $NAME → $RUNDIR"
    bash tools/guard_run.sh "$RUNDIR" "$RUNDIR/stdout.log" -- "$@"
    say "结束 $NAME rc=$?"
}

ev() {  # $1=tag $2=ckpt
    local TAG=$1 CK=$2
    say "评测 $TAG"
    timeout 2400 env -u DISPLAY "$PY" -u -m tools.eval_gate \
        --data datasets/1s/va_s2 --t0 4000 --ks 2000,8000,16000 --n-trials 128 --batch 16 \
        --fit-data datasets/1s/tr_s1 --fit-n 500 --scaler "$SCALER" \
        --model "$TAG=$CK" --out "$BASE/eval_$TAG.json" 2>&1 | tee -a "$LOG" | tail -5
}

say "=== 1) T1 d_z=32 ==="
R1=$(bash tools/new_run.sh t1v4_dz32)
run t1v4_dz32 "$R1" env -u DISPLAY "$PY" -u -m world.learn.train2 t1 \
  --data datasets/1s/tr_s1 --val datasets/1s/va_s2 --run t1v4_dz32 --tier t2 \
  --steps 12000 --batch 32 --length 64 --d-z 32 --hidden 192 --n-layers 2 --lr 3e-3 \
  --rel-gate 0.05 --mem-mb 12000

say "=== 2) T2 d_z=32（A+B + overshoot）==="
R2=$(bash tools/new_run.sh r6_dz32)
run r6_dz32 "$R2" env -u DISPLAY "$PY" -u -m world.learn.train2 t2 \
  --data datasets/1s/tr_s1 --val datasets/1s/va_s2 --run r6_dz32 --tier t2 \
  --init t1v4_dz32/best.pt --scaler "$SCALER" \
  --steps 9000 --batch 24 --length 128 --d-z 32 --hidden 384 --n-layers 2 --lr 1e-3 \
  --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 64 --cl-horizon 512 \
  --cl-chunk 64 --cl-lags 64,256,512 --cl-weight 1.0 --cl-weight-start 0.1 \
  --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.1 \
  --event-weight 10 --event-tau 0.2 --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
  --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
  --sigma-gate 0.008 --rss-cap-mb 32000 --mem-floor-mb 16000 \
  --save-every 500 --mem-mb 24000 --overshoot-lags 32,64,96 --overshoot-weight 0.1 --seed 1
ev DZ32 r6_dz32/last.pt

say "=== 3) 逐因子权重（V/cai ×2）d_z=16 ==="
R3=$(bash tools/new_run.sh r6_fw2)
run r6_fw2 "$R3" env -u DISPLAY "$PY" -u -m world.learn.train2 t2 \
  --data datasets/1s/tr_s1 --val datasets/1s/va_s2 --run r6_fw2 --tier t2 \
  --init t1v4_dz16/best.pt --scaler "$SCALER" \
  --steps 9000 --batch 24 --length 128 --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
  --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 64 --cl-horizon 512 \
  --cl-chunk 64 --cl-lags 64,256,512 --cl-weight 1.0 --cl-weight-start 0.1 \
  --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.1 \
  --event-weight 10 --event-tau 0.2 --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
  --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
  --sigma-gate 0.008 --rss-cap-mb 32000 --mem-floor-mb 16000 \
  --save-every 500 --mem-mb 16000 --overshoot-lags 32,64,96 --overshoot-weight 0.1 \
  --factor-weights "1,1,1,1,1,1,1,2,2" --seed 1
ev FW2 r6_fw2/last.pt

say "=== 全部结束 ==="

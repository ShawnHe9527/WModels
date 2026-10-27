#!/usr/bin/env bash
# v4 实验流水线（DESIGN_v4 §9）。每一步的产物与门禁都写进 results/。
#
# 用法：
#   bash run_v4.sh v3        # 生成 tr/va/te 三组（各 200 条）
#   bash run_v4.sh v4        # T1 自编码（t2 档）
#   bash run_v4.sh v5        # T2 两阶段课程（t2 档）
#   bash run_v4.sh v6        # 配对评测（t2 档）
#   bash run_v4.sh v6b       # T1/T2 消融（t1 档，同 ckpt 配置）
#   bash run_v4.sh all       # 依次全跑
#
# ⚠️ 内存约束：同一时刻只允许一个训练进程（可用内存 1.5–2.7 GB）。
#
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-/home/neuron/miniconda3/envs/worm_sim/bin/python}
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
LOGD=results/logs
mkdir -p "$LOGD" results/checkpoints results/reports results/data

step_v3() {
  for spec in "tr_s1 1" "va_s2 2" "te_s3 3"; do
    set -- $spec
    name=$1; seed=$2
    echo "=========== V3 生成 $name (seed=$seed) ==========="
    $PY -u -m world.gen.datagen2 --name "$name" --n-trial 200 \
        --seed "$seed" --every-secs 20 2>&1 | tee "$LOGD/datagen_$name.log"
  done
}

step_v4() {
  echo "=========== V4 T1 自编码（t2, d_z=16）==========="
  $PY -u -m world.learn.train2 t1 \
      --data tr_s1 --val va_s2 --run t1v4_dz16 --tier t2 \
      --steps 12000 --batch 32 --length 64 \
      --d-z 16 --hidden 192 --n-layers 2 --lr 3e-3 \
      --rel-gate 0.05 --mem-mb 2600 2>&1 | tee "$LOGD/t1v4.log"
}

step_v5() {
  echo "=========== V5 T2 两阶段课程（t2，初始基线配置）==========="
  # 这是 v4 首轮基线复现；当前最优 R2 的完整命令见
  # results/rounds/R2_event_weighting/CONCLUSIONS.md
  $PY -u -m world.learn.train2 t2 \
      --data tr_s1 --val va_s2 --run t2v4_baseline --tier t2 \
      --init t1v4_dz16/best.pt --steps 9000 \
      --batch 32 --length 128 \
      --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
      --cl-stage-a-steps 2000 --cl-start 10 --cl-steps 64 --cl-weight 1.0 \
      --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
      --sigma-gate 0.008 --mem-mb 2600 2>&1 | tee "$LOGD/t2v4_baseline.log"
}

step_v6() {
  echo "=========== V6 配对评测（t2, te_s3）==========="
  # 当前最优为 R2（t2v5_events）；R0–R4 的归档评测见 results/rounds/
  for ck in t2v5_events/last.pt t2v5_events/best.pt; do
    [ -f "results/checkpoints/$ck" ] || continue
    echo "--- $ck ---"
    $PY -u -m world.learn.train2 eval --ckpt "$ck" --data te_s3 --tier t2 \
        --horizons 10,40,200,1000,4000,8000 --n-seq 64 --n-rep 4 \
        --mem-mb 2600 2>&1 | tee "$LOGD/eval_$(basename "$ck" .pt)_t2.log"
  done
}

step_diag() {
  echo "=========== §6.5 诊断 D-a / D-b ==========="
  $PY -u -m tools.diag_latent --ckpt t2v5_events/last.pt \
      --data te_s3 --tier t2 --which ab --n-win 192 --length 512 \
      --mem-mb 2600 2>&1 | tee "$LOGD/diag_t2.log"
}

step_v6b() {
  echo "=========== V6b T1/T2 消融 ==========="
  echo ">>> 先训 t1 档的 T1"
  $PY -u -m world.learn.train2 t1 \
      --data tr_s1 --val va_s2 --run t1v4t1_dz16 --tier t1 \
      --steps 12000 --batch 32 --length 64 \
      --d-z 16 --hidden 192 --n-layers 2 --lr 3e-3 \
      --rel-gate 0.05 --mem-mb 2600 2>&1 | tee "$LOGD/t1v4_t1.log"
  echo ">>> 再训 t1 档的 T2（相同的两阶段课程）"
  $PY -u -m world.learn.train2 t2 \
      --data tr_s1 --val va_s2 --run t2v4t1_dz16 --tier t1 \
      --init t1v4t1_dz16/best.pt --steps 9000 \
      --batch 32 --length 128 \
      --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
      --cl-stage-a-steps 2000 --cl-start 10 --cl-steps 64 --cl-weight 1.0 \
      --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
      --sigma-gate 0.008 --mem-mb 2600 2>&1 | tee "$LOGD/t2v4_t1.log"
  echo ">>> 评测 t1 档"
  $PY -u -m world.learn.train2 eval --ckpt t2v4t1_dz16/last.pt \
      --data te_s3 --tier t1 --horizons 10,40,200,1000,4000,8000 \
      --n-seq 64 --n-rep 4 --mem-mb 2600 2>&1 \
      | tee "$LOGD/eval_t1_tier.log"
}

case "${1:-all}" in
  v3)  step_v3  ;;
  v4)  step_v4  ;;
  v5)  step_v5  ;;
  v6)  step_v6  ;;
  v6b) step_v6b ;;
  diag) step_diag ;;
  all) step_v3; step_v4; step_v5; step_v6; step_diag; step_v6b ;;
  *) echo "未知步骤 $1"; exit 2 ;;
esac

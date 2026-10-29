#!/usr/bin/env bash
# tools/new_run.sh — 建一个符合 AGENTS.md §5 规范的实验目录
#
# 用法:
#   tools/new_run.sh <name> [suffix]
# 例:
#   tools/new_run.sh r6-ca            # runs/2026-10-29-abc1234-r6-ca/
#   tools/new_run.sh r6-ca rerun      # runs/2026-10-29-abc1234-r6-ca-rerun/
#
# 只创建目录 + commit.txt（HEAD、git status、主机、磁盘/内存快照），
# 不覆盖任何已有目录。config.yaml / metrics.json / stdout.log 由调用者填。
set -eu

name=${1:?usage: tools/new_run.sh <name> [suffix]}
suffix=${2:-}
cd "$(dirname "$0")/.."

commit7=$(git rev-parse --short=7 HEAD)
dir="runs/$(date +%Y-%m-%d)-${commit7}-${name}${suffix:+-$suffix}"
if [ -e "$dir" ]; then
  echo "new_run: $dir 已存在，拒绝覆盖" >&2
  exit 1
fi
mkdir -p "$dir"
{
  echo "git_head: $(git rev-parse HEAD)"
  echo "--- git status --porcelain ---"
  git status --porcelain
  echo "--- created $(date -Is) ---"
  hostname
  df -h /home | tail -1
  free -h | head -2
} > "$dir/commit.txt"

echo "$dir"

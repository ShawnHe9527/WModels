#!/usr/bin/env bash
# tools/guard_run.sh — 工作站资源护栏：启动前检查 + 运行中监控 + 自动停机
#
# 用法:
#   tools/guard_run.sh RUN_DIR STDOUT_LOG -- CMD [ARGS...]
#
# 阈值（可用环境变量覆盖）:
#   DISK_START_GB=10   启动要求 /home 可用
#   DISK_WARN_GB=8     运行中告警
#   DISK_STOP_GB=3     运行中停止
#   MEM_START_GB=16    启动要求 MemAvailable
#   MEM_STOP_GB=16     运行中停止
#   RSS_STOP_GB=32     进程 RSS 上限
#   SAMPLE_S=60        采样间隔
#   NICE_LVL=5         进程 nice
#
# 输出: RUN_DIR/monitor.log（每次采样一行 + 停机原因），CMD 的 stdout/stderr 写 STDOUT_LOG。
# 退出码: 0 正常；1 preflight 不通过；2 护栏触发停机；其余为 CMD 自身退出码。
set -u

RUN_DIR=${1:?missing RUN_DIR}
STDOUT_LOG=${2:?missing STDOUT_LOG}
shift 2
[ "${1:-}" = "--" ] && shift
[ $# -ge 1 ] || { echo "guard_run: no command given" >&2; exit 64; }

DISK_START_GB=${DISK_START_GB:-10}
DISK_WARN_GB=${DISK_WARN_GB:-8}
DISK_STOP_GB=${DISK_STOP_GB:-3}
MEM_START_GB=${MEM_START_GB:-16}
MEM_STOP_GB=${MEM_STOP_GB:-16}
RSS_STOP_GB=${RSS_STOP_GB:-32}
SAMPLE_S=${SAMPLE_S:-60}
NICE_LVL=${NICE_LVL:-5}

mkdir -p "$RUN_DIR"
MON="$RUN_DIR/monitor.log"

disk_avail_gb() { df -BG --output=avail /home 2>/dev/null | tail -1 | tr -dc '0-9'; }
mem_avail_gb() { awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo; }
load1() { cut -d' ' -f1 /proc/loadavg; }

D=$(disk_avail_gb); M=$(mem_avail_gb); L=$(load1)
{
  echo "=== guard preflight $(date -Is) ==="
  echo "host=$(hostname)  cmd=$*"
  echo "disk /home  : ${D} GB   (require >= ${DISK_START_GB}, warn < ${DISK_WARN_GB}, stop < ${DISK_STOP_GB})"
  echo "MemAvailable: ${M} GB   (require >= ${MEM_START_GB}, stop < ${MEM_STOP_GB})"
  echo "RSS cap     : ${RSS_STOP_GB} GB"
  echo "load1       : ${L}"
} | tee -a "$MON"

if [ "${D:-0}" -lt "$DISK_START_GB" ]; then
  echo "PREFLIGHT FAIL: disk ${D}G < ${DISK_START_GB}G" | tee -a "$MON"; exit 1
fi
if [ "${M:-0}" -lt "$MEM_START_GB" ]; then
  echo "PREFLIGHT FAIL: mem ${M}G < ${MEM_START_GB}G" | tee -a "$MON"; exit 1
fi

nice -n "$NICE_LVL" "$@" > "$STDOUT_LOG" 2>&1 &
PID=$!
echo "pid=$PID  start=$(date -Is)" >> "$MON"

REASON=""
while kill -0 "$PID" 2>/dev/null; do
  sleep "$SAMPLE_S"
  RSS_KB=$(awk '/^VmRSS:/{print $2}' "/proc/$PID/status" 2>/dev/null || echo 0)
  RSS_GB=$(( ${RSS_KB:-0} / 1048576 ))
  M=$(mem_avail_gb); D=$(disk_avail_gb); L=$(load1)
  echo "[mon $(date -Is)] rss=${RSS_GB}GB avail=${M}GB disk=${D}GB load=${L}" >> "$MON"
  if [ "${RSS_KB:-0}" -gt $(( RSS_STOP_GB * 1048576 )) ]; then REASON="rss>${RSS_STOP_GB}GB"; break; fi
  if [ "${M:-0}" -lt "$MEM_STOP_GB" ]; then REASON="mem<${MEM_STOP_GB}GB"; break; fi
  if [ "${D:-0}" -lt "$DISK_STOP_GB" ]; then REASON="disk<${DISK_STOP_GB}GB"; break; fi
  if [ "${D:-0}" -lt "$DISK_WARN_GB" ]; then echo "[warn] disk=${D}GB < ${DISK_WARN_GB}GB" >> "$MON"; fi
done

if [ -n "$REASON" ]; then
  echo "ABORT: ${REASON}  $(date -Is)" >> "$MON"
  kill -TERM "$PID" 2>/dev/null
  sleep 10
  kill -KILL "$PID" 2>/dev/null
fi
wait "$PID" 2>/dev/null; RC=$?
echo "[exit] rc=$RC  $(date -Is)" >> "$MON"
exit "$RC"

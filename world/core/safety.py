"""资源安全护栏（内存 / CPU / 磁盘）。

## 为什么单独一个模块，而且必须在最前面 import

WSL2 环境：**7.8 GB RAM + 2 GB swap，且 swap 是磁盘文件**。
一旦内存打满 → swap 疯狂读写 → 磁盘 I/O 100% → WSL 直接失联（实测过多次）。

历史上踩过的坑（全部要防住）：

| 坑 | 后果 | 对策 |
|---|---|---|
| `.npz` 不支持 mmap | 760 MB 数据被反复读成 22 GB → 断连 | 只用逐 key `.npy` + mmap |
| `OMP_NUM_THREADS` 只设 `torch.set_num_threads` | 47 线程 → glibc arena 碎片 → OOM | **在 import numpy/torch 之前**设环境变量 |
| `detach()` 以为能省内存 | 计算图已经建好了，省不掉 | 靶子必须在 `no_grad` 下预计算 |
| CPU 上开 gradient checkpointing | 实测 2.69 s → 4.76 s（净亏）| 禁用 |
| 无内存监控 | 跑到一半突然断连，不知道原因 | 每步查 RSS，超阈值优雅退出并落 checkpoint |

## 用法（顺序很重要）

```python
from . import safety
safety.apply_thread_caps()      # ← 必须在 import numpy / torch 之前
import numpy as np
import torch
...
safety.preflight()              # 打印资源现状
```

⚠️ `apply_thread_caps()` 之后再 import numpy/torch 才会生效。
（numpy/torch 在 import 时读这些环境变量初始化线程池。）
"""
from __future__ import annotations

import os
import shutil
import time

# ---------------------------------------------------------------- 线程配置
# 留 2 核给系统，避免把机器占满导致 WSL 卡死
DEFAULT_THREADS = 6
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def apply_thread_caps(n: int = DEFAULT_THREADS) -> int:
    """把 BLAS/OpenMP 线程数钉死。**必须在 import numpy/torch 之前调用。**

    ⚠️ 只调 `torch.set_num_threads()` 是不够的：numpy 的 OpenBLAS 线程池
    不受 torch 管辖，实测会自己开到 47 个线程 ⇒ glibc 为每个线程创建 arena
    ⇒ 虚拟内存暴涨 ⇒ OOM。
    """
    for v in THREAD_ENV_VARS:
        os.environ[v] = str(n)
    # 抑制 tokenizer 并行等隐式开线程
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return n


def apply_torch_threads(n: int = DEFAULT_THREADS) -> None:
    """在 import torch 之后再钉一次（torch 会覆盖环境变量）。"""
    import torch
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass          # 已经初始化过就不能再设，忽略


# ---------------------------------------------------------------- 内存
def rss_mb() -> float:
    """当前进程常驻内存（MB）。读 /proc，不依赖 psutil。"""
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def mem_available_mb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


class MemGuard:
    """内存守卫：超阈值就抛 `MemoryGuardTripped`，让训练循环优雅收尾。

    用法：
    ```python
    guard = MemGuard(limit_mb=3500)
    for step in range(...):
        ...
        guard.check()          # 每步或每 N 步调一次
    ```
    ⚠️ 不要在 `check()` 里直接 `sys.exit()`：那样 checkpoint 来不及落盘。
        抛异常 → 训练循环捕获 → 存 checkpoint → 退出。
    """

    def __init__(self, limit_mb: float = 3500.0, floor_mb: float = 400.0):
        self.limit_mb = float(limit_mb)
        self.floor_mb = float(floor_mb)
        self.peak_mb = 0.0

    def check(self) -> None:
        cur = rss_mb()
        self.peak_mb = max(self.peak_mb, cur)
        if cur > self.limit_mb:
            raise MemoryGuardTripped(
                f"RSS {cur:.0f} MB 超过阈值 {self.limit_mb:.0f} MB")
        avail = mem_available_mb()
        if 0 < avail < self.floor_mb:
            raise MemoryGuardTripped(
                f"系统可用内存只剩 {avail:.0f} MB（< {self.floor_mb:.0f}）")


class MemoryGuardTripped(RuntimeError):
    pass


# ---------------------------------------------------------------- 磁盘
def disk_free_gb(path: str = ".") -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return -1.0


def disk_guard(path: str = ".", need_gb: float = 1.0) -> None:
    free = disk_free_gb(path)
    if 0 <= free < need_gb:
        raise MemoryGuardTripped(
            f"磁盘剩余 {free:.1f} GB < 需要 {need_gb:.1f} GB")


# ---------------------------------------------------------------- 预检
def preflight(limit_mb: float = 3500.0, verbose: bool = True) -> dict:
    info = dict(
        threads=os.environ.get("OMP_NUM_THREADS", "?"),
        rss_mb=rss_mb(),
        mem_available_mb=mem_available_mb(),
        disk_free_gb=disk_free_gb(),
        cpu_count=os.cpu_count(),
    )
    if verbose:
        print("─" * 78)
        print("资源预检")
        print(f"  CPU 逻辑核        : {info['cpu_count']}")
        print(f"  BLAS/OMP 线程     : {info['threads']}")
        print(f"  本进程 RSS        : {info['rss_mb']:.0f} MB "
              f"(护栏 {limit_mb:.0f} MB)")
        print(f"  系统可用内存      : {info['mem_available_mb']:.0f} MB")
        print(f"  磁盘剩余          : {info['disk_free_gb']:.0f} GB")
        ok = (info["mem_available_mb"] > 800 and info["disk_free_gb"] > 2)
        print(f"  判定              : {'✅ 可运行' if ok else '⚠️ 资源紧张'}")
        print("─" * 78)
        info["ok"] = ok
    return info


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def s(self) -> float:
        return time.perf_counter() - self.t0

    def reset(self) -> None:
        self.t0 = time.perf_counter()

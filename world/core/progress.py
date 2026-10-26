"""终端实时进度显示。

## 设计目标

1. **不会"静默"**：上一轮曾因为只在 `% eval_every == 0` 时打印，导致长时间
   看不到任何输出，无法区分"在跑"还是"死了"。
2. **单行原地刷新**：不刷屏，用 `\\r` 覆写同一行。
3. **能看到**：步数 / 进度 / 各损失分量 / lr / 梯度范数 / 吞吐 / ETA / 内存。
4. **自动降级**：输出被重定向到文件（非 tty）时，自动改成每 N 秒一行，
   避免 `\\r` 把日志文件写成一团。
5. **实时**：所有输出都 `flush=True`，管道下也能立刻看到。
"""
from __future__ import annotations

import sys
import time


def _fmt_secs(s: float) -> str:
    s = max(0.0, float(s))
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"


class Progress:
    """训练进度条。

    ```python
    p = Progress(total=20000, desc="T1 自编码器")
    for step in range(20000):
        ...
        p.update(step + 1, loss=..., lr=...)
    p.close()
    ```
    """

    def __init__(self, total: int, desc: str = "", width: int = 28,
                 every_secs: float = 2.0, show_every: int = 0):
        self.total = int(total)
        self.desc = desc
        self.width = int(width)
        self.every_secs = float(every_secs)
        self.show_every = int(show_every)
        self.t0 = time.perf_counter()
        self._last_print = 0.0
        self._last_step = 0
        self._last_t = self.t0
        self.is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._line_len = 0
        self._extra: dict[str, float] = {}

    # ------------------------------------------------------------------
    def update(self, step: int, **extra) -> None:
        self._extra.update(extra)
        now = time.perf_counter()
        force = self.show_every > 0 and step % self.show_every == 0
        if not force and (now - self._last_print) < self.every_secs:
            return
        self._last_print = now
        self._render(step, now)

    def _render(self, step: int, now: float) -> None:
        frac = min(1.0, step / self.total) if self.total else 0.0
        filled = int(round(self.width * frac))
        bar = "█" * filled + "░" * (self.width - filled)

        el = now - self.t0
        rate = step / el if el > 0 else 0.0
        eta = (self.total - step) / rate if rate > 0 else 0.0

        # 瞬时吞吐（最近一段）
        dt = now - self._last_t
        inst = (step - self._last_step) / dt if dt > 0.05 else rate
        self._last_t, self._last_step = now, step

        # 损失等附加项
        parts = []
        for k, v in self._extra.items():
            if isinstance(v, float):
                if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
                    parts.append(f"{k}={v:.3e}")
                else:
                    parts.append(f"{k}={v:.4g}")
            else:
                parts.append(f"{k}={v}")
        extras = ("  " + " ".join(parts)) if parts else ""

        # 内存
        try:
            from .safety import rss_mb
            rss = rss_mb()
            mem = f"  mem={rss:.0f}MB"
        except Exception:
            mem = ""

        line = (f"{self.desc} {frac*100:5.1f}% |{bar}| "
                f"{step}/{self.total} "
                f"[{_fmt_secs(el)}<{_fmt_secs(eta)}, {inst:.0f} it/s]"
                f"{extras}{mem}")

        if self.is_tty:
            pad = max(0, self._line_len - len(line))
            sys.stdout.write("\r" + line + " " * pad)
            self._line_len = len(line)
            sys.stdout.flush()
        else:
            # 非 tty：直接换行输出，日志文件才可读
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def step(self, **extra) -> None:
        """无 step 参数时按内部计数推进。"""
        self._internal = getattr(self, "_internal", 0) + 1
        self.update(self._internal, **extra)

    def close(self, msg: str = "") -> None:
        if self.is_tty and self._line_len:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._line_len = 0
        if msg:
            print(msg, flush=True)


def log(msg: str = "") -> None:
    """带时间戳的即时日志行（不会被进度条覆盖，先换行）。"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kv_block(title: str, d: dict, indent: str = "  ") -> None:
    print(title, flush=True)
    for k, v in d.items():
        if isinstance(v, float):
            if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
                print(f"{indent}{k:<24}{v:.4e}", flush=True)
            else:
                print(f"{indent}{k:<24}{v:.6g}", flush=True)
        else:
            print(f"{indent}{k:<24}{v}", flush=True)

"""指标：与基线 L0（复制上一帧）的配对比较。

## 口径

误差一律在**归一化空间**（已逐因子除以自身 std）计算，
所以 `rel_err` 是"平均几个标准差"，跨因子可比。

$$\varepsilon(k)=\big\langle\big|\hat s_{t+k}-s_{t+k}\big|\big\rangle$$

L0 基线：$\hat s_{t+k}:=s_t$。

## 必须配对

模型与 L0 必须在**同一批**采样序列上算。n=32 时 L0 的采样噪声可达 ±50%，
足以把"赢/输"判反（v2 的教训）。本模块的 `evaluate_horizons` 保证配对。
"""
from __future__ import annotations

import numpy as np


def norm_abs(pred: np.ndarray, true: np.ndarray) -> float:
    """归一化空间的平均绝对误差。"""
    return float(np.abs(pred - true).mean())


def l0_curve(s_all: np.ndarray, t0: np.ndarray, ks, L_max: int) -> dict:
    """L0 误差曲线。s_all: (B, L_max, nf, P) 真实未来。"""
    out = {}
    s0 = s_all[:, 0]
    for k in ks:
        out[k] = norm_abs(s0, s_all[:, k - 1])
    return out


def amplification(err: float, k: int, sigma: float) -> float:
    """放大比 a = ε(闭环) / (√k·σ)。

    a ≈ 1 ⇒ 误差是独立累积（随机游走）
    a ≫ 1 且随 k 单调上升 ⇒ **latent 轨迹在发散**（v2 实测 2.6→79.7）
    a < 1 ⇒ 误差相互抵消
    """
    denom = np.sqrt(max(k, 1)) * max(sigma, 1e-12)
    return float(err / denom)


def per_factor(err_map: dict, names) -> list[dict]:
    """把逐因子误差整理成可打印的行。"""
    rows = []
    for i, nm in enumerate(names):
        rows.append(dict(factor=nm, model=err_map["model"][i],
                         L0=err_map["L0"][i],
                         win=err_map["model"][i] < err_map["L0"][i]))
    return rows


def summarize(rows: list[tuple]) -> str:
    """rows: [(k, ms, model, L0, ratio_ok)] → 表格字符串。"""
    lines = [f"{'k':>6} {'ms':>7} {'模型':>10} {'L0':>10} {'模型/L0':>9} 判定"]
    for k, ms, m, l0, _ in rows:
        r = m / max(l0, 1e-12)
        lines.append(f"{k:>6} {ms:>7.1f} {m:>10.5f} {l0:>10.5f} "
                     f"{r:>9.2f} {'✅ 赢' if m < l0 else '❌ 输'}")
    return "\n".join(lines)

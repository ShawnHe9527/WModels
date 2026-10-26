"""输入生成：棘波序列策略（renewal + 不应期 + burst + ρ 同步）。

## 设计约束（来自 docs/archive/DESIGN_v3.md §5）

1. **per-synapse 的率**，不是"树级 ÷ 长度"
2. **伪随机 + 固定 seed**，每条轨迹记录 seed
3. 生理随机过程：gamma-renewal + 绝对不应期
4. 两个**正交**旋钮：
   - `burst`：时间维度，一次事件后追加 3–6 个棘波（间隙 3–8 ms）
   - `rho`：空间维度，用**共享 master 棘波串**实现
5. 兴奋 / 抑制各有**独立**的 master（不应完全同步）

## ⚠️ ρ 的实现要点（踩过坑）

只调制发放率**不行**：AMPA 的 τ=3 ms 远短于率包络的 σ=80 ms，
Poisson 噪声会淹没相关性（实测 r 仅 0.047）。
必须让各通道**复制**同一批时刻（master），实测 r 达 0.538（ρ=0.6）。

master 必须在**外层算一次**，不能在每通道循环里重采样。

## 实测标定（docs/archive/DESIGN_v3.md §5.2.1）

| rate_e (Hz/syn) | ρ | V max | cai 跨度 | 区域 |
|---|---|---|---|---|
| 60 | 0.0 | −65.2 | 0.014 | 亚阈线性 |
| 200 | 0.0 | −45.7 | 0.068 | 近阈值 |
| 120 | 0.6 | +20.6 | 1.057 | **Ca 平台** |
| 200 | 0.6 | +9.1 | 0.931 | Ca 平台 |

⇒ Ca 平台需 `rate_e ≥ 120` **且** `rho ≥ 0.6` 同时满足。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


# ==============================================================================
@dataclass
class Policy:
    """一条轨迹的输入生成参数（全部可复现）。"""
    rate_e: float          # Hz，每突触兴奋率（基础值）
    rate_i: float          # Hz，每突触抑制率
    rho: float             # [0,1] 空间同步性
    burst_prob: float      # 每事件后追加一簇的概率
    burst_n: tuple = (3, 6)
    burst_isi: tuple = (3.0, 8.0)
    refractory: float = 2.0    # ms，绝对不应期
    regularity: float = 0.0    # 0=Poisson(CV=1)，1=极规则
    jitter: float = 0.2        # 每通道率的空间抖动（比例）

    def to_dict(self) -> dict:
        d = asdict(self)
        d["burst_n"] = list(self.burst_n)
        d["burst_isi"] = list(self.burst_isi)
        return d


def sample_policy(rng: np.random.Generator) -> Policy:
    """分层采样一条协议（见 docs/archive/DESIGN_v3.md §5.2.2）。

    ## 为什么分层

    均匀独立采样 `rate~U(15,220)` × `rho~U(0,0.8)` 会**严重偏多**平台区
    （实测 20 条里 75% 出现 Ca 尖峰）。原因是 `rate·rho` 超过某个阈值就必然
    过阈，均匀采样把大量质量放在超阈区。

    改为三档分层，每档参数范围都经过实测标定：

    | 档 | 名称 | rate_e | rho | 预期行为 |
    |---|---|---|---|---|
    | A | linear | 10–80 | 0–0.35 | 亚阈线性 / 静息 |
    | B | inter | 80–160 | 0.20–0.60 | 强去极化，临界 |
    | C | plateau | 140–220 | 0.55–0.85 | Ca 平台 / 尖峰 |

    占比 35 / 30 / 35 ⇒ 预期平台区 ≈ 35%（门禁 15–45%）。
    """
    u = rng.random()
    if u < 0.45:
        regime, rate_e, rho = ("linear", rng.uniform(5.0, 55.0),
                               rng.uniform(0.0, 0.20))
    elif u < 0.85:
        regime, rate_e, rho = ("inter", rng.uniform(60.0, 110.0),
                               rng.uniform(0.10, 0.40))
    else:
        regime, rate_e, rho = ("plateau", rng.uniform(120.0, 220.0),
                               rng.uniform(0.50, 0.85))

    pol = Policy(
        rate_e=float(rate_e),
        rate_i=float(rate_e * rng.uniform(0.5, 0.85)),
        rho=float(rho),
        burst_prob=0.15 if rng.random() < 0.5 else 0.0,
        regularity=float(rng.uniform(0.0, 1.0)),
        jitter=float(rng.uniform(0.05, 0.4)),
    )
    pol.regime = regime          # 动态属性，供分析与分层评测
    return pol


# ==============================================================================
def _renewal_times(rng, rate_hz, t0, t1, regularity, refractory) -> np.ndarray:
    """gamma-renewal 过程：CV 由 regularity 控制，带绝对不应期。"""
    if rate_hz <= 0:
        return np.empty(0)
    mean_isi = 1000.0 / rate_hz
    # shape k ⇒ CV = 1/sqrt(k)。regularity=0 ⇒ k=1（Poisson）；→1 时 k 增大
    k = 1.0 / max(1.0 - 0.9 * float(regularity), 1e-3)
    out = []
    t = t0 + float(rng.uniform(0, mean_isi))
    n_max = int((t1 - t0) / mean_isi * 6) + 200
    for _ in range(n_max):
        if t >= t1:
            break
        out.append(t)
        t += max(float(rng.gamma(k, mean_isi / k)), refractory)
    return np.asarray(out, dtype=float)


def _add_bursts(rng, ts, t1, prob, n_range, isi_range) -> np.ndarray:
    """在每个事件后按 prob 追加一簇棘波（时间维度）。"""
    if prob <= 0 or len(ts) == 0:
        return ts
    extra = []
    for x in ts:
        if rng.random() < prob:
            n = int(rng.integers(n_range[0], n_range[1] + 1))
            tb = float(x)
            for _ in range(n):
                tb += float(rng.uniform(*isi_range))
                if tb < t1:
                    extra.append(tb)
    return np.sort(np.concatenate([ts, np.asarray(extra, dtype=float)]))


def make_spike_trains(rng: np.random.Generator, nseg: int, t0: float,
                      t1: float, pol: Policy):
    """生成一整套输入。

    返回
    ----
    ex : list[np.ndarray]   每个 compartment 的兴奋棘波时刻
    in_ : list[np.ndarray]  每个 compartment 的抑制棘波时刻
    info : dict             统计量（用于数据门禁）
    """
    span_s = (t1 - t0) / 1000.0

    # ---- 共享 master（只在**这里**算一次）----
    n_master_e = int(round(pol.rho * pol.rate_e * span_s))
    master_e = np.sort(rng.uniform(t0, t1, max(n_master_e, 0)))
    n_master_i = int(round(pol.rho * pol.rate_i * span_s))
    master_i = np.sort(rng.uniform(t0, t1, max(n_master_i, 0)))

    ex, inh = [], []
    for _ in range(nseg):
        jit = 1.0 + pol.jitter * (2.0 * rng.random() - 1.0)
        # 独立部分按 (1−ρ) 的率
        ts_e = _renewal_times(rng, pol.rate_e * jit * (1.0 - pol.rho),
                              t0, t1, pol.regularity, pol.refractory)
        ts_e = np.sort(np.concatenate([ts_e, master_e]))
        ts_e = _add_bursts(rng, ts_e, t1, pol.burst_prob,
                           pol.burst_n, pol.burst_isi)
        ex.append(ts_e)

        ts_i = _renewal_times(rng, pol.rate_i * jit * (1.0 - pol.rho),
                              t0, t1, pol.regularity, pol.refractory)
        ts_i = np.sort(np.concatenate([ts_i, master_i]))
        inh.append(ts_i)

    n_e = sum(len(t) for t in ex)
    n_i = sum(len(t) for t in inh)
    info = dict(
        n_event_e=n_e, n_event_i=n_i,
        rate_e_real=n_e / nseg / span_s,
        rate_i_real=n_i / nseg / span_s,
        n_master_e=int(len(master_e)),
    )
    return ex, inh, info


# ==============================================================================
def make_fixedn_pair(rng: np.random.Generator, nseg: int, t0: float, t1: float,
                     base: Policy, rho_hi: float):
    """对照组：固定总事件数，只改 ρ（见 docs/archive/DESIGN_v3.md §5.6）。

    先按 base（ρ=0）生成，数出每突触平均事件数；再把高 ρ 条件下的
    **独立部分**率反解，使总事件数基本不变。
    """
    span_s = (t1 - t0) / 1000.0
    lo = Policy(**{**base.to_dict(), "rho": 0.0})
    ex_lo, in_lo, info_lo = make_spike_trains(rng, nseg, t0, t1, lo)

    per_syn_target = info_lo["n_event_e"] / nseg
    per_syn_indep = max(per_syn_target - rho_hi * base.rate_e * span_s, 0.0)
    hi = Policy(**{**base.to_dict(), "rho": rho_hi,
                   "rate_e": per_syn_indep / max(span_s, 1e-9)})
    ex_hi, in_hi, info_hi = make_spike_trains(rng, nseg, t0, t1, hi)
    return (ex_lo, in_lo, info_lo), (ex_hi, in_hi, info_hi)

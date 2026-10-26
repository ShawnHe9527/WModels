"""v4 输入生成：目标段自身输入 + tuft 簇状输入 + 胞体 bAP 时刻表。

## 与 v3 `protocol.py` 的关系

- **目标段自身输入**：完全复用 `protocol.make_spike_trains`（rate_e / rate_i /
  ρ 共享 master 棘波串）。v3 的标定结论继续有效。
- **tuft 输入**：本模新加。用**招募比例 $f$**（D5）而不是 ρ，因为
  `exp_bundle` 里齐射数 k 是我们直接指定的，ρ 落地会有二重间接。
- **bAP**：本模新加。由胞体真实 AP 产生（D2），这里只生成时刻表。

## 为什么 tuft 用「簇」

实测（`tools/probe_neighbor_caspike.py --exp syn`）：56 个 tuft 突触上投
**均匀 Poisson**，≥100 Hz 时退化为**一次持续整条 trial 的平台**
（>−45 mV 占 99% 时间，事件数 = 1）。

⇒ 均匀 Poisson 和恒定 DC 一样把输入「拉平」了，只是工具从电流换成电导。
只有「簇」（同步集合放电）才产生离散事件，而这正是皮层传入的真实形态。

## 为什么 bAP 与簇只部分耦合（D3 = 0.4）

若强制锁定（p_lead = 1.0），模型只需学「有 bAP 就有事件」，退化成单比特预测。
保留 60% 偶然性，模型必须真正整合两条通路 ⇒ 任务才有科学价值。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import protocol as P1


# ==============================================================================
@dataclass
class Policy2:
    """一条 v4 轨迹的完整输入参数（全部可复现）。"""

    # ---- 目标段自身输入（沿用 v3）----
    rate_e: float          # Hz，目标段每突触兴奋率
    rate_i: float          # Hz，目标段每突触抑制率
    rho: float             # [0,1] 目标段空间同步性

    # ---- tuft 簇状输入（新）----
    f_tuft: float          # [0,1] 每簇里参与齐射的 tuft 突触比例
    bundle_hz: float       # Hz，簇发生率
    width: float           # ms，簇内齐射时间窗

    # ---- 胞体 bAP（新）----
    bap_rate: float        # Hz，独立 bAP 率（不含 p_lead 触发）
    p_lead: float          # [0,1] 簇前触发 bAP 的概率
    bap_amp: float         # nA，单发窗口 3.5–4.5
    bap_width: float       # ms

    # ---- 沿用 v3 的时序参数 ----
    burst_prob: float = 0.0
    burst_n: tuple = (3, 6)
    burst_isi: tuple = (3.0, 8.0)
    refractory: float = 2.0
    regularity: float = 0.0
    jitter: float = 0.2

    regime: str = "inter"          # 动态属性，不参与 asdict

    def to_dict(self) -> dict:
        d = asdict(self)
        d["burst_n"] = list(self.burst_n)
        d["burst_isi"] = list(self.burst_isi)
        d["regime"] = self.regime
        return d

    def to_policy1(self) -> P1.Policy:
        return P1.Policy(
            rate_e=self.rate_e, rate_i=self.rate_i, rho=self.rho,
            burst_prob=self.burst_prob, burst_n=self.burst_n,
            burst_isi=self.burst_isi, refractory=self.refractory,
            regularity=self.regularity, jitter=self.jitter,
        )


# ==============================================================================
def sample_policy2(rng: np.random.Generator) -> Policy2:
    """分层采样（沿 v3 的三档思路，按 v4 新机制标定）。

    | 档 | f_tuft | bAP 率 | 预期 |
    |---|---|---|---|
    | linear  | 0.05–0.35 | 0–8 Hz   | 亚阈，无平台 |
    | inter   | 0.25–0.60 | 5–14 Hz  | 偶发事件 |
    | plateau | 0.45–1.00 | 10–20 Hz | 稳定平台 |

    ⚠️ O1 密度（17 E）下，无 bAP 时满簇总电导 0.0068 µS < 阈值 0.008 µS
    ⇒ **plateau 档必须同时给高 f 与高 bAP 率**，否则必然失败。
    """
    u = rng.random()
    if u < 0.50:
        regime, f, bp = "linear", rng.uniform(0.10, 0.40), rng.uniform(0.0, 8.0)
        re_, rho = rng.uniform(5.0, 55.0), rng.uniform(0.0, 0.20)
    elif u < 0.80:
        regime, f, bp = "inter", rng.uniform(0.25, 0.55), rng.uniform(5.0, 14.0)
        re_, rho = rng.uniform(60.0, 110.0), rng.uniform(0.10, 0.40)
    else:
        regime, f, bp = "plateau", rng.uniform(0.45, 0.85), rng.uniform(10.0, 20.0)
        re_, rho = rng.uniform(120.0, 220.0), rng.uniform(0.50, 0.85)

    pol = Policy2(
        rate_e=float(re_), rate_i=float(rng.uniform(40.0, 120.0)),
        rho=float(rho),
        f_tuft=float(f),
        bundle_hz=float(rng.uniform(2.0, 8.0)),
        width=float(rng.uniform(3.0, 8.0)),
        bap_rate=float(bp),
        p_lead=0.4,
        bap_amp=float(rng.uniform(3.5, 4.5)),
        bap_width=2.0,
        burst_prob=float(rng.uniform(0.0, 0.25)),
        jitter=float(rng.uniform(0.1, 0.3)),
    )
    pol.regime = regime
    return pol


# ==============================================================================
def make_tuft_bundles(rng: np.random.Generator, n_e: int, n_i: int,
                      t0: float, t1: float, pol: Policy2,
                      rate_i: float = 15.0):
    """生成 tuft 的 E（簇状）与 I（均匀背景）事件时刻表。

    Returns
    -------
    ex : list[np.ndarray]  长度 n_e，每个 tuft 兴奋性突触的事件时刻
    ix : list[np.ndarray]  长度 n_i
    bundles : np.ndarray   簇发生时刻（供 bAP 耦合用）
    """
    span_s = (t1 - t0) / 1000.0
    n_b = int(rng.poisson(max(pol.bundle_hz, 1e-6) * span_s))
    bundles = np.sort(rng.uniform(t0, t1, n_b)) if n_b else np.empty(0)

    ev = [[] for _ in range(n_e)]
    for tb in bundles:
        k = int(rng.binomial(n_e, min(max(pol.f_tuft, 0.0), 1.0)))
        if k <= 0:
            continue
        sel = rng.choice(n_e, size=k, replace=False)
        for j in sel:
            ev[j].append(rng.uniform(tb, tb + pol.width))
    ex = [np.sort(np.asarray(v, dtype=np.float64)) if v
          else np.empty(0, dtype=np.float64) for v in ev]

    # 抑制性：独立均匀 Poisson（背景，不锁定在兴奋簇上）
    ix = []
    for _ in range(n_i):
        n = int(rng.poisson(max(rate_i, 1e-6) * span_s))
        ix.append(np.sort(rng.uniform(t0, t1, n)))

    return ex, ix, bundles


# ==============================================================================
def make_bap_times(rng: np.random.Generator, bundles: np.ndarray,
                   t0: float, t1: float, pol: Policy2,
                   max_n: int = 32) -> np.ndarray:
    """bAP 时刻表：一部分由簇前触发（p_lead），其余独立 Poisson。

    ⚠️ 上限 max_n：胞体用 IClamp 脉冲池实现（`Vector.play` 对
    `IClamp.amp` 不生效，见 DESIGN_v4 §8.2），池大小固定。
    """
    out = []
    for tb in bundles:
        if rng.random() < pol.p_lead:
            out.append(tb - rng.uniform(2.0, 8.0))
    span_s = (t1 - t0) / 1000.0
    n_ind = int(rng.poisson(max(pol.bap_rate, 1e-6) * span_s))
    if n_ind:
        out.extend(rng.uniform(t0, t1, n_ind).tolist())

    ts = np.sort(np.asarray([t_ for t_ in out if t0 - 10.0 <= t_ <= t1],
                            dtype=np.float64))
    if len(ts) > max_n:
        # 超限时均匀抽稀，而不是直接截断（避免系统性丢掉后半段）
        keep = np.sort(rng.choice(len(ts), size=max_n, replace=False))
        ts = ts[keep]
    return np.maximum(ts, t0 + 0.1)

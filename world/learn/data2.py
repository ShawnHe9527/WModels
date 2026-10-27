"""v4 数据读取：85 维原始记录 → token 形式（支持 T1 / T2 两档）。

## 为什么要分两层

磁盘上只存 **85 维原始记录**（`u.dat`，comp-major `[A,N,G]` 重复）。
训练时按需组装成 token：

| 档位 | 组成 | token 数 $P_u$ | 说明 |
|---|---|---|---|
| **T2** | 11 自己的 comp + **17 邻段 comp** | **28** | 全分辨率 |
| **T1** | 11 自己的 comp + **3 邻段 section** | **14** | 按 section 求和 |

T1 是 T2 的**确定性函数** ⇒ 两种观测**完全配对**（同一个仿真的同一份数据），
T1 vs T2 的差异纯粹来自观测分辨率，不含任何样本差异。

## 每个 token 的 5 个通道

| # | 通道 | 说明 |
|---|---|---|
| 0 | AMPA | 标准化后 |
| 1 | NMDA（去门控）| 标准化后 |
| 2 | GABA | 标准化后 |
| 3 | `frac` | 在**自己所属 section 内**的相对位置，∈ (0,1) |
| 4 | `side` | −1 = 近端父段，0 = 目标段本体，+1 = 远端子树 |

后两个通道**不标准化**：它们是几何坐标，本来就 O(1)。
（标准化常数列会让 `scale→0`，除出来是天文数字。）

## 为什么 bAP 不进 token

bAP 作用于**整个细胞**，不是某个 compartment。若也做成 token，
`mean`/`max` 池化会把它平均掉。所以单独作为全局向量 `u_glob` (B,L,1)
拼在池化之后（与 θ 同级）。
"""
from __future__ import annotations

import json
import os

import numpy as np

N_FACTORS = 9
GATE_CH = 3          # AMPA / NMDA / GABA
U_FEAT = GATE_CH + 2  # + frac + side
N_GLOB = 1           # bAP

# apic[51] / apic[50] / apic[52] / apic[53] 的 compartment 数
P_OWN = 11
P_NB = {"t2": 17, "t1": 3}
NB_COMP = (3, 5, 9)          # 邻段各 section 的 comp 数（t2 用）
NB_SIDE = (-1, 1, 1)


# ==============================================================================
def _mm(path, shape):
    return np.memmap(path, dtype=np.float32, mode="r", shape=shape)


class Dataset2:
    """v4 数据集的只读视图。"""

    def __init__(self, root: str, name: str, tier: str = "t2"):
        assert tier in ("t1", "t2"), tier
        self.tier = tier
        self.name = name
        self.dir = os.path.join(root, name)
        with open(os.path.join(self.dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        m = self.meta
        if m.get("version") != "v4":
            raise ValueError(
                f"{name} 不是 v4 数据集（version={m.get('version')}）"
                f"，请用 world.gen.datagen2 重新生成")

        self.N = int(m["n_trial"])
        self.T = int(m["T"])
        self.nseg = int(m["nseg"])
        self.nf = len(m["factor_names"])
        self.dt = float(m["dt"])
        self.u_dim = int(m["u_dim"])
        self.nb_n_e = int(m.get("nb_n_e", 1))

        self.s = _mm(os.path.join(self.dir, "s.dat"),
                     (self.N, self.T, self.nf, self.nseg))
        self.u = _mm(os.path.join(self.dir, "u.dat"),
                     (self.N, self.T, self.u_dim))
        self.th = np.load(os.path.join(self.dir, "th.npy"))

        # 状态 token 的位置编码（目标段自身）
        self.x = ((np.arange(self.nseg) + 0.5) / self.nseg).astype(np.float32)

        # ---- 输入 token 的位置特征（常量，一次算好）----
        self.xu, self.P_u = self._pos_features(tier)

        # ---- 每 trial 的 regime（用于分档评测）----
        self.regimes = np.array(
            [t.get("regime", "") for t in m.get("trials", [])], dtype=object)

    # ------------------------------------------------------------------
    @staticmethod
    def _pos_features(tier: str):
        """→ (xu:(P_u,2) float32 [frac, side], P_u)。"""
        rows = []
        # 目标段自身的 11 个 comp
        for j in range(P_OWN):
            rows.append(((j + 0.5) / P_OWN, 0.0))
        if tier == "t2":
            for n, side in zip(NB_COMP, NB_SIDE):
                for j in range(n):
                    rows.append(((j + 0.5) / n, float(side)))
        else:
            # T1：每个 section 一个"伪 comp"，取段中点
            for _n, side in zip(NB_COMP, NB_SIDE):
                rows.append((0.5, float(side)))
        xu = np.asarray(rows, dtype=np.float32)
        return xu, xu.shape[0]

    def __repr__(self) -> str:
        return (f"Dataset2({self.name}, tier={self.tier}, N={self.N}, "
                f"T={self.T}, nf={self.nf}, nseg={self.nseg}, "
                f"P_u={self.P_u}, u_dim={self.u_dim})")

    # ------------------------------------------------------------------
    def u_tokens(self, raw: np.ndarray) -> np.ndarray:
        """raw:(...,85) → tokens:(..., P_u, 5)。"""
        lead = raw.shape[:-1]
        own = raw[..., 0:33].reshape(*lead, P_OWN, GATE_CH)
        if self.tier == "t2":
            nb = raw[..., 33:84].reshape(*lead, 17, GATE_CH)
        else:
            nb = raw[..., 33:42].reshape(*lead, 3, GATE_CH)
        tok = np.concatenate([own, nb], axis=-2)          # (...,P_u,3)
        return np.concatenate(
            [tok, np.broadcast_to(self.xu, (*lead, self.P_u, 2))],
            axis=-1)

    def u_glob(self, raw: np.ndarray) -> np.ndarray:
        """raw:(...,85) → bAP:(...,1)。"""
        return raw[..., 84:85]


# ==============================================================================
def compute_scaler2(ds: Dataset2, n_sample: int = 400, seed: int = 0) -> dict:
    """逐因子 `s` 与逐通道 `u` 的 loc/scale（只用训练集）。"""
    rng = np.random.default_rng(seed)
    p = rng.integers(0, ds.N, n_sample)
    t = rng.integers(0, ds.T, n_sample)
    S = ds.s[p, t]                          # (n,nf,P)
    U = ds.u[p, t][..., :84]                # (n,84)

    loc_s = S.mean(axis=(0, 2))
    scale_s = np.maximum(S.std(axis=(0, 2)), 1e-6)

    # u 是 comp-major [A,N,G] 重复 ⇒ 按 3 取模分通道
    U3 = U.reshape(n_sample, -1, 3)          # (n,28*3/3=28, 3)
    loc_u = U3.mean(axis=(0, 1))
    scale_u = np.maximum(U3.std(axis=(0, 1)), 1e-9)

    return dict(
        factors=ds.meta["factor_names"],
        u_layout="comp-major [AMPA, NMDA_ungated, GABA] x P",
        loc_s=loc_s.astype(np.float32).tolist(),
        scale_s=scale_s.astype(np.float32).tolist(),
        loc_u=loc_u.astype(np.float32).tolist(),
        scale_u=scale_u.astype(np.float32).tolist(),
    )


def normalizers2(scaler: dict):
    loc_s = np.asarray(scaler["loc_s"], dtype=np.float32)[None, :, None]
    sc_s = np.asarray(scaler["scale_s"], dtype=np.float32)[None, :, None]
    loc_u = np.asarray(scaler["loc_u"], dtype=np.float32)      # (3,)
    sc_u = np.asarray(scaler["scale_u"], dtype=np.float32)

    def ns(s):        # (B, ..., nf, P)
        return (s - loc_s) / sc_s

    def nu_raw(raw):
        """raw:(B,L,85) → 标准化后的 (B,L,85)，bAP 保持不变。"""
        g = raw[..., :84].reshape(*raw.shape[:-1], -1, 3)
        g = (g - loc_u) / sc_u
        out = np.empty_like(raw)
        out[..., :84] = g.reshape(*raw.shape[:-1], 84)
        out[..., 84] = raw[..., 84]
        return out
    return ns, nu_raw


# ==============================================================================
class Sampler2:
    """按窗口随机采样（v4）。"""

    def __init__(self, ds: Dataset2, scaler: dict, batch: int = 64,
                 length: int = 64, seed: int = 1,
                 p_trial: np.ndarray | None = None,
                 hot_frac: float = 0.0,
                 event_center_frac: float = 0.0,
                 event_center_thr: float = -3.1):
        """p_trial    : 每个 trial 的抽样概率（None = 均匀）。仅训练用。
        hot_frac   : batch 里「含事件窗口」的强制比例（0 = 不用）。
                     事件 = trial 的 n_tuft_event>0 或 plateau。
        event_center_frac : batch 里以 Ca episode 中心开窗的比例（0 = 关闭）。
        event_center_thr  : Ca episode 判据（cai 的 log10 值）。

        ⚠️ 这些参数**只能给训练 sampler 用**。验证/评测必须传默认值，
        否则分布变了，a(k) 不可比。
        """
        self.ds = ds
        self.ns, self.nu = normalizers2(scaler)
        self.batch = batch
        self.length = length
        self.rng = np.random.default_rng(seed)
        self.p_trial = None
        if p_trial is not None:
            p = np.asarray(p_trial, dtype=np.float64)
            assert p.shape == (ds.N,), f"p_trial 长度应为 {ds.N}"
            self.p_trial = p / p.sum()
        self.hot_frac = float(hot_frac)
        self.event_center_frac = float(np.clip(event_center_frac, 0.0, 1.0))
        self.event_centers: list[np.ndarray] = [
            np.empty(0, dtype=np.int64) for _ in range(ds.N)]
        # 预计算每个 trial 的 Ca episode 中心。只读 cai 通道，200×40000 约
        # 0.35 GB 的临时读；初始化一次，训练循环里只查表。
        if self.event_center_frac > 0:
            ci = ds.meta["factor_names"].index("cai")
            n_ev = 0
            for i in range(ds.N):
                c = np.asarray(ds.s[i, :, ci, :]).max(axis=1)
                m = c > float(event_center_thr)
                d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
                starts = np.where(d == 1)[0]
                ends = np.where(d == -1)[0]
                if len(starts):
                    self.event_centers[i] = ((starts + ends) // 2).astype(np.int64)
                    n_ev += 1
            if n_ev == 0:
                self.event_center_frac = 0.0

        # 「含事件 trial」判据：**必须用目标段的 Ca 指标**。
        # ⚠️ 不能用 n_tuft_event —— 那是*输入侧*（邻段突触）的事件，
        #    实测 199/200 个 trial 都有，分层会完全失效。
        # 实测区分度（有/无 Ca 事件两组）：cai_span 1.84 > V_span 1.84 >
        #   plateau 1.75 > mid_frac 1.51 > n_tuft_event 0.77。
        # 取 cai_span 前 30% ⇒ 覆盖 95% 的有事件 trial。
        spans = np.array([t.get("cai_span", 0.0)
                          for t in ds.meta["trials"]], dtype=np.float64)
        if len(spans) and self.hot_frac > 0:
            thr = np.percentile(spans, 70.0)
            self.hot = np.where(spans >= thr)[0].astype(np.int64)
        else:
            self.hot = np.zeros(0, dtype=np.int64)
        if self.hot_frac > 0 and len(self.hot) == 0:
            self.hot_frac = 0.0

    def _draw_trials(self, n: int) -> np.ndarray:
        if self.p_trial is None:
            return self.rng.integers(0, self.ds.N, n)
        return self.rng.choice(self.ds.N, n, p=self.p_trial)

    def batch_pair(self):
        """→ (s0, u_tok, u_glob, th, s_true, x)。

        s0     : (B, nf, Ps)
        u_tok  : (B, L, P_u, 5)
        u_glob : (B, L, 1)
        th     : (B, th_dim)
        s_true : (B, L, nf, Ps)
        x      : (Ps,)
        """
        L = self.length
        ds = self.ds
        # 分层采样：一部分窗口强制落在「含事件」的 trial 上
        nh = 0
        if self.hot_frac > 0:
            nh = int(round(self.hot_frac * self.batch))
            nh = min(nh, self.batch)
        p = np.concatenate([self.rng.choice(self.hot, nh),
                            self._draw_trials(self.batch - nh)])
        self.rng.shuffle(p)                    # 防止 hot 总排在前面
        t0 = self.rng.integers(0, ds.T - L - 1, self.batch)
        if self.event_center_frac > 0:
            n_ev = min(int(round(self.event_center_frac * self.batch)),
                       self.batch)
            which = self.rng.choice(self.batch, n_ev, replace=False)
            for j in which:
                cents = self.event_centers[int(p[j])]
                if len(cents) == 0:
                    continue
                c = int(self.rng.choice(cents))
                jit = int(self.rng.integers(-max(L // 4, 1),
                                            max(L // 4, 1) + 1))
                t0[j] = int(np.clip(c - L // 2 + jit, 0, ds.T - L - 1))
        idx = t0[:, None] + np.arange(L)[None, :]          # (B,L)

        s0 = np.asarray(ds.s[p, t0])                        # (B,nf,Ps)
        s_true = np.asarray(ds.s[p[:, None], idx])          # (B,L,nf,Ps)
        raw = np.asarray(ds.u[p[:, None], idx])             # (B,L,85)

        # ---- 标准化（只动电导，不动 bAP）----
        raw = self.nu(raw)
        u_tok = ds.u_tokens(raw).astype(np.float32)          # (B,L,P_u,5)
        u_glob = ds.u_glob(raw).astype(np.float32)           # (B,L,1)
        th = np.asarray(ds.th[p])
        return self.ns(s0), u_tok, u_glob, th, self.ns(s_true), ds.x

    # ------------------------------------------------------------------
    def batch_window(self, t0: int, L: int, n: int, off: int = 0):
        """固定起点的一批窗口（评测用，保证不同模型看到同一批数据）。"""
        ds = self.ds
        p = (np.arange(n) + off) % ds.N
        idx = t0 + np.arange(L)[None, :]
        s0 = np.asarray(ds.s[p, t0])
        s_true = np.asarray(ds.s[p[:, None], idx])
        raw = self.nu(np.asarray(ds.u[p[:, None], idx]))
        return (self.ns(s0), ds.u_tokens(raw).astype(np.float32),
                ds.u_glob(raw).astype(np.float32), np.asarray(ds.th[p]),
                self.ns(s_true), ds.x)

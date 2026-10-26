"""数据集读取、采样、标准化。

## 内存策略

每个数据集 4.2 GB（s 3.2 + u 1.1）。**绝不整体读入**（可用 RAM 仅 1.3–1.8 GB）。
一律 `np.memmap` + 按索引切片。

## 标准化

逐因子（9 个）算 `loc` / `scale`，**只用训练集**。
`cai` 在生成时已取 log10，直接标准化。

输入 `u` 也逐通道标准化（AMPA / NMDA / GABA 三个量纲差很大）。

## 采样

一条样本 = 一段长度为 `L` 的连续时间窗（默认 L=64 步 = 1.6 ms @ dt=0.025）。
窗口起点在 test 相内随机。
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..core import config as C

N_FACTORS = 9
U_CH = 3


# ==============================================================================
def data_dir(root: str, name: str) -> str:
    return os.path.join(root, name)


def _mm(path: str, shape: tuple) -> np.memmap:
    return np.memmap(path, dtype=np.float32, mode="r", shape=shape)


class Dataset:
    """一个数据集的只读视图。"""

    def __init__(self, root: str, name: str):
        self.name = name
        self.dir = data_dir(root, name)
        with open(os.path.join(self.dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        m = self.meta
        self.N = int(m["n_trial"])
        self.T = int(m["T"])
        self.nseg = int(m["nseg"])
        self.nf = len(m["factor_names"])
        self.dt = float(m["dt"])

        self.s = _mm(os.path.join(self.dir, "s.dat"),
                     (self.N, self.T, self.nf, self.nseg))
        self.u = _mm(os.path.join(self.dir, "u.dat"),
                     (self.N, self.T, U_CH, self.nseg))
        self.ub = _mm(os.path.join(self.dir, "ub.dat"),
                      (self.N, self.T, 3))
        self.th = np.load(os.path.join(self.dir, "th.npy"))       # (N,27)
        self.x = ((np.arange(self.nseg) + 0.5) / self.nseg).astype(
            np.float32)          # 位置编码（⚠️ 必须 float32）

    def __repr__(self) -> str:
        return (f"Dataset({self.name}, N={self.N}, T={self.T}, "
                f"nf={self.nf}, nseg={self.nseg}, dt={self.dt})")


# ==============================================================================
def compute_scaler(ds: Dataset, n_sample: int = 400, seed: int = 0) -> dict:
    """逐因子 / 逐通道 的 loc 与 scale。"""
    rng = np.random.default_rng(seed)
    p = rng.integers(0, ds.N, n_sample)
    t = rng.integers(0, ds.T, n_sample)
    S = np.stack([ds.s[p[i], t[i]] for i in range(n_sample)])   # (n,nf,P)
    U = np.stack([ds.u[p[i], t[i]] for i in range(n_sample)])

    loc_s = S.mean(axis=(0, 2))
    scale_s = S.std(axis=(0, 2))
    scale_s = np.maximum(scale_s, 1e-6)

    loc_u = U.mean(axis=(0, 2))
    scale_u = U.std(axis=(0, 2))
    scale_u = np.maximum(scale_u, 1e-9)

    return dict(
        factors=ds.meta["factor_names"],
        u_channels=ds.meta["u_channels"],
        loc_s=loc_s.astype(np.float32).tolist(),
        scale_s=scale_s.astype(np.float32).tolist(),
        loc_u=loc_u.astype(np.float32).tolist(),
        scale_u=scale_u.astype(np.float32).tolist(),
    )


def normalizers(scaler: dict):
    loc_s = np.asarray(scaler["loc_s"], dtype=np.float32)[None, :, None]
    sc_s = np.asarray(scaler["scale_s"], dtype=np.float32)[None, :, None]
    loc_u = np.asarray(scaler["loc_u"], dtype=np.float32)[None, :, None]
    sc_u = np.asarray(scaler["scale_u"], dtype=np.float32)[None, :, None]

    def ns(s):        # (B, ..., nf, P)
        return (s - loc_s) / sc_s

    def nu(u):
        return (u - loc_u) / sc_u
    return ns, nu


# ==============================================================================
class Sampler:
    """按窗口随机采样。"""

    def __init__(self, ds: Dataset, scaler: dict, batch: int = 64,
                 length: int = 64, seed: int = 1):
        self.ds = ds
        self.ns, self.nu = normalizers(scaler)
        self.batch = batch
        self.length = length
        self.rng = np.random.default_rng(seed)

    def batch_pair(self):
        """→ (s0, u, th, s_true, x)。形状 s0:(B,nf,P) u:(B,L,3,P)。

        ⚠️ 用**花式索引**一次取完，不要写 Python 双重循环
        （B=64, L=64 ⇒ 4096 次切片，会让 CPU 成为瓶颈）。
        """
        L = self.length
        ds = self.ds
        p = self.rng.integers(0, ds.N, self.batch)
        t0 = self.rng.integers(0, ds.T - L - 1, self.batch)
        idx = t0[:, None] + np.arange(L)[None, :]          # (B,L)

        s0 = np.asarray(ds.s[p, t0])                        # (B,nf,P)
        s_true = np.asarray(ds.s[p[:, None], idx])          # (B,L,nf,P)
        u = np.asarray(ds.u[p[:, None], idx])               # (B,L,3,P)
        th = np.asarray(ds.th[p])
        return self.ns(s0), self.nu(u), th, self.ns(s_true), ds.x

"""DESIGN_v4 §6.5 的三个诊断（**不改变架构**，只做测量）。

| 诊断 | 内容 | 判据 |
|---|---|---|
| **D-a** | 9 因子分三组报误差（快门控 / `cai` / `V`）| `cai` 误差 ≥ 2× 快门控 且长视界不收敛 ⇒ 才考虑因子分组 |
| **D-b** | 用**线性探针**从 $z_t$ 解码「过去 20 ms 内有没有 bAP」| **AUC > 0.8** ⇒ latent 真的记住了历史依赖（核心卖点）|
| **D-c** | $d_z$ 扫 16/32/48（另跑 train2，不在此处）| 16→32 改善 > 15% ⇒ 瓶颈是容量 |

## 为什么 D-b 是核心卖点

bAP 只以 **1 维事件**进入 $u$，但它对 Ca 阈值有**乘性**影响
（实测 2.25×，见 DESIGN_v4 §2.2），必须被记住 15 ms 以上。
若单帧 $u$ 不够、而线性探针能从 $z_t$ 读出，就直接证明
「隐状态承载历史」——这正是世界模型的核心宣称，且**可证伪**。

## 为什么用线性探针而不是 MLP

要证明的是「信息**存在**于 latent 中」，不是「能被复杂网络挖出来」。
线性探针是最强的证伪形式：**线性可读 ⇒ 信息是显式编码的**。

## 用法

    cd WModel2 && PYTHONPATH=$PWD python -m tools.diag_latent \
        --ckpt t2v5_events/last.pt --data te_s3 --tier t2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.core import safety                     # noqa: E402

THREADS = safety.apply_thread_caps(4)

import numpy as np                                # noqa: E402
import torch                                      # noqa: E402
import torch.nn.functional as F                   # noqa: E402

from world.learn import data2 as D2               # noqa: E402
from world.learn import model as M                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results", "checkpoints")
SCALER = os.path.join(ROOT, "results", "data", "scaler_v4.json")

# 因子分组（与 sim2.FACTOR_NAMES 顺序一致）
FAST = ["Ih.m", "Ca_LVAst.m", "Ca_LVAst.h", "Ca_HVA.h",
        "SKv3_1.m", "SK_E2.z", "Im.m"]
FAST_IDX = list(range(7))
CAI_IDX = [7]
V_IDX = [8]


def log(msg: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _torch(batch):
    s0, u_tok, u_g, th, s_true, x = batch
    u = torch.from_numpy(np.ascontiguousarray(
        np.asarray(u_tok).transpose(0, 1, 3, 2)))
    return (torch.from_numpy(np.asarray(s0)), u,
            torch.from_numpy(np.asarray(u_g)),
            torch.from_numpy(np.asarray(th)),
            torch.from_numpy(np.asarray(s_true)),
            torch.from_numpy(np.asarray(x)))


def auc_score(y, s) -> float:
    """rank-based AUC（不需要 sklearn）。"""
    y = np.asarray(y).ravel().astype(bool)
    s = np.asarray(s).ravel().astype(np.float64)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列：取平均秩
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ==============================================================================
def collect(model, ds, scaler, n_win: int, L: int, seed: int = 0):
    """采 n_win 个窗口，返回 (z, s_true, u_glob, s0)。"""
    sp = D2.Sampler2(ds, scaler, batch=n_win, length=L, seed=seed)
    s0, u_tok, u_g, th, s_true, x = _torch(sp.batch_pair())
    with torch.no_grad():
        z = model.encode(s_true, u_tok, th, x, u_g)
    return z.numpy(), s_true.numpy(), u_g.numpy(), s0.numpy()


def d_b(model, ds, scaler, n_win=256, L=512, lookback_ms=20.0):
    """D-b：从 z_t 线性解码「过去 lookback_ms 内有无 bAP」。

    ⚠️ lookback_ms 上限受窗口长度限制：若 lookback_ms > L*dt，`lo` 恒为 0，
    标签退化成「窗口起点以来累计有无 bAP」，不同 lookback 会给出**完全相同**
    的结果。要测 50 ms 需 L ≥ 2000 步（dt=0.025）。这里给出硬警告。
    """
    span_ms = L * ds.dt
    if lookback_ms > span_ms:
        print(f"  ⚠️ lookback {lookback_ms:.0f} ms > 窗口跨度 {span_ms:.1f} ms，"
              f"标签被截断，请加大 --length")
    z, s_true, u_g, s0 = collect(model, ds, scaler, n_win, L)
    ds_ms = ds.dt
    w = max(1, int(round(lookback_ms / ds_ms)))
    B, T, dz = z.shape

    # 标签：past-window 内 u_glob 的最大值
    ug = u_g[..., 0]                                    # (B,T)
    lab = np.zeros_like(ug)
    for b in range(B):
        c = np.cumsum(ug[b])
        c = np.concatenate([[0.0], c])
        idx = np.arange(T)
        lo = np.maximum(idx - w, 0)
        lab[b] = c[idx + 1] - c[lo]                     # 事件计数
    y = (lab > 0).astype(np.int64)

    # 训练/测试按 trial 切分（避免同一轨迹泄漏）
    n_tr = B // 2
    Xtr = z[:n_tr].reshape(-1, dz)
    ytr = y[:n_tr].reshape(-1)
    Xte = z[n_tr:].reshape(-1, dz)
    yte = y[n_tr:].reshape(-1)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    # 逻辑回归（torch，L2）
    lin = torch.nn.Linear(dz, 2)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=200,
                            line_search_fn="strong_wolfe")
    xt = torch.from_numpy(Xtr.astype(np.float32))
    yt = torch.from_numpy(ytr)
    npos = max(int(ytr.sum()), 1)
    wgt = torch.tensor([1.0, float(len(ytr) - npos) / npos], dtype=torch.float32)

    def closure():
        opt.zero_grad()
        lo = F.cross_entropy(lin(xt), yt, weight=wgt)
        lo = lo + 1e-3 * (lin.weight ** 2).sum()
        lo.backward()
        return lo
    opt.step(closure)

    with torch.no_grad():
        prob = torch.softmax(lin(torch.from_numpy(
            Xte.astype(np.float32))), dim=1)[:, 1].numpy()
    auc = auc_score(yte, prob)

    # 基线：单帧 u 本身能解码多少（对照，证明增益来自 z）
    ug_te = ug[n_tr:].reshape(-1)
    auc_u = auc_score(yte, ug_te)

    print("\n=== D-b：latent 是否记住了 bAP ===")
    print(f"  窗口 {n_win} × {L} 步 = {n_win * L} 个样本；"
          f"标签 = 过去 {lookback_ms:.0f} ms 内有 bAP")
    print(f"  正样本比例（训练半）  {float(ytr.mean()):.3f}")
    print(f"  AUC(latent z)   = {auc:.4f}")
    print(f"  AUC(单帧 u_bap) = {auc_u:.4f}   ← 对照（瞬时指示，无记忆）")
    print(f"  判据 AUC > 0.80  ⇒ "
          f"{'✅ latent 承载历史依赖' if auc > 0.80 else '❌ 未达门禁'}")
    return dict(auc_latent=auc, auc_inst=auc_u,
                pos_rate=float(ytr.mean()), lookback_ms=lookback_ms)


def d_a(model, ds, scaler, horizons=(10, 40, 200, 1000, 4000), n_seq=64,
        t0=4000):
    """D-a：9 因子分三组报误差（模型 vs L0）。"""
    L = max(horizons) + 1
    sp = D2.Sampler2(ds, scaler, batch=n_seq, length=1, seed=0)
    s0, u_tok, u_g, th, s_true, x = sp.batch_window(t0, L, n_seq)
    u = torch.from_numpy(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2)))
    s0t = torch.from_numpy(s0)
    with torch.no_grad():
        outs = model.rollout(s0t, u, torch.from_numpy(th),
                             torch.from_numpy(x),
                             ug_seq=torch.from_numpy(u_g)).numpy()
    st = s_true

    groups = {"快门控": FAST_IDX, "cai": CAI_IDX, "V": V_IDX}
    print("\n=== D-a：分因子误差（归一化空间）===")
    print(f"{'group':>8} " + " ".join(f"{'k='+str(k):>12}" for k in horizons)
          + f" {'模型/L0(k=max)':>15}")
    out = {}
    for gname, gidx in groups.items():
        row = []
        for k in horizons:
            e = np.abs(outs[:, k - 1][:, gidx] - st[:, k][:, gidx]).mean()
            row.append(float(e))
        e0 = np.abs(s0[:, gidx] - st[:, horizons[-1]][:, gidx]).mean()
        ratio = row[-1] / max(float(e0), 1e-12)
        print(f"{gname:>8} " + " ".join(f"{v:>12.4f}" for v in row)
              + f" {ratio:>15.3f}")
        out[gname] = dict(err=row, ratio_last=ratio)
    fast = out["快门控"]["err"][-1]
    cai = out["cai"]["err"][-1]
    print(f"\n  cai / 快门控（k={horizons[-1]}）= "
          f"{cai / max(fast, 1e-12):.2f}×")
    print("  决策规则：若 ≥2× 且长视界不收敛 ⇒ 才引入因子分组（否则保持单一 latent）")
    return out


# ==============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools.diag_latent")
    ap.add_argument("--ckpt", type=str, default="t2v5_events/last.pt")
    ap.add_argument("--data", type=str, default="te_s3")
    ap.add_argument("--tier", type=str, default="t2", choices=["t1", "t2"])
    ap.add_argument("--n-win", type=int, default=256)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--lookback-ms", type=float, default=20.0)
    ap.add_argument("--which", type=str, default="ab",
                    help="要跑哪些诊断：a / b / ab")
    ap.add_argument("--mem-mb", type=float, default=2200.0)
    a = ap.parse_args(argv)

    safety.preflight(limit_mb=a.mem_mb)
    ds = D2.Dataset2(DATA, a.data, tier=a.tier)
    ck = torch.load(os.path.join(RES, a.ckpt), map_location="cpu",
                    weights_only=False)
    cfg = ck.get("cfg") or {}
    scaler = ck.get("scaler") or json.load(open(SCALER, encoding="utf-8"))
    m = M.build(d_z=int(cfg.get("d_z", 16)),
                hidden=int(cfg.get("hidden", 192)),
                n_layers=int(cfg.get("n_layers", 2)),
                n_factors=D2.N_FACTORS, u_dim=D2.U_FEAT,
                th_dim=int(ds.th.shape[1]),
                pred_attn=bool(cfg.get("pred_attn", False)),
                n_heads=int(cfg.get("n_heads", 4)),
                u_in_x=False, n_glob=D2.N_GLOB)
    m.load_state_dict(ck["model"])
    m.eval()
    log(f"载入 {a.ckpt}  {ds}")

    res = dict(ckpt=a.ckpt, tier=a.tier, data=a.data)
    if "a" in a.which:
        res["d_a"] = d_a(m, ds, scaler)
    if "b" in a.which:
        res["d_b"] = d_b(m, ds, scaler, n_win=a.n_win, L=a.length,
                         lookback_ms=a.lookback_ms)

    outp = os.path.join(ROOT, "results", "reports",
                        f"diag_{os.path.basename(a.ckpt).replace('.pt', '')}"
                        f"_{a.tier}.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n⇒ {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

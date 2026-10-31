"""I1：世界模型 rollout + jump 辅助校正（保持世界模型为核心）。

在每个检查点 k ∈ {512,2048,8000}，用 jump 模型（只用 t0 状态 + [t0,t0+k] 的输入，因果）
给出粗预测，按 s ← (1−λ)·s_roll + λ·s_jump 混合，再继续 rollout。

用法：
  env -u DISPLAY python -m tools.hybrid_rollout --model r6_c_1s_oldsc/last.pt \
      --lambda 0,0.25,0.5,1.0 --k 8000 --n 128
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from world.learn import data2 as D2
from tools.plot_v4 import load_model
from tools.jump_probe import u_summary


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_feats(ds, model, sc, ids, t0, delta):
    """单 trial 的 jump 特征：s99（按 scale_s）+ u 统计 + θ。"""
    ss = np.asarray(sc["scale_s"], np.float32)[:, None]
    probe = D2.Sampler2(ds, sc, batch=len(ids), length=1, seed=0)
    # 用 sampler 取 t0 的 s0 与 u token（只为 z 一致性；这里用原始 s）
    out = []
    for i in ids:
        st0 = np.asarray(ds.s[i, t0])
        uw = np.asarray(ds.u[i, t0:t0 + delta])
        out.append(np.concatenate([(st0 / ss).reshape(-1), u_summary(uw),
                                   np.asarray(ds.th[i])]))
    return np.asarray(out, np.float32)


def train_jump(tr, model, sc, deltas, t0s, seeds=1):
    """在 tr 上训练每个 Δ 的 jump 模型（s99 特征）。"""
    ss = np.asarray(sc["scale_s"], np.float32)[:, None]
    X, Y, D = [], [], []
    for t0 in t0s:
        for d in deltas:
            if t0 + d >= tr.s.shape[1] - 1:
                continue
            ids = np.arange(min(160, tr.N))
            X.append(make_feats(tr, model, sc, ids, t0, d))
            st1 = np.asarray(tr.s[ids, t0 + d])
            st0 = np.asarray(tr.s[ids, t0])
            Y.append(((st1 - st0) / ss).reshape(len(ids), -1))
            D += [d] * len(ids)
    X = np.concatenate(X); Y = np.concatenate(Y); D = np.asarray(D)
    out = {}
    for d in deltas:
        sel = D == d
        if sel.sum() < 10:
            continue
        Xd, Yd = X[sel], Y[sel]
        mu, sd = Xd.mean(0), Xd.std(0) + 1e-6
        torch.manual_seed(1)
        net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(),
                            nn.Linear(256, 256), nn.SiLU(),
                            nn.Linear(256, Y.shape[1]))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt = torch.from_numpy(((Xd - mu) / sd)).float()
        Yt = torch.from_numpy(Yd).float()
        for ep in range(300):
            perm = torch.randperm(len(Xt))
            for b in range(0, len(Xt), 128):
                idx = perm[b:b + 128]
                loss = F.mse_loss(net(Xt[idx]), Yt[idx])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        out[d] = (net, mu, sd)
        log(f"jump Δ={d} 训练完成 n={len(Xd)}")
    return out


def jump_pred(jump, ds, sc, i, t0, d):
    net, mu, sd = jump[d]
    x = make_feats(ds, None, sc, [i], t0, d)
    with torch.no_grad():
        dy = net(torch.from_numpy((x - mu) / sd).float()).numpy().reshape(9, 11)
    return np.asarray(ds.s[i, t0]) + dy * np.asarray(sc["scale_s"], np.float32)[:, None]


@torch.no_grad()
def rollout_batch(model, ds, sc, ids, t0, K, jump, lam, checkpoints):
    """批量 rollout（B=len(ids)），含检查点校正。返回 (B,9,11) 归一化状态。"""
    ids = list(ids)
    probe = D2.Sampler2(ds, sc, batch=len(ids), length=K, seed=0)
    s0, u_tok, ug, th, st, x = probe.batch_window(t0, K, len(ids), off=ids[0])
    s0t = torch.from_numpy(np.ascontiguousarray(s0)).float()
    ut = torch.from_numpy(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2))).float()
    ugt = torch.from_numpy(np.ascontiguousarray(ug)).float()
    tht = torch.from_numpy(th).float()
    xt = torch.from_numpy(x).float()
    z = model.encode(s0t.unsqueeze(1), ut[:, :1], tht, xt, ugt[:, :1]).squeeze(1)
    s = s0t.clone()
    for j in range(K):
        jj = min(j + 1, K - 1)
        z_next = model.predict(z.unsqueeze(1), ut[:, jj:jj + 1], tht, xt,
                               ugt[:, jj:jj + 1]).squeeze(1)
        s = model.step_residual(s, z, z_next, tht, xt, ugt[:, jj:jj + 1])
        z = z_next
        if (j + 1) in checkpoints and lam > 0 and (j + 1) in jump:
            xj = make_feats(ds, model, sc, ids, t0, j + 1)      # raw 特征
            net, mu, sd = jump[j + 1]
            with torch.no_grad():
                dy = net(torch.from_numpy((xj - mu) / sd).float()).numpy().reshape(len(ids), 9, 11)
            st0 = np.asarray(ds.s[ids, t0])
            s_j = st0 + dy * np.asarray(sc["scale_s"], np.float32)[:, None]
            _loc = np.asarray(sc["loc_s"], np.float32)[:, None]
            _sc = np.asarray(sc["scale_s"], np.float32)[:, None]
            s_j = (s_j - _loc) / _sc
            s = (1.0 - lam) * s + lam * torch.from_numpy(s_j).float()
    return s.numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="r6_c_1s_oldsc/last.pt")
    ap.add_argument("--down", default="datasets/1s/tr_s1")
    ap.add_argument("--val", default="datasets/1s/va_s2")
    ap.add_argument("--t0", type=int, default=4000)
    ap.add_argument("--k", type=int, default=8000)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--lams", default="0,0.25,0.5,1.0")
    ap.add_argument("--checkpoints", default="512,2048,8000")
    ap.add_argument("--out", default="runs/hybrid.json")
    a = ap.parse_args(argv)

    tr = D2.Dataset2("data", a.down, tier="t2")
    va = D2.Dataset2("data", a.val, tier="t2")
    model, sc, cfg = load_model(a.model, va, "t2")
    cps = tuple(int(x) for x in a.checkpoints.split(","))
    jump = train_jump(tr, model, sc, list(cps), [4000, 8000, 12000])

    _loc = np.asarray(sc["loc_s"], np.float32)[:, None]
    _sc = np.asarray(sc["scale_s"], np.float32)[:, None]
    base = (np.asarray(va.s[np.arange(a.n), a.t0 + a.k]) - _loc) / _sc
    l0 = (np.asarray(va.s[np.arange(a.n), a.t0]) - _loc) / _sc
    den = float(np.mean((l0 - base) ** 2))
    rows = []
    for lam in [float(x) for x in a.lams.split(",")]:
        t = time.time()
        preds = []
        B = 16
        for st in range(0, a.n, B):
            ids = list(range(st, min(st + B, a.n)))
            preds.append(rollout_batch(model, va, sc, ids, a.t0, a.k, jump, lam, cps))
        preds = np.concatenate(preds, 0)
        num1 = float(np.mean(np.abs(preds - base)))
        num2 = float(np.mean((preds - base) ** 2))
        per1 = np.mean(np.abs(preds - base), axis=(1, 2))
        per10 = np.mean(np.abs(l0 - base), axis=(1, 2))
        per2 = np.mean((preds - base) ** 2, axis=(1, 2))
        per20 = np.mean((l0 - base) ** 2, axis=(1, 2))
        rows.append(dict(lam=lam,
                         l1_agg=num1 / float(np.mean(np.abs(l0 - base))),
                         l1_mean=float(np.mean(per1 / per10)),
                         l1_median=float(np.median(per1 / per10)),
                         mse_agg=num2 / den,
                         mse_mean=float(np.mean(per2 / per20)),
                         n=a.n, sec=round(time.time() - t, 1)))
        log(str(rows[-1]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    log(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

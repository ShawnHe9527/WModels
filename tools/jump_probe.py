"""路线 1 诊断：直接 Δ 预测（非因果上界）。

对每个 (trial, t0, Δ) 用 [t0, t0+Δ) 的输入统计量 + 初始状态，直接预测 s_{t0+Δ}，
不经过 rollout。目的是回答：

1. 长视野（Δ=8000）的状态是否**可从输入粗统计中学到**（非因果上界）；
2. 这是否依赖 16 维 latent，还是需要原始 99 维状态（回答“降维是否是瓶颈”）；
3. 与 L0（复制）和 L2（ridge）比，能好多少。

用法：
  env -u DISPLAY python -m tools.jump_probe --data datasets/1s/tr_s1 --val datasets/1s/va_s2 \
      --deltas 512,2048,8000 --t0 4000,8000,12000 --seeds 1,2,3
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


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _scales(sc, ds):
    ss = np.asarray(sc["scale_s"], dtype=np.float32)          # (9,)
    su = np.asarray(sc["scale_u"], dtype=np.float32)          # (Cu,Pu) maybe
    return ss, su


def u_summary(u_win: np.ndarray) -> np.ndarray:
    """u_win:(Δ,85) → (85*4,) = mean,std,max,last。"""
    return np.concatenate([u_win.mean(0), u_win.std(0), u_win.max(0), u_win[-1]])


def build(ds, model, sc, t0s, deltas, max_trials=None, seed=0):
    N, T = ds.N, ds.s.shape[1]
    ss, _ = _scales(sc, ds)
    s_scale = np.asarray(ss, dtype=np.float32)[:, None]     # (9,1)
    xz, xs, yy = [], [], []
    meta = []
    probe = D2.Sampler2(ds, sc, batch=16, length=1, seed=seed)
    n = N if max_trials is None else max_trials
    keep = np.arange(n)                      # batch_window 用 0..n-1
    for t0 in t0s:
        for d in deltas:
            t1 = t0 + d
            if t1 >= T - 1:
                continue
            # z at t0（用 sampler 保证 u token 的排布正确）
            s0, u_tok, ug, th, st, x = probe.batch_window(t0, 1, n, off=0)
            s0t = torch.from_numpy(np.ascontiguousarray(s0)).float()
            ut = torch.from_numpy(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2))).float()
            tht = torch.from_numpy(th).float()
            xt = torch.from_numpy(x).float()
            ugt = torch.from_numpy(ug).float()
            with torch.no_grad():
                z = model.encode(s0t.unsqueeze(1), ut[:, :1], tht, xt,
                                 ugt[:, :1]).squeeze(1).numpy()  # (B,16)
            for j, i in enumerate(keep):
                uw = np.asarray(ds.u[i, t0:t1])                    # (Δ,85)
                st0 = np.asarray(ds.s[i, t0])                       # (9,11)
                st1 = np.asarray(ds.s[i, t1])
                xz.append(np.concatenate([z[j], u_summary(uw), np.asarray(th[i])]))
                xs.append(np.concatenate([st0.reshape(-1) / np.repeat(s_scale.reshape(-1), 11),
                                          u_summary(uw), np.asarray(th[i])]))
                yy.append((st1 - st0) / s_scale)                    # 预测位移
                meta.append((i, t0, d, float(np.mean(st0))))
    return (np.asarray(xz, np.float32), np.asarray(xs, np.float32),
            np.asarray(yy, np.float32), meta)


def l0_ratio(y, base_disp, pred_disp):
    """model/L0：预测位移 vs 真实位移 的 MSE 比（L0 位移=0）。"""
    num = np.mean((pred_disp - y) ** 2, axis=(1, 2))
    den = np.mean(y ** 2, axis=(1, 2))
    return float(num.mean()), float(np.median(num / den))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/1s/tr_s1")
    ap.add_argument("--val", default="datasets/1s/va_s2")
    ap.add_argument("--t0", default="4000,8000,12000")
    ap.add_argument("--deltas", default="512,2048,8000")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default="runs/jump_probe.json")
    a = ap.parse_args(argv)

    tr = D2.Dataset2("data", a.data, tier="t2")
    va = D2.Dataset2("data", a.val, tier="t2")
    m, sc, cfg = load_model("t1v4_dz16/best.pt", tr, "t2")
    t0s = [int(x) for x in a.t0.split(",")]
    deltas = [int(x) for x in a.deltas.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]

    log(f"构建训练表 t0={t0s} Δ={deltas} ...")
    XZ, XS, Y, _ = build(tr, m, sc, t0s, deltas, max_trials=160, seed=0)
    log(f"训练样本 {XZ.shape}")
    z_val, s_val, y_val, meta_val = build(va, m, sc, t0s, deltas, max_trials=128, seed=1)
    log(f"验证样本 {z_val.shape}")

    # 标准化（用训练集统计）
    mu_z, sd_z = XZ.mean(0), XZ.std(0) + 1e-6
    mu_s, sd_s = XS.mean(0), XS.std(0) + 1e-6
    Yf = Y.reshape(len(Y), -1)
    yv = y_val.reshape(len(y_val), -1)

    def trainer(X, Xv, n_in, seed):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(n_in, 256), nn.SiLU(), nn.Linear(256, 256),
                            nn.SiLU(), nn.Linear(256, Yf.shape[1]))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt = torch.from_numpy(((X - X.mean(0)) / (X.std(0) + 1e-6))).float()
        Xvt = torch.from_numpy(((Xv - X.mean(0)) / (X.std(0) + 1e-6))).float()
        yt = torch.from_numpy(Yf).float()
        yvt = torch.from_numpy(yv).float()
        best, best_sd = 1e9, None
        for ep in range(400):
            perm = torch.randperm(len(Xt))
            for b in range(0, len(Xt), 128):
                idx = perm[b:b + 128]
                loss = F.mse_loss(net(Xt[idx]), yt[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                v = float(F.mse_loss(net(Xvt), yvt))
            if v < best:
                best, best_sd = v, {k: t.clone() for k, t in net.state_dict().items()}
        net.load_state_dict(best_sd)
        return net

    rows = []
    for tag, X, Xv in (("z16", XZ, z_val), ("s99", XS, s_val)):
        for seed in seeds:
            net = trainer(X, Xv, X.shape[1], seed)
            with torch.no_grad():
                pred = net(torch.from_numpy(((Xv - X.mean(0)) / (X.std(0) + 1e-6))).float()).numpy()
            for d in deltas:
                sel = np.array([k for k, mm in enumerate(meta_val) if mm[2] == d])
                num = np.mean((pred[sel] - yv[sel]) ** 2, axis=1)
                den = np.mean(yv[sel] ** 2, axis=1)
                rows.append(dict(feat=tag, seed=seed, delta=d, mean=float(num.mean()),
                                 median=float(np.median(num / den)), n=int(len(sel))))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    log(f"→ {a.out}")
    for r in rows:
        log(str(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

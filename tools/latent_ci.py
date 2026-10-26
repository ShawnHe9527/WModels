"""跨轮次的 latent 方向一致性（cos）与不确定度。

用途：判断「R1 的 cos=0.93 与 R2 的 0.41 是否真的是差别」，还是 12 个 trial
的抽样噪声。做法：对每个 run，在同一批 plateau trial 上算逐 trial 的 cos，
给出中位数 + bootstrap 95% 置信区间。

    PYTHONPATH=$PWD python -m tools.latent_ci --runs t2v5_ancC t2v5_events \
        t2v5_events_w5a2 --ks 500 1000 2000

输出：results/reports/latent_ci.json + 终端表格。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from world.core import safety                      # noqa: F401
from world.learn import data2 as D2
from tools.plot_v4 import load_model, regime_rollouts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESC = os.path.join(ROOT, "results", "checkpoints")
REPR = os.path.join(ROOT, "results", "reports")


def log(*a):
    print(*a, flush=True)


def cos_drift(m, ds, sc, trial: int, t0: int, L: int, ks):
    """单个 trial 的闭环 cos 轨迹。"""
    T = lambda z: torch.from_numpy(np.ascontiguousarray(z)).float()  # noqa: E731
    sp = D2.Sampler2(ds, sc, batch=1, length=1, seed=0)
    s0, u, ug, th, st, x = sp.batch_window(t0, L, 1, off=trial)
    uT = torch.from_numpy(np.ascontiguousarray(u.transpose(0, 1, 3, 2))).float()
    with torch.no_grad():
        zt = m.encode(T(st), uT, T(th), T(x), T(ug))
        z = m.encode(T(s0).unsqueeze(1), uT[:, :1], T(th), T(x),
                     T(ug)[:, :1]).squeeze(1)
        zp = []
        for t in range(L - 1):
            j = t + 1
            z = m.predict(z.unsqueeze(1), uT[:, j].unsqueeze(1), T(th), T(x),
                          T(ug)[:, j].unsqueeze(1)).squeeze(1)
            zp.append(z.numpy()[0])
    ZP = np.stack(zp)
    ZT = zt[0].numpy()[1:]
    out = {}
    for k in ks:
        if k > len(ZP):
            continue
        a, b = ZP[k - 1], ZT[k]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        out[k] = dict(cos=float(a @ b / (na * nb + 1e-12)),
                      rz=float(na / max(nb, 1e-12)))
    return out


def boot_ci(v, n=4000, seed=0):
    """中位数 + bootstrap 95% CI（小样本专用）。"""
    v = np.asarray(v, dtype=np.float64)
    if len(v) < 3:
        return float(np.median(v)), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    med = np.median(rng.choice(v, (n, len(v)), replace=True), axis=1)
    return (float(np.median(v)), float(np.percentile(med, 2.5)),
            float(np.percentile(med, 97.5)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data", default="te_s3")
    p.add_argument("--tier", default="t2")
    p.add_argument("--ckpt", default="last.pt")
    p.add_argument("--ks", nargs="+", type=int, default=[500, 1000, 2000])
    p.add_argument("--t0", type=int, default=4000)
    p.add_argument("--n-trials", type=int, default=12)
    a = p.parse_args()

    ds = D2.Dataset2(DATA, a.data, tier=a.tier)
    L = max(a.ks) + 2
    out = {}
    for run in a.runs:
        ck = os.path.join(run, a.ckpt)
        if not os.path.exists(os.path.join(RESC, ck)):
            log(f"跳过 {run}: 找不到 {ck}")
            continue
        m, sc, _ = load_model(ck, ds, a.tier)
        # 与 round_report 用同一批 plateau trial，保证可比
        ids, *_ = regime_rollouts(m, ds, sc, "plateau", L + 1, a.t0,
                                  n=a.n_trials)
        per = {k: [] for k in a.ks}
        rzs = {k: [] for k in a.ks}
        for t in ids:
            r = cos_drift(m, ds, sc, int(t), a.t0, L, a.ks)
            for k in a.ks:
                if k in r:
                    per[k].append(r[k]["cos"])
                    rzs[k].append(r[k]["rz"])
        out[run] = {}
        log(f"\n=== {run}  (plateau n={len(ids)}) ===")
        log(f"{'k':>6}{'cos中位':>10}{'95%CI':>20}{'|z|比中位':>12}{'cos<0.5占比':>13}")
        for k in a.ks:
            c, lo, hi = boot_ci(per[k])
            rz, rzlo, rzhi = boot_ci(rzs[k])
            frac = float(np.mean(np.asarray(per[k]) < 0.5))
            out[run][str(k)] = dict(cos_median=c, cos_lo=lo, cos_hi=hi,
                                    cos_all=per[k], rz_median=rz,
                                    rz_lo=rzlo, rz_hi=rzhi, frac_below05=frac)
            log(f"{k:6d}{c:10.3f}{f'[{lo:+.3f},{hi:+.3f}]':>20}"
                f"{rz:12.3f}{frac:13.2f}")

    os.makedirs(REPR, exist_ok=True)
    pth = os.path.join(REPR, "latent_ci.json")
    with open(pth, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"\n=> {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""方案 1：线性探针 z_t → s_t —— 判断 latent 是否携带慢状态。

对每个模型：
  1. 在固定时间点（60 个）采样所有 trial 的 s_t，过 encoder 得 z_t；
  2. 按 trial 划分 train/test（默认 96/32），对每个因子拟合 ridge；
  3. 报 test R²（相对 train 均值基线），以及 99 维整体的 R²；
  4. 附带 shuffled-z 控制（应为 ~0）。

用法：
    cd /home/Hesj/WModel && export PYTHONPATH=$PWD
    env -u DISPLAY $PY -m tools.probe_latent \
        --data datasets/2s/va_s2_2s --n-trials 128 --n-times 60 \
        --models R2=t2v5_events/last.pt CA=r6_ca_2s_l512_rerun/last.pt \
        --out runs/<run>/probe_latent.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world.core import safety  # noqa: E402

safety.apply_thread_caps(4)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from world.learn import data2 as D2  # noqa: E402
from tools.plot_v4 import load_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def _to_t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).float()


@torch.no_grad()
def collect(m, ds, sc, t0s, n_trials, batch):
    Z, Y, ids = [], [], []
    for t0 in t0s:
        for off in range(0, n_trials, batch):
            b = min(batch, n_trials - off)
            probe = D2.Sampler2(ds, sc, batch=b, length=1, seed=0)
            s0, u_tok, ug, th, st, x = probe.batch_window(t0, 1, b, off=off)
            u = _to_t(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2)))
            z = m.encode(_to_t(s0).unsqueeze(1), u[:, :1], _to_t(th), _to_t(x),
                         _to_t(ug)[:, :1]).squeeze(1)
            Z.append(z.cpu().numpy())
            Y.append(np.asarray(s0))                 # 归一化空间
            ids.append((np.arange(b) + off) % ds.N)
    return (np.concatenate(Z), np.concatenate(Y), np.concatenate(ids))


def ridge_r2(Ztr, Ytr, Zte, Yte, lam=1e-4):
    """逐列 ridge，返回每列 test R²。"""
    Xtr = np.c_[np.ones(len(Ztr)), Ztr]
    Xte = np.c_[np.ones(len(Zte)), Zte]
    A = Xtr.T @ Xtr
    A += lam * np.eye(A.shape[0])
    A[0, 0] -= lam
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    pred = Xte @ W
    ss_res = ((Yte - pred) ** 2).sum(axis=0)
    ss_tot = ((Yte - Ytr.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def run(m, ds, sc, t0s, n_trials, batch, split):
    Z, Y, ids = collect(m, ds, sc, t0s, n_trials, batch)
    tr = ids < split
    te = ~tr
    names = ds.meta["factor_names"]
    Yf = Y.mean(axis=2)                       # (n,9) 按 compartment 平均
    Yflat = Y.reshape(len(Y), -1)             # (n,99)

    r2_f = ridge_r2(Z[tr], Yf[tr], Z[te], Yf[te])
    r2_flat = ridge_r2(Z[tr], Yflat[tr], Z[te], Yflat[te])

    rng = np.random.default_rng(0)
    Zsh = Z[rng.permutation(len(Z))]
    r2_sh = ridge_r2(Zsh[tr], Yf[tr], Zsh[te], Yf[te])

    return dict(
        n_frames=int(len(Z)),
        n_train=int(tr.sum()), n_test=int(te.sum()),
        factor_r2={names[i]: float(r2_f[i]) for i in range(len(names))},
        flat_r2_mean=float(np.mean(r2_flat)),
        flat_r2_p10=float(np.percentile(r2_flat, 10)),
        shuffled_r2_mean=float(np.mean(r2_sh)),
        z_abs_mean=float(np.linalg.norm(Z, axis=1).mean()),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tools.probe_latent")
    ap.add_argument("--data", default="datasets/2s/va_s2_2s")
    ap.add_argument("--tier", default="t2")
    ap.add_argument("--n-trials", type=int, default=128)
    ap.add_argument("--n-times", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--split", type=int, default=96,
                    help="trial id < split 作 train，其余 test")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    ds = D2.Dataset2(DATA, a.data, tier=a.tier)
    n = min(a.n_trials, ds.N)
    t0s = np.linspace(2000, ds.T - 2000, a.n_times).astype(int)
    print(f"data={a.data} n={n} times={len(t0s)} split={a.split}", flush=True)

    out = dict(config=dict(data=a.data, n_trials=n, n_times=int(len(t0s)),
                           batch=a.batch, split_trial=a.split),
               models={})
    for spec in a.models:
        name, ckpt = spec.split("=", 1)
        print(f"=== {name}: {ckpt} ===", flush=True)
        m, sc, _cfg = load_model(ckpt, ds, a.tier)
        out["models"][name] = run(m, ds, sc, t0s, n, a.batch, a.split)
        del m

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"=> {a.out}\n", flush=True)

    names = ds.meta["factor_names"]
    print("model   " + "".join(f"{x:>11}" for x in names) +
          "   flat_mean  shuf", flush=True)
    for name, r in out["models"].items():
        print(f"{name:<7} " + "".join(f"{r['factor_r2'][x]:>11.3f}" for x in names) +
              f"   {r['flat_r2_mean']:>8.3f}  {r['shuffled_r2_mean']:>5.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

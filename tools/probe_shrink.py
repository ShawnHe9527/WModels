"""E0：逐因子 delta 收缩（推理侧，不训练）。

闭环增量按 α 收缩：  ŝ_t = s_{t-1} + α_f · (pred_t − s_{t-1})

目的：验证“慢因子漂移”假设——若把慢门控的增量按住（α<1）能在长视界改善
model/L0，则训练侧加 delta 收缩/慢因子权重就有明确目标强度；否则漂移不是
简单偏置。

用法：
    cd /home/Hesj/WModel && export PYTHONPATH=$PWD
    env -u DISPLAY $PY -m tools.probe_shrink \
        --data datasets/2s/va_s2_2s --t0 4000 --n-trials 128 \
        --ckpt t2v5_events/last.pt --out runs/<run>/shrink.json
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
def rollout_shrunk(m, s0t, u, tht, xt, ugt, alpha, L):
    z = m.encode(s0t.unsqueeze(1), u[:, :1], tht, xt,
                 ugt[:, :1]).squeeze(1)
    s_hat = s0t.clone()
    outs = []
    for t in range(L):
        j = min(t + 1, L - 1)
        z_next = m.predict(z.unsqueeze(1), u[:, j].unsqueeze(1), tht, xt,
                           ugt[:, j].unsqueeze(1)).squeeze(1)
        s_prev = s_hat
        s_hat = m.step_residual(s_prev, z, z_next, tht, xt)
        if alpha is not None:
            s_hat = s_prev + alpha * (s_hat - s_prev)
        z = z_next
        outs.append(s_hat)
    return torch.stack(outs, dim=1)


def evaluate(m, ds, sc, ks, t0, n_tr, batch, alpha, names, model_name):
    kmax = max(ks)
    L = kmax + 1
    acc = {k: dict(model=[], L0=[], regime=[], pf_m=[], pf_0=[]) for k in ks}
    n = min(n_tr, ds.N)
    for off in range(0, n, batch):
        b = min(batch, n - off)
        probe = D2.Sampler2(ds, sc, batch=b, length=1, seed=0)
        s0, u_tok, ug, th, st, x = probe.batch_window(t0, L, b, off=off)
        u = _to_t(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2)))
        s0t, tht, ugt, xt = _to_t(s0), _to_t(th), _to_t(ug), _to_t(x)
        ids = (np.arange(b) + off) % ds.N
        pred = rollout_shrunk(m, s0t, u, tht, xt, ugt, alpha, L).cpu().numpy()
        for k in ks:
            p = pred[:, k - 1]
            t_k = st[:, k]
            acc[k]["model"].append(np.abs(p - t_k).mean(axis=(1, 2)))
            acc[k]["L0"].append(np.abs(s0 - t_k).mean(axis=(1, 2)))
            acc[k]["regime"].append(ds.regimes[ids])
            acc[k]["pf_m"].append(np.abs(p - t_k).mean(axis=2))
            acc[k]["pf_0"].append(np.abs(s0 - t_k).mean(axis=2))
    out = {}
    for k in ks:
        model = np.concatenate(acc[k]["model"])
        L0 = np.concatenate(acc[k]["L0"])
        reg = np.concatenate(acc[k]["regime"])
        pf_m = np.concatenate(acc[k]["pf_m"]).mean(axis=0)
        pf_0 = np.concatenate(acc[k]["pf_0"]).mean(axis=0)
        by = {}
        for r in ("linear", "inter", "plateau"):
            msk = reg == r
            by[r] = dict(n=int(msk.sum()),
                         ratio=float(model[msk].mean() / L0[msk].mean())
                         if msk.any() else None)
        out[k] = dict(ratio=float(model.mean() / L0.mean()),
                      by_regime=by,
                      per_factor={names[i]: float(pf_m[i] / max(pf_0[i], 1e-12))
                                  for i in range(len(names))})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tools.probe_shrink")
    ap.add_argument("--data", default="datasets/2s/va_s2_2s")
    ap.add_argument("--tier", default="t2")
    ap.add_argument("--ckpt", default="t2v5_events/last.pt")
    ap.add_argument("--t0", type=int, default=4000)
    ap.add_argument("--n-trials", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ks", default="200,2000,4000,8000")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    ks = sorted(int(x) for x in a.ks.split(","))
    ds = D2.Dataset2(DATA, a.data, tier=a.tier)
    names = ds.meta["factor_names"]
    m, sc, _cfg = load_model(a.ckpt, ds, a.tier)
    print(f"data={a.data} n={min(a.n_trials, ds.N)} t0={a.t0} ks={ks}",
          flush=True)

    idx = {nm: i for i, nm in enumerate(names)}
    slow3 = [idx["Ih.m"], idx["Ca_LVAst.h"], idx["Ca_HVA.h"]]

    def alpha_vec(overrides):
        v = torch.ones(1, len(names), 1)
        for i, val in overrides.items():
            v[0, i, 0] = val
        return v

    configs = [("baseline", None)]
    for al in (0.85, 0.70, 0.50):
        configs.append((f"slow3={al:.2f}", alpha_vec({i: al for i in slow3})))
    configs.append(("Ca_HVA.h=0.70",
                    alpha_vec({idx["Ca_HVA.h"]: 0.70})))
    configs.append(("slow3=0.00 (freeze)",
                    alpha_vec({i: 0.0 for i in slow3})))

    results = {}
    for cname, alpha in configs:
        print(f"=== {cname} ===", flush=True)
        results[cname] = evaluate(m, ds, sc, ks, a.t0, a.n_trials,
                                  a.batch, alpha, names, cname)
        r = results[cname]
        print("  " + "  ".join(
            f"k={k}:{r[k]['ratio']:.3f}" for k in ks), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(dict(config=dict(data=a.data, ckpt=a.ckpt, t0=a.t0,
                                   ks=ks, n_trials=min(a.n_trials, ds.N)),
                       configs=results), f, ensure_ascii=False, indent=1)
    print(f"\n=> {a.out}", flush=True)

    print(f"\n{'config':<22}" + "".join(f"{'k='+str(k):>10}" for k in ks)
          + "   lin@8k  pla@8k")
    for cname, r in results.items():
        kk = max(ks)
        print(f"{cname:<22}" + "".join(f"{r[k]['ratio']:>10.3f}" for k in ks)
              + f"   {r[kk]['by_regime']['linear']['ratio']:>6.3f}"
              + f"  {r[kk]['by_regime']['plateau']['ratio']:>6.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

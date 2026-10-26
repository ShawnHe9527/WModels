"""Evaluation gate: paired model/L0/L1/L2/L3 comparison on a validation split.

This is a measurement tool, not training. It deliberately writes only the
specified --out JSON and prints a table.

Usage:
    PYTHONPATH=$PWD python -m tools.eval_gate \
        --data va_s2 --t0 4000,16000 --ks 10,40,200,1000,2000 \
        --n-trials 128 --model R2=t2v5_events/last.pt \
        --out runs/.../gate.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.core import safety

safety.apply_thread_caps(4)
import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402

from world.learn import data2 as D2                  # noqa: E402
from tools.plot_v4 import load_model                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SCALER = os.path.join(ROOT, "results", "data", "scaler_v4.json")
SANE_MAX = 100.0


def log(*a):
    print(*a, flush=True)


def _to_t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).float()


def trial_means(ds, ids):
    """Per-trial mean over the full 40000-frame trajectory (L1, noncausal)."""
    out = np.empty((len(ids), ds.nf, ds.nseg), dtype=np.float64)
    for j, i in enumerate(ids):
        acc = np.zeros((ds.nf, ds.nseg), dtype=np.float64)
        for t in range(0, ds.T, 2000):
            acc += ds.s[i, t:t + 2000].sum(axis=0)
        out[j] = acc / ds.T
    return out


def fit_l2_l3(tr, scaler, ks, n_fit=3000, seed=0, lam=1.0):
    """Fit ridge (L2) and per-dim AR(1) (L3) on the training split.

    L2: [s_t(99), u_t(85), theta(27)] -> s_{t+k}(99), affine, ridge.
    L3: per-dim s_{t+k} = a + b*s_t.
    """
    ns, nu = D2.normalizers2(scaler)
    rng = np.random.default_rng(seed)
    Xs, Ys = {}, {}
    for k in ks:
        tmax = tr.T - int(k) - 1
        if tmax <= 1:
            raise ValueError(f"k={k} too large for T={tr.T}")
        p = rng.integers(0, tr.N, n_fit)
        t = rng.integers(1, tmax, n_fit)
        s = ns(np.asarray(tr.s[p, t]))                    # (n,9,11)
        u = nu(np.asarray(tr.u[p, t]))                    # (n,85)
        y = ns(np.asarray(tr.s[p, t + int(k)]))           # (n,9,11)
        X = np.concatenate([s.reshape(n_fit, -1), u,
                            np.asarray(tr.th[p])], axis=1)
        Xs[k] = X
        Ys[k] = y.reshape(n_fit, -1)

    L2 = {}
    for k in ks:
        X, Y = Xs[k], Ys[k]
        Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        n_feat = Xb.shape[1]
        A = Xb.T @ Xb
        A += lam * np.eye(n_feat)
        A[0, 0] -= lam                       # don't regularize intercept
        W = np.linalg.solve(A, Xb.T @ Y)     # (n_feat, 99)
        L2[k] = dict(W=W.astype(np.float64), lam=lam, n_fit=n_fit)

    L3 = {}
    for k in ks:
        X, Y = Xs[k][:, :tr.nf * tr.nseg], Ys[k]
        s = X.reshape(-1, tr.nf, tr.nseg)
        y = Y.reshape(-1, tr.nf, tr.nseg)
        s2 = s.reshape(-1, tr.nf * tr.nseg)
        y2 = y.reshape(-1, tr.nf * tr.nseg)
        mx = s2.mean(0)
        my = y2.mean(0)
        vx = s2.var(0)
        cov = ((s2 - mx) * (y2 - my)).mean(0)
        b = np.divide(cov, vx, out=np.zeros_like(cov), where=vx > 1e-12)
        a = my - b * mx
        L3[k] = dict(a=a, b=b, n_fit=n_fit)

    # keep feature construction for evaluation: s/u/th
    return L2, L3, ns, nu


def predict_l2_l3(k, s0, u0, th, L2, L3, ns, nu):
    # s0 is already normalized by Sampler2; only raw u needs normalization.
    s = s0
    u = nu(u0)
    X = np.concatenate([s.reshape(s.shape[0], -1), u, th], axis=1)
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    p2 = (Xb @ L2[k]["W"]).reshape(s.shape[0], s.shape[1], s.shape[2])
    p3 = (L3[k]["a"] + L3[k]["b"] * s.reshape(s.shape[0], -1)
          ).reshape(s.shape[0], s.shape[1], s.shape[2])
    return p2, p3


def errors(pred, true):
    """(B,F,P) -> per-sample mean abs; mark diverged as nan."""
    a = np.abs(pred - true).mean(axis=(1, 2))
    bad = (~np.isfinite(pred).all(axis=(1, 2))) | \
          (np.abs(pred).max(axis=(1, 2)) > SANE_MAX)
    a = np.where(bad, np.nan, a)
    return a, bad


def boot_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = rng.integers(0, len(x), (n_boot, len(x)))
    m = x[b].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def eval_model(m, ds, sc, ks, t0s, n_trials, batch, L1_by_id, L2, L3, ns, nu,
               fit_trials):
    """Return {k: {model,L0,L1,L2,L3, ...}} plus per-sample arrays."""
    out = {int(k): dict(model=[], L0=[], L1=[], L2=[], L3=[], regime=[],
                        bad=0) for k in ks}
    t_start = time.time()
    for t0 in t0s:
        for off in range(0, n_trials, batch):
            n = min(batch, n_trials - off)
            probe = D2.Sampler2(ds, sc, batch=n, length=1, seed=0)
            s0, u_tok, ug, th, st, x = probe.batch_window(t0, max(ks) + 1,
                                                          n, off=off)
            u = _to_t(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2)))
            s0t, tht, ugt, xt = _to_t(s0), _to_t(th), _to_t(ug), _to_t(x)
            with torch.no_grad():
                pred = m.rollout(s0t, u, tht, xt, ug_seq=ugt).cpu().numpy()
            ids = (np.arange(n) + off) % ds.N
            l1 = L1_by_id[ids]
            u0 = np.asarray(ds.u[ids, t0])
            th0 = np.asarray(ds.th[ids])
            for k in ks:
                k = int(k)
                p = pred[:, k - 1]
                em, bad = errors(p, st[:, k])
                e0, _ = errors(s0, st[:, k])
                e1, _ = errors(l1, st[:, k])
                p2, p3 = predict_l2_l3(k, s0, u0, th0, L2, L3, ns, nu)
                e2, _ = errors(p2, st[:, k])
                e3, _ = errors(p3, st[:, k])
                out[k]["model"].append(em)
                out[k]["L0"].append(e0)
                out[k]["L1"].append(e1)
                out[k]["L2"].append(e2)
                out[k]["L3"].append(e3)
                out[k]["regime"].append(ds.regimes[ids])
                out[k]["bad"] += int(np.asarray(bad).sum())
            log(f"  t0={t0} off={off} n={n} ({time.time()-t_start:.1f}s)")
    for k in out:
        for key in ("model", "L0", "L1", "L2", "L3", "regime"):
            out[k][key] = np.concatenate(out[k][key], axis=0)
    return out


def summarize(block, k):
    d = block[int(k)]
    res = {}
    for key in ("model", "L0", "L1", "L2", "L3"):
        res[key] = boot_ci(d[key])
    m = np.asarray(d["model"])
    l0 = np.asarray(d["L0"])
    valid = np.isfinite(m) & np.isfinite(l0)
    ratio = float(np.nanmean(m[valid]) / np.nanmean(l0[valid])) if valid.any() else float("nan")
    # bootstrap ratio
    idx = np.where(valid)[0]
    if len(idx):
        rng = np.random.default_rng(123)
        b = rng.integers(0, len(idx), (2000, len(idx)))
        rr = (m[idx][b].mean(1) / l0[idx][b].mean(1))
        rci = [float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))]
    else:
        rci = [float("nan"), float("nan")]
    res["ratio"] = ratio
    res["ratio_ci"] = rci
    res["diverged"] = int(np.asarray(d["model"]).size - np.isfinite(m).sum())
    res["n"] = int(m.size)
    # per-regime
    regs = np.asarray(d["regime"])
    res["regimes"] = {}
    for r in ("linear", "inter", "plateau"):
        msk = regs == r
        if not msk.any():
            res["regimes"][r] = dict(n=0)
            continue
        v = msk & valid
        res["regimes"][r] = dict(
            n=int(msk.sum()),
            model=float(np.nanmean(m[v])) if v.any() else float("nan"),
            L0=float(np.nanmean(l0[v])) if v.any() else float("nan"),
            ratio=float(np.nanmean(m[v]) / np.nanmean(l0[v])) if v.any() else float("nan"),
        )
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tools.eval_gate")
    ap.add_argument("--data", default="va_s2")
    ap.add_argument("--tier", default="t2")
    ap.add_argument("--t0", default="4000,16000")
    ap.add_argument("--ks", default="10,40,200,1000,2000")
    ap.add_argument("--n-trials", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--model", action="append", required=True,
                    help="name=ckpt path, can repeat")
    ap.add_argument("--fit-data", default="tr_s1")
    ap.add_argument("--fit-n", type=int, default=3000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    safety.preflight(limit_mb=3000)
    ds = D2.Dataset2(DATA, a.data, tier=a.tier)
    tr = D2.Dataset2(DATA, a.fit_data, tier=a.tier)
    ks = [int(x) for x in a.ks.split(",")]
    t0s = [int(x) for x in a.t0.split(",")]
    n_trials = min(a.n_trials, ds.N)
    log(f"data={a.data} n={n_trials} t0={t0s} ks={ks}")

    # baselines fitted once on the training split
    L2, L3, ns, nu = fit_l2_l3(tr, json.load(open(SCALER, encoding="utf-8")),
                               ks, n_fit=a.fit_n)
    ids = np.arange(ds.N)
    log("computing L1 trial means ...")
    L1_by_id = ns(trial_means(ds, ids))                  # (N,9,11)

    all_out = dict(config=dict(data=a.data, t0=t0s, ks=ks, n_trials=n_trials,
                               fit_data=a.fit_data, fit_n=a.fit_n,
                               batch=a.batch),
                   models={})
    for spec in a.model:
        name, ckpt = spec.split("=", 1)
        log(f"=== {name}: {ckpt} ===")
        m, sc, cfg = load_model(ckpt, ds, a.tier)
        ck = torch.load(os.path.join(ROOT, "results", "checkpoints", ckpt),
                        map_location="cpu", weights_only=False)
        block = eval_model(m, ds, sc, ks, t0s, n_trials, a.batch,
                           L1_by_id, L2, L3, ns, nu, a.fit_data)
        all_out["models"][name] = {
            "ckpt": ckpt,
            "sigma": float(ck.get("sigma", float("nan"))),
            "results": {str(k): summarize(block, k) for k in ks},
        }
        del m
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(all_out, f, ensure_ascii=False, indent=1)
    log(f"=> {a.out}")

    # compact table
    log("\nmodel            k     model/L0  95%CI              L1/L0  L2/L0  L3/L0  div")
    for name, md in all_out["models"].items():
        for k in ks:
            r = md["results"][str(k)]
            log(f"{name:15s} {k:5d}  {r['ratio']:.3f}   "
                f"[{r['ratio_ci'][0]:.3f},{r['ratio_ci'][1]:.3f}]   "
                f"{r['L1'][0]/r['L0'][0]:.3f}  {r['L2'][0]/r['L0'][0]:.3f}  "
                f"{r['L3'][0]/r['L0'][0]:.3f}  {r['diverged']}/{r['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

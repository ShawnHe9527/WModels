"""R5 zero-training calibration for the R2 checkpoint.

Fit a per-factor affine map on the validation split, then apply it to a held-out
split. Only `V` and `cai` are calibrated; other factors are untouched.

This is a diagnostic/measurement tool: it does not modify any checkpoint.

Usage:
    PYTHONPATH=$PWD python -m tools.calibrate_r2 \
        --fit-data va_s2 --eval-data te_s4 --t0 4000 \
        --n-fit 128 --n-eval 128 --batch 16 \
        --out runs/.../calibration.json
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
from tools.ca_event_metrics import ca_event_metrics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SCALER = os.path.join(ROOT, "results", "data", "scaler_v4.json")
CKPT = "t2v5_events/last.pt"
SANE_MAX = 100.0


def log(*a):
    print(*a, flush=True)


def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).float()


def collect(m, ds, sc, n_trials, t0, L, batch):
    """Run rollouts and collect k=2000/8000 states + plateau Ca/V series."""
    ci = ds.meta["factor_names"].index("cai")
    vi = ds.meta["factor_names"].index("V")
    ids, regs = [], []
    S0, P2, Y2, P8, Y8 = [], [], [], [], []
    C0, C1, V0, V1 = [], [], [], []      # plateau cai series / Vmax
    for off in range(0, n_trials, batch):
        n = min(batch, n_trials - off)
        probe = D2.Sampler2(ds, sc, batch=n, length=1, seed=0)
        s0, u_tok, ug, th, st, x = probe.batch_window(t0, L, n, off=off)
        u = _t(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2)))
        with torch.no_grad():
            pred = m.rollout(_t(s0), u, _t(th), _t(x), ug_seq=_t(ug)).numpy()
        ii = (np.arange(n) + off) % ds.N
        ids.append(ii)
        regs.append(ds.regimes[ii])
        S0.append(s0)
        P2.append(pred[:, 1999])
        Y2.append(st[:, 2000])
        P8.append(pred[:, 7999])
        Y8.append(st[:, 8000])
        pm = ds.regimes[ii] == "plateau"
        if pm.any():
            # pred[t] = ŝ_{t+1}; align true at t+1; skip final frame
            C0.append(pred[pm, :L - 1, ci, :])
            C1.append(st[pm, 1:, ci, :])
            V0.append(pred[pm, :, vi, :].max(axis=(1, 2)))
            V1.append(st[pm, 1:, vi, :].max(axis=(1, 2)))
        log(f"  t0={t0} off={off} n={n} plateau={int(pm.sum())}")
    out = dict(ids=np.concatenate(ids), regimes=np.concatenate(regs),
               s0=np.concatenate(S0), p2=np.concatenate(P2),
               y2=np.concatenate(Y2), p8=np.concatenate(P8),
               y8=np.concatenate(Y8), ci=ci, vi=vi)
    if C0:
        out.update(cai_pred=np.concatenate(C0), cai_true=np.concatenate(C1),
                   vmax_pred=np.concatenate(V0), vmax_true=np.concatenate(V1))
    else:
        out.update(cai_pred=np.empty((0, L - 1, ds.nseg)),
                   cai_true=np.empty((0, L - 1, ds.nseg)),
                   vmax_pred=np.empty(0), vmax_true=np.empty(0))
    return out


def fit_gain(block, sc, vi, ci):
    """Fit one amplitude gain per factor around a physically fixed baseline.

    Global affine calibration is too unstable: the resting V dominates the
    regression and a mean offset destroys the peak.  Instead, match the
    *event amplitude* around the resting level:

        x_cal = rest + alpha * (x - rest)

    alpha is the median per-trial ratio (true_max - rest)/(pred_max - rest)
    over plateau trials, clipped to [0.5, 1.5].
    """
    cal = {}
    # plateau series were already filtered in collect()

    # V: rest = -76 mV in physical units -> normalized coordinate
    rest_v = (-76.0 - float(sc["loc_s"][vi])) / float(sc["scale_s"][vi])
    x = np.asarray(block["vmax_pred"], dtype=np.float64)
    y = np.asarray(block["vmax_true"], dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x > rest_v + 1e-6)
    ratios = (y[ok] - rest_v) / (x[ok] - rest_v)
    a = float(np.median(ratios)) if len(ratios) else 1.0
    cal[vi] = (float(np.clip(a, 0.5, 1.5)), rest_v)

    # cai: use the true 20th percentile as the resting level (normalized)
    rest_c = float(np.percentile(block["cai_true"], 20)) if block["cai_pred"].size else 0.0
    xc = block["cai_pred"].max(axis=(1, 2))
    yc = block["cai_true"].max(axis=(1, 2))
    okc = np.isfinite(xc) & np.isfinite(yc) & (xc > rest_c + 1e-6)
    ratiosc = (yc[okc] - rest_c) / (xc[okc] - rest_c)
    ac = float(np.median(ratiosc)) if len(ratiosc) else 1.0
    cal[ci] = (float(np.clip(ac, 0.5, 1.5)), rest_c)
    return cal


def apply_cal(arr, cal):
    """arr: (n,F,P) normalized; apply rest-relative gain."""
    out = arr.copy()
    for f, (a, rest) in cal.items():
        out[:, f, :] = rest + a * (out[:, f, :] - rest)
    return out


def phys(x, sc, f):
    return x * float(sc["scale_s"][f]) + float(sc["loc_s"][f])


def err(pred, true):
    a = np.abs(pred - true).mean(axis=(1, 2))
    bad = (~np.isfinite(pred).all(axis=(1, 2))) | \
          (np.abs(pred).max(axis=(1, 2)) > SANE_MAX)
    return np.where(bad, np.nan, a), int(bad.sum())


def ratio_ci(m, l0, seed=0, n_boot=2000):
    valid = np.isfinite(m) & np.isfinite(l0)
    if not valid.any():
        return float("nan"), [float("nan"), float("nan")], int((~valid).sum())
    a, b = m[valid], l0[valid]
    rng = np.random.default_rng(seed)
    bb = rng.integers(0, len(a), (n_boot, len(a)))
    rr = a[bb].mean(1) / b[bb].mean(1)
    return float(a.mean() / b.mean()), [float(np.percentile(rr, 2.5)),
                                        float(np.percentile(rr, 97.5))], int((~valid).sum())


def ca_metrics(block, sc, ds, use_cal, cal):
    if len(block["cai_pred"]) == 0:
        return {}
    ci, vi = block["ci"], block["vi"]
    cp = block["cai_pred"].copy()
    if use_cal:
        a, rest = cal[ci]
        cp = rest + a * (cp - rest)
    P = phys(cp, sc, ci)[:, :, None, :]
    Y = phys(block["cai_true"], sc, ci)[:, :, None, :]
    return ca_event_metrics(P, Y, 0, ds.dt)


def summarize(block, sc, ds, cal=None):
    out = {}
    ci, vi = block["ci"], block["vi"]
    for k, p, y in ((2000, block["p2"], block["y2"]),
                    (8000, block["p8"], block["y8"])):
        e, bad = err(p, y)
        e0, _ = err(block["s0"], y)
        r, ci95, nbad = ratio_ci(e, e0)
        out[k] = dict(model=float(np.nanmean(e)), L0=float(np.nanmean(e0)),
                      ratio=r, ratio_ci=ci95, n=int(len(e)),
                      diverged=bad, n_bad_ratio=nbad)
        if cal is not None:
            pc = apply_cal(p, cal)
            ec, badc = err(pc, y)
            rc, ci95c, nb = ratio_ci(ec, e0)
            out[k]["calibrated"] = dict(model=float(np.nanmean(ec)),
                                        ratio=rc, ratio_ci=ci95c,
                                        diverged=badc, n_bad_ratio=nb)
    # physical V max on plateau
    if len(block["vmax_pred"]):
        out["V_max"] = dict(
            pred=float(phys(np.asarray(block["vmax_pred"]), sc, vi).max()),
            true=float(phys(np.asarray(block["vmax_true"]), sc, vi).max()),
            n=int(len(block["vmax_pred"])),
        )
        if cal is not None:
            a, rest = cal[vi]
            vmax_cal = rest + a * (np.asarray(block["vmax_pred"]) - rest)
            out["V_max"]["calibrated"] = float(phys(vmax_cal, sc, vi).max())
    # Ca event metrics
    out["ca_before"] = ca_metrics(block, sc, ds, False, {})
    if cal is not None:
        out["ca_after"] = ca_metrics(block, sc, ds, True, cal)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tools.calibrate_r2")
    ap.add_argument("--fit-data", default="va_s2")
    ap.add_argument("--eval-data", default="te_s4")
    ap.add_argument("--t0", type=int, default=4000)
    ap.add_argument("--n-fit", type=int, default=128)
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    safety.preflight(limit_mb=3000)
    ds_fit = D2.Dataset2(DATA, a.fit_data, tier="t2")
    ds_eval = D2.Dataset2(DATA, a.eval_data, tier="t2")
    m, sc, cfg = load_model(CKPT, ds_fit, "t2")
    tr_ci = ds_fit.meta["factor_names"].index("cai")
    vi = ds_fit.meta["factor_names"].index("V")
    L = 8001

    log(f"fit {a.fit_data}: n={min(a.n_fit, ds_fit.N)} t0={a.t0}")
    bf = collect(m, ds_fit, sc, min(a.n_fit, ds_fit.N), a.t0, L, a.batch)
    # Fit only on plateau trials (the regime with Ca/V amplitude extremes).
    cal = fit_gain(bf, sc, vi, tr_ci)
    log(f"calibration gain/rest: V={cal[vi]}, cai={cal[tr_ci]}")

    log(f"eval {a.eval_data}: n={min(a.n_eval, ds_eval.N)} t0={a.t0}")
    be = collect(m, ds_eval, sc, min(a.n_eval, ds_eval.N), a.t0, L, a.batch)
    res = dict(
        fit_data=a.fit_data, eval_data=a.eval_data, t0=a.t0, L=L,
        ckpt=CKPT, alpha_beta={str(k): list(v) for k, v in cal.items()},
        fit_plateau=int((bf["regimes"] == "plateau").sum()),
        before=summarize(be, sc, ds_eval, None),
        after=summarize(be, sc, ds_eval, cal),
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log(f"=> {a.out}")
    for k in (2000, 8000):
        b, c = res["before"][k], res["after"][k]["calibrated"]
        log(f"k={k}: before {b['ratio']:.3f} [{b['ratio_ci'][0]:.3f},"
            f"{b['ratio_ci'][1]:.3f}]  after {c['ratio']:.3f} "
            f"[{c['ratio_ci'][0]:.3f},{c['ratio_ci'][1]:.3f}]")
    log(f"Vmax: before {res['before']['V_max']['pred']:.2f} "
        f"after {res['after']['V_max']['calibrated']:.2f} "
        f"true {res['before']['V_max']['true']:.2f}")
    log(f"Ca F1: before {res['before']['ca_before'].get('peak_f1', float('nan')):.3f} "
        f"after {res['after']['ca_after'].get('peak_f1', float('nan')):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

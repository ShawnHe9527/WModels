"""B：条件校准（在训练集上拟合 per-factor 单调分段线性，推理时应用）。

用法：
  python -m tools.calib fit  --ckpt r6_fw2/last.pt --out runs/calib_fw2.json
  python -m tools.calib eval --ckpt r6_fw2/last.pt --calib runs/calib_fw2.json
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
import torch

from world.learn import data2 as D2
from tools.plot_v4 import load_model
from world.core import safety

def _t(x): return torch.from_numpy(np.ascontiguousarray(x)).float()

def fit(ck, out, n_win=3000):
    safety.preflight(limit_mb=6000)
    ds = D2.Dataset2("data", "datasets/1s/tr_s1", tier="t2")
    m, sc, cfg = load_model(ck, ds, "t2")
    s_tr = D2.Sampler2(ds, sc, batch=32, length=128, seed=0, p_trial=None,
                       hot_frac=0.5, event_center_frac=0.0, event_center_thr=0.02)
    acc_p, acc_t = [], []
    with torch.no_grad():
        for _ in range(max(1, n_win // 32)):
            s0, u_tok, ug, th, st, x = s_tr.batch_pair()
            s0t, stt = _t(s0), _t(st)
            u = _t(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2)))
            tht, xt, ugt = _t(th), _t(x), _t(ug)
            z = m.encode(stt, u, tht, xt, ugt)
            zp = torch.cat([z[:, :1], z[:, :-1]], 1)
            zn = m.predict(zp, u, tht, xt, ugt)
            sp = torch.cat([s0t.unsqueeze(1), stt[:, :-1]], 1)
            s_hat = sp + m.state_delta(zp, zn, tht, xt)
            acc_p.append(s_hat.numpy()); acc_t.append(stt.numpy())
    P = np.concatenate(acc_p, 0); T = np.concatenate(acc_t, 0)  # (N,L,9,11)
    names = ds.meta["factor_names"]
    cal = {}
    for f, n in enumerate(names):
        pf = P[:, :, f, :].ravel(); tf = T[:, :, f, :].ravel()
        # 5 分位分段线性（单调）
        qs = np.quantile(pf, [0.1, 0.3, 0.5, 0.7, 0.9])
        knots = np.unique(np.concatenate([[pf.min()], qs, [pf.max()]]))
        vals = []
        for i in range(len(knots)):
            lo = knots[i] - 1e-9 if i == 0 else knots[i]
            hi = knots[i + 1] + 1e-9 if i + 1 < len(knots) else knots[i] + 1e-9
            msk = (pf >= lo) & (pf < hi)
            vals.append(float(tf[msk].mean()) if msk.any() else float(knots[i]))
        cal[n] = dict(knots=knots.tolist(), vals=vals)
    json.dump(cal, open(out, "w"), indent=1)
    print("fitted", out)

def apply_cal(x, name, cal):
    c = cal[name]
    return np.interp(x, np.asarray(c["knots"]), np.asarray(c["vals"]))

@torch.no_grad()
def rollout(m, ds, sc, i, t0, K, cal=None):
    probe = D2.Sampler2(ds, sc, batch=1, length=K + 1, seed=0)
    s0, u_tok, ug, th, st, x = probe.batch_window(t0, K + 1, 1, off=i)
    s0t, tht, xt, ugt = _t(s0), _t(th), _t(x), _t(ug)
    u = _t(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2)))
    z = m.encode(s0t.unsqueeze(1), u[:, :1], tht, xt, ugt[:, :1])[:, 0]
    s = s0t.clone(); out = []
    for j in range(K):
        jj = min(j + 1, K - 1)
        zn = m.predict(z.unsqueeze(1), u[:, jj:jj + 1], tht, xt, ugt[:, jj:jj + 1])[:, 0]
        s = m.step_residual(s, z, zn, tht, xt, ugt[:, jj:jj + 1]); z = zn
        s_np = s.numpy()
        if cal is not None:
            for f, n in enumerate(ds.meta["factor_names"]):
                if n in cal:
                    s_np[0, f, :] = apply_cal(s_np[0, f, :], n, cal)
        out.append(s_np[0])
    return np.asarray(out)

def evaluate(ck, calpath, n_trials=64):
    ds = D2.Dataset2("data", "datasets/1s/va_s2", tier="t2")
    m, sc, cfg = load_model(ck, ds, "t2")
    loc = np.asarray(sc["loc_s"], np.float32)[:, None]; scl = np.asarray(sc["scale_s"], np.float32)[:, None]
    cal = json.load(open(calpath)) if calpath else None
    regs = np.asarray(ds.regimes); ids = np.where(regs == "plateau")[0][:n_trials]
    t0, K = 4000, 8000
    for tag, c in (("no-cal", None), ("cal", cal)):
        num = den = 0.0; v_mae = []; cai_mae = []
        for i in ids:
            pred = rollout(m, ds, sc, i, t0, K, c) * scl + loc
            true = np.asarray(ds.s[i, t0 + 1:t0 + 1 + K])
            num += np.abs(pred - true).mean(); den += np.abs(true[0] - true).mean()
            v_mae.append(np.abs(pred[-1] - true[-1])[np.asarray(ds.meta["factor_names"]).tolist().index("V")].mean())
            cai_mae.append(np.abs(pred[-1] - true[-1])[np.asarray(ds.meta["factor_names"]).tolist().index("cai")].mean())
        print(f"{tag:7s} ratio={num/den:.3f}  V@200ms={np.mean(v_mae):.2f}mV  cai={np.mean(cai_mae):.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "eval"])
    ap.add_argument("--ckpt", default="r6_fw2/last.pt")
    ap.add_argument("--calib", default="")
    ap.add_argument("--out", default="runs/calib_fw2.json")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a.ckpt, a.out)
    else:
        evaluate(a.ckpt, a.calib, a.n)

if __name__ == "__main__":
    main()

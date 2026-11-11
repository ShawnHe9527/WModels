"""D1–D3：plateau 误差分解 / mid-course oracle / 分档误差曲线。

D1：平台起始时刻、幅度、持续时间、平台内波动 的误差分解
D2：在 t0+2000 帧用真值校正，看 200/400 ms 误差降多少
D3：plateau 的 V 误差随 k 的曲线
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

from world.learn import data2 as D2
from tools.plot_v4 import load_model
from world.core import safety


def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).float()


@torch.no_grad()
def rollout(m, ds, sc, i, t0, K, corr_at=None, corr_state=None):
    probe = D2.Sampler2(ds, sc, batch=1, length=K + 1, seed=0)
    s0, u_tok, ug, th, st, x = probe.batch_window(t0, K + 1, 1, off=i)
    s0t, tht, xt, ugt = _t(s0), _t(th), _t(x), _t(ug)
    u = _t(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2)))
    with torch.no_grad():
        z = m.encode(s0t.unsqueeze(1), u[:, :1], tht, xt, ugt[:, :1])[:, 0]
    s = s0t.clone()
    out = []
    for j in range(K):
        jj = min(j + 1, K - 1)
        zn = m.predict(z.unsqueeze(1), u[:, jj:jj + 1], tht, xt,
                       ugt[:, jj:jj + 1]).squeeze(1)
        s = m.step_residual(s, z, zn, tht, xt, ugt[:, jj:jj + 1])
        z = zn
        if corr_at is not None and (j + 1) == corr_at:
            s = torch.from_numpy(np.ascontiguousarray(corr_state)).float().unsqueeze(0)
            z = m.encode(s.unsqueeze(1), u[:, jj:jj + 1], tht, xt, ugt[:, jj:jj + 1])[:, 0]
        out.append(s.numpy()[0])
    return np.asarray(out)  # (K,9,11) normalized


def plateau_stats(v, dt=0.025, thr=-55.0):
    """v:(T,) mV。返回 (onset_ms, duration_ms, amp_mV)。"""
    above = v > thr
    if not above.any():
        return np.nan, 0.0, np.nan
    idx = np.where(above)[0]
    onset = idx[0] * dt
    # 用最长连续段作为平台
    splits = np.where(np.diff(idx) > 1)[0]
    segs = np.split(idx, splits + 1)
    seg = max(segs, key=len)
    dur = (seg[-1] - seg[0]) * dt
    amp = v[seg].max() - v[0]
    return onset, dur, amp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="r6_fw2/last.pt")
    ap.add_argument("--data", default="datasets/1s/va_s2")
    ap.add_argument("--t0", type=int, default=4000)
    ap.add_argument("--ks", default="2000,4000,8000,16000")
    ap.add_argument("--n-trials", type=int, default=43)
    ap.add_argument("--out", default="runs/err_decomp.json")
    a = ap.parse_args(argv)
    safety.preflight(limit_mb=8000)

    ds = D2.Dataset2("data", a.data, tier="t2")
    m, sc, cfg = load_model(a.ckpt, ds, "t2")
    loc = np.asarray(sc["loc_s"], np.float32)[:, None]
    scl = np.asarray(sc["scale_s"], np.float32)[:, None]
    vi = ds.meta["factor_names"].index("V")
    regs = np.asarray(ds.regimes)
    ids = np.where(regs == "plateau")[0][:a.n_trials]
    K = max(int(v) for v in a.ks.split(","))

    rec = {k: [] for k in a.ks.split(",")}
    d1 = dict(onset_err=[], amp_err=[], dur_err=[], base_err=[])
    d2 = {k: [] for k in a.ks.split(",")}
    for n, i in enumerate(ids):
        t0 = a.t0
        pred = rollout(m, ds, sc, i, t0, K) * scl + loc
        true = np.asarray(ds.s[i, t0 + 1:t0 + 1 + K])
        for kk in rec:
            k = int(kk)
            rec[kk].append(float(np.abs(pred[k - 1] - true[k - 1]).mean()))
        # D1：平台统计（用 V 的最大 compartment）
        vt = true[:, vi, :].max(axis=1)
        vp = pred[:, vi, :].max(axis=1)
        ot, dt_, at = plateau_stats(vt)
        op, dp, ap_ = plateau_stats(vp)
        d1["onset_err"].append(abs(op - ot) if np.isfinite(ot) and np.isfinite(op) else np.nan)
        d1["dur_err"].append(abs(dp - dt_))
        d1["amp_err"].append(abs(ap_ - at) if np.isfinite(at) else np.nan)
        d1["base_err"].append(float(np.abs(vp[0] - vt[0])))
        # D2：mid-course 校正
        corr_k = 2000
        st_true = np.asarray(ds.s[i, t0 + 1 + corr_k])
        pred2 = rollout(m, ds, sc, i, t0, K, corr_at=corr_k, corr_state=st_true) * scl + loc
        for kk in d2:
            k = int(kk)
            if k > corr_k:
                d2[kk].append(float(np.abs(pred2[k - 1] - true[k - 1]).mean()))
        if (n + 1) % 10 == 0:
            print(f"{n+1}/{len(ids)}", flush=True)

    res = dict(
        d1=dict(onset_ms=float(np.nanmean(d1["onset_err"])),
                amp_mV=float(np.nanmean(d1["amp_err"])),
                dur_ms=float(np.nanmean(d1["dur_err"])),
                base_mV=float(np.nanmean(d1["base_err"]))),
        d3={k: float(np.mean(v)) for k, v in rec.items()},
        d2={k: float(np.mean(v)) for k, v in d2.items() if len(v)},
    )
    print(json.dumps(res, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

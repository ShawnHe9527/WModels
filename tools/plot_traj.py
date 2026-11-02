"""画出 A+B 模型在若干 trial 上的 V / cai 轨迹 vs 真值（含 L0 参考）。"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world.learn import data2 as D2
from tools.plot_v4 import load_model

DT = 0.025


@torch.no_grad()
def rollout(m, ds, sc, i, t0, K):
    probe = D2.Sampler2(ds, sc, batch=1, length=K, seed=0)
    s0, u_tok, ug, th, st, x = probe.batch_window(t0, K, 1, off=i)
    s0t = torch.from_numpy(np.ascontiguousarray(s0)).float()
    ut = torch.from_numpy(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2))).float()
    ugt = torch.from_numpy(np.ascontiguousarray(ug)).float()
    tht = torch.from_numpy(th).float()
    xt = torch.from_numpy(x).float()
    z = m.encode(s0t.unsqueeze(1), ut[:, :1], tht, xt, ugt[:, :1]).squeeze(1)
    s = s0t.clone()
    out = []
    for j in range(K):
        jj = min(j + 1, K - 1)
        zn = m.predict(z.unsqueeze(1), ut[:, jj:jj + 1], tht, xt,
                       ugt[:, jj:jj + 1]).squeeze(1)
        s = m.step_residual(s, z, zn, tht, xt, ugt[:, jj:jj + 1])
        z = zn
        out.append(s.squeeze(0).numpy())
    return np.asarray(out), s0.squeeze(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="r6_ab_s2/last.pt")
    ap.add_argument("--data", default="datasets/1s/va_s2")
    ap.add_argument("--t0", type=int, default=4000)
    ap.add_argument("--k", type=int, default=8000)
    ap.add_argument("--out", default="runs/traj.png")
    a = ap.parse_args(argv)

    ds = D2.Dataset2("data", a.data, tier="t2")
    m, sc, cfg = load_model(a.model, ds, "t2")
    loc = np.asarray(sc["loc_s"], np.float32)[:, None]
    scl = np.asarray(sc["scale_s"], np.float32)[:, None]
    regs = np.asarray(ds.regimes)
    picks = []
    for r in ("linear", "inter", "plateau"):
        idx = np.where(regs == r)[0]
        picks.append(int(idx[0]) if len(idx) else 0)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    t = np.arange(1, a.k + 1) * DT
    for c, (i, r) in enumerate(zip(picks, ("linear", "inter", "plateau"))):
        pred, s0 = rollout(m, ds, sc, i, a.t0, a.k)
        pred_raw = pred * scl + loc
        true = np.asarray(ds.s[i, a.t0 + 1:a.t0 + a.k + 1])
        l0 = np.repeat(np.asarray(ds.s[i, a.t0])[None], a.k, axis=0)
        v_pred = pred_raw[:, 8, :].mean(1)
        v_true = true[:, 8, :].mean(1)
        v_l0 = l0[:, 8, :].mean(1)
        c_pred = pred_raw[:, 7, :].max(1)
        c_true = true[:, 7, :].max(1)
        c_l0 = l0[:, 7, :].max(1)
        axes[0, c].plot(t, v_true, "k", lw=1.4, label="true")
        axes[0, c].plot(t, v_pred, "C0", lw=1.0, label=f"model ({a.model.split('/')[0]})")
        axes[0, c].plot(t, v_l0, "C7--", lw=0.8, label="L0 (persist)")
        axes[0, c].set_title(f"{r}  trial={i}")
        axes[0, c].set_ylabel("V (mV, mean)")
        # zoom 0-20 ms
        mz = t <= 20
        axes[1, c].plot(t[mz], v_true[mz], "k", lw=1.4)
        axes[1, c].plot(t[mz], v_pred[mz], "C0", lw=1.0)
        axes[1, c].set_ylabel("V zoom 0-20 ms")
        axes[1, c].set_xlabel("time (ms)")
        axes[2, c].plot(t, c_true, "k", lw=1.4)
        axes[2, c].plot(t, c_pred, "C0", lw=1.0)
        axes[2, c].plot(t, c_l0, "C7--", lw=0.8)
        axes[2, c].set_ylabel("log10(cai) max")
        axes[2, c].set_xlabel("time (ms)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Predicted vs true trajectories — {a.model}  (va_s2, t0={a.t0}, {a.k} steps = {a.k*DT:.0f} ms)",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("saved", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

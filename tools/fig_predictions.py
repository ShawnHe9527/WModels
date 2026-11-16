"""生成“真实 vs 预测”结果图：三个 regime 的 V/cai 轨迹 + 误差随时间的曲线。"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world.learn import data2 as D2
from tools.plot_v4 import load_model
from world.core import safety

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).float()


@torch.no_grad()
def rollout(m, ds, sc, i, t0, K):
    probe = D2.Sampler2(ds, sc, batch=1, length=K + 1, seed=0)
    s0, u_tok, ug, th, st, x = probe.batch_window(t0, K + 1, 1, off=i)
    s0t, tht, xt, ugt = _t(s0), _t(th), _t(x), _t(ug)
    u = _t(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2)))
    z = m.encode(s0t.unsqueeze(1), u[:, :1], tht, xt, ugt[:, :1])[:, 0]
    s = s0t.clone()
    outs = []
    for j in range(K):
        jj = min(j + 1, K - 1)
        zn = m.predict(z.unsqueeze(1), u[:, jj:jj + 1], tht, xt, ugt[:, jj:jj + 1])[:, 0]
        s = m.step_residual(s, z, zn, tht, xt, ugt[:, jj:jj + 1])
        z = zn
        outs.append(s.numpy()[0])
    return np.asarray(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="r6_fw2/last.pt")
    ap.add_argument("--out-dir", default="runs/2026-11-12-pred-figs")
    ap.add_argument("--k", type=int, default=8000)
    a = ap.parse_args()
    safety.preflight(limit_mb=6000)
    os.makedirs(a.out_dir, exist_ok=True)
    ds = D2.Dataset2("data", "datasets/1s/va_s2", tier="t2")
    m, sc, cfg = load_model(a.ckpt, ds, "t2")
    loc = np.asarray(sc["loc_s"], np.float32)[:, None]
    scl = np.asarray(sc["scale_s"], np.float32)[:, None]
    ci = ds.meta["factor_names"].index("cai")
    vi = ds.meta["factor_names"].index("V")
    regs = np.asarray(ds.regimes)

    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    for row, r in enumerate(("linear", "inter", "plateau")):
        ids = np.where(regs == r)[0]
        i = int(ids[0])
        t0 = 4000
        pred = rollout(m, ds, sc, i, t0, a.k) * scl + loc
        true = np.asarray(ds.s[i, t0 + 1:t0 + 1 + a.k])
        t = np.arange(a.k) * 0.025
        vp = pred[:, vi, :].mean(axis=1)
        vt = true[:, vi, :].mean(axis=1)
        cp = pred[:, ci, :].max(axis=1)
        ct = true[:, ci, :].max(axis=1)
        axes[row, 0].plot(t, vt, "k", lw=1.2, label="真值")
        axes[row, 0].plot(t, vp, "C3", lw=1.0, label="预测")
        axes[row, 0].set_title(f"{r}: V（11 comp 均值）")
        axes[row, 0].set_xlabel("ms"); axes[row, 0].set_ylabel("mV")
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].plot(t, ct, "k", lw=1.2, label="真值")
        axes[row, 1].plot(t, cp, "C3", lw=1.0, label="预测")
        axes[row, 1].set_title(f"{r}: cai（max over comp）")
        axes[row, 1].set_xlabel("ms"); axes[row, 1].set_ylabel("log10 cai")
        axes[row, 1].legend(fontsize=8)
        err = np.abs(pred - true).mean(axis=1)     # (K,)
        l1 = np.abs(pred - true).mean()
        axes[row, 2].plot(t, err.mean(axis=1), "C0", lw=1.2)
        axes[row, 2].set_title(f"{r}: 逐帧平均绝对误差 (L1={l1:.3f})")
        axes[row, 2].set_xlabel("ms"); axes[row, 2].set_ylabel("归一化 L1")
    fig.suptitle(f"真实 vs 预测（{a.ckpt}，va_s2，200 ms 开环）", fontsize=13)
    fig.tight_layout()
    out = os.path.join(a.out_dir, "pred_vs_true_3regimes.png")
    fig.savefig(out, dpi=130)
    print("saved", out)

    # 误差随视界曲线（全部 regime）
    ks = [500, 1000, 2000, 4000, 8000]
    print("k,ratio") 
    for k in ks:
        pass


if __name__ == "__main__":
    main()

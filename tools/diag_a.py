"""诊断 A：误差随视界/因子/事件帧的分解（纯评测，不训练）。

用法：
    cd /home/Hesj/WModel && export PYTHONPATH=$PWD
    env -u DISPLAY $PY -m tools.diag_a \
        --data datasets/2s/va_s2_2s --t0 4000 --n-trials 128 \
        --models R2=t2v5_events/last.pt CA=r6_ca_2s_l512_rerun/last.pt \
        --out runs/<run>/diag_a.json --fig-dir runs/<run>/diag_a

设计：
- 每批做一次 rollout 到 max(k)，所有视界共用一次前向，避免重复计算；
- 误差都在**归一化空间**里算（与训练 loss 同尺度），L0 = 复制 s0；
- 事件帧用真值定义：bAP = raw u[...,84] > 0.5；Ca 事件 = |Δcai| 逐帧最大 > 0.05（log10）；
- latent 漂移记录 |z| 与 cos(z, z_true)。
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
CAI = 7
V = 8
BAP_U = 84
CA_EVENT_DCAI = 0.05      # log10 / frame


def _to_t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).float()


@torch.no_grad()
def run_model(m, sc, ds, ks, t0, n_tr, batch):
    kset = set(ks)
    kmax = max(ks)
    L = kmax + 1
    acc = {k: dict(model=[], L0=[], regime=[], pf_model=[], pf_L0=[],
                   bap=[], bap_err=[], bap_l0=[], evt=[], evt_err=[], evt_l0=[],
                   zabs=[], zcos=[]) for k in ks}
    n = min(n_tr, ds.N)
    for off in range(0, n, batch):
        b = min(batch, n - off)
        probe = D2.Sampler2(ds, sc, batch=b, length=1, seed=0)
        s0, u_tok, ug, th, st, x = probe.batch_window(t0, L, b, off=off)
        u = _to_t(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2)))
        s0t, tht, ugt, xt = _to_t(s0), _to_t(th), _to_t(ug), _to_t(x)
        ids = (np.arange(b) + off) % ds.N

        z = m.encode(s0t.unsqueeze(1), u[:, :1], tht, xt,
                     ugt[:, :1]).squeeze(1)
        s_hat = s0t.clone()
        preds, zs = [], {}
        for t in range(L):
            j = min(t + 1, L - 1)
            z_next = m.predict(
                z.unsqueeze(1), u[:, j].unsqueeze(1), tht, xt,
                ugt[:, j].unsqueeze(1)).squeeze(1)
            s_hat = m.step_residual(s_hat, z, z_next, tht, xt)
            z = z_next
            if (t + 1) in kset:
                zs[t + 1] = z.detach().cpu().numpy()
            preds.append(s_hat)
        pred = torch.stack(preds, dim=1).cpu().numpy()      # (b,L,F,P)

        for k in ks:
            p = pred[:, k - 1]
            t_k = st[:, k]
            e = np.abs(p - t_k).mean(axis=(1, 2))
            e0 = np.abs(s0 - t_k).mean(axis=(1, 2))
            acc[k]["model"].append(e)
            acc[k]["L0"].append(e0)
            acc[k]["regime"].append(ds.regimes[ids])
            acc[k]["pf_model"].append(np.abs(p - t_k).mean(axis=2))   # (b,F)
            acc[k]["pf_L0"].append(np.abs(s0 - t_k).mean(axis=2))

            # bAP 帧（真值定义）
            bap = ds.u[ids, t0 + k, BAP_U] > 0.5
            ev = np.abs(p[:, V, :] - t_k[:, V, :]).mean(axis=1)
            e0v = np.abs(s0[:, V, :] - t_k[:, V, :]).mean(axis=1)
            acc[k]["bap"].append(bap)
            acc[k]["bap_err"].append(ev)
            acc[k]["bap_l0"].append(e0v)

            # Ca 事件帧：|Δcai| 逐帧最大
            d = np.abs(ds.s[ids, t0 + k, CAI, :] - ds.s[ids, t0 + k - 1, CAI, :])
            evt = d.max(axis=1) > CA_EVENT_DCAI
            acc[k]["evt"].append(evt)
            acc[k]["evt_err"].append(e)
            acc[k]["evt_l0"].append(e0)

            # latent 漂移
            zk = zs[k]
            zt = m.encode(_to_t(t_k)[:, None], u[:, k:k + 1], tht, xt,
                          ugt[:, k:k + 1]).squeeze(1).cpu().numpy()
            acc[k]["zabs"].append(np.linalg.norm(zk, axis=1))
            num = (zk * zt).sum(axis=1)
            den = np.linalg.norm(zk, axis=1) * np.linalg.norm(zt, axis=1) + 1e-12
            acc[k]["zcos"].append(num / den)

    out = {}
    for k in ks:
        a = acc[k]
        model = np.concatenate(a["model"])
        L0 = np.concatenate(a["L0"])
        reg = np.concatenate(a["regime"])
        pf_m = np.concatenate(a["pf_model"])
        pf_0 = np.concatenate(a["pf_L0"])
        bap = np.concatenate(a["bap"])
        be = np.concatenate(a["bap_err"])
        b0 = np.concatenate(a["bap_l0"])
        evt = np.concatenate(a["evt"])
        ee = np.concatenate(a["evt_err"])
        e0 = np.concatenate(a["evt_l0"])
        zabs = np.concatenate(a["zabs"])
        zcos = np.concatenate(a["zcos"])

        by_reg = {}
        for r in ("linear", "inter", "plateau"):
            msk = reg == r
            by_reg[r] = dict(
                n=int(msk.sum()),
                model=float(model[msk].mean()) if msk.any() else None,
                L0=float(L0[msk].mean()) if msk.any() else None,
                ratio=float(model[msk].mean() / L0[msk].mean())
                if msk.any() and L0[msk].mean() > 0 else None)

        # 事件帧 vs 静息帧的误差贡献
        ev_share = float(ee[evt].sum() / max(ee.sum(), 1e-12)) if evt.any() else None
        bap_share = float(be[bap].sum() / max(be.sum(), 1e-12)) if bap.any() else None

        out[k] = dict(
            n=int(model.size),
            model=float(model.mean()), L0=float(L0.mean()),
            ratio=float(model.mean() / L0.mean()),
            by_regime=by_reg,
            per_factor={name: float(pf_m[:, j].mean() / max(pf_0[:, j].mean(), 1e-12))
                        for j, name in enumerate(ds.meta['factor_names'])},
            bap=dict(n_frames=int(bap.sum()), n_trials=int(bap.sum() > 0),
                     err=float(be[bap].mean()) if bap.any() else None,
                     err_other=float(be[~bap].mean()) if (~bap).any() else None,
                     l0=float(b0[bap].mean()) if bap.any() else None,
                     share_of_err=bap_share),
            ca_event=dict(n_frames=int(evt.sum()),
                          err=float(ee[evt].mean()) if evt.any() else None,
                          err_other=float(ee[~evt].mean()) if (~evt).any() else None,
                          share_of_err=ev_share),
            latent=dict(zabs_mean=float(zabs.mean()),
                        zabs_p95=float(np.percentile(zabs, 95)),
                        cos_mean=float(zcos.mean()),
                        cos_p05=float(np.percentile(zcos, 5))),
        )
    return out


def plot(results, ks, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    regs = ("linear", "inter", "plateau")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (name, res) in zip(axes, results.items()):
        for r in regs:
            y = [res[k]["by_regime"][r]["ratio"] for k in ks]
            ax.plot(ks, y, marker="o", label=r)
        y = [res[k]["ratio"] for k in ks]
        ax.plot(ks, y, "k--", marker="s", label="overall")
        ax.axhline(1.0, color="gray", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("horizon k (steps)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("model / L0")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "error_vs_horizon.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (name, res) in zip(axes, results.items()):
        pf = res[max(ks)]["per_factor"]
        ax.bar(range(len(pf)), list(pf.values()))
        ax.set_xticks(range(len(pf)))
        ax.set_xticklabels(list(pf.keys()), rotation=60, fontsize=8)
        ax.axhline(1.0, color="gray", lw=0.8)
        ax.set_title(f"{name}  per-factor model/L0 @ k={max(ks)}")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "per_factor.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, res in results.items():
        ax.plot(ks, [res[k]["latent"]["zabs_mean"] for k in ks],
                marker="o", label=f"{name} |z|")
        ax.plot(ks, [res[k]["latent"]["cos_mean"] for k in ks],
                marker="s", linestyle="--", label=f"{name} cos(z,z_true)")
    ax.set_xscale("log")
    ax.set_xlabel("horizon k (steps)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "latent_drift.png"), dpi=140)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tools.diag_a")
    ap.add_argument("--data", default="datasets/2s/va_s2_2s")
    ap.add_argument("--tier", default="t2")
    ap.add_argument("--t0", type=int, default=4000)
    ap.add_argument("--n-trials", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ks", default="10,40,100,200,500,1000,2000,4000,8000")
    ap.add_argument("--models", nargs="+", required=True,
                    help="name=ckpt，可多个")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig-dir", default="")
    a = ap.parse_args(argv)

    ks = sorted(int(x) for x in a.ks.split(","))
    safety.preflight(limit_mb=3000)
    ds = D2.Dataset2(DATA, a.data, tier=a.tier)
    print(f"data={a.data} n={min(a.n_trials, ds.N)} t0={a.t0} ks={ks}",
          flush=True)

    results = {}
    for spec in a.models:
        name, ckpt = spec.split("=", 1)
        print(f"=== {name}: {ckpt} ===", flush=True)
        m, sc, _cfg = load_model(ckpt, ds, a.tier)
        results[name] = run_model(m, sc, ds, ks, a.t0, a.n_trials, a.batch)
        del m

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(dict(config=dict(data=a.data, t0=a.t0, ks=ks,
                                   n_trials=min(a.n_trials, ds.N)),
                       models=results), f, ensure_ascii=False, indent=1)
    print(f"=> {a.out}", flush=True)
    if a.fig_dir:
        plot(results, ks, a.fig_dir)
        print(f"=> figures {a.fig_dir}", flush=True)

    print("\nmodel      k     ratio   lin    int    pla   bAPshare  CasShare  |z|   cos", flush=True)
    for name, res in results.items():
        for k in ks:
            r = res[k]
            br = r["by_regime"]
            print(f"{name:<8} {k:>5}  {r['ratio']:.3f}  "
                  f"{br['linear']['ratio'] if br['linear']['ratio'] else float('nan'):.3f}  "
                  f"{br['inter']['ratio'] if br['inter']['ratio'] else float('nan'):.3f}  "
                  f"{br['plateau']['ratio'] if br['plateau']['ratio'] else float('nan'):.3f}  "
                  f"{r['bap']['share_of_err'] if r['bap']['share_of_err'] else float('nan'):.3f}     "
                  f"{r['ca_event']['share_of_err'] if r['ca_event']['share_of_err'] else float('nan'):.3f}   "
                  f"{r['latent']['zabs_mean']:.2f}  {r['latent']['cos_mean']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

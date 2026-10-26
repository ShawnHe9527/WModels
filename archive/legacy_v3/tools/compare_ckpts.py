"""多检查点严格配对对比。

## 为什么需要独立脚本

`world.learn.train eval` 每次采样都换了 `n_rep` 批序列，
所以不同 checkpoint 的误差混入了**采样噪声**（实测同一模型两次 k=40 可差 26%）。
本脚本让**所有模型跑完全相同的输入序列**，误差差异只反映模型本身。

## 用法

    cd WModel2 && PYTHONPATH=$PWD python -m tools.compare_ckpts \
        --data te_s3 --n-seq 32 --n-rep 3 \
        --ckpts t2c_dz16/best.pt t2s3_dz16/stageA.pt t2s3_dz16/last.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.core import safety

THREADS = safety.apply_thread_caps(4)

import numpy as np                    # noqa: E402
import torch                          # noqa: E402

from world.learn import data as D     # noqa: E402
from world.learn import metrics as M  # noqa: E402
from world.learn import model as M2   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results", "checkpoints")


def load(ck_path: str, th_dim: int, ds_nseg: int):
    ck = torch.load(os.path.join(RES, ck_path), map_location="cpu",
                    weights_only=False)
    cfg = ck.get("cfg") or {}
    m = M2.build(d_z=ck.get("d_z", cfg.get("d_z", 16)),
                 hidden=cfg.get("hidden", 192),
                 n_layers=cfg.get("n_layers", 2),
                 n_factors=D.N_FACTORS, u_dim=D.U_CH, th_dim=th_dim,
                 pred_attn=cfg.get("pred_attn", False),
                 n_heads=cfg.get("n_heads", 4))
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck, ck["scaler"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="te_s3")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--horizons", default="10,40,200,1000,4000,8000")
    ap.add_argument("--n-seq", type=int, default=32)
    ap.add_argument("--n-rep", type=int, default=3)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default="")
    ap.add_argument("--mem-mb", type=float, default=2000.0)
    a = ap.parse_args(argv)

    safety.preflight(limit_mb=a.mem_mb)
    ds = D.Dataset(DATA, a.data)
    ks = [int(v) for v in a.horizons.split(",")]
    L = max(ks) + 1
    th_dim = int(ds.th.shape[1])

    # ---- 所有模型共用同一批采样器（同 seed ⇒ 同序列）----
    sampler = D.Sampler(ds, {"loc_s":[0]*9,"scale_s":[1]*9,"loc_u":[0]*3,"scale_u":[1]*3}, batch=a.n_seq, length=L, seed=a.seed)

    models = []
    for p in a.ckpts:
        m, ck, sc = load(p, th_dim, ds.nseg)
        sampler.ns, sampler.nu = D.normalizers(sc)
        models.append((p, m, ck))
        print(f"载入 {p}  step={ck.get('step')}  "
              f"σ={ck.get('sigma', ck.get('val_rel'))}")

    # ---- 一次性采样所有 batch（所有模型共用）----
    sampler.ns, sampler.nu = D.normalizers(models[0][2]["scaler"])
    batches = [sampler.batch_pair() for _ in range(a.n_rep)]

    # 用第 0 个模型的 scaler 统一归一化（各模型 scaler 应一致，来自 T1）
    def to_torch(b):
        s0, u, th, st, x = b
        return tuple(torch.from_numpy(np.asarray(v))
                     for v in (s0, u, th, st, x))

    tb = [to_torch(b) for b in batches]

    results = {}
    for name, model, ck in models:
        sig = float(ck.get("sigma") or ck.get("val_rel") or 0.0)
        rows = []
        with torch.no_grad():
            for (s0, u, th, s_true, x) in tb:
                out = model.rollout(s0, u, th, x, use_residual=True)
                for k in ks:
                    st_k = s_true[:, k].numpy()
                    m = float(np.abs(out[:, k - 1].numpy() - st_k).mean())
                    l0 = float(np.abs(s0.numpy() - st_k).mean())
                    rows.append(dict(k=k, ms=k * ds.dt, model=m, L0=l0,
                                     a=M.amplification(m, k, sig)))
        agg = {}
        for r in rows:
            agg.setdefault(r["k"], []).append(r)
        results[name] = dict(
            sigma=sig,
            rows={k: dict(ms=agg[k][0]["ms"],
                          model=float(np.mean([r["model"] for r in agg[k]])),
                          L0=float(np.mean([r["L0"] for r in agg[k]])),
                          a=float(np.mean([r["a"] for r in agg[k]])))
                  for k in ks})

    # ---- 打印 ----
    print("\n" + "=" * 96)
    print(f"严格配对对比（{a.data}，n={a.n_seq}×{a.n_rep}，"
          f"**所有模型跑完全相同的序列**）")
    hdr = f"{'k':>6} {'ms':>7} {'L0':>9} " + " ".join(
        f"{os.path.basename(os.path.dirname(p)):>14}" for p in a.ckpts)
    print(hdr)
    print("-" * 96)
    for k in ks:
        l0 = results[a.ckpts[0]]["rows"][k]["L0"]
        cells = []
        for p in a.ckpts:
            v = results[p]["rows"][k]["model"]
            cells.append(f"{v:>8.4f}({v/l0:>4.2f})")
        print(f"{k:>6} {k*ds.dt:>7.1f} {l0:>9.5f} " + " ".join(cells))

    print("\n放大比 a = ε/(√k·σ)")
    print(f"{'k':>6} " + " ".join(
        f"{os.path.basename(os.path.dirname(p)):>14}" for p in a.ckpts))
    for k in ks:
        print(f"{k:>6} " + " ".join(
            f"{results[p]['rows'][k]['a']:>14.2f}" for p in a.ckpts))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"\n⇒ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

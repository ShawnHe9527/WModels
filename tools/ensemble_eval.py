"""多模型集成评测：逐视界平均预测，并与单模对比。

用法：
  env -u DISPLAY python -m tools.ensemble_eval \
      --models s1=r6_ab_AB/last.pt s2=r6_ab_s2/last.pt s3=r6_ab_s3/last.pt \
      --data datasets/1s/va_s2 --t0 4000 --ks 2000,8000,16000 --n-trials 128 \
      --out runs/.../ensemble.json
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
import torch

from world.learn import data2 as D2
from tools.plot_v4 import load_model


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def rollout(m, ds, sc, i, t0, K, kset):
    probe = D2.Sampler2(ds, sc, batch=1, length=K, seed=0)
    s0, u_tok, ug, th, st, x = probe.batch_window(t0, K, 1, off=i)
    s0t = torch.from_numpy(np.ascontiguousarray(s0)).float()
    ut = torch.from_numpy(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2))).float()
    ugt = torch.from_numpy(np.ascontiguousarray(ug)).float()
    tht = torch.from_numpy(th).float()
    xt = torch.from_numpy(x).float()
    z = m.encode(s0t.unsqueeze(1), ut[:, :1], tht, xt, ugt[:, :1]).squeeze(1)
    s = s0t.clone()
    rec = {}
    for j in range(K):
        jj = min(j + 1, K - 1)
        zn = m.predict(z.unsqueeze(1), ut[:, jj:jj + 1], tht, xt,
                       ugt[:, jj:jj + 1]).squeeze(1)
        s = m.step_residual(s, z, zn, tht, xt, ugt[:, jj:jj + 1])
        z = zn
        if (j + 1) in kset:
            rec[j + 1] = s.squeeze(0).numpy()
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=ckpt")
    ap.add_argument("--data", default="datasets/1s/va_s2")
    ap.add_argument("--t0", type=int, default=4000)
    ap.add_argument("--ks", default="2000,8000,16000")
    ap.add_argument("--n-trials", type=int, default=128)
    ap.add_argument("--out", default="runs/ensemble.json")
    a = ap.parse_args(argv)

    ds = D2.Dataset2("data", a.data, tier="t2")
    ks = [int(x) for x in a.ks.split(",")]
    K = max(ks)
    # 用第一个模型的 scaler
    models = []
    sc0 = None
    for spec in a.models:
        name, ck = spec.split("=", 1)
        m, sc, cfg = load_model(ck, ds, "t2")
        models.append((name, m))
        if sc0 is None:
            sc0 = sc
    loc = np.asarray(sc0["loc_s"], np.float32)[:, None]
    scl = np.asarray(sc0["scale_s"], np.float32)[:, None]

    preds = {name: [] for name, _ in models}
    for i in range(a.n_trials):
        for name, m in models:
            preds[name].append(rollout(m, ds, sc0, i, a.t0, K, set(ks)))
        if (i + 1) % 32 == 0:
            log(f"{i+1}/{a.n_trials} trials")
    base = {k: np.asarray(ds.s[np.arange(a.n_trials), a.t0 + k]) for k in ks}
    l0 = np.asarray(ds.s[np.arange(a.n_trials), a.t0])

    rows = []
    ens = {k: [] for k in ks}
    for k in ks:
        idx = k - 1
        ps = []
        for name, _ in models:
            p = np.asarray([x[k] for x in preds[name]]) * scl + loc
            ps.append(p)
            den = np.mean(np.abs((l0 - base[k]) / scl))
            num = np.mean(np.abs((p - base[k]) / scl))
            rows.append(dict(model=name, k=k, ratio=float(num / den),
                             n_div=int(np.sum(~np.isfinite(p).all(axis=(1, 2))))))
        e = np.mean(ps, axis=0)
        den = np.mean(np.abs((l0 - base[k]) / scl))
        num = np.mean(np.abs((e - base[k]) / scl))
        rows.append(dict(model="ENSEMBLE", k=k, ratio=float(num / den),
                         n_div=int(np.sum(~np.isfinite(e).all(axis=(1, 2))))))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    log(f"→ {a.out}")
    for r in rows:
        log(str(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

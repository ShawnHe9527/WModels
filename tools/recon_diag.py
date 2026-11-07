"""重建诊断：冻结 Enc/Dec 在“峰帧”与“静息帧”上的重建误差。"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

from world.learn import data2 as D2
from tools.plot_v4 import load_model
from tools.ca_event_metrics import _peaks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="t1v4_dz16/best.pt")
    ap.add_argument("--data", default="datasets/1s/va_s2")
    ap.add_argument("--n-random", type=int, default=300)
    ap.add_argument("--out", default="runs/recon_diag.json")
    a = ap.parse_args(argv)

    ds = D2.Dataset2("data", a.data, tier="t2")
    m, sc, cfg = load_model(a.ckpt, ds, "t2")
    loc = np.asarray(sc["loc_s"], np.float32)[:, None]
    scl = np.asarray(sc["scale_s"], np.float32)[:, None]
    ci, vi = 7, 8
    rng = np.random.default_rng(0)

    peak_frames = []
    for i in range(ds.N):
        c = np.asarray(ds.s[i, :, ci, :]).max(axis=1)
        idx, _, _ = _peaks(c, 0.025)
        peak_frames += [(i, int(t)) for t in idx]
    rng.shuffle(peak_frames)
    peak_frames = peak_frames[:a.n_random]
    rand_frames = [(int(rng.integers(0, ds.N)), int(rng.integers(500, ds.s.shape[1] - 500)))
                   for _ in range(a.n_random)]

    def recon(frames):
        errs = []
        for i, t in frames:
            probe = D2.Sampler2(ds, sc, batch=1, length=1, seed=0)
            s0, u_tok, ug, th, st, x = probe.batch_window(t, 1, 1, off=i)
            st0 = torch.from_numpy(np.ascontiguousarray(s0)).float()
            ut = torch.from_numpy(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2))).float()
            ugt = torch.from_numpy(np.ascontiguousarray(ug)).float()
            tht = torch.from_numpy(th).float()
            xt = torch.from_numpy(x).float()
            with torch.no_grad():
                z = m.encode(st0.unsqueeze(1), ut[:, :1], tht, xt, ugt[:, :1]).squeeze(1)
                rec = m.decode(z.unsqueeze(1), tht, xt).squeeze(1)
            errs.append((rec - st0).abs().squeeze(0).numpy())  # (9,11) normalized
        return np.asarray(errs)  # (N,9,11)

    ep = recon(peak_frames)
    er = recon(rand_frames)
    names = ds.meta["factor_names"]
    print(f"{'factor':12s} {'peak_MAE':>9s} {'rand_MAE':>9s}  ratio")
    rows = {}
    for f, n in enumerate(names):
        mp, mr = float(ep[:, f, :].mean()), float(er[:, f, :].mean())
        rows[n] = dict(peak=mp, rand=mr, ratio=mp / max(mr, 1e-12))
        print(f"{n:12s} {mp:9.4f} {mr:9.4f}  {mp/max(mr,1e-12):6.2f}")
    # 峰帧的 V 空间方差（原始单位）
    vs = []
    for i, t in peak_frames:
        v = np.asarray(ds.s[i, t, vi, :])
        vs.append(float(v.std()))
    print(f"\n峰帧 V 空间 std（mV，11 compartment）: 中位 {np.median(vs):.2f}, P90 {np.percentile(vs,90):.2f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fo:
        json.dump(dict(rows=rows, peak_v_std_median=float(np.median(vs))), fo,
                  ensure_ascii=False, indent=1)
    print("→", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""B1 基线：直接 MLP（无 latent 瓶颈），与 A+B 完全相同的训练配方。

s_{t+1} = s_t + MLP([s_t(99), u_t(85), θ(27)])
训练：一步事件加权 + 闭环（horizon 512, lags 64/256/512, TBPTT chunk 64）+ 课程。
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from world.learn import data2 as D2
from world.core import safety


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class DirectMLP(nn.Module):
    def __init__(self, nf=9, P=11, udim=140, th=27, hidden=256, n_layers=2):
        super().__init__()
        self.nf, self.P = nf, P
        din = nf * P + udim + th
        layers, d = [], din
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, nf * P)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def delta(self, s, u_flat, th):
        B = s.shape[0]
        x = torch.cat([s.reshape(B, -1), u_flat, th], dim=-1)
        return self.net(x).reshape(B, self.nf, self.P)


def _torch(x):
    return torch.from_numpy(np.ascontiguousarray(x)).float()


def eval_ratio(m, ds, sc, ks, t0, n_trials):
    loc = np.asarray(sc["loc_s"], np.float32)[:, None]
    scl = np.asarray(sc["scale_s"], np.float32)[:, None]
    K = max(ks)
    out = {}
    with torch.no_grad():
        for i in range(n_trials):
            probe = D2.Sampler2(ds, sc, batch=1, length=K, seed=0)
            s0, u_tok, ug, th, st, x = probe.batch_window(t0, K, 1, off=i)
            uflat = _torch(u_tok.transpose(0, 1, 3, 2).reshape(1, K, -1))
            tht = _torch(th)
            s = _torch(s0)
            for j in range(K):
                jj = min(j + 1, K - 1)
                s = s + m.delta(s, uflat[:, jj], tht)
                if (j + 1) in ks:
                    pred = (s.numpy() * scl + loc)[0]
                    true = np.asarray(ds.s[i, t0 + j + 1])
                    l0 = np.asarray(ds.s[i, t0])
                    out.setdefault(j + 1, []).append(
                        (np.abs(pred - true).mean(), np.abs(l0 - true).mean()))
    res = {}
    for k in ks:
        arr = np.asarray(out[k])
        res[k] = dict(ratio=float(arr[:, 0].mean() / arr[:, 1].mean()),
                      n=int(len(arr)))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/1s/tr_s1")
    ap.add_argument("--val", default="datasets/1s/va_s2")
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--cl-horizon", type=int, default=512)
    ap.add_argument("--cl-lags", default="64,256,512")
    ap.add_argument("--cl-chunk", type=int, default=64)
    ap.add_argument("--cl-stage-a", type=int, default=2000)
    ap.add_argument("--event-weight", type=float, default=10.0)
    ap.add_argument("--event-tau", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-n", type=int, default=128)
    ap.add_argument("--eval-ks", default="200,2000,8000,16000")
    ap.add_argument("--out", default="runs/b1.json")
    a = ap.parse_args(argv)

    safety.preflight(limit_mb=16000)
    tr = D2.Dataset2("data", a.data, tier="t2")
    va = D2.Dataset2("data", a.val, tier="t2")
    sc = json.load(open("/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json"))
    torch.manual_seed(a.seed)
    m = DirectMLP(hidden=a.hidden)
    log(f"B1 DirectMLP 参数 {sum(p.numel() for p in m.parameters())}")

    s_tr = D2.Sampler2(tr, sc, batch=a.batch, length=a.length, seed=a.seed,
                       p_trial=None, hot_frac=0.5,
                       event_center_frac=0.0, event_center_thr=0.02)
    cl_s = D2.Sampler2(tr, sc, batch=a.batch, length=a.cl_horizon + 2,
                       seed=a.seed + 1000, p_trial=None, hot_frac=0.5,
                       event_center_frac=0.0, event_center_thr=0.02)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    ci = tr.meta["factor_names"].index("cai")

    for step in range(1, a.steps + 1):
        s0, u_tok, ug, th, s_true, x = _torch_pair(s_tr)
        uflat = u_tok.permute(0, 1, 3, 2).reshape(s0.shape[0], a.length, -1)
        s_prev = torch.cat([s0.unsqueeze(1), s_true[:, :-1]], dim=1)
        pred = s_prev + m.delta(s_prev.reshape(-1, 9, 11), uflat.reshape(-1, uflat.shape[-1]),
                                th.repeat_interleave(a.length, 0)).reshape_as(s_prev)
        d1 = (s_true[:, 1:, ci, :] - s_true[:, :-1, ci, :]).abs().amax(dim=2)
        dca = torch.cat([torch.zeros_like(d1[:, :1]), d1], dim=1)
        w = 1.0 + a.event_weight * torch.sigmoid(dca / a.event_tau)
        loss = (w * (pred - s_true).pow(2).mean(dim=(2, 3))).sum() / w.sum()

        if a.cl_horizon > 0:
            c0, cu, cug, cth, cst, cx = _torch_pair(cl_s)
            cuflat = cu.permute(0, 1, 3, 2).reshape(c0.shape[0], a.cl_horizon + 2, -1)
            frac = max(0.0, (step - a.cl_stage_a) / max(a.steps - a.cl_stage_a, 1))
            h = int(round(4 + frac * (a.cl_horizon - 4)))
            sc_ = c0.clone()
            lags = [int(x) for x in a.cl_lags.split(",") if 0 < int(x) <= h]
            cl = 0.0
            for j in range(h):
                sc_ = sc_ + m.delta(sc_, cuflat[:, j + 1], cth)
                if (j + 1) in lags:
                    cl = cl + F.mse_loss(sc_, cst[:, j + 1])
                if a.cl_chunk > 0 and (j + 1) % a.cl_chunk == 0 and (j + 1) < h:
                    sc_ = sc_.detach()
            if lags:
                loss = loss + cl / len(lags)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 0.3 if step > a.cl_stage_a else 1.0)
        if step > a.cl_stage_a:
            for g in opt.param_groups:
                g["lr"] = 2e-4
        opt.step()
        if step % 500 == 0 or step == a.steps:
            log(f"step {step}: loss={loss.item():.3e}")

    eks = [int(x) for x in a.eval_ks.split(",")]
    res = eval_ratio(m, va, sc, eks, 4000, a.eval_n)
    log(str(res))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


def _torch_pair(sampler):
    s0, u_tok, ug, th, s_true, x = sampler.batch_pair()
    return (torch.from_numpy(np.ascontiguousarray(s0)).float(),
            torch.from_numpy(np.ascontiguousarray(u_tok)).float(),
            torch.from_numpy(np.ascontiguousarray(ug)).float(),
            torch.from_numpy(np.ascontiguousarray(th)).float(),
            torch.from_numpy(np.ascontiguousarray(s_true)).float(),
            torch.from_numpy(np.ascontiguousarray(x)).float())


if __name__ == "__main__":
    raise SystemExit(main())

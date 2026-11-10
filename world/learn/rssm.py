"""RSSM-lite：在冻结的 Enc/Dec 上训练“GRU 记忆 + 随机 latent + KL”。

结构：
  z_t = Enc(s_t)                    （冻结）
  q(z_t|s_t) = N(μ_q, σ_q)          （后验头）
  h_t = GRU(z_{t-1}, pooled_u_t)    （记忆）
  p(z_t|h_t) = N(μ_p, σ_p)          （先验头）
  L = MSE(Dec(z_t^s), s_t) + β·KL(q||p)

用法（冒烟）：
  python -m world.learn.rssm --steps 3 --batch 4 --length 32
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from world.learn import data2 as D2
from world.core import safety


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).float()


class RSSM(nn.Module):
    def __init__(self, d_z=16, hidden=128, u_dim=5, h_dim=64):
        super().__init__()
        self.h_dim = h_dim
        self.post = nn.Sequential(nn.Linear(d_z, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 2 * d_z))
        self.prior = nn.Sequential(nn.Linear(h_dim, hidden), nn.SiLU(),
                                   nn.Linear(hidden, 2 * d_z))
        # 后验做残差（μ = z + Δμ），初始 Δμ=0 ⇒ 初始 μ=z，避免 decoder 收到离谱 z
        nn.init.zeros_(self.post[-1].weight)
        nn.init.zeros_(self.post[-1].bias)
        self.gru = nn.GRUCell(d_z + u_dim, h_dim)

    def sample(self, mu, logvar):
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def forward_teacher(self, z_t, u_pool):
        """z_t:(B,L,dz) u_pool:(B,L,5) → (mu_q, lv_q, mu_p, lv_p, h)*L"""
        B, L, _ = z_t.shape
        h = torch.zeros(B, self.h_dim)
        mu_q, lv_q, mu_p, lv_p = [], [], [], []
        z_prev = torch.zeros(B, z_t.shape[-1])
        for t in range(L):
            h = self.gru(torch.cat([z_prev, u_pool[:, t]], -1), h)
            mq, lq = self.post(z_t[:, t]).chunk(2, -1)
            mq = z_t[:, t] + mq
            lq = lq.clamp(-6.0, 2.0)
            mp, lp = self.prior(h).chunk(2, -1)
            mu_q.append(mq); lv_q.append(lq)
            mu_p.append(mp); lv_p.append(lp)
            z_prev = z_t[:, t]
        return (torch.stack(mu_q, 1), torch.stack(lv_q, 1),
                torch.stack(mu_p, 1), torch.stack(lv_p, 1), h)


def kl(mu_q, lv_q, mu_p, lv_p):
    return 0.5 * (lv_p - lv_q + (torch.exp(lv_q) + (mu_q - mu_p) ** 2)
                  / torch.exp(lv_p).clamp(min=1e-6) - 1).mean()


@torch.no_grad()
def rollout_eval(m, enc, dec, ds, sc, ks, t0, n_trials):
    """用先验 μ 闭环 rollout，返回 {k: model/L0}。"""
    loc = np.asarray(sc["loc_s"], np.float32)[:, None]
    scl = np.asarray(sc["scale_s"], np.float32)[:, None]
    K = max(ks)
    out = {k: [] for k in ks}
    for i in range(n_trials):
        probe = D2.Sampler2(ds, sc, batch=1, length=K + 1, seed=0)
        s0, u_tok, ug, th, st, x = probe.batch_window(t0, K + 1, 1, off=i)
        s0t, tht, xt = _t(s0), _t(th), _t(x)
        u = _t(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2)))
        z = enc(s0t.unsqueeze(0), u[:, :1], tht, xt, _t(ug)[:, :1]).squeeze(1)
        h = torch.zeros(1, m.h_dim)
        for j in range(K):
            h = m.gru(torch.cat([z, u[:, j + 1].mean(-1)], -1), h)
            mp, _ = m.prior(h).chunk(2, -1)
            z = mp
            if (j + 1) in ks:
                rec = dec(z.unsqueeze(1), tht, xt).squeeze(1)
                pred = rec.squeeze(0).numpy() * scl + loc
                true = np.asarray(ds.s[i, t0 + j + 1])
                l0 = np.asarray(ds.s[i, t0])
                out[j + 1].append(float(np.abs(pred - true).mean()
                                        / np.abs(l0 - true).mean()))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/1s/tr_s1")
    ap.add_argument("--val", default="datasets/1s/va_s2")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--cl-horizon", type=int, default=512)
    ap.add_argument("--cl-lags", default="64,256,512")
    ap.add_argument("--cl-chunk", type=int, default=64)
    ap.add_argument("--cl-stage-a", type=int, default=2000)
    ap.add_argument("--eval-ks", default="2000,8000,16000")
    ap.add_argument("--eval-n", type=int, default=128)
    ap.add_argument("--out", default="runs/rssm.json")
    a = ap.parse_args(argv)

    safety.preflight(limit_mb=8000)
    tr = D2.Dataset2("data", a.data, tier="t2")
    sc = json.load(open("/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json"))
    torch.manual_seed(a.seed)

    # 冻结的 T1
    import sys
    sys.path.insert(0, "/home/Hesj/WModel")
    from tools.plot_v4 import load_model
    enc_model, sc, cfg = load_model("t1v4_dz16/best.pt", tr, "t2")
    enc, dec = enc_model.enc, enc_model.dec
    for p in list(enc.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)
    enc.eval(); dec.eval()

    m = RSSM(hidden=a.hidden)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    cl_s = D2.Sampler2(tr, sc, batch=a.batch, length=a.cl_horizon + 2,
                       seed=a.seed + 1000)
    s_tr = D2.Sampler2(tr, sc, batch=a.batch, length=a.length, seed=a.seed,
                       p_trial=None, hot_frac=0.5, event_center_frac=0.0,
                       event_center_thr=0.02)
    for step in range(1, a.steps + 1):
        s0, u_tok, ug, th, s_true, x = s_tr.batch_pair()
        s0 = _t(s0); s_true = _t(s_true); th = _t(th); x = _t(x)
        u = _t(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2)))  # (B,L,5,P)
        with torch.no_grad():
            z = enc(s_true, u, th, x, _t(ug))
        u_pool = u.mean(-1)                       # (B,L,5)
        mq, lq, mp, lp, _ = m.forward_teacher(z, u_pool)
        zs = m.sample(mq, lq)
        rec = dec(zs.reshape(-1, z.shape[-1]).unsqueeze(1), th.repeat_interleave(a.length, 0),
                  x).reshape_as(s_true)
        loss_rec = F.mse_loss(rec, s_true)
        loss = loss_rec + a.kl_weight * kl(mq, lq, mp, lp)

        # ---- CL：先验采样闭环 + 多滞后损失 ----
        if a.cl_horizon > 0:
            c0, cu, cug, cth, cst, cx = cl_s.batch_pair()
            c0t, ctht, cxt, cstt = _t(c0), _t(cth), _t(cx), _t(cst)
            cu_t = _t(np.ascontiguousarray(cu.transpose(0, 1, 3, 2)))
            with torch.no_grad():
                zc = enc(cstt, cu_t, ctht, cxt, _t(cug))[:, 0]
            frac = max(0.0, (step - a.cl_stage_a) / max(a.steps - a.cl_stage_a, 1))
            hh = int(round(4 + frac * (a.cl_horizon - 4)))
            lags = [int(v) for v in a.cl_lags.split(",") if 0 < int(v) <= hh]
            hz = torch.zeros(zc.shape[0], m.h_dim)
            zz = zc
            cl = 0.0
            for j in range(hh):
                hz = m.gru(torch.cat([zz, cu_t[:, j + 1].mean(-1)], -1), hz)
                mp2, _ = m.prior(hz).chunk(2, -1)
                zz = mp2
                if (j + 1) in lags:
                    rec2 = dec(zz.unsqueeze(1), ctht, cxt).squeeze(1)
                    cl = cl + F.mse_loss(rec2, cstt[:, j + 1])
                if a.cl_chunk > 0 and (j + 1) % a.cl_chunk == 0 and (j + 1) < hh:
                    zz = zz.detach(); hz = hz.detach()
            if lags:
                loss = loss + cl / len(lags)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 1 == 0:
            log(f"step {step}: rec={loss_rec.item():.4e} kl={kl(mq,lq,mp,lp).item():.4f}")
    eks = [int(v) for v in a.eval_ks.split(",")]
    res = rollout_eval(m, enc, dec, tr, sc, eks, 4000, a.eval_n)
    log(f"eval: {res}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(steps=a.steps, eval=res), open(a.out, "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

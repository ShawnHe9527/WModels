"""因果时序卷积基线（TDNN：固定感受野 C 的因果卷积等价形式）。

输入每步 = [s_t(99), u_tokens(140), bAP(1), θ(27)] = 267；
取最近 C=8 步拼接（2136）→ MLP → Δs(99)。
与 A+B 同配方（教师强制 + CL 512/lags 64,256,512/TBPTT 64 + 课程）。
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from world.learn import data2 as D2
from world.core import safety

C = 8


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class TDNN(nn.Module):
    def __init__(self, nf=9, P=11, udim=140, th=27, hidden=256, ctx=C):
        super().__init__()
        self.nf, self.P, self.ctx = nf, P, ctx
        din = (nf * P + udim + 1 + th) * ctx
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, nf * P))

    def forward(self, hist):
        # hist: (B, ctx, 267) -> (B, nf, P)
        B = hist.shape[0]
        return self.net(hist.reshape(B, -1)).reshape(B, self.nf, self.P)


def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).float()


def feats(s, uflat, bap, th):
    """单步特征 (B,267)。"""
    return torch.cat([s.reshape(s.shape[0], -1), uflat, bap, th], dim=-1)


def eval_ratio(m, ds, sc, ks, t0, n):
    loc = np.asarray(sc["loc_s"], np.float32)[:, None]
    scl = np.asarray(sc["scale_s"], np.float32)[:, None]
    K = max(ks)
    out = {}
    with torch.no_grad():
        for i in range(n):
            probe = D2.Sampler2(ds, sc, batch=1, length=K, seed=0)
            s0, u_tok, ug, th, st, x = probe.batch_window(t0, K, 1, off=i)
            uflat = _t(u_tok.transpose(0, 1, 3, 2).reshape(1, K, -1))
            tht = _t(th)
            bap = _t(ug).reshape(1, K, 1)
            sp = _t(s0)
            f0 = feats(sp, uflat[:, 0], bap[:, 0], tht)
            hist = f0.unsqueeze(0).expand(1, C, -1).clone()
            for j in range(K):
                jj = min(j + 1, K - 1)
                sp = sp + m(hist)
                f = feats(sp, uflat[:, jj], bap[:, jj], tht).unsqueeze(1)
                hist = torch.cat([hist[:, 1:], f], dim=1)
                if (j + 1) in ks:
                    pred = (sp.numpy() * scl + loc)[0]
                    true = np.asarray(ds.s[i, t0 + j + 1])
                    l0 = np.asarray(ds.s[i, t0])
                    out.setdefault(j + 1, []).append(
                        (np.abs(pred - true).mean(), np.abs(l0 - true).mean()))
    return {k: float(np.asarray(out[k])[:, 0].mean() / np.asarray(out[k])[:, 1].mean())
            for k in ks}


def main(argv=None):
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
    ap.add_argument("--out", default="runs/tdnn.json")
    a = ap.parse_args(argv)

    safety.preflight(limit_mb=16000)
    tr = D2.Dataset2("data", a.data, tier="t2")
    va = D2.Dataset2("data", a.val, tier="t2")
    sc = json.load(open("/home/Hesj/QH1_dataHe/WM/scalers/scaler_v4.json"))
    torch.manual_seed(a.seed)
    m = TDNN(hidden=a.hidden)
    log(f"TDNN(ctx={C}) 参数 {sum(p.numel() for p in m.parameters()):,}")

    s_tr = D2.Sampler2(tr, sc, batch=a.batch, length=a.length, seed=a.seed,
                       p_trial=None, hot_frac=0.5, event_center_frac=0.0,
                       event_center_thr=0.02)
    cl_s = D2.Sampler2(tr, sc, batch=a.batch, length=a.cl_horizon + 2,
                       seed=a.seed + 1000, p_trial=None, hot_frac=0.5,
                       event_center_frac=0.0, event_center_thr=0.02)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    ci = tr.meta["factor_names"].index("cai")

    def one_step_hist(s_prev, uflat, bap, th):
        """s_prev (B,L,9,11) → 窗口特征 (B,L,C,267)。"""
        B, L = s_prev.shape[0], s_prev.shape[1]
        fe = torch.cat([s_prev.reshape(B, L, -1), uflat,
                        bap.unsqueeze(-1), th.unsqueeze(1).expand(B, L, -1)],
                       dim=-1)                    # (B,L,267)
        pad = fe[:, :1].expand(B, C - 1, fe.shape[-1])
        fe = torch.cat([pad, fe], dim=1)          # (B,L+C-1,267)
        win = fe.unfold(1, C, 1).permute(0, 1, 3, 2)   # (B,L,C,267)
        return win

    for step in range(1, a.steps + 1):
        s0, u_tok, ug, th, s_true, x = s_tr.batch_pair()
        s0 = _t(s0); u_tok = _t(u_tok); ug = _t(ug); th = _t(th); s_true = _t(s_true)
        uflat = u_tok.permute(0, 1, 3, 2).reshape(s0.shape[0], a.length, -1)
        s_prev = torch.cat([s0.unsqueeze(1), s_true[:, :-1]], dim=1)
        win = one_step_hist(s_prev, uflat, ug[:, :, 0], th)
        B_, L_ = s_prev.shape[0], s_prev.shape[1]
        pred = s_prev + m(win.reshape(B_ * L_, C, -1)).reshape_as(s_prev)
        d1 = (s_true[:, 1:, ci, :] - s_true[:, :-1, ci, :]).abs().amax(dim=2)
        dca = torch.cat([torch.zeros_like(d1[:, :1]), d1], dim=1)
        w = 1.0 + a.event_weight * torch.sigmoid(dca / a.event_tau)
        loss = (w * (pred - s_true).pow(2).mean(dim=(2, 3))).sum() / w.sum()

        # CL：rollout，维护 C 步历史
        c0, cu, cug, cth, cst, cx = cl_s.batch_pair()
        c0 = _t(c0); cu = _t(cu); cug = _t(cug); cth = _t(cth); cst = _t(cst)
        cuflat = cu.permute(0, 1, 3, 2).reshape(c0.shape[0], a.cl_horizon + 2, -1)
        frac = max(0.0, (step - a.cl_stage_a) / max(a.steps - a.cl_stage_a, 1))
        h = int(round(4 + frac * (a.cl_horizon - 4)))
        lags = [int(x) for x in a.cl_lags.split(",") if 0 < int(x) <= h]
        sc_ = c0.clone()
        _f0 = feats(sc_, cuflat[:, 0], cug[:, 0].reshape(-1, 1), cth).unsqueeze(1)
        hist = _f0.expand(-1, C, -1).clone()
        cl = 0.0
        for j in range(h):
            dd = m(hist)
            sc_ = sc_ + dd
            nfe = feats(sc_, cuflat[:, j + 1],
                        cug[:, j + 1].reshape(-1, 1), cth).unsqueeze(1)
            hist = torch.cat([hist[:, 1:], nfe], dim=1)
            if (j + 1) in lags:
                cl = cl + F.mse_loss(sc_, cst[:, j + 1])
            if a.cl_chunk > 0 and (j + 1) % a.cl_chunk == 0 and (j + 1) < h:
                sc_ = sc_.detach()
                hist = hist.detach()
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
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

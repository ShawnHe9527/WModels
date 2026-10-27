"""v4 训练：T1 自编码 → T2 两阶段课程（教师强制 + 闭环尾）→ 配对评测。

## 与 v3 `train.py` 的区别

| | v3 | **v4 `train2.py`** |
|---|---|---|
| 数据 | `Dataset`：u (T,3,11) | **`Dataset2`：u (T,85) → token** |
| token 数 | 11 | **28（T2）/ 14（T1）**，由 `--tier` |
| 全局输入 | 无 | **`u_glob` = bAP 指示**，拼在池化之后 |
| 评测 | 全体平均 | **按 regime 分三档报 $a(k)$** |
| 其余 | 残差输出、两阶段课程、严格配对 | **不变**（v3 已验证有效）|

## 为什么必须按 regime 分档报指标

分层采样让 linear 档占 50%，而这些 trial 的 $V$ 跨距只有 16 mV、
上行占比 0% ⇒ 近乎静态。若只报总体，$a(k)$ 会被这些
「L0 本来就很强」的样本稀释，看不出模型真正赢在哪。

⇒ `cmd_eval` 会同时给**总体**和**分档**两组数。

## 用法

    # T1 自编码
    PYTHONPATH=$PWD python -m world.learn.train2 t1 \
        --data tr_s1 --val va_s2 --run t1v4_dz16 --tier t2 \
        --steps 12000 --d-z 16 --hidden 192 --batch 32 --length 64

    # T2 两阶段课程
    PYTHONPATH=$PWD python -m world.learn.train2 t2 \
        --data tr_s1 --val va_s2 --run t2v5_events --tier t2 \
        --init t1v4_dz16/best.pt --steps 9000 \
        --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 64 \
        --stage-b-lr 2e-4 --stage-b-grad-clip 0.3

    # 配对评测
    PYTHONPATH=$PWD python -m world.learn.train2 eval \
        --ckpt t2v5_events/last.pt --data te_s3 --tier t2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from world.core import safety                        # noqa: E402

THREADS = safety.apply_thread_caps(4)

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import torch.nn.functional as F                      # noqa: E402

from world.core import progress                      # noqa: E402
from world.learn import data2 as D2                  # noqa: E402
from world.learn import model as M                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results", "checkpoints")
SCALER = os.path.join(ROOT, "results", "data", "scaler_v4.json")

# 归一化空间中 |s| 的合理上限（真实数据 |V| 归一化后约 1.7，留 60 倍余量）
SANE_MAX = 100.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cfg(a: argparse.Namespace) -> dict:
    """剔除 func（否则 torch.save 会把模块引用一起 pickle）。"""
    return {k: v for k, v in vars(a).items() if k != "func"}


def _regime_p(ds, spec: str):
    """'0.25,0.35,0.40' → 每 trial 的抽样概率（linear,inter,plateau）。

    语义是**目标组占比**：三个数之和应≈1，各组被抽到的总概率就是这三个数。
    ⚠️ 不能简单地让每个 trial 取权重 w：那样 trial 多的档会占更大份额
    （实测 tr_s1 为 93/68/39，会把 0.25/0.35/0.40 变成 0.37/0.38/0.25）。

    None / "" ⇒ 均匀采样（旧行为）。只给训练 sampler 用。
    """
    if not spec:
        return None
    w = [float(x) for x in str(spec).split(",")]
    if len(w) != 3:
        raise ValueError(f"--regime-weights 需 3 个数（linear,inter,plateau），"
                         f"got {spec!r}")
    m = {"linear": w[0], "inter": w[1], "plateau": w[2]}
    p = np.zeros(ds.N, dtype=np.float64)
    tot = 0.0
    for r, v in m.items():
        msk = (ds.regimes == r)
        n = int(msk.sum())
        if n and v > 0:
            p[msk] = v / n                 # 组内均分 ⇒ 组总概率 = v
            tot += v
    if tot <= 0:
        return None
    p /= p.sum()
    return p


def _log_sampling(s_tr, a, ds) -> None:
    # anchor 配置无论是否启用抽样都打印，便于各轮次事后归因
    log(f"  anchor：weight={a.anchor_weight:.3f} every={a.anchor_every} "
        f"cl={a.anchor_cl:.3f}")
    if not (a.regime_weights or a.hot_frac > 0):
        return
    p = s_tr.p_trial
    if p is not None:
        parts = []
        for r, i in (("linear", 0), ("inter", 1), ("plateau", 2)):
            msk = (ds.regimes == r)
            parts.append(f"{r}={p[msk].sum():.2f}")
        log(f"采样：每 batch 期望档比例 " + "  ".join(parts))
    log(f"  hot_frac={a.hot_frac:.2f}，含事件 trial {len(s_tr.hot)}/{ds.N}")
    if a.event_weight > 0:
        log(f"  事件加权：weight={a.event_weight:.1f} tau={a.event_tau:.3f}")


def save_ckpt(path: str, model, **kw) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(model=model.state_dict(), **kw), path)


# ==============================================================================
def build_model(a, th_dim: int, cfg: dict | None = None):
    """按 CLI 建 v4 模型；若给定 cfg（来自 ckpt），**cfg 优先**。"""
    c = cfg or {}
    d_z = int(c.get("d_z", a.d_z))
    hidden = int(c.get("hidden", a.hidden))
    n_layers = int(c.get("n_layers", a.n_layers))
    tier = str(c.get("tier", getattr(a, "tier", "t2")))
    pred_attn = bool(c.get("pred_attn", getattr(a, "pred_attn", False)))
    n_heads = int(c.get("n_heads", getattr(a, "n_heads", 4)))
    m = M.build(d_z=d_z, hidden=hidden, n_layers=n_layers,
                n_factors=D2.N_FACTORS, u_dim=D2.U_FEAT, th_dim=th_dim,
                pred_attn=pred_attn, n_heads=n_heads,
                u_in_x=False, n_glob=D2.N_GLOB)
    n = m.n_params()
    log(f"模型 enc={n['enc']:,} pred={n['pred']:,} dec={n['dec']:,} "
        f"total={n['total']:,}  d_z={d_z} tier={tier} attn={pred_attn}")
    return m


def _torch(batch):
    """(s0, u_tok, u_glob, th, s_true, x) → torch。

    ⚠️ `u_tok` 是 (B,L,P_u,5)，Encoder 要 (B,L,Cu,Pu) ⇒ 必须 permute。
    """
    s0, u_tok, u_g, th, s_true, x = batch
    u = torch.from_numpy(np.ascontiguousarray(
        np.asarray(u_tok).transpose(0, 1, 3, 2)))          # (B,L,5,Pu)
    return (torch.from_numpy(np.asarray(s0)), u,
            torch.from_numpy(np.asarray(u_g)),
            torch.from_numpy(np.asarray(th)),
            torch.from_numpy(np.asarray(s_true)),
            torch.from_numpy(np.asarray(x)))


# ============================================================================== T1
def cmd_t1(a) -> int:
    log("T1 自编码器（v4）")
    safety.preflight(limit_mb=a.mem_mb)

    tr = D2.Dataset2(DATA, a.data, tier=a.tier)
    va = D2.Dataset2(DATA, a.val, tier=a.tier)
    log(f"训练集 {tr}\n       验证集 {va}")

    scaler = D2.compute_scaler2(tr, seed=0)
    os.makedirs(os.path.dirname(SCALER), exist_ok=True)
    with open(SCALER, "w", encoding="utf-8") as f:
        json.dump(scaler, f, ensure_ascii=False, indent=1)
    log(f"scaler → {SCALER}")
    log(f"  scale_s = {np.round(scaler['scale_s'], 4)}")
    log(f"  scale_u = {np.round(scaler['scale_u'], 6)}")

    s_tr = D2.Sampler2(tr, scaler, batch=a.batch, length=a.length, seed=a.seed)
    s_va = D2.Sampler2(va, scaler, batch=a.batch, length=a.length,
                       seed=a.val_seed + 100)

    model = build_model(a, int(tr.th.shape[1]))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)

    outdir = os.path.join(RES, a.run)
    os.makedirs(outdir, exist_ok=True)
    prog = progress.Progress(a.steps, desc=f"T1 {a.tier} dz={a.d_z}")
    best = float("inf")
    t0 = time.time()

    for step in range(1, a.steps + 1):
        s0, u, ug, th, s_true, x = _torch(s_tr.batch_pair())
        B, T, nf, P = s_true.shape
        s_flat = s0.unsqueeze(1).expand(B, T, nf, P)
        rec = model.reconstruct(s_flat, u, th, x, ug)
        loss = F.mse_loss(rec, s_flat)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                  a.grad_clip))
        opt.step()
        sch.step()
        prog.update(step, loss=f"{loss.item():.3e}", grad=f"{gn:.3f}")

        if step % 500 == 0 or step == a.steps:
            with torch.no_grad():
                sv, uv, ugv, thv, stv, xv = _torch(s_va.batch_pair())
                Bv, Tv, nfv, Pv = stv.shape
                sfv = sv.unsqueeze(1).expand(Bv, Tv, nfv, Pv)
                rv = model.reconstruct(sfv, uv, thv, xv, ugv)
                rel = float((rv - sfv).abs().mean())
            log(f"step {step}: train={loss.item():.3e}  val_rel={rel:.6f}")
            if rel < best:
                best = rel
                save_ckpt(os.path.join(outdir, "best.pt"), model,
                          step=step, val_rel=rel, scaler=scaler,
                          cfg=_cfg(a))

    el = time.time() - t0
    log(f"T1 完成 {el:.0f}s  最佳 val_rel={best:.6f}")
    print("=" * 70)
    print(f"T1 门禁：val_rel = {best:.6f}   判据 ≤ {a.rel_gate}")
    ok = best <= a.rel_gate
    print(f"  压缩比  {D2.N_FACTORS * tr.nseg} / {a.d_z} = "
          f"{D2.N_FACTORS * tr.nseg / a.d_z:.2f}×")
    print(f"  ⇒ {'✅ 通过' if ok else '❌ 未通过'}")
    return 0 if ok else 1


# ============================================================================== T2
def cmd_t2(a) -> int:
    log("T2 单步预测（Enc/Dec 冻结，v4）")
    safety.preflight(limit_mb=a.mem_mb)

    tr = D2.Dataset2(DATA, a.data, tier=a.tier)
    va = D2.Dataset2(DATA, a.val, tier=a.tier)
    scaler_path = getattr(a, "scaler", "") or SCALER
    with open(scaler_path, encoding="utf-8") as f:
        scaler = json.load(f)

    s_tr = D2.Sampler2(tr, scaler, batch=a.batch, length=a.length, seed=a.seed,
                       p_trial=_regime_p(tr, a.regime_weights),
                       hot_frac=a.hot_frac,
                       event_center_frac=a.event_center_frac,
                       event_center_thr=a.event_center_thr)
    s_va = D2.Sampler2(va, scaler, batch=a.batch, length=a.length,
                       seed=a.val_seed + 100)
    _log_sampling(s_tr, a, tr)
    if getattr(a, "event_center_frac", 0.0) > 0:
        log(f"  事件中心开窗：frac={a.event_center_frac:.2f} "
            f"thr={a.event_center_thr:g}（cai log10）")
    if getattr(a, "event_v_weight", 0.0) > 0:
        log(f"  V 事件加权：weight={a.event_v_weight:.1f} "
            f"tau={a.event_v_tau:.3f}")

    model = build_model(a, int(tr.th.shape[1]))
    ck = torch.load(os.path.join(RES, a.init), map_location="cpu",
                    weights_only=False)
    sd = {k: v for k, v in ck["model"].items()
          if k.startswith("enc.") or k.startswith("dec.")}
    model.load_state_dict(sd, strict=False)
    log(f"载入 T1: step={ck.get('step')} val_rel={ck.get('val_rel'):.6f}"
        f"（仅 enc/dec，{len(sd)} 个张量）")
    for p in list(model.enc.parameters()) + list(model.dec.parameters()):
        p.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"可训练参数 {n_train:,}（仅 Predictor）")

    opt = torch.optim.AdamW([p for p in model.parameters()
                             if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    outdir = os.path.join(RES, a.run)
    os.makedirs(outdir, exist_ok=True)
    prog = progress.Progress(a.steps, desc=f"T2 {a.tier}")
    best = float("inf")
    t0 = time.time()
    CAI = tr.meta["factor_names"].index("cai")      # 事件加权用
    VIX = tr.meta["factor_names"].index("V")        # V 事件加权用

    for step in range(1, a.steps + 1):
        s0, u, ug, th, s_true, x = _torch(s_tr.batch_pair())

        # 教师强制：z 全部来自真值
        with torch.no_grad():
            z_true = model.encode(s_true, u, th, x, ug)      # (B,T,dz)
        z_prev = torch.cat([z_true[:, :1], z_true[:, :-1]], dim=1)
        z_next = model.predict(z_prev, u, th, x, ug)

        d_next = model.decode(z_next, th, x)
        with torch.no_grad():
            d_prev = model.decode(z_prev, th, x)
        s_prev = torch.cat([s0.unsqueeze(1), s_true[:, :-1]], dim=1)
        s_hat = s_prev + (d_next - d_prev)
        # ---- ① 事件加权损失 ----
        #
        # 实测 98.6% 的帧是静息的，事件帧误差虽大 2.6 倍，但对 MSE 只贡献 3.7%
        # ⇒ 目标函数几乎不惩罚「事件学不准」。这里按 cai 变化幅度给权重。
        #
        # 注意：只有训练 loss 加权；日志/评测仍用未加权误差，保持可比。
        if a.event_weight > 0 or a.event_v_weight > 0:
            # s_hat[t] 对齐 s_true[t]（s_prev[t]=s_true[t-1]），故差分需 pad 到 L
            se = (s_hat - s_true).pow(2).mean(dim=(2, 3))       # (B,L)
            w = torch.ones_like(se)
            if a.event_weight > 0:
                d1 = (s_true[:, 1:, CAI, :]
                      - s_true[:, :-1, CAI, :]).abs().amax(dim=2)   # (B,L-1)
                dca = torch.cat([torch.zeros_like(d1[:, :1]), d1], dim=1)
                w = w + a.event_weight * torch.sigmoid(
                    dca / max(a.event_tau, 1e-6))
            if a.event_v_weight > 0:
                dv1 = (s_true[:, 1:, VIX, :]
                       - s_true[:, :-1, VIX, :]).abs().amax(dim=2)
                dv = torch.cat([torch.zeros_like(dv1[:, :1]), dv1], dim=1)
                w = w + a.event_v_weight * torch.sigmoid(
                    dv / max(a.event_v_tau, 1e-6))
            loss = (w * se).sum() / w.sum()
        else:
            loss = F.mse_loss(s_hat, s_true)

        # ---- latent 自洽正则（latent consistency）----
        #
        # 目的：把预测出的 z 约束在 encoder 学到的流形上，但**不改变预测目标**
        # （所以不像收缩映射那样有系统偏差）。
        #
        #   ŝ = Dec(ẑ)          → 再喂回 Encoder
        #   ẑ' = Enc(ŝ, u, θ)
        #   L_anchor = λ · |ẑ' − ẑ|²
        #
        # 若 ẑ 是「合法状态编码」，则编码-解码-再编码应保持恒等；
        # 漂移到流形外（实测闭环时 |z| 从 1.15 涨到 13.8）时该项会惩罚。
        #
        # ⚠️ 开销：多一次 Encoder 前向（Encoder 是最大模块）⇒ 每 N 步才算。
        if a.anchor_weight > 0 and (step % max(a.anchor_every, 1) == 0):
            s_rt = model.decode(z_next[:, 1:], th, x)          # (B,T-1,nf,P)
            z_rt = model.encode(s_rt, u[:, 1:], th, x, ug[:, 1:])
            anc = F.mse_loss(z_rt, z_next[:, 1:])
            loss = loss + a.anchor_weight * anc
            anchor_now = float(anc)
        else:
            anchor_now = float("nan")

        # ---- 闭环段 latent 自洽正则（③b）----
        #
        # 上面的 `anc` 只约束**教师强制路径**的 z_next；而真正会外逃的是闭环自己
        # 滚出来的 zc（实测 |z| 从 5.75 涨到 20.62，cos 掉到 0.20）。
        # 这里在闭环循环内再验一次「能不能被 encoder 认回来」。
        #
        # ⚠️ 只在训练里加。推理期绝不做硬投影：实测半径投影 0.46×、
        # roundtrip 投影在 k=200 差 7 倍，都更差。
        anc_cl_now = float("nan")

        # ---- 闭环尾损失 ----
        if a.cl_steps > 0:
            A = a.cl_stage_a_steps
            if A > 0:
                if step <= A:
                    cur_cl = a.cl_start
                    if step == 1:
                        log(f"课程阶段 A：[1,{A}] cl 固定 = {cur_cl}")
                else:
                    if step == A + 1:
                        log(f"课程阶段 B：({A},{a.steps}] cl "
                            f"{a.cl_start} → {a.cl_steps} 线性升")
                        if a.stage_b_lr > 0:
                            log(f"  阶段 B 学习率 {a.lr:.2e} → {a.stage_b_lr:.2e}")
                        save_ckpt(os.path.join(outdir, "stageA.pt"), model,
                                  step=step - 1, sigma=best, scaler=scaler,
                                  cfg=_cfg(a))
                        log(f"  已存阶段 A 末检查点 stageA.pt（σ={best:.6f}）")
                    frac = (step - A) / max(a.steps - A, 1)
                    cur_cl = int(round(a.cl_start
                                       + frac * (a.cl_steps - a.cl_start)))
            elif a.cl_grow_every > 0:
                cur_cl = min(a.cl_steps, a.cl_start
                             + (step // a.cl_grow_every) * a.cl_grow_by)
            else:
                cur_cl = a.cl_steps
            m = min(max(cur_cl, 2), s_true.shape[1] - 2)
            zc = z_true[:, 0]
            sc = s0.clone()
            cl_loss = 0.0
            anc_cl_sum, anc_cl_n = 0.0, 0
            for k in range(m):
                j = k + 1
                zc_next = model.predict(zc.unsqueeze(1), u[:, j].unsqueeze(1),
                                        th, x, ug[:, j].unsqueeze(1)
                                        ).squeeze(1)
                sc_next = model.step_residual(sc, zc, zc_next, th, x)
                cl_loss = cl_loss + F.mse_loss(sc_next, s_true[:, j])
                # ③b：闭环段自洽。乘 m 抵消外层 `/m`，使该项不吃 m 的稀释。
                if (a.anchor_cl > 0
                        and k % max(a.anchor_every, 1) == 0):
                    _zrt = model.encode(
                        model.decode(zc_next.unsqueeze(1), th, x),
                        u[:, j].unsqueeze(1), th, x,
                        ug[:, j].unsqueeze(1)).squeeze(1)
                    _a = F.mse_loss(_zrt, zc_next)
                    cl_loss = cl_loss + a.anchor_cl * _a * m
                    anc_cl_sum += float(_a)
                    anc_cl_n += 1
                sc, zc = sc_next, zc_next
            if anc_cl_n:
                anc_cl_now = anc_cl_sum / anc_cl_n
            # 闭环权重爬坡：阶段 A 就用**轻权重** cl_weight_start，
            # 阶段 B 内线性升到 cl_weight。若阶段 A 用满权重，进入阶段 B
            # 会突然跌回 cl_weight_start（不连续），实测会让长视界变差。
            if A > 0:
                if step <= A:
                    w_now = a.cl_weight_start
                else:
                    frac = (step - A) / max(a.steps - A, 1)
                    w_now = a.cl_weight_start + frac * (a.cl_weight
                                                        - a.cl_weight_start)
            else:
                w_now = a.cl_weight
            loss = loss + w_now * cl_loss / m
            cl_now = m

        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_now = a.grad_clip
        if a.cl_stage_a_steps > 0 and step > a.cl_stage_a_steps:
            if a.stage_b_lr > 0:
                for g in opt.param_groups:
                    g["lr"] = a.stage_b_lr
            if a.stage_b_grad_clip > 0:
                clip_now = a.stage_b_grad_clip
        gn = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], clip_now))
        opt.step()
        extra = dict(loss=f"{loss.item():.3e}", grad=f"{gn:.3f}")
        if a.cl_steps > 0:
            extra["cl"] = int(cl_now)
            extra["w"] = f"{w_now:.2f}"
        if not np.isnan(anchor_now):
            extra["anc"] = f"{anchor_now:.2e}"
        if not np.isnan(anc_cl_now):
            extra["ancC"] = f"{anc_cl_now:.2e}"
        prog.update(step, **extra)

        if step % 500 == 0 or step == a.steps:
            sig = eval_single_step(model, s_va)
            log(f"step {step}: loss={loss.item():.3e}  单步σ={sig:.6f}")
            if sig < best:
                best = sig
                save_ckpt(os.path.join(outdir, "best.pt"), model,
                          step=step, sigma=sig, scaler=scaler, cfg=_cfg(a))

    el = time.time() - t0
    save_ckpt(os.path.join(outdir, "last.pt"), model,
              step=a.steps, sigma=eval_single_step(model, s_va),
              scaler=scaler, cfg=_cfg(a))
    log(f"T2 完成 {el:.0f}s  最佳单步 σ={best:.6f}（另存 last.pt）")
    print("=" * 70)
    print(f"T2 门禁：单步 σ = {best:.6f}   判据 ≤ {a.sigma_gate}")
    ok = best <= a.sigma_gate
    print(f"  ⇒ {'✅ 通过' if ok else '❌ 未通过'}")
    return 0 if ok else 1


# ==============================================================================
@torch.no_grad()
def eval_single_step(model, sampler, n_batch: int = 4) -> float:
    """教师强制单步误差（残差形式，喂真值 latent）。

    ⚠️ 不能用 `model.step_residual`：它内部做 `[:, 0]`，只支持 (B,d_z)
    或 (B,1,d_z)。这里时间维长度为 T−1 > 1，必须直接算残差。
    """
    model.eval()
    tot, n = 0.0, 0
    for _ in range(n_batch):
        s0, u, ug, th, s_true, x = _torch(sampler.batch_pair())
        z = model.encode(s_true, u, th, x, ug)
        z_prev = torch.cat([z[:, :1], z[:, :-1]], dim=1)
        z_next = model.predict(z_prev, u, th, x, ug)
        # 预测 s_{t+1}：用 z_prev[t+1] 与 z_next[t+1] 的解码差
        zp = z_prev[:, 1:]
        zn = z_next[:, 1:]
        pred = s_true[:, :-1] + (model.decode(zn, th, x)
                                 - model.decode(zp, th, x))
        tgt = s_true[:, 1:]
        tot += float((pred - tgt).abs().mean()) * pred.shape[0]
        n += pred.shape[0]
    model.train()
    return tot / max(n, 1)


# ==============================================================================
def cmd_eval(a) -> int:
    """严格配对：同一批序列上，模型与 L0（复制上一帧）逐步对比。

    同时给**总体**与**分 regime** 两组数 —— 后者才是真正的证据，
    因为 linear 档占 50% 且近乎静态，会稀释总体 $a(k)$。
    """
    log("配对评测（v4）")
    safety.preflight(limit_mb=a.mem_mb)

    ds = D2.Dataset2(DATA, a.data, tier=a.tier)
    ck = torch.load(os.path.join(RES, a.ckpt), map_location="cpu",
                    weights_only=False)
    cfg = ck.get("cfg") or {}
    scaler = ck.get("scaler") or json.load(open(SCALER, encoding="utf-8"))
    model = build_model(a, int(ds.th.shape[1]), cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    log(f"载入 {a.ckpt}: step={ck.get('step')} "
        f"σ={ck.get('sigma', ck.get('val_rel'))}")

    sp = D2.Sampler2(ds, scaler, batch=a.n_seq, length=1, seed=0)
    ks = [int(k) for k in a.horizons.split(",")]
    L = max(ks) + 1

    tot = {k: [0.0, 0.0] for k in ks}          # [model, L0]
    diverged: dict[int, bool] = {}
    by_reg = {r: {k: [0.0, 0.0] for k in ks}
              for r in ("linear", "inter", "plateau")}
    cnt = 0
    cnt_reg = {r: 0 for r in by_reg}

    n_used = min(a.n_seq * a.n_rep, ds.N)
    for off in range(0, n_used, a.n_seq):
        n = min(a.n_seq, n_used - off)
        probe = D2.Sampler2(ds, scaler, batch=n, length=1, seed=0)
        s0, u_tok, u_g, th, s_true, x = probe.batch_window(
            a.t0, L, n, off=off)
        u = torch.from_numpy(np.ascontiguousarray(u_tok.transpose(0, 1, 3, 2)))
        s0t = torch.from_numpy(s0)
        tht = torch.from_numpy(th)
        ug = torch.from_numpy(u_g)
        xt = torch.from_numpy(x)
        outs = model.rollout(s0t, u, tht, xt, ug_seq=ug)   # (B,L,nf,P)
        st = torch.from_numpy(s_true)

        regs = ds.regimes[(np.arange(n) + off) % ds.N]
        for k in ks:
            ol = outs[:, k - 1]
            # 数值熔断：归一化空间里 |out| 超过 SANE_MAX 就不能当「模型能力」解读。
            # （实测 noLN/last 在 k=4000 达到 4.5e15；不标记会被误读成真实误差）
            ol_np = ol.detach().cpu().numpy()
            if (~np.isfinite(ol_np)).any() or np.abs(ol_np).max() > SANE_MAX:
                diverged[k] = True
            e = float((ol - st[:, k]).abs().mean())
            e0 = float((s0t - st[:, k]).abs().mean())
            tot[k][0] += e * n
            tot[k][1] += e0 * n
            for r in by_reg:
                msk = (regs == r)
                j = int(msk.sum())
                if j:
                    # 分档只统计该档样本
                    e_r = float((outs[msk, k - 1] - st[msk, k]).abs().mean())
                    e0_r = float((s0t[msk] - st[msk, k]).abs().mean())
                    by_reg[r][k][0] += e_r * j
                    by_reg[r][k][1] += e0_r * j
        for r in by_reg:
            cnt_reg[r] += int((regs == r).sum())
        cnt += n

    sigma = ck.get("sigma", 0.0) or 0.0

    def table(rows_map, ndata, title):
        print(f"\n{title}  (n={ndata})")
        print(f"{'k':>6} {'ms':>8} {'模型':>10} {'L0':>10} "
              f"{'模型/L0':>9} {'a(k)':>8}  备注")
        out = {}
        for k in ks:
            e, e0 = rows_map[k]
            e, e0 = e / max(ndata, 1), e0 / max(ndata, 1)
            ratio = e / max(e0, 1e-12)
            ak = e / max(np.sqrt(k) * sigma, 1e-12) if sigma else float("nan")
            tag = "⚠️ DIVERGED（数值爆炸，不作能力解读）" if diverged.get(k) else ""
            print(f"{k:>6} {k * ds.dt:>8.1f} {e:>10.4f} {e0:>10.4f} "
                  f"{ratio:>9.3f} {ak:>8.2f}  {tag}")
            out[k] = dict(model=e, L0=e0, ratio=ratio, a=ak,
                          diverged=bool(diverged.get(k)))
        return out

    res = dict(ckpt=a.ckpt, tier=a.tier, data=a.data, n=cnt,
               sigma=sigma, horizons=ks)
    res["counts"] = dict(overall=cnt, **{r: cnt_reg[r] for r in by_reg})
    res["overall"] = table(tot, cnt, "总体")
    for r in by_reg:
        if cnt_reg[r]:
            res[r] = table(by_reg[r], cnt_reg[r], f"regime = {r}")

    outp = os.path.join(ROOT, "results", "reports",
                        f"eval_{os.path.basename(a.ckpt).replace('.pt', '')}"
                        f"_{a.tier}_{a.data}.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n⇒ {outp}")

    # ---- 成败判据（DESIGN_v4 §9：a(200) < 1.0）----
    k200 = 200 if 200 in res["overall"] else ks[-1]
    ok = res["overall"][k200]["ratio"] < 1.0
    print(f"判据 a({k200}) = {res['overall'][k200]['ratio']:.3f} < 1.0  "
          f"⇒ {'✅ 优于 L0' if ok else '❌ 未超过 L0'}")
    return 0 if ok else 1


# ==============================================================================
def common(p, run: str, seed: int, tier_default: str = "t2"):
    p.add_argument("--data", type=str, default="tr_s1")
    p.add_argument("--val", type=str, default="va_s2")
    p.add_argument("--tier", type=str, default=tier_default,
                   choices=["t1", "t2"],
                   help="t2=28 token 全分辨率；t1=14 token 按 section 求和")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--d-z", type=int, default=16)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mem-mb", type=float, default=1600.0)
    p.add_argument("--seed", type=int, default=seed)
    p.add_argument("--val-seed", type=int, default=2)
    p.add_argument("--run", type=str, default=run)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="world.learn.train2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("t1")
    common(p1, "t1v4_dz16", 1)
    p1.add_argument("--rel-gate", type=float, default=0.05)
    p1.set_defaults(func=cmd_t1)

    p2 = sub.add_parser("t2")
    common(p2, "t2v4_baseline", 1)
    p2.add_argument("--init", type=str, default="t1v4_dz16/best.pt")
    p2.add_argument("--sigma-gate", type=float, default=0.02)
    p2.add_argument("--cl-steps", type=int, default=10)
    p2.add_argument("--cl-weight", type=float, default=1.0)
    p2.add_argument("--cl-start", type=int, default=4)
    p2.add_argument("--cl-grow-every", type=int, default=0)
    p2.add_argument("--cl-grow-by", type=int, default=1)
    p2.add_argument("--cl-stage-a-steps", type=int, default=0)
    p2.add_argument("--cl-weight-start", type=float, default=0.0,
                    help="阶段 B 起始闭环权重（线性升到 --cl-weight）")
    p2.add_argument("--anchor-weight", type=float, default=0.0,
                    help="latent 自洽正则权重 λ（0=关闭）")
    p2.add_argument("--anchor-every", type=int, default=4,
                    help="每 N 步算一次自洽正则（省 Encoder 开销）")
    p2.add_argument("--anchor-cl", type=float, default=0.0,
                    help="闭环段 latent 自洽正则权重（0=关闭）。"
                         "与 --anchor-weight 不同：后者只约束教师强制路径，"
                         "本项约束闭环自产的 zc（实测 |z| 外逃处）")
    p2.add_argument("--event-weight", type=float, default=0.0,
                    help="① 事件加权强度：0=关闭。事件帧权重 = 1+该值")
    p2.add_argument("--event-tau", type=float, default=0.2,
                    help="① sigmoid 敏感度（归一化单位的 |Δcai| 尺度）")
    p2.add_argument("--event-v-weight", type=float, default=0.0,
                    help="①b V 事件加权强度：0=关闭。V 大幅变化帧额外加权")
    p2.add_argument("--event-v-tau", type=float, default=0.02,
                    help="①b sigmoid 敏感度（归一化单位的 |ΔV| 尺度）")
    p2.add_argument("--event-center-frac", type=float, default=0.0,
                    help="③c batch 内以 Ca episode 中心开窗的比例（0=关闭）")
    p2.add_argument("--event-center-thr", type=float, default=-3.1,
                    help="③c Ca episode 判据（cai 的 log10 值）")
    p2.add_argument("--scaler", type=str, default="",
                    help="归一化 scaler JSON 路径；空=results/data/scaler_v4.json")
    p2.add_argument("--regime-weights", type=str, default="",
                    help="② 每 trial 抽样概率 'linear,inter,plateau'，"
                         "如 '0.25,0.35,0.40'。空=均匀。仅训练用")
    p2.add_argument("--hot-frac", type=float, default=0.0,
                    help="③a batch 内强制含事件 trial 的比例（0=关闭）")
    p2.add_argument("--stage-b-lr", type=float, default=0.0)
    p2.add_argument("--stage-b-grad-clip", type=float, default=0.0)
    p2.add_argument("--pred-attn", action="store_true")
    p2.add_argument("--n-heads", type=int, default=4)
    p2.set_defaults(func=cmd_t2)

    pe = sub.add_parser("eval")
    pe.add_argument("--ckpt", type=str, default="t2v5_events/last.pt")
    pe.add_argument("--data", type=str, default="te_s3")
    pe.add_argument("--tier", type=str, default="t2", choices=["t1", "t2"])
    pe.add_argument("--horizons", type=str,
                    default="10,40,200,1000,4000,8000")
    pe.add_argument("--n-seq", type=int, default=64)
    pe.add_argument("--n-rep", type=int, default=4)
    pe.add_argument("--t0", type=int, default=4000)
    pe.add_argument("--d-z", type=int, default=16)
    pe.add_argument("--hidden", type=int, default=192)
    pe.add_argument("--n-layers", type=int, default=2)
    pe.add_argument("--mem-mb", type=float, default=1600.0)
    pe.set_defaults(func=cmd_eval)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())

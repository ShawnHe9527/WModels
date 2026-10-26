"""训练：T1 自编码器 → T2 单步预测 → 闭环评测。

## 阶段

| 命令 | 内容 | 门禁 |
|---|---|---|
| `t1` | 自编码器：重构 $s_t$（$9\times P$）| `rel_err`；并验证 $d_z$ 信息量 |
| `t2` | 冻结 Enc/Dec，训 Predictor（教师强制）| 单步 $\sigma$ |
| `eval` | 闭环配对评测 vs L0 | 放大比 $a(k)$ |

## 损失

所有张量已在归一化空间（逐因子除以自身 std）⇒ 直接用 MSE，
天然是"逐因子方差归一化"，不会被 $V$ 主导。

## 用法

    cd WModel2 && PYTHONPATH=$PWD python -m world.learn.train t1 \
        --data tr_s1 --val va_s2 --steps 4000 --d-z 8 --run t1_dz8
"""
from __future__ import annotations

import argparse
import json
import os
import time

from ..core import safety
from ..core import progress

THREADS = safety.apply_thread_caps(6)

import numpy as np                      # noqa: E402
import torch                            # noqa: E402
import torch.nn.functional as F         # noqa: E402

from ..core import config as C          # noqa: E402
from . import data as D                 # noqa: E402
from . import metrics as M              # noqa: E402
from . import model as M2               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results", "checkpoints")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cfg(a: argparse.Namespace) -> dict:
    """剔除 func（否则 torch.save 会把模块引用一起 pickle，跨进程加载失败）。"""
    d = {k: v for k, v in vars(a).items() if k != "func"}
    return d


def save_ckpt(path: str, model, **kw) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(model=model.state_dict(), **kw), path)


# ==============================================================================
def build_model(a, th_dim: int, nseg: int, cfg: dict | None = None):
    """按 CLI 参数建模；若给定 cfg（来自 checkpoint），**cfg 优先**。

    ⚠️ 必须用 ckpt 里的 cfg 重建 —— 否则 MLP 版与 cross-attention 版的
       state_dict 键名/形状不兼容，加载直接报错。
    """
    c = cfg or {}
    d_z = int(c.get("d_z", a.d_z))
    hidden = int(c.get("hidden", a.hidden))
    n_layers = int(c.get("n_layers", a.n_layers))
    pred_attn = bool(c.get("pred_attn", getattr(a, "pred_attn", False)))
    n_heads = int(c.get("n_heads", getattr(a, "n_heads", 4)))
    m = M2.build(d_z=d_z, hidden=hidden, n_layers=n_layers,
                 n_factors=D.N_FACTORS, u_dim=D.U_CH, th_dim=th_dim,
                 pred_attn=pred_attn, n_heads=n_heads)
    n = m.n_params()
    log(f"模型参数 enc={n['enc']:,} pred={n['pred']:,} dec={n['dec']:,} "
        f"total={n['total']:,}  d_z={d_z} attn={pred_attn}")
    return m


def batch_to_torch(s0, u, th, s_true, x):
    return (torch.from_numpy(np.asarray(s0)),
            torch.from_numpy(np.asarray(u)),
            torch.from_numpy(np.asarray(th)),
            torch.from_numpy(np.asarray(s_true)),
            torch.from_numpy(np.asarray(x)))


# ============================================================================== T1
def cmd_t1(a: argparse.Namespace) -> int:
    """自编码器：从 s_t（+u_t）重构 s_t。"""
    log("T1 自编码器")
    safety.preflight(limit_mb=a.mem_mb)

    tr = D.Dataset(DATA, a.data)
    va = D.Dataset(DATA, a.val)
    log(f"训练集 {tr}   验证集 {va}")

    scaler = D.compute_scaler(tr, seed=0)
    sdir = os.path.join(ROOT, "results", "data")
    os.makedirs(sdir, exist_ok=True)
    spath = os.path.join(sdir, "scaler.json")
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(scaler, f, ensure_ascii=False, indent=1)
    log(f"scaler → {spath}")
    log(f"  scale_s = {np.round(scaler['scale_s'], 4)}")

    s_tr = D.Sampler(tr, scaler, batch=a.batch, length=a.length, seed=a.seed)
    s_va = D.Sampler(va, scaler, batch=a.batch, length=a.length,
                     seed=a.val_seed + 100)

    model = build_model(a, int(tr.th.shape[1]), tr.nseg)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)

    outdir = os.path.join(RES, a.run)
    os.makedirs(outdir, exist_ok=True)
    prog = progress.Progress(a.steps, desc=f"T1 dz={a.d_z}")
    best = float("inf")
    t0 = time.time()

    for step in range(1, a.steps + 1):
        s0, u, th, s_true, x = batch_to_torch(*s_tr.batch_pair())
        # 逐帧重构（每帧用自己的 u）
        B, T, nf, P = s_true.shape      # ⚠️ 因子维从 s 取（u 只有 3 通道）
        s_flat = s0.unsqueeze(1).expand(B, T, nf, P)
        rec = model.reconstruct(s_flat, u, th, x)
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
                sv, uv, thv, stv, xv = batch_to_torch(*s_va.batch_pair())
                Bv, Tv, nfv, Pv = stv.shape
                sfv = sv.unsqueeze(1).expand(Bv, Tv, nfv, Pv)
                rv = model.reconstruct(sfv, uv, thv, xv)
                rel = float((rv - sfv).abs().mean())
            log(f"step {step}: train={loss.item():.3e}  val_rel={rel:.6f}")
            if rel < best:
                best = rel
                save_ckpt(os.path.join(outdir, "best.pt"), model,
                          step=step, val_rel=rel, scaler=scaler,
                          cfg=_cfg(a), d_z=a.d_z)

    el = time.time() - t0
    log(f"T1 完成 {el:.0f}s  最佳 val_rel={best:.6f}")
    print("=" * 70)
    print(f"T1 门禁：val_rel = {best:.6f}")
    ok = best <= a.rel_gate
    print(f"  判据 ≤ {a.rel_gate}  ⇒ {'✅ 通过' if ok else '❌ 未通过'}")
    print(f"  压缩比  {D.N_FACTORS * tr.nseg} / {a.d_z} = "
          f"{D.N_FACTORS * tr.nseg / a.d_z:.2f}×")
    return 0 if ok else 1


# ============================================================================== T2
def cmd_t2(a: argparse.Namespace) -> int:
    """冻结 Enc/Dec，训 Predictor（教师强制单步）。"""
    log("T2 单步预测（Enc/Dec 冻结）")
    safety.preflight(limit_mb=a.mem_mb)

    tr = D.Dataset(DATA, a.data)
    va = D.Dataset(DATA, a.val)
    with open(os.path.join(ROOT, "results", "data", "scaler.json"),
              encoding="utf-8") as f:
        scaler = json.load(f)

    s_tr = D.Sampler(tr, scaler, batch=a.batch, length=a.length, seed=a.seed)
    s_va = D.Sampler(va, scaler, batch=a.batch, length=a.length,
                     seed=a.val_seed + 100)

    model = build_model(a, int(tr.th.shape[1]), tr.nseg)
    ck = torch.load(os.path.join(RES, a.init), map_location="cpu",
                    weights_only=False)
    # ⚠️ 只加载 enc / dec —— T1 的 Predictor 权重可能来自不同结构
    #    （例如 MLP 版 vs cross-attention 版），形状不兼容。
    sd = {k: v for k, v in ck["model"].items()
          if k.startswith("enc.") or k.startswith("dec.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
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
    prog = progress.Progress(a.steps, desc="T2")
    best = float("inf")
    t0 = time.time()

    for step in range(1, a.steps + 1):
        s0, u, th, s_true, x = batch_to_torch(*s_tr.batch_pair())

        # 教师强制：每步都从**真值** latent 出发（反向深度 = 1）
        with torch.no_grad():
            z_true = model.encode(s_true, u, th, x)      # (B,T,dz)
        z_prev = torch.cat([z_true[:, :1], z_true[:, :-1]], dim=1)
        z_next = model.predict(z_prev, u, th, x)

        # 残差输出：ŝ_{t+1} = s_t + [Dec(z_{t+1}) − Dec(z_t)]
        d_next = model.decode(z_next, th, x)
        with torch.no_grad():
            d_prev = model.decode(z_prev, th, x)
        s_prev = torch.cat([s0.unsqueeze(1), s_true[:, :-1]], dim=1)
        s_hat = s_prev + (d_next - d_prev)
        loss = F.mse_loss(s_hat, s_true)

        # ---- 闭环尾损失（治暴露偏差）----
        # 教师强制下 z_prev 永远是**真值** latent，Predictor 从未见过自己的漂移。
        # 实测：只做教师强制时单步 σ=0.002 但闭环 k=20 误差 0.14（放大比 15×）。
        # 做法：从真值 latent 起，之后只喂自己的 latent 滚 m 步（不重编码）。
        # ⚠️ m 必须小（8–12）：长链在训练早期会把梯度放大到发散。
        if a.cl_steps > 0:
            # ================= 闭环窗长度策略 =================
            # 三种模式（优先级从上到下）：
            #
            # ① 两阶段课程（cl_stage_a_steps > 0）★ 方案 3
            #    阶段 A [1, A]      : cl 固定 = cl_start，专心把**局部精度**练到收敛
            #    阶段 B (A, steps]  : cl 从 cl_start 线性升到 cl_steps，学长程
            #    理由（实测）：旧版"从第 0 步就连续加长"会让长视界**变差**
            #      （cl 课程 4→64 的 k=1000 误差 3.20，而 cl=10 只有 1.35）——
            #      因为早期模型弱，被逼着预测 60 步外会让梯度乱飞，
            #      把已经学好的局部精度破坏掉。
            #
            # ② 连续加长（cl_grow_every > 0）：旧的方案，作为对照保留
            # ③ 固定（其余）：cl = cl_steps
            A = a.cl_stage_a_steps
            if A > 0:
                if step <= A:
                    cur_cl = a.cl_start
                    if step == 1:
                        log(f"课程阶段 A：[1,{A}] cl 固定 = {cur_cl}")
                else:
                    if step == A + 1:
                        log(f"课程阶段 B：({A},{a.steps}] cl {a.cl_start} → {a.cl_steps} 线性升")
                        if a.stage_b_lr > 0:
                            log(f"  阶段 B 学习率 {a.lr:.2e} → {a.stage_b_lr:.2e}")
                        if a.stage_b_grad_clip > 0:
                            log(f"  阶段 B 梯度裁剪 {a.grad_clip} → {a.stage_b_grad_clip}")
                        _ = 0
                        save_ckpt(os.path.join(outdir, "stageA.pt"), model,
                                  step=step - 1, sigma=best, scaler=scaler,
                                  cfg=_cfg(a), d_z=a.d_z)
                        log(f"  已存阶段 A 末检查点 stageA.pt（σ={best:.6f}）")
                        log("  ⚠️ 注意：best.pt 按 σ 选优，")
                        log("     阶段 B 会牺牲单步精度换长视界 ⇒ 评测时用 last.pt")
                    frac = (step - A) / max(a.steps - A, 1)
                    cur_cl = int(round(a.cl_start
                                       + frac * (a.cl_steps - a.cl_start)))
            elif a.cl_grow_every > 0:
                cur_cl = min(a.cl_steps,
                             a.cl_start + (step // a.cl_grow_every) * a.cl_grow_by)
            else:
                cur_cl = a.cl_steps
            m = min(max(cur_cl, 2), s_true.shape[1] - 2)
            zc = z_true[:, 0]                       # 从真值 latent 起
            sc = s0.clone()
            cl_loss = 0.0
            for k in range(m):
                zc_next = model.predict(zc.unsqueeze(1),
                                        u[:, k + 1].unsqueeze(1),
                                        th, x).squeeze(1)
                sc_next = model.step_residual(sc, zc, zc_next, th, x)
                cl_loss = cl_loss + F.mse_loss(sc_next, s_true[:, k + 1])
                sc, zc = sc_next, zc_next
            loss = loss + a.cl_weight * cl_loss / m
            cl_now = m

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # 阶段的 lr / grad-clip 可能不同（阶段 B 需要更低 lr 才稳）
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
        _extra = dict(loss=f"{loss.item():.3e}", grad=f"{gn:.3f}")
        if a.cl_steps > 0:
            _extra["cl"] = int(cl_now)
        prog.update(step, **_extra)

        if step % 500 == 0 or step == a.steps:
            sig = eval_single_step(model, s_va)
            log(f"step {step}: loss={loss.item():.3e}  单步σ={sig:.6f}")
            if sig < best:
                best = sig
                save_ckpt(os.path.join(outdir, "best.pt"), model,
                          step=step, sigma=sig, scaler=scaler, cfg=_cfg(a),
                          d_z=a.d_z)

    el = time.time() - t0
    # ⚠️ 必须存 last.pt：本任务的"最优"不等于"单步 σ 最小"。
    #    阶段 B 专门练长视界，会**牺牲**一点单步精度来换长视界；
    #    若只按 σ 选优，会永远选中阶段 A 的模型，方案 3 的收益就被漏掉。
    save_ckpt(os.path.join(outdir, "last.pt"), model,
              step=a.steps, sigma=eval_single_step(model, s_va),
              scaler=scaler, cfg=_cfg(a), d_z=a.d_z)
    log(f"T2 完成 {el:.0f}s  最佳单步 σ={best:.6f}（另存 last.pt）")
    print("=" * 70)
    print(f"T2 门禁：单步 σ = {best:.6f}   判据 ≤ {a.sigma_gate}")
    ok = best <= a.sigma_gate
    print(f"  ⇒ {'✅ 通过' if ok else '❌ 未通过'}")
    return 0 if ok else 1


@torch.no_grad()
def eval_single_step(model, sampler, n_batch: int = 4) -> float:
    """教师强制单步误差（残差形式，喂真值 latent）。"""
    model.eval()
    errs = []
    for _ in range(n_batch):
        s0, u, th, s_true, x = batch_to_torch(*sampler.batch_pair())
        z_true = model.encode(s_true, u, th, x)
        z_prev = torch.cat([z_true[:, :1], z_true[:, :-1]], dim=1)
        z_next = model.predict(z_prev, u, th, x)
        s_prev = torch.cat([s0.unsqueeze(1), s_true[:, :-1]], dim=1)
        s_hat = s_prev + (model.decode(z_next, th, x)
                          - model.decode(z_prev, th, x))
        errs.append(float((s_hat - s_true).abs().mean()))
    model.train()
    return float(np.mean(errs))


# ============================================================================== eval
def cmd_eval(a: argparse.Namespace) -> int:
    """闭环配对评测 vs L0（同一批样本）。"""
    log("闭环配对评测 vs L0")
    safety.preflight(limit_mb=a.mem_mb)

    ds = D.Dataset(DATA, a.data)
    ck = torch.load(os.path.join(RES, a.ckpt), map_location="cpu",
                    weights_only=False)
    scaler = ck["scaler"]
    model = build_model(a, int(ds.th.shape[1]), ds.nseg, cfg=ck.get("cfg"))
    model.load_state_dict(ck["model"])
    model.eval()
    log(f"载入 {a.ckpt}  step={ck.get('step')}  "
        f"σ={ck.get('sigma', ck.get('val_rel'))}")

    ks = [int(v) for v in a.horizons.split(",")]
    # ‼️ 要 max(k)+1 个真值帧：outs[t] = ŝ[t0+t+1]，视界 k 需比对 s[t0+k]
    L = max(ks) + 1
    sampler = D.Sampler(ds, scaler, batch=a.n_seq, length=L, seed=a.seed)

    rows, amps = [], []
    sigma = float(ck.get("sigma", 0.0)) or eval_single_step(model, sampler)

    for rep in range(a.n_rep):
        s0, u, th, s_true, x = batch_to_torch(*sampler.batch_pair())
        with torch.no_grad():
            out = model.rollout(s0, u, th, x, use_residual=True)
        for k in ks:
            # outs[t] = ŝ[t0+t+1] ⇒ 视界 k 对应 outs[:, k-1]，真值 s[t0+k]
            m = float((out[:, k - 1] - s_true[:, k]).abs().mean())
            l0 = float((s0 - s_true[:, k]).abs().mean())
            rows.append((k, k * ds.dt, m, l0))
            amps.append(M.amplification(m, k, sigma))

    # 汇总
    agg = {}
    for k, ms, m, l0 in rows:
        agg.setdefault(k, []).append((m, l0))
    print("=" * 74)
    print(f"闭环评测（n={a.n_seq}×{a.n_rep}，配对，σ={sigma:.6f}）")
    print(f"{'k':>6} {'ms':>7} {'模型':>10} {'L0':>10} {'模型/L0':>9} "
          f"{'放大a':>7} 判定")
    wins = 0
    for k in ks:
        ms_ = np.mean([v[0] for v in agg[k]])
        l0_ = np.mean([v[1] for v in agg[k]])
        a_ = np.mean([amps[i] for i, r in enumerate(rows) if r[0] == k])
        win = ms_ < l0_
        wins += int(win)
        print(f"{k:>6} {k*ds.dt:>7.1f} {ms_:>10.5f} {l0_:>10.5f} "
              f"{ms_/max(l0_,1e-12):>9.2f} {a_:>7.2f} "
              f"{'✅ 赢' if win else '❌ 输'}")
    print("=" * 74)
    print(f"胜出 {wins}/{len(ks)} 个视界")
    print("放大比 a=ε/(√k·σ)：a≈1 随机游走；a≫1 且随 k 上升 ⇒ latent 发散")

    os.makedirs(os.path.join(ROOT, "results", "reports"), exist_ok=True)
    outp = os.path.join(ROOT, "results", "reports", f"eval_{a.run}.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(dict(ckpt=a.ckpt, sigma=sigma,
                       rows=[dict(k=k, ms=ms, model=m, L0=l0, a=aa)
                             for (k, ms, m, l0), aa in zip(rows, amps)],
                       wins=wins), f, ensure_ascii=False, indent=1)
    print(f"⇒ {outp}")
    return 0


# ==============================================================================
def common(p, run: str, seed: int):
    p.add_argument("--data", type=str, default="tr_s1")
    p.add_argument("--val", type=str, default="va_s2")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--length", type=int, default=64,
                   help="窗口长度（步）；dt=0.025 ⇒ 64 步 = 1.6 ms")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--d-z", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mem-mb", type=float, default=1600.0)
    p.add_argument("--seed", type=int, default=seed)
    p.add_argument("--val-seed", type=int, default=2)
    p.add_argument("--run", type=str, default=run)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="world.learn.train")
    sub = ap.add_subparsers(required=True)

    p1 = sub.add_parser("t1", help="自编码器")
    common(p1, "t1", 1)
    p1.add_argument("--rel-gate", type=float, default=0.05)
    p1.set_defaults(func=cmd_t1)

    p2 = sub.add_parser("t2", help="单步预测")
    common(p2, "t2", 1)
    p2.add_argument("--init", type=str, default="t1_dz8/best.pt")
    p2.add_argument("--sigma-gate", type=float, default=0.02)
    p2.add_argument("--cl-steps", type=int, default=10,
                    help="闭环尾损失滚动步数的**上限**（0=纯教师强制）")
    p2.add_argument("--cl-weight", type=float, default=1.0)
    p2.add_argument("--cl-start", type=int, default=4,
                    help="课程式闭环窗的起始步数")
    p2.add_argument("--cl-grow-every", type=int, default=0,
                    help="每多少步加长一次（0=不做课程式）")
    p2.add_argument("--cl-stage-a-steps", type=int, default=0,
                    help="★两阶段课程：前 A 步 cl 固定在 cl_start（0=关闭）")
    p2.add_argument("--stage-b-lr", type=float, default=0.0,
                    help="阶段 B 的独立学习率（0=沿用 --lr）。"
                         "实测：lr=8e-4 配 cl=200 会让 σ 从 0.0044 退化到 0.0097，"
                         "日志里 grad 峰值 13.1 ⇒ 长链 BPTT 不稳")
    p2.add_argument("--stage-b-grad-clip", type=float, default=0.0,
                    help="阶段 B 的独立梯度裁剪（0=沿用 --grad-clip）")
    p2.add_argument("--cl-grow-by", type=int, default=1)
    p2.add_argument("--pred-attn", action="store_true",
                    help="Predictor 用 cross-attention（z 作 query、突触作 kv）")
    p2.add_argument("--n-heads", type=int, default=4)
    p2.set_defaults(func=cmd_t2)

    pe = sub.add_parser("eval", help="闭环配对评测")
    common(pe, "eval", 1234)
    pe.add_argument("--ckpt", type=str, default="t2/best.pt")
    pe.add_argument("--horizons", type=str, default="10,40,200,1000,4000")
    pe.add_argument("--n-seq", type=int, default=16)
    pe.add_argument("--n-rep", type=int, default=2)
    pe.set_defaults(func=cmd_eval)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())

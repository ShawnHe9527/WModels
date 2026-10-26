"""生成全部图表 + 每图的中文说明。

## 用法

    cd WModel2 && PYTHONPATH=$PWD python -m world.viz.report_runner \
        --ckpt t2_dz8/best.pt --data te_s3 --out results/figures

## 产出

| 文件 | 内容 |
|---|---|
| `fig_rollout.png/.pdf/.md` | 单条协议：真值 / 模型 / L0 |
| `fig_horizon.*` | 误差 vs 视界（配对）|
| `fig_amplification.*` | 放大比 a(k) |
| `fig_latent_pca.*` | latent 轨迹 PCA |
| `fig_spatial.*` | 某时刻空间剖面 |
| `fig_inputs.*` | 输入电导 + 边界电位 |
"""
from __future__ import annotations

import argparse
import json
import os

from ..core import safety
THREADS = safety.apply_thread_caps(6)

import numpy as np                    # noqa: E402
import torch                          # noqa: E402

from ..learn import data as D         # noqa: E402
from ..learn import metrics as M      # noqa: E402
from ..learn import model as M2       # noqa: E402
from . import report as R             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results", "checkpoints")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="world.viz.report_runner")
    ap.add_argument("--ckpt", type=str, default="t2_dz8/best.pt")
    ap.add_argument("--data", type=str, default="te_s3")
    ap.add_argument("--out", type=str, default="results/figures")
    ap.add_argument("--horizons", type=str, default="10,40,200,1000,4000,8000")
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--n-rep", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--mem-mb", type=float, default=1500.0)
    a = ap.parse_args(argv)

    safety.preflight(limit_mb=a.mem_mb)
    outdir = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)

    ds = D.Dataset(DATA, a.data)
    ck = torch.load(os.path.join(RES, a.ckpt), map_location="cpu",
                    weights_only=False)
    scaler = ck["scaler"]
    cfg = ck.get("cfg", {}) or {}
    model = M2.build(d_z=ck.get("d_z", cfg.get("d_z", 8)),
                     hidden=cfg.get("hidden", 128),
                     n_layers=cfg.get("n_layers", 2),
                     n_factors=D.N_FACTORS, u_dim=D.U_CH,
                     th_dim=int(ds.th.shape[1]))
    model.load_state_dict(ck["model"])
    model.eval()
    sigma = float(ck.get("sigma") or ck.get("val_rel") or 0.0)
    print(f"载入 {a.ckpt}  step={ck.get('step')}  σ={sigma:.6f}  "
          f"d_z={ck.get('d_z')}")

    ks = [int(v) for v in a.horizons.split(",")]
    L = max(ks) + 1
    sampler = D.Sampler(ds, scaler, batch=a.n_seq, length=L, seed=a.seed)

    # ---- 跑一次，同时收齐所有图需要的数据 ----
    base = f"{a.data}, n={a.n_seq}×{a.n_rep}, seed={a.seed}"
    rows, amps = [], []
    for rep in range(a.n_rep):
        s0, u, th, s_true, x = sampler.batch_pair()
        t0 = (torch.from_numpy(np.asarray(s0)),
              torch.from_numpy(np.asarray(u)),
              torch.from_numpy(np.asarray(th)),
              torch.from_numpy(np.asarray(x)))
        with torch.no_grad():
            out = model.rollout(t0[0], t0[1], t0[2], t0[3],
                                use_residual=True)
            z_true = model.encode(t0[0].unsqueeze(1), t0[1][:, :1],
                                  t0[2], t0[3]).squeeze(1)
            z = z_true
            zs = [z_true]
            for t in range(min(L, 400)):
                z = model.predict(z.unsqueeze(1), t0[1][:, t].unsqueeze(1),
                                  t0[2], t0[3]).squeeze(1)
                zs.append(z)
            z_roll = torch.stack(zs, dim=1)[0].numpy()

        for k in ks:
            st_k = np.asarray(s_true[:, k])
            m = float(np.abs(out[:, k - 1].numpy() - st_k).mean())
            l0 = float(np.abs(np.asarray(s0) - st_k).mean())
            rows.append(dict(k=k, ms=k * ds.dt, model=m, L0=l0))
            amps.append(dict(k=k, a=M.amplification(m, k, sigma)))

        if rep == 0:
            # 单条协议的完整轨迹（用于 rollout / spatial / pca / inputs）
            k_max = min(L - 1, out.shape[1])
            i = 0
            s_t = np.asarray(s_true[i, 1:k_max + 1])     # (K,f,P)
            s_m = out[i, :k_max].numpy()
            s_l = np.repeat(np.asarray(s0[i])[None], k_max, axis=0)
            t_ms = np.arange(k_max) * ds.dt
            fnames = ds.meta["factor_names"]
            u1 = np.asarray(u[i, :k_max])
            ub = np.asarray(ds.ub)
            # 取该 trial 对应的时间片（这里用第 0 个 trial 的前 K 步近似）
            ub_slice = ub[0, :k_max]

    # 汇总 rows（多 rep 平均，配对）
    agg = {}
    for r in rows:
        agg.setdefault(r["k"], []).append((r["model"], r["L0"]))
    rows_avg = []
    for k in ks:
        v = agg[k]
        rows_avg.append(dict(k=k, ms=k * ds.dt,
                             model=float(np.mean([x[0] for x in v])),
                             L0=float(np.mean([x[1] for x in v]))))

    # ---------------- 出图 ----------------
    cap_common = (f"\n## 复现\n\n```bash\ncd WModel2 && PYTHONPATH=$PWD "
                  f"python -m world.viz.report_runner --ckpt {a.ckpt} "
                  f"--data {a.data} --n-seq {a.n_seq} --n-rep {a.n_rep} "
                  f"--seed {a.seed}\n```\n\n数据：{base}；"
                  f"σ={sigma:.6f}；d_z={ck.get('d_z')}\n")

    R.fig_rollout(outdir, t_ms, s_t, s_m, s_l, fnames, ds.nseg,
                  "# Closed-loop rollout\n\n"
                  "**回答什么**：世界模型能否在**自回归**闭环中跟上真实状态。\n\n"
                  "**怎么读**：灰=真值；蓝=模型闭环（每步喂自己的预测）；"
                  "红虚线=L0（复制起始帧，从不更新）。\n"
                  "每条曲线取自**中间 compartment**（第 6/11 段）。\n\n"
                  "**判据**：蓝线若长期贴在灰线上 ⇒ 好；蓝线若漂向红线 ⇒ "
                  "模型退化成了 L0。\n" + cap_common)

    R.fig_horizon(outdir, rows_avg,
                  "# Error vs horizon\n\n"
                  "**回答什么**：在哪些视界上模型优于 L0。\n\n"
                  "**怎么读**：左图 log-log，纵轴是归一化误差（"
                  "已逐因子除以自身 std，所以跨因子可比）。"
                  "右图是 模型/L0 的比值，**低于 1 的黑点线即为赢**。\n\n"
                  "**注意**：模型与 L0 在**同一批样本**上算（配对），"
                  "否则 L0 的采样噪声可达 ±50%。\n" + cap_common)

    R.fig_amplification(outdir, rows_avg, sigma,
                        "# Amplification ratio\n\n"
                        "**回答什么**：闭环误差是「独立累积」还是「被放大」。\n\n"
                        "**怎么读**：$a(k)=\\varepsilon/(\\sqrt{k}\\sigma)$。\n"
                        "- $a\\approx1$：随机游走式累积（正常）\n"
                        "- $a\\gg1$ 且随 $k$ 单调上升：**latent 轨迹在发散**\n"
                        "- $a<1$：误差相互抵消\n\n"
                        "v2 的教训：曾实测 $a$ 从 2.6 涨到 79.7，"
                        "说明 Predictor 的 Jacobian 谱半径 > 1。\n" + cap_common)

    R.fig_pca(outdir, z_roll[:min(L, 400) + 1],
              z_roll[:min(L, 400) + 1],
              "# Latent trajectory (PCA)\n\n"
              "**回答什么**：闭环 latent 是否落在真实 latent 的流形上。\n\n"
              "**怎么读**：两条轨迹从同一点出发。若闭环轨迹迅速跑出真值"
              "轨迹覆盖的区域 ⇒ latent 发散，这正是放大比异常的根因。\n\n"
              "（本图为示意：两组用的是同一段 latent，"
              "真值侧取编码器输出的前 K 步。）\n" + cap_common)

    R.fig_spatial(outdir, s_t[-1], s_m[-1], fnames, t_ms[-1],
                  "# Spatial profile\n\n"
                  "**回答什么**：误差是集中在某几段，还是全局均匀。\n\n"
                  "**怎么读**：横轴是段内归一化位置 $x\\in(0,1)$，"
                  "每个子图一个因子。灰圆=真值，蓝方=模型。\n\n"
                  "**背景**：该段电紧张衰减仅 4%，所以真值曲线通常很平；"
                  "若模型曲线起伏很大，说明它没学到「等电位」这个事实。\n"
                  + cap_common)

    R.fig_inputs(outdir, t_ms, u1, ub_slice,
                 "# Synaptic input and boundary potentials\n\n"
                 "**回答什么**：协议长什么样，输入强度是否合理。\n\n"
                 "**怎么读**：上图三条曲线是 11 个 compartment 的"
                 "**均值**，阴影是 min..max 范围（宽度反映空间同步性 ρ）。\n"
                 "下图是 3 个邻居段的电位——它们是外生输入，"
                 "因为本段两端都连着别的段。\n\n"
                 "**注意**：NMDA 通道是**去门控**的 "
                 "$\\tilde g=g_{\\rm NMDA}/{\\rm mggate}$，"
                 "因为 $g_{\\rm NMDA}$ 含 $V$ 依赖，闭环时不可用。\n"
                 + cap_common)

    print(f"\n⇒ 图已写入 {outdir}")
    print("每个 .png 旁边有同名 .md 说明")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""绘图与报告：每张图配一份中文说明。

## 约定（results/figures/README.md）

- 坐标轴与图例用**英文**（本机无中文字体，中文会显示方块）
- 同时输出 `.png`（300 dpi）与 `.pdf`（矢量）
- 每张图旁边生成同名 `.md`，含：回答什么问题 / 怎么读 / 关键数值 / 结论 / 复现命令

## 配色

| 对象 | 颜色 | 线型 |
|---|---|---|
| 真值 | 灰 `#555555` | 实线 |
| 世界模型 | 蓝 `#1f77b4` | 实线 |
| L0（复制上帧）| 红 `#d62728` | 虚线 |
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

C_TRUE = "#555555"
C_MODEL = "#1f77b4"
C_L0 = "#d62728"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def _save(fig, outdir: str, name: str, caption: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight")
    with open(os.path.join(outdir, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(caption)
    plt.close(fig)


# ==============================================================================
def fig_rollout(outdir, t_ms, s_true, s_model, s_l0, names, nseg, caption):
    """单条协议：真值 / 模型闭环 / L0，9 因子各一格。"""
    nf = len(names)
    fig, axes = plt.subplots(nf, 1, figsize=(7.2, 1.35 * nf), sharex=True)
    mid = nseg // 2
    for i, nm in enumerate(names):
        ax = axes[i]
        ax.plot(t_ms, s_true[:, i, mid], color=C_TRUE, lw=1.6, label="truth")
        ax.plot(t_ms, s_model[:, i, mid], color=C_MODEL, lw=1.2,
                label="world model")
        ax.plot(t_ms, s_l0[:, i, mid], color=C_L0, lw=1.0, ls="--",
                label="L0 (copy last)")
        ax.set_ylabel(nm, fontsize=8)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle("Closed-loop rollout: truth vs world model vs L0 "
                 f"(compartment {mid+1}/{nseg})", fontsize=10)
    _save(fig, outdir, "fig_rollout", caption)


def fig_horizon(outdir, rows, caption):
    """误差 vs 视界（log-log）+ 模型/L0 比值。"""
    k = np.array([r["k"] for r in rows])
    ms = np.array([k_ * 0 for k_ in k], dtype=float)
    model = np.array([r["model"] for r in rows])
    l0 = np.array([r["L0"] for r in rows])
    ms = np.array([r["ms"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax1.loglog(ms, model, "o-", color=C_MODEL, label="world model")
    ax1.loglog(ms, l0, "s--", color=C_L0, label="L0 (copy last)")
    ax1.set_xlabel("horizon (ms)")
    ax1.set_ylabel("normalized error")
    ax1.legend()

    ratio = model / np.maximum(l0, 1e-12)
    ax2.semilogx(ms, ratio, "o-", color=C_MODEL)
    ax2.axhline(1.0, color="k", lw=0.8, ls=":")
    ax2.set_xlabel("horizon (ms)")
    ax2.set_ylabel("model / L0")
    ax2.set_title("<1 means beating L0", fontsize=9)
    for x, y in zip(ms, ratio):
        ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=7)
    fig.suptitle("Error vs horizon (paired comparison)", fontsize=10)
    _save(fig, outdir, "fig_horizon", caption)


def fig_amplification(outdir, rows, sigma, caption):
    """放大比 a(k)=ε/(√k·σ) 随 k 的曲线（稳定性诊断）。"""
    k = np.array([r["k"] for r in rows], dtype=float)
    model = np.array([r["model"] for r in rows])
    a = model / (np.sqrt(k) * max(sigma, 1e-12))
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.loglog(k, a, "o-", color=C_MODEL)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("horizon k (steps)")
    ax.set_ylabel(r"amplification  $a=\varepsilon/(\sqrt{k}\sigma)$")
    ax.set_title("a~1: random walk;  a>>1 and rising: latent divergence",
                 fontsize=9)
    for kk, aa in zip(k, a):
        ax.annotate(f"{aa:.2f}", (kk, aa), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7)
    _save(fig, outdir, "fig_amplification", caption)


def fig_pca(outdir, z_true, z_pred, caption):
    """latent 轨迹 2D PCA：真值 vs 闭环。"""
    Z = np.concatenate([z_true, z_pred], axis=0)
    Zc = Z - Z.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = Zc @ Vt[:2].T
    n = len(z_true)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(P[:n, 0], P[:n, 1], color=C_TRUE, lw=1.4, label="truth latent")
    ax.plot(P[n:, 0], P[n:, 1], color=C_MODEL, lw=1.2, label="closed-loop")
    ax.scatter([P[0, 0]], [P[0, 1]], color="k", s=18, zorder=5,
               label="start")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    ax.set_title("Latent trajectory (PCA)", fontsize=9)
    _save(fig, outdir, "fig_latent_pca", caption)


def fig_spatial(outdir, s_true_t, s_pred_t, names, t_ms, caption):
    """某一时刻的空间剖面：11 段 × 9 因子，真值 vs 预测。"""
    nf = len(names)
    P = s_true_t.shape[1]
    xs = (np.arange(P) + 0.5) / P
    fig, axes = plt.subplots(1, nf, figsize=(1.45 * nf, 2.6), sharey=False)
    for i, nm in enumerate(names):
        ax = axes[i]
        ax.plot(xs, s_true_t[i], "o-", ms=3, color=C_TRUE, lw=1.2)
        ax.plot(xs, s_pred_t[i], "s--", ms=3, color=C_MODEL, lw=1.0)
        ax.set_title(nm, fontsize=7)
        ax.set_xlabel("x", fontsize=7)
    axes[0].set_ylabel("value")
    fig.suptitle(f"Spatial profile at t={t_ms:.1f} ms "
                 "(gray=truth, blue=model)", fontsize=9)
    _save(fig, outdir, "fig_spatial", caption)


def fig_inputs(outdir, t_ms, u, ub, caption):
    """输入：三个电导通道（11 段的均值与范围）+ 3 个边界电位。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    lab = ["AMPA", "NMDA (ungated)", "GABA"]
    cols = ["#1f77b4", "#2ca02c", "#9467bd"]
    for c in range(3):
        m = u[:, c, :].mean(axis=1)
        lo = u[:, c, :].min(axis=1)
        hi = u[:, c, :].max(axis=1)
        ax1.plot(t_ms, m, color=cols[c], lw=1.3, label=lab[c])
        ax1.fill_between(t_ms, lo, hi, color=cols[c], alpha=0.2)
    ax1.set_ylabel(r"conductance ($\mu$S)")
    ax1.legend(fontsize=8, ncol=3)
    ax1.set_title("Synaptic input (mean across 11 compartments, band=min..max)",
                  fontsize=9)
    nm = ["apic[50](1.0) parent", "apic[52](0.0) child1",
          "apic[53](0.0) child2"]
    for i in range(3):
        ax2.plot(t_ms, ub[:, i], lw=1.1, label=nm[i])
    ax2.set_ylabel("boundary V (mV)")
    ax2.set_xlabel("time (ms)")
    ax2.legend(fontsize=8, ncol=3)
    _save(fig, outdir, "fig_inputs", caption)


def fig_attention(outdir, attn, u_mag, caption, fname="fig_attention"):
    """突触注意力图：模型在每一步"关注"哪些突触。

    attn : (T, nseg)  每个 compartment 的平均注意力权重
    u_mag: (T, nseg)  对应时刻的兴奋电导（AMPA+NMDA），用于对照

    这张图回答：**模型学到的是"整个段一起被驱动"，还是"某几段为主"？**
    """
    T, P = attn.shape
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    im0 = axes[0].imshow(u_mag.T, aspect="auto", origin="lower",
                         cmap="viridis")
    axes[0].set_ylabel("compartment")
    axes[0].set_title("Excitatory conductance (AMPA+NMDA, ungated)", fontsize=9)
    plt.colorbar(im0, ax=axes[0], fraction=0.03, pad=0.01)

    im1 = axes[1].imshow(attn.T, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_ylabel("compartment")
    axes[1].set_xlabel("time (steps)")
    axes[1].set_title("Attention weight of Predictor (query=latent, "
                      "key=each synapse)", fontsize=9)
    plt.colorbar(im1, ax=axes[1], fraction=0.03, pad=0.01)

    fig.suptitle("Which synapses does the world model attend to?", fontsize=10)
    _save(fig, outdir, fname, caption)

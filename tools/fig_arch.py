"""当前世界模型（A+B 版本）的结构框架图。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "WenQuanYi Micro Hei",
                                          "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.patches import FancyBboxPatch


def box(ax, x, y, w, h, text, fc="#eef3fb", fs=9, ec="k"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, style="-|>", color="k"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, lw=1.2, color=color))


def main():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15); ax.set_ylim(0, 9); ax.axis("off")

    # ---------- 输入 ----------
    box(ax, 0.2, 6.3, 1.9, 0.9, "s_t\n(9 × 11 = 99)", fc="#fdf2e9")
    box(ax, 0.2, 5.1, 1.9, 0.9, "u_t\n(85 = 17 × 5)", fc="#fdf2e9")
    box(ax, 0.2, 3.9, 1.9, 0.9, "θ (27)", fc="#fdf2e9")

    # ---------- Encoder ----------
    box(ax, 2.7, 4.4, 2.1, 2.5,
        "Encoder (T1, frozen)\nDeepSets:\nper-compartment MLP\n[ s(9)+pos ] 与 u(5)\n↓ mean + max pooling\n↓ + θ + bAP\nMLP → LayerNorm", fs=8)
    arrow(ax, 2.1, 6.75, 2.7, 6.4)
    arrow(ax, 2.1, 5.55, 2.7, 5.9)
    arrow(ax, 2.1, 4.35, 2.7, 4.9)

    # ---------- z_t ----------
    box(ax, 5.2, 5.4, 1.3, 0.8, "z_t\n(16)", fc="#e8f6ef")
    arrow(ax, 4.8, 5.65, 5.2, 5.8)

    # ---------- Predictor ----------
    box(ax, 7.0, 4.4, 2.0, 1.9,
        "Predictor (T2)\nMLP 2×192\nDeepSets(u)\n零初始化输出\nΔz = f(z_t, u_t, θ)", fs=8)
    arrow(ax, 6.5, 5.8, 7.0, 5.6)

    box(ax, 9.4, 5.4, 1.3, 0.8, "z_{t+1}\n(16)", fc="#e8f6ef")
    arrow(ax, 9.0, 5.6, 9.4, 5.8)

    # ---------- Decoder ----------
    box(ax, 11.3, 4.4, 1.9, 1.9,
        "Decoder (T1, frozen)\nper-compartment MLP\n[z, θ, pos] → 9 factors", fs=8)
    arrow(ax, 10.7, 5.8, 11.3, 5.6)

    box(ax, 13.6, 5.4, 1.2, 0.8, "Dec(z)\n(99)", fc="#fdf2e9")
    arrow(ax, 13.2, 5.8, 13.6, 5.8)

    # ---------- residual output ----------
    box(ax, 11.3, 2.6, 3.5, 0.9,
        "s_hat_{t+1} = s_t + [ Dec(z_{t+1}) − Dec(z_t) ]", fc="#fff8dc", fs=10)
    arrow(ax, 12.6, 4.4, 12.9, 3.5)
    ax.annotate("", xy=(11.9, 3.05), xytext=(2.2, 6.3),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#888",
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(5.2, 3.25, "skip: s_t", fontsize=8, color="#555")

    # ---------- training panel ----------
    box(ax, 0.2, 0.3, 14.6, 1.6,
        "训练（T2，Enc/Dec 冻结）：\n"
        "① 一步损失（L=128 窗口，teacher forcing）\n"
        "② 长视界闭环损失：rollout K=512（课程 4→512），在 64 / 256 / 512 步加损失，"
        "TBPTT 每 64 步 detach\n"
        "③ 事件加权（cai 变化 ×10）、regime 加权（linear/inter/plateau）、anchor-cl 自洽",
        fc="#f2f2f2", fs=10)
    ax.text(0.2, 2.15, "推理/评测：从 s_0 闭环 rollout，逐帧输出 s_hat_{t+1}；"
                       "指标 model/L0（相对持久性基线）", fontsize=9, color="#333")

    ax.text(0.2, 8.55, "WModel2 世界模型结构（R7：长视界 TBPTT 版）",
            fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig("runs/2026-11-01-e5dee3f-fig-traj/arch.png", dpi=140)
    print("saved arch.png")


if __name__ == "__main__":
    main()

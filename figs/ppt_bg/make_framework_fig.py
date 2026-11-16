import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

# Chinese font
for fam in ["Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK JP"]:
    try:
        font_manager.findfont(fam, fallback_to_default=False)
        plt.rcParams["font.family"] = fam
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

C_BG = "#F7F9FC"
C_IN = "#DCEBFA"
C_ENC = "#DDF2E3"
C_LAT = "#FFE8C7"
C_PRED = "#EADCF7"
C_DEC = "#F9D9D9"
C_OUT = "#F2F2F2"
C_TXT = "#1F2937"
C_EDGE = "#6B7280"

def box(ax, x, y, w, h, text, fc, fs=10, bold=False, rounded=0.02, lw=1.2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.006,rounding_size={rounded}",
                       linewidth=lw, edgecolor=C_EDGE, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, color=C_TXT, fontweight=("bold" if bold else "normal"),
            zorder=3, linespacing=1.25)
    return p

def arrow(ax, x1, y1, x2, y2, color="#374151", lw=1.6, style="-|>", rad=0.0, ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=12,
                        linewidth=lw, color=color, zorder=4,
                        connectionstyle=f"arc3,rad={rad}", linestyle=ls)
    ax.add_patch(a)

fig = plt.figure(figsize=(14, 6.6), dpi=220, facecolor="white")

# ============ LEFT: Stage 1 ============
ax = fig.add_axes([0.02, 0.07, 0.47, 0.84])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.01,0.01),0.98,0.98, boxstyle="round,pad=0.01,rounding_size=0.02",
                            facecolor=C_BG, edgecolor="#D1D5DB", linewidth=1.0, zorder=0))
ax.text(0.5, 0.955, "Stage 1：单段可行性验证（当前）", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#111827")

# inputs
box(ax, 0.04, 0.72, 0.20, 0.12, "状态 $s_t$\n9×11", C_IN, fs=11, bold=True)
box(ax, 0.04, 0.56, 0.20, 0.12, "输入 $u_t$\n28×5 + bAP", C_IN, fs=11, bold=True)
box(ax, 0.04, 0.40, 0.20, 0.12, "静态条件 $\\theta$\n27", C_IN, fs=11, bold=True)
box(ax, 0.04, 0.24, 0.20, 0.12, "位置 $x$\n11", C_IN, fs=11, bold=True)

# encoder
box(ax, 0.30, 0.44, 0.16, 0.40, "Encoder\n逐室 MLP\n+ mean/max 池化", C_ENC, fs=10.5, bold=True)
arrow(ax, 0.24, 0.78, 0.30, 0.70)
arrow(ax, 0.24, 0.62, 0.30, 0.62)
arrow(ax, 0.24, 0.46, 0.30, 0.56)
arrow(ax, 0.24, 0.30, 0.30, 0.50)

# latent
box(ax, 0.52, 0.56, 0.14, 0.14, "$z_t$\n16 维", C_LAT, fs=11, bold=True)
arrow(ax, 0.46, 0.64, 0.52, 0.63)

# predictor
box(ax, 0.50, 0.24, 0.22, 0.18, "Predictor\n$z_{t+1}=z_t+\\Delta z$\n条件：$u_t,\\theta$, bAP", C_PRED, fs=9.8, bold=True)
arrow(ax, 0.59, 0.56, 0.59, 0.42)
arrow(ax, 0.24, 0.62, 0.50, 0.33, rad=0.15, ls="--", color="#6B7280")
ax.text(0.30, 0.40, "$u_t$", fontsize=9, color="#6B7280", ha="center")

# decoder
box(ax, 0.75, 0.54, 0.20, 0.18, "Decoder\n共享 MLP\n条件：$\\theta, x$", C_DEC, fs=9.6, bold=True)
arrow(ax, 0.66, 0.63, 0.75, 0.63)
arrow(ax, 0.72, 0.33, 0.85, 0.54, rad=-0.12, ls="--", color="#6B7280")

# output
box(ax, 0.74, 0.76, 0.22, 0.12, "输出 $\\hat{s}_{t+1}$\n9×11", C_OUT, fs=11, bold=True)
arrow(ax, 0.85, 0.72, 0.85, 0.76)

# closed loop
box(ax, 0.72, 0.06, 0.24, 0.10, "闭环 rollout\n5–200 ms", C_OUT, fs=9.5)
arrow(ax, 0.96, 0.11, 0.96, 0.24, rad=0.25, color="#6B7280", ls="--")
arrow(ax, 0.94, 0.11, 0.72, 0.11, color="#6B7280", ls="--")
ax.text(0.50, 0.13, "目标：99→16，可解码 / 可干预", ha="center", va="center",
        fontsize=9.5, color="#374151")

# ============ RIGHT: Stage 2 ============
ax2 = fig.add_axes([0.51, 0.07, 0.47, 0.84])
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
ax2.add_patch(FancyBboxPatch((0.01,0.01),0.98,0.98, boxstyle="round,pad=0.01,rounding_size=0.02",
                             facecolor=C_BG, edgecolor="#D1D5DB", linewidth=1.0, zorder=0))
ax2.text(0.5, 0.955, "Stage 2：全神经元分层模型（目标）", ha="center", va="center",
         fontsize=14, fontweight="bold", color="#111827")

# hierarchy levels
box(ax2, 0.28, 0.70, 0.44, 0.12, "全局 latent  $z_g$\n（全细胞上下文）", C_LAT, fs=11, bold=True)
box(ax2, 0.10, 0.48, 0.30, 0.12, "branch latent $z_b$\n（分支局部状态）", C_LAT, fs=10.5, bold=True)
box(ax2, 0.58, 0.48, 0.30, 0.12, "branch latent $z_b$\n（分支局部状态）", C_LAT, fs=10.5, bold=True)
box(ax2, 0.06, 0.24, 0.18, 0.10, "compartment\n$s_t$ 9×639", C_IN, fs=9.5, bold=True)
box(ax2, 0.28, 0.24, 0.18, 0.10, "compartment\n$s_t$ 9×639", C_IN, fs=9.5, bold=True)
box(ax2, 0.50, 0.24, 0.18, 0.10, "compartment\n$s_t$ 9×639", C_IN, fs=9.5, bold=True)
box(ax2, 0.72, 0.24, 0.18, 0.10, "compartment\n$s_t$ 9×639", C_IN, fs=9.5, bold=True)

# arrows up
arrow(ax2, 0.15, 0.34, 0.20, 0.48)
arrow(ax2, 0.37, 0.34, 0.28, 0.48)
arrow(ax2, 0.59, 0.34, 0.68, 0.48)
arrow(ax2, 0.81, 0.34, 0.78, 0.48)
ax2.text(0.5, 0.38, "逐室编码 → 分支池化", ha="center", fontsize=9.5, color="#374151")
arrow(ax2, 0.25, 0.60, 0.38, 0.70, rad=0.1)
arrow(ax2, 0.73, 0.60, 0.62, 0.70, rad=-0.1)
ax2.text(0.5, 0.66, "全局池化 / 图消息传递", ha="center", fontsize=9.5, color="#374151")

# predictor + decoder
box(ax2, 0.28, 0.06, 0.44, 0.12, "分支/全局 Predictor\n图消息传递 + TBPTT", C_PRED, fs=9.8, bold=True)
arrow(ax2, 0.38, 0.18, 0.38, 0.24, color="#6B7280", ls="--")
arrow(ax2, 0.62, 0.18, 0.62, 0.24, color="#6B7280", ls="--")
box(ax2, 0.70, 0.70, 0.24, 0.12, "共享 Decoder\n逐室 9 因子", C_DEC, fs=9.5, bold=True)
arrow(ax2, 0.72, 0.76, 0.70, 0.76, color="#6B7280", ls="--")
arrow(ax2, 0.73, 0.60, 0.82, 0.70, color="#6B7280", ls="--")
arrow(ax2, 0.84, 0.82, 0.84, 0.865, color="#374151")
ax2.text(0.84, 0.87, "输出：V / cai / 门控\n全部 compartment", ha="center", va="center",
         fontsize=9.2, color="#374151")

# input context
box(ax2, 0.05, 0.06, 0.18, 0.12, "$u_t$ 逐段电导\n$\\theta$ 逐段属性\n$x$ 逐室位置", C_IN, fs=8.8)
arrow(ax2, 0.23, 0.12, 0.28, 0.12)

# bottom note
ax2.text(0.5, 0.015, "5.7k 状态 → 分支/全局 latent；目标：全细胞可解码、可干预",
         ha="center", va="bottom", fontsize=9.2, color="#374151")

# suptitle
fig.text(0.5, 0.985, "从单段到全神经元：压缩—预测—解码 的分层世界模型",
         ha="center", va="top", fontsize=15, fontweight="bold", color="#111827")

out = "figs/ppt_bg/out"
os.makedirs(out, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"framework_stage1_stage2.{ext}"),
                dpi=220, bbox_inches="tight", facecolor="white")
print("saved", out)

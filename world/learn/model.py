"""世界模型：集合编码（DeepSets）+ 残差输出。

## 为什么用集合编码（docs/archive/DESIGN_v3.md §6.2）

`nseg` 会随目标段变化（11、9、…）。若用固定 `33→hidden` 的 MLP，
`nseg=9` 时输入变 27 维，权重形状不匹配 ⇒ 完全不能用。

集合编码的做法：

```
对每个 compartment i:
    f_i = [gA_i, gN_i, gG_i, x_i]      (+θ 拼接在池化后)
    h_i = MLP_in(f_i)                  ← ★ 共享权重
z_u = Pool(h_0..h_{nseg-1})
```

⇒ `nseg` 变化只改变元素个数，结构完全兼容。

⚠️ **共享的是 MLP 参数，不是数据。** 各 compartment 的电导是**独立**输入。

## 池化方式

`Pool` = concat[mean, max]，保留"平均活跃度"与"最活跃那一段"两类信息。
位置信息 $x_i=(i+0.5)/nseg$ 已作为输入进 MLP，所以池化不会完全丢空间信息。

## 残差输出（docs/archive/DESIGN_v3.md §6.5）

$$\hat s_{t+1}=\hat s_t+\big[\mathrm{Dec}(z_{t+1})-\mathrm{Dec}(z_t)\big]$$

依据：解码器有恒定的系统性偏差（v2 实测 1.6e-3），相减即消。
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_FACTORS = 9


def _mlp(i: int, h: int, o: int, n: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


# ==============================================================================
class Encoder(nn.Module):
    """(s, u, θ, x[, u_glob]) → latent z。

    s : (B, T, 9, Ps)
    u : (B, T, Cu, Pu)
    θ : (B, th_dim)
    x : (Ps,)  状态 token 的位置编码（广播）

    ## v4 扩展（默认关闭，v3 行为不变）

    * `u_in_x=False`：`u` 的位置信息**已烘焙在通道里**
      （v4 每 token 通道 = `[AMPA, NMDA, GABA, frac, side]`）。
      这样 `Pu` 可以 ≠ `Ps`（v4：Ps=11, Pu=28）。
    * `n_glob=k`：额外拼一个**全局向量**（v4：bAP 指示，k=1）。
      bAP 作用于整个细胞而非某个 compartment，
      如果也当 token 就会在 mean/max 池化里被抹平。
    """

    def __init__(self, n_factors: int = 9, u_dim: int = 3, th_dim: int = 27,
                 d_z: int = 8, hidden: int = 128, n_layers: int = 2,
                 u_in_x: bool = True, n_glob: int = 0):
        super().__init__()
        self.d_z = d_z
        self.u_in_x = bool(u_in_x)
        self.n_glob = int(n_glob)
        cu = u_dim + (1 if self.u_in_x else 0)
        self.mlp_s = _mlp(n_factors + 1, hidden, hidden, n_layers)
        self.mlp_u = _mlp(cu, hidden, hidden, n_layers)
        # 池化(mean‖max) ×2 分支 → 融合
        self.fuse = _mlp(4 * hidden + th_dim + self.n_glob,
                         hidden, d_z, n_layers)
        self.ln = nn.LayerNorm(d_z)

    def forward(self, s, u, th, x, u_glob=None):
        """s:(B,T,F,Ps)  u:(B,T,Cu,Pu) → z:(B,T,d_z)。"""
        B, T, F, Ps = s.shape
        Pu = u.shape[-1]
        sp = s.permute(0, 1, 3, 2)              # (B,T,Ps,F)
        up = u.permute(0, 1, 3, 2)              # (B,T,Pu,Cu)
        xsb = x.view(1, 1, Ps, 1).expand(B, T, Ps, 1)
        fs = torch.cat([sp, xsb], dim=-1)       # (B,T,Ps,F+1)
        if self.u_in_x:
            # v3：u 与 s 共享同一套 compartment 位置
            xub = x.view(1, 1, Pu, 1).expand(B, T, Pu, 1)
            fu = torch.cat([up, xub], dim=-1)
        else:
            fu = up
        hs = self.mlp_s(fs)                     # (B,T,Ps,H)
        hu = self.mlp_u(fu)
        parts = [hs.mean(2), hs.amax(2), hu.mean(2), hu.amax(2)]
        thb = th.unsqueeze(1).expand(B, T, th.shape[-1])   # (B,T,th)
        parts.append(thb)
        if self.n_glob:
            if u_glob is None:
                raise ValueError("Encoder 需要 u_glob（n_glob>0）")
            parts.append(u_glob)                # (B,T,n_glob)
        z = self.fuse(torch.cat(parts, dim=-1))
        return self.ln(z)


class Predictor(nn.Module):
    """(z_t, u_t, θ) → z_{t+1}（残差式 z + Δ）。

    两种输入聚合方式：

    * `attn=False`：DeepSets —— 每个 compartment 一个 token，均值+最大池化。
    * `attn=True` ：**cross-attention** —— $z_t$ 作 query，
      22/33 个突触作 key/value。见 `docs/COMPARE_Yao2026.md` §5。

    ## 为什么加注意力版（motivation：Yao et al. 2026, Nat Commun）

    文献认为 AI 的 attention 对应的正是**单个神经元内部的树突整合**。
    把 $z_t$ 当 query、把各突触的电导当 key/value，等价于让模型学
    "当前状态应该关注哪几个突触" —— 而且这个注意力权重**可直接可视化**，
    是"哪些突触在驱动这段树突"的定量证据。

    ⚠️ 保底：注意力版初始化时把输出层置零（残差恒等），
    与 MLP 版一致，避免训练初期把 latent 推飞。
    """

    def __init__(self, d_z: int = 8, u_dim: int = 3, th_dim: int = 27,
                 hidden: int = 128, n_layers: int = 2, u_summary: int = 16,
                 attn: bool = False, n_heads: int = 4,
                 u_in_x: bool = True, n_glob: int = 0):
        super().__init__()
        self.u_summary = u_summary
        self.attn_mode = bool(attn)
        self.u_in_x = bool(u_in_x)
        self.n_glob = int(n_glob)
        cu = u_dim + (1 if self.u_in_x else 0)

        if self.attn_mode:
            d_model = u_summary * n_heads if u_summary * n_heads <= 256 else 128
            d_model = max(32, (d_model // n_heads) * n_heads)
            self.d_model = d_model
            self.q = nn.Linear(d_z, d_model)
            self.kv = _mlp(cu, hidden, 2 * d_model, 1)
            self.attn = nn.MultiheadAttention(d_model, n_heads,
                                              batch_first=True)
            merged = d_model * 2               # 上下文 + 注意力加权均值
        else:
            self.mlp_u = _mlp(cu, hidden, u_summary, 1)
            merged = u_summary

        self.net = _mlp(d_z + merged + th_dim + self.n_glob,
                        hidden, d_z, n_layers)
        self.ln = nn.LayerNorm(d_z)
        # ‼️ `use_ln_out` 必须为 False！
        #
        # 旧实现是 `return self.ln(z + de)`。LayerNorm 会**重写 z 本身**：
        # 实测 |LN(z) − z| / |z| = 0.063，而真实 Δz 只有 0.00063。
        # 也就是说「零初始化 = 恒等」这个前提**被 LN 破坏了**：
        # 即使 de=0，输出也不是 z，而是被扭曲 6.3% 的 z。
        # 后果：predictor 的 z 误差 = 0.065 = 真实 Δz 的 **103 倍**，
        # 单步误差 0.0082 而 L0 只有 0.00071（差 11.5×），永远赢不了。
        #
        # 保留 `self.ln` 只为旧 checkpoints 的 state_dict 兼容（strict=False）。
        self.use_ln_out = False
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.last_attn = None                  # 供可视化（(B,T,P)）

    def _tokens(self, u, x):
        """u:(B,T,Cu,Pu) → tokens:(B*T,Pu,Cu+(1 if u_in_x))。"""
        B, T, C, P = u.shape
        up = u.permute(0, 1, 3, 2).reshape(B * T, P, C)
        if not self.u_in_x:
            return up, B, T, P
        xb = x.view(1, P, 1).expand(B * T, P, 1)
        return torch.cat([up, xb], dim=-1), B, T, P

    def forward(self, z, u, th, x, u_glob=None):
        """z:(B,T,d_z)  u:(B,T,Cu,Pu) → z_next:(B,T,d_z)。"""
        B, T, dz = z.shape
        tok, B2, T2, P = self._tokens(u, x)
        zf = z.reshape(B * T, dz)
        thb = th.unsqueeze(1).expand(B, T, th.shape[-1]).reshape(B * T, -1)

        if self.attn_mode:
            kv = self.kv(tok)                              # (BT,P,2*d_model)
            k, v = kv.chunk(2, dim=-1)
            q = self.q(zf).unsqueeze(1)                    # (BT,1,d_model)
            ctx, w = self.attn(q, k, v, need_weights=True,
                               average_attn_weights=True)  # (BT,1,d), (BT,1,P)
            self.last_attn = w.squeeze(1).reshape(B, T, P)
            ctx = ctx.squeeze(1)                           # (BT,d_model)
            # 注意力加权均值（显式的"读出的突触内容"）
            am = torch.bmm(w, v).squeeze(1)                # (BT,d_model)
            us = torch.cat([ctx, am], dim=-1)
        else:
            hu = self.mlp_u(tok)
            uu = torch.cat([hu.mean(1), hu.amax(1)], dim=-1)
            us = uu[..., :self.u_summary]

        de = self.net(torch.cat([zf, us, thb] +
                                ([u_glob.reshape(B * T, -1)]
                                 if self.n_glob else []),
                                dim=-1)).reshape(B, T, dz)
        out = z + de
        if getattr(self, "use_ln_out", False):
            out = self.ln(out)
        return out


class Decoder(nn.Module):
    """(z, θ, x) → ŝ（每 compartment 一次，共享权重）。"""

    def __init__(self, d_z: int = 8, n_factors: int = 9, th_dim: int = 27,
                 hidden: int = 128, n_layers: int = 2):
        super().__init__()
        self.net = _mlp(d_z + th_dim + 1, hidden, n_factors, n_layers)

    def forward(self, z, th, x):
        """z:(B,T,d_z) → ŝ:(B,T,F,P)。"""
        B, T, dz = z.shape
        P = x.numel()
        zb = z.unsqueeze(2).expand(B, T, P, dz)
        thb = th.unsqueeze(1).unsqueeze(2).expand(B, T, P, th.shape[-1])
        xb = x.view(1, 1, P, 1).expand(B, T, P, 1)
        out = self.net(torch.cat([zb, thb, xb], dim=-1))   # (B,T,P,F)
        return out.permute(0, 1, 3, 2)                     # → (B,T,F,P)


# ==============================================================================
class WorldModel(nn.Module):
    def __init__(self, n_factors: int = 9, u_dim: int = 3, th_dim: int = 27,
                 d_z: int = 8, hidden: int = 128, n_layers: int = 2,
                 pred_attn: bool = False, n_heads: int = 4,
                 u_in_x: bool = True, n_glob: int = 0):
        super().__init__()
        self.d_z = d_z
        self.n_factors = n_factors
        self.pred_attn = bool(pred_attn)
        self.u_in_x = bool(u_in_x)
        self.n_glob = int(n_glob)
        self.enc = Encoder(n_factors, u_dim, th_dim, d_z, hidden, n_layers,
                           u_in_x=u_in_x, n_glob=n_glob)
        self.pred = Predictor(d_z, u_dim, th_dim, hidden, n_layers,
                              attn=pred_attn, n_heads=n_heads,
                              u_in_x=u_in_x, n_glob=n_glob)
        self.dec = Decoder(d_z, n_factors, th_dim, hidden, n_layers)

    # ------------------------------------------------------------------
    def encode(self, s, u, th, x, u_glob=None):
        return self.enc(s, u, th, x, u_glob)

    def predict(self, z, u, th, x, u_glob=None):
        return self.pred(z, u, th, x, u_glob)

    def decode(self, z, th, x):
        return self.dec(z, th, x)

    def reconstruct(self, s, u, th, x, u_glob=None):
        return self.decode(self.encode(s, u, th, x, u_glob), th, x)

    # ------------------------------------------------------------------
    def step_residual(self, s_prev, z_prev, z_next, th, x):
        """残差步：ŝ = s_prev + [Dec(z_next) − Dec(z_prev)]。

        z_prev / z_next 可以是 (B,d_z) 或 (B,T,d_z)；统一升成 3D。
        """
        if z_next.dim() == 2:
            z_next = z_next.unsqueeze(1)
        if z_prev.dim() == 2:
            z_prev = z_prev.unsqueeze(1)
        d_next = self.decode(z_next, th, x)
        d_prev = self.decode(z_prev, th, x)
        return s_prev + (d_next[:, 0] - d_prev[:, 0])

    @torch.no_grad()
    def rollout(self, s0, u_seq, th, x, use_residual: bool = True,
                ug_seq=None):
        """闭环滚动。s0:(B,F,P) u_seq:(B,L,Cu,Pu) → outs:(B,L,F,P)。

        ‼️ 时间对齐（极易错，实测错过一次导致闭环全崩）：

        训练时的映射是 `predict(z_j, u_{j+1}) = z_{j+1}`
        （T2 里 `z_prev[t] = z_true[t-1]` 搭配 `u[t]`）。
        所以闭环第 t 步要用 **u_{t+1}** 而不是 u_t。

        返回的 `outs[t]` = $\\hat s_{t+1}$（已经推了 t+1 步）。
        """
        L = u_seq.shape[1]
        z = self.encode(s0.unsqueeze(1), u_seq[:, :1], th, x,
                        None if ug_seq is None else ug_seq[:, :1]
                        ).squeeze(1)
        s_hat = s0.clone()
        outs = []
        for t in range(L):
            j = min(t + 1, L - 1)                # ★ u_{t+1}
            ut = u_seq[:, j]
            gt = None if ug_seq is None else ug_seq[:, j]
            z_next = self.predict(z.unsqueeze(1), ut.unsqueeze(1), th, x,
                                  None if gt is None else gt.unsqueeze(1)
                                  ).squeeze(1)
            if use_residual:
                s_hat = self.step_residual(s_hat, z, z_next, th, x)
            else:
                s_hat = self.decode(z_next.unsqueeze(1), th, x)[:, 0]
            z = z_next
            outs.append(s_hat)
        return torch.stack(outs, dim=1)

    # ------------------------------------------------------------------
    def n_params(self) -> dict:
        c = lambda m: sum(p.numel() for p in m.parameters())   # noqa: E731
        return dict(enc=c(self.enc), pred=c(self.pred), dec=c(self.dec),
                    total=c(self))


def build(d_z: int = 8, hidden: int = 128, n_layers: int = 2,
          n_factors: int = 9, u_dim: int = 3, th_dim: int = 27,
          pred_attn: bool = False, n_heads: int = 4,
          u_in_x: bool = True, n_glob: int = 0) -> WorldModel:
    return WorldModel(n_factors, u_dim, th_dim, d_z, hidden, n_layers,
                      pred_attn=pred_attn, n_heads=n_heads,
                      u_in_x=u_in_x, n_glob=n_glob)

# 后续修改方向计划（2026-11-04，待 E1-strong 结论后启动）

> 结合 `docs/ROADMAP.md`、`docs/PLAN_ENSEMBLE_5SEED.md` 与"主流世界模型可借鉴清单"整理。
> **本文件只做规划，不启动实验**；等当前（E1-strong seed 2/3）结论出来，再按 §5 的判据逐项启动。

## 1. 对"可借鉴清单"的判断

| 来源 | 可借 | 用在哪 | 适用性 |
|---|---|---|---|
| PlaNet/Dreamer | **latent overshooting**（latent 空间多步一致性） | 长视界 | **高**：便宜，可作辅助损失 |
| PlaNet/Dreamer | imagination rollout | 长视界 | 已有（CL rollout）；缺 value/policy，不适用 |
| PlaNet/Dreamer | KL balancing | 稳定性 | **条件适用**：需要随机 latent（RSSM）才有意义 |
| Ha 2018 | MDN/混合密度 | 事件头 | **高**：事件时刻是多模态的，MSE 只能给均值 |
| IRIS | 离散 token | 事件/离散状态 | 中：需重训 encoder，优先级低于事件头 |
| Director | 分层 world model | 全细胞 Stage 2/3 | **高**（但属于整细胞阶段） |
| JEPA | latent 预测、不做 pixel 重建 | 思路参考 | 我们已在 latent 空间预测（T2），保留物理 decoder 是必要的 |

**结论**：清单可用，但要分层——**latent overshooting / MDN 事件头 / ensemble+EMA** 是短期可做的；
**KL balancing / 离散 token** 依赖随机 latent 或重训 encoder，属中期；**Director** 属整细胞阶段。

## 2. Tier 0：已在计划中的（先做）

| 项 | 解决 | 成本 | 状态 |
|---|---|---|---|
| **5-seed 单独训练 + 集成** | seed 方差 / 不确定性 | 4 新 seed ≈10 h | 方案已写（`PLAN_ENSEMBLE_5SEED.md`） |
| **EMA + 闭环验证选模** | seed 方差 / 选模 | ~1 h 代码 | 待实现 |
| **Huber 鲁棒损失** | 重尾误差 | 1 seed 筛查 ~2.5 h | 待筛查 |

## 3. Tier 1：便宜的（1–2 天，按性价比）

1. **latent overshooting（latent 多步一致性）**
   - 在 CL 损失里加：把真实 `s_{t+k}` 编码成 `z_{t+k}`（k=64/256），与 rollout 的 `z` 做 MSE；
   - 解决：长视界漂移（latent 层面的多步一致性）；
   - 成本：~20 行 + 1 seed 筛查 2.5 h；**与现有 CL 状态损失互补**。
2. **事件头 / MDN**
   - 分类“未来 Δ ms 内是否有 Ca 峰” + 回归峰时刻/幅度；或 MDN 输出 Δs 的混合分量；
   - 解决：事件 F1（当前 0.25）；
   - 成本：1–2 天 + 1 seed 筛查；**先做最简单的二分类+时刻回归**。
3. **多视界直接预测头（jump 辅助）**
   - 已有 jump 证据（0.48–0.58）；把 jump 预测作为**辅助损失**（或与 rollout 门控融合）；
   - 解决：长视界 + 事件；
   - 成本：~1 天。
4. **EMA 权重 + 闭环验证选模**
   - 解决：seed 方差；成本最低。

## 4. Tier 2：结构性的（3–7 天，需先诊断）

1. **RSSM-lite（确定性 h + 小随机 z）**
   - **前置诊断**：检查 latent 是否装不下“快变量”（例如 z 对 `dV/dt`、Ca 峰时刻的线性探针 R² 低）；
   - 若诊断支持，再投入 2–3 天；否则不做。
2. **KL balancing**：只随 RSSM 一起做。
3. **离散 token（VQ/IRIS）**：仅当事件头仍不足、且愿意重训 encoder 时。
4. **快慢双头**：**已试并失败（E2）**，无新证据不重复。

## 5. 决策规则（等 E1-strong 结论）

| E1-strong 结果 | 基座 | 启动项 |
|---|---|---|
| 三 seed 均值 ≤0.52 且 F1 ≥0.30 | **E1-strong** | 5-seed 集成（补 seed 4/5）+ EMA + latent overshooting |
| 仅主指标复现、F1 不升 | E1-strong（主）/ A+B（事件） | 5-seed 集成 + 事件头 |
| 未复现 | **A+B** | 5-seed 集成（补 seed 2–5）+ Huber + latent overshooting；事件头作为中期 |

## 6. 整细胞阶段（Tier 3，非近期）

- Director 式分层 + 图网络 + 每 branch latent；数据 370 GB/组、TBPTT 必用；
- 先做 64–128 segment 的子集验证，再扩全细胞；
- 与当前 `apical_51` 结果作为 Stage 1 对照。

## 7. 明确不做（避免重复）

- 快慢双头（E2 已崩）；dense 损失点（128/128 发散）；更长视界单独加码（H1024 未复现）；
- 扩散/大 Transformer（CPU + 数据量不匹配）；RL 的 reward/value 头（无回报信号）。

# results/ — 当前 v4 产出

## 活跃目录

| 目录 | 内容 |
|---|---|
| `checkpoints/` | 当前 v4 权重：`t1v4_dz16`、`t2v4_anchor`、`t2v5_ancC`、`t2v5_noancC`、`t2v5_events`（R2 最优）、`t2v5_events_w5a2`、`t2v5_events_w8` |
| `rounds/` | R0–R4 六轮归档：每轮 `README.md` / `metrics.json` / `figures/` / `logs/` / `ckpt/`；总索引 `README.md`，对比图 `compare_rounds.png`、`summary_rounds.png` |
| `logs/` | 当前训练/生成日志（`t1v4.log`、`t2v5_*.log`、`t2v4_anchor.log`、`datagen_te_s3.log`） |
| `reports/` | 当前评测/诊断 JSON（R1/R2 的 `eval_last_t2_te_s3.json`、`diag_last_t2.json`、`latent_ci.json`、`ca_events_k2000.json`、`eval_stageA_t2_te_s3.json`） |
| `figs/` | R2 阶段的诊断图（误差-视界、latent 记忆、空间剖面等） |
| `data/` | `scaler_v4.json`（训练集归一化尺度） |

## 入口

- 跨轮总表：`rounds/README.md`
- 单轮结论：`rounds/R*/CONCLUSIONS.md`（R2/R3/R4 有；R0/R1/R1b 由 `README.md` + `metrics.json` 给出）
- 当前最优：R2 `t2v5_events`（`model/L0@200=0.562`，`model/L0@8000=0.911`，无发散）

## 历史归档

`results/archive/legacy_v3/` 存放：

- v3 / 早期 v4 的 checkpoints（`t1_dz16`、`t2_dz16`、`t2c*`、`t2s3*`、`t2v4_dz16*` 等）
- 旧 logs、旧 reports（`STATUS_*`、`cmp_*`、旧 `eval_t2*`）
- 旧 `figures/`
- 旧 `scaler.json`

归档区**不要用于当前结论**；当前结果以 `rounds/` 为准。

## 约定

- 大数据集放在 `WModel2/data/`，不放进 `results/`
- `rounds/` 下的 `metrics.json` 是机器可读真源；`README.md` 由 `tools.round_report` 生成，`CONCLUSIONS.md` 人工撰写、不被覆盖
- 结果文件名/JSON 内保留数据版本、随机种子、关键超参

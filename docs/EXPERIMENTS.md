# 实验索引（归档用，2026-11-16）

> 目的：保留每个实验的**脚本、配置、结果目录与权重位置**，便于后续对比与论文归档。
> 详细结论见 `PROJECT_STATE.md` §2.x；计划见 `docs/WM_PLAN.md`；标准见 `docs/EVAL_STANDARD.md`。

## 0. 目录约定

- 脚本：`tools/night_*.sh`、`tools/reeval*.sh`、`tools/cmp_*.sh`
- 结果：`runs/<date>-<commit7>-<run>/`（`stdout.log`、`monitor.log`、`commit.txt`；部分含 `config.yaml`/`eval_*.json`）
- 权重：`results/checkpoints/<run>/last.pt`（NAS）
- 汇总：`runs/<date>-night-*/night.log`（训练/评测流水）

## 1. 主实验

| # | 实验 | 权重目录 | 关键配置 | 200 ms（同口径） | 判定 |
|---|---|---|---|---|---|
| 1 | Baseline | `r6_b_*` | CL=64 | 0.83 / 3.13 / 4.78（3 seed） | 2/3 退化 |
| 2 | A | `r6_ab_A`, `r6_a_s2`, `r6_a_s3` | horizon 512, lags {512} | 0.538 / 0.884 / 0.836 | ❌ 未复现 |
| 3 | **A+B** | `r6_ab_AB`, `r6_ab_s2..s5` | 64/256/512 | **0.576**（5 seed） | 基线 |
| 4 | H1024 | `r6_b_h1024_s2/s3` | lags 64..1024 | 0.588（3 seed） | ❌ |
| 5 | H2048 | `r6_h2048` | lags ..2048 | 0.554（1 seed） | 未确认 |
| 6 | E1（V 事件加权） | `r6_e1_vevent`, `r6_e1_s2/s3` | event_v_weight 10 | 0.867（3 seed） | ❌ |
| 7 | E1-strong | `r6_e1strong`, `r6_e1s_s2/s3` | 30 / tau 0.01 | 0.604（3 seed） | ❌ |
| 8 | Huber | `r6_huber` | huber 1e-3 | 0.590 | ❌ |
| 9 | **OV** | `r6_ov`, `r6_ov_s2..s5` | overshoot 32/64/96 | **0.493**（5 seed） | ✅ 采纳 |
| 10 | **FW2** | `r6_fw2`, `r6_fw2_s2..s5` | OV + cai/V×2 | **0.508 / 0.563**（5 seed） | ✅ 主候选 |
| 11 | pinball | `r6_pinball`, `r6_pinball2` | q 0.85 / 0.9 | 0.660 / 0.425 | ⚠️ 不一致 |
| 12 | 长滞后 OV | `r6_ovlong`, `_s2`, `_s3` | lags 128/256 | 0.568（3 seed） | ❌ |
| 13 | 事件输出（cai） | `r6_eventout`, `r6_eventout_thr` | spike-and-slab | 0.596 / 0.532 | ❌ |
| 14 | VH（独立 V 头） | `r6_vhead`, `r6_vhead_s2` | v_head | 0.513 / 0.555 | ⚠️ 不一致 |
| 15 | u-hist | `r6_uhist` | u-hist 3 | 0.702 | ❌ |
| 16 | 多视界头 | `r6_mh` | mh 64/256 | 0.670 | ❌ |
| 17 | d_z=32 | `r6_dz32` | T1/T2 d_z 32 | 0.589 | ❌ 无增益 |
| 18 | Decoder 微调 | `r6_finetune` | 解冻 Dec 2e-4 | 2.091 | ❌ 崩 |
| 19 | 2 s 数据训练 | `r6_2s` | `tr_s1_2s` | 0.611 | ❌ |
| 20 | RSSM-lite | `runs/2026-11-10-rssm` | 随机 latent + KL | 1.130 | ❌ KL 塌缩 |
| 21 | **E1（关事件权重）** | `r6_e1`, `r6_e1_s2`, `_s3`, `_s5` | event_weight 0 | **0.443**（4 seed） | ⚠️ 权衡（峰差） |
| 22 | EW3 | `r6_ew3`, `r6_ew3_s2` | event_weight 3 | 0.463 / 0.578 | ⚠️ F1 好、峰不稳 |
| 23 | EOT | `r6_eot_s2/s3` | event-out thr 0.5 | 0.560 / 0.710 | ❌ |
| 24 | VEV（V 事件输出） | `r6_vev`, `r6_vev_s2` | event-out V | 0.481 / 0.388 | ❌ 峰未修好 |
| 25 | 非 latent 基线 | `b1-direct-mlp`, `b2-direct-gru`, `b3-tdnn` | MLP/GRU/TDNN | 1.115 / 299 / 2.652 | ❌ 长程失败 |
| 26 | 5-seed 集成 | `runs/2026-11-05-ensemble5` | 5 模型平均 | 0.530（64 trial） | ✅ 稳健性 |

## 2. 关键脚本

| 脚本 | 用途 |
|---|---|
| `tools/night_ab_seeds_2026_10_31.sh` | A/A+B 多 seed |
| `tools/night_all_2026_11_02.sh` | H1024 + 延长评测 + dense + P3a |
| `tools/night_e1_2026_11_04.sh` / `night_e1strong...` | E1 / E1-strong |
| `tools/night_bc_2026_11_05.sh` | Huber + overshoot |
| `tools/night_ov_seeds*` | OV 多 seed |
| `tools/night_fw2_seeds*` | FW2 多 seed |
| `tools/night_pinball*` | pinball |
| `tools/night_vhead*` / `night_v_event*` | V 头 / V 事件输出 |
| `tools/night_e01_2026_11_12.sh` / `night_ew3*` / `night_eot*` | 事件权重消融 |
| `tools/reeval_hist.sh` / `reeval128.sh` | 多指标复查 |
| `tools/cmp_e1_fw2.sh` / `cmp_ew3_eot.sh` | 同 seed 对比 |
| `tools/bias_check.py` / `regime_bias.py` / `err_decomp.py` | 偏差/分解诊断 |
| `tools/fig_cases.py` / `fig_predictions.py` | 结果图 |

## 3. 归档规则

1. **不删除**任何 `runs/` 目录或 `results/checkpoints/` 权重；
2. 新实验写入新目录（`runs/<date>-<commit7>-<name>`），不覆盖；
3. 每次重要结论更新 `PROJECT_STATE.md`，并把脚本/结果提交 git；
4. 论文只用 `EVAL_STANDARD.md` 的多指标口径，不再用单一 `model/L0`。

# docs/ — 当前文档索引

## 当前有效

| 文件 | 作用 | 状态 |
|---|---|---|
| [`DESIGN_v4.md`](DESIGN_v4.md) | **当前唯一实验方案**（输入/状态/协议/门禁/训练/评测） | ✅ 2026-10-22 定档并执行 |
| [`MODEL_FACTS.md`](MODEL_FACTS.md) | NEURON 模型几何、通道、突触、单位等实测事实 | ✅ 引用数字前先查 |
| [`COMPARE_Yao2026.md`](COMPARE_Yao2026.md) | 与 Yao et al. 2026 的注意力/树突整合对照 | 📎 文献参考 |
| [`L5PC_exp_使用手册.md`](L5PC_exp_使用手册.md) | 官方 `L5PC_exp.py` 的使用说明 | 📦 资产参考 |

结果入口不在 `docs/`：

- 跨轮索引：`results/rounds/README.md`
- 单轮结论：`results/rounds/R*/CONCLUSIONS.md`
- 图：`results/rounds/summary_rounds.png`、`results/rounds/compare_rounds.png`

## 历史归档（不要作为当前决策依据）

`docs/archive/` 存放：

- `DESIGN_v3.md`、`DESIGN_世界模型_定稿_v2.md`、`DESIGN_世界模型_定稿_v1.md` —— 已被 v4 取代的设计
- `STATUS_迁移与作废清单.md` —— WModel → WModel2 的迁移记录
- 旧实验计划、审计、事故调查与旧文档索引

旧 v3 代码在 `archive/legacy_v3/code/`；旧 v3 结果在 `results/archive/legacy_v3/`。

## 文档可信度规则

1. **代码是准，文档是解释。** 冲突时以当前代码 + 实跑输出为准。
2. 任何“实测”结论必须附：脚本名、随机种子、样本数、日期。
3. 已被推翻的结论不删除，标记 `❌ 已推翻（原因 + 日期）`，避免后人再次引用。
4. 当前设计只认 `DESIGN_v4.md`；v2/v3 只用于追溯。

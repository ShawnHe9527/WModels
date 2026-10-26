# archive/legacy_v3 — 历史代码归档

这里存放 **v2 / v3 时代**、已不再参与当前 v4 入口的代码与工具。

当前 v4 入口只依赖：

```text
run_v4.sh
world/core/{safety,progress,config}.py
world/gen/{datagen2,sim2,protocol,protocol2,secinfo}.py
world/learn/{data2,model,train2}.py
tools/{round_report,rounds_index,fig_rounds_summary,ca_event_metrics,
       latent_ci,diag_latent,plot_v4,verify_env,v0_check,
       probe_drive_modes,probe_neighbor_caspike}.py
```

归档内容：

| 路径 | 说明 |
|---|---|
| `code/world/gen/datagen.py`、`sim.py` | v3 数据生成与电压钳/旧仿真 |
| `code/world/learn/{data,train,metrics}.py` | v3 数据集、单体训练、旧指标 |
| `code/world/viz/` | 旧 `report_runner` / `report` 绘图 |
| `code/world/eval/__init__.py` | 空包占位 |
| `tools/compare_ckpts.py` | 旧跨 ckpt 对比 |
| `tools/stage_b_selfcheck.py` | 旧阶段 B 自检 |
| `tools/report_v4.py` | 旧的 STATUS_v4 生成器（结论硬编码，已过时） |
| `tools/probe_*.py`（部分） | 早期一次性探针；当前保留的两个探针不在此处 |

**不要从这里 import，也不要把这里的结论当作当前结果。**
当前设计与结果：

- `docs/DESIGN_v4.md`
- `results/rounds/README.md`
- `results/rounds/R*/CONCLUSIONS.md`

对应的旧文档在 `docs/archive/`，旧结果在 `results/archive/legacy_v3/`。

# R3 结论说明（压过冲 + 拉回方向：**两项假设均未成立**）

> 人工撰写，不会被 `tools.round_report` 覆盖。
> 自动数值见同目录 `README.md` / `metrics.json`；跨轮索引见 `../README.md`。

---

## 0. 一句话结论

**R3 是一次失败的调参轮，不予采纳。R2 的配置（`--event-weight 10 --anchor-cl 0.1`）
仍是目前最优解。** 但它替我们排除了两条假设，并且暴露了一个**评估方法上的严重缺陷**
（单 trial 指标不可信），后者价值可能比这一轮训练本身更大。

---

## 1. R3 动了什么

只动两个旋钮，其余与 R2 完全一致（batch 24 / length 128 / cl-steps 64 /
档权重 0.25,0.35,0.40 / hot-frac 0.5 / 9000 步）：

| 旋钮 | R2 | R3 | 想修的问题 |
|---|---|---|---|
| `--event-weight` | 10 | **5** | V 上限过冲 19.7 mV（真值 13.4） |
| `--anchor-cl` | 0.1 | **0.2** | cos(k=2000) 从 0.928 掉到 0.410 |

训练本身很健康：9000 步 / 3130 s，**σ = 0.000285**（五轮最低），无 OOM。

---

## 2. 结果：三项退步，一项"改善"是假的

| 指标 | R2 | **R3** | 判定 |
|---|---|---|---|
| cai precision | 0.696 | **0.155** | ❌ 崩塌 |
| cai recall | 0.867 | **0.273** | ❌ 崩塌 |
| V 上限 | 19.7 | **10.8** | ❌ 变成不足（真值 13.4） |
| cai RMSE | 0.1688 | **0.2510** | ❌ |
| plateau 比值中位 | 0.0639 | 0.0855 | ❌ |
| inter 比值中位 | 0.0096 | 0.0134 | ❌ |
| linear 比值中位 | 0.0067 | 0.0070 | ≈ 持平 |
| a(200) | **0.562** | 0.605 | ❌ 反而更差 |
| k=8000 比值 | **0.911** | 1.473 | ❌ |
| cos(k=2000) 中位 | **0.961** | 0.869 | ❌ 反而更差 |
| σ（训练） | 0.000310 | **0.000285** | ✅ 但无意义（见 §4） |

**除了训练 loss，R3 在所有下游指标上都不如 R2。**

---

## 3. 两条假设的裁决

### 3.1 `--anchor-cl 0.1 → 0.2`：**假设被证伪**

- 它想修的问题是「cos(k=2000) 只有 0.410」。但复算发现：**0.410 是单个 trial 的值**。
  拿 12 个 plateau trial 求中位数，R2 在 k=2000 的 cos 是 **0.961**，是五轮里最好的。
- 也就是说 **R2 根本不存在方向性问题**，这个旋钮是在修一个不存在的病。
- 加大权重不但没帮助，中位 cos 反而从 0.961 掉到 0.869。
- ⇒ 结论：**`--anchor-cl 0.1` 是够的；不要为了"更稳"盲目加大，它会压制有效信息。**

### 3.2 `--event-weight 10 → 5`：**方向对，幅度过头**

- `event-weight` 实际控制的是「**模型敢把 Ca 事件预测多大**」：
  - weight 10 → V 上限 **19.7 mV**（过冲 +47%）
  - weight 5 → V 上限 **10.8 mV**（不足 −19%）
  - ⇒ 真值 13.4 mV，最优应在 **weight ≈ 7～8** 之间（未测）。
- 为什么减半会让 P/R 崩塌到 0.155/0.273（而不是温和下降）？因为
  **cai 事件的 precision/recall 用的是一个固定的绝对阈值**：

  ```python
  TH = -3.10                       # log10[Ca2+]，tools/round_report.py
  at = true[:, ci, :].max(1) >= TH
  ap = pred[:, ci, :].max(1) >= TH
  ```

  幅度一旦整体降到阈值以下，模型就**整段检测不到事件**，于是既漏报（recall↓）
  又只剩零星错报（precision↓）。**这是一个阈值伪影，不是"模型变傻了"。**

---

## 4. 更重要的收获：单 trial 指标会骗人

这是本轮真正的技术发现。用新增的 `tools/latent_ci.py` 在**同一批 12 个 plateau trial**
上算逐 trial 的 cos，得到中位数 + bootstrap 95% CI：

| run | k=1000 cos 中位 [95%CI] | k=2000 cos 中位 [95%CI] | k=2000 掉到 <0.5 的 trial 占比 |
|---|---|---|---|
| R1 `t2v5_ancC` | 0.942 [+0.841,+0.968] | 0.859 [+0.766,+0.916] | 8% |
| R2 `t2v5_events` | 0.970 [+0.908,+0.989] | **0.961** [+0.576,+0.975] | **25%** |
| R3 `t2v5_events_w5a2` | 0.971 [+0.547,+0.980] | 0.869 [+0.419,+0.930] | 33% |

读法：
1. **中位数层面** R2 > R3 > R1，顺序与旧的单 trial 结论**完全相反**。
2. 但 R2 的 CI 下界只有 0.576 —— 因为它有 **25% 的 trial 会崩**（分布双峰）：
   大多数 trial 极准，少数几个 trial 方向彻底跑飞。
3. ⇒ **真正要解决的问题不是"平均漂移"，而是"少数 trial 崩掉"**。
   这是一个鲁棒性问题，靠全局加大 anchor 权重解决不了（R3 已证明会伤到好 trial）。

**已据此修正工具**（`tools/round_report.py` / `tools/rounds_index.py`）：
latent 表改为 **12 trial 中位数 + 95% CI + cos<0.5 占比**，不再用单个 trial。
索引表同时加了 `cai R` 列——**只看 P 不看 R 会重演 R1b 的误判**。

---

## 5. 下一步（R4）建议

**基线回退到 R2 的配置**，然后只做一件事：

```
--event-weight 10 → 8        # 目标 V 上限 13~17 mV，保持 P/R 不崩
--anchor-cl   0.2 → 0.1      # 回退，已证伪
```

判据（缺一不可，全部用**跨 12 trial 中位数**或确定性统计量）：

| 指标 | 目标 | 依据 |
|---|---|---|
| cai precision | ≥ 0.60 | R2 = 0.696 |
| cai recall | ≥ 0.80 | R2 = 0.867 |
| V 上限 | 13 ~ 17 mV | R2 19.7 偏高 / R3 10.8 偏低 |
| plateau 比值中位 | ≤ 0.070 | R2 = 0.0639 |
| a(200) | ≤ 0.58 | R2 = 0.562 |
| k=8000 比值 | ≤ 1.2 | R2 = 0.911 |
| cos(k=2000) 中位 | ≥ 0.90 | R2 = 0.961 |
| cos(k=2000)<0.5 占比 | ≤ 25% | R2 = 25%，此列是真正的技术债 |

另建议并行做一件几乎零成本的事：**给 Ca 事件指标补一个不依赖绝对阈值的版本**
（例如按幅度回归，或用逐模型自适应阈值），避免以后再用「幅度缩小 ⇒ P/R 崩塌」
这种伪影来误导结论。但**不能替换**现有 TH=-3.10 的定义，否则历史轮次不可比；
应以新增列的方式并存。

---

## 6. 复现命令

```bash
cd /home/neuron/L5PC/WModel2
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

python -u -m world.learn.train2 t2 \
  --data tr_s1 --val va_s2 --run t2v5_events_w5a2 --tier t2 --init t1v4_dz16/best.pt \
  --steps 9000 --batch 24 --length 128 --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
  --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 64 --cl-weight 1.0 --cl-weight-start 0.1 \
  --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.2 \
  --event-weight 5 --event-tau 0.2 --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
  --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 --sigma-gate 0.008 --mem-mb 3300

python -u -m tools.round_report --run t2v5_events_w5a2 --round R3_overshoot_fix \
  --title "R2 基础上 event-weight 10→5 + anchor-cl 0.1→0.2" --k 2000 --mem-mb 3200

# 单 trial vs 跨 trial 的对比（本轮的关键证据）
python -u -m tools.latent_ci --runs t2v5_ancC t2v5_events t2v5_events_w5a2 \
  --ks 500 1000 2000
```

"""当前 v4 实验的全局常量。

只保留 `sim2` / `secinfo` 真正使用的量：

* NEURON 温度、步长（`sim2` 会用自己的 `dt` 覆盖，这里供 θ 提取使用）
* 目标 section
* 状态 / 通道 / 区域清单（θ_s 的布局依赖它们）

实验窗口、采样协议、优化超参都不在这里：它们由
`world/gen/protocol2.py`、`world/gen/datagen2.py`、`world/learn/train2.py`
的命令行参数决定，避免出现两份互相矛盾的默认值。
"""
from __future__ import annotations

# ---------------------------------------------------------------- NEURON 常量
CELS = 34.0                 # °C
DT = 0.025                 # ms，v4 正式数据步长
SECTION = "apical_51"      # 目标 section

# ---------------------------------------------------------------- 状态定义
# 7 个慢门控 + cai + V；逐 compartment 记录（nseg=11 ⇒ 9×11=99 维）
STATE_GATES = (
    ("Ih", "m"),
    ("Ca_LVAst", "m"), ("Ca_LVAst", "h"),
    ("Ca_HVA", "h"),
    ("SKv3_1", "m"),
    ("SK_E2", "z"),
    ("Im", "m"),
)

# ---------------------------------------------------------------- θ_s 布局
CHANNEL_BARS = (
    "pas", "Ih", "Im", "Ca_HVA", "Ca_LVAst", "SK_E2", "SKv3_1", "NaTs2_t",
)
AXON_BARS = ("NaTa_t", "Nap_Et2", "K_Pst", "K_Tst")
REGIONS = ("basal", "apical", "axon", "soma")

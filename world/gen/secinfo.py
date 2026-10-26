"""NEURON 会话管理 + section 查找 + θ_s 提取 + 状态读取。

## 设计原则

1. **一个进程内只初始化一次 NEURON**（hoc 加载很慢，且重复加载会段错误）
2. **每次需要"干净"的段时，新建 cell 实例**（⚠️ 不能复用，否则
   `SEClamp` / 修改过的 `pas` 会残留；且旧 cell 被 GC 后
   `SectionUnit(sec)` 会报 `ReferenceError: can't access a deleted section`）
3. θ_s 的每个数字**都来自 `psection()` 实测**，不猜、不抄文档

## ⚠️ 踩过的坑（写在这里避免重犯）

- `sec.parentseg()` 而非 `sec.parent`（后者在无父时抛 `TypeError("No value")`）
- `psection()['density_mechs'][mech]['gXbar']` 返回**长度 = nseg 的 list**
- `cell.all` 是 `SectionList`，**没有 `len()`**，要用 `for s in cell.all`
- 每次新建 cell 后，**旧的 section 引用全部失效** ⇒ 不能跨调用缓存 `sec`
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from ..core import config as C

_H_READY = False


# ==============================================================================
# NEURON 初始化（只做一次）
# ==============================================================================
def init_neuron():
    """加载模型定义。⚠️ 顺序不能变，否则段错误。"""
    global _H_READY
    from neuron import h
    if not _H_READY:
        h.load_file("stdrun.hoc")
        h.load_file("import3d.hoc")
        h.load_file("L5PCbiophys5b.hoc")
        h.load_file("L5PCtemplate_2.hoc")
        h.dt = C.DT
        h.celsius = C.CELS
        _H_READY = True
    return h


def new_cell():
    """新建一个**干净**的细胞实例。返回 (cell, catalogs)。

    ⚠️ 必须每次都新建：修改过 `pas` 或加了 `SEClamp` 的 cell 不能复用。
    """
    h = init_neuron()
    cell = h.L5PCtemplate("morphologies/cell1.asc")
    from L5PC_exp import build_catalogs
    cat, _secsegs, _segs, _sortd = build_catalogs(cell)
    return cell, cat


def free_cell(cell) -> None:
    """删除整个细胞实例的所有 section。

    ⚠️ **必须调用**：每条协议都 `new_cell()`（196 个 section）。
    若累积 200 条协议 ⇒ 39200 个 section + 数万个 Vector，
    在 7.8 GB / 2 GB 可用内存的 WSL2 上会直接 OOM。
    """
    h = init_neuron()
    if cell is None:
        return
    for sec in list(cell.all):
        try:
            h.delete_section(sec=sec)
        except Exception:
            pass


def get_section(cell, cat, sid: str = C.SECTION):
    """按 section_id 找到 cell 里的 section 对象。"""
    row = cat[cat["section_id"] == sid]
    if len(row) == 0:
        raise KeyError(f"找不到 section_id={sid}")
    target = row.iloc[0]["sec_name"]
    short = target.split(".")[-1]
    for s in cell.all:
        if s.name().split(".")[-1] == short:
            return s, row.iloc[0]
    raise KeyError(f"cell 中没有 {target}")


# ==============================================================================
# 几何量
# ==============================================================================
def geometry(sec) -> dict:
    """几何事实（全部实测）。

    ⚠️ 直径沿段可能变化（taper），必须逐 compartment 取再看首尾。
    """
    xs = np.linspace(0.0, 1.0, int(sec.nseg))
    diam = np.array([float(sec(x).diam) for x in xs], dtype=float)
    L = float(sec.L)
    nseg = int(sec.nseg)
    Ra = float(sec.Ra)
    cm = float(sec.cm)

    # 轴向电阻（圆柱公式，用于交叉校验）
    d_mean_cm = float(diam.mean()) * 1e-4
    R_ax = Ra * (L * 1e-4) / (np.pi * (d_mean_cm / 2.0) ** 2) / 1e6   # MΩ

    return dict(
        L_um=L, nseg=nseg, Ra=Ra, cm=cm,
        diam_um=diam,                       # 逐 compartment
        diam_mean_um=float(diam.mean()),
        diam_cv=float(diam.std() / max(abs(diam.mean()), 1e-12)),  # 变异系数
        taper=float(diam[-1] / max(diam[0], 1e-12)),
        R_ax_Mohm=float(R_ax),
    )


# ==============================================================================
# 通道密度
# ==============================================================================
_BAR_KEYS = {
    "pas": "g", "Ih": "gIhbar", "Im": "gImbar", "Ca_HVA": "gCa_HVAbar",
    "Ca_LVAst": "gCa_LVAstbar", "SK_E2": "gSK_E2bar",
    "SKv3_1": "gSKv3_1bar", "NaTs2_t": "gNaTs2_tbar",
    "NaTa_t": "gNaTa_tbar", "Nap_Et2": "gNap_Et2bar",
    "K_Pst": "gK_Pstbar", "K_Tst": "gK_Tstbar",
}


def channel_bars(sec) -> dict:
    """逐机制的 gbar（S/cm²），逐 compartment 的 list → 取均值 + 变异系数。

    ⚠️ `psection()['density_mechs'][mech][key]` 返回**长度 = nseg 的 list**。
    ⚠️ 机制不存在时不应静默填 0 —— 要显式记「不存在」（后面编码成 −12）。
    """
    dm = sec.psection()["density_mechs"]
    out: dict[str, Any] = {}
    for mech in C.CHANNEL_BARS + C.AXON_BARS:
        key = _BAR_KEYS[mech]
        if mech in dm and key in dm[mech]:
            v = np.atleast_1d(np.asarray(dm[mech][key], dtype=float))
            out[mech] = dict(present=True,
                             mean=float(v.mean()),
                             cv=float(v.std() / max(abs(v.mean()), 1e-30)))
        else:
            out[mech] = dict(present=False, mean=0.0, cv=0.0)

    # CaDynamics 参数（记忆载体）
    cad = dict(present=False, gamma=0.0, decay=0.0, min_cai=0.0)
    if "CaDynamics_E2" in dm:
        d = dm["CaDynamics_E2"]
        cad = dict(present=True,
                   gamma=float(np.atleast_1d(d.get("gamma", [0.0])).mean()),
                   decay=float(np.atleast_1d(d.get("decay", [0.0])).mean()),
                   min_cai=float(np.atleast_1d(d.get("minCai", [0.0])).mean()))
    out["CaDynamics_E2"] = cad

    # pas 的反转电位
    e_pas = (float(np.atleast_1d(dm["pas"]["e"]).mean())
             if "pas" in dm and "e" in dm["pas"] else -90.0)
    out["pas"]["e"] = e_pas
    # Ca 通道的 cai0（nernst 参考）
    out["Ca_HVA"]["cai0"] = (float(np.atleast_1d(
        dm["Ca_HVA"].get("cai0_Ca_HVA", [5e-5])).mean())
        if "Ca_HVA" in dm else 5e-5)
    return out


# ==============================================================================
# θ_s 向量（定稿 v2 §3.1，27 维）
# ==============================================================================
LOGFLOOR = -12.0     # 通道不存在时 log10(gbar) 的编码（比最小的 1e-9 还小）


def _log10_pos(x: float, floor: float = 1e-12) -> float:
    return float(np.log10(max(abs(x), floor)))


def build_theta(sec, region: str) -> tuple[np.ndarray, dict]:
    """→ (θ_s: (27,) float32, 供人读的 dict)。

    布局（**顺序锁定，改动必须同步改 config.THETA_LAYOUT**）：
        0-3    几何:     log10 L, log10 d̄, taper, log10 nseg
        4-11   通道 8种: log10 gbar（不存在 → −12）
        12-15  axon 4种: 同上
        16-18  CaDyn:   log10 γ, log10 decay, log10 minCai
        19-22  被动:     log10 g_leak, e_leak, c_m, log10 Ra
        23-26  区域:     one-hot（basal, apical, axon, soma）
    """
    g = geometry(sec)
    ch = channel_bars(sec)

    th = np.zeros(27, dtype=np.float32)
    th[0] = _log10_pos(g["L_um"])
    th[1] = _log10_pos(g["diam_mean_um"])
    th[2] = float(g["taper"])
    th[3] = _log10_pos(float(g["nseg"]))

    for i, mech in enumerate(C.CHANNEL_BARS):
        th[4 + i] = (_log10_pos(ch[mech]["mean"]) if ch[mech]["present"]
                     else LOGFLOOR)
    for i, mech in enumerate(C.AXON_BARS):
        th[12 + i] = (_log10_pos(ch[mech]["mean"]) if ch[mech]["present"]
                      else LOGFLOOR)

    cad = ch["CaDynamics_E2"]
    th[16] = _log10_pos(cad["gamma"]) if cad["present"] else LOGFLOOR
    th[17] = _log10_pos(cad["decay"]) if cad["present"] else LOGFLOOR
    th[18] = _log10_pos(cad["min_cai"]) if cad["present"] else LOGFLOOR

    th[19] = _log10_pos(ch["pas"]["mean"])
    th[20] = float(ch["pas"]["e"])
    th[21] = float(g["cm"])
    th[22] = _log10_pos(g["Ra"])

    for i, r in enumerate(C.REGIONS):
        th[23 + i] = 1.0 if r == region else 0.0

    readable = dict(section_id=C.SECTION,
                    region=region,
                    geom={k: v for k, v in g.items() if k != "diam_um"},
                    diam_um=g["diam_um"].tolist(),
                    channels={k: v for k, v in ch.items()},
                    theta=th.tolist())
    return th, readable


THETA_LAYOUT = (
    ["log10_L", "log10_d", "taper", "log10_nseg"]
    + [f"log10_g_{m}" for m in C.CHANNEL_BARS]
    + [f"log10_g_{m}" for m in C.AXON_BARS]
    + ["log10_gamma", "log10_decay", "log10_minCai"]
    + ["log10_g_leak", "e_leak", "cm", "log10_Ra"]
    + [f"region_{r}" for r in C.REGIONS]
)
assert len(THETA_LAYOUT) == 27, len(THETA_LAYOUT)


# ==============================================================================
# 状态读取（从 SectionUnit 的实例读）
# ==============================================================================
def read_state(u, with_v: bool = False) -> tuple[np.ndarray, list[str]]:
    """从 `SectionUnit` 读状态向量。

    ⚠️ 导频点 P=5 会把 nseg=11 的空间分布压成 5 点。
    本方案要求**逐 compartment**（实测：池化到 5 点最差误差 94.4%），
    所以这里用 `u.V`（nseg 维）与 `u.gates`（P 维）……

    ⚠️⚠️ **重要限制**：`SectionUnit` 的门控只存在于 P 个**导频点**上
    （`state_dim = n_gate*P + 1`，与 nseg 无关）。这是它的设计（为了让
    状态维与 nseg 解耦）。

    因此本模块**暂时只能用导频点的门控** + **逐 compartment 的 V**。
    若后续要逐 compartment 的门控，必须改用 NEURON 原生
    `seg.hX` 读取（`_ref_h` 等），成本高但可行。
    -> 已记录为待办，见 ensure_state_layout()。
    """
    names: list[str] = []
    parts: list[np.ndarray] = []

    if with_v:
        V = np.asarray(u.V, dtype=float)
        parts.append(V)
        names += [f"V[{i}]" for i in range(len(V))]

    for key in C.STATE_GATES:
        if key in u.gates:
            g = np.asarray(u.gates[key], dtype=float)
            parts.append(g)
            names += [f"{key[0]}.{key[1]}[p{i}]" for i in range(len(g))]

    cai = np.atleast_1d(np.asarray(u.cai, dtype=float))
    parts.append(cai)
    names += [f"cai[p{i}]" for i in range(len(cai))]

    return np.concatenate(parts).astype(np.float32), names


def state_layout(nseg: int, P: int, with_v: bool = False) -> list[str]:
    names = [f"V[{i}]" for i in range(nseg)] if with_v else []
    for m, gname in C.STATE_GATES:
        names += [f"{m}.{gname}[p{i}]" for i in range(P)]
    names += [f"cai[p{i}]" for i in range(P)]
    return names


# ==============================================================================
# 给探针用的自检
# ==============================================================================
def selfcheck() -> dict:
    cell, cat = new_cell()
    sec, row = get_section(cell, cat)
    th, readable = build_theta(sec, str(row["region"]))
    g = readable["geom"]
    return dict(
        section=sec.name(), region=row["region"], nseg=g["nseg"],
        L_um=g["L_um"], diam_mean_um=g["diam_mean_um"], taper=g["taper"],
        R_ax_Mohm=g["R_ax_Mohm"], theta_dim=len(th),
        theta_nonzero=int((th > LOGFLOOR + 1e-6).sum()),
        channels_present=[m for m in C.CHANNEL_BARS + C.AXON_BARS
                          if readable["channels"][m]["present"]],
    )


if __name__ == "__main__":
    print(json.dumps(selfcheck(), ensure_ascii=False, indent=2))

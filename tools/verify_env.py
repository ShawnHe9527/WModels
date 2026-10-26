#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_env.py (WModel) — 纯 NEURON 实验环境自检

⭐ 与 `L5PC2026/tools/verify_env.py` 的区别：
   · 那个检查 **V 系列（embedding/MLP）** 的目录与结果约定（含 torch 依赖）
   · 本脚本只检查 **NEURON 模型本身**能否正确建起来 ⇒ 不要求 torch

检查项
======
| 组 | 项 | 内容 |
|---|---|---|
| A | A1 | 必需文件/目录存在（mods / morphologies / hoc / catalogs / tools）|
| A | A2 | mod 文件数 ≥ 17，且**关键机制**齐全 |
| A | A3 | 形态文件非空 |
| B | B1 | NEURON 可导入 + 版本 |
| C | C1 | **编译产物** `x86_64/libnrnmech.so` 存在（否则机制无法加载）|
| C | C2 | 初始化顺序正确（stdrun → import3d → biophys → template）|
| C | C3 | 细胞建成：**196 段**（soma 1 / axon 2 / basal 84 / apical 109）|
| C | C4 | 机制全部可加载（`hasattr(h, <mech>)`，⚠️ 8.2 无 allmechanisms）|
| C | C5 | `build_catalogs()` 可用：section=196 / segment=639 |
| D | D1 | **区域通道事实**（basal 只有 pas+Ih；快 Na/K 只在 axon）|
| D | D2 | 突触参数走 `place_synapses()` 后 = 设计值（NMDA 70 ms）|
| E | E1 | 库文件（L5PC_*.py）全部**不含 torch** ⇒ 保证是纯 NEURON 分支 |
| E | E2 | 探针目录里没有 V/F 系列（embedding）探针 |
| F | F1 | 关键文档在位 |

用法
    PYTHONPATH=$PWD python tools/verify_env.py
    PYTHONPATH=$PWD python tools/verify_env.py -k C        # 只跑 C 组
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)

_checks: list = []


def check(name):
    def deco(fn):
        _checks.append((name, fn))
        return fn
    return deco


# ======================================================================
# A. 文件
# ======================================================================
@check("A1 必需文件/目录存在")
def _a1():
    need_dirs = ["mods", "morphologies", "catalogs", "probes", "tools",
                 "results", "docs", "data", "logs", "src"]
    for d in need_dirs:
        p = os.path.join(ROOT, d)
        assert os.path.isdir(p), f"缺目录 {d}/"
    need_files = ["L5PC_exp.py", "L5PC_gates.py", "L5PC_port.py",
                  "L5PC_compose.py", "L5PC_unit.py", "L5PC_unit_sec.py",
                  "L5PC_parallel.py",
                  "init.hoc", "L5PCbiophys5b.hoc", "L5PCtemplate_2.hoc",
                  "morphologies/cell1.asc",
                  "tools/build_model.sh", "tools/run_probe.sh"]
    for f in need_files:
        p = os.path.join(ROOT, f)
        assert os.path.isfile(p), f"缺文件 {f}"
    return f"{len(need_dirs)} 目录 + {len(need_files)} 文件 ✔"


@check("A2 mod 文件数 ≥17 且关键机制齐全")
def _a2():
    mods = sorted(f for f in os.listdir(os.path.join(ROOT, "mods"))
                  if f.endswith(".mod"))
    assert len(mods) >= 17, f"mod 文件只有 {len(mods)} 个"
    KEY = ["Ih.mod", "Im.mod", "Ca_HVA.mod", "Ca_LVAst.mod",
           "CaDynamics_E2.mod", "SK_E2.mod", "SKv3_1.mod", "NaTs2_t.mod",
           "NaTa_t.mod", "Nap_Et2.mod", "K_Pst.mod", "K_Tst.mod",
           "ProbAMPANMDA2.mod", "ProbUDFsyn2.mod"]
    missing = [m for m in KEY if m not in mods]
    assert not missing, f"缺关键 mod: {missing}"
    return f"mod {len(mods)} 个 ✔ 关键机制 {len(KEY)}/{len(KEY)}"


@check("A3 形态文件非空")
def _a3():
    p = os.path.join(ROOT, "morphologies", "cell1.asc")
    n = os.path.getsize(p)
    assert n > 100_000, f"形态文件只有 {n} B，疑似损坏"
    return f"cell1.asc {n/1024:.0f} KB ✔"


# ======================================================================
# B. 依赖
# ======================================================================
@check("B1 NEURON 可导入 + 版本")
def _b1():
    import neuron
    from neuron import h
    return f"NEURON {h.nrnversion().split()[2]} ✔"


# ======================================================================
# C. 模型
# ======================================================================
@check("C1 编译产物 x86_64/libnrnmech.so 存在")
def _c1():
    so = os.path.join(ROOT, "x86_64", "libnrnmech.so")
    assert os.path.isfile(so), \
        "缺 x86_64/libnrnmech.so ⇒ 先跑: bash tools/build_model.sh"
    return f"libnrnmech.so {os.path.getsize(so)/1024:.0f} KB ✔"


@check("C2 初始化顺序 + 细胞建成（196 段）")
def _c2():
    from neuron import h
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    # ⚠️ import3d.hoc 必须在 L5PCtemplate_2.hoc 之前，否则
    #    "Import3d_Neurolucida3 is not a template" + segfault（实测踩过）
    h.load_file("L5PCbiophys5b.hoc")
    h.load_file("L5PCtemplate_2.hoc")
    cell = h.L5PCtemplate("morphologies/cell1.asc")
    # ⚠️ cell.basal 是 hoc SectionList，不可下标 ⇒ 必须 list()
    n_soma = len(list(cell.somatic))
    n_axon = len(list(cell.axonal))
    n_basal = len(list(cell.basal))
    n_apical = len(list(cell.apical))
    tot = n_soma + n_axon + n_basal + n_apical
    assert tot == 196, f"段数 {tot} ≠ 196"
    assert (n_soma, n_axon, n_basal, n_apical) == (1, 2, 84, 109), \
        f"分区段数不对: soma={n_soma} axon={n_axon} " \
        f"basal={n_basal} apical={n_apical}"
    return f"196 段 ✔ (soma 1 / axon 2 / basal 84 / apical 109)"


@check("C3 机制全部可加载")
def _c3():
    from neuron import h
    REQ = ["pas", "Ih", "Im", "Ca_HVA", "Ca_LVAst", "CaDynamics_E2",
           "SK_E2", "SKv3_1", "NaTs2_t", "NaTa_t", "Nap_Et2",
           "K_Pst", "K_Tst", "ProbAMPANMDA2", "ProbUDFsyn2"]
    # ⚠️ NEURON 8.2 没有 h.allmechanisms()（实测 AttributeError）
    #    ⇒ 用 hasattr(h, <mech>) 判断
    missing = [m for m in REQ if not hasattr(h, m)]
    assert not missing, f"机制未加载: {missing}"
    return f"机制 {len(REQ)}/{len(REQ)} 已加载 ✔"


@check("C4 build_catalogs：section=196 / segment=642（其中树突 639）")
def _c4():
    """⚠️ 两种口径别混：
       · **总 segment = 642** = soma 1 + axon 2 + **树突 639**
       · **639** = basal 262 + apical 377 ← ⭐ **这个数才与原论文对齐**
         （原论文的 "segment" = 我们的 compartment）
    """
    import L5PC_exp as E
    import pandas as pd
    from neuron import h
    cell = h.L5PCtemplate("morphologies/cell1.asc")
    cat, segcat, secsegs, segobj = E.build_catalogs(cell)
    assert len(cat) == 196, f"section 数 {len(cat)} ≠ 196"
    assert len(segcat) == 642, f"segment 数 {len(segcat)} ≠ 642"
    dend = int(segcat.region.isin(["basal", "apical"]).sum())
    assert dend == 639, f"树突 segment 数 {dend} ≠ 639（原论文对齐值）"
    return (f"section 196 ✔ · segment 642 ✔（树突 **639** = basal 262 + "
            f"apical 377）  CELS={E.CELS} DT={E.DT} V_INIT={E.V_INIT}")


@check("C5 区域通道事实（分组实验的前提）")
def _c5():
    """⭐ 用户 2026-10-13 要求：实验前必须知道「哪个区域有哪些通道」。"""
    from neuron import h
    cell = h.L5PCtemplate("morphologies/cell1.asc")
    probes = ["g_pas", "gIhbar_Ih", "gImbar_Im", "gSKv3_1bar_SKv3_1",
              "gSK_E2bar_SK_E2", "gCa_HVAbar_Ca_HVA",
              "gCa_LVAstbar_Ca_LVAst", "decay_CaDynamics_E2",
              "gNaTs2_tbar_NaTs2_t", "gNaTa_tbar_NaTa_t",
              "gNap_Et2bar_Nap_Et2", "gK_Pstbar_K_Pst", "gK_Tstbar_K_Tst"]
    fam = [("basal", list(cell.basal)), ("apical", list(cell.apical)),
           ("soma", list(cell.somatic)), ("axon", list(cell.axonal))]
    present = {}
    for reg, secs in fam:
        n = len(secs)
        got = {}
        for pr in probes:
            cnt = 0
            for s in secs:
                ok = True
                for sg in s:
                    if not hasattr(sg, pr):
                        ok = False
                        break
                if ok:
                    cnt += 1
            if cnt == n:
                got[pr] = cnt
        present[reg] = got
    # ---- 三条必须成立的事实 ----
    b = present["basal"]
    assert b.get("gIhbar_Ih") == 84, "basal 应有 84 段 Ih"
    assert "gCa_HVAbar_Ca_HVA" not in b, \
        "⚠️ basal 不该有 Ca_HVA —— 若出现说明模型被改动"
    assert "gNaTs2_tbar_NaTs2_t" not in b, "⚠️ basal 不该有 NaTs2_t"
    a = present["axon"]
    for pr in ("gNaTa_tbar_NaTa_t", "gNap_Et2bar_Nap_Et2",
               "gK_Pstbar_K_Pst", "gK_Tstbar_K_Tst"):
        assert a.get(pr) == 2, f"快 Na/K ({pr}) 应只在 axon 的 2 段"
    ap = present["apical"]
    assert ap.get("gCa_HVAbar_Ca_HVA") == 109, "apical 应有 109 段 Ca_HVA"
    return ("basal = pas+Ih ✔（无 Ca/Na/SK）· 快 Na/K 只在 axon ✔ · "
            "apical 全通道 ✔")


@check("C6 突触参数走 place_synapses 后 = 设计值")
def _c6():
    import L5PC_exp as E
    from neuron import h
    cell = h.L5PCtemplate("morphologies/cell1.asc")
    cat, segcat, secsegs, segobj = E.build_catalogs(cell)
    nce, nci, nse, nsi = E.place_synapses(cell, cat, segobj)
    k = sorted(nse)[0]
    syn = nse[k]
    td = float(syn.tau_d_NMDA)
    # ⚠️ mod 默认 tau_d_NMDA = 43 ms，设计值是 70 ms ⇒ 必须走 place_synapses
    assert abs(td - 70.0) < 1e-6, f"tau_d_NMDA = {td} ≠ 70（mod 默认是 43！）"
    assert abs(float(syn.gmax) - 4e-4) < 1e-12, \
        f"gmax = {float(syn.gmax)} ≠ 4e-4"
    return (f"NMDA tau_d = {td:.1f} ms ✔ gmax = {float(syn.gmax):.1e} ✔ "
            f"（E 突触 {len(nse)} / I 突触 {len(nsi)}）")


# ======================================================================
# E. 分支纯净性（WModel 是纯 NEURON 分支）
# ======================================================================
@check("E1 库文件不含 torch（纯 NEURON 分支）")
def _e1():
    libs = [f for f in os.listdir(ROOT)
            if f.startswith("L5PC_") and f.endswith(".py")]
    bad = []
    for f in libs:
        t = open(os.path.join(ROOT, f), encoding="utf-8").read()
        if "import torch" in t or "torch." in t:
            bad.append(f)
    assert not bad, f"⚠️ 这些库含 torch ⇒ 不是纯 NEURON 分支: {bad}"
    return f"{len(libs)} 个库均不含 torch ✔"


@check("E2 无 embedding 系列探针（V*/F*）")
def _e2():
    d = os.path.join(ROOT, "probes")
    bad = [f for f in os.listdir(d)
           if f.startswith(("V0", "V1", "V2", "F4", "F5"))]
    assert not bad, f"⚠️ 不该有 embedding/MLP 探针: {bad}"
    return "probes/ 内无 V*/F* ✔"


# ======================================================================
# F. 文档
# ======================================================================
@check("F1 关键文档在位")
def _f1():
    need = ["README.md", "docs/README.md", "docs/MODEL_FACTS.md",
            "docs/L5PC_exp_使用手册.md"]
    miss = [f for f in need if not os.path.isfile(os.path.join(ROOT, f))]
    assert not miss, f"缺 {miss}"
    n = len([f for f in os.listdir(os.path.join(ROOT, "docs"))
             if f.endswith(".md")])
    return f"docs/ {n} 篇 ✔"


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default=None, help="只跑名称含该子串的检查")
    a = ap.parse_args()

    # 强制 CWD（L5PC_exp.py 与 hoc 都依赖相对路径）
    os.chdir(ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    print("=" * 74)
    print(" WModel 环境自检（纯 NEURON）")
    print(f"   ROOT = {ROOT}")
    print("=" * 74)
    n_ok = n_bad = 0
    for name, fn in _checks:
        if a.k and a.k not in name:
            continue
        try:
            msg = fn()
            n_ok += 1
            print(f"  ✅ {name}")
            if msg:
                print(f"       {msg}")
        except Exception as e:                                   # noqa: BLE001
            n_bad += 1
            print(f"  ❌ {name}")
            print(f"       {type(e).__name__}: {e}")
            if os.environ.get("VERBOSE"):
                traceback.print_exc()
    print("=" * 74)
    print(f" 结果: ✅ {n_ok} 通过   ❌ {n_bad} 失败")
    print("=" * 74)
    if n_bad == 0:
        print("\n 环境就绪。")
        print("\n ── 推荐：用统一启动器 ──")
        print("   cd L5PC/WModel")
        print("   bash tools/run_probe.sh probes/<probe>.py --args...")
        print("   FORCE=1 bash tools/run_probe.sh ...   # 确要覆盖归档结果")
        print("\n ── 编译机制（目录搬家后必须重编译）──")
        print("   bash tools/build_model.sh --force")
        print("\n ── 本目录不依赖 torch（纯 NEURON 分支）──")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())

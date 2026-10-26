#!/usr/bin/env python3
"""
L5PC_exp.py — L5PC 多隔室模型的编号、刺激控制与数据产出（P0）
=============================================================
基于 Hay et al. 2011 / Beniaguev et al. 2021 的 L5b 锥体细胞模型。

本脚本实现三类目标：
  【1】Section 编号系统 —— 胞体 / 轴突 / 基树突 / 顶树突 分开编号，
       并提供 section -> segment 的完整映射，便于自由选取与刺激。
  【2】Section 级刺激控制 —— 延续现有 segment 分割策略（nseg = 1+2*int(L/40)），
       但以 section 为控制节点：激活某 section = 激活它内部所有 segment。
  【3】一次性产出三种数据：
       (a) 阈下特征   —— 每个 section 的平均阈下电压 / 峰值电压 / 去极化面积（连续值）
       (b) 二值激活   —— 每个 section 是否发生超阈值发放（0/1）
       (c) 胞体标签   —— 胞体是否放电（0/1） + 胞体阈下电压轨迹

编号约定：
  Section ID:  {region}_{i}        例如  basal_3  apical_50  soma_0  axon_1
  Segment ID:  {region}_{i}_{j}    例如  basal_3_0  = 基树突第3个section、最靠近胞体的第1个segment
           j 从 0 开始，j=0 表示最靠近胞体（x=0 端），j 越大越靠近树突末梢。

运行方式（需使用带 NEURON 的 Python 环境）：
  cd L5PC2021
  python L5PC_exp.py                      # 单次仿真 + 提取特征
  python L5PC_exp.py --trials 20          # 多次随机仿真，生成小数据集
  python L5PC_exp.py --duration 2000      # 自定义仿真时长（ms）

依赖：L5PCbiophys5b.hoc / L5PCtemplate_2.hoc / morphologies/cell1.asc / mods/
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境下保存图片
import matplotlib.pyplot as plt

# ============================================================
# 环境配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from neuron import h, gui

h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")

# ============================================================
# 全局可调参数
# ============================================================
SPIKE_THRESHOLD   = -20.0   # 超阈值发放判据 (mV)，与 run_L5PC_gui.count_spikes 一致
REST_VOLTAGE      = -76.0   # 静息电位基线 (mV)，用于计算去极化面积
V_INIT            = -76.0   # 仿真初始化膜电位
DT                = 0.025   # 积分步长 (ms)
CELS              = 34.0    # 温度 (deg C)

# ============================================================
# 1) Section / Segment 编号系统
# ============================================================
# 区域遍历顺序（决定全局索引），soma/axon 在前，basal/apical 树突在后
REGION_ORDER = ["soma", "axon", "basal", "apical"]

# 区域 -> 模板 SectionList 名称
REGION_TO_LIST = {
    "soma":   "somatic",
    "axon":   "axonal",
    "basal":  "basal",
    "apical": "apical",
}


def _sections_in_region(cell, region):
    """返回某区域的 Section 对象列表（按 NEURON 内部顺序）。"""
    seclist = getattr(cell, REGION_TO_LIST[region])
    return list(seclist)


def build_catalogs(cell):
    """
    构建两个 DataFrame：
      section_catalog : 每个 section 一行，含编号、长度、nseg、距胞体距离等
      segment_catalog : 每个 segment 一行，含编号、所属 section、x 位置、距胞体距离等
    并返回辅助字典 section_id -> segment_id 列表、segment_id -> (section, seg) 对象。
    """
    # 先设置距离原点在胞体中心，使 h.distance 返回"距胞体的路径距离"
    soma_sec = _sections_in_region(cell, "soma")[0]
    h.distance(0, 0.5, sec=soma_sec)

    section_rows = []
    segment_rows = []
    section_segments = {}   # section_id -> [segment_id, ...]
    segment_obj = {}        # segment_id -> (Section, seg)

    global_idx = 0
    for region in REGION_ORDER:
        secs = _sections_in_region(cell, region)
        for i, sec in enumerate(secs):
            section_id = f"{region}_{i}"
            # 该 section 近端（x=0）距胞体的路径距离
            prox_dist = h.distance(0.0, sec=sec)
            dist_dist = h.distance(1.0, sec=sec)

            section_rows.append({
                "section_id":     section_id,
                "region":         region,
                "region_index":   i,
                "global_index":   global_idx,
                "nseg":           sec.nseg,
                "length_um":      sec.L,
                "proximal_dist_um": prox_dist,
                "distal_dist_um":   dist_dist,
                "sec_name":       sec.name(),
            })

            seg_ids = []
            for j, seg in enumerate(sec):
                seg_id = f"{section_id}_{j}"
                seg_ids.append(seg_id)
                segment_obj[seg_id] = (sec, seg)
                # 该 segment 中心距胞体的路径距离
                seg_dist = h.distance(seg.x, sec=sec)
                segment_rows.append({
                    "segment_id":       seg_id,
                    "section_id":       section_id,
                    "region":           region,
                    "region_index":     i,
                    "seg_index_in_sec": j,   # 0 = 近胞体端
                    "x":                seg.x,
                    "distance_um":      seg_dist,
                })

            section_segments[section_id] = seg_ids
            global_idx += 1

    section_catalog = pd.DataFrame(section_rows)
    segment_catalog = pd.DataFrame(segment_rows)
    return section_catalog, segment_catalog, section_segments, segment_obj


# ============================================================
# 2) 突触放置（每 segment 一对兴奋 + 抑制，与 run_L5PC_gui 一致）
# ============================================================
def place_synapses(cell, section_catalog, segment_obj):
    """
    在每个树突 segment（basal + apical）放置 1 个兴奋性 + 1 个抑制性突触。
    返回：exc/ inh 的 NetCon 列表（按 segment_id 索引），以及 syn 对象。
    """
    ex_netcons = {}   # segment_id -> NetCon
    inh_netcons = {}
    ex_synapses = {}
    inh_synapses = {}

    # 直接遍历 segment_obj，在树突 segment 上放置突触
    for seg_id, (sec, seg) in segment_obj.items():
        # 仅在树突上放置突触
        if "dend" not in sec.name() and "apic" not in sec.name():
            continue

        # 兴奋性突触（NMDA 为主，短时可塑性 off）
        syn_ex = h.ProbAMPANMDA2(seg)
        syn_ex.tau_r_AMPA = 0.3
        syn_ex.tau_d_AMPA = 3.0
        syn_ex.tau_r_NMDA = 2.0
        syn_ex.tau_d_NMDA = 70.0
        syn_ex.gmax = 0.0004
        syn_ex.e = 0
        syn_ex.Use = 1.0
        syn_ex.u0 = 0.0
        syn_ex.Dep = 0.0
        syn_ex.Fac = 0.0

        nc_ex = h.NetCon(None, syn_ex)
        nc_ex.delay = 0
        nc_ex.weight[0] = 1.0

        # 抑制性突触（GABA_A，短时可塑性 off）
        syn_inh = h.ProbUDFsyn2(seg)
        syn_inh.tau_r = 0.2
        syn_inh.tau_d = 8.0
        syn_inh.gmax = 0.001
        syn_inh.e = -80
        syn_inh.Use = 1.0
        syn_inh.u0 = 0.0
        syn_inh.Dep = 0.0
        syn_inh.Fac = 0.0

        nc_inh = h.NetCon(None, syn_inh)
        nc_inh.delay = 0
        nc_inh.weight[0] = 1.0

        ex_netcons[seg_id] = nc_ex
        inh_netcons[seg_id] = nc_inh
        ex_synapses[seg_id] = syn_ex
        inh_synapses[seg_id] = syn_inh

    return ex_netcons, inh_netcons, ex_synapses, inh_synapses


# ============================================================
# 3) Section 级刺激控制
# ============================================================
def make_poisson_schedule(section_ids, segment_ids, section_segments,
                          ex_netcons, inh_netcons,
                          duration_ms, ex_rate_hz, inh_rate_hz, seed):
    """
    生成一个 schedule 函数，对选定 section（展开为所有 segment）和/或
    直接指定的 segment 施加 Poisson 脉冲输入。

    参数：
      section_ids  : 要激活的 section 列表，如 ['basal_3', 'apical_50']
      segment_ids  : 要直接激活的 segment 列表，如 ['apical_50_0', 'apical_50_2']
      ex_rate_hz   : 每个 segment 的兴奋性平均发放率（Hz）
      inh_rate_hz  : 每个 segment 的抑制性平均发放率（Hz）
    返回：可直接交给 h.FInitializeHandler 的 schedule 函数。
    """
    rng = np.random.default_rng(seed)

    # 展开：section -> 所有 segment；segment 直接加入；最后去重
    target_segs = []
    for sid in section_ids:
        target_segs.extend(section_segments[sid])
    for seg_id in segment_ids:
        target_segs.append(seg_id)
    target_segs = list(dict.fromkeys(target_segs))  # 去重，保持顺序

    ex_spikes = {}   # segment_id -> spike 时间列表
    inh_spikes = {}
    for seg_id in target_segs:
        n_ex = rng.poisson(ex_rate_hz * duration_ms / 1000.0)
        if n_ex > 0:
            ex_spikes[seg_id] = sorted(rng.uniform(0, duration_ms, n_ex).tolist())
        n_inh = rng.poisson(inh_rate_hz * duration_ms / 1000.0)
        if n_inh > 0:
            inh_spikes[seg_id] = sorted(rng.uniform(0, duration_ms, n_inh).tolist())

    def schedule():
        for seg_id, times in ex_spikes.items():
            for t in times:
                ex_netcons[seg_id].event(t)
        for seg_id, times in inh_spikes.items():
            for t in times:
                inh_netcons[seg_id].event(t)

    return schedule


# ============================================================
# 4) 仿真与数据提取
# ============================================================
class L5PCExperiment:
    """封装：加载模型 -> 编号 -> 放突触 -> 刺激 -> 运行 -> 提取特征。"""

    def __init__(self, morphology_file="morphologies/cell1.asc"):
        print("=" * 60)
        print(" 加载 L5PC 模型 (Hay et al. 2011 / Beniaguev et al. 2021)")
        print("=" * 60)

        h.load_file("L5PCbiophys5b.hoc")
        h.load_file("L5PCtemplate_2.hoc")

        self.cell = h.L5PCtemplate(morphology_file)
        self.soma_sec = _sections_in_region(self.cell, "soma")[0]

        # 编号系统
        (self.section_catalog, self.segment_catalog,
         self.section_segments, self.segment_obj) = build_catalogs(self.cell)

        # 突触放置
        (self.ex_netcons, self.inh_netcons,
         self.ex_synapses, self.inh_synapses) = place_synapses(
             self.cell, self.section_catalog, self.segment_obj)

        # 全局参数
        h.celsius = CELS
        h.dt = DT

        # 统计信息
        n_sec = len(self.section_catalog)
        n_seg = len(self.segment_catalog)
        n_exc = len(self.ex_netcons)
        print(f"[INFO] 模型加载完成: {n_sec} sections, {n_seg} segments, "
              f"{n_exc} 个树突 segment 各含一对突触")
        for region in REGION_ORDER:
            cnt = (self.section_catalog["region"] == region).sum()
            print(f"    {region:6s}: {cnt} sections")

    # ---- 编号查询辅助 ----
    def section_ids(self, region=None):
        """返回 section_id 列表，可按 region 过滤。"""
        if region is None:
            return self.section_catalog["section_id"].tolist()
        return self.section_catalog.loc[
            self.section_catalog["region"] == region, "section_id"
        ].tolist()

    def segments_of(self, section_id):
        """返回某 section 的所有 segment_id 列表（j 从近胞体端到远端）。"""
        return self.section_segments[section_id]

    def show_section(self, section_id):
        """打印某 section 的详细信息（便于识别/选取）。"""
        row = self.section_catalog[
            self.section_catalog["section_id"] == section_id
        ].iloc[0]
        print(f"Section {section_id}: region={row['region']} "
              f"L={row['length_um']:.1f}um nseg={row['nseg']} "
              f"距胞体 {row['proximal_dist_um']:.0f}~{row['distal_dist_um']:.0f}um")
        for seg_id in self.segments_of(section_id):
            seg = self.segment_catalog[
                self.segment_catalog["segment_id"] == seg_id
            ].iloc[0]
            print(f"    {seg_id:16s}  x={seg['x']:.3f}  "
                  f"距胞体 {seg['distance_um']:.1f}um")

    # ---- 刺激 + 运行 ----
    def run(self, section_ids=None, segment_ids=None, duration_ms=1000,
            ex_rate_hz=500.0, inh_rate_hz=400.0, seed=42):
        """
        运行一次仿真。
          section_ids : 要激活的 section（会展开为其所有 segment）
          segment_ids : 要直接激活的 segment
          两者都为空时，默认激活所有树突 section。
          返回本次仿真的原始记录 dict（含每 segment 电压轨迹等）。
        """
        if section_ids is None and segment_ids is None:
            section_ids = self.section_ids("basal") + self.section_ids("apical")

        section_ids = section_ids or []
        segment_ids = segment_ids or []

        # 实际被激活的 segment 列表（展开 section + 直接 segment，去重）
        active_segments = list(dict.fromkeys(
            [s for sid in section_ids for s in self.section_segments[sid]]
            + list(segment_ids)
        ))

        t_start = time.time()

        # 建立刺激 schedule
        scheduler = make_poisson_schedule(
            section_ids, segment_ids, self.section_segments,
            self.ex_netcons, self.inh_netcons,
            duration_ms, ex_rate_hz, inh_rate_hz, seed)
        fih = h.FInitializeHandler(scheduler)

        # 记录向量（每次新建，避免累积）
        rec_t = h.Vector().record(h._ref_t)
        rec_v_soma = h.Vector().record(self.soma_sec(0.5)._ref_v)

        # 记录所有树突 segment 电压（用于 section 特征）
        seg_voltage_vecs = {}
        for seg_id, (sec, seg) in self.segment_obj.items():
            if "dend" in sec.name() or "apic" in sec.name():
                seg_voltage_vecs[seg_id] = h.Vector().record(seg._ref_v)

        # 运行
        h.tstop = duration_ms
        h.finitialize(V_INIT)
        h.continuerun(duration_ms)

        elapsed = time.time() - t_start
        print(f"[INFO] 仿真完成: {duration_ms}ms, 耗时 {elapsed:.2f}s, "
              f"激活 {len(section_ids)} 个 section + {len(segment_ids)} 个 segment "
              f"= {len(active_segments)} 个 segment")

        # 转为 numpy
        time_arr = np.array(rec_t)
        soma_v = np.array(rec_v_soma)
        seg_voltages = {sid: np.array(v) for sid, v in seg_voltage_vecs.items()}

        return {
            "time": time_arr,
            "soma_v": soma_v,
            "seg_voltages": seg_voltages,
            "active_sections": list(section_ids),
            "active_segments": active_segments,
        }

    # ---- 特征提取 ----
    def extract_features(self, record):
        """
        从一次仿真的记录中提取三类数据：
          (a) section 阈下特征   (连续值)
          (b) section 二值激活   (0/1)
          (c) 胞体放电标签 + 胞体阈下电压轨迹
        """
        time_arr = record["time"]
        soma_v = record["soma_v"]
        seg_voltages = record["seg_voltages"]

        dt = time_arr[1] - time_arr[0] if len(time_arr) > 1 else DT

        def sub_clip(v):
            """阈下电压：把超过发放阈值的部分截掉（保持阈下）。"""
            return np.clip(v, None, SPIKE_THRESHOLD)

        def depolarization_area(v):
            """去极化面积：∫ max(0, V - V_rest) dt。"""
            return float(np.sum(np.maximum(v - REST_VOLTAGE, 0.0)) * dt)

        # ---- 胞体标签 ----
        soma_peak = float(soma_v.max())
        soma_spike = int(soma_peak > SPIKE_THRESHOLD)
        soma_sub_v = sub_clip(soma_v)

        # ---- 逐 segment 计算，再聚合到 section ----
        seg_mean_sub = {}
        seg_peak = {}
        seg_area = {}
        for seg_id, v in seg_voltages.items():
            seg_mean_sub[seg_id] = float(sub_clip(v).mean())
            seg_peak[seg_id] = float(v.max())
            seg_area[seg_id] = depolarization_area(v)

        section_rows = []
        for _, srow in self.section_catalog.iterrows():
            sid = srow["section_id"]
            region = srow["region"]
            seg_ids = self.section_segments[sid]

            if region in ("basal", "apical"):
                # 树突 section：聚合其所有 segment
                means = [seg_mean_sub[s] for s in seg_ids]
                peaks = [seg_peak[s] for s in seg_ids]
                areas = [seg_area[s] for s in seg_ids]
                mean_sub = float(np.mean(means))
                peak_v = float(np.max(peaks))
                area = float(np.mean(areas))
                activated = int(peak_v > SPIKE_THRESHOLD)
            else:
                # soma / axon：暂无突触，记为 0（可后续扩展）
                mean_sub = float(sub_clip(soma_v).mean()) if region == "soma" else 0.0
                peak_v = soma_peak if region == "soma" else 0.0
                area = depolarization_area(soma_v) if region == "soma" else 0.0
                activated = soma_spike if region == "soma" else 0

            section_rows.append({
                "section_id": sid,
                "region": region,
                "mean_subthreshold_v": mean_sub,
                "peak_v": peak_v,
                "depolarization_area": area,
                "activated": activated,
            })

        section_features = pd.DataFrame(section_rows)

        return {
            "section_features": section_features,
            "soma_spike": soma_spike,
            "soma_sub_v": soma_sub_v,
            "soma_v": soma_v,
            "time": time_arr,
        }


# ============================================================
# 4.5) 可视化
# ============================================================
def plot_trial(exp, record, features, plot_section_ids=None,
               plot_segment_ids=None, save_path=None,
               show_spike_threshold=True):
    """
    绘制一次仿真的可视化结果：
      左图：胞体电压轨迹（标注是否放电 + 发放阈值线）
      中间：选定 section 内部各 segment 的电压轨迹（叠加）
      右侧：直接指定的单个 segment 电压轨迹

    参数：
      exp             : L5PCExperiment 实例
      record          : exp.run() 返回的记录 dict
      features        : exp.extract_features() 返回的特征 dict
      plot_section_ids: 要绘制的 section 列表（每个一个子图）
      plot_segment_ids: 要绘制的 segment 列表（每个一个子图）
      save_path       : 保存路径（None 则只返回 fig）
      show_spike_threshold: 是否画发放阈值线
    """
    time_arr = record["time"]
    soma_v = record["soma_v"]
    seg_voltages = record["seg_voltages"]
    soma_spike = features["soma_spike"]

    plot_section_ids = plot_section_ids or []
    plot_segment_ids = plot_segment_ids or []

    n_cols = 1 + len(plot_section_ids) + len(plot_segment_ids)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    # ---- 上排：胞体电压 ----
    ax = axes[0]
    ax.plot(time_arr, soma_v, color="black", linewidth=1.0)
    ax.axhline(SPIKE_THRESHOLD, color="red", linestyle="--", linewidth=1,
               label=f"spike threshold ({SPIKE_THRESHOLD} mV)")
    ax.axhline(REST_VOLTAGE, color="blue", linestyle=":", linewidth=1,
               label=f"rest ({REST_VOLTAGE} mV)")
    ax.set_title(f"Soma V (spike={'Yes' if soma_spike else 'No'})")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.legend(fontsize=7)

    # ---- 下排：各选定 section 的 segment 电压 ----
    for k, sid in enumerate(plot_section_ids):
        ax = axes[k + 1]
        seg_ids = exp.segments_of(sid)
        # 该 section 的峰值（用于标题）
        row = exp.section_catalog[
            exp.section_catalog["section_id"] == sid
        ].iloc[0]
        for seg_id in seg_ids:
            if seg_id in seg_voltages:
                v = seg_voltages[seg_id]
                ax.plot(time_arr, v, linewidth=0.8,
                        label=seg_id.split("_")[-1])  # 用 segment 内序号作图例
        if show_spike_threshold:
            ax.axhline(SPIKE_THRESHOLD, color="red", linestyle="--",
                       linewidth=0.8)
        ax.set_title(f"{sid}  (L={row['length_um']:.0f}um, "
                     f"dist={row['proximal_dist_um']:.0f}um)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (mV)")
        ax.legend(fontsize=6, ncol=1)

    # ---- 直接指定的 segment 电压 ----
    for k, seg_id in enumerate(plot_segment_ids):
        ax = axes[1 + len(plot_section_ids) + k]
        if seg_id in seg_voltages:
            ax.plot(time_arr, seg_voltages[seg_id],
                    color="purple", linewidth=1.2, label=seg_id)
        if show_spike_threshold:
            ax.axhline(SPIKE_THRESHOLD, color="red", linestyle="--",
                       linewidth=0.8)
        ax.set_title(seg_id)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (mV)")
        ax.legend(fontsize=6, ncol=1)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"[INFO] 图已保存: {save_path}")
        return save_path
    return fig


# ============================================================
# 4.6) 形态标注可视化
# ============================================================
def get_section_3d(sec):
    """提取一个 Section 的 3D 折线点集和对应的 arc length（单位 um）。"""
    sec.push()
    try:
        n = int(h.n3d())
        pts = []
        arcs = []
        for i in range(n):
            pts.append((h.x3d(i), h.y3d(i), h.z3d(i)))
            arcs.append(h.arc3d(i))
    finally:
        h.pop_section()
    return pts, arcs


def segment_center_3d(sec, seg_x):
    """
    返回某 segment（中心位于 section 的 seg_x 处）在 3D 空间中的坐标。
    通过沿 section 3D 折线的 arc length 线性插值得到。
    """
    pts, arcs = get_section_3d(sec)
    if not pts:
        return (0.0, 0.0, 0.0)
    target = seg_x * sec.L
    # 处理目标落在范围外的情况
    if target <= arcs[0]:
        return pts[0]
    if target >= arcs[-1]:
        return pts[-1]
    for k in range(len(arcs) - 1):
        if arcs[k] <= target <= arcs[k + 1]:
            span = arcs[k + 1] - arcs[k]
            if span <= 0:
                return pts[k]
            t = (target - arcs[k]) / span
            x = pts[k][0] + t * (pts[k + 1][0] - pts[k][0])
            y = pts[k][1] + t * (pts[k + 1][1] - pts[k][1])
            z = pts[k][2] + t * (pts[k + 1][2] - pts[k][2])
            return (x, y, z)
    return pts[-1]


def plot_morphology_markup(exp, section_ids=None, segment_ids=None,
                           save_path=None, projection="xy"):
    """
    在神经元形态（X-Y 俯视投影）上标注选中的 section 和 segment。

    参数：
      section_ids : 要高亮（红色粗线）的 section 列表
      segment_ids : 要标注（蓝色圆点）的 segment 列表
      projection  : 投影平面，"xy"（默认俯视）或 "xz"（侧视）
    """
    # 区域基础配色
    base_colors = {
        "soma":   "#2ecc71",
        "axon":   "#e67e22",
        "basal":  "#c0392b",
        "apical": "#2980b9",
    }
    # 投影轴选择
    axis_map = {"x": 0, "y": 1, "z": 2}
    a0, a1 = axis_map[projection[0]], axis_map[projection[1]]

    fig, ax = plt.subplots(figsize=(8, 11))

    # ---- 1. 画所有 section（细线、半透明）----
    for region in REGION_ORDER:
        for sec in _sections_in_region(exp.cell, region):
            pts, _ = get_section_3d(sec)
            if len(pts) < 2:
                continue
            xs = [p[a0] for p in pts]
            ys = [p[a1] for p in pts]
            ax.plot(xs, ys, color=base_colors[region], linewidth=0.6, alpha=0.55)

    # ---- 2. 高亮选中的 section ----
    if section_ids:
        for sid in section_ids:
            row = exp.section_catalog[
                exp.section_catalog["section_id"] == sid
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            sec = _sections_in_region(exp.cell, row["region"])[row["region_index"]]
            pts, _ = get_section_3d(sec)
            xs = [p[a0] for p in pts]
            ys = [p[a1] for p in pts]
            ax.plot(xs, ys, color="red", linewidth=3.0, zorder=3)
            # 标注 section id（取折线中点）
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            ax.annotate(sid, (cx, cy), fontsize=6, color="red",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="red", alpha=0.85), zorder=6)

    # ---- 3. 标注选中的 segment（蓝色圆点）----
    if segment_ids:
        for sgid in segment_ids:
            if sgid not in exp.segment_obj:
                continue
            sec, seg = exp.segment_obj[sgid]
            cx, cy, cz = segment_center_3d(sec, seg.x)
            px = (cx, cy, cz)[a0]
            py = (cx, cy, cz)[a1]
            ax.scatter(px, py, s=50, color="blue", edgecolors="black",
                       linewidth=0.8, zorder=5)
            ax.annotate(sgid.split("_")[-1], (px, py), fontsize=6,
                        color="blue", ha="center", va="bottom", zorder=6)

    ax.set_aspect("equal")
    ax.set_xlabel(f"{projection[0].upper()} (um)")
    ax.set_ylabel(f"{projection[1].upper()} (um)")
    ax.set_title(f"Morphology markup ({projection.upper()} projection)")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] 形态标注图已保存: {save_path}")
        return save_path
    return fig


# ============================================================
# 5) 数据保存
# ============================================================
def save_outputs(output_dir, trial_idx, features, section_catalog,
                 segment_catalog, active_sections):
    """保存一次仿真的三类数据到磁盘。"""
    os.makedirs(output_dir, exist_ok=True)

    prefix = os.path.join(output_dir, f"trial_{trial_idx}")

    # (a) section 特征表（含阈下特征 + 二值激活）
    features["section_features"].to_csv(f"{prefix}_section_features.csv", index=False)

    # (b) 胞体标签 + 轨迹
    np.savez(
        f"{prefix}_soma.npz",
        time=features["time"],
        soma_v=features["soma_v"],
        soma_sub_v=features["soma_sub_v"],
        soma_spike=features["soma_spike"],
        active_sections=np.array(active_sections, dtype=object),
    )

    return prefix


def save_catalogs(output_dir, section_catalog, segment_catalog):
    """保存编号目录（section / segment），供后续研究复用。"""
    os.makedirs(output_dir, exist_ok=True)
    section_catalog.to_csv(os.path.join(output_dir, "section_catalog.csv"),
                           index=False)
    segment_catalog.to_csv(os.path.join(output_dir, "segment_catalog.csv"),
                           index=False)


# ============================================================
# 6) 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="L5PC P0 实验脚本")
    parser.add_argument("--trials", type=int, default=1,
                        help="仿真次数（随机选取激活 section）")
    parser.add_argument("--duration", type=int, default=1000,
                        help="每次仿真时长 (ms)")
    parser.add_argument("--ex-rate", type=float, default=500.0,
                        help="每个 segment 兴奋性输入率 (Hz)")
    parser.add_argument("--inh-rate", type=float, default=400.0,
                        help="每个 segment 抑制性输入率 (Hz)")
    parser.add_argument("--outdir", default="L5PC_exp_outputs",
                        help="输出目录")
    parser.add_argument("--plot", type=lambda s: s.lower() in ("1", "true", "yes", "on"),
                        default=True, help="是否绘制并保存图像 (默认开)")
    parser.add_argument("--plot-sections", nargs="+", default=None,
                        help="指定要绘制的 section_id 列表，如 basal_0 apical_50；"
                             "不填则自动选激活 section 中峰值最高的 4 个")
    parser.add_argument("--mark-sections", nargs="+", default=None,
                        help="要在形态图上高亮标注的 section_id 列表")
    parser.add_argument("--mark-segments", nargs="+", default=None,
                        help="要在形态图上标注的 segment_id 列表")
    parser.add_argument("--activate-sections", nargs="+", default=None,
                        help="指定要激活的 section_id 列表（覆盖随机选取），"
                             "如 apical_20 basal_3")
    parser.add_argument("--activate-segments", nargs="+", default=None,
                        help="指定要直接激活的 segment_id 列表，"
                             "如 apical_20_0 apical_20_2")
    parser.add_argument("--plot-segments", nargs="+", default=None,
                        help="指定要在电压轨迹图中绘制的 segment_id 列表")
    args = parser.parse_args()

    # 初始化实验（加载模型、编号、放突触）
    exp = L5PCExperiment()

    # 保存编号目录
    save_catalogs(args.outdir, exp.section_catalog, exp.segment_catalog)
    print(f"[INFO] 编号目录已保存到 {args.outdir}/")

    # 打印编号示例（帮助理解编号规则）
    print("\n[*] 编号示例:")
    for demo_sid in ["soma_0", "axon_0", "basal_0", "apical_0", "apical_50"]:
        if demo_sid in exp.section_catalog["section_id"].values:
            exp.show_section(demo_sid)
            print()

    # 运行仿真并提取特征
    rng = np.random.default_rng(0)
    all_basal = exp.section_ids("basal")
    all_apical = exp.section_ids("apical")

    for trial in range(args.trials):
        # 过滤非法编号
        if args.activate_sections is not None:
            active_sections = [s for s in args.activate_sections
                               if s in exp.section_catalog["section_id"].values]
        else:
            active_sections = None

        if args.activate_segments is not None:
            active_segments = [s for s in args.activate_segments
                               if s in exp.segment_catalog["segment_id"].values]
        else:
            active_segments = None

        # 两者都没指定时，随机选取 section 激活
        if active_sections is None and active_segments is None:
            n_basal = rng.integers(0, len(all_basal) + 1)
            n_apical = rng.integers(0, len(all_apical) + 1)
            active_sections = list(rng.choice(all_basal, size=n_basal, replace=False)) + \
                              list(rng.choice(all_apical, size=n_apical, replace=False))

        n_sec = len(active_sections) if active_sections else 0
        n_seg = len(active_segments) if active_segments else 0
        print(f"\n{'='*60}")
        print(f" Trial {trial}: 激活 {n_sec} 个 section + {n_seg} 个 segment")
        print(f"{'='*60}")

        record = exp.run(section_ids=active_sections,
                         segment_ids=active_segments,
                         duration_ms=args.duration,
                         ex_rate_hz=args.ex_rate, inh_rate_hz=args.inh_rate,
                         seed=trial)
        features = exp.extract_features(record)

        # 打印摘要
        n_act = int(features["section_features"]["activated"].sum())
        print(f"[结果] 胞体放电: {features['soma_spike']}, "
              f"超阈值 section 数: {n_act}")

        # 绘图
        if args.plot:
            # 确定要绘制的 section
            if args.plot_sections is not None:
                plot_sids = [s for s in args.plot_sections
                             if s in exp.section_catalog["section_id"].values]
            else:
                # 自动选：激活 section 中峰值最高的 4 个
                act_df = features["section_features"][
                    features["section_features"]["region"].isin(["basal", "apical"])
                ].sort_values("peak_v", ascending=False)
                plot_sids = act_df["section_id"].head(4).tolist()

            # 确定要绘制的 segment
            if args.plot_segments is not None:
                plot_segs = [s for s in args.plot_segments
                             if s in exp.segment_catalog["segment_id"].values]
            else:
                plot_segs = []

            if plot_sids or plot_segs:
                plot_path = os.path.join(args.outdir, f"trial_{trial}.png")
                plot_trial(exp, record, features, plot_sids,
                           plot_segment_ids=plot_segs,
                           save_path=plot_path)

        # 形态标注图
        if args.mark_sections or args.mark_segments:
            markup_path = os.path.join(args.outdir, f"trial_{trial}_markup.png")
            plot_morphology_markup(exp,
                                   section_ids=args.mark_sections,
                                   segment_ids=args.mark_segments,
                                   save_path=markup_path)

        save_outputs(args.outdir, trial, features,
                     exp.section_catalog, exp.segment_catalog,
                     record["active_sections"])

    print(f"\n[OK] 所有数据已保存到 {args.outdir}/")


if __name__ == "__main__":
    main()

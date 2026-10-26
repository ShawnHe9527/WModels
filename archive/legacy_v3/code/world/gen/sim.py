"""单条轨迹的 NEURON 仿真与记录。

## 关键设计

1. **整细胞 + 单段观测**：跑完整 196 段，但只驱动 `apical_51` 的 11 对突触，
   只记录该段的 9 因子 × 11 compartment。
2. **复用细胞**：细胞、22 个突触、NetCon、记录向量**只建一次**，
   每条轨迹只换事件表。`finitialize` 会把状态与记录向量重置。
3. **dt 必须显式重设**：`h.run()` 会把 dt 重置为 0.025（见 DESIGN_v3 §7.2）。
   本模块用 `finitialize` → 设 dt → `continuerun`。

## 输出（每轨迹）

| 数组 | 形状 | 含义 |
|---|---|---|
| `s` | `(T, 9, nseg)` | 状态（9 因子 × nseg）|
| `u` | `(T, 3, nseg)` | 突触电导 [AMPA, NMDA(去门控), GABA] |
| `ub` | `(T, 3)` | 边界电位 [父段, 子段1, 子段2] |

T = t_test / dt（默认 1000 / 0.025 = 40000）。
"""
from __future__ import annotations

import os

import numpy as np

MODDIR = "/home/neuron/L5PC/WModel2/neuron/mods/x86_64/libnrnmech.so"
HOCODIR = "/home/neuron/L5PC/WModel2/neuron/hoc"
MORPH = "/home/neuron/L5PC/WModel2/neuron/morphologies/cell1.asc"

# 9 个状态因子（顺序即 s 的第 1 维）
FACTOR_NAMES = [
    "Ih.m", "Ca_LVAst.m", "Ca_LVAst.h", "Ca_HVA.h",
    "SKv3_1.m", "SK_E2.z", "Im.m", "cai", "V",
]
# (机制, 门控字母) → 属性名 f"{gate}_{mech}"
GATES = [
    ("Ih", "m"), ("Ca_LVAst", "m"), ("Ca_LVAst", "h"), ("Ca_HVA", "h"),
    ("SKv3_1", "m"), ("SK_E2", "z"), ("Im", "m"),
]
N_FACTORS = len(FACTOR_NAMES)          # 9

SECTION = "apical_51"
# 边界：父段 apic[50](1.0)、两个子段 apic[52](0.0)/apic[53](0.0)
BOUNDARY = (("apic", 50, 1.0), ("apic", 52, 0.0), ("apic", 53, 0.0))

EVENTS: list = []          # [(NetCon, np.ndarray of times), ...]
KEEP: list = []            # ⚠️ 必须长期持有，否则 section 被删 ⇒ segfault
_DLL = False
_FIH = False


def add_all_events():
    for nc, ts in EVENTS:
        for t_ in ts:
            nc.event(float(t_))


# ==============================================================================
class Sim:
    """可复用的仿真器：建一次，跑多次。"""

    def __init__(self, dt: float = 0.025, t_settle: float = 100.0,
                 t_test: float = 1000.0,
                 gmax_e: float = 0.0004, gmax_i: float = 0.0010):
        global _DLL
        from neuron import h
        self.h = h
        if not _DLL:
            h.nrn_load_dll(MODDIR)
            _DLL = True
        cwd = os.getcwd()
        os.chdir(HOCODIR)
        for f in ("stdrun.hoc", "import3d.hoc", "L5PCbiophys5b.hoc",
                  "L5PCtemplate_2.hoc"):
            h.load_file(f)
        os.chdir(cwd)
        h.celsius = 34.0

        self.dt = dt
        self.t_settle = t_settle
        self.t_test = t_test
        self.T = int(round(t_test / dt))

        self.cell = h.L5PCtemplate(MORPH)
        KEEP.append(self.cell)                      # ⚠️ 防 GC
        self.sec = self.cell.apic[51]
        self.nseg = int(self.sec.nseg)

        # ---- 11 对突触（官方参数）----
        self.syn_e, self.syn_i = [], []
        self.nc_e, self.nc_i = [], []
        for i in range(self.nseg):
            seg = self.sec((i + 0.5) / self.nseg)
            KEEP.append(seg)
            e = h.ProbAMPANMDA2(seg)
            e.tau_r_AMPA, e.tau_d_AMPA = 0.3, 3.0
            e.tau_r_NMDA, e.tau_d_NMDA = 2.0, 70.0
            e.gmax, e.e = gmax_e, 0
            ne = h.NetCon(None, e)
            ne.delay, ne.weight[0] = 0.0, 1
            ii = h.ProbUDFsyn2(seg)
            ii.tau_r, ii.tau_d = 0.2, 8.0
            ii.gmax, ii.e = gmax_i, -80
            ni = h.NetCon(None, ii)
            ni.delay, ni.weight[0] = 0.0, 1
            KEEP.extend([e, ne, ii, ni])
            self.syn_e.append(e)
            self.syn_i.append(ii)
            self.nc_e.append(ne)
            self.nc_i.append(ni)

        # ---- 记录向量（复用）----
        self.rec_s = []           # 9 × nseg
        self.segs = [self.sec((i + 0.5) / self.nseg) for i in range(self.nseg)]
        KEEP.extend(self.segs)
        for (m, g) in GATES:
            attr = f"{g}_{m}"
            self.rec_s.append([h.Vector().record(getattr(sg, "_ref_" + attr))
                               for sg in self.segs])
        self.rec_s.append([h.Vector().record(sg._ref_cai) for sg in self.segs])
        self.rec_s.append([h.Vector().record(sg._ref_v) for sg in self.segs])

        self.rec_ga = [h.Vector().record(s._ref_g_AMPA) for s in self.syn_e]
        self.rec_gn = [h.Vector().record(s._ref_g_NMDA) for s in self.syn_e]
        # ⚠️ mggate 是 ASSIGNED 变量（不是 RANGE）⇒ 没有 _ref_mggate，
        #    不可记录。但它只依赖 V：mggate = 1/(1+exp(-0.08*v)*mg/3.57)，
        #    mg=1 mM ⇒ 用记录的 V 反算即可（见 run()）。
        self.rec_gi = [h.Vector().record(s._ref_g) for s in self.syn_i]

        self.rec_ub = []
        for (nm, idx, x) in BOUNDARY:
            secb = getattr(self.cell, nm)[idx]
            sg = secb(x)
            KEEP.append(sg)
            self.rec_ub.append(h.Vector().record(sg._ref_v))
        self.rec_t = h.Vector().record(h._ref_t)

        self._arm()

    # ------------------------------------------------------------------
    def _arm(self):
        global _FIH
        if _FIH:
            return
        h = self.h
        h('objref fih_gen')
        h.fih_gen = h.FInitializeHandler('nrnpython("add_all_events()")')
        KEEP.append(h.fih_gen)
        import __main__
        __main__.add_all_events = add_all_events
        _FIH = True

    # ------------------------------------------------------------------
    def run(self, ex_times, in_times, v_init: float = -76.0):
        """跑一条轨迹，返回 (s, u, ub)。

        ex_times / in_times : list[np.ndarray]，长度 = nseg
        """
        EVENTS.clear()
        for i in range(self.nseg):
            EVENTS.append((self.nc_e[i], ex_times[i]))
            EVENTS.append((self.nc_i[i], in_times[i]))

        h = self.h
        h.v_init = v_init
        h.tstop = self.t_settle + self.t_test
        h.finitialize(v_init)      # 触发 FInitializeHandler ⇒ 投递事件
        h.dt = self.dt             # ★ 必须在 finitialize 之后（run() 会重置）
        h.continuerun(h.tstop)

        # ---- 切出 test 相 ----
        t = np.asarray(self.rec_t)
        i0 = int(round(self.t_settle / self.dt))
        sl = slice(i0, i0 + self.T)

        S = np.empty((self.T, N_FACTORS, self.nseg), dtype=np.float32)
        for f in range(N_FACTORS - 2):
            block = np.asarray(self.rec_s[f])[:, sl]           # (nseg, T)
            S[:, f, :] = block.T
        cai = np.asarray(self.rec_s[N_FACTORS - 2])[:, sl]
        S[:, 7, :] = np.log10(np.maximum(cai, 1e-12)).T
        S[:, 8, :] = np.asarray(self.rec_s[N_FACTORS - 1])[:, sl].T

        U = np.empty((self.T, 3, self.nseg), dtype=np.float32)
        ga = np.asarray([np.asarray(v)[sl] for v in self.rec_ga])   # (nseg,T)
        gn = np.asarray([np.asarray(v)[sl] for v in self.rec_gn])
        gi = np.asarray([np.asarray(v)[sl] for v in self.rec_gi])
        U[:, 0, :] = ga.T
        # ★ 去门控 NMDA：mggate 只依赖 V，用记录的 V 反算
        Vseg = S[:, 8, :].T                          # (nseg, T)
        mggate = 1.0 / (1.0 + np.exp(-0.08 * Vseg) * (1.0 / 3.57))
        U[:, 1, :] = (gn / np.maximum(mggate, 1e-9)).T
        U[:, 2, :] = gi.T

        UB = np.stack([np.asarray(v)[sl] for v in self.rec_ub], axis=1)
        return S, U, UB.astype(np.float32)

    # ------------------------------------------------------------------
    def theta(self) -> np.ndarray:
        """段参数 θ_s（27 维）。不依赖 catalog，直接从 section 读。"""
        from . import secinfo as SI
        th, _ = SI.build_theta(self.sec, "apical")
        return np.asarray(th, dtype=np.float32)

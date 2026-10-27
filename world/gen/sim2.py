"""v4 单轨迹仿真：目标段自身突触 + tuft 簇状突触 + 胞体真实 bAP。

## 与 v3 `sim.py` 的区别

| | v3 `Sim` | **v4 `Sim2`** |
|---|---|---|
| 目标段突触 | 11 E + 11 I | **不变** |
| 邻段 | 无驱动，只记边界 V | **`apic[50]/[52]/[53]` 各 comp 1 E + 1 I**（共 17+17）|
| 胞体 | 无 | **`IClamp` 脉冲池，诱发真实 AP** |
| $u$ | `(T,3,11)` + `ub (T,3)` 边界电位 | **`(T,43)` 见下** |
| 门禁 | V_max / cai_span | 5 门（平台占比 / 双峰性 / 事件数 / bAP 数 / 数值）|

## $u$ 的布局（**全分辨率 T2，85 维**）

每 3 维为 1 个 compartment 的 `[AMPA, NMDA(去门控), GABA]`（**comp-major**）：

| 索引 | 内容 |
|---|---|
| `0:33`   | 目标段（11 comp）|
| `33:84`  | 邻段（17 comp，按 `apic[50]`→`apic[52]`→`apic[53]` 顺序）|
| `84`     | **bAP 指示**（胞体脉冲 2 ms 内 = 1）|

邻段在 `33:84` 内部的偏移：

| 邻段 | compartment 范围 | 索引范围 |
|---|---|---|
| `apic[50]` | 0–2 | 33–41 |
| `apic[52]` | 3–7 | 42–56 |
| `apic[53]` | 8–16 | 57–83 |

### 为什么存全分辨率而不是 43 维

**T1（43 维，按 section 求和）是 T2 的确定性函数**：
对每个 section 的 compartment 求和即得。所以只需存 T2，
运行时用 `to_t1(U)` 派生即可 —— **零额外存储、零额外仿真，
两种观测是完全配对的**（同一轨迹的同一份数据）。

这比生成两份数据集更干净：两次生成会引入与观测无关的
d1 差异，让 T1 vs T2 的对比无法严格配对。

——

## 为什么编码器参数量不随这个维数变

`Encoder` 是 DeepSets：**每个 compartment 一个 token**，
每个 token 的特征是 `[3 通道, x]` = **4 维**，与 token 个数无关。

| $u$ 总维 | token 数 | 每 token 维 | Encoder 参数量（H=192, d_z=16）|
|---|---|---|---|
| 43 | 3（T1，三个伪 comp）| 4 | **344,304** |
| 85 | 17 | 4 | **344,304** |

⇒ **涨维只增加池化的序列长度，不增加任何参数**（计算量随 token 数线性增长，可忽略）。

## 为什么 NMDA 必须去门控

`mggate = 1/(1+exp(-0.08 V)/3.57)`：V=−75 时 0.0088，V=+40 时 0.989。
原始 $g_{NMDA}$ 在静息被压小 114 倍、在平台放大 100 倍 —— **即使没有新事件**。
直接喂原始值 = 把 $s$（V）泄露进 $u$，模型可绕过因果学习。
"""
from __future__ import annotations

import os

import numpy as np

# Project-relative paths: the checkout can live anywhere after migration.
# `neuron/mods/x86_64/libnrnmech.so` must be rebuilt on the target machine
# (`nrnivmodl`) because it is platform-specific.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
MODDIR = os.path.join(_ROOT, "neuron", "mods", "x86_64", "libnrnmech.so")
HOCODIR = os.path.join(_ROOT, "neuron", "hoc")
MORPH = os.path.join(_ROOT, "neuron", "morphologies", "cell1.asc")

FACTOR_NAMES = [
    "Ih.m", "Ca_LVAst.m", "Ca_LVAst.h", "Ca_HVA.h",
    "SKv3_1.m", "SK_E2.z", "Im.m", "cai", "V",
]
GATES = [
    ("Ih", "m"), ("Ca_LVAst", "m"), ("Ca_LVAst", "h"), ("Ca_HVA", "h"),
    ("SKv3_1", "m"), ("SK_E2", "z"), ("Im", "m"),
]
N_FACTORS = len(FACTOR_NAMES)          # 9

SECTION = "apical_51"
TARGET_IDX = 51
PARENT_IDX = 50                        # 上游（近端）
CHILD_IDX = (52, 53)                   # 下游（远端 tuft）
NB_IDX = (PARENT_IDX,) + CHILD_IDX     # 3 个邻段

U_DIM = 3 * 11 + 3 * (3 + 5 + 9) + 1     # 33 + 51 + 1 = 85
U_NB_OFFSET = 33
U_BAP_IDX = U_DIM - 1
# 邻段在 u[33:84] 里的 compartment 起止（与 NB_IDX 顺序一致）
NB_SLICES = ((0, 3), (3, 8), (8, 17))    # apic[50] / apic[52] / apic[53]

G_E, G_I = 0.0004, 0.0010
MAXP = 32                              # 胞体脉冲池大小

EVENTS: list = []
KEEP: list = []
_READY = False
_FIH = False


def add_all_events_v2():
    for nc, ts in EVENTS:
        for t_ in ts:
            nc.event(float(t_))


# ==============================================================================
class Sim2:
    """v4 仿真器：建一次，跑多次。"""

    def __init__(self, dt: float = 0.025, t_settle: float = 100.0,
                 t_test: float = 1000.0,
                 gmax_e: float = G_E, gmax_i: float = G_I,
                 nb_n_e: int = 1, nb_n_i: int = 1):
        """`nb_n_e` / `nb_n_i`：**邻段**每 compartment 的突触数（D4）。

        目标段始终是 1 E + 1 I per compartment（与 v3 一致，保证可比）。
        """
        global _READY, _FIH
        from neuron import h
        self.h = h
        if not _READY:
            h.nrn_load_dll(MODDIR)
            cwd = os.getcwd()
            os.chdir(HOCODIR)
            for f in ("stdrun.hoc", "import3d.hoc", "L5PCbiophys5b.hoc",
                      "L5PCtemplate_2.hoc"):
                h.load_file(f)
            os.chdir(cwd)
            _READY = True
        h.celsius = 34.0

        self.dt = dt
        self.t_settle = t_settle
        self.t_test = t_test
        self.T = int(round(t_test / dt))
        self.gmax_e, self.gmax_i = gmax_e, gmax_i

        self.cell = h.L5PCtemplate(MORPH)
        KEEP.append(self.cell)                     # ⚠️ 防 GC ⇒ 否则 segfault

        # ---- 目标段 ----
        self.sec = self.cell.apic[TARGET_IDX]
        self.nseg = int(self.sec.nseg)
        self.segs = [self.sec((i + 0.5) / self.nseg) for i in range(self.nseg)]
        KEEP.extend(self.segs)

        self.syn_e, self.syn_i = [], []
        self.nc_e, self.nc_i = [], []
        for sg in self.segs:
            self._add_pair(self.syn_e, self.nc_e, self.syn_i, self.nc_i,
                           sg, 1, 1)          # 目标段固定 1 E + 1 I

        # ---- 邻段（O1：官方 1 E + 1 I per compartment）----
        self.nb = []            # [{idx, n, segs, syn_e, nc_e, syn_i, nc_i}]
        for idx in NB_IDX:
            sec = self.cell.apic[idx]
            n = int(sec.nseg)
            segs = [sec((i + 0.5) / n) for i in range(n)]
            KEEP.extend(segs)
            d = dict(idx=idx, n=n, segs=segs, syn_e=[], nc_e=[],
                     syn_i=[], nc_i=[], e_seg=[], i_seg=[])
            for ci, sg in enumerate(segs):
                b_e, b_i = len(d["syn_e"]), len(d["syn_i"])
                self._add_pair(d["syn_e"], d["nc_e"], d["syn_i"], d["nc_i"],
                               sg, nb_n_e, nb_n_i)
                # 记录每个突触属于哪个 compartment（去门控要按 seg 取 V）
                d["e_seg"].extend([ci] * (len(d["syn_e"]) - b_e))
                d["i_seg"].extend([ci] * (len(d["syn_i"]) - b_i))
            self.nb.append(d)
        self.n_nb_e = sum(len(d["syn_e"]) for d in self.nb)     # 17
        self.n_nb_i = sum(len(d["syn_i"]) for d in self.nb)     # 17

        # ---- 胞体脉冲池（⚠️ Vector.play 对 IClamp.amp 不生效）----
        # ⚠️ cell.soma 是 SectionList，必须 [0]
        self.soma_seg = self.cell.soma[0](0.5)
        KEEP.append(self.soma_seg)
        self.pulses = [h.IClamp(self.soma_seg) for _ in range(MAXP)]
        KEEP.extend(self.pulses)
        for icp in self.pulses:
            icp.delay, icp.dur, icp.amp = 1e9, 0.1, 0.0

        # ---- 记录 ----
        self.rec_s = []
        for (m, g) in GATES:
            attr = "_ref_" + f"{g}_{m}"
            self.rec_s.append([h.Vector().record(getattr(sg, attr))
                               for sg in self.segs])
        self.rec_s.append([h.Vector().record(sg._ref_cai) for sg in self.segs])
        self.rec_s.append([h.Vector().record(sg._ref_v) for sg in self.segs])

        self.rec_ga = [h.Vector().record(s._ref_g_AMPA) for s in self.syn_e]
        self.rec_gn = [h.Vector().record(s._ref_g_NMDA) for s in self.syn_e]
        self.rec_gi = [h.Vector().record(s._ref_g) for s in self.syn_i]

        for d in self.nb:
            d["rec_ga"] = [h.Vector().record(s._ref_g_AMPA) for s in d["syn_e"]]
            d["rec_gn"] = [h.Vector().record(s._ref_g_NMDA) for s in d["syn_e"]]
            d["rec_gi"] = [h.Vector().record(s._ref_g) for s in d["syn_i"]]
            d["rec_v"] = [h.Vector().record(sg._ref_v) for sg in d["segs"]]
            d["rec_cai"] = [h.Vector().record(sg._ref_cai) for sg in d["segs"]]

        self.rec_v_soma = h.Vector().record(self.soma_seg._ref_v)
        self.rec_t = h.Vector().record(h._ref_t)

        self._arm()

    # ------------------------------------------------------------------
    def _add_pair(self, syn_e, nc_e, syn_i, nc_i, sg, n_e, n_i):
        h = self.h
        for _ in range(n_e):
            e = h.ProbAMPANMDA2(sg)
            e.tau_r_AMPA, e.tau_d_AMPA = 0.3, 3.0
            e.tau_r_NMDA, e.tau_d_NMDA = 2.0, 70.0
            e.gmax, e.e = self.gmax_e, 0
            nc = h.NetCon(None, e)
            nc.delay, nc.weight[0] = 0.0, 1
            KEEP.extend([e, nc])
            syn_e.append(e)
            nc_e.append(nc)
        for _ in range(n_i):
            ii = h.ProbUDFsyn2(sg)
            ii.tau_r, ii.tau_d = 0.2, 8.0
            ii.gmax, ii.e = self.gmax_i, -80
            nc = h.NetCon(None, ii)
            nc.delay, nc.weight[0] = 0.0, 1
            KEEP.extend([ii, nc])
            syn_i.append(ii)
            nc_i.append(nc)

    # ------------------------------------------------------------------
    def _arm(self):
        global _FIH
        if _FIH:
            return
        h = self.h
        h('objref fih_v2')
        h.fih_v2 = h.FInitializeHandler('nrnpython("add_all_events_v2()")')
        KEEP.append(h.fih_v2)
        import __main__
        __main__.add_all_events_v2 = add_all_events_v2
        _FIH = True

    # ------------------------------------------------------------------
    def set_bap(self, times, amp=4.0, width=2.0):
        """设置胞体脉冲。未被使用的 IClamp 推到 1e9（永不触发）。"""
        times = np.asarray(times, dtype=np.float64).ravel()[:MAXP]
        self.bap_times = times
        self.bap_amp, self.bap_width = amp, width
        for j, icp in enumerate(self.pulses):
            if j < len(times):
                icp.delay, icp.dur, icp.amp = float(times[j]), width, amp
            else:
                icp.delay, icp.dur, icp.amp = 1e9, 0.1, 0.0

    # ------------------------------------------------------------------
    def run(self, own_e, own_i, tuft_e, tuft_i, bap_times=None,
            bap_amp=4.0, bap_width=2.0, v_init: float = -76.0):
        EVENTS.clear()
        for i in range(self.nseg):
            EVENTS.append((self.nc_e[i], own_e[i]))
            EVENTS.append((self.nc_i[i], own_i[i]))

        k = 0
        for d in self.nb:
            for nc in d["nc_e"]:
                EVENTS.append((nc, tuft_e[k]))
                k += 1
        k = 0
        for d in self.nb:
            for nc in d["nc_i"]:
                EVENTS.append((nc, tuft_i[k]))
                k += 1

        if bap_times is not None:
            self.set_bap(bap_times, amp=bap_amp, width=bap_width)
        else:
            self.set_bap(np.empty(0))

        h = self.h
        h.v_init = v_init
        h.tstop = self.t_settle + self.t_test
        h.finitialize(v_init)      # ★ 触发 FInitializeHandler ⇒ 投递事件
        h.dt = self.dt             # ★ 必须在 finitialize 之后
        h.continuerun(h.tstop)

        t = np.asarray(self.rec_t)
        i0 = int(round(self.t_settle / self.dt))
        sl = slice(i0, i0 + self.T)
        T = self.T

        # ---- s (T,9,11) ----
        S = np.empty((T, N_FACTORS, self.nseg), dtype=np.float32)
        for f in range(N_FACTORS - 2):
            S[:, f, :] = np.asarray(self.rec_s[f])[:, sl].T
        cai = np.asarray(self.rec_s[N_FACTORS - 2])[:, sl]
        S[:, 7, :] = np.log10(np.maximum(cai, 1e-12)).T
        S[:, 8, :] = np.asarray(self.rec_s[N_FACTORS - 1])[:, sl].T

        # ---- u (T,85) ，comp-major ----
        U = np.empty((T, U_DIM), dtype=np.float32)
        ga = np.asarray([np.asarray(v)[sl] for v in self.rec_ga])   # (11,T)
        gn = np.asarray([np.asarray(v)[sl] for v in self.rec_gn])
        gi = np.asarray([np.asarray(v)[sl] for v in self.rec_gi])
        Vt = np.asarray(self.rec_s[N_FACTORS - 1])[:, sl]           # (11,T)
        own = np.empty((T, self.nseg, 3), dtype=np.float32)
        own[:, :, 0] = ga.T
        own[:, :, 1] = (gn / np.maximum(_mggate(Vt), 1e-9)).T
        own[:, :, 2] = gi.T
        U[:, 0:33] = own.reshape(T, 33)

        nb_stats = []
        for j, d in enumerate(self.nb):
            Vn = np.asarray([np.asarray(v)[sl] for v in d["rec_v"]])   # (n,T)
            gan = np.asarray([np.asarray(v)[sl] for v in d["rec_ga"]])  # (nE,T)
            gnn = np.asarray([np.asarray(v)[sl] for v in d["rec_gn"]])
            gin = np.asarray([np.asarray(v)[sl] for v in d["rec_gi"]])  # (nI,T)
            cain = np.asarray([np.asarray(v)[sl] for v in d["rec_cai"]])
            mg_e = _mggate(Vn)[np.asarray(d["e_seg"], dtype=int)]
            # 按 compartment 汇总（密度>1 时同一 comp 有多个突触）
            n = d["n"]
            e_seg = np.asarray(d["e_seg"], dtype=int)
            i_seg = np.asarray(d["i_seg"], dtype=int)
            gA_c = np.zeros((n, T)); gN_c = np.zeros((n, T))
            gG_c = np.zeros((n, T))
            for ci in range(n):
                if (e_seg == ci).any():
                    gA_c[ci] = gan[e_seg == ci].sum(axis=0)
                    gN_c[ci] = (gnn / np.maximum(mg_e, 1e-9))[e_seg == ci].sum(axis=0)
                if (i_seg == ci).any():
                    gG_c[ci] = gin[i_seg == ci].sum(axis=0)

            o = U_NB_OFFSET + 3 * NB_SLICES[j][0]
            blk = np.empty((T, n, 3), dtype=np.float32)
            blk[:, :, 0] = gA_c.T
            blk[:, :, 1] = gN_c.T
            blk[:, :, 2] = gG_c.T
            U[:, o:o + 3 * n] = blk.reshape(T, 3 * n)
            nb_stats.append(dict(idx=d["idx"], V=Vn, cai=cain, g_sum=gA_c))

        # ---- bAP 指示 ----
        U[:, U_BAP_IDX] = _bap_trace(t[i0:i0 + T], self.bap_times, self.bap_width)

        self.last_nb = nb_stats
        self.last_v_soma = np.asarray(self.rec_v_soma)[sl]
        return S, U

    # ------------------------------------------------------------------
    def theta(self) -> np.ndarray:
        from . import secinfo as SI
        th, _ = SI.build_theta(self.sec, "apical")
        return np.asarray(th, dtype=np.float32)


# ==============================================================================
def _mggate(V):
    """与 `ProbAMPANMDA2.mod` 一致的 Mg 门控（`_ref_mggate` 不存在，只能反算）。"""
    return 1.0 / (1.0 + np.exp(-0.08 * V) * (1.0 / 3.57))


def _bap_trace(t, times, width):
    out = np.zeros(t.shape, dtype=np.float32)
    if times is None or len(times) == 0:
        return out
    for t0 in times:
        out[(t >= t0) & (t < t0 + width)] = 1.0
    return out


def to_t1(U):
    """把 85 维全分辨率 $u$ 压成 43 维（DESIGN_v4 §4.3 T1）。

    **T1 是 T2 的确定性函数**（按 section 对 compartment 求和）：

    | T1 索引 | 内容 |
    |---|---|
    | `0:33`  | 目标段 11 comp × 3 通道（不变）|
    | `33:36` | `apic[50]` 对 3 comp 求和 |
    | `36:39` | `apic[52]` 对 5 comp 求和 |
    | `39:42` | `apic[53]` 对 9 comp 求和 |
    | `42`    | bAP 指示 |

    ⇒ 同一份 T2 数据可无损派生 T1，两种观测**完全配对**，
    让 T1 vs T2 的对比不含任何与观测无关的差异。
    """
    U = np.asarray(U)
    T = U.shape[0]
    out = np.empty((T, 43), dtype=U.dtype)
    out[:, 0:33] = U[:, 0:33]
    for k, (lo, hi) in enumerate(NB_SLICES):
        blk = U[:, U_NB_OFFSET + 3 * lo: U_NB_OFFSET + 3 * hi]
        blk = blk.reshape(T, hi - lo, 3)
        out[:, 33 + 3 * k: 36 + 3 * k] = blk.sum(axis=1)
    out[:, 42] = U[:, U_BAP_IDX]
    return out

# 工作站交接文档 — WModel2 / DSH Agent

> 交接时间：2026-10-28  
> 工作站：`a6000middle`，`192.168.1.94`  
> 账户：`Hesj`，home `/home/Hesj`  
> 项目：`/home/Hesj/WModel`  
> 远端仓库：`https://github.com/ShawnHe9527/WModels`  
> 当前代码基线：`61ea84e`（本文档提交后会更新；以 `git log -1` 为准）

## 0. 新 Agent 先读什么

按顺序读：

1. `AGENTS.md` — 项目事实、禁区、实验规范；
2. `PROJECT_STATE.md` — 当前结论与待确认问题；
3. `docs/PLAN_R6_C_A_AND_MIGRATION.md` — C+A/C-only 实验计划；
4. 本文件 — 工作站交接状态。

**不要跳过 `AGENTS.md` 的“禁区”。** 本文件只补充工作站当前状态，不替代项目规则。

---

## 1. 一句话状态

R2 仍是当前最优；R5 零训练标定、R5-H1、R5-B 均失败。  
下一步是 **2 s 数据 + C+A（`length=512`，9000 步）**；C-only 只在 C+A 有改善后做。  
工作站已装好 DSH、项目已 clone、SMB `dataHe` 已挂载、`data`/`checkpoints` 软链接已建立；**Python/NEURON 环境、T1/R2 权重、2 s 数据尚未就绪。**

---

## 2. 已完成 / 未完成

### 已完成

- DSH 用户级安装：`/home/Hesj/.local/bin/dsh`，`0.2.1-rc.2`
- Node 22：`/home/Hesj/envs/agent22/bin/node`（v22.23.4）
- `DSH_HOME=/home/Hesj/.dsh`
- 项目 clone：`/home/Hesj/WModel`，`main` 分支
- SMB 子目录挂载：
  - `//192.168.1.65/QH1_Labdata/dataHe` → `/home/Hesj/QH1_dataHe`
  - 只暴露 `dataHe`，不挂整个共享
- 项目软链接：
  - `/home/Hesj/WModel/data` → `/home/Hesj/QH1_dataHe/WM`
  - `/home/Hesj/WModel/results/checkpoints` → `/home/Hesj/QH1_dataHe/WM/checkpoints`
- `data` 和 `checkpoints` 写测试已通过
- `.git/info/exclude` 已忽略这两个软链接

### 未完成

- `/home/Hesj/envs/worm_sim` 尚未创建
- `neuron/mods` 尚未 `nrnivmodl`
- `tools/v0_check` 尚未通过
- 本地 T1/R2 checkpoint 尚未拷到工作站
- 2 s 数据尚未生成
- `scaler_v4_2s.json` 尚未生成
- C+A 尚未训练
- `te_s4_2s` 尚未评测

---

## 3. 身份与路径

```text
工作站 IP       192.168.1.94
用户            Hesj
home            /home/Hesj
项目            /home/Hesj/WModel
DSH             /home/Hesj/.local/bin/dsh
DSH_HOME        /home/Hesj/.dsh
Node            /home/Hesj/envs/agent22/bin/node
conda           /home/Hesj/miniconda3/bin/conda
SMB 服务器      192.168.1.65
SMB 共享        QH1_Labdata
SMB 子目录      dataHe/WM
本地挂载点      /home/Hesj/QH1_dataHe
```

**不要混淆：**

- `192.168.1.94` 是工作站；
- `192.168.1.65` 是 NAS；
- `192.168.1.64` 是另一台 NAS（QH2），不是本项目默认路径；
- `Y:`/`Z:` 是 Windows 盘符，Linux 不能用。

---

## 4. 绝对不要做

- 不要用 GPU；GPU 被其他人占满，训练只跑 CPU；
- 不要 `sudo apt install` / 升级共享 conda / 改 `/etc/fstab`；
- 不要 `rm -rf` 删除 `data`、`results/checkpoints`、`runs`；
- 不要 `cat /home/Hesj/.smbcred`，不要提交密码/token；
- 不要动 `/home/Hemp/...`、`/home/zzb/...` 或其他用户的挂载；
- 不要动 `.137` 的 `HempData`（dmesg 里反复报错的是别人的挂载）；
- 不要在 `/mnt/c` 下运行任何东西；
- 不要同时跑数据生成和训练。

---

## 5. 第一步：验证交接状态

在**工作站 Shell**执行：

```bash
whoami
hostname -I
cd /home/Hesj/WModel
git switch main
git status -sb
git log --oneline -1

ls -ld data results/checkpoints
mount | grep QH1_dataHe
df -hT /home/Hesj/QH1_dataHe
```

预期：

- `whoami` = `Hesj`
- `hostname -I` 包含 `192.168.1.94`
- `git status` 干净或只有已知文档改动
- `data` 指向 `/home/Hesj/QH1_dataHe/WM`
- `results/checkpoints` 指向 `/home/Hesj/QH1_dataHe/WM/checkpoints`
- `mount` 显示 `//192.168.1.65/QH1_Labdata/dataHe on /home/Hesj/QH1_dataHe type cifs`
- `df` 显示 NAS 容量

写测试：

```bash
cd /home/Hesj/WModel
touch data/.write_test && rm data/.write_test && echo "data writable"
touch results/checkpoints/.write_test && rm results/checkpoints/.write_test && echo "checkpoints writable"
```

如果不是 writable，先修存储，不要训练。

---

## 6. 建立 conda 虚拟环境

不要动 base；用独立前缀。

```bash
export CONDA_PKGS_DIRS=/home/Hesj/.conda/pkgs
export CONDA_ENVS_DIRS=/home/Hesj/envs
mkdir -p /home/Hesj/.conda/pkgs /home/Hesj/envs

/home/Hesj/miniconda3/bin/conda create \
  -p /home/Hesj/envs/worm_sim \
  python=3.10 -y

export PATH=/home/Hesj/envs/worm_sim/bin:$PATH
hash -r
which python
python -V
```

预期 Python 3.10.x。

安装 CPU 依赖：

```bash
python -m pip install --upgrade pip

python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.4.1+cpu

python -m pip install \
  numpy==2.0.2 NEURON==8.2.0 scipy==1.15.3 matplotlib==3.10.8
```

验证：

```bash
python -c "import torch, numpy; print(torch.__version__, numpy.__version__, torch.cuda.is_available())"
python -c "import neuron; print(neuron.__version__)"
```

`torch.cuda.is_available()` 应为 `False`；不要尝试用 GPU。

---

## 7. 编译 NEURON 并自检

```bash
cd /home/Hesj/WModel/neuron/mods
which nrnivmodl
nrnivmodl
```

如果 `nrnivmodl` 不在 PATH：

```bash
/home/Hesj/envs/worm_sim/bin/nrnivmodl
```

检查：

```bash
ls -l /home/Hesj/WModel/neuron/mods/x86_64/libnrnmech.so
```

端到端：

```bash
cd /home/Hesj/WModel
export PYTHONPATH=$PWD

/home/Hesj/envs/worm_sim/bin/python -m tools.v0_check \
  --n 1 --seed 99 --mem-mb 2500
```

必须 **1/1 通过**。不通过就不要生成数据。

---

## 8. 存储与重挂载

当前挂载是临时的，重启会消失；不要写 `/etc/fstab`。

先检查：

```bash
mount | grep QH1_dataHe
ls -ld /home/Hesj/QH1_dataHe
```

如果重启后没有挂载，由 Hesj（有 sudo）执行：

```bash
sudo mkdir -p /home/Hesj/QH1_dataHe

sudo mount -t cifs \
  //192.168.1.65/QH1_Labdata/dataHe \
  /home/Hesj/QH1_dataHe \
  -o credentials=/home/Hesj/.smbcred,\
uid=1007,gid=1007,\
file_mode=0660,dir_mode=0770,vers=3.1.1
```

**不要挂整个 `QH1_Labdata` 到可被项目随意访问的目录；只挂 `dataHe` 子目录。**

挂好后重新测：

```bash
cd /home/Hesj/WModel
touch data/.write_test && rm data/.write_test && echo ok
```

---

## 9. 从本机 WSL 拷贝 T1/R2 权重

这些文件不在 Git 里。当前本机 WSL 的源路径（按实际为准）：

```text
/home/neuron/L5PC/WModel2/results/checkpoints/t1v4_dz16
/home/neuron/L5PC/WModel2/results/checkpoints/t2v5_events
/home/neuron/L5PC/WModel2/results/data/scaler_v4.json
```

在**本机 WSL**执行：

```bash
scp -r /home/neuron/L5PC/WModel2/results/checkpoints/t1v4_dz16 \
  Hesj@192.168.1.94:/home/Hesj/QH1_dataHe/WM/checkpoints/

scp -r /home/neuron/L5PC/WModel2/results/checkpoints/t2v5_events \
  Hesj@192.168.1.94:/home/Hesj/QH1_dataHe/WM/checkpoints/

scp /home/neuron/L5PC/WModel2/results/data/scaler_v4.json \
  Hesj@192.168.1.94:/home/Hesj/QH1_dataHe/WM/
```

它们只有 MB 级，可以走 SSH；**不要传 17 GB 的 `data/`**。

工作站检查：

```bash
ls -l /home/Hesj/QH1_dataHe/WM/checkpoints/t1v4_dz16/best.pt
ls -l /home/Hesj/QH1_dataHe/WM/checkpoints/t2v5_events/last.pt
```

如果这些文件缺失，C+A 无法用 T1 初始化；先解决，不要直接训练。

---

## 10. 生成 2 s 数据

数据写到 `data/` 软链接，即 NAS 的 `dataHe/WM`。

```bash
cd /home/Hesj/WModel
export PYTHONPATH=$PWD
PY=/home/Hesj/envs/worm_sim/bin/python

$PY -u -m world.gen.datagen2 --name tr_s1_2s \
  --n-trial 200 --seed 1 --t-settle 100 --t-test 2000 --every-secs 20

$PY -u -m world.gen.datagen2 --name va_s2_2s \
  --n-trial 200 --seed 2 --t-settle 100 --t-test 2000 --every-secs 20

$PY -u -m world.gen.datagen2 --name te_s4_2s \
  --n-trial 200 --seed 4 --t-settle 100 --t-test 2000 --every-secs 20
```

预计约 35 GB、1.5–2 h。生成后：

```bash
du -sh data/tr_s1_2s data/va_s2_2s data/te_s4_2s
df -hT /home/Hesj/QH1_dataHe
```

`te_s4_2s` 是盲测；生成后不要用它调参。

---

## 11. 生成新 scaler

```bash
cd /home/Hesj/WModel
$PY -m tools.make_scaler \
  --data tr_s1_2s \
  --out results/data/scaler_v4_2s.json \
  --seed 0
```

检查：

```bash
ls -l results/data/scaler_v4_2s.json
```

---

## 12. 跑 C+A

```bash
cd /home/Hesj/WModel
export PYTHONPATH=$PWD
PY=/home/Hesj/envs/worm_sim/bin/python

$PY -u -m world.learn.train2 t2 \
  --data tr_s1_2s --val va_s2_2s \
  --run r6_ca_2s_l512 --tier t2 \
  --init t1v4_dz16/best.pt \
  --scaler results/data/scaler_v4_2s.json \
  --steps 9000 --batch 24 --length 512 \
  --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
  --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 128 \
  --cl-weight 1.0 --cl-weight-start 0.1 \
  --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.1 \
  --event-weight 10 --event-tau 0.2 \
  --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
  --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
  --sigma-gate 0.008 --mem-mb 5000
```

**不要加** `--event-center-frac` / `--event-v-weight`（R5-B 已失败）。
**不要**把 `--cl-stage-a-steps` 压到 500（H1 的教训）。

预计 4–5 h，峰值 RSS 4.5–5.5 GB；一次只跑一个训练。

---

## 13. 评测

先确认 C+A 产物：

```bash
ls -l results/checkpoints/r6_ca_2s_l512/last.pt
```

在 `te_s4_2s` 上评 R2（zero-shot）和 C+A：

```bash
cd /home/Hesj/WModel
export PYTHONPATH=$PWD
PY=/home/Hesj/envs/worm_sim/bin/python

$PY -m tools.eval_gate \
  --data te_s4_2s --t0 4000,16000 --ks 10,200,2000,8000 \
  --n-trials 128 --batch 16 \
  --model R2=t2v5_events/last.pt \
  --model CA=r6_ca_2s_l512/last.pt \
  --out runs/<date>-<commit>-r6-ca/eval_te_s4_2s.json
```

- 同一测试集、同一 t0、同一 batch；
- 报 overall + linear/inter/plateau；
- 报 Vmax、Ca F1；
- 最终盲测只用 `te_s4_2s` 一次，不要用它调参。

**判据**：

- `te_s4_2s` k=2000/8000 不差于 R2；
- 无 `DIVERGED`；
- Vmax 比值 `pred/true ∈ [0.8,1.2]`；
- Ca F1 不低于 R2。

---

## 14. C-only 的门槛

只有 C+A 有改善才跑 C-only：

```bash
$PY -u -m world.learn.train2 t2 \
  --data tr_s1_2s --val va_s2_2s \
  --run r6_c_only_2s --tier t2 \
  --init t1v4_dz16/best.pt \
  --scaler results/data/scaler_v4_2s.json \
  --steps 9000 --batch 24 --length 128 \
  --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
  --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 64 \
  --cl-weight 1.0 --cl-weight-start 0.1 \
  --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.1 \
  --event-weight 10 --event-tau 0.2 \
  --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
  --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
  --sigma-gate 0.008 --mem-mb 5000
```

C-only 的用途：区分“数据长度”与“视界”贡献。

---

## 15. DSH 交接步骤

新 Agent 在**工作站**启动/使用 DSH：

```bash
source /home/Hesj/env.sh
tmux new -s agent2
cd /home/Hesj/WModel
dsh --profile web --host 127.0.0.1 --port 18082 --no-open
```

- 端口用 18082，避免和已有 18081 冲突；
- 本机 SSH 隧道：

```bash
ssh -N -L 18082:127.0.0.1:18082 Hesj@192.168.1.94
```

- 浏览器打开新 token URL；
- 工作目录选 `/home/Hesj/WModel`；
- 先只读 `AGENTS.md`、`PROJECT_STATE.md`、`docs/PLAN_R6_C_A_AND_MIGRATION.md`、本文件。

如果已有 DSH 窗口仍在运行，不要抢占 18081；用新端口 18082。

---

## 16. 新 Agent 的启动提示词

可以把下面这段直接发给新 DSH：

```text
你接手 WModel2 工作站实验。工作目录 /home/Hesj/WModel，账户 Hesj，工作站 192.168.1.94。
先读：AGENTS.md、PROJECT_STATE.md、docs/PLAN_R6_C_A_AND_MIGRATION.md、docs/HANDOFF_WORKSTATION_AGENT.md。
当前状态：R2 最优；H1/B 失败；下一步只做 2 s 数据 + C+A（L=512，9000 步）。
不要用 GPU；不要改 /etc/fstab；不要动他人挂载；不要删 data/results。
第一步只做环境验证：git status、data 软链接、mount、worm_sim 环境、nrnivmodl、v0_check。
在 v0_check 1/1 通过、T1/R2 ckpt 就位之前，不要生成数据或训练。
每次实验前先给方案（假设、命令、判据、耗时、失败预案），等我确认后再执行。
```

---

## 17. 验收清单

新 Agent 完成环境后，逐项报告：

- [ ] `git status` 干净
- [ ] `data` / `results/checkpoints` 软链接存在且可写
- [ ] `mount | grep QH1_dataHe` 正常
- [ ] `worm_sim` Python 3.10
- [ ] `torch 2.4.1+cpu`，`cuda.is_available() == False`
- [ ] `nrnivmodl` 生成 `libnrnmech.so`
- [ ] `v0_check --n 1` 1/1 通过
- [ ] T1、R2 checkpoint 已就位
- [ ] 2 s 数据三组已生成
- [ ] `scaler_v4_2s.json` 已生成
- [ ] C+A 训练完成
- [ ] `te_s4_2s` 评测完成

---

## 18. 停止条件

- `v0_check` 不通过：停止，不生成数据；
- `data` 不可写 / mount 丢失：停止，先修存储；
- T1/R2 checkpoint 缺失：停止，不训练；
- 2 s 数据出现 NaN/大小异常：停止；
- 训练 RSS 超过 6 GB 或系统可用内存 <600 MB：停止；
- `te_s4_2s` 只用一次，不能根据它继续调参。

---

## 19. 安全与凭据

- `/home/Hesj/.smbcred` 权限 600，含 NAS 密码；不要 `cat`、不要提交；
- DSH Web token 只在启动时打印，不要写进仓库；
- 不要截图/聊天里暴露密码或 token；
- 挂载是临时的；重启后重新执行 mount，不要写 fstab；
- 只操作 `QH1_dataHe/WM`，不要动 `dataHe` 下其他目录。

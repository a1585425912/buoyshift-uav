# UAV DPC 模块化实验：算法流程与 Cluster 机制详细总结

适用分支：`cluster-node-prior-exploration`（默认 `cluster_mode=node_prior`、`uav_kf_prior_mode=dpc_only`）。
本文档是对 `dfpc_experiment` 包的完整技术总结：物理场景、双时间尺度主循环、六种对比方法、
Kalman 滤波实现、cluster 内部机制（重点为 node_prior 模式），以及每一处"为降低误差/提升
后续算法性能"而做的设计。参数值均为 `config.py` 中的默认值，行号以当前分支代码为准。

---

## 1. 问题背景

N 个海上浮标节点组成分布式阵列，相干发射指向一个匀速直线飞行的 UAV（接收方）。
第 i 个节点的理想相位命令是 θᵢ = k·dᵢ（k 为波数，dᵢ 为该节点到 UAV 的真实距离），
使所有节点信号在 UAV 处同相叠加，理想合成功率 P_ideal = (Σ ampᵢ)²。

困难在于每个节点只能获得带噪声的位置信息：

- 节点对 UAV 位置的观测误差（σ=3 m）：UAV 轨迹必须先被估计出来；
- 节点对自身位置的观测误差（σ=1 m）：浮标在海浪中漂移；
- 持续的系统/校准相位误差 εᵢ（σ=3°，整个 trial 内固定）。

距离估计误差 Δd 直接映射为相位误差 k·Δd（20 MHz 时 λ=15 m，1 m 距离误差 ≈ 24° 相位误差），
相位误差使相干叠加退化为部分相干。实验的目标是量化：共识（DPC）、Kalman 滤波、
簇内协作分别能把归一化功率 P/P_ideal 从随机相位水平提升多少。

---

## 2. 物理场景与观测模型（scenario.py / sim_core.py / backend.py）

### 2.1 部署与运动

- N 个浮标初始均匀分布在半径 500 m 的圆盘上（`deploy_buoys_in_disk`），高度 0。
- UAV 起点 (−0.65·R, −0.35·R, 120 m)，速度 20 m/s，航向 20°，严格匀速直线
  （`uav_center_true` 每短步 += v·Ts）。
- 浮标真实位置 = 慢变中心 + 小尺度有界偏移，三个成分每短步（dt=Ts=0.05 s）更新一次：
  1. **海浪漂移**（`gaussian_drift_displacement`）：位移 ~ N(v̄·dt, σ_d·√dt·I)，
     均值速度 5 m/s、方向 0°、扩散强度 0.5 m/√s——非零均值漂移是本项目的默认场景；
  2. **小尺度扰动**（`gaussian_short_offset_step`）：零均值布朗增量（σ=0.08 m/√s），
     经指数记忆（相关时间 1.2 s）衰减，形成 OU 型有界抖动；
  3. **偏移吸收**：每步按 (1−e^(−dt/1.2s)) 把一部分偏移并入慢变中心，
     随后偏移矢量被裁剪回半径 1 m 内（`clip_offsets`），防止随机游走发散。

### 2.2 观测模型（`observe_positions`）

目前是"真实位置 + 各向同性高斯误差"的简化模型：

- 每节点对 UAV 位置的观测：p_u_true + N(0, 3²)（每个节点独立采样）；
- 节点对自身位置的观测：自身真实位置 + N(0, 1²)。

### 2.3 通信图与共识矩阵（`metropolis_hastings_W`）

- 随机无向图：每对节点以 p=0.03 相连，再叠加一个环形骨干保证无孤立节点；
- Metropolis–Hastings 双随机权重：W_ij = 1/(max(deg_i, deg_j)+1)，对角补行和。
  双随机性保证纯平均共识 X←W@X 收敛到全网均值（均值一致性），这是所有共识步骤
  "不引入系统性偏移"的基础。
- 共识乘法在 CuPy/CUDA 上以 float32 加速（`consensus_linear_accel`），无 GPU 时回退 CPU float64。

### 2.4 信道与功率（`geometry_terms`）

- 单节点幅度 ampᵢ = √P_tx / dᵢ^α（P_tx=5，α=1）；
- φ_true = k·dᵢ 为真实传播相位；
- P_ideal = (Σ ampᵢ)²，P_single_mean = mean(amp²)，P_single_best = max(amp²)。

### 2.5 系统相位误差

ε ~ N(0, 3°²)，**每个 trial 只采样一次并全程保持**（`eps_trial`）。刻意不做逐步独立采样：
校准类误差物理上是缓变的，逐步独立采样会在几何平滑变化时产生非物理的功率锯齿。

---

## 3. 双时间尺度实验框架（experiment.py 主循环）

- 一次 trial 共 T_long 个长块（默认 20，sweep 常用 8），每块 K=30 个 Ts=0.05 s 短步。
  **iteration 既是算法迭代也是一次物理短步**：每步真值推进 Ts、重新观测、重跑算法。
- 长块边界处额外推进一个短步，保证记录样本严格等间隔 Ts，块边界不重复同一物理时刻。
- 双时间尺度 Kalman：
  - `uav_kf_short`：dt=Ts，每短步 predict+update，块内跟踪；块首从长尺度 KF **只同步速度均值**
    （不重置协方差——重置会把速度钉死在初始不确定性附近）；
  - `uav_kf_long`：dt=Ts，但每块只在块末更新一次（先 predict_n_steps(K)），用短尺度 KF
    收敛后的位置作观测，块间逐步收敛速度；更新后再做一步全局共识。
  - 初始速度用"真实 UAV 速度 + N(0, 2²)"，避免从零学起的瞬态（这是有意保留的
    温和初始化，对齐旧版双时间尺度实现）。
- Monte Carlo：mc_trials 个 trial（默认 50，sweep 常用 20），seed = seed₀ + trial·1009。
  场景、观测、簇内测量用三个独立 RNG 流（seed, seed+3000003, seed+4000003）。

---

## 4. 每个短步的算法主链

以 sidx 为全局短步序号。原始观测 (uav_obs_short, buoy_obs_short) 在每步开头重新生成。

### 第 1 环节：本地轨迹估计（trajectory.py）

每个节点独立维护最小二乘充分统计量 (Σt, Σt², Σxyz, Σt·xyz)，每步加入自己最新的
UAV 位置观测后，闭式解出 UAV 匀速直线轨迹，输出 9 维状态
`[bx,vx,by,vy,bz,vz, dx,dy,dz]`（截距、笛卡尔速度、单位航向）。
用笛卡尔速度而非角度表示，避免 ±π 缠绕；方向分量在共识后重新单位化
（`normalize_trajectory_directions`，速度作为回退）。

### 第 2 环节：全局 DPC 共识 → DPC 方法

对 9 维轨迹状态做**一步**全网共识 W@X（当前真实迭代模型每 iteration 一次），
方向归一化后按当前时刻 t 外插出 UAV 位置估计 `uav_est_dfpc`。
这相当于每个节点都拿到"全网平均轨迹参数"，单节点观测噪声被 1/√N 压缩。
**DPC 方法的 UAV 侧到此为止，节点侧直接用原始自身观测**（两路公平对比）。

### 第 3 环节：UAV 短时 KF 融合（UAV+Node-KF DPC 方法）

predict(dt=Ts) 后，按 `uav_kf_prior_mode` 注入 DPC 信息：

- `dpc_only`（本分支默认）：把 `uav_est_dfpc` 作为一份**位置伪观测**
  （R = 3² m²）正常走 KF 更新，再融合一份新的原始观测 z_u，最后一步共识；
- `none`（旧路径）：`inject_position_state` 直接覆写 KF 位置并清空位置-速度协方差，
  之后的原始观测更新会把好估计重新拉向单节点噪声——这就是默认值改为 dpc_only 的原因；
- `dpc_plus_cluster`：额外再注入一份簇内 UAV 轨迹先验（配合 uav_prior 模式，当前默认不用）。

### 第 4 环节：节点（浮标）KF

每节点 predict + 用新的自身原始观测 z_b 更新（R = 1² m²），跟踪海浪漂移。
浮标是不同物理目标，**绝不对不同浮标的绝对位置做共识平均**（会抹掉各自真实的漂移差异）。
该 KF 的位置 `node_est_kf` 供 UAV+Node-KF DPC 方法使用。

### 第 5 环节：node_prior 簇内校正（Cluster 系列方法，详见第 6 节）

并行维护第二套节点 KF `node_kf_prior`（与第 4 环节同参数、同原始观测），
额外吸收簇内共识轨迹信息。其位置 `cluster_node_est` 供
Cluster DPC / Cluster UAV+Node-KF DPC 两个方法用作节点侧估计。

### 第 6 环节：相位补偿、功率与误差评估（metrics.py）

对每种方法，用其 (UAV 估计, 节点估计) 计算：

- d̂ᵢ = ‖p_u_est − node_estᵢ‖，θᵢ = k·d̂ᵢ（硬件上模 2π 表示，不属于算法）；
- 残余相位 = wrap_π(θ − φ_true + ε) 再**模 π 折叠到 [−90°, 90°)**
  （`effective_phase_errors`）：发射端允许 0/π 极性反转，折叠是物理等价操作而非截断；
- 功率 P = |Σ ampᵢ·e^{jθresᵢ}|²；归一化功率 P/P_ideal（dB 后即 norm_db）；
- phase_std：折叠后残余相位的**普通样本标准差**（沿用旧实验 σ_φ 口径，非圆统计），
  phase_rmse：相对 0 的 RMS；随机相位参考的理论底线 ≈ 180°/√12 ≈ 51.96°；
- 距离/节点/UAV RMSE 用于分解误差来源。

---

## 5. Kalman 滤波实现细节（kalman.py）

- `BatchCVKalman3D`：N 个目标并行的 6 维 [x,y,z,vx,vy,vz] 恒速滤波器，
  einsum 批量算 P 传播，**Joseph 形式协方差更新** + 显式对称化——长时间序列
  和高频 sweep 中比标准形式更稳定，避免 P 失去半正定；
- z 轴过程噪声缩放 0.05（高度实际固定，但状态保留 z/vz 以便将来接入高度变化）；
- `velocity_propagate=False` 可把 F 的位置-速度耦合块置零（保留接口）；
- `update_position_measurement`：以自定义 R 注入伪观测，用后恢复原 R——
  这是"DPC 先验/簇内共识进 KF"的统一入口；
- `inject_position_state`（旧路径）：覆写位置并清空交叉协方差，
  保留供 `uav_kf_prior_mode=none` 对照。

---

## 6. Cluster 机制详解（cluster.py）

### 6.1 分簇（`communication_graph_clusters`）

在通信图上随机选 n_clusters=40 个种子，多源 BFS 把每个节点划入最近可达的簇；
不连通的残余节点归入邻居最多的簇。簇 = 一组内部两两可达（经通信边）的节点，
簇内交互不产生额外全局流量。

### 6.2 四种 cluster_mode

| 模式 | 簇内做什么 | 输出去向 | 状态 |
|---|---|---|---|
| `trajectory` | 簇内节点互相投票 UAV 轨迹参数，α=0.8 直接混合 | 混合后的轨迹直接用于 Cluster DPC | 旧机制，保留对照 |
| `localization` | 运动先验 + 导频相对约束迭代定位节点 | 定位结果**直接作为**节点位置 | 被node_prior取代 |
| `uav_prior` | 簇内 UAV 轨迹子图共识（`cluster_trajectory_consensus`） | 作为 UAV KF 的额外先验（dpc_plus_cluster） | 保留，默认不用 |
| **`node_prior`** | 簇内观测者对每个节点轨迹做共识估计 | 共识轨迹作为**弱伪观测回灌**节点 KF | **本分支默认** |

### 6.3 node_prior 模式逐步流程（`ClusterTrajectoryLocalizer.update`）

每个短步、每个簇内执行：

1. **绝对基准 = 节点自身运动轨迹**：以 `node_kf_prior` 的 6 维状态为绝对位置先验
   （运动模型提供绝对参考，这是与"纯相对定位"的根本区别）；
2. **导频相对测量**：簇内每对节点 (j,i) 测得 pᵢ − pⱼ + N(0, 0.2²)（vector 约束，
   也有只用距离的 range 变体）；由相邻两步相对位置差分得到**相对速度观测**；
   （仿真中相对测量基于真值加噪，建模簇内导频互测，σ=0.2 m 远小于自身观测 1 m）；
3. **多观测者投票**：对簇内每个目标 i，其余每个观测者 j 生成估计
   "j 的轨迹状态 + (pᵢ − pⱼ 测量值)"，共 M−1 份 6 维轨迹估计（M 为簇大小）；
4. **簇内 DPC 共识**：对每个目标，观测者估计向量在簇子图 W 上迭代
   cluster_localization_iterations=5 步平均共识，使所有观测者对同一目标的估计收敛一致；
5. **防自印证**：目标 i 自己那一行不用它自己的估计，而是用其余观测者的均值填充种子，
   保证共识结果不被目标自身先验锁死；
6. **回灌而非替换**（本模式名称的由来）：共识轨迹**不直接覆盖**节点状态
   （`blend_with_own=False`，返回纯共识值），而是以一份 σ=1 m 的位置伪观测
   （`cluster_node_prior_noise_std`）经 `update_position_measurement` 融入 `node_kf_prior`。
   KF 按当前协方差自动加权：相对测量噪声大时权重自动低；
   共识信息跨步累积在 KF 状态里，而不是每步被覆盖。

### 6.4 误差缩减机理（为什么簇内操作能提高后续算法性能）

1. **独立噪声的平均**：M−1 个观测者的独立投票经共识平均，随机误差 ~ σ/√(M−1)；
   相对测量噪声 0.2 m 比自身观测 1 m 小 5 倍，簇内几何信息质量天然更高；
2. **几何刚化**：vector/range 相对约束把簇内节点的相对几何"锁住"，
   直接压低 node_rmse → 距离估计 d̂ᵢ 更准 → 残余相位 k·Δd 更小；
3. **软融合的抗污染性**：弱伪观测回灌让 KF 保留自己的运动模型记忆，
   相对测量的噪声尖峰不会被直接写进位置；对比 localization 模式（硬替换），
   node_prior 对簇大小不均、测量离群更鲁棒；
4. **速度维度的额外信息**：相对位置差分出的相对速度观测也进共识，
   对漂移中的浮标（5 m/s 涌浪）跟踪有帮助。

---

## 7. "降低误差 / 提升后续算法性能"的操作汇总

| # | 操作 | 位置 | 机理 |
|---|---|---|---|
| 1 | 9 维笛卡尔轨迹参数化 + 共识后方向单位化 | trajectory.py | 避免角度缠绕，共识平均的量无歧义 |
| 2 | MH 双随机 W + 环形骨干 | sim_core.py | 平均共识无偏、无孤立节点 |
| 3 | DPC 估计以伪观测进 KF（dpc_only） | experiment.py 第3环节 | 保留 KF 协方差信息，正确加权融合而非覆写 |
| 4 | 长尺度 KF 块末更新、短尺度只同步速度均值 | experiment.py | 速度块间收敛，位置跟踪不被重置打断 |
| 5 | Joseph 形式 + 对称化协方差更新 | kalman.py | 长时间数值稳定，P 保持半正定 |
| 6 | 节点 KF 不做跨节点绝对共识 | experiment.py 第4环节 | 浮标是不同物理目标，绝对位置不可平均 |
| 7 | 簇内观测者共识 + 防自印证 | cluster.py | 1/√(M−1) 噪声压缩，无自证偏差 |
| 8 | 共识轨迹弱伪观测回灌 node KF | experiment.py 第5环节 | 软融合、跨步累积、抗测量离群 |
| 9 | ε trial 内持续 | experiment.py | 物理校准误差建模，避免非功率锯齿 |
| 10 | 残余相位模 π 折叠 | sim_core.py | 0/π 极性选择的物理等价 |
| 11 | 功率先线性平均再转 dB | experiment.py / sweeps.py | dB 域直接平均会高估功率 |
| 12 | 尾段（后 20%）统计 + trial/block 级诊断 | sweeps.py | 排除收敛瞬态；回升事件可定位到 seed |

---

## 8. 方法对照与指标体系（constants.py / metrics.py）

| 方法 | UAV 侧估计 | 节点侧估计 | 用途 |
|---|---|---|---|
| Random Phase Reference | θ=0 | — | 理论下限（相位 std ≈ 51.96°） |
| No Algorithm | 当前原始观测 | 当前原始观测 | 开环单步基线，无共识无滤波 |
| DPC | 共识轨迹外插 | 原始观测 | 仅 UAV 侧协作 |
| Cluster DPC | 共识轨迹外插 | node_prior KF 位置 | 仅节点侧簇内协作 |
| UAV+Node-KF DPC | 短时 KF（DPC 伪观测） | 节点 KF | 双侧滤波，无簇 |
| Cluster UAV+Node-KF DPC | 短时 KF（DPC 伪观测） | node_prior KF 位置 | 全链路（本分支主方法） |

核心指标：norm_db = 10log₁₀(P/P_ideal)（越接近 0 dB 越好）；
gain_over_single_mean/best_db（相对单节点的阵列增益，节点数实验主指标）；
phase_std/rmse（°）；distance/node/uav RMSE（m）；tail = 后 20% 短步的稳态统计。
理论参考：完全随机相位时 P/P_ideal ≈ 1/N（-10log₁₀N dB），全相干为 0 dB。

---

## 9. 实验入口、脚本与输出

主入口（共用 `add_common_args`，参数在 config.py 集中定义）：

- `dfpc_experiment/run_experiment.py`：单组实验（一次配置 × MC）；
- `dfpc_experiment/run_sweeps.py`：**因子扫描实验**，因素含 node_count / connectivity /
  frequency / node_position_noise / uav_position_error / wave_speed，
  每个因素输出 summary、trial/block 级诊断 CSV 与摘要图；frequency 额外做
  相邻点功率回升检测（rebound_threshold_db=0.5）并定位主导 trial seed；
- `dfpc_experiment/run_note33_unified_sweeps.py`：以统一口径复现 Note3.3 的四组 sweep
  （仅 Random/No Algorithm/DPC 三方法）；
- `dfpc_experiment/run_kf_array_experiment.py`：仅保留两条 KF 支路的多天线（1/2/4/8 阵元）
  实验，总功率固定，含浮标姿态误差与阵元差分校准误差；
- `dfpc_experiment/recompute_no_algorithm_iid.py`：用解析 i.i.d. 随机相位基线
  （σ_φ = 180°/√3）重新统计既有结果；
- `relative_spatial_consensus_experiment.py`（仓库根）：纯空域分解实验，把路径误差拆成
  "UAV-Node 相对几何误差"与"节点自身误差"，验证平均共识对不一致性的抑制；
- `work/`：高频/先验机制的探索性 sweep 脚本（high_freq_prior_sweep、
  high_freq_cluster_points、noise_compare_prior、parallel_freq_sweep）。

输出目录：默认 `modular_dfpc_outputs/`，可用 `--out_dir` 指定（`outputs/` 已被 gitignore，
入库结果需 `git add -f`）。单组实验输出曲线 CSV/NPZ/PNG/报告，sweep 输出
`modular_dfpc_factor_sweeps_summary.csv` 及各因素 summary / diagnostics / 图。

---

## 10. 本分支的探索假设

`cluster-node-prior-exploration` 相对 main 的默认值变化仅两处（commit 91a389c）：

- `cluster_mode`: localization → **node_prior**
- `uav_kf_prior_mode`: none → **dpc_only**

探索假设：**簇内协作信息只用于增强节点自身轨迹（经 KF 回灌），UAV 估计链只吃 DPC
伪观测**——即"节点先验"路线，而不是旧的"簇内 UAV 轨迹投票/先验"路线。
因子实验（node_count / frequency 等 sweep）在上述默认下运行，用以验证该配置下：
1) 归一化功率是否随节点数饱和而非异常线性增长；
2) 对照组（No Algorithm / Random Reference）相位误差是否保持在理论底线（≈51.96°）
且与 N 无关；3) Cluster 系列方法对 DPC/KF 系列的增益来源。

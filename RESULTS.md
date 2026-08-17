# 实验结果总结（cluster-node-prior-exploration 分支）

本文件汇总在探索默认值（`cluster_mode=node_prior`、`uav_kf_prior_mode=dpc_only`）下完成的
三批实验结果与结论。算法机制详见 `ALGORITHM_SUMMARY.md`。

数据位置（`outputs/` 默认被 gitignore，以下结果目录已 `git add -f` 入库）：

- `outputs/sweep_node_freq_20mc_nodeprior/`：因子扫描（node_count 12 点 + frequency 7 点，20 MC trial，
  N=1000 基准、T_long=8、K=30、20 MHz），含 summary / trial / block 三级诊断 CSV、逐因素图、
  回升检测报告，以及两份附加分析图（`cluster_sensitivity_node_freq.png`、`kf_cluster_design_dynamics.png`）。
- `outputs/branch_compare_main_localization/` 与 `outputs/branch_compare_nodeprior/`：
  main 与探索分支的配对对比实验（同 seed 各 20 trial，N=1000、20 MHz、T_long=8、K=30）。

---

## 1. 因子扫描核心结果（node_prior 模式）

### 1.1 归一化功率 vs 节点数：无异常线性增长

归一化功率（÷理想全相干功率）从 N=20 到 N=2000 完全饱和：DPC -0.76→-0.78 dB，
KF-DPC -0.09 dB 恒定，dB/decade 斜率为 0.00（线性增长异常的判据是 ≈+10）。
Random Reference 与模 π 折叠理论精确吻合：底值 (2/π)²=-3.923 dB + (1-(2/π)²)/N 修正，
N=2000 时实测 -3.916 dB，偏差 <0.01 dB——证明归一化分母正确。

### 1.2 对照组相位误差：全部在理论底线

- Random Reference：51.93–51.99°，钉在均匀分布底线 180/√12≈51.96°，与 N、频率均无关；
- No Algorithm：50.4–51.1°（vs N）、37.0°→51.95°（10→30 MHz，低频部分有效属正常物理）；
- 频率回升检测（0.5 dB 阈值）：0 事件；任意阈值下 DPC/KF-DPC 功率曲线 0 次相邻回升。

### 1.3 频率扫描物理一致

DPC 功率 -0.20→-3.92 dB（10→150 MHz，渐近随机底线）；KF-DPC 全程优于 DPC 0.7–3.9 dB。
DPC 距离/UAV RMSE 不随频率变化（估计链不依赖载波相位）——自洽性验证通过。

### 1.4 Cluster 方法敏感性（由 trial 级诊断汇总；summary CSV 不含 Cluster 列）

节点数（簇数固定 40，簇大小≈N/40）：

| N | 20 | 100 | 500 | 2000 |
|---|---|---|---|---|
| Cluster DPC 功率 (dB) | -0.147 | -0.103 | -0.076 | -0.067 |
| 对 DPC 增益 (dB) | +0.61 | +0.68 | +0.71 | +0.71 |
| 对 KF-DPC 增益 (dB) | -0.031 | +0.013 | +0.028 | +0.031 |
| Cluster 节点 RMSE (m) | 0.63 | 0.49 | 0.43 | 0.42 |

- 簇内定位收益随 N 增强并在 N≥500 饱和；N≤30 时对 KF 支路为负增益（簇太小，伪观测引入的噪声大于收益）。

频率（N=1000）：

| 频率 | 10 | 30 | 50 | 80 | 100 | 150 MHz |
|---|---|---|---|---|---|---|
| Cluster DPC 功率 (dB) | -0.026 | -0.144 | -0.378 | -0.945 | -1.441 | -2.678 |
| 对 DPC 增益 (dB) | +0.18 | +1.54 | +2.93 | +2.95 | +2.47 | +1.24 |
| 对 KF-DPC 增益 (dB) | +0.01 | +0.07 | +0.19 | +0.46 | +0.63 | +0.69 |

- Cluster 对 DPC 增益峰值在 50–80 MHz；对 KF-DPC 增益随频率单调上升（高频段节点误差成为可控瓶颈）。

### 1.5 KF+cluster（node_prior 链）动态特性

- 收敛加速：普通节点 KF 需 ~3 个长块学会 5 m/s 涌浪漂移（node RMSE 0.66→0.54 m），
  node_prior 链块 0 即达 0.44 m；
- 小簇失效（N=30）：node_prior 节点 RMSE 0.61 m 全程劣于普通 KF 的 0.53 m——伪观测 σ=1 m
  过保守之外，小簇场景共识本身质量不足；
- 高频（150 MHz）：比 KF-DPC 高 +0.69 dB，相位 std 低 ~4.6°。

## 2. 异常扫描结论

对全部 sweep 数据做了三类系统扫描：单调性违例（该降反升）、块级平台化/冻结、trial 级离群值。
**无真实异常**：

- frequency 功率曲线任意阈值下 0 次回升；node_count 仅 6 次 ≤0.01 dB 的 MC 噪声级抖动；
- 块级无 ≥4 块平台、无完全重复值（仿真无冻结）；"恒定"项均为设计使然
  （DPC 节点 RMSE=√3·1 m、Cluster 节点 RMSE 与频率无关、随机底线 (2/π)² 等）；
- 唯一系统性离群：seed 17308 在所有频率点稳定偏低 ~3σ（最大 -0.19 dB）——同一组 seed 跨点复用
  导致的场景级重复，非计算错误，已被 trial 平均稀释。

## 3. main 与探索分支配对对比（同 seed，各 20 trial）

| 方法 | main (localization) | node_prior | 差距 |
|---|---|---|---|
| Cluster DPC 功率 (dB) | **-0.027** | -0.071 | main 赢 0.043（20/20 trial） |
| Cluster DPC 节点 RMSE (m) | **0.183** | 0.427 | main 好 0.24 |
| Cluster KF-DPC 功率 (dB) | -0.102 | **-0.065** | node_prior 赢 0.038（20/20） |
| KF-DPC 功率 (dB) | -0.176 | **-0.095** | node_prior 赢 0.081（20/20） |

误差分解（node_prior 分支，N=1000/20 MHz）：

| 方法 | UAV RMSE (m) | 节点 RMSE (m) | 功率 (dB) |
|---|---|---|---|
| DPC | 0.154 | 1.732 | -0.777 |
| Cluster DPC | 0.154 | 0.427 | -0.071 |
| KF-DPC | 0.033 | 0.539 | -0.095 |
| Cluster KF-DPC | 0.033 | 0.427 | -0.065 |

结论：

1. `uav_kf_prior_mode: none→dpc_only` 是纯收益（KF-DPC +0.081 dB；main 的硬注入使 UAV RMSE
   恶化到 0.566 m，劣于 DPC 共识的 0.154 m）；
2. `cluster_mode: localization→node_prior` 在节点定位上退步（0.183→0.427 m）：localization 的
   α=0.8 硬混合把 σ=0.2 m 的高精度相对几何直接用足，而 node_prior 给共识登记的信任度 σ=1 m
   远大于其真实误差（~0.18 m），Kalman 增益过小——**先验方差失配，非架构问题**；
3. DPC→KF-DPC 的 0.68 dB 提升主要由（被簇链复制的）节点 KF 承载，UAV KF 在 20 MHz 仅贡献
   0.006 dB（DPC 共识 UAV 估计已好至 0.154 m，误差预算由节点侧主导；该结论随频率升高而减弱）。

## 4. 后续改进方向（已论证未实施）

1. 把 `cluster_node_prior_noise_std` 从 1.0 降到 ≈0.2，或在线自适应
   `Rᵢ = Var_j[簇内观测者对 i 的投票]`——预期同时获得 localization 的精度（≈0.2 m、-0.03 dB）
   与 node_prior 的稳健性（小簇自动降权）；
2. 小簇（M<3）跳过伪观测注入或按 1/(M-1) 缩放 R，消除 N≤30 的负增益；
3. main 侧若保留 α 混合，建议换成方差加权（σ² 来自节点 KF 的 P 与投票样本方差），
   并叠加 innovation 门控抗离群。

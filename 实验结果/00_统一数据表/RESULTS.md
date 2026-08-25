# 基于通信子图的完整 Sweep 结果

## 实验版本

- 本地分支：`demo/communication-subgraph-cluster`
- Sweep 补丁提交：`710f30d Include subgraph cluster methods in factor sweeps`
- Cluster 模式：`subgraph`
- Monte Carlo：每个点 20 次，种子序列在同一因素内保持一致
- 计算设备：CUDA

Sweep 汇总和绘图现已包含 DPC、UAV+Node-KF DPC、Cluster DPC 和
Cluster UAV+Node-KF DPC。功率先在线性域进行 Monte Carlo 平均，再转换为 dB。

## 节点数 Sweep

- 节点数：20、30、50、75、100、150、200、300、500、1000、1500、2000
- 固定参数：20 MHz、通信边概率 0.03、40 个子图、8 个长块、每块 15 个短步

没有出现异常的线性增长：

| 方法 | 归一化功率跨度 | 单节点增益斜率 |
|---|---:|---:|
| DPC | 0.0359 dB | 20.0024 dB/decade |
| UAV+Node-KF DPC | 0.0142 dB | 19.9946 dB/decade |
| Cluster DPC | 0.0911 dB | 20.0362 dB/decade |
| Cluster UAV+Node-KF DPC | 0.0470 dB | 20.0097 dB/decade |

四种方法的自动 `suspicious_linear_growth` 标志均为 `False`。约
20 dB/decade 的单节点增益斜率对应理想相干功率的正常标度
`P_coherent / P_single ∝ N²`，不是归一化功率被错误地按节点数累加。

Cluster 方法在大节点数下缓慢改善：Cluster 节点 RMSE 从小规模时约
0.55 m 降到 N=2000 时的 0.461 m。这与固定 40 个子图时平均子图规模随
N 增大有关，属于当前实验设计中的结构性收益。

关键文件：

- `node_frequency/node_count_summary.csv`
- `node_frequency/node_count_scaling_diagnostics.csv`
- `node_frequency/node_count_all_methods_tail_power_db.png`
- `node_frequency/node_count_all_methods_node_rmse.png`

## 连通性 Sweep

为观察收敛速度，使用更明显的瞬态场景：

- N=300，40 个子图，20 MHz
- UAV 观测噪声标准差 10 m
- UAV KF 初始速度不确定度 10 m/s
- 8 个长块，每块 30 个短步，共 240 步（11.95 s）
- 通信边概率从 0.0005 扫描到 0.10

收敛时间定义为：K 步滑动平均首次进入该方法自身尾段稳态均值下方
0.5 dB 的带宽，并至少持续半个滑动窗口。

| 方法 | p=0.0005 | p=0.01 | p=0.10 | 低到高连通性降幅 |
|---|---:|---:|---:|---:|
| DPC | 5.55 s | 4.45 s | 1.90 s | 65.8% |
| UAV+Node-KF DPC | 4.35 s | 1.95 s | 1.45 s | 66.7% |
| Cluster DPC | 5.75 s | 4.60 s | 2.00 s | 65.2% |
| Cluster UAV+Node-KF DPC | 4.30 s | 1.95 s | 1.45 s | 66.3% |

连通性提高后四种方法的收敛时间近似单调下降。KF 两条曲线在
`p≈0.03` 后达到约 1.45 s 的饱和区；DPC 两条曲线到 p=0.10 仍有小幅收益。
瞬态损失同样明显下降，例如 Cluster UAV+Node-KF DPC 从 1.078 dB
下降到 0.086 dB。

关键文件：

- `connectivity_convergence/connectivity_convergence_summary.csv`
- `connectivity_convergence/connectivity_time_curves.csv`
- `connectivity_convergence/connectivity_all_methods_convergence_time.png`
- `connectivity_convergence/connectivity_all_methods_transient_deficit.png`
- `connectivity_convergence/connectivity_selected_time_curves.png`

## 频率 Sweep

- 频率：10、20、30、50、80、100、150 MHz
- N=1000、通信边概率 0.03、40 个子图

随频率提高，相同米级定位误差产生更大的相位误差，四种方法的归一化功率
整体下降。Cluster UAV+Node-KF DPC 在全部频率点保持最佳或并列最佳。
没有相邻频率点出现超过 0.5 dB 的异常回升；DPC 在 100 到 150 MHz
仅有约 0.019 dB 的微小回升，处于随机相位性能底附近，不构成异常趋势。

关键文件：

- `node_frequency/frequency_summary.csv`
- `node_frequency/frequency_all_methods_tail_power_db.png`
- `node_frequency/frequency_all_methods_phase_std_deg.png`

## 解释限制

当前通信图仍是随机边图并带环形骨架，`global_connectivity` 表示额外通信边
概率，不是物理距离半径。论文如果要把结果解释为通信范围，应进一步使用
距离阈值图或随机几何图。

此外，节点数 Sweep 固定为 40 个子图，因此节点数增加同时增大平均子图规模。
若要把“节点数量效应”和“子图规模效应”完全分离，建议增加一组
`n_clusters ∝ N`、平均子图规模固定的对照 Sweep。

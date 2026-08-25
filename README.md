# UAV DPC 模块化实验项目

这是从原实验中独立迁移出来的新项目，代码只依赖本目录内的 `dfpc_experiment` 包，不再引用旧的实验文件夹。

## 目录结构

- `dfpc_experiment/config.py`：实验参数和命令行接口，优先修改这里。
- `dfpc_experiment/scenario.py`：UAV、节点位置和观测模型。
- `dfpc_experiment/trajectory.py`：节点在线估计 UAV 匀速轨迹，并输出位置、速度及协方差。
- `dfpc_experiment/kalman.py`：UAV 与浮标的批量三维恒速 Kalman 滤波；UAV KF 直接融合六维轨迹观测。
- `dfpc_experiment/sim_core.py`：从旧仿真文件迁移来的基础数学函数。
- `dfpc_experiment/experiment.py`：单次完整实验和 Monte Carlo 主循环。
- `dfpc_experiment/sweeps.py`：敏感性分析实验。
- `dfpc_experiment/metrics.py`：相位误差、合成功率、相对单节点增益等指标。
- `dfpc_experiment/plots.py`：绘图函数。
- `dfpc_experiment/debug_tools.py`：调试输出接口。
- `dfpc_experiment/run_experiment.py`：运行单组实验。
- `dfpc_experiment/run_sweeps.py`：运行因素扫描实验。

## 快速运行

```powershell
python uav_dfpc_modular_project\dfpc_experiment\run_experiment.py --device cpu
```

默认浮标采用二维高斯漂移—扩散模型：均值速度 5 m/s、方向 0°、扩散强度 0.5 m/√s。
`Ts` 是浮标运动积分步长，`K` 是每个长块的短时间步/共识迭代次数，block 物理时长恒为 `K*Ts`。
`TL` 仅为兼容旧命令行参数保留，不再参与实验主循环。
所有 DPC 距离均由三维坐标计算；当前 UAV 高度固定为 120 m、浮标高度固定为 0 m，
但 KF 状态已保留 `z` 和 `vz`，以后可以直接接入高度变化。
普通 DPC 仍使用最近 `K` 份观测的滚动共识轨迹计算相位。KF-DPC 只用最初
`K` 份观测拟合的位置、速度和 OLS 协方差初始化一次 UAV KF；此后历史信息
保存在持续递归的 KF 后验中。每个 `Ts` 先分别对预测状态和当前新增的位置观测
执行一次邻域共识，再按共识观测的有效协方差更新 KF；每份原始观测只进入一次。

运行一个较小的调试实验：

```powershell
python uav_dfpc_modular_project\dfpc_experiment\run_experiment.py --N 100 --mc_trials 2 --T_long 2 --K 5 --device cpu --debug
```

运行节点数扫描：

```powershell
python uav_dfpc_modular_project\dfpc_experiment\run_sweeps.py --factors node_count --node_count_values 50,100,200 --device cpu
```

扫描海浪均值速度：

```powershell
python uav_dfpc_modular_project\dfpc_experiment\run_sweeps.py --factors wave_speed --wave_speed_values 0.5,1,2,5 --device cpu
```

使用 Monte Carlo 线性功率平均检查 KF/DPC 频率曲线：

```powershell
python uav_dfpc_modular_project\dfpc_experiment\run_sweeps.py --factors frequency --frequency_values 30,50,100,150,200,250,300 --mc_trials 50 --device cpu
```

输出中的 `frequency_dfpc_vs_kf_final_power_db.png` 是与旧版
`full_sweep_final_power_db.png` 对应的直接对比图。固定米级定位误差在高频下会转成更大相位误差，
因此高频功率下降是正常物理结果；应通过多次 Monte Carlo 平均消除单场景产生的偶然深谷，而不是对结果曲线做后处理平滑。

所有方法默认采用理想 0/pi 极性选择：残余相位按模 pi 折叠到 `[-90, 90) deg`，再进行功率和相位误差统计；这是极性反转的物理等价，不是截断。

调试频率曲线中的偶然回升：

```powershell
python uav_dfpc_modular_project\dfpc_experiment\run_sweeps.py `
  --factors frequency `
  --frequency_values 30,50,100,150,200,250,300 `
  --mc_trials 20 `
  --rebound_threshold_db 0.5 `
  --debug `
  --debug_trial 0 `
  --debug_blocks=-1 `
  --debug_iterations=-1 `
  --debug_max_nodes 32 `
  --device cpu
```

主要诊断文件：

- `frequency_rebound_events.csv`：相邻频率回升、主导 trial seed 和自动原因分类；
- `frequency_trial_diagnostics.csv`：每个频率、每个 trial 的 final/tail 统计；
- `frequency_block_trial_diagnostics.csv`：继续定位到具体物理 block；
- `frequency/point_XX/debug/debug_node_samples.csv`：指定 trial/block 下逐节点三维坐标、距离误差与残余相位；
- `frequency_rebound_debug_report.md`：回升事件的人可读索引。

输出默认保存在 `uav_dfpc_modular_project/modular_dfpc_outputs` 下，也可以用 `--out_dir` 指定新位置。

## 已归档实验结果

当前分支已经完成的节点数、连通度、频率、联合网格和版本对比实验，统一整理在
[`实验结果`](实验结果/README.md) 目录。该目录保留论文分析所需的标准化数据表、关键图和报告，
并说明各类实验的功率、相位与定位误差口径。

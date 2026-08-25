# main Cluster 与 v1.1.1 完整 Sweep 对比

## 结论

仓库 `main`（commit `0a16f07d7a2da8fb730cb3a72086cd8d972a80b6`）的默认
`localization` Cluster 明显优于 v1.1.1 默认 `node_prior` Cluster，尤其解决了节点估计随
block 恶化的问题。但 main 仍不满足原期望顺序：

`Cluster-KF-DPC > KF-DPC > Cluster-DPC > DPC > Random`

main 的常见实际顺序（N≥50）是：

`Cluster-DPC > Cluster-KF-DPC > KF-DPC > DPC > Random`

31 个扫描点中，严格满足原期望全部四个不等式的点数仍为 0/31。

## 实验口径

- main commit：`0a16f07d7a2da8fb730cb3a72086cd8d972a80b6`
- 对照 release：`v1.1.1`，commit `aefba116fa838b1e851100ccccf38b0188edf82c`
- 两边使用相同参数、相同 20 个 seed、相同 CPU 聚合脚本
- `N=1000`、`T_long=8`、`K=30`，每点 20 次 Monte Carlo
- 节点数 12 点、连通性 12 点、频率 7 点，共 620 个 trial 任务
- main：186 条聚合记录、3720 条 trial 方法记录、29760 条 block 方法记录
- 运行失败 0，聚合 NaN/Inf 0，main 的 31 项单元测试全部通过

## main 自身排序

| 相邻关系 | main 通过点数 | v1.1.1 通过点数 |
|---|---:|---:|
| Cluster-KF-DPC > KF-DPC | 29/31 | 0/31 |
| KF-DPC > Cluster-DPC | 0/31 | 31/31 |
| Cluster-DPC > DPC | 31/31 | 22/31 |
| DPC > Random | 31/31 | 31/31 |

main 中 Cluster-KF-DPC 只在 N=20、N=30 时没有严格高于 KF-DPC，两者相等。这两个
节点数都不超过默认簇数 40，簇内没有有效 peer 时 localization 不做额外修正，符合预期。
从 N=50 开始 Cluster-KF-DPC 超过 KF-DPC；N≥100 后 20/20 配对试验均支持这一关系。

## 基准点：N=1000、p=0.03、20 MHz

| 方法 | main (dB) | v1.1.1 (dB) | main−v1.1.1 |
|---|---:|---:|---:|
| Cluster DPC | -0.027 | -0.499 | +0.472 |
| Cluster-KF-DPC | -0.102 | -0.494 | +0.391 |
| KF-DPC | -0.176 | -0.095 | -0.081 |
| DPC | -0.777 | -0.777 | 0.000 |
| Random | -3.917 | -3.917 | 0.000 |

main 在基准点将 Cluster 节点 RMSE 降至 0.183 m；v1.1.1 为 1.629 m，普通节点 KF
约为 0.539 m。因此 main 的两个 Cluster 方法都获得明显收益。

## 31 个扫描点上的分支差异

以下“平均”是 31 个扫描点的 dB 差值算术平均，用于概括分支差异，不代表新的物理总指标：

| 方法 | main−v1.1.1 平均 | 最小 | 最大 | main 更强的点数 |
|---|---:|---:|---:|---:|
| Cluster DPC | +1.353 dB | +0.120 | +3.363 | 31/31 |
| Cluster-KF-DPC | +0.976 dB | -0.016 | +2.950 | 30/31 |
| KF-DPC | -0.303 dB | -1.031 | -0.020 | 0/31 |
| DPC | 0.000 dB | 0.000 | 0.000 | 0/31（完全一致） |
| Random | 0.000 dB | 0.000 | 0.000 | 0/31（完全一致） |

Cluster-KF-DPC 唯一未优于 v1.1.1 的点是 `p=0.0005`，差 -0.016 dB，同 seed 配对
胜率 55%，属于接近噪声级的差别。Cluster DPC 在 31/31 点均优于 v1.1.1，配对支持率
均为 100%。

## 为什么 main Cluster 更好

### main：localization

main 使用普通节点 KF 作为自身位置先验，簇内 peer 根据相对位置测量生成目标节点估计，
完成簇内共识后按 `cluster_alpha=0.8` 直接修正最终节点状态。该修正只用于当前步评估，
不会再次写回节点 KF 协方差。

因此它不会把相关的 cluster 结果反复当成独立 Kalman 测量，也不会在单节点簇上用自己的
状态更新自己。

### v1.1.1：node_prior

v1.1.1 将簇内共识作为固定 `R=0.2²I` 的伪观测，每个短步回灌第二套节点 KF。单节点簇
也执行零创新自更新；较大簇中，共识结果与 KF 状态高度相关，却仍被当作独立测量，导致
协方差过度收缩和共同误差积累。

block 级证据（20 trial 均值）：

| block | main Cluster 节点 RMSE | v1.1.1 Cluster 节点 RMSE |
|---:|---:|---:|
| 0 | 0.216 m | 0.435 m |
| 1 | 0.201 m | 0.841 m |
| 3 | 0.186 m | 1.121 m |
| 5 | 0.184 m | 1.446 m |
| 7 | 0.183 m | 1.699 m |

main localization 收敛并保持稳定，v1.1.1 node_prior 则持续恶化。

## 为什么 main 仍是 Cluster-DPC 最强

这次是完整 main 分支运行，不是只替换 `cluster.py`，所以还包含非 Cluster 差异：

- main 默认 `uav_kf_prior_mode=none`，且没有 v1.1.1 的保守协方差共识修复；
- main 基准点 UAV-KF RMSE 约 0.566 m，v1.1.1 约 0.030 m；
- main 的 DPC UAV RMSE 约 0.154 m，反而好于 main 的 UAV-KF；
- Cluster-DPC 与 Cluster-KF-DPC 使用相同的高精度 localization 节点位置，前者的 UAV
  估计更准，所以 Cluster-DPC 高于 Cluster-KF-DPC。

也就是说，main 证明了 localization Cluster 节点路径有效，但 main 的旧 UAV-KF 路径拖累了
Cluster-KF-DPC。v1.1.1 则相反：UAV-KF 已修好，但 node_prior Cluster 节点路径失效。

## 建议

下一步最有信息量的实验不是直接选择整个 main 或整个 v1.1.1，而是在 v1.1.1 的 KF 与
协方差修复基础上，仅恢复 main 的 `localization` Cluster 路径，再以同 seed 重跑。这样才能
隔离验证“main Cluster + v1.1.1 KF”的组合排序。

本轮只检出、运行和比较 main，没有修改 main 或 v1.1.1 的算法代码。

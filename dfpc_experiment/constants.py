"""实验方法名和指标名。

这些字符串会被多个模块共用，包括实验主循环、画图、CSV 输出和
sweep 汇总。集中放在这里可以避免不同文件里写出不一致的列名。
"""

# 评估基线，不属于保留的算法分支。
METHOD_NO_ALG = "No Algorithm"

# 保留的两个算法：DPC 使用轨迹状态通信更新；KF-DPC 在通信更新前
# 先用跨时刻的 UAV/浮标后验预测与当前观测完成 Kalman 更新。
METHOD_DPC = "DPC"
METHOD_KF_DPC = "KF-DPC"

# Backward-compatible Python identifiers for existing experiment scripts.  New
# reports and tables use DPC because the algorithm has been renamed.
METHOD_DFPC = METHOD_DPC
METHOD_KF_DFPC = METHOD_KF_DPC
METHODS = [
    METHOD_NO_ALG,
    METHOD_DFPC,
    METHOD_KF_DFPC,
]

# 每个迭代步都会保存这些指标。
# 注意：
# - gain_linear/norm_db 是相对于理想多节点相干功率 P_ideal 的归一化功率。
# - gain_over_single_* 是多节点相对于单节点发射的增益，适合节点数实验。
# - *_rmse 用来解释相位误差来自 UAV 估计还是节点位置估计。
METRIC_KEYS = [
    "gain_linear",
    "norm_db",
    "power_linear",
    "ideal_power_linear",
    "single_node_mean_power_linear",
    "single_node_best_power_linear",
    "gain_over_single_mean_linear",
    "gain_over_single_mean_db",
    "gain_over_single_best_linear",
    "gain_over_single_best_db",
    "phase_std_deg",
    "phase_rmse_deg",
    "distance_rmse",
    "node_rmse",
    "uav_rmse",
    "uav_line_intercept_rmse",
    "uav_velocity_rmse",
]

# 这些指标需要用 nanmean 做 Monte Carlo 平均；仅随机相位理论参考没有定位误差。
ERROR_KEYS = [
    "phase_std_deg",
    "phase_rmse_deg",
    "distance_rmse",
    "node_rmse",
    "uav_rmse",
    "uav_line_intercept_rmse",
    "uav_velocity_rmse",
]

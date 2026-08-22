"""实验方法名和指标名。

这些字符串会被多个模块共用，包括实验主循环、画图、CSV 输出和
sweep 汇总。集中放在这里可以避免不同文件里写出不一致的列名。
"""

# 无算法基线：所有节点不做相位补偿，直接发射。
METHOD_RANDOM_REFERENCE = "Random Phase Reference"
METHOD_NO_ALG = "No Algorithm"

# 每个节点只使用当前原始位置观测进行开环相位补偿，不做共识或滤波。

# DPC 方法：节点根据估计距离补偿相位。
METHOD_DPC = "DPC"
METHOD_NODE_KF_DPC = "DPC + Node-KF"
METHOD_KF_DPC = "UAV+Node-KF DPC"
METHOD_CLUSTER_DPC = "Cluster DPC"
METHOD_CLUSTER_KF_DPC = "Cluster UAV+Node-KF DPC"

# Backward-compatible Python identifiers for existing experiment scripts.  New
# reports and tables use DPC because the algorithm has been renamed.
METHOD_DFPC = METHOD_DPC
METHOD_NODE_KF_DFPC = METHOD_NODE_KF_DPC
METHOD_KF_DFPC = METHOD_KF_DPC
METHOD_CLUSTER_DFPC = METHOD_CLUSTER_DPC
METHOD_CLUSTER_KF_DFPC = METHOD_CLUSTER_KF_DPC
METHODS = [
    METHOD_RANDOM_REFERENCE,
    METHOD_NO_ALG,
    METHOD_DFPC,
    METHOD_NODE_KF_DFPC,
    METHOD_CLUSTER_DFPC,
    METHOD_KF_DFPC,
    METHOD_CLUSTER_KF_DFPC,
]

# Methods retained when cluster trajectory correction is disabled.
CORE_METHODS = [
    METHOD_RANDOM_REFERENCE,
    METHOD_NO_ALG,
    METHOD_DFPC,
    METHOD_NODE_KF_DFPC,
    METHOD_KF_DFPC,
]

# 每个时间步都会保存这些指标。
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

"""实验主循环。

这个文件把各模块串起来：
1. 初始化场景；
2. 每个长块推进 K 个 Ts 短步并产生观测；
3. 最近 K 次观测构成连续滑动轨迹窗口；预热完成后，每个短步都用完整 UAV
   轨迹 [位置,速度] 及协方差更新窗口级 KF；
4. 滑动窗口每次前移一个 Ts，不在长块边界清空轨迹信息；
5. 多个 Monte Carlo trial 做平均。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import backend
from .cluster import (
    ClusterTrajectoryLocalizer,
    cluster_trajectory_consensus,
    communication_graph_clusters,
    rotating_cluster_trajectory_correction,
)
from .config import ExperimentConfig, validate_config
from .constants import (
    CORE_METHODS,
    ERROR_KEYS,
    METHOD_CLUSTER_DFPC,
    METHOD_CLUSTER_KF_DFPC,
    METHOD_DFPC,
    METHOD_KF_DFPC,
    METHOD_NODE_KF_DFPC,
    METHOD_NO_ALG,
    METHOD_RANDOM_REFERENCE,
    METHODS,
    METRIC_KEYS,
)
from .debug_tools import DebugRecorder
from .kalman import (
    BatchCVKalman3D,
    make_cv3d_filter,
)
from .metrics import evaluate_dfpc, mean_stack, random_phase_reference_metrics
from .scenario import (
    advance_truth_one_short_step,
    current_truth,
    geometry_terms,
    init_scene,
    observe_positions,
)
from .trajectory import (
    INTERCEPT_COLUMNS,
    VELOCITY_COLUMNS,
    TrajectoryPrediction,
    consensus_independent_covariance,
    init_line_estimator,
    normalize_trajectory_directions,
    predict_positions,
    predict_trajectory,
    trajectory_state_at_time,
    trajectory_state_from_position_velocity,
    update_local_line_estimates,
)


def consensus_covariance(W: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Conservatively mix covariance after consensus with unknown correlation."""
    weights = np.asarray(W, dtype=np.float64)
    covariance = np.asarray(P, dtype=np.float64)
    if weights.shape[0] != weights.shape[1] or weights.shape[0] != covariance.shape[0]:
        raise ValueError("W and P must have matching leading dimensions")
    return np.einsum("ij,jkl->ikl", weights, covariance)


def dpc_trajectory_prediction(
    line_params: np.ndarray,
    local_prediction: TrajectoryPrediction,
    weight_matrix: np.ndarray,
    position_model_std: float,
    velocity_model_std: float,
    time_s: float,
) -> TrajectoryPrediction:
    """Build the full-state DPC pseudo-measurement after one consensus step."""
    covariance = consensus_independent_covariance(
        weight_matrix,
        local_prediction.covariance,
    )
    position_model_var = max(float(position_model_std), 0.0) ** 2
    covariance[:, :3, :3] += position_model_var * np.eye(3)[None, :, :]
    velocity_model_var = max(float(velocity_model_std), 0.0) ** 2
    covariance[:, 3:, 3:] += velocity_model_var * np.eye(3)[None, :, :]
    return TrajectoryPrediction(
        state=trajectory_state_at_time(line_params, time_s),
        covariance=covariance,
    )


def extrapolate_window_filter(
    kf: BatchCVKalman3D | None,
    elapsed_s: float,
    fallback: TrajectoryPrediction,
) -> np.ndarray:
    """Return a causal UAV state without mutating the window-level filter."""
    if kf is None:
        return fallback.state.copy()
    elapsed = max(float(elapsed_s), 0.0)
    position = kf.positions + elapsed * kf.velocities
    return np.column_stack([position, kf.velocities])


def update_window_filter(
    kf: BatchCVKalman3D | None,
    trajectory: TrajectoryPrediction,
    cfg: ExperimentConfig,
    update_interval_s: float,
    weight_matrix: np.ndarray,
    weight_matrix_gpu: object,
    extra_trajectory_state: np.ndarray | None = None,
    extra_trajectory_covariance: np.ndarray | None = None,
) -> BatchCVKalman3D:
    """Initialize or update one UAV KF after a complete observation window."""
    if kf is None:
        kf = make_cv3d_filter(
            trajectory.positions,
            update_interval_s,
            cfg.uav_obs_noise,
            cfg.uav_kf_accel_std,
            cfg.uav_kf_initial_velocity_std,
            initial_velocity_xyz=trajectory.velocities,
        )
        kf.x = trajectory.state.copy()
        kf.P = trajectory.covariance.copy()
    else:
        kf.predict()
        kf.update_trajectory_measurement(
            trajectory.state,
            trajectory.covariance,
        )

    if extra_trajectory_state is not None:
        if extra_trajectory_covariance is None:
            raise ValueError("extra trajectory covariance is required")
        kf.update_trajectory_measurement(
            extra_trajectory_state,
            extra_trajectory_covariance,
        )

    kf.x = backend.consensus_linear_accel(
        weight_matrix,
        weight_matrix_gpu,
        kf.x,
        1,
    ).copy()
    kf.P = consensus_covariance(weight_matrix, kf.P)
    return kf


def run_single_trial(
    cfg: ExperimentConfig,
    seed: int,
    trial_index: int = 0,
    debug: DebugRecorder | None = None,
) -> dict[str, Any]:
    """运行一次 Monte Carlo trial。

    注意：这里的 iteration 是算法迭代，同时也是物理世界的一次 Ts 短步。
    相邻 block 之间物理世界严格推进 K*Ts 秒。
    """
    rng_scene = np.random.default_rng(seed)
    rng_obs = np.random.default_rng(seed + 3000003)
    rng_cluster_partition = np.random.default_rng(seed + 4000003)
    # Keep graph-partition traversal from shifting the relative-measurement
    # noise stream when connectivity changes.
    rng_cluster_measurement = np.random.default_rng(seed + 5000003)

    fc_hz = float(cfg.fc_mhz) * 1e6
    lam = 3e8 / fc_hz
    k_const = 2.0 * np.pi / lam
    iter_count = cfg.K
    total_steps = cfg.T_long * iter_count
    block_duration_s = float(cfg.K) * float(cfg.Ts)

    state = init_scene(cfg, rng_scene)
    methods = METHODS if cfg.enable_cluster else CORE_METHODS
    cluster_labels = (
        communication_graph_clusters(
            state.W_global,
            cfg.n_clusters,
            rng_cluster_partition,
        )
        if cfg.enable_cluster
        else None
    )
    adjacency = state.W_global > 1e-12
    np.fill_diagonal(adjacency, False)
    cluster_sizes = (
        np.bincount(cluster_labels) if cluster_labels is not None else np.array([], dtype=np.int64)
    )
    graph_diagnostics = {
        "mean_degree": float(np.mean(np.sum(adjacency, axis=1))),
        "cluster_count": float(cluster_sizes.size),
        "mean_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes.size else np.nan,
    }
    # System/calibration phase error is persistent over one trial.  Drawing a
    # new independent value at every short step creates non-physical power
    # zigzags even when the geometry moves smoothly.
    eps_trial = rng_scene.normal(0.0, np.deg2rad(cfg.system_phase_std_deg), size=cfg.N)
    true_line_params = np.array(
        [
            state.uav_center_true[0],
            state.uav_velocity_true[0],
            state.uav_center_true[1],
            state.uav_velocity_true[1],
            state.uav_center_true[2],
            state.uav_velocity_true[2],
        ],
        dtype=np.float64,
    )
    uav_kf_window: BatchCVKalman3D | None = None
    uav_kf_cluster_window: BatchCVKalman3D | None = None
    uav_kf_window_time_s: float | None = None
    uav_kf_window_updates = 0
    line_estimator = init_line_estimator(cfg.N, window_size=cfg.K)
    node_kf: BatchCVKalman3D | None = None
    node_kf_prior: BatchCVKalman3D | None = None
    cluster_localizer = (
        ClusterTrajectoryLocalizer(cfg.N)
        if cfg.enable_cluster and cfg.cluster_mode in {"subgraph", "localization", "node_prior"}
        else None
    )
    metrics = {
        method: {key: np.zeros(total_steps, dtype=np.float64) for key in METRIC_KEYS}
        for method in methods
    }
    cluster_diagnostics = {
        "effective_peer_count_mean": np.full(total_steps, np.nan, dtype=np.float64),
        "nodes_with_subgraph_peers": np.zeros(total_steps, dtype=np.int64),
    }
    for key in ["distance_rmse", "node_rmse", "uav_rmse", "uav_line_intercept_rmse", "uav_velocity_rmse"]:
        metrics[METHOD_RANDOM_REFERENCE][key].fill(np.nan)

    for block in range(cfg.T_long):
        # 长块只用于输出分段；轨迹窗口跨块连续滑动，不在边界重置。
        # The previous block ends at short step K-1. Advance once before the
        # next block so every recorded sample is exactly Ts apart and block
        # boundaries never duplicate the same physical instant.
        if block > 0:
            advance_truth_one_short_step(cfg, state, rng_scene, cfg.Ts)
        # 观测窗口开始：读取真实位置并生成窗口内第一份观测。
        time_s = block * block_duration_s
        p_u_true, buoy_true_short = current_truth(state)
        uav_obs_short, buoy_obs_short = observe_positions(cfg, p_u_true, buoy_true_short, rng_obs)
        # 每个节点只使用最近 K 份观测拟合三维匀速轨迹。
        line_params_dfpc = update_local_line_estimates(line_estimator, time_s, uav_obs_short)
        local_trajectory_prediction = predict_trajectory(
            line_params_dfpc,
            line_estimator,
            time_s,
            cfg.uav_obs_noise,
            cfg.uav_kf_initial_velocity_std,
        )

        # 节点（浮标）短时间尺度 KF：dt=Ts，velocity_propagate=True（浮标随 Ts 漂移），
        # 初始速度取海浪平均速度。每个 iteration predict+update 用一份新原始短观测 z_b_short。
        if node_kf is None:
            node_kf = make_cv3d_filter(
                buoy_obs_short,
                cfg.Ts,
                cfg.buoy_center_obs_noise,
                cfg.buoy_kf_accel_std,
                cfg.buoy_kf_initial_velocity_std,
                initial_velocity_xyz=state.buoy_wave_mean_velocity,
                velocity_propagate=True,
            )
            # 浮标属于不同物理目标：每个浮标只维护自己的 KF 后验，
            # 不对不同浮标的绝对位置做共识平均。
            node_state_kf = node_kf.x.copy()

            if cfg.enable_cluster and cfg.cluster_mode == "node_prior":
                node_kf_prior = make_cv3d_filter(
                    buoy_obs_short,
                    cfg.Ts,
                    cfg.buoy_center_obs_noise,
                    cfg.buoy_kf_accel_std,
                    cfg.buoy_kf_initial_velocity_std,
                    initial_velocity_xyz=state.buoy_wave_mean_velocity,
                    velocity_propagate=True,
                )
                node_state_kf_prior = node_kf_prior.x.copy()

        for iteration in range(iter_count):
            sidx = block * iter_count + iteration
            if iteration > 0:
                advance_truth_one_short_step(cfg, state, rng_scene, cfg.Ts)
                time_s = (block * cfg.K + iteration) * float(cfg.Ts)
                p_u_true, buoy_true_short = current_truth(state)
                uav_obs_short, buoy_obs_short = observe_positions(cfg, p_u_true, buoy_true_short, rng_obs)
                line_params_dfpc = update_local_line_estimates(line_estimator, time_s, uav_obs_short)
                local_trajectory_prediction = predict_trajectory(
                    line_params_dfpc,
                    line_estimator,
                    time_s,
                    cfg.uav_obs_noise,
                    cfg.uav_kf_initial_velocity_std,
                )

            # 解除“block 内真值固定”：除 iteration 0 外，每个短步都推进一个 Ts 后重新观测。
            # ---- 1) 短时间尺度 DPC：对三维轨迹、三轴速度和单位航向做一次邻居共识。
            cluster_uav_prior_state = None
            cluster_uav_prior_covariance = None
            if cfg.enable_cluster and (
                cfg.cluster_mode == "uav_prior"
                or cfg.uav_kf_prior_mode == "dpc_plus_cluster"
            ):
                cluster_prior_line_params = cluster_trajectory_consensus(
                    line_params_dfpc,
                    state.W_global,
                    cluster_labels,
                    consensus_steps=cfg.cluster_localization_iterations,
                    noise_std=cfg.cluster_trajectory_noise_std,
                    rng=rng_cluster_measurement,
                )
                cluster_prior_line_params = normalize_trajectory_directions(
                    cluster_prior_line_params
                )
                cluster_uav_prior_state = trajectory_state_at_time(
                    cluster_prior_line_params,
                    time_s,
                )
                cluster_uav_prior_covariance = local_trajectory_prediction.covariance.copy()
                cluster_uav_prior_covariance[:, :3, :3] += (
                    max(float(cfg.cluster_uav_prior_noise_std), 0.0) ** 2
                    * np.eye(3)[None, :, :]
                )
                cluster_uav_prior_covariance[:, 3:, 3:] += (
                    max(float(cfg.cluster_trajectory_noise_std), 0.0) ** 2
                    * np.eye(3)[None, :, :]
                )
            line_params_dfpc = backend.consensus_linear_accel(
                state.W_global,
                state.W_global_gpu,
                line_params_dfpc,
                1,
            )
            line_params_dfpc = normalize_trajectory_directions(line_params_dfpc)
            uav_trajectory_dfpc = dpc_trajectory_prediction(
                line_params_dfpc,
                local_trajectory_prediction,
                state.W_global,
                cfg.uav_dpc_prior_noise_std,
                cfg.uav_kf_accel_std * cfg.Ts,
                time_s,
            )
            uav_est_dfpc = uav_trajectory_dfpc.positions

            # ---- 2) 滑动窗口 UAV KF：预热 K 份观测后，每个 Ts 更新一次。
            if line_estimator.count >= cfg.K:
                standard_extra_state = (
                    cluster_uav_prior_state
                    if cfg.uav_kf_prior_mode == "dpc_plus_cluster"
                    else None
                )
                standard_extra_covariance = (
                    cluster_uav_prior_covariance
                    if standard_extra_state is not None
                    else None
                )
                uav_kf_window = update_window_filter(
                    uav_kf_window,
                    uav_trajectory_dfpc,
                    cfg,
                    cfg.Ts,
                    state.W_global,
                    state.W_global_gpu,
                    standard_extra_state,
                    standard_extra_covariance,
                )
                if cfg.enable_cluster and cfg.cluster_mode == "uav_prior":
                    uav_kf_cluster_window = update_window_filter(
                        uav_kf_cluster_window,
                        uav_trajectory_dfpc,
                        cfg,
                        cfg.Ts,
                        state.W_global,
                        state.W_global_gpu,
                        cluster_uav_prior_state,
                        cluster_uav_prior_covariance,
                    )
                uav_kf_window_time_s = time_s
                uav_kf_window_updates += 1

            elapsed_from_window_s = (
                0.0
                if uav_kf_window_time_s is None
                else time_s - uav_kf_window_time_s
            )
            uav_window_state = extrapolate_window_filter(
                uav_kf_window,
                elapsed_from_window_s,
                uav_trajectory_dfpc,
            )
            uav_est_kf = uav_window_state[:, :3]
            kf_velocity = uav_window_state[:, 3:]
            cluster_uav_est_kf = uav_est_kf
            cluster_kf_velocity = kf_velocity
            if uav_kf_cluster_window is not None:
                cluster_window_state = extrapolate_window_filter(
                    uav_kf_cluster_window,
                    elapsed_from_window_s,
                    uav_trajectory_dfpc,
                )
                cluster_uav_est_kf = cluster_window_state[:, :3]
                cluster_kf_velocity = cluster_window_state[:, 3:]

            # ---- 3) 短时间尺度节点 KF：predict（dt=Ts）+ 用一份新原始短观测 z_b 更新。
            #     浮标不做跨节点共识，每个节点只使用自己的 KF 状态。
            if sidx > 0:
                node_kf.predict()
            z_b_short_kf = buoy_true_short + rng_obs.normal(0.0, cfg.buoy_center_obs_noise, size=(cfg.N, 3))
            if sidx > 0:
                node_kf.update(z_b_short_kf)
            node_state_kf = node_kf.x.copy()
            node_est_kf = node_state_kf[:, :3]
            if node_kf_prior is not None:
                if sidx > 0:
                    node_kf_prior.predict()
                if sidx > 0:
                    node_kf_prior.update(z_b_short_kf)
                node_state_kf_prior = node_kf_prior.x.copy()
                assert cluster_localizer is not None
                cluster_node_consensus = cluster_localizer.update(
                    node_state_kf_prior,
                    state.W_global,
                    cluster_labels,
                    buoy_true_short,
                    rng_cluster_measurement,
                    relative_noise_std=cfg.cluster_relative_noise_std,
                    constraint_type=cfg.cluster_constraint_type,
                    consensus_steps=cfg.cluster_localization_iterations,
                    dt=cfg.Ts,
                    blend_with_own=False,
                )
                node_kf_prior.update_position_measurement(
                    cluster_node_consensus[:, :3],
                    cfg.cluster_node_prior_noise_std,
                )
                node_state_kf_prior = node_kf_prior.x.copy()

            # 2. 当前真值下的几何量、相位噪声、无算法基线（每个短步真值都变，必须重算）。
            phi_true, amp, p_ideal, p_single_mean, p_single_best = geometry_terms(
                cfg, p_u_true, buoy_true_short, k_const
            )
            eps = eps_trial
            random_reference_result = random_phase_reference_metrics(
                phi_true,
                amp,
                eps,
                p_ideal,
                p_single_mean,
                p_single_best,
            )

            # 开环原始观测基线：每个节点仅使用自己当前的 UAV/节点位置
            # 观测形成相位命令，不做节点间共识，也不做滤波。
            no_alg_result = evaluate_dfpc(
                uav_obs_short,
                buoy_obs_short,
                p_u_true,
                buoy_true_short,
                phi_true,
                amp,
                eps,
                k_const,
                p_ideal,
                p_single_mean,
                p_single_best,
            )
            no_alg_result["uav_line_intercept_rmse"] = np.nan
            no_alg_result["uav_velocity_rmse"] = np.nan

            kf_intercept_xy = uav_est_kf[:, :2] - time_s * kf_velocity[:, :2]
            kf_line_params = np.column_stack(
                [kf_intercept_xy[:, 0], kf_velocity[:, 0], kf_intercept_xy[:, 1], kf_velocity[:, 1]]
            )

            # 3. 短时间尺度 DPC 指标（节点估计用当前原始观测 buoy_obs_short，保持两路公平）。
            dfpc_result = evaluate_dfpc(
                uav_est_dfpc,
                buoy_obs_short,
                p_u_true,
                buoy_true_short,
                phi_true,
                amp,
                eps,
                k_const,
                p_ideal,
                p_single_mean,
                p_single_best,
            )
            intercept_error = line_params_dfpc[:, INTERCEPT_COLUMNS] - true_line_params[INTERCEPT_COLUMNS]
            velocity_error = line_params_dfpc[:, VELOCITY_COLUMNS] - true_line_params[VELOCITY_COLUMNS]
            dfpc_result["uav_line_intercept_rmse"] = float(
                np.sqrt(np.mean(np.sum(intercept_error**2, axis=1)))
            )
            dfpc_result["uav_velocity_rmse"] = float(
                np.sqrt(np.mean(np.sum(velocity_error**2, axis=1)))
            )

            # Ablation: DPC UAV estimate plus Node-KF, without clustering.
            node_kf_dfpc_result = evaluate_dfpc(
                uav_est_dfpc,
                node_est_kf,
                p_u_true,
                buoy_true_short,
                phi_true,
                amp,
                eps,
                k_const,
                p_ideal,
                p_single_mean,
                p_single_best,
            )
            node_kf_dfpc_result["uav_line_intercept_rmse"] = dfpc_result["uav_line_intercept_rmse"]
            node_kf_dfpc_result["uav_velocity_rmse"] = dfpc_result["uav_velocity_rmse"]

            # KF 方法指标：UAV 使用窗口级后验/外推，浮标使用短步节点 KF 后验。
            kf_result = evaluate_dfpc(
                uav_est_kf,
                node_est_kf,
                p_u_true,
                buoy_true_short,
                phi_true,
                amp,
                eps,
                k_const,
                p_ideal,
                p_single_mean,
                p_single_best,
            )
            kf_result["uav_line_intercept_rmse"] = float(
                np.sqrt(np.mean(np.sum((kf_intercept_xy - true_line_params[[0, 2]]) ** 2, axis=1)))
            )
            kf_result["uav_velocity_rmse"] = float(
                np.sqrt(np.mean(np.sum((kf_velocity[:, :2] - true_line_params[[1, 3]]) ** 2, axis=1)))
            )

            # ---- 4) Cluster 环节。默认按通信图划分子图，每个节点只接受
            #      同一子图内、与其直接相连的邻居给出的相对定位结果；无邻居
            #      时保留自身 KF。子图定位结果随后进入 DPC/KF-DPC 相位补偿。
            #      --cluster_mode=localization/trajectory 保留旧兼容路径。
            if cfg.enable_cluster:
                if cfg.cluster_mode in {"subgraph", "localization"}:
                    assert cluster_localizer is not None
                    localize_nodes = (
                        cluster_localizer.update_subgraph
                        if cfg.cluster_mode == "subgraph"
                        else cluster_localizer.update
                    )
                    localization_options = {
                        "relative_noise_std": cfg.cluster_relative_noise_std,
                        "constraint_type": cfg.cluster_constraint_type,
                        "consensus_steps": cfg.cluster_localization_iterations,
                        "dt": cfg.Ts,
                    }
                    if cfg.cluster_mode == "localization":
                        localization_options["alpha"] = cfg.cluster_alpha
                    cluster_node_state = localize_nodes(
                        node_state_kf,
                        state.W_global,
                        cluster_labels,
                        buoy_true_short,
                        rng_cluster_measurement,
                        **localization_options,
                    )
                    valid_cluster_nodes = cluster_localizer.last_effective_peer_count > 0.0
                    if np.any(valid_cluster_nodes):
                        cluster_diagnostics["effective_peer_count_mean"][sidx] = float(
                            np.mean(cluster_localizer.last_effective_peer_count[valid_cluster_nodes])
                        )
                    cluster_diagnostics["nodes_with_subgraph_peers"][sidx] = int(
                        np.count_nonzero(valid_cluster_nodes)
                    )
                    cluster_node_est = cluster_node_state[:, :3]
                    cluster_uav_est_dfpc = uav_est_dfpc
                    cluster_dfpc_result = evaluate_dfpc(
                        cluster_uav_est_dfpc,
                        cluster_node_est,
                        p_u_true,
                        buoy_true_short,
                        phi_true,
                        amp,
                        eps,
                        k_const,
                        p_ideal,
                        p_single_mean,
                        p_single_best,
                    )
                    cluster_dfpc_result["uav_line_intercept_rmse"] = dfpc_result["uav_line_intercept_rmse"]
                    cluster_dfpc_result["uav_velocity_rmse"] = dfpc_result["uav_velocity_rmse"]

                    cluster_uav_est_kf = uav_est_kf
                    cluster_kf_velocity = kf_velocity
                    cluster_kf_result = evaluate_dfpc(
                        cluster_uav_est_kf,
                        cluster_node_est,
                        p_u_true,
                        buoy_true_short,
                        phi_true,
                        amp,
                        eps,
                        k_const,
                        p_ideal,
                        p_single_mean,
                        p_single_best,
                    )
                    cluster_kf_intercept_xy = (
                        cluster_uav_est_kf[:, :2] - time_s * cluster_kf_velocity[:, :2]
                    )
                    cluster_kf_result["uav_line_intercept_rmse"] = float(
                        np.sqrt(np.mean(np.sum((cluster_kf_intercept_xy - true_line_params[[0, 2]]) ** 2, axis=1)))
                    )
                    cluster_kf_result["uav_velocity_rmse"] = float(
                        np.sqrt(np.mean(np.sum((cluster_kf_velocity[:, :2] - true_line_params[[1, 3]]) ** 2, axis=1)))
                    )
                elif cfg.cluster_mode == "uav_prior":
                    cluster_node_est = node_est_kf
                    cluster_uav_est_dfpc = uav_est_dfpc
                    cluster_dfpc_result = dict(dfpc_result)
                    cluster_kf_result = evaluate_dfpc(
                        cluster_uav_est_kf,
                        node_est_kf,
                        p_u_true,
                        buoy_true_short,
                        phi_true,
                        amp,
                        eps,
                        k_const,
                        p_ideal,
                        p_single_mean,
                        p_single_best,
                    )
                    cluster_kf_intercept_xy = (
                        cluster_uav_est_kf[:, :2] - time_s * cluster_kf_velocity[:, :2]
                    )
                    cluster_kf_result["uav_line_intercept_rmse"] = float(
                        np.sqrt(np.mean(np.sum((cluster_kf_intercept_xy - true_line_params[[0, 2]]) ** 2, axis=1)))
                    )
                    cluster_kf_result["uav_velocity_rmse"] = float(
                        np.sqrt(np.mean(np.sum((cluster_kf_velocity[:, :2] - true_line_params[[1, 3]]) ** 2, axis=1)))
                    )
                elif cfg.cluster_mode == "node_prior":
                    assert node_kf_prior is not None
                    cluster_node_est = node_kf_prior.x[:, :3]
                    cluster_uav_est_dfpc = uav_est_dfpc
                    cluster_dfpc_result = evaluate_dfpc(
                        cluster_uav_est_dfpc,
                        cluster_node_est,
                        p_u_true,
                        buoy_true_short,
                        phi_true,
                        amp,
                        eps,
                        k_const,
                        p_ideal,
                        p_single_mean,
                        p_single_best,
                    )
                    cluster_dfpc_result["uav_line_intercept_rmse"] = dfpc_result["uav_line_intercept_rmse"]
                    cluster_dfpc_result["uav_velocity_rmse"] = dfpc_result["uav_velocity_rmse"]
                    cluster_uav_est_kf = uav_est_kf
                    cluster_kf_velocity = kf_velocity
                    cluster_kf_result = evaluate_dfpc(
                        cluster_uav_est_kf,
                        cluster_node_est,
                        p_u_true,
                        buoy_true_short,
                        phi_true,
                        amp,
                        eps,
                        k_const,
                        p_ideal,
                        p_single_mean,
                        p_single_best,
                    )
                    cluster_kf_intercept_xy = (
                        cluster_uav_est_kf[:, :2] - time_s * cluster_kf_velocity[:, :2]
                    )
                    cluster_kf_result["uav_line_intercept_rmse"] = float(
                        np.sqrt(np.mean(np.sum((cluster_kf_intercept_xy - true_line_params[[0, 2]]) ** 2, axis=1)))
                    )
                    cluster_kf_result["uav_velocity_rmse"] = float(
                        np.sqrt(np.mean(np.sum((cluster_kf_velocity[:, :2] - true_line_params[[1, 3]]) ** 2, axis=1)))
                    )
                else:
                    cluster_line_params = rotating_cluster_trajectory_correction(
                        line_params_dfpc,
                        cluster_labels,
                        rng_cluster_measurement,
                        cfg.cluster_alpha,
                        cfg.cluster_trajectory_noise_std,
                    )
                    cluster_line_params = normalize_trajectory_directions(cluster_line_params)
                    cluster_uav_est_dfpc = predict_positions(cluster_line_params, time_s, cfg.uav_height)
                    cluster_dfpc_result = evaluate_dfpc(
                        cluster_uav_est_dfpc,
                        buoy_obs_short,
                        p_u_true,
                        buoy_true_short,
                        phi_true,
                        amp,
                        eps,
                        k_const,
                        p_ideal,
                        p_single_mean,
                        p_single_best,
                    )
                    cluster_intercept_error = (
                        cluster_line_params[:, INTERCEPT_COLUMNS] - true_line_params[INTERCEPT_COLUMNS]
                    )
                    cluster_velocity_error = (
                        cluster_line_params[:, VELOCITY_COLUMNS] - true_line_params[VELOCITY_COLUMNS]
                    )
                    cluster_dfpc_result["uav_line_intercept_rmse"] = float(
                        np.sqrt(np.mean(np.sum(cluster_intercept_error**2, axis=1)))
                    )
                    cluster_dfpc_result["uav_velocity_rmse"] = float(
                        np.sqrt(np.mean(np.sum(cluster_velocity_error**2, axis=1)))
                    )

                    cluster_kf_line_params = trajectory_state_from_position_velocity(
                        uav_est_kf,
                        kf_velocity,
                        time_s,
                    )
                    cluster_kf_line_params = rotating_cluster_trajectory_correction(
                        cluster_kf_line_params,
                        cluster_labels,
                        rng_cluster_measurement,
                        cfg.cluster_alpha,
                        cfg.cluster_trajectory_noise_std,
                    )
                    cluster_kf_line_params = normalize_trajectory_directions(cluster_kf_line_params)
                    cluster_uav_est_kf = predict_positions(cluster_kf_line_params, time_s, cfg.uav_height)
                    cluster_kf_velocity = cluster_kf_line_params[:, VELOCITY_COLUMNS]
                    cluster_kf_result = evaluate_dfpc(
                        cluster_uav_est_kf,
                        node_est_kf,
                        p_u_true,
                        buoy_true_short,
                        phi_true,
                        amp,
                        eps,
                        k_const,
                        p_ideal,
                        p_single_mean,
                        p_single_best,
                    )
                    cluster_kf_intercept_xy = cluster_uav_est_kf[:, :2] - time_s * cluster_kf_velocity[:, :2]
                    cluster_kf_result["uav_line_intercept_rmse"] = float(
                        np.sqrt(np.mean(np.sum((cluster_kf_intercept_xy - true_line_params[[0, 2]]) ** 2, axis=1)))
                    )
                    cluster_kf_result["uav_velocity_rmse"] = float(
                        np.sqrt(np.mean(np.sum((cluster_kf_velocity[:, :2] - true_line_params[[1, 3]]) ** 2, axis=1)))
                    )

            results = {
                METHOD_RANDOM_REFERENCE: random_reference_result,
                METHOD_NO_ALG: no_alg_result,
                METHOD_DFPC: dfpc_result,
                METHOD_NODE_KF_DFPC: node_kf_dfpc_result,
                METHOD_KF_DFPC: kf_result,
            }
            if cfg.enable_cluster:
                results[METHOD_CLUSTER_DFPC] = cluster_dfpc_result
                results[METHOD_CLUSTER_KF_DFPC] = cluster_kf_result
            for method, result in results.items():
                for key, val in result.items():
                    metrics[method][key][sidx] = val

            if debug is not None:
                # 4. 如果打开 --debug，只保存指定 block/iteration 的细节，避免文件过大。
                debug.capture(
                    trial_index=trial_index,
                    seed=seed,
                    block=block,
                    iteration=iteration,
                    method=METHOD_DFPC,
                    p_u_true=p_u_true,
                    p_u_est=uav_est_dfpc,
                    node_true_xyz=buoy_true_short,
                    node_est_xyz=buoy_obs_short,
                    phi_true=phi_true,
                    amp=amp,
                    eps=eps,
                    k_const=k_const,
                    metric=dfpc_result,
                    line_params=line_params_dfpc,
                )
                debug.capture(
                    trial_index=trial_index,
                    seed=seed,
                    block=block,
                    iteration=iteration,
                    method=METHOD_KF_DFPC,
                    p_u_true=p_u_true,
                    p_u_est=uav_est_kf,
                    node_true_xyz=buoy_true_short,
                    node_est_xyz=node_est_kf,
                    phi_true=phi_true,
                    amp=amp,
                    eps=eps,
                    k_const=k_const,
                    metric=kf_result,
                    line_params=kf_line_params,
                )

    return {
        "metrics": metrics,
        "args": cfg.__dict__.copy(),
        "methods": methods,
        "frequency_MHz": float(cfg.fc_mhz),
        "wavelength_m": lam,
        "total_steps": total_steps,
        "trial_index": trial_index,
        "seed": seed,
        "physical_duration_s": (total_steps - 1) * float(cfg.Ts),
        "uav_kf_window_updates": uav_kf_window_updates,
        "uav_kf_window_size": int(cfg.K),
        "cluster_diagnostics": cluster_diagnostics,
        "graph_diagnostics": graph_diagnostics,
    }


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    """运行完整实验，包括 Monte Carlo 平均。"""
    validate_config(cfg)
    mc_trials = max(int(cfg.mc_trials), 1)
    device = backend.resolve_device(cfg)
    print(f"Global consensus device: {device}", flush=True)
    debug = DebugRecorder.from_config(cfg)
    methods = METHODS if cfg.enable_cluster else CORE_METHODS

    trials = []
    for trial in range(mc_trials):
        seed = int(cfg.seed) + trial * int(cfg.mc_seed_stride)
        print(f"MC trial {trial + 1}/{mc_trials} seed={seed}", flush=True)
        trials.append(run_single_trial(cfg, seed, trial, debug))

    total_steps = trials[0]["total_steps"]
    metrics = {
        method: {key: np.zeros(total_steps, dtype=np.float64) for key in METRIC_KEYS}
        for method in methods
    }
    cluster_diagnostics = {
        key: mean_stack([res["cluster_diagnostics"][key] for res in trials])
        for key in trials[0]["cluster_diagnostics"]
    }
    graph_diagnostics = {
        key: float(np.nanmean([res["graph_diagnostics"][key] for res in trials]))
        for key in trials[0]["graph_diagnostics"]
    }
    for method in methods:
        # 功率类指标必须先在线性域平均，再转成 dB。
        for key in [
            "power_linear",
            "ideal_power_linear",
            "single_node_mean_power_linear",
            "single_node_best_power_linear",
        ]:
            stack = np.stack([res["metrics"][method][key] for res in trials], axis=0)
            metrics[method][key] = np.mean(stack, axis=0)
        metrics[method]["gain_linear"] = (
            metrics[method]["power_linear"]
            / np.maximum(metrics[method]["ideal_power_linear"], 1e-30)
        )
        metrics[method]["norm_db"] = 10.0 * np.log10(np.maximum(metrics[method]["gain_linear"], 1e-30))
        metrics[method]["gain_over_single_mean_linear"] = (
            metrics[method]["power_linear"] / np.maximum(metrics[method]["single_node_mean_power_linear"], 1e-30)
        )
        metrics[method]["gain_over_single_mean_db"] = 10.0 * np.log10(
            np.maximum(metrics[method]["gain_over_single_mean_linear"], 1e-30)
        )
        metrics[method]["gain_over_single_best_linear"] = (
            metrics[method]["power_linear"] / np.maximum(metrics[method]["single_node_best_power_linear"], 1e-30)
        )
        metrics[method]["gain_over_single_best_db"] = 10.0 * np.log10(
            np.maximum(metrics[method]["gain_over_single_best_linear"], 1e-30)
        )
        for key in ERROR_KEYS:
            # 相位/RMSE 类指标直接对 trial 求平均。
            metrics[method][key] = mean_stack([res["metrics"][method][key] for res in trials])

    # 每个长 block 的最终迭代点，方便观察动态场景下每个 block 的收敛终值。
    iter_count = cfg.K
    block_final_rows = []
    for block in range(cfg.T_long):
        final_idx = (block + 1) * iter_count - 1
        for method in methods:
            block_final_rows.append(
                {
                    "long_block": block,
                    "physical_time_s": ((block + 1) * cfg.K - 1) * float(cfg.Ts),
                    "method": method,
                    "final_norm_power_db": float(metrics[method]["norm_db"][final_idx]),
                    "final_gain_over_single_mean_db": float(metrics[method]["gain_over_single_mean_db"][final_idx]),
                    "final_gain_over_single_best_db": float(metrics[method]["gain_over_single_best_db"][final_idx]),
                    "final_phase_std_deg": float(metrics[method]["phase_std_deg"][final_idx]),
                    "final_phase_rmse_deg": float(metrics[method]["phase_rmse_deg"][final_idx]),
                    "final_distance_rmse_m": ""
                    if np.isnan(metrics[method]["distance_rmse"][final_idx])
                    else float(metrics[method]["distance_rmse"][final_idx]),
                    "final_node_rmse_m": ""
                    if np.isnan(metrics[method]["node_rmse"][final_idx])
                    else float(metrics[method]["node_rmse"][final_idx]),
                    "final_uav_rmse_m": ""
                    if np.isnan(metrics[method]["uav_rmse"][final_idx])
                    else float(metrics[method]["uav_rmse"][final_idx]),
                    "final_uav_line_intercept_rmse_m": ""
                    if np.isnan(metrics[method]["uav_line_intercept_rmse"][final_idx])
                    else float(metrics[method]["uav_line_intercept_rmse"][final_idx]),
                    "final_uav_velocity_rmse_mps": ""
                    if np.isnan(metrics[method]["uav_velocity_rmse"][final_idx])
                    else float(metrics[method]["uav_velocity_rmse"][final_idx]),
                }
            )

    # 频率 sweep 调试所需的非聚合数据：保留每个 trial 和每个物理 block，
    # 避免只看 Monte Carlo 均值时无法判断回升来自哪个随机样本。
    trial_summary_rows: list[dict[str, Any]] = []
    trial_block_rows: list[dict[str, Any]] = []
    tail = slice(int(0.8 * total_steps), None)
    for trial_res in trials:
        trial_metrics = trial_res["metrics"]
        trial_index = int(trial_res["trial_index"])
        trial_seed = int(trial_res["seed"])
        for method in methods:
            method_metrics = trial_metrics[method]
            tail_gain = float(np.mean(method_metrics["gain_linear"][tail]))
            trial_summary_rows.append(
                {
                    "trial_index": trial_index,
                    "seed": trial_seed,
                    "method": method,
                    "tail_power_db": float(10.0 * np.log10(max(tail_gain, 1e-30))),
                    "final_power_db": float(method_metrics["norm_db"][-1]),
                    "tail_phase_std_deg": float(np.nanmean(method_metrics["phase_std_deg"][tail])),
                    "tail_phase_rmse_deg": float(np.nanmean(method_metrics["phase_rmse_deg"][tail])),
                    "tail_distance_rmse_m": ""
                    if np.all(np.isnan(method_metrics["distance_rmse"][tail]))
                    else float(np.nanmean(method_metrics["distance_rmse"][tail])),
                    "tail_node_rmse_m": ""
                    if np.all(np.isnan(method_metrics["node_rmse"][tail]))
                    else float(np.nanmean(method_metrics["node_rmse"][tail])),
                    "tail_uav_rmse_m": ""
                    if np.all(np.isnan(method_metrics["uav_rmse"][tail]))
                    else float(np.nanmean(method_metrics["uav_rmse"][tail])),
                }
            )
            for block in range(cfg.T_long):
                final_idx = (block + 1) * iter_count - 1
                trial_block_rows.append(
                    {
                        "trial_index": trial_index,
                        "seed": trial_seed,
                        "long_block": block,
                        "physical_time_s": ((block + 1) * cfg.K - 1) * float(cfg.Ts),
                        "method": method,
                        "block_final_power_db": float(method_metrics["norm_db"][final_idx]),
                        "block_final_gain_linear": float(method_metrics["gain_linear"][final_idx]),
                        "block_final_phase_std_deg": float(method_metrics["phase_std_deg"][final_idx]),
                        "block_final_phase_rmse_deg": float(method_metrics["phase_rmse_deg"][final_idx]),
                        "block_final_distance_rmse_m": ""
                        if np.isnan(method_metrics["distance_rmse"][final_idx])
                        else float(method_metrics["distance_rmse"][final_idx]),
                        "block_final_node_rmse_m": ""
                        if np.isnan(method_metrics["node_rmse"][final_idx])
                        else float(method_metrics["node_rmse"][final_idx]),
                        "block_final_uav_rmse_m": ""
                        if np.isnan(method_metrics["uav_rmse"][final_idx])
                        else float(method_metrics["uav_rmse"][final_idx]),
                    }
                )

    return {
        "metrics": metrics,
        "block_final_rows": block_final_rows,
        "args": cfg.__dict__.copy(),
        "methods": methods,
        "frequency_MHz": trials[0]["frequency_MHz"],
        "wavelength_m": trials[0]["wavelength_m"],
        "total_steps": total_steps,
        "mc_trials": mc_trials,
        "compute_device": device,
        "debug_recorder": debug,
        "physical_duration_s": trials[0]["physical_duration_s"],
        "uav_kf_window_updates": trials[0]["uav_kf_window_updates"],
        "uav_kf_window_size": trials[0]["uav_kf_window_size"],
        "trial_summary_rows": trial_summary_rows,
        "trial_block_rows": trial_block_rows,
        "cluster_diagnostics": cluster_diagnostics,
        "graph_diagnostics": graph_diagnostics,
    }

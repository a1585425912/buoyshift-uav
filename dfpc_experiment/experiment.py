"""实验主循环。

这个文件把各模块串起来：
1. 初始化场景；
2. 每个长块推进 K 个 Ts 短步并产生观测；
3. 短时间尺度：block 内 K 次 iteration，每个 iteration 先做 DPC 共识得到 UAV 估计，
   再把该估计作为观测喂给短时间尺度 KF（dt=Ts，状态 [x,y,z,vx,vy,vz]）更新位置；
4. 长时间尺度：block 结束时，用短尺度 KF 收敛后的位置作为观测，更新长时间尺度 KF
   （dt=Ts，每个 block 推进 K 次，物理间隔 K*Ts），使速度在块间逐步收敛；
   短尺度 KF 在每块开始时同步长尺度后验的速度；
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
    inject_position_state,
    make_cv3d_filter,
    predict_n_steps,
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
    init_line_estimator,
    normalize_trajectory_directions,
    predict_positions,
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
    line_estimator = init_line_estimator(cfg.N)
    uav_kf_long: BatchCVKalman3D | None = None
    uav_kf_short: BatchCVKalman3D | None = None
    uav_kf_prior_short: BatchCVKalman3D | None = None
    node_kf: BatchCVKalman3D | None = None
    node_kf_prior: BatchCVKalman3D | None = None
    cluster_localizer = (
        ClusterTrajectoryLocalizer(cfg.N)
        if cfg.enable_cluster and cfg.cluster_mode in {"subgraph", "localization", "node_prior"}
        else None
    )
    # UAV KF 的初始速度：用真实 UAV 速度 + 高斯噪声作为初值，对齐旧版双时间尺度
    # 中“以长时间尺度速度作为短时间尺度 KF 初值”的做法；噪声小以避免从零起步的瞬态。
    uav_kf_velocity_true_xyz = state.uav_velocity_true.copy()
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
        # The previous block ends at short step K-1. Advance once before the
        # next block so every recorded sample is exactly Ts apart and block
        # boundaries never duplicate the same physical instant.
        if block > 0:
            advance_truth_one_short_step(cfg, state, rng_scene, cfg.Ts)
        # 长时间尺度块开始：读取真实位置、生成一次观测、初始化/重置本块所需的 KF。
        time_s = block * block_duration_s
        p_u_true, buoy_true_short = current_truth(state)
        uav_obs_short, buoy_obs_short = observe_positions(cfg, p_u_true, buoy_true_short, rng_obs)
        # 每个节点用自身截至当前时刻的全部观测拟合 x=bx+vx*t, y=by+vy*t。
        line_params_dfpc = update_local_line_estimates(line_estimator, time_s, uav_obs_short)

        # 双时间尺度 UAV KF：
        #   - 长时间尺度 uav_kf_long：状态 [x,y,z,vx,vy,vz]，dt=Ts，
        #     每个长块推进 K 次，维持块间速度；物理推进量仍为 K*Ts。
        #     首次构造用“真实 UAV 速度 + 小噪声”作初值，避免从零起步的瞬态。
        #   - 短时间尺度 uav_kf_short：dt=Ts，velocity_propagate=True（block 内 UAV 随每步 Ts 推进）。
        #     每块开始时从 long KF 的共识后验（状态+协方差）重置。
        if uav_kf_long is None:
            uav_kf_long = make_cv3d_filter(
                uav_obs_short,
                cfg.Ts,
                cfg.uav_obs_noise,
                cfg.uav_kf_accel_std,
                cfg.uav_kf_initial_velocity_std,
                initial_velocity_xyz=uav_kf_velocity_true_xyz
                + rng_obs.normal(0.0, cfg.uav_kf_velocity_init_std, size=3),
            )
            uav_kf_short = make_cv3d_filter(
                uav_obs_short,
                cfg.Ts,
                cfg.uav_obs_noise,
                cfg.uav_kf_accel_std,
                cfg.uav_kf_initial_velocity_std,
                initial_velocity_xyz=uav_kf_velocity_true_xyz
                + rng_obs.normal(0.0, cfg.uav_kf_velocity_init_std, size=3),
                velocity_propagate=True,
            )
            if cfg.enable_cluster and cfg.cluster_mode == "uav_prior":
                uav_kf_prior_short = make_cv3d_filter(
                    uav_obs_short,
                    cfg.Ts,
                    cfg.uav_obs_noise,
                    cfg.uav_kf_accel_std,
                    cfg.uav_kf_initial_velocity_std,
                    initial_velocity_xyz=uav_kf_velocity_true_xyz
                    + rng_obs.normal(0.0, cfg.uav_kf_velocity_init_std, size=3),
                    velocity_propagate=True,
                )
        else:
            # Propagate the long-timescale filter over exactly the K short
            # physical steps that elapsed since its previous update.  The
            # truth advances K*Ts per block, not TL, so a block-sized predict
            # with dt=K*Ts also fits the physical time axis.
            predict_n_steps(uav_kf_long, cfg.K)
        # 短时间尺度 UAV KF 从长尺度后验同步速度均值。
        # The short-timescale KF remains continuous across block boundaries,
        # so its position/velocity covariance is not reset here; overwriting
        # the covariance with the long-timescale P kept the velocity estimate
        # pinned near its initialization uncertainty.
        uav_kf_short.x[:, 3:] = uav_kf_long.x[:, 3:].copy()
        if uav_kf_prior_short is not None:
            uav_kf_prior_short.x[:, 3:] = uav_kf_long.x[:, 3:].copy()

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

            # 解除“block 内真值固定”：除 iteration 0 外，每个短步都推进一个 Ts 后重新观测。
            # ---- 1) 短时间尺度 DPC：对三维轨迹、三轴速度和单位航向做一次邻居共识。
            cluster_uav_prior_est = None
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
                cluster_uav_prior_est = predict_positions(
                    cluster_prior_line_params,
                    time_s,
                    cfg.uav_height,
                )
            line_params_dfpc = backend.consensus_linear_accel(
                state.W_global,
                state.W_global_gpu,
                line_params_dfpc,
                1,
            )
            line_params_dfpc = normalize_trajectory_directions(line_params_dfpc)
            uav_est_dfpc = predict_positions(line_params_dfpc, time_s, cfg.uav_height)

            # ---- 2) 短时间尺度 UAV KF（DPC 作先验、原始短观测 z_u 作似然）：
            #        predict（dt=Ts，速度积分进位置）→ 注入 DPC 估计作位置先验 →
            #        update 用一份新的原始短观测 z_u_short_kf → consensus。
            # The first short-time sample is already the filter's initial
            # state; predicting before it would map the t=0 measurement onto
            # t=Ts.  Every later step advances exactly one Ts.
            if sidx > 0:
                uav_kf_short.predict()
            # Legacy path overwrites KF state with DPC then updates raw
            # observations, which pulls the good DPC estimate back toward
            # single-node noise.  The dpc_only branch instead treats DPC as a
            # normal position pseudo-measurement so the KF fuses it properly.
            if cfg.uav_kf_prior_mode == "none":
                inject_position_state(uav_kf_short, uav_est_dfpc, cfg.uav_dpc_prior_noise_std)
            else:
                uav_kf_short.update_position_measurement(
                    uav_est_dfpc,
                    cfg.uav_dpc_prior_noise_std,
                )
                if (
                    cfg.uav_kf_prior_mode == "dpc_plus_cluster"
                    and cluster_uav_prior_est is not None
                ):
                    uav_kf_short.update_position_measurement(
                        cluster_uav_prior_est,
                        cfg.cluster_uav_prior_noise_std,
                    )
            if cluster_uav_prior_est is not None and uav_kf_prior_short is not None:
                if sidx > 0:
                    uav_kf_prior_short.predict()
                inject_position_state(
                    uav_kf_prior_short, uav_est_dfpc, cfg.uav_dpc_prior_noise_std
                )
                uav_kf_prior_short.update_position_measurement(
                    cluster_uav_prior_est,
                    cfg.cluster_uav_prior_noise_std,
                )
            z_u_short_kf = p_u_true + rng_obs.normal(0.0, cfg.uav_obs_noise, size=(cfg.N, 3))
            if sidx > 0:
                uav_kf_short.update(z_u_short_kf)
                if uav_kf_prior_short is not None:
                    uav_kf_prior_short.update(z_u_short_kf)
            uav_state_short_kf = uav_kf_short.x.copy()
            uav_state_short_kf = backend.consensus_linear_accel(
                state.W_global, state.W_global_gpu, uav_state_short_kf, 1
            )
            uav_kf_short.x = uav_state_short_kf.copy()
            uav_kf_short.P = consensus_covariance(state.W_global, uav_kf_short.P)
            uav_est_kf = uav_state_short_kf[:, :3]
            kf_velocity = uav_state_short_kf[:, 3:]
            cluster_uav_est_kf = uav_est_kf
            cluster_kf_velocity = kf_velocity
            if uav_kf_prior_short is not None:
                uav_prior_state = uav_kf_prior_short.x.copy()
                uav_prior_state = backend.consensus_linear_accel(
                    state.W_global,
                    state.W_global_gpu,
                    uav_prior_state,
                    1,
                )
                uav_kf_prior_short.x = uav_prior_state.copy()
                uav_kf_prior_short.P = consensus_covariance(
                    state.W_global, uav_kf_prior_short.P
                )
                cluster_uav_est_kf = uav_prior_state[:, :3]
                cluster_kf_velocity = uav_prior_state[:, 3:]

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

            # 短时间尺度 KF 方法的指标：位置用短尺度 UAV/节点 KF 后验。
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

        # ---- 长时间尺度 KF 更新：用本块短尺度 KF 收敛后的位置作为观测，更新 long KF
        #      的 [x,y,z,vx,vy,vz]，使速度在块间逐步收敛；再做一次全局共识。
        if block == 0:
            # Block 0 starts at t=0; its first long-timescale observation is
            # the short-scale estimate at t=(K-1)*Ts, so propagate K-1 steps
            # before the update instead of applying a stale initial state.
            predict_n_steps(uav_kf_long, cfg.K - 1)
        uav_kf_long.update(uav_est_kf)
        uav_kf_long.x = backend.consensus_linear_accel(
            state.W_global,
            state.W_global_gpu,
            uav_kf_long.x,
            1,
        ).copy()
        uav_kf_long.P = consensus_covariance(state.W_global, uav_kf_long.P)

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
        "trial_summary_rows": trial_summary_rows,
        "trial_block_rows": trial_block_rows,
        "cluster_diagnostics": cluster_diagnostics,
        "graph_diagnostics": graph_diagnostics,
    }

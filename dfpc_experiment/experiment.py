"""实验主循环。

这个文件把各模块串起来：
1. 初始化场景；
2. 每个长块推进 K 个迭代步，相邻迭代间隔为 Ts，并产生观测；
3. 每个节点用连续观测拟合 UAV 轨迹 [位置,速度] 及协方差；
4. DPC 对完整轨迹参数 [截距,速度] 做通信更新，KF-DPC 先本地滤波再做轨迹共识；
5. 多个 Monte Carlo trial 做平均。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import backend
from .config import ExperimentConfig, validate_config
from .constants import (
    ERROR_KEYS,
    METHOD_DFPC,
    METHOD_KF_DFPC,
    METHOD_NODE_KF_DPC,
    METHOD_NO_ALG,
    METHOD_SHORE_BROADCAST,
    METHODS,
    METRIC_KEYS,
)
from .debug_tools import DebugRecorder
from .dpc import (
    DistanceRMSEConvergenceTracker,
    consensus_trajectory_state,
    conservative_consensus_covariance,
    dpc_state_update,
    kf_dpc_state_update,
)
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
    init_line_estimator,
    predict_trajectory,
    update_local_line_estimates,
)


def consensus_covariance(W: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Backward-compatible export of conservative posterior covariance fusion."""
    return conservative_consensus_covariance(W, P)


def dpc_trajectory_prediction(
    local_prediction: TrajectoryPrediction,
    time_s: float,
    weight_matrix: np.ndarray,
    weight_matrix_gpu: object,
    position_model_std: float,
    velocity_model_std: float,
) -> TrajectoryPrediction:
    """Reach consensus on full UAV trajectories, then predict the current state."""
    return dpc_state_update(
        local_prediction,
        time_s,
        weight_matrix,
        weight_matrix_gpu,
        position_model_std,
        velocity_model_std,
    )


def initialize_uav_filter(
    trajectory: TrajectoryPrediction,
    cfg: ExperimentConfig,
    update_interval_s: float,
    time_s: float,
    weight_matrix: np.ndarray,
    weight_matrix_gpu: object,
) -> BatchCVKalman3D:
    """Initialize local UAV posteriors, then perform the first W update."""
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
    # At step 0 only position is observed.  Velocity is an unobserved prior and
    # must keep its configured uncertainty instead of inheriting an artificial
    # reduction from spatial consensus of identical zero-velocity placeholders.
    velocity_var = max(float(cfg.uav_kf_initial_velocity_std), 1e-9) ** 2
    kf.P[:, :3, 3:] = 0.0
    kf.P[:, 3:, :3] = 0.0
    kf.P[:, 3:, 3:] = velocity_var * np.eye(3)[None, :, :]

    kf.x, _ = consensus_trajectory_state(
        kf.x,
        time_s,
        weight_matrix,
        weight_matrix_gpu,
    )
    kf.P = consensus_covariance(weight_matrix, kf.P)

    return kf


def update_uav_filter(
    kf: BatchCVKalman3D,
    local_trajectory: TrajectoryPrediction,
    time_s: float,
    weight_matrix: np.ndarray,
    weight_matrix_gpu: object,
) -> None:
    """Advance from k-1, update locally at k, then fuse posterior states."""
    kf_dpc_state_update(kf, local_trajectory, time_s, weight_matrix, weight_matrix_gpu)


def run_single_trial(
    cfg: ExperimentConfig,
    seed: int,
    trial_index: int = 0,
    debug: DebugRecorder | None = None,
) -> dict[str, Any]:
    """运行一次 Monte Carlo trial。

    注意：iteration 是算法迭代编号，相邻两次迭代对应 Ts 秒的物理时间。
    相邻 block 之间物理世界严格推进 K*Ts 秒。
    """
    rng_scene = np.random.default_rng(seed)
    rng_obs = np.random.default_rng(seed + 3000003)
    rng_broadcast = np.random.default_rng(seed + 6000007)

    fc_hz = float(cfg.fc_mhz) * 1e6
    lam = 3e8 / fc_hz
    k_const = 2.0 * np.pi / lam
    iter_count = cfg.K
    total_steps = cfg.T_long * iter_count
    block_duration_s = float(cfg.K) * float(cfg.Ts)

    state = init_scene(cfg, rng_scene)
    methods = [*METHODS, METHOD_SHORE_BROADCAST] if cfg.shore_broadcast_enabled else list(METHODS)
    adjacency = state.W_global > 1e-12
    np.fill_diagonal(adjacency, False)
    graph_diagnostics = {
        "mean_degree": float(np.mean(np.sum(adjacency, axis=1))),
    }
    # System/calibration phase error is persistent over one trial.  Drawing a
    # new independent value at every short step creates non-physical power
    # zigzags even when the geometry moves smoothly.
    eps_trial = rng_scene.normal(0.0, np.deg2rad(cfg.system_phase_std_deg), size=cfg.N)
    uav_kf: BatchCVKalman3D | None = None
    uav_kf_trajectory_initializations = 0
    uav_kf_position_updates = 0
    # UAV trajectory statistics retain the complete history.  K belongs to the
    # buoy short-time motion/output block and does not reset or gate the UAV KF.
    line_estimator = init_line_estimator(cfg.N)
    dpc_convergence = DistanceRMSEConvergenceTracker(
        cfg.dpc_distance_rmse_delta_tol_m,
        cfg.dpc_consensus_hold_steps,
    )
    kf_dpc_convergence = DistanceRMSEConvergenceTracker(
        cfg.dpc_distance_rmse_delta_tol_m,
        cfg.dpc_consensus_hold_steps,
    )
    node_kf_dpc_convergence = DistanceRMSEConvergenceTracker(
        cfg.dpc_distance_rmse_delta_tol_m,
        cfg.dpc_consensus_hold_steps,
    )
    node_kf: BatchCVKalman3D | None = None

    # Optional shared pre-roll.  Its samples live at negative time and are not
    # written to metrics, so the first plotted sample remains formal time k=0.
    for warmup_index in range(cfg.warmup_steps):
        warmup_time_s = (warmup_index - cfg.warmup_steps) * float(cfg.Ts)
        p_u_true, buoy_true_short = current_truth(state)
        uav_obs_short, buoy_obs_short = observe_positions(cfg, p_u_true, buoy_true_short, rng_obs)
        local_line_params = update_local_line_estimates(line_estimator, warmup_time_s, uav_obs_short)
        local_trajectory_prediction = predict_trajectory(
            local_line_params,
            line_estimator,
            warmup_time_s,
            cfg.uav_obs_noise,
            cfg.uav_kf_initial_velocity_std,
        )
        if uav_kf is None:
            uav_kf = initialize_uav_filter(
                local_trajectory_prediction,
                cfg,
                cfg.Ts,
                warmup_time_s,
                state.W_global,
                state.W_global_gpu,
            )
            uav_kf_trajectory_initializations += 1
        else:
            update_uav_filter(
                uav_kf,
                local_trajectory_prediction,
                warmup_time_s,
                state.W_global,
                state.W_global_gpu,
            )
            uav_kf_position_updates += 1

        if node_kf is None:
            node_kf = make_cv3d_filter(
                buoy_obs_short,
                cfg.Ts,
                cfg.buoy_center_obs_noise,
                cfg.buoy_kf_accel_std,
                cfg.buoy_kf_initial_velocity_std,
                initial_velocity_xyz=state.buoy_wave_mean_velocity,
                position_diffusion=(
                    cfg.buoy_center_accumulation_ratio
                    * cfg.buoy_random_displacement_std
                    / np.sqrt(cfg.Ts)
                ),
                velocity_propagate=True,
            )
        else:
            node_kf.predict()
            node_kf.update(buoy_obs_short)
        advance_truth_one_short_step(cfg, state, rng_scene, cfg.Ts)

    # The trajectory intercept is defined at formal time zero, after pre-roll.
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
    metrics = {
        method: {key: np.zeros(total_steps, dtype=np.float64) for key in METRIC_KEYS}
        for method in methods
    }
    consensus_diagnostics = {
        "dpc_distance_rmse_delta_m": np.full(total_steps, np.nan, dtype=np.float64),
        "dpc_phase_ready": np.zeros(total_steps, dtype=bool),
        "kf_dpc_distance_rmse_delta_m": np.full(total_steps, np.nan, dtype=np.float64),
        "kf_dpc_phase_ready": np.zeros(total_steps, dtype=bool),
        "node_kf_dpc_distance_rmse_delta_m": np.full(total_steps, np.nan, dtype=np.float64),
        "node_kf_dpc_phase_ready": np.zeros(total_steps, dtype=bool),
    }
    for key in ["distance_rmse", "node_rmse", "uav_rmse", "uav_line_intercept_rmse", "uav_velocity_rmse"]:
        metrics[METHOD_NO_ALG][key].fill(np.nan)

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
        # 每个节点持续累积历史观测，拟合三维匀速轨迹；长块边界不清空。
        local_line_params = update_local_line_estimates(line_estimator, time_s, uav_obs_short)
        local_trajectory_prediction = predict_trajectory(
            local_line_params,
            line_estimator,
            time_s,
            cfg.uav_obs_noise,
            cfg.uav_kf_initial_velocity_std,
        )

        # 节点（浮标）持久 KF：dt=Ts，初始速度取洋流平均速度；它与 UAV-KF
        # 一样跨所有物理时刻递推，不存在独立的“短尺度 KF”或块边界重置。
        if node_kf is None:
            node_kf = make_cv3d_filter(
                buoy_obs_short,
                cfg.Ts,
                cfg.buoy_center_obs_noise,
                cfg.buoy_kf_accel_std,
                cfg.buoy_kf_initial_velocity_std,
                initial_velocity_xyz=state.buoy_wave_mean_velocity,
                position_diffusion=(
                    cfg.buoy_center_accumulation_ratio
                    * cfg.buoy_random_displacement_std
                    / np.sqrt(cfg.Ts)
                ),
                velocity_propagate=True,
            )
            # 浮标属于不同物理目标：每个浮标只维护自己的 KF 后验，
            # 不对不同浮标的绝对位置做共识平均。
            node_state_kf = node_kf.x.copy()

        for iteration in range(iter_count):
            sidx = block * iter_count + iteration
            if iteration > 0:
                advance_truth_one_short_step(cfg, state, rng_scene, cfg.Ts)
                time_s = (block * cfg.K + iteration) * float(cfg.Ts)
                p_u_true, buoy_true_short = current_truth(state)
                uav_obs_short, buoy_obs_short = observe_positions(cfg, p_u_true, buoy_true_short, rng_obs)
                local_line_params = update_local_line_estimates(line_estimator, time_s, uav_obs_short)
                local_trajectory_prediction = predict_trajectory(
                    local_line_params,
                    line_estimator,
                    time_s,
                    cfg.uav_obs_noise,
                    cfg.uav_kf_initial_velocity_std,
                )

            # 解除“block 内真值固定”：除 iteration 0 外，每次迭代都推进 Ts 秒后重新观测。
            # ---- 1) DPC：各节点先从连续观测估计 [轨迹截距,速度]，
            #          再用 W 更新整条轨迹，最后预测当前位置。
            uav_trajectory_dfpc = dpc_trajectory_prediction(
                local_trajectory_prediction,
                time_s,
                state.W_global,
                state.W_global_gpu,
                cfg.uav_dpc_prior_noise_std,
                cfg.uav_kf_accel_std * cfg.Ts,
            )
            uav_est_dfpc = uav_trajectory_dfpc.positions
            if uav_trajectory_dfpc.trajectory_parameters is None:
                raise RuntimeError("DPC trajectory consensus did not return parameters")
            line_params_dfpc = uav_trajectory_dfpc.trajectory_parameters
            # ---- 2) 持久 UAV KF：k=0 初始化；k>=1 由 k-1 后验预测，
            #          使用本地轨迹观测更新，最后才通过 W 融合 UAV 后验。
            if uav_kf is None:
                uav_kf = initialize_uav_filter(
                    local_trajectory_prediction,
                    cfg,
                    cfg.Ts,
                    time_s,
                    state.W_global,
                    state.W_global_gpu,
                )
                uav_kf_trajectory_initializations += 1
            elif uav_kf is not None:
                update_uav_filter(
                    uav_kf,
                    local_trajectory_prediction,
                    time_s,
                    state.W_global,
                    state.W_global_gpu,
                )
                uav_kf_position_updates += 1

            uav_state = uav_trajectory_dfpc.state if uav_kf is None else uav_kf.x
            uav_est_kf = uav_state[:, :3]
            kf_velocity = uav_state[:, 3:]
            # ---- 3) 持久节点 KF：predict（dt=Ts）+ 用本轮同一份节点观测 z_b 更新。
            #     浮标不做跨节点共识，每个节点只使用自己的 KF 状态。
            if sidx > 0 or cfg.warmup_steps > 0:
                node_kf.predict()
            z_b_short_kf = buoy_obs_short
            if sidx > 0 or cfg.warmup_steps > 0:
                node_kf.update(z_b_short_kf)
            node_state_kf = node_kf.x.copy()
            node_est_kf = node_state_kf[:, :3]

            # 2. 当前真值下的几何量、相位噪声、无算法基线（每次迭代真值都变，必须重算）。
            phi_true, amp, p_ideal, p_single_mean, p_single_best = geometry_terms(
                cfg, p_u_true, buoy_true_short, k_const
            )
            eps = eps_trial
            # 无算法组：保留纯随机相位发射，不使用位置观测、共识或滤波。
            no_alg_result = random_phase_reference_metrics(
                phi_true,
                amp,
                eps,
                p_ideal,
                p_single_mean,
                p_single_best,
            )

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

            # 仅节点 KF：UAV 估计与普通 DPC 完全相同，只替换浮标位置后验。
            node_kf_result = evaluate_dfpc(
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

            broadcast_result = None
            if cfg.shore_broadcast_enabled:
                broadcast_uav_fix = p_u_true + rng_broadcast.normal(
                    0.0, cfg.shore_broadcast_uav_noise, size=3
                )
                broadcast_uav_est = np.broadcast_to(broadcast_uav_fix, (cfg.N, 3))
                broadcast_node_est = buoy_true_short + rng_broadcast.normal(
                    0.0, cfg.shore_broadcast_node_noise, size=(cfg.N, 3)
                )
                broadcast_result = evaluate_dfpc(
                    broadcast_uav_est,
                    broadcast_node_est,
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
                broadcast_result["uav_line_intercept_rmse"] = np.nan
                broadcast_result["uav_velocity_rmse"] = np.nan
            node_kf_result["uav_line_intercept_rmse"] = dfpc_result[
                "uav_line_intercept_rmse"
            ]
            node_kf_result["uav_velocity_rmse"] = dfpc_result["uav_velocity_rmse"]

            # KF 方法指标：UAV 使用窗口级后验/外推，浮标使用逐迭代节点 KF 后验。
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

            # 4. 收敛判据：相邻物理时刻的全节点距离误差 RMSE 变化量。
            # 这里使用仿真真值计算 distance_rmse，因此该判据用于实验评估；真实系统中
            # 需要用可观测的距离残差或创新统计量替代真值误差。
            dpc_rmse_delta, dpc_ready = dpc_convergence.update(dfpc_result["distance_rmse"])
            kf_dpc_rmse_delta, kf_dpc_ready = kf_dpc_convergence.update(
                kf_result["distance_rmse"]
            )
            node_kf_dpc_rmse_delta, node_kf_dpc_ready = node_kf_dpc_convergence.update(
                node_kf_result["distance_rmse"]
            )
            consensus_diagnostics["dpc_distance_rmse_delta_m"][sidx] = dpc_rmse_delta
            consensus_diagnostics["dpc_phase_ready"][sidx] = dpc_ready
            consensus_diagnostics["kf_dpc_distance_rmse_delta_m"][sidx] = kf_dpc_rmse_delta
            consensus_diagnostics["kf_dpc_phase_ready"][sidx] = kf_dpc_ready
            consensus_diagnostics["node_kf_dpc_distance_rmse_delta_m"][sidx] = (
                node_kf_dpc_rmse_delta
            )
            consensus_diagnostics["node_kf_dpc_phase_ready"][sidx] = node_kf_dpc_ready

            results = {
                METHOD_NO_ALG: no_alg_result,
                METHOD_DFPC: dfpc_result,
                METHOD_NODE_KF_DPC: node_kf_result,
                METHOD_KF_DFPC: kf_result,
            }
            if broadcast_result is not None:
                results[METHOD_SHORE_BROADCAST] = broadcast_result
            for method, result in results.items():
                for key, val in result.items():
                    metrics[method][key][sidx] = val

            if debug is not None:
                # 5. 如果打开 --debug，只保存指定 block/iteration 的细节，避免文件过大。
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
                    method=METHOD_NODE_KF_DPC,
                    p_u_true=p_u_true,
                    p_u_est=uav_est_dfpc,
                    node_true_xyz=buoy_true_short,
                    node_est_xyz=node_est_kf,
                    phi_true=phi_true,
                    amp=amp,
                    eps=eps,
                    k_const=k_const,
                    metric=node_kf_result,
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
        "uav_kf_window_updates": uav_kf_trajectory_initializations,
        "uav_kf_trajectory_initializations": uav_kf_trajectory_initializations,
        "uav_kf_position_updates": uav_kf_position_updates,
        "uav_kf_initialization_step": -int(cfg.warmup_steps),
        "uav_kf_window_size": 1,
        "consensus_diagnostics": consensus_diagnostics,
        "graph_diagnostics": graph_diagnostics,
    }


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    """运行完整实验，包括 Monte Carlo 平均。"""
    validate_config(cfg)
    mc_trials = max(int(cfg.mc_trials), 1)
    device = backend.resolve_device(cfg)
    print(f"Global consensus device: {device}", flush=True)
    debug = DebugRecorder.from_config(cfg)
    trials = []
    for trial in range(mc_trials):
        seed = int(cfg.seed) + trial * int(cfg.mc_seed_stride)
        print(f"MC trial {trial + 1}/{mc_trials} seed={seed}", flush=True)
        trials.append(run_single_trial(cfg, seed, trial, debug))

    methods = trials[0]["methods"]
    total_steps = trials[0]["total_steps"]
    metrics = {
        method: {key: np.zeros(total_steps, dtype=np.float64) for key in METRIC_KEYS}
        for method in methods
    }
    consensus_diagnostics = {
        key: mean_stack([res["consensus_diagnostics"][key] for res in trials])
        for key in trials[0]["consensus_diagnostics"]
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
        "uav_kf_trajectory_initializations": trials[0]["uav_kf_trajectory_initializations"],
        "uav_kf_position_updates": trials[0]["uav_kf_position_updates"],
        "uav_kf_initialization_step": trials[0]["uav_kf_initialization_step"],
        "uav_kf_window_size": trials[0]["uav_kf_window_size"],
        "trial_summary_rows": trial_summary_rows,
        "trial_block_rows": trial_block_rows,
        "consensus_diagnostics": consensus_diagnostics,
        "graph_diagnostics": graph_diagnostics,
    }

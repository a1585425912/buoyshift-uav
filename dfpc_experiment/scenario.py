"""场景生成、运动模型和观测模型。

这个文件对应实验中的“物理场景”部分：
1. 节点部署；
2. UAV 确定性运动；
3. 节点短时间尺度随机运动和可选洋流漂移；
4. 节点对 UAV / 自身位置的带噪声观测。

如果导师要求修改浮标运动模型或定位误差模型，优先改这里。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import backend
from . import sim_core as sim
from .config import ExperimentConfig


@dataclass
class SceneState:
    """一个 Monte Carlo trial 内随时间演化的真实状态。"""

    # 全局通信图的 Metropolis-Hastings 共识矩阵。
    W_global: np.ndarray
    W_global_gpu: object

    # 节点真实位置 = 随洋流及累积小位移移动的中心 + 当前剩余随机偏移。
    buoy_center_true: np.ndarray
    buoy_offset_true: np.ndarray

    # UAV 当前真实位置及固定速度，构成参数化匀速直线轨迹。
    uav_center_true: np.ndarray
    uav_velocity_true: np.ndarray

    # 固定洋流速度向量，单位 m/s。
    buoy_wave_mean_velocity: np.ndarray


def init_scene(cfg: ExperimentConfig, rng_scene: np.random.Generator) -> SceneState:
    """初始化一个 trial 的通信图、节点初始位置和 UAV 初始轨迹。"""
    W_global = sim.metropolis_hastings_W(cfg.N, cfg.global_connectivity, rng_scene)
    W_global_gpu = backend.make_gpu_consensus_matrix(W_global, cfg)
    buoy_xy = sim.deploy_buoys_in_disk(cfg.N, cfg.area_radius, rng_scene)
    buoy_center_true = np.column_stack([buoy_xy, np.zeros(cfg.N, dtype=np.float64)])
    buoy_offset_xy = backend.gaussian_position_offsets(
        cfg.N, cfg.buoy_random_displacement_std, rng_scene
    )
    buoy_offset_true = np.column_stack(
        [buoy_offset_xy, np.zeros(cfg.N, dtype=np.float64)]
    )

    heading = np.deg2rad(cfg.heading_deg)
    uav_start_x = -0.65 * cfg.area_radius if cfg.uav_start_x is None else float(cfg.uav_start_x)
    uav_start_y = -0.35 * cfg.area_radius if cfg.uav_start_y is None else float(cfg.uav_start_y)
    uav_center_true = np.array([uav_start_x, uav_start_y, cfg.uav_height], dtype=np.float64)
    uav_vel_true = np.array(
        [cfg.uav_speed * np.cos(heading), cfg.uav_speed * np.sin(heading), 0.0],
        dtype=np.float64,
    )
    return SceneState(
        W_global=W_global,
        W_global_gpu=W_global_gpu,
        buoy_center_true=buoy_center_true,
        buoy_offset_true=buoy_offset_true,
        uav_center_true=uav_center_true,
        uav_velocity_true=uav_vel_true,
        buoy_wave_mean_velocity=float(cfg.buoy_wave_speed)
        * np.array(
            [np.cos(np.deg2rad(cfg.buoy_wave_heading_deg)), np.sin(np.deg2rad(cfg.buoy_wave_heading_deg)), 0.0]
        ),
    )


def advance_truth_one_short_step(
    cfg: ExperimentConfig,
    state: SceneState,
    rng_scene: np.random.Generator,
    dt: float | None = None,
) -> None:
    """推进一个短时间尺度真实状态。

    UAV 按确定轨迹匀速运动。浮标中心按固定洋流速度和方向平移；
    每个时刻独立采样一次零均值高斯小位移，其中固定比例累积到中心，
    剩余部分作为当前时刻相对中心的偏移。
    """
    step_dt = float(cfg.Ts if dt is None else dt)
    if step_dt <= 0.0:
        raise ValueError("physical time step must be positive")

    state.uav_center_true = state.uav_center_true + state.uav_velocity_true * step_dt
    displacement_xy = backend.gaussian_position_offsets(
        cfg.N, cfg.buoy_random_displacement_std, rng_scene
    )
    displacement = np.column_stack(
        [displacement_xy, np.zeros(cfg.N, dtype=np.float64)]
    )
    accumulation_ratio = float(cfg.buoy_center_accumulation_ratio)
    state.buoy_center_true = (
        state.buoy_center_true
        + state.buoy_wave_mean_velocity * step_dt
        + accumulation_ratio * displacement
    )
    state.buoy_offset_true = np.column_stack(
        [
            (1.0 - accumulation_ratio) * displacement_xy,
            np.zeros(cfg.N, dtype=np.float64),
        ]
    )


def advance_truth_one_long_block(
    cfg: ExperimentConfig,
    state: SceneState,
    rng_scene: np.random.Generator,
) -> None:
    """推进一个长块，即 K 个持续 Ts 的物理短时间步。"""
    duration = float(cfg.K) * float(cfg.Ts)
    short_dt = float(cfg.Ts)
    if duration < 0.0 or short_dt <= 0.0:
        raise ValueError("K must be non-negative and Ts must be positive")
    full_steps = int(np.floor(duration / short_dt + 1e-12))
    for _ in range(full_steps):
        advance_truth_one_short_step(cfg, state, rng_scene, short_dt)
    remainder = duration - full_steps * short_dt
    if remainder > 1e-12:
        advance_truth_one_short_step(cfg, state, rng_scene, remainder)


def current_truth(state: SceneState) -> tuple[np.ndarray, np.ndarray]:
    """返回当前 UAV 真实位置和节点真实位置。"""
    buoy_true_short = state.buoy_center_true + state.buoy_offset_true
    p_u_true = state.uav_center_true.copy()
    return p_u_true, buoy_true_short


def observe_positions(
    cfg: ExperimentConfig,
    p_u_true: np.ndarray,
    buoy_true_short: np.ndarray,
    rng_obs: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """生成观测/估计位置。

    目前仍是“真实位置 + 高斯误差”的简化模型。后续如果要接入
    文献中的导频定位误差模型或测距模型，可以替换这个函数。
    """
    uav_obs_short = p_u_true + rng_obs.normal(0.0, cfg.uav_obs_noise, size=(cfg.N, 3))
    buoy_obs_short = buoy_true_short + rng_obs.normal(0.0, cfg.buoy_center_obs_noise, size=(cfg.N, 3))
    return uav_obs_short, buoy_obs_short


def geometry_terms(
    cfg: ExperimentConfig,
    p_u_true: np.ndarray,
    buoy_true_short: np.ndarray,
    k_const: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """计算相位和功率归一化所需的几何量。

    返回：
    - phi_true: 每个节点到 UAV 的真实传播相位 k*d；
    - amp: 每个节点到 UAV 的路径损耗幅度；
    - p_ideal: 所有节点完全相干时的理想功率；
    - p_single_mean / p_single_best: 单节点基准功率，用于节点数实验。
    """
    d_true = np.linalg.norm(p_u_true[None, :] - buoy_true_short, axis=1)
    phi_true = k_const * d_true
    amp = np.sqrt(cfg.tx_power) / np.maximum(d_true, 1e-6) ** cfg.path_loss_alpha
    p_ideal = float(np.sum(amp) ** 2)
    p_single = amp**2
    return phi_true, amp, p_ideal, float(np.mean(p_single)), float(np.max(p_single))

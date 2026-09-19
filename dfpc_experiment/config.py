"""模块化 DPC 实验的参数配置与命令行解析。

这个文件是最适合你改实验参数的地方。其它模块尽量只读取
`ExperimentConfig`，不直接写死参数，这样后续改频率、节点数、
误差模型、调试开关时不用到处翻代码。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PACKAGE_DIR.parent


@dataclass
class ExperimentConfig:
    # -------------------------
    # 基本场景参数
    # -------------------------
    # N: 节点/浮标数量。
    # fc_mhz: 工作频率，频率越高，同样距离误差会造成更大的相位误差。
    # T_long: 浮标运动/输出长块数。
    # K: 每个长块内的算法迭代步数；相邻迭代间隔为 Ts，块时长恒为 K*Ts。
    N: int = 2000
    fc_mhz: float = 20.0
    T_long: int = 20
    K: int = 30
    # Number of shared, unrecorded observations immediately before formal k=0.
    warmup_steps: int = 0
    area_radius: float = 500.0
    uav_height: float = 120.0
    uav_start_x: float | None = None
    uav_start_y: float | None = None
    uav_speed: float = 15.0
    heading_deg: float = 20.0
    # Legacy CLI/output field. Effective block duration is always K * Ts.
    TL: float = 1.0
    Ts: float = 0.05

    # -------------------------
    # 定位误差与运动模型参数
    # -------------------------
    # uav_obs_noise: 每个节点对 UAV 位置观测/估计的高斯误差标准差。
    # buoy_center_obs_noise: 节点自身位置估计误差标准差。
    # TL: 兼容旧命令行参数的保留字段；当前物理时间轴由 K*Ts 决定，
    #     TL 不再参与主循环。
    # Ts: 浮标运动模型的数值积分子步，单位 s；不再与 K 绑定。
    # buoy_random_displacement_std: 每个时刻相对当前中心独立采样的位置偏移标准差。
    # buoy_center_accumulation_ratio: 每步随机小位移中累积到下一时刻中心的比例。
    # system_phase_std_deg: 额外系统相位误差，例如同步误差。
    uav_obs_noise: float = 3.0
    # Additive position-model uncertainty for the full DPC trajectory
    # pseudo-measurement; OLS supplies the remaining position/velocity covariance.
    uav_dpc_prior_noise_std: float = 0.3
    buoy_random_displacement_std: float = 0.03
    buoy_center_accumulation_ratio: float = 0.05
    buoy_center_obs_noise: float = 1.0
    system_phase_std_deg: float = 3.0

    # Optional shore-station broadcast.  One common UAV fix and one fix per
    # buoy are delivered instantaneously to every transmitter at each step.
    shore_broadcast_enabled: bool = False
    shore_broadcast_uav_noise: float = 0.0
    shore_broadcast_node_noise: float = 0.0

    # -------------------------
    # 三维恒速 Kalman 滤波参数
    # -------------------------
    # UAV/浮标状态统一为 [x,y,z,vx,vy,vz]；当前真实高度固定，但滤波仍处理 z。
    uav_kf_accel_std: float = 0.5
    uav_kf_initial_velocity_std: float = 20.0
    # 保留给历史独立实验脚本的速度初始化噪声；窗口级主流程不读取真实 UAV 速度。
    uav_kf_velocity_init_std: float = 2.0
    # Mean current velocity is fixed; stochastic center accumulation is wired
    # separately as position diffusion when the buoy KF is constructed.
    buoy_kf_accel_std: float = 0.0
    # The simulated current speed/direction are fixed and known exactly.  Keep
    # their velocity covariance effectively fixed so noisy position fixes do
    # not create a spurious velocity bias that accumulates into late-time drift.
    buoy_kf_initial_velocity_std: float = 0.0

    # 位移中心按固定洋流速度和方向确定性移动。
    buoy_wave_speed: float = 0.3
    buoy_wave_heading_deg: float = 0.0

    # -------------------------
    # 通信/功率/共识参数
    # -------------------------
    # tx_power: 单个节点发射功率。
    # path_loss_alpha: 幅度路径损耗指数，amp ~ 1 / d^alpha。
    # 默认为 0，表示暂不考虑路径损耗，传播损耗系数恒为 1。
    # global_connectivity: 通信图随机连接概率。
    # uav_consensus_steps: 保留的批量共识接口；主循环每个 iteration 做一次。
    tx_power: float = 5.0
    path_loss_alpha: float = 0.0
    global_connectivity: float = 0.05
    uav_consensus_steps: int = 3
    # Convergence uses the absolute change of distance RMSE between physical
    # steps k and k-1.  It is a simulation-only criterion because d_true is
    # unavailable to a deployed network.
    dpc_distance_rmse_delta_tol_m: float = 0.05
    dpc_consensus_hold_steps: int = 3

    # -------------------------
    # Monte Carlo 与计算后端
    # -------------------------
    mc_trials: int = 20
    mc_seed_stride: int = 1009
    seed: int = 5200
    device: str = "auto"
    gpu_dtype: str = "float32"
    out_dir: str = str(EXPERIMENT_DIR / "modular_dfpc_outputs")

    # -------------------------
    # 调试接口
    # -------------------------
    # debug=True 后会输出每个指定 block/iteration 的中间量：
    # UAV 真值/估计、节点真值/估计、距离误差、相位残差等。
    debug: bool = False
    debug_dir: str | None = None
    debug_trial: int = 0
    debug_blocks: str = "0"
    debug_iterations: str = "0,-1"
    debug_max_nodes: int = 32


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """把 ExperimentConfig 中的参数暴露成命令行参数。

    `run_experiment.py` 和 `run_sweeps.py` 都复用这个函数，所以新加
    参数时通常只需要在 `ExperimentConfig` 和这里各加一次。
    """
    parser.add_argument("--N", type=int, default=ExperimentConfig.N)
    parser.add_argument("--fc_mhz", type=float, default=ExperimentConfig.fc_mhz)
    parser.add_argument("--T_long", type=int, default=ExperimentConfig.T_long)
    parser.add_argument("--K", type=int, default=ExperimentConfig.K)
    parser.add_argument("--warmup_steps", type=int, default=ExperimentConfig.warmup_steps)
    parser.add_argument("--area_radius", type=float, default=ExperimentConfig.area_radius)
    parser.add_argument("--uav_height", type=float, default=ExperimentConfig.uav_height)
    parser.add_argument("--uav_start_x", type=float, default=None)
    parser.add_argument("--uav_start_y", type=float, default=None)
    parser.add_argument("--uav_speed", type=float, default=ExperimentConfig.uav_speed)
    parser.add_argument("--heading_deg", type=float, default=ExperimentConfig.heading_deg)
    parser.add_argument(
        "--TL",
        type=float,
        default=ExperimentConfig.TL,
        help="legacy field kept for backward compatibility; effective block duration is K*Ts",
    )
    parser.add_argument("--Ts", type=float, default=ExperimentConfig.Ts)

    parser.add_argument("--uav_obs_noise", type=float, default=ExperimentConfig.uav_obs_noise)
    parser.add_argument(
        "--uav_dpc_prior_noise_std",
        type=float,
        default=ExperimentConfig.uav_dpc_prior_noise_std,
    )
    parser.add_argument(
        "--buoy_random_displacement_std",
        type=float,
        default=ExperimentConfig.buoy_random_displacement_std,
    )
    parser.add_argument(
        "--buoy_center_accumulation_ratio",
        type=float,
        default=ExperimentConfig.buoy_center_accumulation_ratio,
    )
    parser.add_argument("--buoy_center_obs_noise", type=float, default=ExperimentConfig.buoy_center_obs_noise)
    parser.add_argument("--system_phase_std_deg", type=float, default=ExperimentConfig.system_phase_std_deg)
    parser.add_argument("--shore_broadcast_enabled", action="store_true")
    parser.add_argument("--shore_broadcast_uav_noise", type=float, default=ExperimentConfig.shore_broadcast_uav_noise)
    parser.add_argument("--shore_broadcast_node_noise", type=float, default=ExperimentConfig.shore_broadcast_node_noise)
    parser.add_argument("--uav_kf_accel_std", type=float, default=ExperimentConfig.uav_kf_accel_std)
    parser.add_argument("--uav_kf_initial_velocity_std", type=float, default=ExperimentConfig.uav_kf_initial_velocity_std)
    parser.add_argument("--uav_kf_velocity_init_std", type=float, default=ExperimentConfig.uav_kf_velocity_init_std)
    parser.add_argument("--buoy_kf_accel_std", type=float, default=ExperimentConfig.buoy_kf_accel_std)
    parser.add_argument("--buoy_kf_initial_velocity_std", type=float, default=ExperimentConfig.buoy_kf_initial_velocity_std)
    parser.add_argument("--buoy_wave_speed", type=float, default=ExperimentConfig.buoy_wave_speed)
    parser.add_argument("--buoy_wave_heading_deg", type=float, default=ExperimentConfig.buoy_wave_heading_deg)

    parser.add_argument("--tx_power", type=float, default=ExperimentConfig.tx_power)
    parser.add_argument("--path_loss_alpha", type=float, default=ExperimentConfig.path_loss_alpha)
    parser.add_argument("--global_connectivity", type=float, default=ExperimentConfig.global_connectivity)
    parser.add_argument("--uav_consensus_steps", type=int, default=ExperimentConfig.uav_consensus_steps)
    parser.add_argument(
        "--dpc_distance_rmse_delta_tol_m",
        type=float,
        default=ExperimentConfig.dpc_distance_rmse_delta_tol_m,
    )
    parser.add_argument(
        "--dpc_consensus_hold_steps",
        type=int,
        default=ExperimentConfig.dpc_consensus_hold_steps,
    )
    parser.add_argument("--mc_trials", type=int, default=ExperimentConfig.mc_trials)
    parser.add_argument("--mc_seed_stride", type=int, default=ExperimentConfig.mc_seed_stride)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=ExperimentConfig.device)
    parser.add_argument("--gpu_dtype", choices=["float32", "float64"], default=ExperimentConfig.gpu_dtype)
    parser.add_argument("--out_dir", type=str, default=ExperimentConfig.out_dir)

    parser.add_argument("--debug", action="store_true", help="write per-step debug CSV snapshots")
    parser.add_argument("--debug_dir", type=str, default=None)
    parser.add_argument("--debug_trial", type=int, default=ExperimentConfig.debug_trial)
    parser.add_argument("--debug_blocks", type=str, default=ExperimentConfig.debug_blocks)
    parser.add_argument("--debug_iterations", type=str, default=ExperimentConfig.debug_iterations)
    parser.add_argument("--debug_max_nodes", type=int, default=ExperimentConfig.debug_max_nodes)


def parse_experiment_args() -> ExperimentConfig:
    """解析单次实验入口的命令行参数，返回统一配置对象。"""
    parser = argparse.ArgumentParser(description="Modular windowed trajectory-KF DPC experiment")
    add_common_args(parser)
    return ExperimentConfig(**vars(parser.parse_args()))


def validate_config(cfg: ExperimentConfig) -> None:
    """在启动重计算前集中检查会破坏物理含义或数组形状的参数。"""
    if cfg.N < 1 or cfg.T_long < 1 or cfg.K < 1:
        raise ValueError("N, T_long and K must be positive")
    if cfg.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if cfg.TL <= 0.0 or cfg.Ts <= 0.0:
        raise ValueError("TL and Ts must both be positive")
    if not 0.0 <= cfg.buoy_center_accumulation_ratio <= 1.0:
        raise ValueError("buoy_center_accumulation_ratio must be in [0, 1]")
    if cfg.dpc_consensus_hold_steps < 1:
        raise ValueError("dpc_consensus_hold_steps must be positive")
    nonnegative = {
        "uav_speed": cfg.uav_speed,
        "uav_obs_noise": cfg.uav_obs_noise,
        "uav_dpc_prior_noise_std": cfg.uav_dpc_prior_noise_std,
        "dpc_distance_rmse_delta_tol_m": cfg.dpc_distance_rmse_delta_tol_m,
        "buoy_random_displacement_std": cfg.buoy_random_displacement_std,
        "buoy_center_obs_noise": cfg.buoy_center_obs_noise,
        "buoy_wave_speed": cfg.buoy_wave_speed,
        "uav_kf_accel_std": cfg.uav_kf_accel_std,
        "uav_kf_initial_velocity_std": cfg.uav_kf_initial_velocity_std,
        "buoy_kf_accel_std": cfg.buoy_kf_accel_std,
        "buoy_kf_initial_velocity_std": cfg.buoy_kf_initial_velocity_std,
        "shore_broadcast_uav_noise": cfg.shore_broadcast_uav_noise,
        "shore_broadcast_node_noise": cfg.shore_broadcast_node_noise,
    }
    invalid = [name for name, value in nonnegative.items() if value < 0.0]
    if invalid:
        raise ValueError(f"these parameters must be non-negative: {', '.join(invalid)}")

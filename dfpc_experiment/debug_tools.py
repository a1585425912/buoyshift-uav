"""调试快照工具。

打开 `--debug` 后，这个模块会把指定 block/iteration 的中间变量写成 CSV。
用途是定位异常点：例如某个频率下功率突然下降时，可以检查是 UAV 估计误差、
节点位置误差、距离误差还是相位残差导致的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import sim_core as sim

from .config import ExperimentConfig
from .io_utils import write_rows


def parse_index_filter(text: str, size: int) -> set[int]:
    """解析调试索引。

    支持：
    - "all" 或 "*"：保存全部；
    - "0,-1"：保存第一个和最后一个；
    - "0,2,5"：保存指定索引。
    """
    text = (text or "").strip().lower()
    if text in {"all", "*"}:
        return set(range(size))
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx = size + idx
        if 0 <= idx < size:
            out.add(idx)
    return out


@dataclass
class DebugRecorder:
    """收集并输出调试 CSV。

    summary_rows 是每个捕获点一行；
    node_rows 是每个捕获点下若干节点的详细误差。
    """

    cfg: ExperimentConfig
    enabled: bool
    block_filter: set[int] = field(default_factory=set)
    iter_filter: set[int] = field(default_factory=set)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    node_rows: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: ExperimentConfig) -> "DebugRecorder":
        """根据命令行参数构造调试记录器。"""
        return cls(
            cfg=cfg,
            enabled=bool(cfg.debug),
            block_filter=parse_index_filter(cfg.debug_blocks, cfg.T_long),
            iter_filter=parse_index_filter(cfg.debug_iterations, cfg.K),
        )

    def should_capture(self, trial_index: int, block: int, iteration: int) -> bool:
        """判断当前 trial/block/iteration 是否需要保存调试快照。"""
        return (
            self.enabled
            and trial_index == int(self.cfg.debug_trial)
            and block in self.block_filter
            and iteration in self.iter_filter
        )

    def capture(
        self,
        *,
        trial_index: int,
        seed: int,
        block: int,
        iteration: int,
        method: str,
        p_u_true: np.ndarray,
        p_u_est: np.ndarray,
        node_true_xyz: np.ndarray,
        node_est_xyz: np.ndarray,
        phi_true: np.ndarray,
        amp: np.ndarray,
        eps: np.ndarray,
        k_const: float,
        metric: dict[str, float],
        line_params: np.ndarray | None = None,
    ) -> None:
        """保存一个调试快照。

        这里重新计算 d_hat、d_true 和 residual phase，是为了把“误差来源”
        直接写进 CSV，后续不用再回到代码里推。
        """
        if not self.should_capture(trial_index, block, iteration):
            return

        node_est_xyz = np.asarray(node_est_xyz, dtype=np.float64)
        node_true_xyz = np.asarray(node_true_xyz, dtype=np.float64)
        d_hat = np.linalg.norm(p_u_est - node_est_xyz, axis=1)
        d_true = np.linalg.norm(p_u_true[None, :] - node_true_xyz, axis=1)
        theta = k_const * d_hat
        residual = sim.effective_phase_errors(theta, phi_true, eps)

        mean_line = None if line_params is None else np.mean(line_params, axis=0)
        self.summary_rows.append(
            # 汇总表：适合快速看某个异常点的整体状态。
            {
                "trial_index": trial_index,
                "seed": seed,
                "long_block": block,
                "iteration_step": iteration,
                "method": method,
                "uav_true_x": float(p_u_true[0]),
                "uav_true_y": float(p_u_true[1]),
                "uav_true_z": float(p_u_true[2]),
                "uav_est_mean_x": float(np.mean(p_u_est[:, 0])),
                "uav_est_mean_y": float(np.mean(p_u_est[:, 1])),
                "uav_est_mean_z": float(np.mean(p_u_est[:, 2])),
                "line_bx_mean": "" if mean_line is None else float(mean_line[0]),
                "line_vx_mean": "" if mean_line is None else float(mean_line[1]),
                "line_by_mean": "" if mean_line is None else float(mean_line[2]),
                "line_vy_mean": "" if mean_line is None else float(mean_line[3]),
                "line_bz_mean": "" if mean_line is None or len(mean_line) < 9 else float(mean_line[4]),
                "line_vz_mean": "" if mean_line is None or len(mean_line) < 9 else float(mean_line[5]),
                "direction_x_mean": "" if mean_line is None or len(mean_line) < 9 else float(mean_line[6]),
                "direction_y_mean": "" if mean_line is None or len(mean_line) < 9 else float(mean_line[7]),
                "direction_z_mean": "" if mean_line is None or len(mean_line) < 9 else float(mean_line[8]),
                "norm_db": float(metric["norm_db"]),
                "gain_over_single_mean_db": float(metric["gain_over_single_mean_db"]),
                "phase_std_deg": float(metric["phase_std_deg"]),
                "distance_rmse": float(metric["distance_rmse"]) if not np.isnan(metric["distance_rmse"]) else "",
                "node_rmse": float(metric["node_rmse"]) if not np.isnan(metric["node_rmse"]) else "",
                "uav_rmse": float(metric["uav_rmse"]) if not np.isnan(metric["uav_rmse"]) else "",
                "uav_line_intercept_rmse": float(metric["uav_line_intercept_rmse"]),
                "uav_velocity_rmse": float(metric["uav_velocity_rmse"]),
                "residual_phase_mean_deg": float(np.rad2deg(np.mean(residual))),
                "residual_phase_std_deg": float(np.rad2deg(sim.sample_phase_std(residual))),
                "residual_phase_rmse_deg": float(np.rad2deg(sim.phase_rmse(residual))),
            }
        )

        max_nodes = min(int(self.cfg.debug_max_nodes), node_true_xyz.shape[0])
        for node_id in range(max_nodes):
            self.node_rows.append(
                # 节点表：适合检查少数节点的距离误差和相位残差是否特别大。
                {
                    "trial_index": trial_index,
                    "seed": seed,
                    "long_block": block,
                    "iteration_step": iteration,
                    "method": method,
                    "node_id": node_id,
                    "node_true_x": float(node_true_xyz[node_id, 0]),
                    "node_true_y": float(node_true_xyz[node_id, 1]),
                    "node_true_z": float(node_true_xyz[node_id, 2]),
                    "node_est_x": float(node_est_xyz[node_id, 0]),
                    "node_est_y": float(node_est_xyz[node_id, 1]),
                    "node_est_z": float(node_est_xyz[node_id, 2]),
                    "uav_est_x": float(p_u_est[node_id, 0]),
                    "uav_est_y": float(p_u_est[node_id, 1]),
                    "uav_est_z": float(p_u_est[node_id, 2]),
                    "line_bx": "" if line_params is None else float(line_params[node_id, 0]),
                    "line_vx": "" if line_params is None else float(line_params[node_id, 1]),
                    "line_by": "" if line_params is None else float(line_params[node_id, 2]),
                    "line_vy": "" if line_params is None else float(line_params[node_id, 3]),
                    "line_bz": "" if line_params is None or line_params.shape[1] < 9 else float(line_params[node_id, 4]),
                    "line_vz": "" if line_params is None or line_params.shape[1] < 9 else float(line_params[node_id, 5]),
                    "direction_x": "" if line_params is None or line_params.shape[1] < 9 else float(line_params[node_id, 6]),
                    "direction_y": "" if line_params is None or line_params.shape[1] < 9 else float(line_params[node_id, 7]),
                    "direction_z": "" if line_params is None or line_params.shape[1] < 9 else float(line_params[node_id, 8]),
                    "d_true": float(d_true[node_id]),
                    "d_hat": float(d_hat[node_id]),
                    "distance_error": float(d_hat[node_id] - d_true[node_id]),
                    "residual_phase_deg": float(np.rad2deg(residual[node_id])),
                    "amp": float(amp[node_id]),
                }
            )

    def write(self, out_dir: Path) -> None:
        """把调试快照写到 debug 目录。"""
        if not self.enabled:
            return
        debug_dir = Path(self.cfg.debug_dir) if self.cfg.debug_dir else out_dir / "debug"
        write_rows(debug_dir / "debug_step_summary.csv", self.summary_rows)
        write_rows(debug_dir / "debug_node_samples.csv", self.node_rows)

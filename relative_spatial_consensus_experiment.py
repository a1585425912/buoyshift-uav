# relative_spatial_consensus_experiment.py
# -*- coding: utf-8 -*-

"""
实验目标：
1. 纯空域模型，不考虑电状态，不考虑KF，不依赖UAV回传；
2. 将空间路径误差分解为：
   - UAV-Node相对路径几何误差：由 rho_hat 不一致/不准确导致；
   - node自身误差：由浮标自身位置估计误差导致；
3. 使用平均共识降低各node对UAV相对位置估计的不一致性；
4. 比较：
   - full error: 相对路径误差 + node自身误差
   - relative-only error: 仅相对路径误差
   - self-only error: 仅node自身误差
   - ideal: 完美对齐
5. 频率范围：30-50 MHz
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

from dfpc_experiment.sim_core import apply_polarity_correction


def configure_matplotlib_fonts():
    """Use an installed CJK font when plotting Chinese labels."""
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Microsoft JhengHei",
        "STSong",
        "STXihei",
        "KaiTi",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


configure_matplotlib_fonts()


# =========================
# 0. 全局参数
# =========================

SEED = 2026
rng = np.random.default_rng(SEED)

C0 = 3e8  # 光速，m/s

# 节点与场景参数
N = 200
SEA_RADIUS = 1000.0       # 浮标网络覆盖半径，m
UAV_HEIGHT = 800.0        # UAV高度，m
UAV_XY = np.array([500.0, -300.0])  # UAV在水平面上的相对位置，m

# 误差参数
SIGMA_RHO = 80.0          # 每个node对UAV相对位置估计误差标准差，m
SIGMA_SELF = 2.0          # node自身位置估计误差标准差，m

# 如果存在所有节点共同的UAV相对位置系统偏差，可以在这里设置
# 注意：普通平均共识不能消除公共系统偏差，只能消除节点间不一致的随机估计误差
COMMON_RHO_BIAS = np.array([0.0, 0.0, 0.0])

# 共识参数
MAX_ITER = 60
CONNECTIVITY = 0.05      # 边数 / 完全图边数
MC_RUNS = 200

# 频率设置
FREQ_LIST = [30e6, 40e6, 50e6]
MAIN_FREQ = 40e6

# 是否使用路径衰减
# False: 只研究相位相干影响，所有node幅度相同
# True : 使用 a_n = 1 / d_n
USE_PATH_LOSS = False

OUT_DIR = Path("results_relative_spatial_consensus")
OUT_DIR.mkdir(exist_ok=True)


# =========================
# 1. 工具函数
# =========================

def uniform_disk_points(n, radius, rng):
    """
    在半径为 radius 的圆盘内均匀生成 n 个点。
    输出 shape = (n, 2)
    """
    r = radius * np.sqrt(rng.random(n))
    theta = 2 * np.pi * rng.random(n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def build_connected_graph(n, connectivity, rng):
    """
    构造一个无向连通图。
    connectivity = 实际边数 / 完全图边数
    先生成随机树保证连通，再补边达到目标连通度。
    """
    max_edges = n * (n - 1) // 2
    target_edges = int(connectivity * max_edges)
    target_edges = max(target_edges, n - 1)

    adj = np.zeros((n, n), dtype=bool)

    # 随机树，保证连通
    for i in range(1, n):
        j = rng.integers(0, i)
        adj[i, j] = True
        adj[j, i] = True

    current_edges = n - 1

    # 补边
    while current_edges < target_edges:
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i != j and not adj[i, j]:
            adj[i, j] = True
            adj[j, i] = True
            current_edges += 1

    return adj


def metropolis_hastings_matrix(adj):
    """
    根据无向图构造 Metropolis-Hastings 共识矩阵 W。
    W 为对称双随机矩阵。
    """
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    W = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j]:
                W[i, j] = 1.0 / (max(deg[i], deg[j]) + 1.0)

    for i in range(n):
        W[i, i] = 1.0 - W[i].sum()

    return W


def second_largest_eigenvalue_modulus(W):
    """
    计算 W 的第二大特征值模，用于衡量共识收敛速度。
    越小，收敛越快。
    """
    eigvals = np.linalg.eigvals(W)
    eig_abs = np.sort(np.abs(eigvals))[::-1]
    return eig_abs[1].real


def coherent_gain_db(d_true, d_est, freq_hz, use_path_loss=False):
    """
    根据真实路径 d_true 和估计补偿路径 d_est 计算归一化相干增益。

    发射补偿相位：
        theta_n = 2*pi/lambda * d_est_n

    UAV处剩余相位：
        residual_n = 2*pi/lambda * (d_est_n - d_true_n)

    归一化相干增益：
        G = |sum a_n exp(j residual_n)|^2 / (sum a_n)^2
    """
    wavelength = C0 / freq_hz
    k0 = 2 * np.pi / wavelength

    residual_phase = k0 * (d_est - d_true)
    residual_phase = apply_polarity_correction(residual_phase)

    if use_path_loss:
        amp = 1.0 / np.maximum(d_true, 1e-9)
    else:
        amp = np.ones_like(d_true)

    E = np.sum(amp * np.exp(1j * residual_phase))
    P = np.abs(E) ** 2
    P_ideal = np.sum(amp) ** 2

    gain = P / P_ideal
    gain_db = 10 * np.log10(np.maximum(gain, 1e-15))

    return gain_db


def run_consensus_experiment(
    W,
    delta_true,
    rho_true,
    freq_hz,
    sigma_rho,
    sigma_self,
    max_iter,
    mc_runs,
    rng,
    common_rho_bias=None,
    use_path_loss=False,
):
    """
    运行 Monte Carlo 实验。

    delta_true:
        node相对于浮标网络参考点的位置，shape=(N,3)

    rho_true:
        UAV相对于浮标网络参考点的位置，shape=(3,)

    每个node的本地UAV相对位置估计：
        rho_hat_n = rho_true + common_bias + random_error_n

    每个node自身位置估计：
        delta_hat_n = delta_true_n + self_error_n

    输出：
        full_error_curve:
            相对路径误差 + node自身误差

        rel_only_curve:
            仅相对路径误差，node位置理想

        self_only_curve:
            仅node自身误差，UAV相对位置理想
    """
    n = delta_true.shape[0]

    if common_rho_bias is None:
        common_rho_bias = np.zeros(3)

    full_all = np.zeros((mc_runs, max_iter + 1))
    rel_only_all = np.zeros((mc_runs, max_iter + 1))
    self_only_all = np.zeros((mc_runs, max_iter + 1))

    d_true = np.linalg.norm(rho_true[None, :] - delta_true, axis=1)

    for mc in range(mc_runs):
        # 每个node对UAV相对位置的本地估计误差
        rho_noise = rng.normal(0.0, sigma_rho, size=(n, 3))

        # 这里默认UAV高度估计也可能有误差。
        # 如果你认为高度已知，可以取消下一行注释：
        # rho_noise[:, 2] = 0.0

        rho_states = rho_true[None, :] + common_rho_bias[None, :] + rho_noise

        # node自身位置估计误差
        self_error = rng.normal(0.0, sigma_self, size=(n, 3))

        # 浮标在海面上，默认只考虑水平位置误差
        self_error[:, 2] = 0.0

        delta_hat = delta_true + self_error

        # self-only 情况：UAV相对位置完全正确，只有node自身误差
        d_est_self_only = np.linalg.norm(rho_true[None, :] - delta_hat, axis=1)
        self_only_gain = coherent_gain_db(
            d_true, d_est_self_only, freq_hz, use_path_loss
        )

        for it in range(max_iter + 1):
            # full error：共识后的rho估计 + node自身误差
            d_est_full = np.linalg.norm(rho_states - delta_hat, axis=1)

            # relative-only：共识后的rho估计 + 理想node位置
            d_est_rel_only = np.linalg.norm(rho_states - delta_true, axis=1)

            full_all[mc, it] = coherent_gain_db(
                d_true, d_est_full, freq_hz, use_path_loss
            )

            rel_only_all[mc, it] = coherent_gain_db(
                d_true, d_est_rel_only, freq_hz, use_path_loss
            )

            self_only_all[mc, it] = self_only_gain

            # 共识迭代
            if it < max_iter:
                rho_states = W @ rho_states

    return {
        "full_mean": full_all.mean(axis=0),
        "full_std": full_all.std(axis=0),
        "rel_only_mean": rel_only_all.mean(axis=0),
        "rel_only_std": rel_only_all.std(axis=0),
        "self_only_mean": self_only_all.mean(axis=0),
        "self_only_std": self_only_all.std(axis=0),
    }


# =========================
# 2. 生成空域场景
# =========================

# 参考点取浮标网络中心
p_ref = np.array([0.0, 0.0, 0.0])

# node真实位置
node_xy = uniform_disk_points(N, SEA_RADIUS, rng)
delta_true = np.column_stack([
    node_xy[:, 0],
    node_xy[:, 1],
    np.zeros(N)
])

# UAV真实相对位置
rho_true = np.array([UAV_XY[0], UAV_XY[1], UAV_HEIGHT])

# 通信图与W矩阵
adj = build_connected_graph(N, CONNECTIVITY, rng)
W = metropolis_hastings_matrix(adj)

lambda2 = second_largest_eigenvalue_modulus(W)
avg_degree = adj.sum(axis=1).mean()

print("========== 场景参数 ==========")
print(f"N = {N}")
print(f"sea radius = {SEA_RADIUS:.1f} m")
print(f"UAV relative position rho_true = {rho_true}")
print(f"connectivity = {CONNECTIVITY}")
print(f"average degree = {avg_degree:.2f}")
print(f"lambda2(W) = {lambda2:.4f}")
print(f"sigma_rho = {SIGMA_RHO:.2f} m")
print(f"sigma_self = {SIGMA_SELF:.2f} m")
print(f"MC runs = {MC_RUNS}")
print("==============================")


# =========================
# 3. 实验一：误差分解曲线
# =========================

result_main = run_consensus_experiment(
    W=W,
    delta_true=delta_true,
    rho_true=rho_true,
    freq_hz=MAIN_FREQ,
    sigma_rho=SIGMA_RHO,
    sigma_self=SIGMA_SELF,
    max_iter=MAX_ITER,
    mc_runs=MC_RUNS,
    rng=rng,
    common_rho_bias=COMMON_RHO_BIAS,
    use_path_loss=USE_PATH_LOSS,
)

iters = np.arange(MAX_ITER + 1)

plt.figure(figsize=(8, 5))
plt.plot(iters, result_main["full_mean"], label="full: 相对路径误差 + node自身误差")
plt.plot(iters, result_main["rel_only_mean"], label="relative-only: 仅UAV-Node相对路径误差")
plt.plot(iters, result_main["self_only_mean"], label="self-only: 仅node自身误差")
plt.axhline(0.0, linestyle="--", linewidth=1, label="ideal: 完美相干 0 dB")
plt.xlabel("Consensus iteration")
plt.ylabel("Normalized coherent gain (dB)")
plt.title(f"Error decomposition under spatial consensus, f = {MAIN_FREQ/1e6:.0f} MHz")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

fig1 = OUT_DIR / "01_error_decomposition_vs_iteration.png"
plt.savefig(fig1, dpi=300)
plt.close()


# =========================
# 4. 实验二：30/40/50 MHz 对比
# =========================

freq_summary = {}

for f in FREQ_LIST:
    res = run_consensus_experiment(
        W=W,
        delta_true=delta_true,
        rho_true=rho_true,
        freq_hz=f,
        sigma_rho=SIGMA_RHO,
        sigma_self=SIGMA_SELF,
        max_iter=MAX_ITER,
        mc_runs=MC_RUNS,
        rng=rng,
        common_rho_bias=COMMON_RHO_BIAS,
        use_path_loss=USE_PATH_LOSS,
    )
    freq_summary[f] = res

plt.figure(figsize=(8, 5))
for f in FREQ_LIST:
    plt.plot(
        iters,
        freq_summary[f]["full_mean"],
        label=f"{f/1e6:.0f} MHz"
    )

plt.axhline(0.0, linestyle="--", linewidth=1, label="ideal: 0 dB")
plt.xlabel("Consensus iteration")
plt.ylabel("Normalized coherent gain (dB)")
plt.title("Frequency comparison: 30-50 MHz")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

fig2 = OUT_DIR / "02_frequency_comparison_30_40_50MHz.png"
plt.savefig(fig2, dpi=300)
plt.close()


# =========================
# 5. 实验三：网络连通度对共识效果的影响
# =========================

connectivity_list = [0.02, 0.05, 0.10, 0.20]
conn_results = {}

for c in connectivity_list:
    adj_c = build_connected_graph(N, c, rng)
    W_c = metropolis_hastings_matrix(adj_c)
    lambda2_c = second_largest_eigenvalue_modulus(W_c)

    res_c = run_consensus_experiment(
        W=W_c,
        delta_true=delta_true,
        rho_true=rho_true,
        freq_hz=MAIN_FREQ,
        sigma_rho=SIGMA_RHO,
        sigma_self=SIGMA_SELF,
        max_iter=MAX_ITER,
        mc_runs=MC_RUNS,
        rng=rng,
        common_rho_bias=COMMON_RHO_BIAS,
        use_path_loss=USE_PATH_LOSS,
    )

    conn_results[c] = {
        "lambda2": lambda2_c,
        "avg_degree": adj_c.sum(axis=1).mean(),
        "result": res_c,
    }

plt.figure(figsize=(8, 5))
for c in connectivity_list:
    label = (
        f"c={c:.2f}, "
        f"D={conn_results[c]['avg_degree']:.1f}, "
        f"lambda2={conn_results[c]['lambda2']:.3f}"
    )
    plt.plot(iters, conn_results[c]["result"]["full_mean"], label=label)

plt.axhline(0.0, linestyle="--", linewidth=1, label="ideal: 0 dB")
plt.xlabel("Consensus iteration")
plt.ylabel("Normalized coherent gain (dB)")
plt.title(f"Connectivity comparison, f = {MAIN_FREQ/1e6:.0f} MHz")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()

fig3 = OUT_DIR / "03_connectivity_comparison.png"
plt.savefig(fig3, dpi=300)
plt.close()


# =========================
# 6. 实验四：node自身误差残差地板
# =========================

sigma_self_list = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
self_floor_full = []
self_floor_only = []

for sigma_self in sigma_self_list:
    res_s = run_consensus_experiment(
        W=W,
        delta_true=delta_true,
        rho_true=rho_true,
        freq_hz=MAIN_FREQ,
        sigma_rho=SIGMA_RHO,
        sigma_self=sigma_self,
        max_iter=MAX_ITER,
        mc_runs=MC_RUNS,
        rng=rng,
        common_rho_bias=COMMON_RHO_BIAS,
        use_path_loss=USE_PATH_LOSS,
    )

    self_floor_full.append(res_s["full_mean"][-1])
    self_floor_only.append(res_s["self_only_mean"][-1])

plt.figure(figsize=(8, 5))
plt.plot(sigma_self_list, self_floor_full, marker="o", label="after consensus: full error")
plt.plot(sigma_self_list, self_floor_only, marker="s", label="residual floor: self-only")
plt.axhline(0.0, linestyle="--", linewidth=1, label="ideal: 0 dB")
plt.xlabel("Node self-position error std sigma_self (m)")
plt.ylabel("Final normalized coherent gain (dB)")
plt.title(f"Residual floor caused by node self error, f = {MAIN_FREQ/1e6:.0f} MHz")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

fig4 = OUT_DIR / "04_node_self_error_floor.png"
plt.savefig(fig4, dpi=300)
plt.close()


# =========================
# 7. 打印结果
# =========================

print("\n========== 输出图像 ==========")
print(fig1)
print(fig2)
print(fig3)
print(fig4)

print("\n========== 主实验最终结果 ==========")
print(f"frequency = {MAIN_FREQ/1e6:.0f} MHz")
print(f"full final gain       = {result_main['full_mean'][-1]:.3f} dB")
print(f"relative-only final   = {result_main['rel_only_mean'][-1]:.3f} dB")
print(f"self-only final floor = {result_main['self_only_mean'][-1]:.3f} dB")

print("\n========== 频率对比最终结果 ==========")
for f in FREQ_LIST:
    print(
        f"{f/1e6:.0f} MHz: "
        f"full final gain = {freq_summary[f]['full_mean'][-1]:.3f} dB, "
        f"self-only floor = {freq_summary[f]['self_only_mean'][-1]:.3f} dB"
    )

print("\n========== 连通度对比 ==========")
for c in connectivity_list:
    print(
        f"c={c:.2f}, "
        f"avg degree={conn_results[c]['avg_degree']:.2f}, "
        f"lambda2={conn_results[c]['lambda2']:.4f}, "
        f"final gain={conn_results[c]['result']['full_mean'][-1]:.3f} dB"
    )

print("\nDone.")

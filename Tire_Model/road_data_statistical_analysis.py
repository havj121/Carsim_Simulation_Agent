import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# 配置路径，确保能找到 analyze_friction_utilization 和 carsim_agent 模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from tire_data_preprocessing import load_all_wheels_data

# ------------------------------
# 1. 数据统计分析函数
# ------------------------------
def statistical_analysis(data, output_dir="./Tire_Model/Statistical_Analysis"):
    """
    对采集的道路数据进行统计分析
    """
    if data.empty:
        print("No data for statistical analysis.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # 1.1 数据时长统计分析
    # 假设采样频率是固定的，或者通过 Time 列计算
    unique_mu = data["mu_road"].unique()
    
    # 计算总时长 (假设每个点代表一个采样周期，从数据中提取)
    # 实际上由于是合并的数据集，我们需要按不同的仿真序列计算时长
    # 这里简单处理：假设采样间隔为 dt
    dt = 0.025 # 默认 25ms，或者根据数据计算
    if len(data) > 1:
        # 尝试从 Time 列估算 dt (取众数或平均值)
        diffs = data["Time"].diff().dropna()
        if not diffs.empty:
            dt = diffs.median()
    
    total_points = len(data)
    total_duration = total_points * dt
    
    print("\n" + "="*40)
    print("📊 DATA DURATION STATISTICS")
    print(f"Total points: {total_points}")
    print(f"Estimated Total Duration: {total_duration:.2f} s")
    
    duration_info = []
    for mu in unique_mu:
        mu_data = data[data["mu_road"] == mu]
        mu_points = len(mu_data)
        mu_duration = mu_points * dt
        print(f"Mu = {mu:.2f}: {mu_points} points, Duration: {mu_duration:.2f} s")
        duration_info.append({"mu": mu, "points": mu_points, "duration": mu_duration})
    print("="*40 + "\n")

    # 1.2 横向力与侧偏角符号一致性分析
    analyze_sign_consistency(data)

    # 1.3 绘制直方图分布
    plot_histograms(data, output_dir)
    
    # 1.4 绘制力学特性散点图
    plot_mechanics_scatters(data, output_dir)

def analyze_sign_consistency(data):
    """
    分析横向力 Fy 与侧偏角 Alpha 的符号一致性 (Fy * Alpha < 0 则不一致)
    """
    wheels = ["L1", "R1", "L2", "R2"]
    print("="*40)
    print("🚩 FY & ALPHA SIGN INCONSISTENCY ANALYSIS")
    
    total_inconsistent = 0
    total_points = len(data) * 4
    
    for w in wheels:
        fy = data[f"fy_{w}"]
        alpha = data[f"alpha_{w}"]
        
        # 统计符号不一致的点 (排除接近 0 的点以减少噪声影响)
        mask_inconsistent = (fy * alpha < 0) & (np.abs(fy) > 50) & (np.abs(alpha) > 0.01)
        inconsistent_count = mask_inconsistent.sum()
        ratio = (inconsistent_count / len(data)) * 100
        
        print(f"Wheel {w}: {inconsistent_count} inconsistent points ({ratio:.2f}%)")
        total_inconsistent += inconsistent_count
        
    total_ratio = (total_inconsistent / total_points) * 100
    print(f"Overall Inconsistency Ratio: {total_ratio:.2f}%")
    print("="*40 + "\n")

def plot_histograms(data, output_dir):
    """
    分析统计分布：
    1. 质心侧偏角 (Beta) 分布图
    2. 后轮纵向滑移率 (Kappa) 分布图 (限制 0-3)
    3. 后轮纵向滑移率 (Kappa) 累计频率分布图 (CDF)
    4. 四轮侧偏角 (Alpha) 分布图 (统计 +/- 40 deg 比例)
    5. 轮胎力分布 (Fx, Fy, Fz)
    """
    # 1. 质心侧偏角 (Beta) 单独保存
    plt.figure(figsize=(10, 6))
    beta_deg = data["beta"] * (180.0 / np.pi)
    plt.hist(beta_deg, bins=50, color='plum', edgecolor='black', alpha=0.7)
    plt.title("Vehicle Sideslip Angle (Beta) Distribution", fontsize=14)
    plt.xlabel("Beta (deg)")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "vehicle_beta_distribution.png"), dpi=150)
    plt.close()

    # 2. 后轮纵向滑移率 (Kappa) 分布图 (限制 0-3)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Rear Wheel Longitudinal Slip Distribution (Range 0-3)", fontsize=16)
    
    for i, w in enumerate(["L2", "R2"]):
        kappa_data = data[f"kappa_{w}"]
        # 计算 0-1 之间的比例
        in_range_count = ((kappa_data >= 0) & (kappa_data <= 1.0)).sum()
        ratio = (in_range_count / len(kappa_data)) * 100
        
        axes[i].hist(kappa_data, bins=50, range=(0, 3.0), color='orange', edgecolor='black', alpha=0.7)
        axes[i].set_title(f"Wheel {w} (0-1 Ratio: {ratio:.2f}%)")
        axes[i].set_xlabel("Longitudinal Slip Kappa")
        axes[i].set_ylabel("Count")
        axes[i].grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "rear_wheels_kappa_distribution.png"), dpi=150)
    plt.close()

    # 3. 后轮纵向滑移率 (Kappa) 累计频率分布图 (CDF)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Rear Wheel Longitudinal Slip Cumulative Frequency (CDF)", fontsize=16)
    
    for i, w in enumerate(["L2", "R2"]):
        kappa_data = data[f"kappa_{w}"].values
        # 排序并计算累积概率 (使用全体数据进行归一化)
        sorted_kappa = np.sort(kappa_data)
        y_cdf = np.arange(1, len(sorted_kappa) + 1) / len(sorted_kappa)
        
        # 绘制 CDF 曲线 (手动绘制以确保归一化一致)
        axes[i].step(sorted_kappa, y_cdf, color='blue', linewidth=2, where='post')
        
        # 标记关键阈值下的比例 (如 0.1, 0.5, 1.0, 10.0)
        thresholds = [0.1, 0.5, 1.0, 10.0]
        y_text = 0.8
        for th in thresholds:
            ratio = (kappa_data <= th).mean() * 100
            axes[i].axvline(th, color='red', linestyle='--', alpha=0.5)
            # 找到对应的 y 值 (CDF 值)
            axes[i].text(th + 0.2, y_text, f'<{th}: {ratio:.1f}%', fontsize=10, color='red')
            y_text -= 0.1
            
        axes[i].set_title(f"Wheel {w} Global Cumulative Distribution")
        axes[i].set_xlabel("Longitudinal Slip Kappa")
        axes[i].set_ylabel("Cumulative Frequency (Total = 100%)")
        axes[i].grid(True, linestyle='--', alpha=0.5)
        axes[i].set_xlim(0, 15.0) # 仅显示 0-15 范围
        axes[i].set_ylim(0, 1.05)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "rear_wheels_kappa_cdf.png"), dpi=150)
    plt.close()

    # 4. 四轮侧偏角 (Alpha) 合并分布图
    plt.figure(figsize=(12, 7))
    wheels = ["L1", "R1", "L2", "R2"]
    colors = ['skyblue', 'salmon', 'lightgreen', 'plum']
    
    # 将四轮 alpha 数据合并成一个长序列进行统计
    all_alpha_deg = pd.concat([data[f"alpha_{w}"] for w in wheels]) * (180.0 / np.pi)
    
    # 统计 +/- 40 deg 比例
    in_range_mask = (all_alpha_deg >= -40.0) & (all_alpha_deg <= 40.0)
    in_range_ratio = (in_range_mask.sum() / len(all_alpha_deg)) * 100
    
    plt.hist(all_alpha_deg, bins=80, range=(-60, 60), color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(40, color='red', linestyle='--', label='40 deg')
    plt.axvline(-40, color='red', linestyle='--', label='-40 deg')
    plt.title(f"All Wheels Slip Angle (Alpha) Distribution\n(+/- 40 deg Ratio: {in_range_ratio:.2f}%)", fontsize=14)
    plt.xlabel("Slip Angle (deg)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "all_wheels_alpha_distribution.png"), dpi=150)
    plt.close()

    # 4. 轮胎力分布 (Fx, Fy, Fz) - 保持原样
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle("Tire Forces Distribution (Fx, Fy, Fz)", fontsize=20)
    wheel_labels = ["Front Left", "Front Right", "Rear Left", "Rear Right"]
    
    for i, w in enumerate(wheels):
        axes[0, i].hist(data[f"fx_{w}"], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, i].set_title(f"{wheel_labels[i]} Fx")
        axes[1, i].hist(data[f"fy_{w}"], bins=50, color='salmon', edgecolor='black', alpha=0.7)
        axes[1, i].set_title(f"{wheel_labels[i]} Fy")
        axes[2, i].hist(data[f"fz_{w}"], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[2, i].set_title(f"{wheel_labels[i]} Fz")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "tire_forces_histogram.png"), dpi=150)
    plt.close()

def plot_mechanics_scatters(data, output_dir):
    """
    1. 所有轮胎横向力 vs 侧偏角 (按 mu 分 Axes)
    2. 后轮纵向力 vs 滑移率 (按 mu 分 Axes)
    """
    unique_mu = sorted(data["mu_road"].unique())
    num_mu = len(unique_mu)
    wheels = ["L1", "R1", "L2", "R2"]
    rear_wheels = ["L2", "R2"]
    
    # 1. 横向力特性分布 (按 Mu 绘制 Axes)
    fig1, axes1 = plt.subplots(1, num_mu, figsize=(10 * num_mu, 8), squeeze=False)
    axes1 = axes1.flatten()
    fig1.suptitle("Lateral Force vs Slip Angle Characteristics (By Road Mu)", fontsize=18)
    
    for i, mu in enumerate(unique_mu):
        mu_data = data[data["mu_road"] == mu]
        all_alpha = pd.concat([mu_data[f"alpha_{w}"] for w in wheels]) * (180.0 / np.pi)
        all_fy = pd.concat([mu_data[f"fy_{w}"] for w in wheels])
        all_fz = pd.concat([mu_data[f"fz_{w}"] for w in wheels])
        
        sc = axes1[i].scatter(all_alpha, all_fy, c=all_fz, s=10, alpha=0.6, cmap='viridis')
        axes1[i].set_title(f"Road Mu = {mu}")
        axes1[i].set_xlabel("Slip Angle (deg)")
        axes1[i].set_ylabel("Lateral Force Fy (N)")
        axes1[i].grid(True, linestyle='--', alpha=0.5)
        plt.colorbar(sc, ax=axes1[i], label="Vertical Load Fz (N)")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "mechanics_lateral_comparison.png"), dpi=150)
    plt.close()

    # 2. 纵向力特性分布 (按 Mu 绘制 Axes)
    fig2, axes2 = plt.subplots(1, num_mu, figsize=(10 * num_mu, 8), squeeze=False)
    axes2 = axes2.flatten()
    fig2.suptitle("Longitudinal Force vs Slip Ratio Characteristics (By Road Mu)", fontsize=18)
    
    for i, mu in enumerate(unique_mu):
        mu_data = data[data["mu_road"] == mu]
        rear_kappa = pd.concat([mu_data[f"kappa_{w}"] for w in rear_wheels])
        rear_fx = pd.concat([mu_data[f"fx_{w}"] for w in rear_wheels])
        rear_fz = pd.concat([mu_data[f"fz_{w}"] for w in rear_wheels])
        
        sc = axes2[i].scatter(rear_kappa, rear_fx, c=rear_fz, s=15, alpha=0.6, cmap='plasma')
        axes2[i].set_title(f"Road Mu = {mu}")
        axes2[i].set_xlabel("Longitudinal Slip Kappa")
        axes2[i].set_ylabel("Longitudinal Force Fx (N)")
        axes2[i].grid(True, linestyle='--', alpha=0.5)
        axes2[i].set_xlim(-0.1, 1.1) # 限制显示主要范围
        plt.colorbar(sc, ax=axes2[i], label="Vertical Load Fz (N)")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "mechanics_longitudinal_comparison.png"), dpi=150)
    plt.close()
    print(f"✅ Saved mechanics comparison plots to {output_dir}")

# ------------------------------
# 3. 主程序
# ------------------------------
if __name__ == "__main__":
    # 支持两种加载方式，默认优先读取预处理后的缓存文件
    data_cache_path = "./data/combined_tire_data.pkl"
    
    # 方式 1: 直接读取预处理后的数据集 (默认)
    if os.path.exists(data_cache_path):
        print(f"📂 Loading preprocessed data from cache: {data_cache_path}")
        combined_df = pd.read_pickle(data_cache_path)
    else:
        # 方式 2: 列出 hist_dirs, 调用预处理函数进行实时处理
        print("⚠️ Cache not found. Loading from raw simulation directories...")
        hist_dirs = [
            r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260329_192653",
            r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260329_134701",
        ]
        
        all_dfs = []
        for hist_dir in hist_dirs:
            if os.path.exists(hist_dir):
                df_util = load_all_wheels_data(hist_dir)
                if not df_util.empty:
                    all_dfs.append(df_util)
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
        else:
            combined_df = pd.DataFrame()

    if not combined_df.empty:
        # 执行统计分析
        statistical_analysis(combined_df)
    else:
        print("❌ No data found for analysis.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from analyze_friction_utilization import load_all_wheels_data

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
    dt = 0.001 # 默认 1ms，或者根据数据计算
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

    # 1.2 绘制直方图分布
    plot_histograms(data, output_dir)
    
    # 1.3 绘制力学特性散点图
    plot_mechanics_scatters(data, output_dir)

def plot_histograms(data, output_dir):
    """
    分析每个轮胎的垂向力、横向力、纵向力、质心侧偏角、后轮纵向滑移率的分布直方图
    """
    wheels = ["L1", "R1", "L2", "R2"]
    wheel_labels = ["Front Left", "Front Right", "Rear Left", "Rear Right"]
    
    # 1. 轮胎力分布 (Fx, Fy, Fz)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle("Tire Forces Distribution (Fx, Fy, Fz)", fontsize=20)
    
    for i, w in enumerate(wheels):
        # Fx
        axes[0, i].hist(data[f"fx_{w}"], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, i].set_title(f"{wheel_labels[i]} Fx")
        axes[0, i].set_xlabel("Longitudinal Force (N)")
        
        # Fy
        axes[1, i].hist(data[f"fy_{w}"], bins=50, color='salmon', edgecolor='black', alpha=0.7)
        axes[1, i].set_title(f"{wheel_labels[i]} Fy")
        axes[1, i].set_xlabel("Lateral Force (N)")
        
        # Fz
        axes[2, i].hist(data[f"fz_{w}"], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[2, i].set_title(f"{wheel_labels[i]} Fz")
        axes[2, i].set_xlabel("Vertical Force (N)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "tire_forces_histogram.png"), dpi=150)
    plt.close()

    # 2. 质心侧偏角 (Beta) 和 后轮纵向滑移率 (Slip)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Vehicle Beta and Rear Wheel Slip Distribution", fontsize=18)
    
    # 质心侧偏角 (转换为度)
    axes[0].hist(data["beta"] * (180.0 / np.pi), bins=50, color='plum', edgecolor='black', alpha=0.7)
    axes[0].set_title("Vehicle Sideslip Angle (Beta)")
    axes[0].set_xlabel("Beta (deg)")
    
    # 后轮滑移率 L2
    axes[1].hist(data["slip_L2"], bins=50, color='orange', edgecolor='black', alpha=0.7)
    axes[1].set_title("Rear Left Wheel Slip (Kappa_L2)")
    axes[1].set_xlabel("Longitudinal Slip")
    
    # 后轮滑移率 R2
    axes[2].hist(data["slip_R2"], bins=50, color='orange', edgecolor='black', alpha=0.7)
    axes[2].set_title("Rear Right Wheel Slip (Kappa_R2)")
    axes[2].set_xlabel("Longitudinal Slip")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "beta_slip_histogram.png"), dpi=150)
    plt.close()

def plot_mechanics_scatters(data, output_dir):
    """
    同一种路面系数下：
    - 所有轮胎横向力 vs 轮胎侧偏角 (Colormap: Fz)
    - 后轮纵向力 vs 滑移率 (Colormap: Fz)
    """
    unique_mu = data["mu_road"].unique()
    wheels = ["L1", "R1", "L2", "R2"]
    rear_wheels = ["L2", "R2"]
    
    for mu in unique_mu:
        mu_data = data[data["mu_road"] == mu]
        
        # 创建画布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f"Tire Mechanics Analysis (Road Mu = {mu})", fontsize=18)
        
        # 1. 横向力 vs 侧偏角 (所有轮胎合并)
        all_alpha = pd.concat([mu_data[f"alpha_{w}"] for w in wheels])
        all_fy = pd.concat([mu_data[f"fy_{w}"] for w in wheels])
        all_fz = pd.concat([mu_data[f"fz_{w}"] for w in wheels])
        
        sc1 = ax1.scatter(all_alpha * (180.0 / np.pi), all_fy, c=all_fz, 
                          s=10, alpha=0.6, cmap='viridis')
        ax1.set_title("Lateral Force vs Slip Angle (All Wheels)")
        ax1.set_xlabel("Slip Angle (deg)")
        ax1.set_ylabel("Lateral Force Fy (N)")
        ax1.grid(True, linestyle='--', alpha=0.5)
        cbar1 = plt.colorbar(sc1, ax=ax1)
        cbar1.set_label("Vertical Load Fz (N)")
        
        # 2. 后轮纵向力 vs 滑移率
        rear_slip = pd.concat([mu_data[f"slip_{w}"] for w in rear_wheels])
        rear_fx = pd.concat([mu_data[f"fx_{w}"] for w in rear_wheels])
        rear_fz = pd.concat([mu_data[f"fz_{w}"] for w in rear_wheels])
        
        sc2 = ax2.scatter(rear_slip, rear_fx, c=rear_fz, 
                          s=15, alpha=0.6, cmap='plasma')
        ax2.set_title("Longitudinal Force vs Slip Ratio (Rear Wheels)")
        ax2.set_xlabel("Longitudinal Slip Kappa")
        ax2.set_ylabel("Longitudinal Force Fx (N)")
        ax2.grid(True, linestyle='--', alpha=0.5)
        cbar2 = plt.colorbar(sc2, ax=ax2)
        cbar2.set_label("Vertical Load Fz (N)")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f"tire_mechanics_mu_{mu:.2f}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ Saved mechanics scatter plot for mu={mu} to {save_path}")

# ------------------------------
# 3. 主程序
# ------------------------------
if __name__ == "__main__":
    # 需要分析的数据集路径 (与之前保持一致)
    hist_dirs = [
        r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260324_202921",
        r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260313_093433"
    ]
    
    all_dfs = []
    for hist_dir in hist_dirs:
        if os.path.exists(hist_dir):
            df_util = load_all_wheels_data(hist_dir)
            if not df_util.empty:
                all_dfs.append(df_util)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # 执行统计分析
        statistical_analysis(combined_df)
    else:
        print("No data found for analysis.")

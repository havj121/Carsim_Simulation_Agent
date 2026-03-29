import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tire_data_preprocessing import load_all_wheels_data

# ------------------------------
# 2. 摩擦圆绘图函数
# ------------------------------
def plot_friction_circles(data, output_dir="./Tire_Model/Friction_Analysis"):
    """
    根据不同的摩擦系数 mu_road 分组绘图
    """
    if data.empty:
        print("No data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # 按照 mu_road 分组
    unique_mu = data["mu_road"].unique()
    print(f"Unique friction coefficients found: {unique_mu}")

    for mu in unique_mu:
        mu_data = data[data["mu_road"] == mu]
        
        # 创建一个 Figure，包含 4 个子图 (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f"Friction Utilization Distribution (Road Mu = {mu})", fontsize=16)
        
        wheels = ["L1", "R1", "L2", "R2"]
        wheel_names = {"L1": "Front Left", "R1": "Front Right", "L2": "Rear Left", "R2": "Rear Right"}
        
        for idx, w in enumerate(wheels):
            ax = axes[idx // 2, idx % 2]
            
            ux = mu_data[f"util_x_{w}"]
            uy = mu_data[f"util_y_{w}"]
            util_comb = mu_data[f"util_comb_{w}"]
            
            # 绘制散点，使用 colormap，根据综合利用系数设置颜色
            # 设置颜色条范围在 0 到 1.0 之间
            sc = ax.scatter(ux, uy, c=util_comb, s=8, alpha=0.7, cmap='viridis', 
                            vmin=0, vmax=1.0, label='Simulation Data')
            
            # 绘制摩擦边界圆 (Radius = mu)
            theta = np.linspace(0, 2*np.pi, 200)
            x_circle = mu * np.cos(theta)
            y_circle = mu * np.sin(theta)
            ax.plot(x_circle, y_circle, 'r--', lw=2, label=f'Friction Limit (mu={mu})')
            
            # 绘图修饰
            ax.set_title(f"Wheel: {wheel_names[w]}")
            ax.set_xlabel("Longitudinal Utilization (Fx/Fz)")
            ax.set_ylabel("Lateral Utilization (Fy/Fz)")
            ax.set_xlim(-mu*1.5, mu*1.5)
            ax.set_ylim(-mu*1.5, mu*1.5)
            ax.set_aspect('equal', 'box')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # 添加颜色条
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Friction Utilization (Combined)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f"friction_circle_mu_{mu:.2f}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ Saved friction circle plot for mu={mu} to {save_path}")

def plot_axle_friction_circles(data, output_dir="./Tire_Model/Friction_Analysis"):
    """
    绘制前后轴的摩擦利用分析图 (每张图包含 2 个子图: 前轴, 后轴)
    """
    if data.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    unique_mu = data["mu_road"].unique()

    for mu in unique_mu:
        mu_data = data[data["mu_road"] == mu]
        
        # 创建一个 Figure，包含 2 个子图 (1x2)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f"Axle Friction Utilization (Road Mu = {mu})", fontsize=16)
        
        axle_keys = ["front", "rear"]
        axle_names = {"front": "Front Axle", "rear": "Rear Axle"}
        
        for idx, axle in enumerate(axle_keys):
            ax = axes[idx]
            
            ux = mu_data[f"util_x_{axle}"]
            uy = mu_data[f"util_y_{axle}"]
            util_comb = mu_data[f"util_comb_{axle}"]
            
            # 绘制散点，使用 colormap，根据综合利用系数设置颜色
            # 设置颜色条范围在 0 到 1.0 之间
            sc = ax.scatter(ux, uy, c=util_comb, s=8, alpha=0.7, cmap='plasma', 
                            vmin=0, vmax=1.0, label='Simulation Data')
            
            # 绘制摩擦边界圆
            theta = np.linspace(0, 2*np.pi, 200)
            x_circle = mu * np.cos(theta)
            y_circle = mu * np.sin(theta)
            ax.plot(x_circle, y_circle, 'r--', lw=2, label=f'Friction Limit (mu={mu})')
            
            ax.set_title(f"Axle: {axle_names[axle]}")
            ax.set_xlabel("Longitudinal Utilization (Fx/Fz)")
            ax.set_ylabel("Lateral Utilization (Fy/Fz)")
            ax.set_xlim(-mu*1.5, mu*1.5)
            ax.set_ylim(-mu*1.5, mu*1.5)
            ax.set_aspect('equal', 'box')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # 添加颜色条
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Friction Utilization (Combined)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f"axle_friction_mu_{mu:.2f}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ Saved axle friction circle plot for mu={mu} to {save_path}")

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
            if not os.path.exists(hist_dir):
                print(f"Warning: Directory {hist_dir} not found. Skipping.")
                continue
            
            # 加载并计算
            df_util = load_all_wheels_data(hist_dir)
            if not df_util.empty:
                all_dfs.append(df_util)
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            print(f"Merged datasets. Total points: {len(combined_df)}")
        else:
            combined_df = pd.DataFrame()
    
    if not combined_df.empty:
        # 绘图：单轮
        plot_friction_circles(combined_df)
        
        # 绘图：前后轴
        plot_axle_friction_circles(combined_df)
    else:
        print("❌ No data found for analysis.")
        
    print("Analysis script finished.")

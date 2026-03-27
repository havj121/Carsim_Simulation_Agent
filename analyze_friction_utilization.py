import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from carsim_agent import CarsimAgent

# ------------------------------
# 1. 数据加载与处理逻辑 (参考 TireModel_Exptanh.py)
# ------------------------------
def load_all_wheels_data(hist_dir):
    """
    加载历史数据，并计算四轮的摩擦利用系数:
    util_x = Fx / Fz
    util_y = Fy / Fz
    """
    required_vars = [
        "Time", "Vx", "Ax", "Ay", "AV_Y", "Beta", "Steer_L1", "Steer_L2",
        "Alpha_L1", "Alpha_R1", "Alpha_L2", "Alpha_R2",
        "Kappa_L1", "Kappa_R1", "Kappa_L2", "Kappa_R2",
        "Fz_L1", "Fz_R1", "Fz_L2", "Fz_R2",
        "Fy_L1", "Fy_L2", "Fy_R1", "Fy_R2",
        "Fx_L1", "Fx_L2", "Fx_R1", "Fx_R2",
        "AVy_L1", "AVy_R1", "AVy_L2", "AVy_R2"
    ]
    
    print(f"Loading historical data from: {hist_dir}...")
    batch_results = CarsimAgent.load_historical_data(hist_dir, required_vars=required_vars)
    
    all_rows = []
    for res in batch_results:
        if res.get("status") != "success" or not res.get("extracted_data"):
            continue
        
        df = pd.DataFrame(res["extracted_data"])
        if len(df) == 0:
            continue
            
        # 获取当前工况的路面摩擦系数 mu
        mu_road = res.get("config", {}).get("MU_ROAD_CONSTANT(1)", 1.0)

        # 1. 提取物理参数用于计算 (参考 TireModel_Exptanh.py)
        g = 9.81
        vehicle_param = {"a": 1.04, "b": 1.56, "r_eff": 0.287}
        speed = df["Vx"].values / 3.6
        beta = df["Beta"].values * (np.pi / 180.0)
        r = df["AV_Y"].values * (np.pi / 180.0)
        delta = ((df["Steer_L1"].values + df["Steer_L2"].values) / 2.0) * (np.pi / 180.0)
        
        # 增加 epsilon 防止除零
        Vx_safe = np.maximum(speed * np.cos(beta), 1e-6)
        V = speed / np.cos(beta + 1e-12)
        alpha_f_calc = np.arctan2(V * np.sin(beta) + vehicle_param["a"] * r, Vx_safe) - delta
        alpha_r_calc = np.arctan2(V * np.sin(beta) - vehicle_param["b"] * r, Vx_safe)
        
        Ux = V * np.cos(beta)
        Vy = V * np.sin(beta)
        U_long_f = np.maximum(Ux * np.cos(delta) + (Vy + vehicle_param["a"] * r) * np.sin(delta), 1e-6)
        U_long_r = np.maximum(Ux, 1e-6)
        
        omega_L1 = df["AVy_L1"].values * (2.0 * np.pi / 60.0)
        omega_R1 = df["AVy_R1"].values * (2.0 * np.pi / 60.0)
        omega_L2 = df["AVy_L2"].values * (2.0 * np.pi / 60.0)
        omega_R2 = df["AVy_R2"].values * (2.0 * np.pi / 60.0)
        
        omega_f = 0.5 * (omega_L1 + omega_R1)
        omega_r = 0.5 * (omega_L2 + omega_R2)
        sigma_f_calc = vehicle_param["r_eff"] * omega_f / U_long_f - 1.0
        sigma_r_calc = vehicle_param["r_eff"] * omega_r / U_long_r - 1.0
        
        # 计算轴综合滑移率
        kappa_f_calc = np.sqrt(np.tan(alpha_f_calc)**2 + sigma_f_calc**2 + 1e-9)
        kappa_r_calc = np.sqrt(np.tan(alpha_r_calc)**2 + sigma_r_calc**2 + 1e-9)

        # 提取车轮滑移数据
        alpha_L1 = df["Alpha_L1"].values * (np.pi / 180.0)
        alpha_R1 = df["Alpha_R1"].values * (np.pi / 180.0)
        alpha_L2 = df["Alpha_L2"].values * (np.pi / 180.0)
        alpha_R2 = df["Alpha_R2"].values * (np.pi / 180.0)
        kappa_L1 = df["Kappa_L1"].values
        kappa_R1 = df["Kappa_R1"].values
        kappa_L2 = df["Kappa_L2"].values
        kappa_R2 = df["Kappa_R2"].values
        
        # 计算各轮综合滑移率
        kappa_L1_calc = np.sqrt(np.tan(alpha_L1)**2 + kappa_L1**2 + 1e-9)
        kappa_R1_calc = np.sqrt(np.tan(alpha_R1)**2 + kappa_R1**2 + 1e-9)
        kappa_L2_calc = np.sqrt(np.tan(alpha_L2)**2 + kappa_L2**2 + 1e-9)
        kappa_R2_calc = np.sqrt(np.tan(alpha_R2)**2 + kappa_R2**2 + 1e-9)

        # 提取各个车轮的数据并计算利用系数
        wheels = ["L1", "R1", "L2", "R2"]
        for i in range(len(df)):
            row = {"mu_road": mu_road, "Time": df["Time"].iloc[i]}
            
            # 1. 各个车轮的力
            fz_l1, fz_r1 = df["Fz_L1"].iloc[i], df["Fz_R1"].iloc[i]
            fz_l2, fz_r2 = df["Fz_L2"].iloc[i], df["Fz_R2"].iloc[i]
            fx_l1, fx_r1 = df["Fx_L1"].iloc[i], df["Fx_R1"].iloc[i]
            fx_l2, fx_r2 = df["Fx_L2"].iloc[i], df["Fx_R2"].iloc[i]
            fy_l1, fy_r1 = df["Fy_L1"].iloc[i], df["Fy_R1"].iloc[i]
            fy_l2, fy_r2 = df["Fy_L2"].iloc[i], df["Fy_R2"].iloc[i]

            # 2. 计算轴力
            fz_f = fz_l1 + fz_r1
            fz_r = fz_l2 + fz_r2
            fx_f = fx_l1 + fx_r1
            fx_r = fx_l2 + fx_r2
            fy_f = fy_l1 + fy_r1
            fy_r = fy_l2 + fy_r2

            # 3. 计算利用系数 (增加 epsilon 防止除零)
            row["util_x_L1"] = fx_l1 / max(abs(fz_l1), 1e-3)
            row["util_y_L1"] = fy_l1 / max(abs(fz_l1), 1e-3)
            row["util_x_R1"] = fx_r1 / max(abs(fz_r1), 1e-3)
            row["util_y_R1"] = fy_r1 / max(abs(fz_r1), 1e-3)
            row["util_x_L2"] = fx_l2 / max(abs(fz_l2), 1e-3)
            row["util_y_L2"] = fy_l2 / max(abs(fz_l2), 1e-3)
            row["util_x_R2"] = fx_r2 / max(abs(fz_r2), 1e-3)
            row["util_y_R2"] = fy_r2 / max(abs(fz_r2), 1e-3)

            # 各轮综合滑移率
            row["kappa_L1"] = kappa_L1_calc[i]
            row["kappa_R1"] = kappa_R1_calc[i]
            row["kappa_L2"] = kappa_L2_calc[i]
            row["kappa_R2"] = kappa_R2_calc[i]

            # 4. 计算轴利用系数
            row["util_x_front"] = fx_f / max(abs(fz_f), 1e-3)
            row["util_y_front"] = fy_f / max(abs(fz_f), 1e-3)
            row["util_x_rear"] = fx_r / max(abs(fz_r), 1e-3)
            row["util_y_rear"] = fy_r / max(abs(fz_r), 1e-3)

            # 轴综合滑移率
            row["kappa_front"] = kappa_f_calc[i]
            row["kappa_rear"] = kappa_r_calc[i]

            all_rows.append(row)
            
    data = pd.DataFrame(all_rows)
    print(f"Total points collected: {len(data)}")
    return data

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
            kappa = mu_data[f"kappa_{w}"]
            
            # 绘制散点，使用 colormap，根据 kappa 设置颜色
            sc = ax.scatter(ux, uy, c=kappa, s=8, alpha=0.7, cmap='viridis', label='Simulation Data')
            
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
            cbar.set_label('Combined Slip Ratio (kappa)')

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
            kappa = mu_data[f"kappa_{axle}"]
            
            # 绘制散点，使用 colormap，根据 kappa 设置颜色
            sc = ax.scatter(ux, uy, c=kappa, s=8, alpha=0.7, cmap='plasma', label='Simulation Data')
            
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
            cbar.set_label('Combined Slip Ratio (kappa)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f"axle_friction_mu_{mu:.2f}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ Saved axle friction circle plot for mu={mu} to {save_path}")

# ------------------------------
# 3. 主程序
# ------------------------------
if __name__ == "__main__":
    # 使用与 TireModel_Exptanh.py 相同的默认目录
    hist_dir = r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260313_093433"
    
    if not os.path.exists(hist_dir):
        print(f"Warning: Directory {hist_dir} not found. Please modify the 'hist_dir' variable in the script.")
    else:
        # 加载并计算
        df_util = load_all_wheels_data(hist_dir)
        
        # 绘图：单轮
        plot_friction_circles(df_util)
        
        # 绘图：前后轴
        plot_axle_friction_circles(df_util)
        
    print("Analysis script finished.")

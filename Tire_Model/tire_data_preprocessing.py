import numpy as np
import pandas as pd
import os
import sys

# 配置路径，确保能找到 carsim_agent 模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from carsim_agent import CarsimAgent

# ------------------------------
# 轮胎数据预处理函数
# ------------------------------
def load_all_wheels_data(hist_dir):
    """
    加载历史数据，并进行预处理：
    1. 计算四轮摩擦利用系数
    2. 修正轮胎力符号：横向力 Fy 乘以 -1 (使其与侧偏角同号)
    3. 补充物理量用于后续分析
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
    
    print(f"Loading and preprocessing data from: {hist_dir}...")
    batch_results = CarsimAgent.load_historical_data(hist_dir, required_vars=required_vars)
    
    # 1. 提取物理参数
    vehicle_param = {"m": 1323.9, "a": 1.04, "b": 1.56, "h_cg": 0.54, "r_eff": 0.287}
    g = 9.81
    
    all_rows = []
    for res in batch_results:
        if res.get("status") != "success" or not res.get("extracted_data"):
            continue
        
        df = pd.DataFrame(res["extracted_data"])
        if len(df) == 0:
            continue
            
        # 获取当前工况的路面摩擦系数 mu
        mu_road = res.get("config", {}).get("MU_ROAD_CONSTANT(1)", 1.0)

        # 1. 提取物理参数
        speed = df["Vx"].values / 3.6
        beta = df["Beta"].values * (np.pi / 180.0)
        r = df["AV_Y"].values * (np.pi / 180.0)
        ax = df["Ax"].values * g
        delta = ((df["Steer_L1"].values + df["Steer_L2"].values) / 2.0) * (np.pi / 180.0)
        
        Vx_safe = np.maximum(speed * np.cos(beta), 1e-6)
        V = speed / np.cos(beta + 1e-12)
        
        # 计算轴侧偏角和滑移率 (用于轴利用率分析)
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
        
        # 组合滑移率 (Combined Slip Ratio)
        comb_kappa_f_calc = np.sqrt(np.tan(alpha_f_calc)**2 + sigma_f_calc**2 + 1e-9)
        comb_kappa_r_calc = np.sqrt(np.tan(alpha_r_calc)**2 + sigma_r_calc**2 + 1e-9)

        # 提取车轮原始数据
        alpha_L1 = df["Alpha_L1"].values * (np.pi / 180.0)
        alpha_R1 = df["Alpha_R1"].values * (np.pi / 180.0)
        alpha_L2 = df["Alpha_L2"].values * (np.pi / 180.0)
        alpha_R2 = df["Alpha_R2"].values * (np.pi / 180.0)
        kappa_L1 = df["Kappa_L1"].values
        kappa_R1 = df["Kappa_R1"].values
        kappa_L2 = df["Kappa_L2"].values
        kappa_R2 = df["Kappa_R2"].values
        
        # 组合滑移率 (Combined Slip Ratio)
        comb_kappa_L1 = np.sqrt(np.tan(alpha_L1)**2 + kappa_L1**2 + 1e-9)
        comb_kappa_R1 = np.sqrt(np.tan(alpha_R1)**2 + kappa_R1**2 + 1e-9)
        comb_kappa_L2 = np.sqrt(np.tan(alpha_L2)**2 + kappa_L2**2 + 1e-9)
        comb_kappa_R2 = np.sqrt(np.tan(alpha_R2)**2 + kappa_R2**2 + 1e-9)

        # 计算轴荷分配 (根据加速度 ax)
        m, a, b, h_cg = vehicle_param["m"], vehicle_param["a"], vehicle_param["b"], vehicle_param["h_cg"]
        L = a + b
        Fz_f_calc = (m * g * b - m * ax * h_cg) / L
        Fz_r_calc = (m * g * a + m * ax * h_cg) / L

        for i in range(len(df)):
            row = {"mu_road": mu_road, "Time": df["Time"].iloc[i], "Vx_kmh": df["Vx"].iloc[i], "speed": speed[i]}
            
            # 1. 提取力并应用符号反转 (Fy * -1)
            fz_l1, fz_r1 = df["Fz_L1"].iloc[i], df["Fz_R1"].iloc[i]
            fz_l2, fz_r2 = df["Fz_L2"].iloc[i], df["Fz_R2"].iloc[i]
            fx_l1, fx_r1 = df["Fx_L1"].iloc[i], df["Fx_R1"].iloc[i]
            fx_l2, fx_r2 = df["Fx_L2"].iloc[i], df["Fx_R2"].iloc[i]
            
            # 横向力乘以 -1，使其与侧偏角同号
            fy_l1, fy_r1 = df["Fy_L1"].iloc[i] * -1.0, df["Fy_R1"].iloc[i] * -1.0
            fy_l2, fy_r2 = df["Fy_L2"].iloc[i] * -1.0, df["Fy_R2"].iloc[i] * -1.0

            # 2. 计算轴力
            fz_f_act = fz_l1 + fz_r1
            fz_r_act = fz_l2 + fz_r2
            fx_f = fx_l1 + fx_r1
            fx_r = fx_l2 + fx_r2
            fy_f = fy_l1 + fy_r1
            fy_r = fy_l2 + fy_r2

            # 3. 计算利用系数 (过滤低载荷)
            fz_min_threshold = 500.0
            def calc_util(f, fz):
                return f / fz if abs(fz) > fz_min_threshold else 0.0

            row["util_x_L1"], row["util_y_L1"] = calc_util(fx_l1, fz_l1), calc_util(fy_l1, fz_l1)
            row["util_x_R1"], row["util_y_R1"] = calc_util(fx_r1, fz_r1), calc_util(fy_r1, fz_r1)
            row["util_x_L2"], row["util_y_L2"] = calc_util(fx_l2, fz_l2), calc_util(fy_l2, fz_l2)
            row["util_x_R2"], row["util_y_R2"] = calc_util(fx_r2, fz_r2), calc_util(fy_r2, fz_r2)

            # 综合利用系数
            for w in ["L1", "R1", "L2", "R2"]:
                row[f"util_comb_{w}"] = np.sqrt(row[f"util_x_{w}"]**2 + row[f"util_y_{w}"]**2)

            # 4. 补充物理量
            row["fz_L1"], row["fz_R1"] = fz_l1, fz_r1
            row["fz_L2"], row["fz_R2"] = fz_l2, fz_r2
            row["fx_L1"], row["fx_R1"] = fx_l1, fx_r1
            row["fx_L2"], row["fx_R2"] = fx_l2, fx_r2
            row["fy_L1"], row["fy_R1"] = fy_l1, fy_r1
            row["fy_L2"], row["fy_R2"] = fy_l2, fy_r2
            
            row["alpha_L1"], row["alpha_R1"] = alpha_L1[i], alpha_R1[i]
            row["alpha_L2"], row["alpha_R2"] = alpha_L2[i], alpha_R2[i]
            
            # 纵向滑移率 (Longitudinal Slip)
            row["kappa_L1"], row["kappa_R1"] = kappa_L1[i], kappa_R1[i]
            row["kappa_L2"], row["kappa_R2"] = kappa_L2[i], kappa_R2[i]
            
            # 综合滑移率 (Combined Slip)
            row["comb_kappa_L1"], row["comb_kappa_R1"] = comb_kappa_L1[i], comb_kappa_R1[i]
            row["comb_kappa_L2"], row["comb_kappa_R2"] = comb_kappa_L2[i], comb_kappa_R2[i]
            
            row["beta"] = beta[i]
            row["yaw_rate"] = r[i]

            # 轴级数据 (用于建模)
            row["alpha_f"] = alpha_f_calc[i]
            row["alpha_r"] = alpha_r_calc[i]
            row["sigma_f"] = sigma_f_calc[i]
            row["sigma_r"] = sigma_r_calc[i]
            row["Fz_f"] = Fz_f_calc[i]
            row["Fz_r"] = Fz_r_calc[i]
            row["Fy_f"] = fy_f
            row["Fy_r"] = fy_r
            row["Fx_f"] = fx_f
            row["Fx_r"] = fx_r
            row["mu_Fz_f"] = mu_road * Fz_f_calc[i]
            row["mu_Fz_r"] = mu_road * Fz_r_calc[i]
            
            # 实测轴级数据
            row["alpha_f_act"] = 0.5 * (alpha_L1[i] + alpha_R1[i])
            row["alpha_r_act"] = 0.5 * (alpha_L2[i] + alpha_R2[i])
            row["kappa_f_act"] = 0.5 * (kappa_L1[i] + kappa_R1[i])
            row["kappa_r_act"] = 0.5 * (kappa_L2[i] + kappa_R2[i])
            row["Fz_f_act"] = fz_f_act
            row["Fz_r_act"] = fz_r_act

            # 轴利用率
            row["util_x_front"], row["util_y_front"] = calc_util(fx_f, fz_f_act), calc_util(fy_f, fz_f_act)
            row["util_x_rear"], row["util_y_rear"] = calc_util(fx_r, fz_r_act), calc_util(fy_r, fz_r_act)
            row["util_comb_front"] = np.sqrt(row["util_x_front"]**2 + row["util_y_front"]**2)
            row["util_comb_rear"] = np.sqrt(row["util_x_rear"]**2 + row["util_y_rear"]**2)
            
            row["comb_kappa_front"], row["comb_kappa_rear"] = comb_kappa_f_calc[i], comb_kappa_r_calc[i]

            all_rows.append(row)
            
    data = pd.DataFrame(all_rows)
    print(f"Total points collected: {len(data)}")
    return data

if __name__ == "__main__":
    # 1. 定义需要合并的数据集路径，2个数据集
    hist_dirs = [
        r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260329_192653",
        # r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260329_135507",
        r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260329_134701",
        # r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260324_202921"
    ]
    
    all_dfs = []
    total_samples = 0
    for hist_dir in hist_dirs:
        if os.path.exists(hist_dir):
            df_batch = load_all_wheels_data(hist_dir)
            if not df_batch.empty:
                all_dfs.append(df_batch)
                print(f"Loaded {len(df_batch)} points from {os.path.basename(hist_dir)}")
                total_samples += len(df_batch)
    
    if all_dfs:
        # 2. 合并数据
        combined_data = pd.concat(all_dfs, ignore_index=True)
        
        # 3. 创建保存路径并保存为 pickle 文件 (方便保留 DataFrame 类型信息)
        save_dir = "./data"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "combined_tire_data.pkl")
        
        combined_data.to_pickle(save_path)
        print(f"\n✅ All data processed and saved to: {save_path}")
        print(f"Total samples: {len(combined_data)}")
    else:
        print("❌ No data found to process.")

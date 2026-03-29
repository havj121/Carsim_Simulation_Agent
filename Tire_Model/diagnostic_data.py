import pandas as pd
import numpy as np
import os
import sys

# 配置路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from tire_data_preprocessing import load_all_wheels_data

if __name__ == "__main__":
    hist_dirs = [
        r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260329_135507",
        r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260329_134701",
    ]
    
    all_dfs = []
    for hist_dir in hist_dirs:
        if os.path.exists(hist_dir):
            df = load_all_wheels_data(hist_dir)
            if not df.empty:
                all_dfs.append(df)
    
    data = pd.concat(all_dfs, ignore_index=True)
    print(f"Total data: {len(data)}")
    
    # Front axle check
    mask_f = (data["alpha_f"] * data["Fy_f"] >= 0)
    print(f"Front axle valid: {mask_f.sum()} / {len(data)}")
    
    # Rear axle check
    mask_r_sign = (data["alpha_r"] * data["Fy_r"] >= 0)
    data_r_temp = data[mask_r_sign]
    print(f"Rear axle sign consistency: {len(data_r_temp)} / {len(data)}")
    
    mask_speed = (data_r_temp["Vx_kmh"] >= 10.0)
    mask_slip = (np.abs(data_r_temp["sigma_r"]) <= 1.2)
    data_r_final = data_r_temp[mask_speed & mask_slip]
    print(f"Rear axle final valid: {len(data_r_final)} / {len(data)}")
    print(f"Max sigma_r: {data_r_temp['sigma_r'].abs().max()}")
    print(f"Min Vx_kmh: {data_r_temp['Vx_kmh'].min()}")

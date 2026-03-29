import numpy as np
import pandas as pd
import os
import sys

# 配置路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from tire_data_preprocessing import load_all_wheels_data

def deep_analyze_anomalies(data):
    if data.empty:
        print("No data to analyze.")
        return

    wheels = ["L1", "R1", "L2", "R2"]
    wheel_names = {"L1": "Front Left", "R1": "Front Right", "L2": "Rear Left", "R2": "Rear Right"}
    
    print("\n" + "="*80)
    print("🔍 DEEP ANALYSIS OF TIRE DATA ANOMALIES")
    print("="*80)

    for w in wheels:
        print(f"\n>>> Analyzing Wheel {wheel_names[w]} ({w})")
        
        # 1. 分析侧偏角与横向力符号相反 (Alpha vs Fy)
        # 注意：预处理已经将 Fy 乘以 -1，使其与 Alpha 标准情况下同号
        # 所以这里的“符号相反”指的是 (alpha * fy < 0)
        alpha = data[f"alpha_{w}"]
        fy = data[f"fy_{w}"]
        fz = data[f"fz_{w}"]
        
        # 过滤极小值
        mask_valid = (np.abs(alpha) > 1e-4) & (np.abs(fy) > 1.0)
        alpha_v = alpha[mask_valid]
        fy_v = fy[mask_valid]
        fz_v = fz[mask_valid]
        
        opposite_mask = (alpha_v * fy_v < 0)
        opp_count = opposite_mask.sum()
        total_valid = len(alpha_v)
        opp_ratio = (opp_count / total_valid) * 100 if total_valid > 0 else 0
        
        print(f"--- [Alpha vs Fy Sign Anomaly] ---")
        print(f"  Ratio: {opp_ratio:.2f}% ({opp_count}/{total_valid})")
        
        if opp_count > 0:
            opp_fz = fz_v[opposite_mask]
            opp_alpha = alpha_v[opposite_mask]
            opp_fy = fy_v[opposite_mask]
            
            print(f"  Characteristics:")
            print(f"    - Avg Vertical Load (Fz): {opp_fz.mean():.1f} N (Overall Avg: {fz_v.mean():.1f} N)")
            print(f"    - Fz < 1000N Ratio: {(opp_fz < 1000).sum() / opp_count * 100:.1f}%")
            print(f"    - Avg Alpha (Abs): {np.abs(opp_alpha).mean():.4f} rad")
            print(f"    - Avg Fy (Abs): {np.abs(opp_fy).mean():.1f} N")
            print(f"    - Scenario: Low Fz or Zero-crossing transients.")

        # 2. 分析滑移率异常大 (Kappa > 1.0 or extremely large)
        # CarSim 中的 Kappa 定义通常是 (Rw*w - Vx)/Vx，在 Vx 极小时会非常大
        kappa = data[f"kappa_{w}"]
        
        large_kappa_mask = (np.abs(kappa) > 1.0)
        large_kappa_count = large_kappa_mask.sum()
        total_points = len(data)
        large_kappa_ratio = (large_kappa_count / total_points) * 100
        
        print(f"--- [Extremely Large Kappa (>1.0)] ---")
        print(f"  Ratio: {large_kappa_ratio:.2f}% ({large_kappa_count}/{total_points})")
        
        if large_kappa_count > 0:
            l_kappa = kappa[large_kappa_mask]
            l_fz = fz[large_kappa_mask]
            l_fx = data[f"fx_{w}"][large_kappa_mask]
            
            # 统计超过 100 的极端值
            extreme_mask = (np.abs(kappa) > 100)
            extreme_count = extreme_mask.sum()
            
            print(f"  Characteristics:")
            print(f"    - Max Kappa: {np.abs(l_kappa).max():.2f}")
            print(f"    - Extreme Kappa (>100) Count: {extreme_count}")
            print(f"    - Avg Fz during Large Kappa: {l_fz.mean():.1f} N")
            print(f"    - Fz < 500N Ratio: {(l_fz < 500).sum() / large_kappa_count * 100:.1f}%")
            print(f"    - Scenario: Wheel lock/spin or very low vehicle speed (Vx -> 0).")

if __name__ == "__main__":
    hist_dirs = [
        r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260329_135507",
        r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260329_134701"
    ]
    
    all_dfs = []
    for hist_dir in hist_dirs:
        if os.path.exists(hist_dir):
            df = load_all_wheels_data(hist_dir)
            if not df.empty:
                all_dfs.append(df)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        deep_analyze_anomalies(combined_df)
    else:
        print("No data found.")

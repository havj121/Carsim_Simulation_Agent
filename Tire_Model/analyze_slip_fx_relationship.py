import numpy as np
import pandas as pd
import os
import sys

# 配置路径，确保能找到 analyze_friction_utilization 模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from tire_data_preprocessing import load_all_wheels_data

def analyze_slip_fx_signs(data):
    """
    分析所有轮胎纵向滑移率 (kappa) 与 纵向力 (fx) 的符号关系
    """
    if data.empty:
        print("No data to analyze.")
        return

    wheels = ["L1", "R1", "L2", "R2"]
    wheel_names = {"L1": "Front Left", "R1": "Front Right", "L2": "Rear Left", "R2": "Rear Right"}
    
    print("\n" + "="*60)
    print("📈 TIRE SLIP RATIO (Kappa) VS LONGITUDINAL FORCE (Fx) SIGN RELATIONSHIP")
    print("="*60)
    
    for w in wheels:
        slip = data[f"kappa_{w}"]
        fx = data[f"fx_{w}"]
        
        # 过滤掉数值非常小的点（认为是 0，避免符号误判）
        mask = (np.abs(slip) > 1e-4) & (np.abs(fx) > 1.0)
        slip_f = slip[mask]
        fx_f = fx[mask]
        
        if len(slip_f) == 0:
            print(f"Wheel {wheel_names[w]}: No significant data points.")
            continue
            
        # 计算符号是否相同 (slip * fx > 0)
        same_sign = (slip_f * fx_f > 0).sum()
        opposite_sign = (slip_f * fx_f < 0).sum()
        total = len(slip_f)
        
        same_percent = (same_sign / total) * 100
        opp_percent = (opposite_sign / total) * 100
        
        print(f"Wheel {wheel_names[w]} ({w}):")
        print(f"  - Total significant points: {total}")
        print(f"  - Opposite Sign (Slip > 0, Fx < 0 or vice versa): {opposite_sign} ({opp_percent:.2f}%)")
        print(f"  - Same Sign (Slip > 0, Fx > 0 or Slip < 0, Fx < 0): {same_sign} ({same_percent:.2f}%)")
        
        # 结论判断
        if opp_percent > 95:
            conclusion = "🔴 NEGATIVELY CORRELATED"
        elif same_percent > 95:
            conclusion = "🟢 POSITIVELY CORRELATED"
        else:
            conclusion = "🟡 MIXED RELATIONSHIP"
            
        print(f"  - Conclusion: {conclusion}\n")

if __name__ == "__main__":
    hist_dirs = [
        r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260329_135507"
    ]
    
    all_dfs = []
    for hist_dir in hist_dirs:
        if os.path.exists(hist_dir):
            df = load_all_wheels_data(hist_dir)
            if not df.empty:
                all_dfs.append(df)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        analyze_slip_fx_signs(combined_df)
    else:
        print("No data found for analysis.")

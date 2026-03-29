import numpy as np
import matplotlib.pyplot as plt
import os

# 创建保存结果的目录
OUTPUT_DIR = "./Tire_Model/Sensitivity_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Magic Formula (Pacejka) 模型定义
def magic_formula(alpha, B, C, D, E):
    """
    alpha: 侧偏角 (rad)
    B: 刚度因子, C: 形状因子, D: 峰值因子, E: 曲率因子
    """
    return D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))))

# 2. ExpTanh 模型定义
def exp_tanh_model(alpha, a1, a2, a3, a4, a5):
    """
    a1: 均衡位置, a2: 峰值因子, a3: 饱和区衰减控制, a4: 初始斜率控制, a5: 中心位置
    """
    return a1 + a2 * np.tanh(a4 * (alpha - a5)) * np.exp(-a3 * np.abs(alpha))

# 设置侧偏角范围 (0 到 0.5 rad)
alpha_range = np.linspace(0, 0.5, 500)

# 基准参数设置
mf_base = {'B': 10, 'C': 1.9, 'D': 1.0, 'E': 0.97}
et_base = {'a1': 0.0, 'a2': 1.0, 'a3': 0.5, 'a4': 10.0, 'a5': 0.0}

fluctuations = [-0.1, -0.05, 0.0, 0.05, 0.1]
labels = ["-10%", "-5%", "Base", "+5%", "+10%"]
colors = ['red', 'orange', 'black', 'green', 'blue']
styles = ['--', '--', '-', '--', '--']

def analyze_sensitivity(model_func, base_params, model_name):
    """
    对模型的每一个因子进行灵敏度分析并绘图
    """
    param_names = list(base_params.keys())
    
    for param_to_vary in param_names:
        plt.figure(figsize=(10, 6))
        
        # 计算基准线用于波动大小分析
        base_curve = model_func(alpha_range, **base_params)
        
        for f, label, color, style in zip(fluctuations, labels, colors, styles):
            # 复制基准参数
            current_params = base_params.copy()
            # 对指定因子进行波动
            if f == 0:
                current_val = current_params[param_to_vary]
            else:
                current_val = current_params[param_to_vary] * (1 + f)
            
            current_params[param_to_vary] = current_val
            
            # 计算预测值
            y_pred = model_func(alpha_range, **current_params)
            
            # 计算输出结果的波动比例 (以最大偏差相对于基准最大值的比例表示)
            if f == 0:
                out_fluct_str = ""
            else:
                max_base = np.max(np.abs(base_curve))
                if max_base > 1e-6:
                    # 计算最大偏差比例
                    out_fluct = (np.max(np.abs(y_pred - base_curve)) / max_base) * 100
                    out_fluct_str = f", Out Var: {out_fluct:.2f}%"
                else:
                    out_fluct_str = ""

            # 绘图
            plt.plot(alpha_range, y_pred, color=color, linestyle=style, 
                     label=f"{label} ({current_val:.4f}){out_fluct_str}", lw=2 if f==0 else 1.5)

        plt.title(f'Sensitivity Analysis: {model_name} - Parameter {param_to_vary}', fontsize=14)
        plt.xlabel('Slip Angle (rad)', fontsize=12)
        plt.ylabel('Lateral Force Fy', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        # 保存图片
        save_path = os.path.join(OUTPUT_DIR, f"{model_name}_{param_to_vary}_sensitivity.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved sensitivity plot for {model_name} parameter {param_to_vary} to {save_path}")

def compare_sensitivity(mf_param, et_param):
    """
    在一张图中对比 Magic Formula 和 ExpTanh 模型中对应因子的灵敏度
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # 1. Magic Formula 部分
    mf_curve_base = magic_formula(alpha_range, **mf_base)
    for f, label, color, style in zip(fluctuations, labels, colors, styles):
        current_params = mf_base.copy()
        current_val = current_params[mf_param] * (1 + f) if f != 0 else current_params[mf_param]
        current_params[mf_param] = current_val
        y_pred = magic_formula(alpha_range, **current_params)
        
        # 计算波动比例
        out_fluct_str = ""
        if f != 0:
            max_base = np.max(np.abs(mf_curve_base))
            out_fluct = (np.max(np.abs(y_pred - mf_curve_base)) / max_base) * 100
            out_fluct_str = f", Out Var: {out_fluct:.2f}%"
            
        ax1.plot(alpha_range, y_pred, color=color, linestyle=style, 
                 label=f"{label} ({current_val:.3f}){out_fluct_str}", lw=2 if f==0 else 1.5)
    
    ax1.set_title(f'Magic Formula Sensitivity: {mf_param}', fontsize=14)
    ax1.set_xlabel('Slip Angle (rad)', fontsize=12)
    ax1.set_ylabel('Lateral Force Fy', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='best', fontsize='small')

    # 2. ExpTanh 部分
    et_curve_base = exp_tanh_model(alpha_range, **et_base)
    for f, label, color, style in zip(fluctuations, labels, colors, styles):
        current_params = et_base.copy()
        current_val = current_params[et_param] * (1 + f) if f != 0 else current_params[et_param]
        current_params[et_param] = current_val
        y_pred = exp_tanh_model(alpha_range, **current_params)
        
        # 计算波动比例
        out_fluct_str = ""
        if f != 0:
            max_base = np.max(np.abs(et_curve_base))
            out_fluct = (np.max(np.abs(y_pred - et_curve_base)) / max_base) * 100
            out_fluct_str = f", Out Var: {out_fluct:.2f}%"
            
        ax2.plot(alpha_range, y_pred, color=color, linestyle=style, 
                 label=f"{label} ({current_val:.3f}){out_fluct_str}", lw=2 if f==0 else 1.5)
    
    ax2.set_title(f'ExpTanh Model Sensitivity: {et_param}', fontsize=14)
    ax2.set_xlabel('Slip Angle (rad)', fontsize=12)
    ax2.set_ylabel('Lateral Force Fy', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='best', fontsize='small')

    plt.suptitle(f'Comparison of Model Sensitivity: MF {mf_param} vs ExpTanh {et_param}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图片
    save_name = f"Compare_{mf_param}_vs_{et_param}_sensitivity.png"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved comparison plot: {save_path}")

if __name__ == "__main__":
    print("🚀 Starting sensitivity analysis for Magic Formula...")
    analyze_sensitivity(magic_formula, mf_base, "Magic_Formula")
    
    print("\n🚀 Starting sensitivity analysis for ExpTanh Model...")
    analyze_sensitivity(exp_tanh_model, et_base, "ExpTanh_Model")
    
    print("\n🚀 Starting comparison of model sensitivity...")
    # 1. 比较 E (Curvature) 与 a3 (Decay)
    compare_sensitivity('E', 'a3')
    
    # 2. 比较 B (Stiffness) 与 a4 (Initial Slope)
    compare_sensitivity('B', 'a4')
    
    print(f"\n✨ Analysis complete. Results are in the '{OUTPUT_DIR}' directory.")

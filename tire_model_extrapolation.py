import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 外推能力分析 (Extrapolation Analysis) 的原理与实现方法：
# 1. 原理：
#    - 轮胎模型通常在有限的侧偏角范围内（如 0 到 0.3 rad）进行实验或训练。
#    - 外推能力是指当侧偏角超过训练范围（如进入 0.5 到 2.0 rad 的极限工况）时，模型的物理一致性表现。
#    - Magic Formula (MF) 基于正弦函数，当其曲率因子 E > 1 时，在大侧偏角下可能出现非物理的剧烈振荡甚至反向力。
#    - ExpTanh 模型基于双曲正切与指数衰减的乘积，具有天然的物理一致性：其在大侧偏角下会平滑地趋向于 a1 (通常为0) 或渐进衰减，不会出现振荡。
# 2. 实现方法：
#    - 定义一个远超正常饱和区的侧偏角范围 (0 到 2.0 rad)。
#    - 设定基准参数 (Normal) 和由于神经网络预测偏差或极限工况导致的扰动参数 (Drifted/Extreme)。
#    - 对比观察两条模型曲线在外推区 (Extrapolation Zone) 的走势，特别是是否存在正负号翻转或高频波动。
# ==============================================================================
# 创建保存结果的目录
OUTPUT_DIR = "./Tire_Model/Extrapolation_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def magic_formula(alpha, B, C, D, E):
    """
    标准 Pacejka Magic Formula
    alpha: 侧偏角 (rad)
    B: 刚度因子, C: 形状因子, D: 峰值因子, E: 曲率因子
    """
    return D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))))

def exp_tanh_model(alpha, a1, a2, a3, a4, a5):
    """
    最新的 ExpTanh 模型
    a1: 均衡位置, a2: 峰值因子, a3: 饱和区衰减控制, a4: 初始斜率控制, a5: 中心位置
    """
    return a1 + a2 * np.tanh(a4 * (alpha - a5)) * np.exp(-a3 * np.abs(alpha))

# 设定滑移角范围：从 0 到 1.5 rad (远超正常的饱和区)
alpha_extrap = np.linspace(0, 1.5, 1000)

# 1. Magic Formula 基准参数与外推风险参数
mf_params = [10, 1.9, 1.0, 1]
mf_extreme = [10, 1.9, 1.0, 1.2]  # E > 1, 模拟外推时的正弦波振荡风险

# 2. ExpTanh 基准参数与外推鲁棒参数 (更新为 5 参数模型)
# a1=0, a2=1.0, a3=0.5, a4=10, a5=0
et_params = [0.0, 1.0, 0.5, 10.0, 0.0]
et_extreme = [0.0, 1.0, 0.1, 10.0, 0.0] # a3 变小，衰减变慢，但依然保持物理单调性

def compare_extrapolation():
    plt.figure(figsize=(14, 7))

    # --- Magic Formula 外推表现 ---
    plt.subplot(1, 2, 1)
    plt.plot(alpha_extrap, magic_formula(alpha_extrap, *mf_params), 'k', lw=2, label='Normal MF (E=0.97)')
    plt.plot(alpha_extrap, magic_formula(alpha_extrap, *mf_extreme), 'r--', lw=2, label='Extreme MF (E=1.25, Risky)')
    
    # 标注外推区
    plt.axvspan(0.3, 1.5, color='gray', alpha=0.1, label='Extrapolation Zone (> 0.3 rad)')
    plt.axhline(0, color='black', lw=1)
    plt.title("Magic Formula: Extrapolation Risk\n(Potential non-physical oscillation at high angles)", fontsize=13)
    plt.xlabel("Slip Angle (rad)", fontsize=11)
    plt.ylabel("Lateral Force (Normalized)", fontsize=11)
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # --- ExpTanh 外推表现 ---
    plt.subplot(1, 2, 2)
    plt.plot(alpha_extrap, exp_tanh_model(alpha_extrap, *et_params), 'k', lw=2, label='Normal ExpTanh (a3=0.5)')
    plt.plot(alpha_extrap, exp_tanh_model(alpha_extrap, *et_extreme), 'b--', lw=2, label='Extreme ExpTanh (a3=0.1, Robust)')
    
    plt.axvspan(0.3, 1.5, color='gray', alpha=0.1, label='Extrapolation Zone (> 0.3 rad)')
    plt.axhline(0, color='black', lw=1)
    plt.title("ExpTanh: Robust Extrapolation\n(Always smooth decay, no oscillation)", fontsize=13)
    plt.xlabel("Slip Angle (rad)", fontsize=11)
    plt.ylabel("Lateral Force (Normalized)", fontsize=11)
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Tire Model Extrapolation Capability Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存分析图
    save_path = os.path.join(OUTPUT_DIR, "tire_model_extrapolation_analysis.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Extrapolation analysis plot saved as {save_path}")

if __name__ == "__main__":
    compare_extrapolation()
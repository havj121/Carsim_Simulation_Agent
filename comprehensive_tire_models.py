import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from carsim_agent import CarsimAgent

# ==========================================
# 1. 理论模型 (Theoretical Models)
# ==========================================

class TheoreticalModel:
    """理论模型基类"""
    def forward(self, x, params):
        raise NotImplementedError

class LinearModel(TheoreticalModel):
    """1. 线性模型: Fy = Ca * alpha"""
    def __init__(self):
        self.name = "Linear"
        self.param_count = 1 # Ca

    def forward(self, alpha, params):
        Ca = params[:, 0:1]
        return Ca * alpha

class MagicFormulaModel(TheoreticalModel):
    """
    2. Magic Formula (Pacejka): Fy = D*sin(C*atan(B*x - E*(B*x - atan(B*x))))
    B: 刚度因子, C: 形状因子, D: 峰值因子, E: 曲率因子
    """
    def __init__(self):
        self.name = "MagicFormula"
        self.param_count = 4 # B, C, D, E

    def forward(self, alpha, params):
        B, C, D, E = params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4]
        # 限制参数物理范围 (防止训练时数值溢出)
        B = torch.abs(B) + 1e-3
        C = torch.clamp(C, 1.0, 2.0)
        D = torch.abs(D)
        return D * torch.sin(C * torch.atan(B * alpha - E * (B * alpha - torch.atan(B * alpha))))

class ExpTanhModel(TheoreticalModel):
    """
    3. ExpTanh 模型: Fy = a1 + a2 * tanh(a4 * (alpha - a5)) * exp(-a3 * |alpha|)
    a1: 均衡位置, a2: 峰值因子, a3: 饱和区衰减控制, a4: 初始斜率控制, a5: 中心位置
    """
    def __init__(self):
        self.name = "ExpTanh"
        self.param_count = 5 # a1, a2, a3, a4, a5

    def forward(self, alpha, params):
        a1, a2, a3, a4, a5 = params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4], params[:, 4:5]
        a3 = torch.exp(a3) # 保证衰减系数非负
        a4 = torch.exp(a4) # 保证斜率系数非负
        return a1 + a2 * torch.tanh(a4 * (alpha - a5)) * torch.exp(-a3 * torch.abs(alpha))

# ==========================================
# 2. 数据驱动模型 (Data-Driven Models)
# ==========================================

class PureNNTireModel(nn.Module):
    """4. 纯神经网络模型"""
    def __init__(self, input_dim=4):
        super(PureNNTireModel, self).__init__()
        self.name = "PureNN"
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 64), # +1 是因为包含核心输入 alpha (或 tan alpha)
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, alpha, feat):
        x = torch.cat([alpha, feat], dim=1)
        return self.net(x)

class PINNTireModel(nn.Module):
    """
    5. PINN 模型 (物理增强神经网络)
    特点：神经网络预测理论模型的参数，实现数据驱动与物理约束的结合。
    能够灵活选择不同的理论模型。
    """
    def __init__(self, theory_model: TheoreticalModel, input_dim=4):
        super(PINNTireModel, self).__init__()
        self.theory = theory_model
        self.name = f"PINN_{theory_model.name}"
        
        # MLP 用于预测理论模型的参数
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, theory_model.param_count)
        )

    def forward(self, alpha, feat):
        # 1. 神经网络输出物理参数
        params = self.mlp(feat)
        # 2. 物理参数带入理论公式计算最终出力
        return self.theory.forward(alpha, params)

# ==========================================
# 3. 数据处理与验证逻辑
# ==========================================

def get_data_from_hist(hist_dir, tire_id="L1"):
    """复用历史数据读取逻辑"""
    target_vars = [
        "Time", "Alpha_L1", "Alpha_R1", "Alpha_L2", "Alpha_R2",
        "Fz_L1", "Fz_R1", "Fz_L2", "Fz_R2", "Fy_L1", "Fy_R1", "Fy_L2", "Fy_R2",
        "Vx", "Beta", "AV_Y"
    ]
    batch_results = CarsimAgent.load_historical_data(hist_dir, required_vars=target_vars)
    
    X_alpha, X_feat, Y_fy, MU = [], [], [], []
    
    for res in batch_results:
        if res.get("status") != "success" or not res.get("extracted_data"):
            continue
        df = pd.DataFrame(res.get("extracted_data"))
        mu = res.get("config", {}).get("MU_ROAD_CONSTANT(1)", 0.8)
        
        # 提取指定轮胎数据
        alpha = df[f'Alpha_{tire_id}'].values * (np.pi / 180.0)
        fz = df[f'Fz_{tire_id}'].values
        fy = df[f'Fy_{tire_id}'].values
        vx = df['Vx'].values / 3.6
        beta = df['Beta'].values * (np.pi / 180.0)
        r = df['AV_Y'].values * (np.pi / 180.0)
        
        for i in range(len(df)):
            X_alpha.append([np.tan(alpha[i])]) # 使用 tan(alpha) 作为核心输入
            X_feat.append([vx[i], beta[i], r[i], fz[i]])
            Y_fy.append([fy[i]])
            MU.append(mu)
            
    return np.array(X_alpha), np.array(X_feat), np.array(Y_fy), np.array(MU)

def train_and_verify(model, X_alpha, X_feat, Y_fy, MU, epochs=500):
    print(f"\n🚀 Training Model: {model.name}")
    
    alpha_t = torch.FloatTensor(X_alpha)
    feat_t = torch.FloatTensor(X_feat)
    fy_t = torch.FloatTensor(Y_fy)
    mu_t = torch.FloatTensor(MU).view(-1, 1)
    fz_t = feat_t[:, 3:4]
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    history = []
    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        pred = model(alpha_t, feat_t)
        loss_data = criterion(pred, fy_t)
        
        # 物理偏差损失 (PINN 特有)
        loss_phys = torch.tensor(0.0)
        if "PINN" in model.name:
            # 物理约束：|Fy| <= mu * Fz
            loss_phys = 0.1 * torch.mean(torch.relu(torch.abs(pred) - mu_t * torch.abs(fz_t)))
            
        total_loss = loss_data + loss_phys
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Total Loss: {total_loss.item():.4f} (MSE: {loss_data.item():.4f})")
        history.append(total_loss.item())
    
    # 可视化 3D 结果
    model.eval()
    with torch.no_grad():
        fy_pred = model(alpha_t, feat_t).numpy()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_alpha, X_feat[:, 3], Y_fy, s=5, alpha=0.3, label='Actual')
    ax.scatter(X_alpha, X_feat[:, 3], fy_pred, s=5, alpha=0.3, color='red', label='Predicted')
    ax.set_title(f"3D Verification: {model.name}")
    ax.set_xlabel("tan(alpha)")
    ax.set_ylabel("Fz [N]")
    ax.set_zlabel("Fy [N]")
    ax.legend()
    plt.savefig(f"verification_{model.name}.png")
    plt.close()
    return history

if __name__ == "__main__":
    hist_dir = r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260313_093433"
    
    if not os.path.exists(hist_dir):
        print(f"❌ Hist dir not found: {hist_dir}")
    else:
        # 1. 获取数据
        X_a, X_f, Y_f, MU_val = get_data_from_hist(hist_dir, tire_id="L1")
        
        # --- 测试各种模型 ---
        
        # 示例 A: 纯神经网络模型
        nn_model = PureNNTireModel()
        train_and_verify(nn_model, X_a, X_f, Y_f, MU_val)
        
        # 示例 B: PINN + ExpTanh (灵活组合)
        pinn_et = PINNTireModel(theory_model=ExpTanhModel())
        train_and_verify(pinn_et, X_a, X_f, Y_f, MU_val)
        
        # 示例 C: PINN + MagicFormula (灵活组合)
        pinn_mf = PINNTireModel(theory_model=MagicFormulaModel())
        train_and_verify(pinn_mf, X_a, X_f, Y_f, MU_val)
        
        # 示例 D: 合并所有轮胎数据训练统一模型 (以 PINN_ExpTanh 为例)
        print("\n--- Training Unified Model (L1 + R1) ---")
        X_a_r1, X_f_r1, Y_f_r1, MU_r1 = get_data_from_hist(hist_dir, tire_id="R1")
        X_a_all = np.vstack([X_a, X_a_r1])
        X_f_all = np.vstack([X_f, X_f_r1])
        Y_f_all = np.vstack([Y_f, Y_f_r1])
        MU_all = np.concatenate([MU_val, MU_r1])
        
        unified_model = PINNTireModel(theory_model=ExpTanhModel())
        train_and_verify(unified_model, X_a_all, X_f_all, Y_f_all, MU_all)
        
        print("\n✨ All model tests completed. Plots saved.")

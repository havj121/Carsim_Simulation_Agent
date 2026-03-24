import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from carsim_agent import CarsimAgent

# ==========================================
# 1. Neural-ExpTanh 模型定义 
# ==========================================
class NeuralExpTanhTire(nn.Module):
    def __init__(self, tire_id="L1", input_vars=["Vx", "Beta", "AV_Y", "Fz"], output_vars=["Fy"]):
        super(NeuralExpTanhTire, self).__init__()
        self.tire_id = tire_id
        # 默认输入变量（MLP输入，不包含核心 tan(alpha)）
        self.input_vars = input_vars 
        # 默认输出变量（每个输出对应一组 5 个 ExpTanh 系数）
        self.output_vars = output_vars
        
        # 物理约束相关参数
        self.mu_var = "MU" # 附着系数
        self.fz_var = "Fz" # 垂直载荷
        self.alpha_var = "Alpha" #  slip angle
        
        # 自动配置 MLP 维度
        self._build_model()
        
        # 训练所需数据
        self.X_z = None     # tan(alpha)
        self.X_feat = None  # MLP 输入特征
        self.Y_targets = None # 输出目标
        self.MU = None      
        self.Fz = None      
        
        # 训练历史
        self.history = {'epoch': [], 'loss': [], 'loss_mse': [], 'loss_phys': []}

    def _build_model(self):
        """根据输入输出变量配置，构建 MLP 网络"""
        in_dim = len(self.input_vars)
        # 每个输出变量需要 5 个系数来描述其 ExpTanh 曲线
        out_dim = len(self.output_vars) * 5
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, out_dim) 
        )

    def set_vars(self, input_vars=None, output_vars=None):
        """动态修改输入/输出变量，并重新构建模型"""
        if input_vars is not None:
            self.input_vars = input_vars
        if output_vars is not None:
            self.output_vars = output_vars
        self._build_model()
        print(f"🔄 Model reconfigured for {self.tire_id}: Inputs={self.input_vars}, Outputs={self.output_vars}")

    def forward(self, z, feat):
        """
        z: tan(alpha) - shape (N, 1)
        feat: 其他特征输入 - shape (N, in_dim)
        """
        all_coeffs = self.mlp(feat) # shape (N, num_outputs * 5)
        
        predictions = []
        for i in range(len(self.output_vars)):
            # 提取该输出对应的 5 个系数
            start_idx = i * 5
            c1 = all_coeffs[:, start_idx : start_idx+1]
            c2 = all_coeffs[:, start_idx+1 : start_idx+2]
            c3 = torch.exp(all_coeffs[:, start_idx+2 : start_idx+3]) # 非负
            c4 = torch.exp(all_coeffs[:, start_idx+3 : start_idx+4]) # 非负
            c5 = all_coeffs[:, start_idx+4 : start_idx+5]
            
            # ExpTanh 公式
            term_exp = torch.exp(-c3 * torch.abs(z))
            term_tanh = torch.tanh(c4 * (z - c5))
            pred = c1 + c2 * term_exp * term_tanh
            predictions.append(pred)
            
        return torch.cat(predictions, dim=1) # shape (N, num_outputs)

    def set_data(self, X_z, X_feat, Y_targets, MU, Fz):
        self.X_z = X_z
        self.X_feat = X_feat
        self.Y_targets = Y_targets
        self.MU = MU
        self.Fz = Fz

    def train_model(self, epochs=5000, lr=0.01, lambda_phys=0.1):
        if self.X_z is None:
            raise ValueError(f"No data set for tire {self.tire_id}")
            
        print(f"\n>>> Training model for tire: {self.tire_id} (Samples: {len(self.X_z)})")
        
        # 转 Tensor
        z_tensor = torch.FloatTensor(self.X_z)
        feat_tensor = torch.FloatTensor(self.X_feat)
        y_tensor = torch.FloatTensor(self.Y_targets)
        mu_tensor = torch.FloatTensor(self.MU).view(-1, 1)
        fz_tensor = torch.FloatTensor(self.Fz).view(-1, 1)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs + 1):
            self.train()
            optimizer.zero_grad()
            
            y_pred = self.forward(z_tensor, feat_tensor)
            loss_mse = criterion(y_pred, y_tensor)
            
            # 物理约束：|Fy_pred| <= mu * Fz (仅对 Fy 进行约束，如果 Fy 在输出中)
            loss_phys = torch.tensor(0.0)
            if "Fy" in self.output_vars:
                fy_idx = self.output_vars.index("Fy")
                fy_pred = y_pred[:, fy_idx:fy_idx+1]
                loss_phys = torch.mean(torch.relu(torch.abs(fy_pred) - mu_tensor * torch.abs(fz_tensor)))
            
            loss = loss_mse + lambda_phys *loss_phys
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Tire {self.tire_id} | Epoch {epoch} | Loss: {loss.item():.4f} (MSE: {loss_mse.item():.4f})")
                self.history['epoch'].append(epoch)
                self.history['loss'].append(loss.item())
                self.history['loss_mse'].append(loss_mse.item())
                self.history['loss_phys'].append(loss_phys.item())

# ==========================================
# 2. 数据处理辅助函数
# ==========================================
def process_agent_results(batch_results):
    """
    按轮胎分离数据: {'L1': (X_z, X_feat, Y_fy, MU, Fz, Fx), ...}
    """
    tire_data = {tire: {'X_z': [], 'X_feat': [], 'Fy': [], 'MU': [], 'Fz': [], 'Fx': []} for tire in ['L1', 'R1', 'L2', 'R2']}
    
    for i, res in enumerate(batch_results):
        if res.get("status") != "success" or not res.get("extracted_data"):
            continue
            
        mu = res.get("config", {}).get("MU_ROAD_CONSTANT(1)", 0.8)
        df = pd.DataFrame(res.get("extracted_data"))
        
        for tire in tire_data.keys():
            # 基础建模变量
            required_cols = [f'Alpha_{tire}', f'Fz_{tire}', f'Fy_{tire}', f'Fx_{tire}', 'Vx', 'Beta', 'AV_Y']
            if not all(col in df.columns for col in required_cols):
                continue
                
            alpha_rad = df[f'Alpha_{tire}'].values * (np.pi / 180.0)
            fz = df[f'Fz_{tire}'].values
            fy = df[f'Fy_{tire}'].values
            fx = df[f'Fx_{tire}'].values
            v = df['Vx'].values / 3.6
            beta = df['Beta'].values * (np.pi / 180.0)
            r = df['AV_Y'].values * (np.pi / 180.0)
            z = np.tan(alpha_rad)
            
            for j in range(len(df)):
                tire_data[tire]['X_z'].append([z[j]])
                # 特征组合：[Vx, Beta, AV_Y, Fz]
                tire_data[tire]['X_feat'].append([v[j], beta[j], r[j], fz[j]])
                tire_data[tire]['Fy'].append([fy[j]])
                tire_data[tire]['Fx'].append([fx[j]])
                tire_data[tire]['MU'].append(mu)
                tire_data[tire]['Fz'].append(fz[j])
                
    final_tire_data = {}
    for tire, data in tire_data.items():
        if len(data['X_z']) > 0:
            final_tire_data[tire] = {
                'X_z': np.array(data['X_z']),
                'X_feat': np.array(data['X_feat']),
                'Fy': np.array(data['Fy']),
                'Fx': np.array(data['Fx']),
                'MU': np.array(data['MU']),
                'Fz': np.array(data['Fz'])
            }
    return final_tire_data

# ==========================================
# 3. 运行主流程
# ==========================================
def run_tire_modeling(historical_dir=None, template_path=None, mods_json_path=None):
    """
    获取仿真数据支持两种模式：
    1. 历史数据：提供 historical_dir
    2. 新仿真：提供 template_path 和 mods_json_path
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 建模所需目标变量：Alpha, Fx, Fy, Fz (4 tires) + Vx, Beta, AV_Y
    target_vars = [
        "Time", 
        "Alpha_L1", "Alpha_R1", "Alpha_L2", "Alpha_R2",
        "Fx_L1", "Fx_R1", "Fx_L2", "Fx_R2",
        "Fy_L1", "Fy_R1", "Fy_L2", "Fy_R2",
        "Fz_L1", "Fz_R1", "Fz_L2", "Fz_R2",
        "Vx", "Beta", "AV_Y"
    ]
    
    # 1. 获取仿真数据
    if historical_dir:
        print(f"📂 Loading historical data from: {historical_dir}")
        batch_results = CarsimAgent.load_historical_data(historical_dir, required_vars=target_vars)
    elif template_path and mods_json_path:
        print(f"🚀 Starting new simulation using template: {template_path}")
        agent = CarsimAgent(template_path=template_path)
        with open(mods_json_path, 'r') as f:
            batch_configs = json.load(f)
        batch_results = agent.get_simulation_data(batch_configs, target_vars=target_vars)
    else:
        raise ValueError("Must provide either historical_dir OR (template_path AND mods_json_path)")

    return process_agent_results(batch_results)

def visualize_3d(tire_id, X_z, Fz, Y_true, Y_pred, MU, suffix=""):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*']
    unique_mus = np.unique(MU)
    for idx, mu in enumerate(unique_mus):
        mask = (MU == mu)
        marker = markers[idx % len(markers)]
        ax.scatter(X_z[mask], Fz[mask], Y_true[mask], s=10, alpha=0.4, marker=marker, label=f'CarSim (MU={mu})')
        ax.scatter(X_z[mask], Fz[mask], Y_pred[mask], s=10, alpha=0.4, marker=marker, color='red', label=f'Model (MU={mu})')
    
    ax.set_title(f"Tire {tire_id} 3D Modeling {suffix}")
    ax.set_xlabel("tan(alpha)")
    ax.set_ylabel("Fz [N]")
    ax.set_zlabel("Fy [N]")
    ax.set_xlim([-10, 10])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"tire_model_verification_{tire_id}_{suffix}.png")
    plt.close()

if __name__ == "__main__":
    hist_dir = r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260313_093433"
    
    # 获取数据
    tire_data_dict = run_tire_modeling(historical_dir=hist_dir)
    
    # --- 示例 1: 输入单个轮胎的数据训练单个轮胎模型 ---
    print("\n--- Example 1: Individual Tire Model (L1) ---")
    l1_data = tire_data_dict['L1']
    model_l1 = NeuralExpTanhTire(tire_id="L1")
    model_l1.set_data(l1_data['X_z'], l1_data['X_feat'], l1_data['Fy'], l1_data['MU'], l1_data['Fz'])
    model_l1.train_model(epochs=300)
    
    # 验证可视化
    model_l1.eval()
    with torch.no_grad():
        fy_pred_l1 = model_l1(torch.FloatTensor(l1_data['X_z']), torch.FloatTensor(l1_data['X_feat'])).numpy()
    visualize_3d("L1", l1_data['X_z'], l1_data['Fz'], l1_data['Fy'], fy_pred_l1, l1_data['MU'], suffix="individual")

    # --- 示例 2: 合并所有轮胎数据训练一个统一轮胎模型 ---
    print("\n--- Example 2: Unified Tire Model (All Tires) ---")
    # 合并所有轮胎的数据
    all_X_z = np.vstack([d['X_z'] for d in tire_data_dict.values()])
    all_X_feat = np.vstack([d['X_feat'] for d in tire_data_dict.values()])
    all_Fy = np.vstack([d['Fy'] for d in tire_data_dict.values()])
    all_MU = np.concatenate([d['MU'] for d in tire_data_dict.values()])
    all_Fz = np.concatenate([d['Fz'] for d in tire_data_dict.values()])
    
    model_unified = NeuralExpTanhTire(tire_id="Unified")
    model_unified.set_data(all_X_z, all_X_feat, all_Fy, all_MU, all_Fz)
    model_unified.train_model(epochs=300)
    
    # 验证可视化 (以 L1 数据验证统一模型)
    model_unified.eval()
    with torch.no_grad():
        fy_pred_unified_l1 = model_unified(torch.FloatTensor(l1_data['X_z']), torch.FloatTensor(l1_data['X_feat'])).numpy()
    visualize_3d("L1", l1_data['X_z'], l1_data['Fz'], l1_data['Fy'], fy_pred_unified_l1, l1_data['MU'], suffix="unified")


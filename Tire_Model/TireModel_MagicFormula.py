import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg') # 强制使用非交互式后端，防止在某些环境下挂起
import matplotlib.pyplot as plt

# 配置路径，确保能找到 tire_data_preprocessing 模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from tire_data_preprocessing import load_all_wheels_data

# ------------------------------
# 1. 参数设置
# ------------------------------
g = 9.81
lambda_penalty = 0.01
batch_size = 256
epochs = 100
learning_rate = 1e-3
# 暂时强制使用 CPU 以避免 CUDA 兼容性问题
device = torch.device('cpu')

output_dir = "./Tire_Model/MagicFormula"
os.makedirs(output_dir, exist_ok=True)

vehicle_param = {
    "m": 1323.9,
    "a": 1.04,
    "b": 1.56,
    "h_cg": 0.54,
    "r_eff": 0.287
}

# ------------------------------
# 2. 数据过滤函数 (与 ExpTanh 保持一致)
# ------------------------------
def filter_tire_data(data, axle='front'):
    """
    数据筛选逻辑：
    1. 剔除横向力与侧偏角符号不一致的数据 (所有轴)
    2. 后轴额外剔除：速度 < 10km/h，纵向滑移率 > 10
    """
    if data.empty:
        return data
    
    total_initial = len(data)
    dt = 0.01 # 假设 10ms
    
    # ① 剔除横向力与侧偏角符号不一致的数据
    if axle == 'front':
        mask_sign = (data["alpha_f"] * data["Fy_f"] >= 0)
    else:
        mask_sign = (data["alpha_r"] * data["Fy_r"] >= 0)
        
    rejected_sign = total_initial - mask_sign.sum()
    data_filtered = data[mask_sign].copy()
    
    print(f"\n--- Data Filtering Summary ({axle.capitalize()} Axle) ---")
    print(f"1. Sign Consistency Filter:")
    print(f"   - Rejected: {rejected_sign} points ({rejected_sign/total_initial*100:.2f}%)")
    
    # ② 后轴组合建模额外剔除
    if axle == 'rear':
        total_before_rear = len(data_filtered)
        # 剔除速度 < 10km/h (包括负数)
        mask_speed = (data_filtered["Vx_kmh"] >= 10.0)
        # 剔除纵向滑移率 > 10
        mask_slip = (np.abs(data_filtered["sigma_r"]) <= 10.0)
        
        mask_combined = mask_speed & mask_slip
        rejected_rear = total_before_rear - mask_combined.sum()
        data_filtered = data_filtered[mask_combined].copy()
        
        print(f"2. Rear Axle Specific Filter (Speed < 10km/h or Slip > 10):")
        print(f"   - Rejected: {rejected_rear} points ({rejected_rear/total_initial*100:.2f}%)")

    final_count = len(data_filtered)
    final_duration = final_count * dt
    print(f"Final Data Status:")
    print(f"   - Remaining Points: {final_count}")
    print(f"   - Estimated Duration: {final_duration:.2f} s")
    print(f"   - Total Rejected Ratio: {(total_initial - final_count)/total_initial*100:.2f}%")
    
    return data_filtered.reset_index(drop=True)

# ------------------------------
# 3. Magic Formula 模型定义
# ------------------------------
class MagicFormula(nn.Module):
    """ Magic Formula (Pacejka): F = D * sin(C * arctan(B*z - E*(B*z - arctan(B*z))))
        其中 B, C, D, E 由小型神经网络生成以解耦工况。
    """
    def __init__(self, input_dim, hidden_units=[32, 32], output_dim=4):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        
        # 手动初始化偏置，让模型初始输出合理的参数
        with torch.no_grad():
            self.net[-1].bias.data[0] = 0.0  # D: sigmoid(0)*1.5 = 0.75
            self.net[-1].bias.data[1] = 1.0  # B: softplus(1)*15 = 19.6
            self.net[-1].bias.data[2] = 0.5  # C: softplus(0.5)+1.0 = 1.97
            self.net[-1].bias.data[3] = 0.0  # E: tanh(0) = 0.0

    def forward(self, z, feat, mu_Fz):
        """
        z: 输入滑移量 (alpha 或 kappa) [batch, 1]
        feat: 特征 [batch, input_dim]
        mu_Fz: 物理缩放因子 (mu * Fz) [batch, 1]
        """
        p = self.net(feat)          # [batch, 4]
        
        # 物理参数约束与缩放
        D_scale = torch.sigmoid(p[:, 0:1]) * 1.5           # 峰值倍数 0~1.5
        B = torch.nn.functional.softplus(p[:, 1:2]) * 15.0 # 刚度系数
        C = torch.nn.functional.softplus(p[:, 2:3]) + 1.0  # 形状因子 > 1.0
        E = torch.tanh(p[:, 3:4])                          # 曲率因子 -1~1
        
        # Magic Formula 核心计算
        Bz = B * z
        phi = Bz - E * (Bz - torch.atan(Bz))
        F_coeff = D_scale * torch.sin(C * torch.atan(phi))
        
        # 输出物理力： F = F_coeff * mu_Fz
        return F_coeff * mu_Fz

class CombinedMagicFormula(nn.Module):
    """ 组合滑移模型：总力 Ftot = MagicFormula(kappa)
        Fy = (tan(alpha)/kappa) * Ftot
        Fx = (sigma/kappa) * Ftot
    """
    def __init__(self, input_dim_feat, hidden_units=[16, 16]):
        super().__init__()
        self.tot_model = MagicFormula(input_dim_feat, hidden_units, output_dim=4)

    def forward(self, alpha, sigma, feat, mu_Fz):
        """
        alpha: 侧偏角 [batch, 1]
        sigma: 纵向滑移率 [batch, 1]
        feat: 特征 [batch, input_dim]
        mu_Fz: 物理缩放因子 (mu * Fz) [batch, 1]
        """
        # 综合滑移 kappa = sqrt(tan^2(alpha) + sigma^2)
        tan_alpha = torch.tan(alpha)
        kappa = torch.sqrt(tan_alpha**2 + sigma**2 + 1e-9)
        
        Ftot = self.tot_model(kappa, feat, mu_Fz)
        
        # 按照分量比例分配
        Fy = (tan_alpha / kappa) * Ftot
        Fx = (sigma / kappa) * Ftot
        return Fy, Fx

# ------------------------------
# 4. 训练函数 (与 ExpTanh 保持一致)
# ------------------------------
def train_pure_slip(model, dataloader, optimizer, device, lambda_penalty):
    """ 训练纯滑移模型（前轴侧向物理力）"""
    model.train()
    total_loss, total_mse, total_penalty = 0, 0, 0
    force_scale = 1000.0
    for alpha, feat, mu_Fz, target_Fy in dataloader:
        alpha = alpha.to(device)
        feat = feat.to(device)
        mu_Fz = mu_Fz.to(device)
        target_Fy = target_Fy.to(device)

        pred_Fy = model(alpha, feat, mu_Fz)
        
        # 使用缩放后的 MSE
        mse = nn.functional.mse_loss(pred_Fy / force_scale, target_Fy / force_scale)
        
        # 物理不等式约束
        penalty = torch.mean(torch.relu(torch.abs(pred_Fy) - mu_Fz)**2) / (force_scale**2)
        loss = mse + lambda_penalty * penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse.item()
        total_penalty += penalty.item()
        
    num_batches = len(dataloader)
    return {
        "loss": total_loss / num_batches,
        "mse": total_mse / num_batches,
        "penalty": total_penalty / num_batches
    }

def train_combined_slip(model, dataloader, optimizer, device, lambda_penalty):
    """ 训练组合滑移模型（后轴物理力）"""
    model.train()
    total_loss, total_mse, total_penalty = 0, 0, 0
    force_scale = 1000.0
    for alpha, sigma, feat, mu_Fz, target_Fy, target_Fx in dataloader:
        alpha = alpha.to(device)
        sigma = sigma.to(device)
        feat = feat.to(device)
        mu_Fz = mu_Fz.to(device)
        target_Fy = target_Fy.to(device)
        target_Fx = target_Fx.to(device)

        pred_Fy, pred_Fx = model(alpha, sigma, feat, mu_Fz)
        
        # 损失包含 Fx, Fy 物理力 (使用缩放因子)
        mse = nn.functional.mse_loss(pred_Fy / force_scale, target_Fy / force_scale) + \
              nn.functional.mse_loss(pred_Fx / force_scale, target_Fx / force_scale)
        
        # 总力物理软约束
        Ftot_pred = torch.sqrt(pred_Fy**2 + pred_Fx**2 + 1e-9)
        penalty = torch.mean(torch.relu(Ftot_pred - mu_Fz)**2) / (force_scale**2)
        
        loss = mse + lambda_penalty * penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse.item()
        total_penalty += penalty.item()

    num_batches = len(dataloader)
    return {
        "loss": total_loss / num_batches,
        "mse": total_mse / num_batches,
        "penalty": total_penalty / num_batches
    }

def plot_training_history(history, title, filename):
    """ 绘制训练过程中的损失曲线 """
    epochs_list = range(1, len(history["loss"]) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, history["loss"], label='Total Loss (Total)')
    plt.plot(epochs_list, history["mse"], label='Prediction Error Loss (MSE)', linestyle='--')
    plt.plot(epochs_list, history["penalty_weighted"], label='Penalty Loss (Lambda * Penalty)', linestyle=':')
    
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value (log scale)')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Training history plot saved as {save_path}")

def plot_error_distribution(true_val, pred_val, set_name, model_name, filename):
    """ 绘制模型预测误差百分比的分布直方图 """
    epsilon = 100.0
    error_percent = np.abs(pred_val - true_val) / (np.abs(true_val) + epsilon) * 100.0
    
    plt.figure(figsize=(10, 6))
    plt.hist(error_percent.flatten(), bins=50, color='skyblue', edgecolor='black', alpha=0.7, range=(0, 50))
    plt.axvline(np.mean(error_percent), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(error_percent):.2f}%')
    plt.axvline(np.median(error_percent), color='green', linestyle='dashed', linewidth=1, label=f'Median: {np.median(error_percent):.2f}%')
    
    plt.title(f"Force Error Percentage Distribution - {model_name} ({set_name})")
    plt.xlabel("Error Percentage (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Error distribution plot saved as {save_path}")

# ------------------------------
# 5. 主程序
# ------------------------------
if __name__ == "__main__":
    # 1. 从预处理后的文件加载数据
    data_path = "./data/combined_tire_data.pkl"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found at {data_path}. Please run tire_data_preprocessing.py first.")
        sys.exit()
        
    print(f"📂 Loading preprocessed data from: {data_path}")
    data_full = pd.read_pickle(data_path)
    print(f"Total samples loaded: {len(data_full)}")
    
    # ------------------------------
    # 建模：前轴横向力 Fy_f
    # ------------------------------
    print("\n" + "="*50)
    print("🚀 Training Front Axle Magic Formula Model (Fy_f)...")
    print("="*50)
    
    data_f = filter_tire_data(data_full, axle='front')
    
    feat_f_cols = ['yaw_rate', 'speed', 'beta']
    scaler_f = StandardScaler()
    X_f_feat_scaled = scaler_f.fit_transform(data_f[feat_f_cols]).astype(np.float32)
    
    alpha_f = data_f['alpha_f'].values.reshape(-1, 1).astype(np.float32)
    mu_Fz_f = data_f['mu_Fz_f'].values.reshape(-1, 1).astype(np.float32)
    Fy_f = data_f['Fy_f'].values.reshape(-1, 1).astype(np.float32)
    
    train_idx_f, test_idx_f = train_test_split(np.arange(len(data_f)), test_size=0.2, random_state=42)
    
    train_f_ds = TensorDataset(torch.tensor(alpha_f[train_idx_f]), 
                                torch.tensor(X_f_feat_scaled[train_idx_f]), 
                                torch.tensor(mu_Fz_f[train_idx_f]),
                                torch.tensor(Fy_f[train_idx_f]))
    loader_f_train = DataLoader(train_f_ds, batch_size=batch_size, shuffle=True)
    
    pure_model = MagicFormula(input_dim=len(feat_f_cols)).to(device)
    optimizer_pure = optim.Adam(pure_model.parameters(), lr=learning_rate)
    scheduler_pure = optim.lr_scheduler.ReduceLROnPlateau(optimizer_pure, 'min', patience=20, factor=0.5)
    
    history_pure = {"loss": [], "mse": [], "penalty_weighted": []}
    for epoch in range(epochs):
        res = train_pure_slip(pure_model, loader_f_train, optimizer_pure, device, lambda_penalty)
        history_pure["loss"].append(res["loss"])
        history_pure["mse"].append(res["mse"])
        history_pure["penalty_weighted"].append(res["penalty"] * lambda_penalty)
        scheduler_pure.step(res["loss"])
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {res['loss']:.4f}")
            
    plot_training_history(history_pure, "Front Axle Magic Formula Training", "pure_slip_history.png")
    torch.save(pure_model.state_dict(), os.path.join(output_dir, 'pure_slip_magic_formula_front.pth'))

    # 计算前轴误差分布
    pure_model.eval()
    with torch.no_grad():
        pred_f_train = pure_model(torch.tensor(alpha_f[train_idx_f]).to(device), 
                                  torch.tensor(X_f_feat_scaled[train_idx_f]).to(device),
                                  torch.tensor(mu_Fz_f[train_idx_f]).to(device)).cpu().numpy()
        plot_error_distribution(Fy_f[train_idx_f], pred_f_train, "Train", "Front Lateral", "front_error_train.png")
        
        pred_f_test = pure_model(torch.tensor(alpha_f[test_idx_f]).to(device), 
                                 torch.tensor(X_f_feat_scaled[test_idx_f]).to(device),
                                 torch.tensor(mu_Fz_f[test_idx_f]).to(device)).cpu().numpy()
        plot_error_distribution(Fy_f[test_idx_f], pred_f_test, "Test", "Front Lateral", "front_error_test.png")

    # ------------------------------
    # 建模：后轴组合力 Fy_r, Fx_r
    # ------------------------------
    print("\n" + "="*50)
    print("🚀 Training Rear Axle Magic Formula Combined Model (Fy_r, Fx_r)...")
    print("="*50)
    
    data_r = filter_tire_data(data_full, axle='rear')
    
    feat_r_cols = ['yaw_rate', 'speed']
    scaler_r = StandardScaler()
    X_r_feat_scaled = scaler_r.fit_transform(data_r[feat_r_cols]).astype(np.float32)
    
    alpha_r = data_r['alpha_r'].values.reshape(-1, 1).astype(np.float32)
    sigma_r = data_r['sigma_r'].values.reshape(-1, 1).astype(np.float32)
    mu_Fz_r = data_r['mu_Fz_r'].values.reshape(-1, 1).astype(np.float32)
    Fy_r = data_r['Fy_r'].values.reshape(-1, 1).astype(np.float32)
    Fx_r = data_r['Fx_r'].values.reshape(-1, 1).astype(np.float32)
    
    train_idx_r, test_idx_r = train_test_split(np.arange(len(data_r)), test_size=0.2, random_state=42)
    
    train_r_ds = TensorDataset(torch.tensor(alpha_r[train_idx_r]), torch.tensor(sigma_r[train_idx_r]), 
                                torch.tensor(X_r_feat_scaled[train_idx_r]), 
                                torch.tensor(mu_Fz_r[train_idx_r]),
                                torch.tensor(Fy_r[train_idx_r]), torch.tensor(Fx_r[train_idx_r]))
    loader_r_train = DataLoader(train_r_ds, batch_size=batch_size, shuffle=True)
    
    combined_model = CombinedMagicFormula(input_dim_feat=len(feat_r_cols)).to(device)
    optimizer_combined = optim.Adam(combined_model.parameters(), lr=learning_rate)
    scheduler_combined = optim.lr_scheduler.ReduceLROnPlateau(optimizer_combined, 'min', patience=20, factor=0.5)
    
    history_combined = {"loss": [], "mse": [], "penalty_weighted": []}
    for epoch in range(epochs):
        res = train_combined_slip(combined_model, loader_r_train, optimizer_combined, device, lambda_penalty)
        history_combined["loss"].append(res["loss"])
        history_combined["mse"].append(res["mse"])
        history_combined["penalty_weighted"].append(res["penalty"] * lambda_penalty)
        scheduler_combined.step(res["loss"])
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {res['loss']:.4f}")
            
    plot_training_history(history_combined, "Rear Axle Magic Formula Training", "combined_slip_history.png")
    torch.save(combined_model.state_dict(), os.path.join(output_dir, 'combined_slip_magic_formula_rear.pth'))
    
    # 计算后轴误差分布
    combined_model.eval()
    with torch.no_grad():
        pred_r_fy_train, pred_r_fx_train = combined_model(torch.tensor(alpha_r[train_idx_r]).to(device), 
                                                          torch.tensor(sigma_r[train_idx_r]).to(device), 
                                                          torch.tensor(X_r_feat_scaled[train_idx_r]).to(device),
                                                          torch.tensor(mu_Fz_r[train_idx_r]).to(device))
        pred_r_fy_train, pred_r_fx_train = pred_r_fy_train.cpu().numpy(), pred_r_fx_train.cpu().numpy()
        plot_error_distribution(Fy_r[train_idx_r], pred_r_fy_train, "Train", "Rear Lateral (Fy)", "rear_fy_error_train.png")
        plot_error_distribution(Fx_r[train_idx_r], pred_r_fx_train, "Train", "Rear Longitudinal (Fx)", "rear_fx_error_train.png")
        
        pred_r_fy_test, pred_r_fx_test = combined_model(torch.tensor(alpha_r[test_idx_r]).to(device), 
                                                        torch.tensor(sigma_r[test_idx_r]).to(device), 
                                                        torch.tensor(X_r_feat_scaled[test_idx_r]).to(device),
                                                        torch.tensor(mu_Fz_r[test_idx_r]).to(device))
        pred_r_fy_test, pred_r_fx_test = pred_r_fy_test.cpu().numpy(), pred_r_fx_test.cpu().numpy()
        plot_error_distribution(Fy_r[test_idx_r], pred_r_fy_test, "Test", "Rear Lateral (Fy)", "rear_fy_error_test.png")
        plot_error_distribution(Fx_r[test_idx_r], pred_r_fx_test, "Test", "Rear Longitudinal (Fx)", "rear_fx_error_test.png")

    print("\n🏁 REAR AXLE TRAINING COMPLETE.")
    print("\n✅ All Magic Formula models trained, saved, and evaluation plots generated in", output_dir)

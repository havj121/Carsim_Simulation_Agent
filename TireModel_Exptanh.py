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
import os
from carsim_agent import CarsimAgent

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

vehicle_param = {
    "m": 1323.9,
    "a": 1.04,
    "b": 1.56,
    "h_cg": 0.54,
    "r_eff": 0.287
}

# ------------------------------
# 2. 数据加载与预处理
# ------------------------------
def load_and_preprocess_from_hist(hist_dir):
    required_vars = [
        "Time", "Vx", "Beta", "AV_Y", "Ax",
        "Steer_L1", "Steer_L2",
        "Alpha_L1", "Alpha_L2", "Alpha_R1", "Alpha_R2",
        "Kappa_L1", "Kappa_L2", "Kappa_R1", "Kappa_R2",
        "Fz_L1", "Fz_L2", "Fz_R1", "Fz_R2",
        "Fy_L1", "Fy_L2", "Fy_R1", "Fy_R2",
        "Fx_L1", "Fx_L2", "Fx_R1", "Fx_R2",
        "AVy_L1", "AVy_R1", "AVy_L2", "AVy_R2"
    ]
    batch_results = CarsimAgent.load_historical_data(hist_dir, required_vars=required_vars)
    rows = []
    for res in batch_results:
        if res.get("status") != "success" or not res.get("extracted_data"):
            continue
        df = pd.DataFrame(res["extracted_data"])
        if len(df) == 0:
            continue
        # 获取当前工况的摩擦系数
        mu_road = res.get("config", {}).get("MU_ROAD_CONSTANT(1)", 1.0)
        Vx_kmh = df["Vx"].values
        speed = Vx_kmh / 3.6
        beta = df["Beta"].values * (np.pi / 180.0)
        r = df["AV_Y"].values * (np.pi / 180.0)
        delta = ((df["Steer_L1"].values + df["Steer_L2"].values) / 2.0) * (np.pi / 180.0)
        alpha_L1 = df["Alpha_L1"].values * (np.pi / 180.0)
        alpha_R1 = df["Alpha_R1"].values * (np.pi / 180.0)
        alpha_L2 = df["Alpha_L2"].values * (np.pi / 180.0)
        alpha_R2 = df["Alpha_R2"].values * (np.pi / 180.0)
        kappa_L1 = df["Kappa_L1"].values
        kappa_R1 = df["Kappa_R1"].values
        kappa_L2 = df["Kappa_L2"].values
        kappa_R2 = df["Kappa_R2"].values
        Fz_L1 = df["Fz_L1"].values
        Fz_R1 = df["Fz_R1"].values
        Fz_L2 = df["Fz_L2"].values
        Fz_R2 = df["Fz_R2"].values
        Fy_L1 = df["Fy_L1"].values
        Fy_R1 = df["Fy_R1"].values
        Fy_L2 = df["Fy_L2"].values
        Fy_R2 = df["Fy_R2"].values
        Fx_L1 = df["Fx_L1"].values
        Fx_R1 = df["Fx_R1"].values
        Fx_L2 = df["Fx_L2"].values
        Fx_R2 = df["Fx_R2"].values
        omega_L1 = df["AVy_L1"].values * (2.0 * np.pi / 60.0)
        omega_R1 = df["AVy_R1"].values * (2.0 * np.pi / 60.0)
        omega_L2 = df["AVy_L2"].values * (2.0 * np.pi / 60.0)
        omega_R2 = df["AVy_R2"].values * (2.0 * np.pi / 60.0)
        ax_g = df["Ax"].values
        ax = ax_g * g
        m = vehicle_param["m"]
        a = vehicle_param["a"]
        b = vehicle_param["b"]
        h_cg = vehicle_param["h_cg"]
        r_eff = vehicle_param["r_eff"]
        
        # 增加 epsilon 防止除零
        Vx_safe = np.maximum(speed * np.cos(beta), 1e-6)
        V = speed / np.cos(beta + 1e-12)
        alpha_f_calc = np.arctan2(V * np.sin(beta) + a * r, Vx_safe) - delta
        alpha_r_calc = np.arctan2(V * np.sin(beta) - b * r, Vx_safe)
        Ux = V * np.cos(beta)
        Vy = V * np.sin(beta)
        U_long_f = np.maximum(Ux * np.cos(delta) + (Vy + a * r) * np.sin(delta), 1e-6)
        U_long_r = np.maximum(Ux, 1e-6)
        omega_f = 0.5 * (omega_L1 + omega_R1)
        omega_r = 0.5 * (omega_L2 + omega_R2)
        sigma_f_calc = r_eff * omega_f / U_long_f - 1.0
        sigma_r_calc = r_eff * omega_r / U_long_r - 1.0
        
        alpha_f_act = 0.5 * (alpha_L1 + alpha_R1)
        alpha_r_act = 0.5 * (alpha_L2 + alpha_R2)
        kappa_f_act = 0.5 * (kappa_L1 + kappa_R1)
        kappa_r_act = 0.5 * (kappa_L2 + kappa_R2)
        Fz_f_act = Fz_L1 + Fz_R1
        Fz_r_act = Fz_L2 + Fz_R2
        L = a + b
        Fz_f_calc = (m * g * b - m * ax * h_cg) / L
        Fz_r_calc = (m * g * a + m * ax * h_cg) / L
        Fy_f = Fy_L1 + Fy_R1
        Fy_r = Fy_L2 + Fy_R2
        Fx_f = Fx_L1 + Fx_R1
        Fx_r = Fx_L2 + Fx_R2
        for i in range(len(df)):
            # 💡 重要：检查并修正 alpha 和 Fy 的符号一致性
            # 保证 Fy 和 alpha 同号，以符合 tanh(a2 * z) 模型
            alpha_f_val = alpha_f_calc[i]
            Fy_f_val = Fy_f[i]
            if alpha_f_val * Fy_f_val < 0:
                Fy_f_val = -Fy_f_val # 强制同号

            alpha_r_val = alpha_r_calc[i]
            Fy_r_val = Fy_r[i]
            if alpha_r_val * Fy_r_val < 0:
                Fy_r_val = -Fy_r_val # 强制同号
            
            # 同理处理纵向力
            sigma_r_val = sigma_r_calc[i]
            Fx_r_val = Fx_r[i]
            if sigma_r_val * Fx_r_val < 0:
                Fx_r_val = -Fx_r_val # 强制同号

            rows.append({
                "yaw_rate": r[i],
                "speed": speed[i],
                "beta": beta[i],
                "Fz_f": Fz_f_calc[i],
                "Fz_r": Fz_r_calc[i],
                "alpha_f": alpha_f_val,
                "alpha_r": alpha_r_val,
                "sigma_f": sigma_f_calc[i],
                "sigma_r": sigma_r_val,
                "alpha_f_act": alpha_f_act[i],
                "alpha_r_act": alpha_r_act[i],
                "kappa_f_act": kappa_f_act[i],
                "kappa_r_act": kappa_r_act[i],
                "Fz_f_act": Fz_f_act[i],
                "Fz_r_act": Fz_r_act[i],
                "Fy_f": Fy_f_val,
                "Fy_r": Fy_r_val,
                "Fx_f": Fx_f[i],
                "Fx_r": Fx_r_val,
                "Vx_kmh": Vx_kmh[i],
                "mu_road": mu_road, # 添加摩擦系数
                "mu_Fz_f": mu_road * Fz_f_calc[i], # 新增 mu * Fz_f
                "mu_Fz_r": mu_road * Fz_r_calc[i] # 新增 mu * Fz_r
            })
    data = pd.DataFrame(rows)
    if len(data) == 0:
        return data
    mask = (data["Vx_kmh"] >= 5.0) & (data["speed"] > 0.0)
    data = data.loc[mask].reset_index(drop=True)
    return data
# ------------------------------
# 3. ExpTanh 模型定义
# ------------------------------
class ExpTanh(nn.Module):
    """ ExpTanh 曲线： F = a1 * tanh(a2 * z) * exp(-a3 * |z|)
        其中 a1~a3 由小型神经网络生成以解耦工况。
    """
    def __init__(self, input_dim, hidden_units=[32, 32], output_dim=3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h)) # 添加 BN 层加速收敛
            layers.append(nn.LeakyReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        # 💡 重要：手动初始化偏置，让模型初始输出合理的参数
        with torch.no_grad():
            self.net[-1].bias.data[0] = 0.0  # sigmoid(0) = 0.5
            self.net[-1].bias.data[1] = 1.0  # softplus(1)*10 初始刚度
            self.net[-1].bias.data[2] = -2.0 # softplus(-2) 初始衰减极小

    def forward(self, z, feat, mu_Fz):
        """
        z: 输入滑移量 (alpha 或 kappa) [batch, 1]
        feat: 特征 [batch, input_dim]
        mu_Fz: 物理缩放因子 (mu * Fz) [batch, 1]
        """
        p = self.net(feat)          # [batch, 3]
        
        # 物理参数约束与缩放
        # 现在输出的是无量纲力系数，a1 代表峰值附着系数倍数
        a1 = torch.sigmoid(p[:, 0:1]) * 1.5              # 增加范围 0~1.5 (原 1.1)
        a2 = torch.nn.functional.softplus(p[:, 1:2]) * 15.0 # 增加刚度系数 15 (原 10)
        a3 = torch.nn.functional.softplus(p[:, 2:3])      # 衰减率
        
        # 增加 clamp 防止数值爆炸
        exp_input = -a3 * torch.abs(z)
        exp_term = torch.exp(torch.clamp(exp_input, -20, 0))
        tanh_term = torch.tanh(a2 * z)
        
        F_coeff = a1 * tanh_term * exp_term
        # 💡 直接输出物理力： F = F_coeff * mu_Fz
        return F_coeff * mu_Fz


class CombinedExpTanh(nn.Module):
    """ 组合滑移模型：总力 Ftot = ExpTanh(kappa)
        Fy = (tan(alpha)/kappa) * Ftot
        Fx = (sigma/kappa) * Ftot
    """
    def __init__(self, input_dim_feat, hidden_units=[16, 16]):
        super().__init__()
        self.tot_model = ExpTanh(input_dim_feat, hidden_units, output_dim=3)

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
# 4. 训练函数
# ------------------------------
def train_pure_slip(model, dataloader, optimizer, device, lambda_penalty):
    """ 训练纯滑移模型（前轴侧向物理力）"""
    model.train()
    total_loss, total_mse, total_penalty = 0, 0, 0
    force_scale = 1000.0 # 💡 缩放因子：物理力级太大，除以 1000 使 MSE 在 1~10 数量级
    for alpha, feat, mu_Fz, target_Fy in dataloader:
        alpha = alpha.to(device)
        feat = feat.to(device)
        mu_Fz = mu_Fz.to(device)
        target_Fy = target_Fy.to(device)

        pred_Fy = model(alpha, feat, mu_Fz)
        
        # 💡 使用缩放后的 MSE
        mse = nn.functional.mse_loss(pred_Fy / force_scale, target_Fy / force_scale)
        
        # 物理不等式约束也需要相应缩放
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
        
        # 总力物理软约束也需缩放
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
    plt.plot(epochs_list, history["loss"], label='Total Loss')
    plt.plot(epochs_list, history["mse"], label='MSE Loss', linestyle='--')
    plt.plot(epochs_list, history["penalty_weighted"], label='Lambda * Penalty', linestyle=':')
    
    plt.yscale('log') # 物理力级 MSE 很大，用对数坐标展示更清晰
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value (log scale)')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(filename)
    plt.close()
    print(f"📊 Training history plot saved as {filename}")

# ------------------------------
# 5. 主程序
# ------------------------------
if __name__ == "__main__":
    hist_dir = r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260313_093433"
    if not os.path.exists(hist_dir):
        print(f"Hist dir not found: {hist_dir}")
        exit(0)
        
    data = load_and_preprocess_from_hist(hist_dir)
    print(f"Loaded {len(data)} data points.")
    if len(data) == 0:
        print("No data found. Exiting.")
        exit(0)

    # ---- 1. 特征工程与标准化 ----
    # 前轴特征 feat_f: [r, V, beta]
    # 后轴特征 feat_r: [r, V]
    # mu_Fz 作为物理缩放因子，不参与标准化
    feat_f_cols = ['yaw_rate', 'speed', 'beta']
    feat_r_cols = ['yaw_rate', 'speed']
    
    scaler_f = StandardScaler()
    scaler_r = StandardScaler()
    
    X_f_feat_scaled = scaler_f.fit_transform(data[feat_f_cols]).astype(np.float32)
    X_r_feat_scaled = scaler_r.fit_transform(data[feat_r_cols]).astype(np.float32)
    
    # 提取输入滑移量、物理缩放因子与目标力 (物理量级)
    alpha_f = data['alpha_f'].values.reshape(-1, 1).astype(np.float32)
    Fy_f = data['Fy_f'].values.reshape(-1, 1).astype(np.float32)
    mu_Fz_f = data['mu_Fz_f'].values.reshape(-1, 1).astype(np.float32)
    
    alpha_r = data['alpha_r'].values.reshape(-1, 1).astype(np.float32)
    sigma_r = data['sigma_r'].values.reshape(-1, 1).astype(np.float32)
    Fy_r = data['Fy_r'].values.reshape(-1, 1).astype(np.float32)
    Fx_r = data['Fx_r'].values.reshape(-1, 1).astype(np.float32)
    mu_Fz_r = data['mu_Fz_r'].values.reshape(-1, 1).astype(np.float32)

    # ---- 2. 数据集划分 ----
    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # 构建 DataLoader (目标为物理力)
    train_f_ds = TensorDataset(torch.tensor(alpha_f[train_idx]), 
                               torch.tensor(X_f_feat_scaled[train_idx]), 
                               torch.tensor(mu_Fz_f[train_idx]),
                               torch.tensor(Fy_f[train_idx]))
    test_f_ds = TensorDataset(torch.tensor(alpha_f[test_idx]), 
                              torch.tensor(X_f_feat_scaled[test_idx]), 
                              torch.tensor(mu_Fz_f[test_idx]),
                              torch.tensor(Fy_f[test_idx]))
    loader_f_train = DataLoader(train_f_ds, batch_size=batch_size, shuffle=True)
    loader_f_test = DataLoader(test_f_ds, batch_size=batch_size, shuffle=False)

    train_r_ds = TensorDataset(torch.tensor(alpha_r[train_idx]), torch.tensor(sigma_r[train_idx]), 
                               torch.tensor(X_r_feat_scaled[train_idx]), 
                               torch.tensor(mu_Fz_r[train_idx]),
                               torch.tensor(Fy_r[train_idx]), torch.tensor(Fx_r[train_idx]))
    test_r_ds = TensorDataset(torch.tensor(alpha_r[test_idx]), torch.tensor(sigma_r[test_idx]), 
                              torch.tensor(X_r_feat_scaled[test_idx]), 
                              torch.tensor(mu_Fz_r[test_idx]),
                              torch.tensor(Fy_r[test_idx]), torch.tensor(Fx_r[test_idx]))
    loader_r_train = DataLoader(train_r_ds, batch_size=batch_size, shuffle=True)
    loader_r_test = DataLoader(test_r_ds, batch_size=batch_size, shuffle=False)

    # ---- 3. 模型实例化 ----
    pure_model = ExpTanh(input_dim=len(feat_f_cols)).to(device)
    optimizer_pure = optim.Adam(pure_model.parameters(), lr=learning_rate)
    scheduler_pure = optim.lr_scheduler.ReduceLROnPlateau(optimizer_pure, 'min', patience=20, factor=0.5)

    combined_model = CombinedExpTanh(input_dim_feat=len(feat_r_cols)).to(device)
    optimizer_combined = optim.Adam(combined_model.parameters(), lr=learning_rate)
    scheduler_combined = optim.lr_scheduler.ReduceLROnPlateau(optimizer_combined, 'min', patience=20, factor=0.5)

    # ---- 4. 训练循环 ----
    history_pure = {"loss": [], "mse": [], "penalty_weighted": []}
    print("Training pure slip model (Front Axle Coeff)...")
    for epoch in range(epochs):
        res = train_pure_slip(pure_model, loader_f_train, optimizer_pure, device, lambda_penalty)
        history_pure["loss"].append(res["loss"])
        history_pure["mse"].append(res["mse"])
        history_pure["penalty_weighted"].append(res["penalty"] * lambda_penalty)
        
        scheduler_pure.step(res["loss"])
        
        if (epoch+1) % 20 == 0:
            curr_lr = optimizer_pure.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}, Loss (Scaled MSE): {res['loss']:.4f}, LR: {curr_lr:.6f}")

    plot_training_history(history_pure, "Pure Slip Training Loss (Front Axle)", "pure_slip_history.png")

    history_combined = {"loss": [], "mse": [], "penalty_weighted": []}
    print("\nTraining combined slip model (Rear Axle Coeff)...")
    for epoch in range(epochs):
        res = train_combined_slip(combined_model, loader_r_train, optimizer_combined, device, lambda_penalty)
        history_combined["loss"].append(res["loss"])
        history_combined["mse"].append(res["mse"])
        history_combined["penalty_weighted"].append(res["penalty"] * lambda_penalty)
        
        scheduler_combined.step(res["loss"])
        
        if (epoch+1) % 50 == 0:
            curr_lr = optimizer_combined.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}, Loss (Scaled MSE): {res['loss']:.4f}, LR: {curr_lr:.6f}")

    plot_training_history(history_combined, "Combined Slip Training Loss (Rear Axle)", "combined_slip_history.png")

    # ---- 5. 评估与可视化 (还原为物理力) ----
    pure_model.eval()
    combined_model.eval()
    with torch.no_grad():
        # 前轴预测物理力
        pred_f_phys_torch = pure_model(torch.tensor(alpha_f[test_idx]).to(device), 
                                       torch.tensor(X_f_feat_scaled[test_idx]).to(device),
                                       torch.tensor(mu_Fz_f[test_idx]).to(device))
        pred_f_phys = pred_f_phys_torch.cpu().numpy()
        mse_f_phys = np.mean((pred_f_phys - Fy_f[test_idx])**2)
        print(f"\nPure slip test MSE (Front Physical Force): {mse_f_phys:.4f}")

        # 后轴预测物理力
        pred_r_fy_phys_torch, pred_r_fx_phys_torch = combined_model(torch.tensor(alpha_r[test_idx]).to(device), 
                                                                    torch.tensor(sigma_r[test_idx]).to(device), 
                                                                    torch.tensor(X_r_feat_scaled[test_idx]).to(device),
                                                                    torch.tensor(mu_Fz_r[test_idx]).to(device))
        pred_r_fy_phys = pred_r_fy_phys_torch.cpu().numpy()
        pred_r_fx_phys = pred_r_fx_phys_torch.cpu().numpy()
        
        mse_r_fy_phys = np.mean((pred_r_fy_phys - Fy_r[test_idx])**2)
        mse_r_fx_phys = np.mean((pred_r_fx_phys - Fx_r[test_idx])**2)
        print(f"Combined slip test MSE (Rear Fy Physical): {mse_r_fy_phys:.4f}")
        print(f"Combined slip test MSE (Rear Fx Physical): {mse_r_fx_phys:.4f}")

    # 可视化前轴 (物理力)
    idx_p = np.random.choice(len(test_idx), 500, replace=False)
    plt.figure(figsize=(8,5))
    plt.scatter(alpha_f[test_idx][idx_p], Fy_f[test_idx][idx_p], s=1, label='True (Physical)')
    plt.scatter(alpha_f[test_idx][idx_p], pred_f_phys[idx_p], s=1, label='Predicted (Physical)')
    plt.xlabel('Front slip angle (rad)')
    plt.ylabel('Front lateral force (N)')
    plt.legend(); plt.title('Front Axle Validation (Physical Force)'); plt.savefig('pure_slip_front_validation.png'); plt.close()

    # 可视化后轴 Fy (物理力)
    plt.figure(figsize=(8,5))
    plt.scatter(alpha_r[test_idx][idx_p], Fy_r[test_idx][idx_p], s=1, label='True (Physical)')
    plt.scatter(alpha_r[test_idx][idx_p], pred_r_fy_phys[idx_p], s=1, label='Predicted (Physical)')
    plt.xlabel('Rear slip angle (rad)')
    plt.ylabel('Rear lateral force (N)')
    plt.legend(); plt.title('Rear Axle Fy Validation (Physical Force)'); plt.savefig('combined_slip_rear_fy_validation.png'); plt.close()

    # 可视化后轴 Fx (物理力)
    plt.figure(figsize=(8,5))
    plt.scatter(sigma_r[test_idx][idx_p], Fx_r[test_idx][idx_p], s=1, label='True (Physical)')
    plt.scatter(sigma_r[test_idx][idx_p], pred_r_fx_phys[idx_p], s=1, label='Predicted (Physical)')
    plt.xlabel('Rear slip ratio')
    plt.ylabel('Rear longitudinal force (N)')
    plt.legend(); plt.title('Rear Axle Fx Validation (Physical Force)'); plt.savefig('combined_slip_rear_fx_validation.png'); plt.close()

    torch.save(pure_model.state_dict(), 'pure_slip_expTanh_front.pth')
    torch.save(combined_model.state_dict(), 'combined_slip_expTanh_rear.pth')
    print("\nModels saved successfully.")
    print("Script execution completed.")

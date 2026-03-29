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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 配置路径，确保能找到 tire_data_preprocessing 模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from tire_data_preprocessing import load_all_wheels_data

# ------------------------------
# 1. 参数设置与配置
# ------------------------------
CONFIG = {
    "model_type": "MLP", # 可选: "ExpTanh", "MagicFormula", "MLP"
    "hidden_units": [32, 32],    # 方便切换网络深度和宽度
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "lambda_penalty": 0.01,
    "random_seed": 42,           # 增加随机种子配置
    "device": "cpu",             # "cpu" 或 "cuda"
    "output_base_dir": "./Tire_Model/Integrated_Models"
}

# 设置随机种子确保可复现性
np.random.seed(CONFIG["random_seed"])
torch.manual_seed(CONFIG["random_seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["random_seed"])

# 默认输出路径 (可被外部脚本修改)
DEFAULT_OUTPUT_DIR = os.path.join(CONFIG["output_base_dir"], CONFIG["model_type"])

# ------------------------------
# 2. 通用基础 MLP 结构
# ------------------------------
class BaseBackbone(nn.Module):
    """
    可配置的通用多层感知机骨架
    """
    def __init__(self, input_dim, hidden_units, output_dim):
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

    def forward(self, x):
        return self.net(x)

# ------------------------------
# 3. 各种理论模型组件
# ------------------------------
class ExpTanhCore(nn.Module):
    def forward(self, z, p):
        # p: [batch, 3] -> a1, a2, a3
        a1 = torch.sigmoid(p[:, 0:1]) * 1.5
        a2 = torch.nn.functional.softplus(p[:, 1:2]) * 15.0
        a3 = torch.nn.functional.softplus(p[:, 2:3])
        
        exp_input = -a3 * torch.abs(z)
        exp_term = torch.exp(torch.clamp(exp_input, -20, 0))
        tanh_term = torch.tanh(a2 * z)
        return a1 * tanh_term * exp_term

class MagicFormulaCore(nn.Module):
    def forward(self, z, p):
        # p: [batch, 4] -> D, B, C, E
        D = torch.sigmoid(p[:, 0:1]) * 1.5
        B = torch.nn.functional.softplus(p[:, 1:2]) * 15.0
        C = torch.nn.functional.softplus(p[:, 2:3]) + 1.0
        E = torch.tanh(p[:, 3:4])
        
        Bz = B * z
        phi = Bz - E * (Bz - torch.atan(Bz))
        return D * torch.sin(C * torch.atan(phi))

class MLPCore(nn.Module):
    """ 纯神经网络模型：直接通过滑移率和特征预测无量纲力系数 """
    def forward(self, z, p):
        # p 已经是神经网络直接输出的力系数，不需要理论公式转换
        # 为了保持接口一致，这里的 p 就是 F_coeff
        # 增加一个 sigmoid 限制范围以保持物理合理性 (约 0~1.5 mu)
        return torch.tanh(p) * 1.2 # 使用 tanh 限制在 -1.2 ~ 1.2 之间

# ------------------------------
# 4. 统一轮胎模型封装
# ------------------------------
class UnifiedTireModel(nn.Module):
    def __init__(self, input_dim, hidden_units, model_type="ExpTanh"):
        super().__init__()
        self.model_type = model_type
        
        if model_type == "ExpTanh":
            self.backbone = BaseBackbone(input_dim, hidden_units, 3)
            self.core = ExpTanhCore()
        elif model_type == "MagicFormula":
            self.backbone = BaseBackbone(input_dim, hidden_units, 4)
            self.core = MagicFormulaCore()
        elif model_type == "MLP":
            # 纯 MLP 模式下，将 z (滑移率) 也作为输入进入 backbone
            self.backbone = BaseBackbone(input_dim + 1, hidden_units, 1)
            self.core = MLPCore()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, z, feat, mu_Fz):
        if self.model_type == "MLP":
            # 将滑移率并入特征向量
            combined_feat = torch.cat([z, feat], dim=1)
            p = self.backbone(combined_feat)
        else:
            p = self.backbone(feat)
            
        F_coeff = self.core(z, p)
        return F_coeff * mu_Fz

class CombinedTireModel(nn.Module):
    """ 组合滑移封装层 """
    def __init__(self, input_dim_feat, hidden_units, model_type="ExpTanh"):
        super().__init__()
        self.tot_model = UnifiedTireModel(input_dim_feat, hidden_units, model_type)

    def forward(self, alpha, sigma, feat, mu_Fz):
        tan_alpha = torch.tan(alpha)
        kappa = torch.sqrt(tan_alpha**2 + sigma**2 + 1e-9)
        Ftot = self.tot_model(kappa, feat, mu_Fz)
        Fy = (tan_alpha / kappa) * Ftot
        Fx = (sigma / kappa) * Ftot
        return Fy, Fx

# ------------------------------
# 5. 辅助函数 (数据过滤、训练、绘图)
# ------------------------------
def filter_tire_data(data, axle='front'):
    if data.empty: return data
    total_initial = len(data)
    if axle == 'front':
        mask_sign = (data["alpha_f"] * data["Fy_f"] >= 0)
    else:
        mask_sign = (data["alpha_r"] * data["Fy_r"] >= 0)
    data_filtered = data[mask_sign].copy()
    if axle == 'rear':
        mask_speed = (data_filtered["Vx_kmh"] >= 10.0)
        mask_slip = (np.abs(data_filtered["sigma_r"]) <= 10.0)
        data_filtered = data_filtered[mask_speed & mask_slip].copy()
    print(f"--- {axle.capitalize()} Axle: {len(data_filtered)}/{total_initial} points remained ---")
    return data_filtered.reset_index(drop=True)

def train_model(model, dataloader, optimizer, device, is_combined=False):
    model.train()
    total_loss, total_mse, total_penalty = 0, 0, 0
    force_scale = 1000.0
    for batch in dataloader:
        if is_combined:
            alpha, sigma, feat, mu_Fz, target_Fy, target_Fx = [b.to(device) for b in batch]
            pred_Fy, pred_Fx = model(alpha, sigma, feat, mu_Fz)
            mse = nn.functional.mse_loss(pred_Fy/force_scale, target_Fy/force_scale) + \
                  nn.functional.mse_loss(pred_Fx/force_scale, target_Fx/force_scale)
            Ftot_pred = torch.sqrt(pred_Fy**2 + pred_Fx**2 + 1e-9)
            penalty = torch.mean(torch.relu(Ftot_pred - mu_Fz)**2) / (force_scale**2)
        else:
            z, feat, mu_Fz, target_F = [b.to(device) for b in batch]
            pred_F = model(z, feat, mu_Fz)
            mse = nn.functional.mse_loss(pred_F/force_scale, target_F/force_scale)
            penalty = torch.mean(torch.relu(torch.abs(pred_F) - mu_Fz)**2) / (force_scale**2)
            
        loss = mse + CONFIG["lambda_penalty"] * penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_mse += mse.item()
        total_penalty += penalty.item()
    return total_loss/len(dataloader), total_mse/len(dataloader), total_penalty/len(dataloader)

def plot_history(history, title, filename, output_dir=None):
    if output_dir is None: output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(history["loss"], label='Total')
    plt.plot(history["mse"], label='MSE', linestyle='--')
    plt.yscale('log')
    plt.title(title); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, filename)); plt.close()

def plot_error(true_val, pred_val, title, filename, mode="percent", output_dir=None):
    """
    绘制误差分布图
    mode: "percent" 为百分比误差 (%), "abs" 为绝对值误差 (N)
    """
    if output_dir is None: output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    if mode == "percent":
        error = np.abs(pred_val - true_val) / (np.abs(true_val) + 100.0) * 100.0
        unit = "%"
        hist_range = (0, 50)
    else:
        error = np.abs(pred_val - true_val)
        unit = "N"
        hist_range = (0, 1000) # 绝对力误差范围一般在 1000N 以内观察

    plt.figure(figsize=(10, 6))
    plt.hist(error.flatten(), bins=50, range=hist_range, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(error), color='red', linestyle='--', label=f'Mean: {np.mean(error):.2f}{unit}')
    plt.axvline(np.median(error), color='green', linestyle='--', label=f'Median: {np.median(error):.2f}{unit}')
    
    plt.title(f"{title} ({mode.capitalize()} Error)")
    plt.xlabel(f"Error ({unit})")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"📊 Saved {mode} error plot: {filename}")

# ------------------------------
# 6. 主程序流程
# ------------------------------
if __name__ == "__main__":
    os.makedirs(CONFIG["output_base_dir"], exist_ok=True)
    output_dir = os.path.join(CONFIG["output_base_dir"], CONFIG["model_type"])
    os.makedirs(output_dir, exist_ok=True)

    data_path = "./data/combined_tire_data.pkl"
    if not os.path.exists(data_path):
        print("❌ Data not found."); sys.exit()
    data_full = pd.read_pickle(data_path)
    device = torch.device(CONFIG["device"])

    # --- 前轴模型 ---
    print(f"\n🚀 Training Front Axle [{CONFIG['model_type']}]...")
    df_f = filter_tire_data(data_full, 'front')
    feat_cols_f = ['yaw_rate', 'speed', 'beta']
    scaler_f = StandardScaler()
    X_feat_f = scaler_f.fit_transform(df_f[feat_cols_f]).astype(np.float32)
    z_f = df_f['alpha_f'].values.reshape(-1, 1).astype(np.float32)
    mu_fz_f = df_f['mu_Fz_f'].values.reshape(-1, 1).astype(np.float32)
    target_f = df_f['Fy_f'].values.reshape(-1, 1).astype(np.float32)
    
    idx_train, idx_test = train_test_split(np.arange(len(df_f)), test_size=0.2, random_state=CONFIG["random_seed"])
    train_ds = TensorDataset(torch.tensor(z_f[idx_train]), torch.tensor(X_feat_f[idx_train]), 
                             torch.tensor(mu_fz_f[idx_train]), torch.tensor(target_f[idx_train]))
    loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    
    model_f = UnifiedTireModel(len(feat_cols_f), CONFIG["hidden_units"], CONFIG["model_type"]).to(device)
    opt = optim.Adam(model_f.parameters(), lr=CONFIG["learning_rate"])
    hist = {"loss": [], "mse": []}
    for e in range(CONFIG["epochs"]):
        l, m, p = train_model(model_f, loader, opt, device)
        hist["loss"].append(l); hist["mse"].append(m)
        if (e+1)%20==0: print(f"Epoch {e+1}, Loss: {l:.4f}")
    
    plot_history(hist, f"Front {CONFIG['model_type']} History", "front_history.png")
    model_f.eval()
    with torch.no_grad():
        pred = model_f(torch.tensor(z_f[idx_test]).to(device), torch.tensor(X_feat_f[idx_test]).to(device), 
                       torch.tensor(mu_fz_f[idx_test]).to(device)).cpu().numpy()
        # 评价前轴：百分比误差与绝对误差
        plot_error(target_f[idx_test], pred, f"Front {CONFIG['model_type']} Fy", "front_fy_error_percent.png", mode="percent")
        plot_error(target_f[idx_test], pred, f"Front {CONFIG['model_type']} Fy", "front_fy_error_abs.png", mode="abs")

    # --- 后轴模型 ---
    print(f"\n🚀 Training Rear Axle [{CONFIG['model_type']}]...")
    df_r = filter_tire_data(data_full, 'rear')
    feat_cols_r = ['yaw_rate', 'speed']
    scaler_r = StandardScaler()
    X_feat_r = scaler_r.fit_transform(df_r[feat_cols_r]).astype(np.float32)
    alpha_r = df_r['alpha_r'].values.reshape(-1, 1).astype(np.float32)
    sigma_r = df_r['sigma_r'].values.reshape(-1, 1).astype(np.float32)
    mu_fz_r = df_r['mu_Fz_r'].values.reshape(-1, 1).astype(np.float32)
    target_fy_r = df_r['Fy_r'].values.reshape(-1, 1).astype(np.float32)
    target_fx_r = df_r['Fx_r'].values.reshape(-1, 1).astype(np.float32)

    idx_train, idx_test = train_test_split(np.arange(len(df_r)), test_size=0.2, random_state=CONFIG["random_seed"])
    train_ds_r = TensorDataset(torch.tensor(alpha_r[idx_train]), torch.tensor(sigma_r[idx_train]),
                               torch.tensor(X_feat_r[idx_train]), torch.tensor(mu_fz_r[idx_train]),
                               torch.tensor(target_fy_r[idx_train]), torch.tensor(target_fx_r[idx_train]))
    loader_r = DataLoader(train_ds_r, batch_size=CONFIG["batch_size"], shuffle=True)
    
    model_r = CombinedTireModel(len(feat_cols_r), CONFIG["hidden_units"], CONFIG["model_type"]).to(device)
    opt_r = optim.Adam(model_r.parameters(), lr=CONFIG["learning_rate"])
    hist_r = {"loss": [], "mse": []}
    for e in range(CONFIG["epochs"]):
        l, m, p = train_model(model_r, loader_r, opt_r, device, is_combined=True)
        hist_r["loss"].append(l); hist_r["mse"].append(m)
        if (e+1)%20==0: print(f"Epoch {e+1}, Loss: {l:.4f}")

    plot_history(hist_r, f"Rear {CONFIG['model_type']} History", "rear_history.png")
    model_r.eval()
    with torch.no_grad():
        p_fy, p_fx = model_r(torch.tensor(alpha_r[idx_test]).to(device), torch.tensor(sigma_r[idx_test]).to(device),
                             torch.tensor(X_feat_r[idx_test]).to(device), torch.tensor(mu_fz_r[idx_test]).to(device))
        p_fy, p_fx = p_fy.cpu().numpy(), p_fx.cpu().numpy()
        # 评价后轴：百分比误差与绝对误差
        plot_error(target_fy_r[idx_test], p_fy, f"Rear {CONFIG['model_type']} Fy", "rear_fy_error_percent.png", mode="percent")
        plot_error(target_fy_r[idx_test], p_fy, f"Rear {CONFIG['model_type']} Fy", "rear_fy_error_abs.png", mode="abs")
        plot_error(target_fx_r[idx_test], p_fx, f"Rear {CONFIG['model_type']} Fx", "rear_fx_error_percent.png", mode="percent")
        plot_error(target_fx_r[idx_test], p_fx, f"Rear {CONFIG['model_type']} Fx", "rear_fx_error_abs.png", mode="abs")

    print(f"\n✅ Integration training complete. Results in {output_dir}")

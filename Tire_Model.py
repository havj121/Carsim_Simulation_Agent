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
# 1. 获取 CarSim 仿真数据 (批量仿真)
# ==========================================
def get_carsim_tire_data(template_dir):
    # 实例化 Agent
    agent = CarsimAgent(template_path=template_dir)
    
    # 目标提取变量
    target_vars = [
        "Time", 
        "Alpha_L1", "Alpha_R1", "Alpha_L2", "Alpha_R2",
        "Fz_L1", "Fz_R1", "Fz_L2", "Fz_R2",
        "Fy_L1", "Fy_R1", "Fy_L2", "Fy_R2",
        "Vx", "Beta", "AV_Y"
    ]
    
    agent.options.run_mode = "mods"
    agent.options.extract_data = True
    # 关键：显式设置提取变量
    agent.options.default_extract_vars = target_vars
    agent.options.keep_detailed_results = True
    agent.options.delete_sim_files = False
    
    # 手动设置提取方式为 CSV
    agent.options.extraction_method = "csv"
    
    # 加载批量仿真配置
    batch_mods_path = "batch_mods.json"
    if not os.path.exists(batch_mods_path):
        # 如果不存在，报错
        raise FileNotFoundError(f"Batch configuration file {batch_mods_path} not found. "
                                "Please create it with the required format.")
    
    with open(batch_mods_path, 'r') as f:
        batch_configs = json.load(f)
        
    print(f"🚀 Running BATCH CarSim simulation ({len(batch_configs)} tasks) to collect tire data...")
    # 运行批量仿真，显式传递 extract_vars
    results = agent.run_batch(batch_configs, processes=4)
    
    # 重新检查结果并显式从 CSV 提取，如果 run_batch 的 extract_vars 逻辑有问题
    for res in results:
        if res.get("status") == "success":
            res_path = res.get("results_path")
            csv_path = os.path.join(res_path, "LastRun.csv")
            if os.path.exists(csv_path):
                # 重新提取所需的变量
                agent.extractor.extract_to_csv(csv_path, target_vars=target_vars, method="csv")
    
    return results

def process_carsim_batch_results(batch_results):
    """
    处理来自 agent.run_batch 的结果列表 (内存中的数据或 CSV 路径)
    按轮胎分离数据: {'L1': (X_z, X_feat, Y_fy), ...}
    """
    tire_data = {tire: {'X_z': [], 'X_feat': [], 'Y_fy': []} for tire in ['L1', 'R1', 'L2', 'R2']}
    
    # 由于 get_carsim_tire_data 现在总是返回 results 列表
    for res in batch_results:
        # 优先从重新提取的 _extracted.csv 读取，因为内存中的数据可能列不全
        if res.get("status") == "success" and res.get("results_path"):
            extracted_csv = os.path.join(res.get("results_path"), "LastRun_extracted.csv")
            if os.path.exists(extracted_csv):
                print(f"Processing task {res.get('task_id')} from Extracted CSV: {extracted_csv}")
                df = pd.read_csv(extracted_csv)
                _extract_from_df(df, tire_data)
                continue
                
        if res.get("status") == "success" and res.get("extracted_data"):
            print(f"Processing task {res.get('task_id')} from memory...")
            df = pd.DataFrame(res.get("extracted_data"))
            _extract_from_df(df, tire_data)
        elif res.get("status") == "success" and res.get("results_path"):
            # 兜底：如果内存中没有，尝试从原始文件读取
            csv_path = os.path.join(res.get("results_path"), "LastRun.csv")
            if os.path.exists(csv_path):
                print(f"Processing task {res.get('task_id')} from Original CSV: {csv_path}")
                df = pd.read_csv(csv_path)
                _extract_from_df(df, tire_data)
            
    # 转换为 numpy 数组
    final_tire_data = {}
    for tire, data in tire_data.items():
        final_tire_data[tire] = (
            np.array(data['X_z']),
            np.array(data['X_feat']),
            np.array(data['Y_fy'])
        )
            
    return final_tire_data

def _extract_from_df(df, tire_data):
    for tire in tire_data.keys():
        # 批量仿真可能没有提取所有列，需要检查
        required_cols = [f'Alpha_{tire}', f'Fz_{tire}', f'Fy_{tire}', 'Vx', 'Beta', 'AV_Y']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing columns for tire {tire} in data: {[col for col in required_cols if col not in df.columns]}")
            continue
            
        alpha_rad = df[f'Alpha_{tire}'].values * (np.pi / 180.0)
        fz = df[f'Fz_{tire}'].values
        fy = df[f'Fy_{tire}'].values
        v = df['Vx'].values / 3.6
        beta = df['Beta'].values * (np.pi / 180.0)
        r = df['AV_Y'].values * (np.pi / 180.0)
        
        z = np.tan(alpha_rad)
        
        for i in range(len(df)):
            tire_data[tire]['X_z'].append([z[i]])
            tire_data[tire]['X_feat'].append([v[i], beta[i], r[i], fz[i]])
            tire_data[tire]['Y_fy'].append([fy[i]])

# ==========================================
# 2. Neural-ExpTanh 模型定义 
# ==========================================
class NeuralExpTanhTire(nn.Module):
    def __init__(self, feature_dim=4):
        super(NeuralExpTanhTire, self).__init__()
        # MLP 分支：预测 ExpTanh 的 5 个系数
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 5) 
        )

    def forward(self, z, feat):
        coeffs = self.mlp(feat)
        c1 = coeffs[:, 0:1]
        c2 = coeffs[:, 1:2]
        c3 = torch.exp(coeffs[:, 2:3]) # 保证非负 
        c4 = torch.exp(coeffs[:, 3:4]) # 保证非负
        c5 = coeffs[:, 4:5]
        
        # ExpTanh 核心公式: Fy = c1 + c2 * exp(-c3*|z|) * tanh(c4*(z - c5))
        term_exp = torch.exp(-c3 * torch.abs(z))
        term_tanh = torch.tanh(c4 * (z - c5))
        fy_pred = c1 + c2 * term_exp * term_tanh
        return fy_pred

# ==========================================
# 3. 运行主流程
# ==========================================
def run_tire_modeling():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(current_dir, "Drifting_Template")
    
    # 1. 获取仿真数据
    try:
        batch_results = get_carsim_tire_data(template_dir)
        tire_data_dict = process_carsim_batch_results(batch_results)
    except Exception as e:
        print(f"Error getting CarSim data: {e}")
        return

    # 准备存储模型和验证结果
    models = {}
    verification_results = {}

    # 遍历每个轮胎进行训练
    for tire, (X_z, X_feat, Y_fy) in tire_data_dict.items():
        if len(X_z) == 0:
            print(f"No data for tire {tire}, skipping...")
            continue
            
        print(f"\n>>> Training model for tire: {tire} (Samples: {len(X_z)})")
        
        # 数据缩放
        feat_mean = X_feat.mean(axis=0)
        feat_std = X_feat.std(axis=0)
        feat_std[feat_std == 0] = 1.0
        X_feat_scaled = (X_feat - feat_mean) / feat_std
        
        # 转 Tensor
        z_tensor = torch.FloatTensor(X_z)
        feat_tensor = torch.FloatTensor(X_feat_scaled)
        fy_tensor = torch.FloatTensor(Y_fy)

        # 初始化模型
        model = NeuralExpTanhTire(feature_dim=4)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # 训练循环
        for epoch in range(501):
            model.train()
            optimizer.zero_grad()
            fy_pred = model(z_tensor, feat_tensor)
            loss = criterion(fy_pred, fy_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Tire {tire} | Epoch {epoch} | Loss: {loss.item():.4f}")

        # 保存模型和预测结果用于可视化
        models[tire] = model
        model.eval()
        with torch.no_grad():
            fy_pred_all = model(z_tensor, feat_tensor).numpy()
        verification_results[tire] = (X_z, Y_fy, fy_pred_all)

    # 4. 验证与可视化 (4个子图)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, tire in enumerate(['L1', 'R1', 'L2', 'R2']):
        if tire in verification_results:
            X_z, Y_fy, fy_pred = verification_results[tire]
            axes[i].scatter(X_z, Y_fy, s=2, alpha=0.5, label='CarSim Data')
            axes[i].scatter(X_z, fy_pred, s=2, alpha=0.5, label='Model Prediction', color='red')
            axes[i].set_title(f"Tire {tire} Lateral Force Modeling")
            axes[i].set_xlabel("tan(alpha)")
            axes[i].set_ylabel("Fy [N]")
            axes[i].legend()
            axes[i].grid(True)
        else:
            axes[i].set_title(f"Tire {tire} - No Data")

    plt.tight_layout()
    plt.savefig("tire_model_verification.png")
    print("\nVerification plot saved as tire_model_verification.png")

if __name__ == "__main__":
    run_tire_modeling()

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# 添加路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from TireModel import (
    UnifiedTireModel, CombinedTireModel, filter_tire_data, 
    train_model, plot_history, plot_error, CONFIG
)

# ------------------------------
# 1. 实验配置
# ------------------------------
STUDY_CONFIG = {
    "models": ["ExpTanh", "MagicFormula", "MLP"],
    "hidden_units": [32, 32],
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "random_seed": 42,
    "base_output_dir": "./Tire_Model/Comparison_Study"
}

# 设置全局种子
np.random.seed(STUDY_CONFIG["random_seed"])
torch.manual_seed(STUDY_CONFIG["random_seed"])

# ------------------------------
# 2. 实验运行核心函数
# ------------------------------
def run_experiment(exp_name, data_full, model_type, train_indices=None, test_indices=None, axle='front'):
    """
    运行单次实验并保存结果
    """
    exp_dir = os.path.join(STUDY_CONFIG["base_output_dir"], exp_name, model_type)
    os.makedirs(exp_dir, exist_ok=True)
    
    device = torch.device("cpu")
    
    # 数据准备
    df = filter_tire_data(data_full, axle)
    if axle == 'front':
        feat_cols = ['yaw_rate', 'speed', 'beta']
        z = df['alpha_f'].values.reshape(-1, 1).astype(np.float32)
        mu_fz = df['mu_Fz_f'].values.reshape(-1, 1).astype(np.float32)
        target = df['Fy_f'].values.reshape(-1, 1).astype(np.float32)
    else:
        feat_cols = ['yaw_rate', 'speed']
        alpha = df['alpha_r'].values.reshape(-1, 1).astype(np.float32)
        sigma = df['sigma_r'].values.reshape(-1, 1).astype(np.float32)
        mu_fz = df['mu_Fz_r'].values.reshape(-1, 1).astype(np.float32)
        target_fy = df['Fy_r'].values.reshape(-1, 1).astype(np.float32)
        target_fx = df['Fx_r'].values.reshape(-1, 1).astype(np.float32)

    scaler = StandardScaler()
    X_feat = scaler.fit_transform(df[feat_cols]).astype(np.float32)

    # 索引分配
    if train_indices is None or test_indices is None:
        all_idx = np.arange(len(df))
        train_indices, test_indices = train_test_split(all_idx, test_size=0.2, random_state=STUDY_CONFIG["random_seed"])

    # 构建模型
    if axle == 'front':
        model = UnifiedTireModel(len(feat_cols), STUDY_CONFIG["hidden_units"], model_type).to(device)
        train_ds = TensorDataset(torch.tensor(z[train_indices]), torch.tensor(X_feat[train_indices]), 
                                 torch.tensor(mu_fz[train_indices]), torch.tensor(target[train_indices]))
    else:
        model = CombinedTireModel(len(feat_cols), STUDY_CONFIG["hidden_units"], model_type).to(device)
        train_ds = TensorDataset(torch.tensor(alpha[train_indices]), torch.tensor(sigma[train_indices]),
                                 torch.tensor(X_feat[train_indices]), torch.tensor(mu_fz[train_indices]),
                                 torch.tensor(target_fy[train_indices]), torch.tensor(target_fx[train_indices]))

    loader = DataLoader(train_ds, batch_size=STUDY_CONFIG["batch_size"], shuffle=True)
    opt = optim.Adam(model.parameters(), lr=STUDY_CONFIG["learning_rate"])
    
    history = {"loss": [], "mse": []}
    print(f"Starting training for {STUDY_CONFIG['epochs']} epochs...")
    for e in range(STUDY_CONFIG["epochs"]):
        l, m, p = train_model(model, loader, opt, device, is_combined=(axle=='rear'))
        history["loss"].append(l); history["mse"].append(m)
        if (e+1) % 10 == 0:
            print(f"  Epoch {e+1}/{STUDY_CONFIG['epochs']} - Loss: {l:.4f}")
    
    print(f"Training complete. Saving results to {exp_dir}...")
    
    # 保存历史 (带上轴信息)
    with open(os.path.join(exp_dir, f"history_{axle}.json"), 'w') as f:
        json.dump(history, f)

    # 预测与评估
    model.eval()
    results = {}
    with torch.no_grad():
        if axle == 'front':
            # 训练集预测
            pred_train = model(torch.tensor(z[train_indices]).to(device), torch.tensor(X_feat[train_indices]).to(device), 
                               torch.tensor(mu_fz[train_indices]).to(device)).cpu().numpy()
            # 测试集预测
            pred_test = model(torch.tensor(z[test_indices]).to(device), torch.tensor(X_feat[test_indices]).to(device), 
                              torch.tensor(mu_fz[test_indices]).to(device)).cpu().numpy()
            
            results = {
                "train_true": target[train_indices].tolist(), "train_pred": pred_train.tolist(),
                "test_true": target[test_indices].tolist(), "test_pred": pred_test.tolist(),
                "train_alpha": z[train_indices].tolist(), "test_alpha": z[test_indices].tolist()
            }
        else:
            p_fy_tr, p_fx_tr = model(torch.tensor(alpha[train_indices]).to(device), torch.tensor(sigma[train_indices]).to(device),
                                    torch.tensor(X_feat[train_indices]).to(device), torch.tensor(mu_fz[train_indices]).to(device))
            p_fy_te, p_fx_te = model(torch.tensor(alpha[test_indices]).to(device), torch.tensor(sigma[test_indices]).to(device),
                                    torch.tensor(X_feat[test_indices]).to(device), torch.tensor(mu_fz[test_indices]).to(device))
            
            # 计算组合滑移 kappa = sqrt(tan(alpha)^2 + sigma^2)
            train_kappa = np.sqrt(np.tan(alpha[train_indices])**2 + sigma[train_indices]**2).tolist()
            test_kappa = np.sqrt(np.tan(alpha[test_indices])**2 + sigma[test_indices]**2).tolist()

            results = {
                "train_true_fy": target_fy[train_indices].tolist(), "train_pred_fy": p_fy_tr.cpu().numpy().tolist(),
                "train_true_fx": target_fx[train_indices].tolist(), "train_pred_fx": p_fx_tr.cpu().numpy().tolist(),
                "test_true_fy": target_fy[test_indices].tolist(), "test_pred_fy": p_fy_te.cpu().numpy().tolist(),
                "test_true_fx": target_fx[test_indices].tolist(), "test_pred_fx": p_fx_te.cpu().numpy().tolist(),
                "train_kappa": train_kappa, "test_kappa": test_kappa
            }

    with open(os.path.join(exp_dir, f"results_{axle}.json"), 'w') as f:
        json.dump(results, f)
    
    print(f"Finished experiment: {exp_name} - {model_type}")
    return history, results

# ------------------------------
# 3. 实验场景定义
# ------------------------------
def run_all_studies():
    data_path = "./data/combined_tire_data.pkl"
    data_full = pd.read_pickle(data_path)
    
    all_results = {
        "Baseline": {"front": {}, "rear": {}},
        "DataVolume": {"front": {}, "rear": {}},
        "Extrapolation": {"front": {}, "rear": {}}
    }
    
    for axle in ['front', 'rear']:
        # 3.1 基础设置 (Baseline)
        print(f"\n--- Running Baseline Comparison ({axle}) ---")
        for m_type in STUDY_CONFIG["models"]:
            hist, res = run_experiment("1_Baseline", data_full, m_type, axle=axle)
            all_results["Baseline"][axle][m_type] = (hist, res)

        # 3.2 数量影响分析 (Data Volume)
        print(f"\n--- Running Data Volume Impact Analysis ({axle}) ---")
        df_axle = filter_tire_data(data_full, axle)
        all_idx = np.arange(len(df_axle))
        train_idx_80, test_idx_20 = train_test_split(all_idx, test_size=0.2, random_state=STUDY_CONFIG["random_seed"])
        train_idx_60, _ = train_test_split(train_idx_80, test_size=0.4, random_state=STUDY_CONFIG["random_seed"])
        
        for m_type in STUDY_CONFIG["models"]:
            hist, res = run_experiment("2_DataVolume", data_full, m_type, 
                                       train_indices=train_idx_60, test_indices=test_idx_20, axle=axle)
            all_results["DataVolume"][axle][m_type] = (hist, res)

        # 3.3 外延性分析 (Extrapolation)
        print(f"\n--- Running Extrapolation Analysis ({axle}) ---")
        df_axle = filter_tire_data(data_full, axle)
        if axle == 'front':
            sort_val = np.abs(df_axle['alpha_f'].values)
        else:
            # 后轴按综合滑移排序
            tan_alpha = np.tan(df_axle['alpha_r'].values)
            sigma = df_axle['sigma_r'].values
            sort_val = np.sqrt(tan_alpha**2 + sigma**2)
            
        sorted_indices = np.argsort(sort_val)
        train_idx_extra = sorted_indices[:int(0.8 * len(sorted_indices))]
        test_idx_extra = sorted_indices[int(0.8 * len(sorted_indices)):]
        
        for m_type in STUDY_CONFIG["models"]:
            hist, res = run_experiment("3_Extrapolation", data_full, m_type, 
                                       train_indices=train_idx_extra, test_indices=test_idx_extra, axle=axle)
            all_results["Extrapolation"][axle][m_type] = (hist, res)

    # ------------------------------
    # 4. 对比分析绘图
    # ------------------------------
    plot_comparisons(all_results)

def plot_comparisons(all_results):
    os.makedirs(STUDY_CONFIG["base_output_dir"], exist_ok=True)
    
    for axle in ['front', 'rear']:
        # ① 对比三种模型 (Baseline)
        if all_results["Baseline"][axle]:
            plt.figure(figsize=(18, 5))
            plt.suptitle(f"Baseline Model Comparison - {axle.capitalize()} Axle", fontsize=16)
            
            # Loss Curves
            plt.subplot(1, 3, 1)
            plotted_loss = False
            for m_type in STUDY_CONFIG["models"]:
                if m_type in all_results["Baseline"][axle]:
                    loss_data = np.array(all_results["Baseline"][axle][m_type][0]["loss"])
                    valid_idx = loss_data > 0
                    if np.any(valid_idx):
                        plt.plot(np.where(valid_idx, loss_data, np.nan), label=m_type)
                        plotted_loss = True
            if plotted_loss:
                plt.yscale('log')
            plt.title("Loss Curves"); plt.legend(); plt.grid(True)
            
            # Test Percent Error
            plt.subplot(1, 3, 2)
            for m_type in STUDY_CONFIG["models"]:
                if m_type not in all_results["Baseline"][axle]: continue
                res = all_results["Baseline"][axle][m_type][1]
                if axle == 'front':
                    true, pred = np.array(res["test_true"]), np.array(res["test_pred"])
                else:
                    true = np.sqrt(np.array(res["test_true_fy"])**2 + np.array(res["test_true_fx"])**2)
                    pred = np.sqrt(np.array(res["test_pred_fy"])**2 + np.array(res["test_pred_fx"])**2)
                
                error = np.abs(pred - true) / (np.abs(true) + 100.0) * 100.0
                mean_err = np.mean(error)
                plt.hist(error.flatten(), bins=50, alpha=0.4, label=f"{m_type} (Mean:{mean_err:.2f}%)", range=(0, 30))
            plt.title("Test Percent Error (%)"); plt.legend(); plt.grid(True)
            
            # Test Absolute Error
            plt.subplot(1, 3, 3)
            for m_type in STUDY_CONFIG["models"]:
                if m_type not in all_results["Baseline"][axle]: continue
                res = all_results["Baseline"][axle][m_type][1]
                if axle == 'front':
                    true, pred = np.array(res["test_true"]), np.array(res["test_pred"])
                else:
                    true = np.sqrt(np.array(res["test_true_fy"])**2 + np.array(res["test_true_fx"])**2)
                    pred = np.sqrt(np.array(res["test_pred_fy"])**2 + np.array(res["test_pred_fx"])**2)
                
                error_abs = np.abs(pred - true)
                mean_abs = np.mean(error_abs)
                plt.hist(error_abs.flatten(), bins=50, alpha=0.4, label=f"{m_type} (Mean:{mean_abs:.1f}N)", range=(0, 1000))
            plt.title("Test Absolute Error (N)"); plt.legend(); plt.grid(True)
            
            plt.savefig(os.path.join(STUDY_CONFIG["base_output_dir"], f"1_baseline_comparison_{axle}.png"))
            plt.close()

        # ② 数量影响与外延分析对比图
        for study_name in ["DataVolume", "Extrapolation"]:
            if not all_results[study_name][axle]: continue
            
            plt.figure(figsize=(20, 10))
            plt.suptitle(f"{study_name} Analysis - {axle.capitalize()} Axle (Solid: Baseline, Dashed: {study_name})", fontsize=16)
            
            # 1. 损失函数对比
            plt.subplot(2, 2, 1)
            plotted_loss_comp = False
            for m_type in STUDY_CONFIG["models"]:
                if m_type in all_results["Baseline"][axle] and m_type in all_results[study_name][axle]:
                    # Baseline
                    loss_base = np.array(all_results["Baseline"][axle][m_type][0]["loss"])
                    valid_base = loss_base > 0
                    if np.any(valid_base):
                        plt.plot(np.where(valid_base, loss_base, np.nan), label=f"{m_type}-Base")
                        plotted_loss_comp = True
                    # Study
                    loss_study = np.array(all_results[study_name][axle][m_type][0]["loss"])
                    valid_study = loss_study > 0
                    if np.any(valid_study):
                        plt.plot(np.where(valid_study, loss_study, np.nan), label=f"{m_type}-{study_name[:4]}", linestyle='--')
                        plotted_loss_comp = True
            if plotted_loss_comp:
                plt.yscale('log')
            plt.title("Loss Curves Comparison"); plt.legend(ncol=2); plt.grid(True)
            
            # 2. 训练集绝对误差对比
            plt.subplot(2, 2, 2)
            for m_type in STUDY_CONFIG["models"]:
                if m_type in all_results[study_name][axle]:
                    res_s = all_results[study_name][axle][m_type][1]
                    if axle == 'front':
                        err_s = np.abs(np.array(res_s["train_pred"]) - np.array(res_s["train_true"]))
                    else:
                        err_s = np.abs(np.sqrt(np.array(res_s["train_pred_fy"])**2 + np.array(res_s["train_pred_fx"])**2) - 
                                       np.sqrt(np.array(res_s["train_true_fy"])**2 + np.array(res_s["train_true_fx"])**2))
                    
                    plt.hist(err_s.flatten(), bins=50, alpha=0.3, label=f"{m_type} (Mean:{np.mean(err_s):.1f}N)", range=(0, 1000))
            plt.title(f"Train Absolute Error (N) - {study_name}"); plt.legend(); plt.grid(True)

            # 3. 测试集百分比误差对比
            plt.subplot(2, 2, 3)
            hist_range_pct = (0, 40) if study_name == "DataVolume" else (0, 100)
            for m_type in STUDY_CONFIG["models"]:
                if m_type in all_results[study_name][axle]:
                    res_s = all_results[study_name][axle][m_type][1]
                    if axle == 'front':
                        true, pred = np.array(res_s["test_true"]), np.array(res_s["test_pred"])
                    else:
                        true = np.sqrt(np.array(res_s["test_true_fy"])**2 + np.array(res_s["test_true_fx"])**2)
                        pred = np.sqrt(np.array(res_s["test_pred_fy"])**2 + np.array(res_s["test_pred_fx"])**2)
                    
                    err_pct = np.abs(pred - true) / (np.abs(true) + 100.0) * 100.0
                    plt.hist(err_pct.flatten(), bins=50, alpha=0.3, label=f"{m_type} (Mean:{np.mean(err_pct):.2f}%)", range=hist_range_pct)
            plt.title(f"Test Percent Error (%) - {study_name}"); plt.legend(); plt.grid(True)

            # 4. 测试集绝对误差对比
            plt.subplot(2, 2, 4)
            hist_range_abs = (0, 1200) if study_name == "DataVolume" else (0, 3000)
            for m_type in STUDY_CONFIG["models"]:
                if m_type in all_results[study_name][axle]:
                    res_s = all_results[study_name][axle][m_type][1]
                    if axle == 'front':
                        true, pred = np.array(res_s["test_true"]), np.array(res_s["test_pred"])
                    else:
                        true = np.sqrt(np.array(res_s["test_true_fy"])**2 + np.array(res_s["test_true_fx"])**2)
                        pred = np.sqrt(np.array(res_s["test_pred_fy"])**2 + np.array(res_s["test_pred_fx"])**2)
                    
                    err_abs = np.abs(pred - true)
                    plt.hist(err_abs.flatten(), bins=50, alpha=0.3, label=f"{m_type} (Mean:{np.mean(err_abs):.1f}N)", range=hist_range_abs)
            plt.title(f"Test Absolute Error (N) - {study_name}"); plt.legend(); plt.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(STUDY_CONFIG["base_output_dir"], f"comparison_{study_name}_{axle}.png"))
            plt.close()

if __name__ == "__main__":
    run_all_studies()

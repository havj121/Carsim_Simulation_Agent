import os
import numpy as np
import matplotlib.pyplot as plt
import json

# ------------------------------
# 1. 配置
# ------------------------------
STUDY_CONFIG = {
    "models": ["ExpTanh", "MagicFormula", "MLP"],
    "base_output_dir": "./Tire_Model/Comparison_Study"
}

def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def plot_comparisons_from_disk():
    os.makedirs(STUDY_CONFIG["base_output_dir"], exist_ok=True)
    
    # 构造数据字典
    all_results = {
        "Baseline": {"front": {}, "rear": {}},
        "DataVolume": {"front": {}, "rear": {}},
        "Extrapolation": {"front": {}, "rear": {}}
    }
    
    exp_map = {
        "Baseline": "1_Baseline",
        "DataVolume": "2_DataVolume",
        "Extrapolation": "3_Extrapolation"
    }

    # 加载数据
    for study_name, folder_name in exp_map.items():
        for axle in ['front', 'rear']:
            for m_type in STUDY_CONFIG["models"]:
                exp_dir = os.path.join(STUDY_CONFIG["base_output_dir"], folder_name, m_type)
                hist = load_json(os.path.join(exp_dir, f"history_{axle}.json"))
                res = load_json(os.path.join(exp_dir, f"results_{axle}.json"))
                
                # 兼容旧版本文件名
                if hist is None:
                    hist = load_json(os.path.join(exp_dir, "history.json"))
                if res is None:
                    res = load_json(os.path.join(exp_dir, "results.json"))
                
                if hist and res:
                    all_results[study_name][axle][m_type] = (hist, res)
                else:
                    print(f"⚠️ Warning: Missing data for {study_name} - {axle} - {m_type}")

    # ------------------------------
    # 绘图逻辑
    # ------------------------------
    for axle in ['front', 'rear']:
        # ① 基础对比 (Baseline)
        if all_results["Baseline"][axle]:
            plt.figure(figsize=(18, 5))
            plt.suptitle(f"Baseline Model Comparison - {axle.capitalize()} Axle", fontsize=16)
            
            # Loss
            plt.subplot(1, 3, 1)
            for m_type in STUDY_CONFIG["models"]:
                if m_type in all_results["Baseline"][axle]:
                    plt.plot(all_results["Baseline"][axle][m_type][0]["loss"], label=m_type)
            plt.yscale('log'); plt.title("Loss Curves"); plt.legend(); plt.grid(True)
            
            # Test Percent Error
            plt.subplot(1, 3, 2)
            for m_type in STUDY_CONFIG["models"]:
                if m_type in all_results["Baseline"][axle]:
                    res = all_results["Baseline"][axle][m_type][1]
                    if axle == 'front':
                        if "test_true" in res:
                            true, pred = np.array(res["test_true"]), np.array(res["test_pred"])
                        elif "test_true_fy" in res: # 兼容旧版本
                            true, pred = np.array(res["test_true_fy"]), np.array(res["test_pred_fy"])
                        else:
                            print(f"⏩ Skipping {m_type} {axle} Baseline: Expected front keys not found.")
                            continue
                    else:
                        # 检查 JSON 中是否存在 fy/fx 键
                        if "test_true_fy" in res:
                            true = np.sqrt(np.array(res["test_true_fy"])**2 + np.array(res["test_true_fx"])**2)
                            pred = np.sqrt(np.array(res["test_pred_fy"])**2 + np.array(res["test_pred_fx"])**2)
                        else:
                            print(f"⏩ Skipping {m_type} {axle} Baseline: Expected rear keys not found.")
                            continue
                    
                    error = np.abs(pred - true) / (np.abs(true) + 100.0) * 100.0
                    mean_err = np.mean(error)
                    plt.hist(error.flatten(), bins=50, alpha=0.4, label=f"{m_type} (Mean:{mean_err:.2f}%)", range=(0, 30))
            plt.title("Test Percent Error (%)"); plt.legend(); plt.grid(True)
            
            # Test Absolute Error
            plt.subplot(1, 3, 3)
            for m_type in STUDY_CONFIG["models"]:
                if m_type in all_results["Baseline"][axle]:
                    res = all_results["Baseline"][axle][m_type][1]
                    if axle == 'front':
                        if "test_true" in res:
                            true, pred = np.array(res["test_true"]), np.array(res["test_pred"])
                        elif "test_true_fy" in res: # 兼容旧版本
                            true, pred = np.array(res["test_true_fy"]), np.array(res["test_pred_fy"])
                        else: continue
                    else:
                        if "test_true_fy" in res:
                            true = np.sqrt(np.array(res["test_true_fy"])**2 + np.array(res["test_true_fx"])**2)
                            pred = np.sqrt(np.array(res["test_pred_fy"])**2 + np.array(res["test_pred_fx"])**2)
                        else: continue
                    
                    error_abs = np.abs(pred - true)
                    mean_abs = np.mean(error_abs)
                    plt.hist(error_abs.flatten(), bins=50, alpha=0.4, label=f"{m_type} (Mean:{mean_abs:.1f}N)", range=(0, 1000))
            plt.title("Test Absolute Error (N)"); plt.legend(); plt.grid(True)
            
            plt.savefig(os.path.join(STUDY_CONFIG["base_output_dir"], f"1_baseline_comparison_{axle}.png"))
            plt.close()

        # ② 数量影响与外延分析
        for study_name in ["DataVolume", "Extrapolation"]:
            if not all_results[study_name][axle]: continue
            
            plt.figure(figsize=(20, 10))
            plt.suptitle(f"{study_name} Analysis - {axle.capitalize()} Axle (Solid: Baseline, Dashed: {study_name})", fontsize=16)
            
            # 1. 损失函数对比
            plt.subplot(2, 2, 1)
            for m_type in STUDY_CONFIG["models"]:
                if m_type in all_results["Baseline"][axle] and m_type in all_results[study_name][axle]:
                    plt.plot(all_results["Baseline"][axle][m_type][0]["loss"], label=f"{m_type}-Base")
                    plt.plot(all_results[study_name][axle][m_type][0]["loss"], label=f"{m_type}-{study_name[:4]}", linestyle='--')
            plt.yscale('log'); plt.title("Loss Curves Comparison"); plt.legend(ncol=2); plt.grid(True)
            
            # 2. 训练集绝对误差对比
            plt.subplot(2, 2, 2)
            for m_type in STUDY_CONFIG["models"]:
                if m_type in all_results[study_name][axle]:
                    res_s = all_results[study_name][axle][m_type][1]
                    if axle == 'front':
                        if "train_pred" in res_s and "train_true" in res_s:
                            err_s = np.abs(np.array(res_s["train_pred"]) - np.array(res_s["train_true"]))
                        elif "train_pred_fy" in res_s and "train_true_fy" in res_s: # 兼容旧版本或带有后缀的键名
                            err_s = np.abs(np.array(res_s["train_pred_fy"]) - np.array(res_s["train_true_fy"]))
                        else: continue
                    else:
                        if "train_pred_fy" in res_s:
                            err_s = np.abs(np.sqrt(np.array(res_s["train_pred_fy"])**2 + np.array(res_s["train_pred_fx"])**2) - 
                                           np.sqrt(np.array(res_s["train_true_fy"])**2 + np.array(res_s["train_true_fx"])**2))
                        else: continue
                    
                    plt.hist(err_s.flatten(), bins=50, alpha=0.3, label=f"{m_type} (Mean:{np.mean(err_s):.1f}N)", range=(0, 1000))
            plt.title(f"Train Absolute Error (N) - {study_name}"); plt.legend(); plt.grid(True)

            # 3. 测试集百分比误差对比
            plt.subplot(2, 2, 3)
            for m_type in STUDY_CONFIG["models"]:
                if m_type in all_results[study_name][axle]:
                    res_s = all_results[study_name][axle][m_type][1]
                    if axle == 'front':
                        if "test_true" in res_s:
                            true, pred = np.array(res_s["test_true"]), np.array(res_s["test_pred"])
                        elif "test_true_fy" in res_s: # 兼容旧版本
                            true, pred = np.array(res_s["test_true_fy"]), np.array(res_s["test_pred_fy"])
                        else: continue
                    else:
                        if "test_true_fy" in res_s:
                            true = np.sqrt(np.array(res_s["test_true_fy"])**2 + np.array(res_s["test_true_fx"])**2)
                            pred = np.sqrt(np.array(res_s["test_pred_fy"])**2 + np.array(res_s["test_pred_fx"])**2)
                        else: continue
                    
                    err_pct = np.abs(pred - true) / (np.abs(true) + 100.0) * 100.0
                    plt.hist(err_pct.flatten(), bins=50, alpha=0.3, label=f"{m_type} (Mean:{np.mean(err_pct):.2f}%)", range=(0, 40))
            plt.title(f"Test Percent Error (%) - {study_name}"); plt.legend(); plt.grid(True)

            # 4. 测试集绝对误差对比
            plt.subplot(2, 2, 4)
            for m_type in STUDY_CONFIG["models"]:
                if m_type in all_results[study_name][axle]:
                    res_s = all_results[study_name][axle][m_type][1]
                    if axle == 'front':
                        if "test_true" in res_s:
                            true, pred = np.array(res_s["test_true"]), np.array(res_s["test_pred"])
                        elif "test_true_fy" in res_s: # 兼容旧版本
                            true, pred = np.array(res_s["test_true_fy"]), np.array(res_s["test_pred_fy"])
                        else: continue
                    else:
                        if "test_true_fy" in res_s:
                            true = np.sqrt(np.array(res_s["test_true_fy"])**2 + np.array(res_s["test_true_fx"])**2)
                            pred = np.sqrt(np.array(res_s["test_pred_fy"])**2 + np.array(res_s["test_pred_fx"])**2)
                        else: continue
                    
                    err_abs = np.abs(pred - true)
                    plt.hist(err_abs.flatten(), bins=50, alpha=0.3, label=f"{m_type} (Mean:{np.mean(err_abs):.1f}N)", range=(0, 1200))
            plt.title(f"Test Absolute Error (N) - {study_name}"); plt.legend(); plt.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(STUDY_CONFIG["base_output_dir"], f"comparison_{study_name}_{axle}.png"))
            plt.close()
    
    print(f"✅ All comparison plots generated in {STUDY_CONFIG['base_output_dir']}")

if __name__ == "__main__":
    plot_comparisons_from_disk()

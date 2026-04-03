import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
                
                if hist is not None and res is not None:
                    all_results[study_name][axle][m_type] = (hist, res)
                    print(f"✅ Loaded data for {study_name} - {axle} - {m_type}")
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
            plotted_loss = False
            for m_type in STUDY_CONFIG["models"]:
                if m_type in all_results["Baseline"][axle]:
                    loss_data = np.array(all_results["Baseline"][axle][m_type][0]["loss"])
                    # 过滤掉非正值以防 log 坐标系报错
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
            
            save_path = os.path.join(STUDY_CONFIG["base_output_dir"], f"1_baseline_comparison_{axle}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"📊 Saved: {save_path}")

        # ② 数量影响与外延分析
        for study_name in ["DataVolume", "Extrapolation"]:
            if not all_results[study_name][axle]: continue
            
            if study_name == "DataVolume":
                # DataVolume 特殊处理：均值对比柱状图
                plt.figure(figsize=(18, 6))
                plt.suptitle(f"Data Volume Impact Analysis - {axle.capitalize()} Axle (Mean Error Comparison)", fontsize=16)
                
                metrics = ["Train Absolute Error (N)", "Test Percent Error (%)", "Test Absolute Error (N)"]
                x = np.arange(len(STUDY_CONFIG["models"]))
                width = 0.35
                
                for i, metric in enumerate(metrics):
                    plt.subplot(1, 3, i + 1)
                    base_means = []
                    vol_means = []
                    
                    for m_type in STUDY_CONFIG["models"]:
                        if m_type in all_results["Baseline"][axle] and m_type in all_results["DataVolume"][axle]:
                            res_b = all_results["Baseline"][axle][m_type][1]
                            res_v = all_results["DataVolume"][axle][m_type][1]
                            
                            def get_metrics(res, axle_type):
                                if axle_type == 'front':
                                    t_true = np.array(res.get("train_true", res.get("train_true_fy", [])))
                                    t_pred = np.array(res.get("train_pred", res.get("train_pred_fy", [])))
                                    te_true = np.array(res.get("test_true", res.get("test_true_fy", [])))
                                    te_pred = np.array(res.get("test_pred", res.get("test_pred_fy", [])))
                                else:
                                    t_true = np.sqrt(np.array(res["train_true_fy"])**2 + np.array(res["train_true_fx"])**2)
                                    t_pred = np.sqrt(np.array(res["train_pred_fy"])**2 + np.array(res["train_pred_fx"])**2)
                                    te_true = np.sqrt(np.array(res["test_true_fy"])**2 + np.array(res["test_true_fx"])**2)
                                    te_pred = np.sqrt(np.array(res["test_pred_fy"])**2 + np.array(res["test_pred_fx"])**2)
                                
                                err_t_abs = np.mean(np.abs(t_pred - t_true))
                                err_te_pct = np.mean(np.abs(te_pred - te_true) / (np.abs(te_true) + 100.0) * 100.0)
                                err_te_abs = np.mean(np.abs(te_pred - te_true))
                                return [err_t_abs, err_te_pct, err_te_abs]

                            m_b = get_metrics(res_b, axle)
                            m_v = get_metrics(res_v, axle)
                            base_means.append(m_b[i])
                            vol_means.append(m_v[i])
                        else:
                            base_means.append(0); vol_means.append(0)

                    bars1 = plt.bar(x - width/2, base_means, width, label='Baseline (100%)', color='skyblue', alpha=0.8)
                    bars2 = plt.bar(x + width/2, vol_means, width, label='DataVolume (60%)', color='salmon', alpha=0.8)
                    
                    # 在柱状图上标注数值
                    def autolabel(bars):
                        for bar in bars:
                            height = bar.get_height()
                            plt.annotate(f'{height:.1f}',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3),  # 3 points vertical offset
                                        textcoords="offset points",
                                        ha='center', va='bottom', fontsize=9)
                    autolabel(bars1); autolabel(bars2)
                    
                    plt.xticks(x, STUDY_CONFIG["models"])
                    plt.title(metric); plt.legend(); plt.grid(axis='y', alpha=0.3)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                save_path = os.path.join(STUDY_CONFIG["base_output_dir"], f"comparison_DataVolume_{axle}.png")
                plt.savefig(save_path)
                plt.close()
                print(f"📊 Saved: {save_path}")
                continue # 跳过常规 DataVolume 绘图

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
            hist_range_pct = (0, 40) if study_name == "DataVolume" else (0, 100)
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
                    plt.hist(err_pct.flatten(), bins=50, alpha=0.3, label=f"{m_type} (Mean:{np.mean(err_pct):.2f}%)", range=hist_range_pct)
            plt.title(f"Test Percent Error (%) - {study_name}"); plt.legend(); plt.grid(True)

            # 4. 测试集绝对误差对比
            plt.subplot(2, 2, 4)
            hist_range_abs = (0, 1200) if study_name == "DataVolume" else (0, 3000)
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
                    plt.hist(err_abs.flatten(), bins=50, alpha=0.3, label=f"{m_type} (Mean:{np.mean(err_abs):.1f}N)", range=hist_range_abs)
            plt.title(f"Test Absolute Error (N) - {study_name}"); plt.legend(); plt.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(STUDY_CONFIG["base_output_dir"], f"comparison_{study_name}_{axle}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"📊 Saved: {save_path}")

        # ③ 散点图绘制 (Fy-Alpha 或 Ftot-Kappa)
        for study_name in ["Baseline", "DataVolume", "Extrapolation"]:
            if not all_results[study_name][axle]: continue
            
            plt.figure(figsize=(10, 6))
            plt.suptitle(f"{study_name} Scatter Plot - {axle.capitalize()} Axle (Blue: Train, Red: Test)", fontsize=16)
            
            # 由于同一种对比试验下不同模型的训练/测试集是一致的，只需取第一个模型的数据即可
            m_type = STUDY_CONFIG["models"][0]
            if m_type in all_results[study_name][axle]:
                res = all_results[study_name][axle][m_type][1]
                
                if axle == 'front':
                    # X-axis: Alpha
                    x_train = np.array(res.get("train_alpha", []))
                    x_test = np.array(res.get("test_alpha", []))
                    # Y-axis: Fy
                    y_train = np.array(res.get("train_true", []))
                    y_test = np.array(res.get("test_true", []))
                    x_label = "Side Slip Angle Alpha (rad)"
                    y_label = "Lateral Force Fy (N)"
                else:
                    # X-axis: Kappa
                    x_train = np.array(res.get("train_kappa", []))
                    x_test = np.array(res.get("test_kappa", []))
                    # Y-axis: Ftot = sqrt(Fx^2 + Fy^2)
                    y_train = np.sqrt(np.array(res.get("train_true_fy", []))**2 + np.array(res.get("train_true_fx", []))**2)
                    y_test = np.sqrt(np.array(res.get("test_true_fy", []))**2 + np.array(res.get("test_true_fx", []))**2)
                    x_label = "Combined Slip Kappa"
                    y_label = "Total Tire Force Ftot (N)"
                    plt.xlim(0, 10) # 限制后轴横轴范围为 0-10
                
                if len(x_train) > 0:
                    plt.scatter(x_train.flatten(), y_train.flatten(), s=2, alpha=0.3, color='blue', label='Train')
                if len(x_test) > 0:
                    plt.scatter(x_test.flatten(), y_test.flatten(), s=2, alpha=0.3, color='red', label='Test')
                
                plt.xlabel(x_label); plt.ylabel(y_label); plt.legend(); plt.grid(True)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(STUDY_CONFIG["base_output_dir"], f"scatter_{study_name}_{axle}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"📊 Saved Scatter: {save_path}")
    
    print(f"All comparison plots generated in {STUDY_CONFIG['base_output_dir']}")

if __name__ == "__main__":
    plot_comparisons_from_disk()

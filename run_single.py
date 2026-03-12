import os
from carsim_agent import CarsimAgent, CarsimVisualizer

# 配置路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(CURRENT_DIR, "Drifting_Template")
MODS_JSON_PATH = os.path.join(CURRENT_DIR, "temp_mods.json")

if __name__ == "__main__":
    # 实例化 Agent
    agent = CarsimAgent(template_path=TEMPLATE_DIR)
    
    # 设置选项
    agent.options.run_mode = "mods"
    agent.options.extract_data = True
    # 单次运行，保留详细结果
    agent.options.keep_detailed_results = True
    
    # 运行仿真
    print(f"\n🚀 Running Single Simulation with parameters in '{MODS_JSON_PATH}'...")
    result = agent.run(mods_json_path=MODS_JSON_PATH)
    
    if result and result.get("status") == "success":
        print(f"\n✅ Simulation finished successfully!")
        print(f"Results Path: {result.get('results_path')}")
        print(f"Log Path: {result.get('log_path')}")
    else:
        print(f"\n❌ Simulation failed. See logs for details.")

    # 对结果进行可视化
    if result and result.get("status") == "success":
        viz = CarsimVisualizer()
        results_path = result.get("results_path")
        
        # 如果 keep_detailed_results 为 True, results_path 指向文件夹，我们需要加载其中的 LastRun.csv
        if os.path.isdir(results_path):
            csv_path = os.path.join(results_path, "LastRun.csv")
        else:
            csv_path = results_path
            
        print(f"\n📊 Visualizing results from: {csv_path}")
        viz.load_data(csv_path)
        
        # 保存图像以避免阻塞
        viz.plot_variable_vs_time("Vx", save_path="run_single_vx.png")
        # 对R2的Fx和Fy进行可视化
        viz.plot_multi_variables_vs_time(["Fx_R2", "Fy_R2"], save_path="run_single_forces.png")
        viz.plot_xy("Alpha_R2", "Fy_R2", save_path="run_single_alpha_fy.png")
        viz.plot_xy("Kappa_R2", "Fx_R2", save_path="run_single_kappa_fx.png")
    else:
        print("\n⚠️ Skipping visualization due to simulation failure.")

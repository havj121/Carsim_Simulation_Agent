import os
import json
from carsim_agent import CarsimAgent, CarsimVisualizer

# 配置路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(CURRENT_DIR, "Drifting_Template")
MODS_JSON_PATH = os.path.join(CURRENT_DIR, "temp_mods.json")
# 参考运行目录
REFERENCE_DIR = r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Run_4df90ea9-2b3f-4916-84f7-8d33fe413b8f"
REFERENCE_CSV = os.path.join(REFERENCE_DIR, "LastRun.csv")

# 图像保存目录
VALIDATION_DIR = os.path.join(CURRENT_DIR, "Validation_Plots")

if __name__ == "__main__":
    if not os.path.exists(VALIDATION_DIR):
        os.makedirs(VALIDATION_DIR)

    # 1. 实例化 Agent 并运行当前仿真
    agent = CarsimAgent(template_path=TEMPLATE_DIR)
    agent.options.run_mode = "mods"
    agent.options.extract_data = True
    agent.options.keep_detailed_results = True
    
    print(f"\n🚀 Running current simulation with parameters in '{MODS_JSON_PATH}'...")
    result = agent.run(mods_json_path=MODS_JSON_PATH)
    
    if result and result.get("status") == "success":
        new_results_path = result.get("results_path")
        # 确定新运行的 CSV 路径
        if os.path.isdir(new_results_path):
            new_csv = os.path.join(new_results_path, "LastRun.csv")
        else:
            new_csv = new_results_path
            
        print(f"\n✅ Simulation finished. Comparing with: {REFERENCE_CSV}")
        
        # 2. 初始化可视化器并加载两个数据源
        viz = CarsimVisualizer()
        
        # 为了使用 plot_comparison，我们需要依次加载数据到 self.data
        # 但是 load_data 每次都会覆盖 self.data，除非我们稍微修改逻辑或手动合并。
        # 让我们手动处理一下数据的加载。
        viz.load_data(new_csv) # 加载为 Task 0
        task0_data = viz.data[0]
        
        viz.load_data(REFERENCE_CSV) # 加载为 Task 1
        task1_data = viz.data[0]
        
        # 合并数据以进行对比
        viz.data = [task0_data, task1_data]
        
        # 3. 确定对比变量 (使用默认提取变量，注意大小写修复)
        # 从 CarsimAgent 的默认设置中获取，但移除 Time
        compare_vars = [v for v in agent.options.default_extract_vars if v.lower() != "time"]
        # 修复 VX 为 Vx
        compare_vars = [v if v != "VX" else "Vx" for v in compare_vars]
        
        print(f"📊 Generating comparison plots for: {compare_vars}")
        
        labels = ["Current Run", "Reference Run"]
        for var in compare_vars:
            save_path = os.path.join(VALIDATION_DIR, f"compare_{var.lower()}.png")
            viz.plot_comparison(var, task_indices=[0, 1], labels=labels, save_path=save_path)
            
        print(f"\n✅ All comparison plots saved to: {VALIDATION_DIR}")
    else:
        print(f"\n❌ Current simulation failed. Cannot perform comparison.")

import os
from carsim_agent import CarsimVisualizer

def test_visualizer():
    # 1. 初始化可视化器
    viz = CarsimVisualizer()
    
    # 2. 路径配置 (使用最新的批量结果文件)
    batch_json = r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Batch_Run_20260312_112129\batch_results_summary.json"
    
    if not os.path.exists(batch_json):
        print(f"Test file not found: {batch_json}")
        return

    # 3. 加载数据 (JSON)
    print("\n--- Loading Batch Results ---")
    viz.load_data(batch_json)
    
    # 4. 获取任务和变量信息
    tasks = viz.get_available_tasks()
    print(f"Available tasks ({len(tasks)}):")
    for t in tasks:
        print(f"  Index {t['index']}: {t['task_id']} - Config: {t['config']}")
        
    vars = viz.get_available_variables(task_index=0)
    print(f"\nAvailable variables for comparison:\n  {vars}")

    # 5. 批量对比所有变量
    print("\n--- Generating Comparison Plots for All Variables ---")
    
    # 创建保存目录
    output_dir = "batch_comparison_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 获取所有任务索引
    task_indices = [t['index'] for t in tasks]
    
    # 生成有意义的任务标签 (例如: V90_Mu0.5_S90)
    labels = []
    for t in tasks:
        config = t['config']
        v = config.get('SPEED_TARGET_CONSTANT(1)', '?')
        mu = config.get('MU_ROAD_CONSTANT(1)', '?')
        s = config.get('STEER_SW_CONSTANT', '?')
        labels.append(f"V{v}_Mu{mu}_S{s}")

    # 遍历所有变量 (排除 Time)
    compare_vars = [v for v in vars if v.lower() != 'time']
    
    for var in compare_vars:
        # 文件名清理 (移除可能导致路径问题的字符)
        safe_name = var.replace("(", "_").replace(")", "_").lower()
        save_path = os.path.join(output_dir, f"compare_{safe_name}.png")
        
        print(f"Plotting comparison for: {var} ...")
        viz.plot_comparison(
            variable=var, 
            task_indices=task_indices, 
            labels=labels, 
            title=f"Batch Comparison: {var}",
            save_path=save_path
        )

    print(f"\n✅ All batch comparison plots saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    test_visualizer()

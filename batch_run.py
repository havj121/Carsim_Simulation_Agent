import os
import json
from carsim_agent import CarsimAgent

# 配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(CURRENT_DIR, "Drifting_Template")
BATCH_MODS_FILE = os.path.join(CURRENT_DIR, "batch_mods.json")

if __name__ == "__main__":
    # 1. 加载配置
    with open(BATCH_MODS_FILE, 'r', encoding='utf-8') as f:
        batch_configs = json.load(f)
        
    # 2. 实例化 Agent
    agent = CarsimAgent(template_path=TEMPLATE_DIR)
    
    # 3. 设置选项
    agent.options.run_mode = "mods"
    agent.options.extract_data = True
    # 设置批量运行时不保留每个任务的详细结果文件夹，以节省空间
    agent.options.keep_detailed_results = False
    
    # 4. 调用集成的批量运行方法
    # 默认并行进程数为 4
    results = agent.run_batch(batch_configs, processes=4)
    
    print(f"\nBatch task completed. Total tasks: {len(results)}")

import os
from carsim_agent import CarsimAgent

if __name__ == "__main__":
    # 配置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(current_dir, "Drifting_Template")
    
    # 实例化智能体
    agent = CarsimAgent(template_path=template_dir)
    
    # # 模式 1: 通过 mods_json_path 直接运行
    json_path = os.path.join(current_dir, "mods_speed120_mu06.json")
    print("\n--- Testing JSON Mode ---")
    try:
        agent.run(mods_json_path=json_path)
        
    except Exception as e:
        print(f"JSON Mode failed as expected: {e}")
    
    # # 模式 2: 通过 prompt 运行 (原有模式)
    # print("\n--- Testing Prompt Mode ---")
    # user_prompt = "运行Carsim仿真，将车速设置为常数100 km/h, 道路摩擦系数设置为 0.8"
    # agent.run(prompt=user_prompt)

    # 模式 3: 数据提取测试 (extract_data=True)
    # print("\n--- Testing Data Extraction Mode ---")
    # extract_prompt = "运行仿真并提取数据：车速110 km/h"
    # # 自定义提取变量
    # custom_vars = ["Time", "VX", "Steer_SW"]
    # agent.run(prompt=extract_prompt, extract_data=True, extract_vars=custom_vars)

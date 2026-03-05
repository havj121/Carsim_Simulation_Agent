import os
from carsim_agent import CarsimAgent

if __name__ == "__main__":
    # 配置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(current_dir, "Drifting_Template")
    
    # 实例化智能体
    agent = CarsimAgent(template_path=template_dir)
    
    # 模式 1: 通过 mods_json_path 直接运行
    json_path = os.path.join(current_dir, "mods_speed120_mu06.json")
    print("\n--- Testing JSON Mode ---")
    agent.run(mods_json_path=json_path)
    
    # 模式 2: 通过 prompt 运行 (原有模式)
    # print("\n--- Testing Prompt Mode ---")
    # user_prompt = "运行Carsim仿真，将车速设置为常数100 km/h, 道路摩擦系数设置为 0.8"
    # agent.run(prompt=user_prompt)

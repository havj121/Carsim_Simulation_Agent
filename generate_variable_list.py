import pandas as pd
import os

def generate_excel():
    # 整理变量列表
    variable_data = [
        # 车辆整体运动学/动力学变量
        {"变量名符号": "Time", "含义": "仿真时间", "单位": "s"},
        {"变量名符号": "Vx", "含义": "质心纵向速度", "单位": "km/h"},
        {"变量名符号": "Ax", "含义": "质心纵向加速度", "单位": "g"},
        {"变量名符号": "Ay", "含义": "质心横向加速度", "单位": "g"},
        {"变量名符号": "AV_Y", "含义": "横摆角速度 (Yaw Rate)", "单位": "deg/s"},
        {"变量名符号": "Beta", "含义": "质心侧偏角 (Side Slip Angle)", "单位": "deg"},
        
        # 转向系统变量
        {"变量名符号": "Steer_L1", "含义": "左前轮 (L1) 转向角", "单位": "deg"},
        {"变量名符号": "Steer_R1", "含义": "右前轮 (R1) 转向角", "单位": "deg"},
        {"变量名符号": "Steer_L2", "含义": "左后轮 (L2) 转向角", "单位": "deg"},
        {"变量名符号": "Steer_R2", "含义": "右后轮 (R2) 转向角", "单位": "deg"},
        {"变量名符号": "Steer_SW", "含义": "方向盘转角", "单位": "deg"},

        # 轮胎力学变量 (按车轮 L1, R1, L2, R2)
        {"变量名符号": "Alpha_L1/R1/L2/R2", "含义": "各轮轮胎侧偏角", "单位": "deg"},
        {"变量名符号": "Kappa_L1/R1/L2/R2", "含义": "各轮轮胎纵向滑移率", "单位": "-"},
        {"变量名符号": "Fx_L1/R1/L2/R2", "含义": "各轮轮胎纵向力", "单位": "N"},
        {"变量名符号": "Fy_L1/R1/L2/R2", "含义": "各轮轮胎横向力 (侧偏力)", "单位": "N"},
        {"变量名符号": "Fz_L1/R1/L2/R2", "含义": "各轮轮胎垂向载荷", "单位": "N"},
        {"变量名符号": "AVy_L1/R1/L2/R2", "含义": "各轮车轮旋转角速度", "单位": "rpm"},
        {"变量名符号": "Mz_L1/R1/L2/R2", "含义": "各轮轮胎回正力矩", "单位": "N-m"},

        # 环境/控制变量
        {"变量名符号": "MU_ROAD", "含义": "路面附着系数 (通常通过配置获取)", "单位": "-"},
        {"变量名符号": "Throttle", "含义": "油门开度", "单位": "-"},
        {"变量名符号": "Brake", "含义": "制动压力或踏板行程", "单位": "MPa/mm"}
    ]

    # 为了使表格更详细，展开具体的轮位
    detailed_data = []
    wheels = ['L1', 'R1', 'L2', 'R2']
    
    for item in variable_data:
        if "/" in item["变量名符号"]:
            prefix = item["变量名符号"].split("_")[0]
            for w in wheels:
                detailed_data.append({
                    "变量名符号": f"{prefix}_{w}",
                    "含义": item["含义"].replace("各轮", f"{w}轮"),
                    "单位": item["单位"]
                })
        else:
            detailed_data.append(item)

    df = pd.DataFrame(detailed_data)
    
    # 保存为 Excel
    output_path = "CarSim_Extracted_Variables.xlsx"
    df.to_excel(output_path, index=False)
    print(f"✅ 变量列表已成功生成: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_excel()

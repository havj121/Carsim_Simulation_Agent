import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict, Any

class CarsimVisualizer:
    """
    仿真结果可视化类，支持读取批量 JSON 结果和单次 CSV 结果。
    """
    def __init__(self):
        self.data: List[Dict[str, Any]] = []  # 存储加载的数据，统一为列表格式

    def load_data(self, file_path: str):
        """
        加载仿真数据文件 (JSON 或 CSV)。
        :param file_path: 数据文件路径
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                if isinstance(raw_data, list):
                    # 批量结果格式
                    self.data = raw_data
                else:
                    # 单个结果格式包装成列表
                    self.data = [raw_data]
            print(f"Successfully loaded batch data from JSON: {file_path}")
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            # 包装成与 JSON 类似的格式
            self.data = [{
                "task_id": "Single_Run",
                "config": {},
                "status": "success",
                "extracted_data": df.to_dict(orient='records')
            }]
            print(f"Successfully loaded single run data from CSV: {file_path}")
        else:
            raise ValueError(f"Unsupported file format: {ext}. Support .json and .csv")

    def _get_df(self, task_index: int) -> pd.DataFrame:
        """获取指定任务索引的 DataFrame"""
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        if task_index >= len(self.data):
            raise IndexError(f"Task index {task_index} out of range (total tasks: {len(self.data)})")
        
        extracted = self.data[task_index].get("extracted_data")
        if not extracted:
            raise ValueError(f"No extracted_data found for task index {task_index}")
        
        return pd.DataFrame(extracted)

    def plot_variable_vs_time(self, variable: str, task_index: int = 0, title: str = None, save_path: str = None):
        """
        可视化单变量随时间的变化曲线。
        :param variable: 变量名 (如 'Vx', 'Steer_SW')
        :param task_index: 任务索引 (默认为 0)
        :param title: 图像标题
        :param save_path: 保存图像的路径 (可选)
        """
        df = self._get_df(task_index)
        if 'Time' not in df.columns:
            raise ValueError("'Time' column not found in data.")
        if variable not in df.columns:
            raise ValueError(f"Variable '{variable}' not found. Available: {df.columns.tolist()}")

        plt.figure(figsize=(10, 6))
        plt.plot(df['Time'], df[variable], label=variable)
        plt.xlabel('Time (s)')
        plt.ylabel(variable)
        plt.title(title if title else f"{variable} vs Time (Task {task_index})")
        plt.grid(True)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

    def plot_multi_variables_vs_time(self, variables: List[str], task_index: int = 0, title: str = None, save_path: str = None):
        """
        在同一图像中可视化多个变量随时间的变化曲线。
        :param variables: 变量名列表
        :param task_index: 任务索引
        :param title: 图像标题
        :param save_path: 保存图像的路径 (可选)
        """
        df = self._get_df(task_index)
        if 'Time' not in df.columns:
            raise ValueError("'Time' column not found in data.")

        plt.figure(figsize=(10, 6))
        for var in variables:
            if var in df.columns:
                plt.plot(df['Time'], df[var], label=var)
            else:
                print(f"Warning: Variable '{var}' not found and skipped.")

        plt.xlabel('Time (s)')
        plt.ylabel('Values')
        plt.title(title if title else f"Multiple Variables vs Time (Task {task_index})")
        plt.grid(True)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

    def plot_xy(self, x_var: str, y_var: str, task_index: int = 0, title: str = None, scatter: bool = True, save_path: str = None):
        """
        可视化 X 轴变量与 Y 轴变量的关系 (如 X-Y 轨迹)。
        :param x_var: X 轴变量名
        :param y_var: Y 轴变量名
        :param task_index: 任务索引
        :param title: 图像标题
        :param scatter: 是否使用散点图 (默认为 True)
        :param save_path: 保存图像的路径 (可选)
        """
        df = self._get_df(task_index)
        if x_var not in df.columns or y_var not in df.columns:
            raise ValueError(f"Required variables not found. Available: {df.columns.tolist()}")

        plt.figure(figsize=(10, 6))
        if scatter:
            plt.scatter(df[x_var], df[y_var], label=f"{y_var} vs {x_var}", s=10)
        else:
            plt.plot(df[x_var], df[y_var], label=f"{y_var} vs {x_var}")
            
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.title(title if title else f"{y_var} vs {x_var} (Task {task_index})")
        plt.grid(True)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

    def plot_comparison(self, variable: str, task_indices: List[int], labels: List[str] = None, title: str = None, save_path: str = None):
        """
        在同一图像中对比多个任务的同一变量。
        :param variable: 变量名
        :param task_indices: 任务索引列表
        :param labels: 每个任务的标签列表 (默认为 Task X)
        :param title: 图像标题
        :param save_path: 保存路径
        """
        if not self.data:
            raise ValueError("No data loaded.")

        plt.figure(figsize=(10, 6))
        
        for i, idx in enumerate(task_indices):
            df = self._get_df(idx)
            label = labels[i] if labels and i < len(labels) else f"Task {idx}"
            
            if 'Time' in df.columns and variable in df.columns:
                plt.plot(df['Time'], df[variable], label=label)
            else:
                print(f"Warning: Variable '{variable}' or 'Time' not found in task {idx}.")

        plt.xlabel('Time (s)')
        plt.ylabel(variable)
        plt.title(title if title else f"Comparison: {variable}")
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Comparison plot saved to: {save_path}")
        else:
            plt.show()

    def get_available_tasks(self):
        """获取当前已加载任务的信息列表"""
        return [{"index": i, "task_id": d.get("task_id"), "config": d.get("config")} for i, d in enumerate(self.data)]

    def get_available_variables(self, task_index: int = 0):
        """获取指定任务的可选变量名列表"""
        df = self._get_df(task_index)
        return df.columns.tolist()

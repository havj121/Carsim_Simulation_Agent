import os
import sys
import time

# 将当前目录添加到 sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# 从 config.py 读取路径并处理 DLL 路径（针对 Python 3.8+）
try:
    from config import MATLAB_BIN_PATH
    if os.path.exists(MATLAB_BIN_PATH):
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(MATLAB_BIN_PATH)
        else:
            os.environ['PATH'] = MATLAB_BIN_PATH + os.pathsep + os.environ.get('PATH', '')
except ImportError:
    print("⚠️ Warning: Could not import config.py. Path setup might be incomplete.")

# 初始化 Matlab/Simulink 引擎
try:
    import matlab.engine
except ImportError:
    print("❌ MATLAB Python Engine not found. Please install it first.")
    from setup_matlab_engine import setup_matlab_engine
    if setup_matlab_engine(force=True):
        import matlab.engine
        print("✅ MATLAB Python Engine installed successfully.")
    else:
        print("❌ MATLAB Python Engine installation failed.")
        sys.exit(1)

def run_simulink_setup():
    print("🚀 Starting MATLAB engine...")
    try:
        eng = matlab.engine.start_matlab()
        print("✅ MATLAB engine started.")
    except Exception as e:
        print(f"❌ Failed to start MATLAB engine: {e}")
        return

    # 模板文件名和路径
    template_dir = os.path.join(CURRENT_DIR, "Template")
    template_slx = os.path.join(template_dir, "carsim_template_auto.slx")
    
    if not os.path.exists(template_slx):
        print(f"🔍 Template file not found. Building it now...")
        eng.cd(template_dir, nargout=0)
        
        try:
            print(f"🛠️ Executing MATLAB function 'build_carsim_simulink_template'...")
            eng.eval("build_carsim_simulink_template", nargout=0)
            
            # 显式保存
            slx_path = template_slx.replace("\\", "/")
            eng.save_system('carsim_template_auto', slx_path, nargout=0)
            eng.close_system('carsim_template_auto', nargout=0)
            
            time.sleep(2)
            if os.path.exists(template_slx):
                print(f"✅ Template file created successfully at: {template_slx}")
            else:
                print("⚠️ Build script finished but .slx file was not found.")
        except Exception as e:
            print(f"❌ Error during MATLAB build: {e}")
    else:
        print(f"✅ Template file already exists: {template_slx}")
        
    # 示例：运行一个简单的命令
    res = eng.sqrt(42.0)
    print(f"MATLAB Test: sqrt(42) = {res}")
    
    print("🚪 Closing MATLAB engine...")
    eng.quit()

if __name__ == "__main__":
    run_simulink_setup()

import os
import sys
import io
import time
import matlab.engine

# 配置目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

LOG_FILE = os.path.join(CURRENT_DIR, "debug_log.txt")

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    print(msg)
    sys.stdout.flush()

# 处理 MATLAB DLL 路径
try:
    from config import MATLAB_BIN_PATH
    if os.path.exists(MATLAB_BIN_PATH) and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(MATLAB_BIN_PATH)
except ImportError:
    pass

def run_simulink_setup(version="v1"):
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
    log(f"Starting MATLAB engine for version: {version}...")
    
    try:
        eng = matlab.engine.start_matlab()
        log("✅ MATLAB engine started.")
    except Exception as e:
        log(f"❌ Failed to start MATLAB engine: {e}")
        return

    model_name = f"carsim_template_{version}"
    template_dir = os.path.join(CURRENT_DIR, "Template")
    slx_path = os.path.join(template_dir, f"{model_name}.slx").replace("\\", "/")
    
    log(f"🛠️ Building model: {model_name}")
    eng.cd(template_dir, nargout=0)
    
    out, err = io.StringIO(), io.StringIO()
    try:
        func_name = f"build_carsim_simulink_template_{version}"
        log(f"Executing: {func_name}...")
        
        eng.eval(f"status = {func_name}('{model_name}', '{slx_path}')", nargout=0, stdout=out, stderr=err)
        status = eng.workspace['status']
        
        if out.getvalue(): log(f"MATLAB Output:\n{out.getvalue()}")
        if err.getvalue(): log(f"MATLAB Error:\n{err.getvalue()}")
        
        if status:
            log(f"✅ SUCCESS: Model saved at {slx_path}")
        else:
            log("❌ FAILURE: MATLAB function returned error status.")
            
    except Exception as e:
        log(f"❌ CRITICAL ERROR during build: {e}")
    finally:
        log("🚪 Closing MATLAB engine...")
        time.sleep(1)
        eng.quit()

if __name__ == "__main__":
    # 可选参数: "v1", "v2" 或 "v3"
    target_version = "v2"
    run_simulink_setup(version=target_version)

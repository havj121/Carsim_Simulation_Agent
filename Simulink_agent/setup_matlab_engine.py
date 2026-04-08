import os
import sys
import subprocess

def setup_matlab_engine(python_exe=None):
    """
    Check if MATLAB Python Engine is installed. If not, install it.
    This script strictly checks for Python version compatibility.
    If 'python_exe' is provided, it will use that interpreter for installation.
    """
    # 1. Get path from config.py
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from config import MATLAB_PYTHON_ENGINE_PATH, TARGET_PYTHON_EXE
        if python_exe is None:
            python_exe = TARGET_PYTHON_EXE
            
        print(f"🔍 Using Python interpreter: {python_exe}")
        print(f"📂 MATLAB Engine source path: {MATLAB_PYTHON_ENGINE_PATH}")
    except (ImportError, AttributeError) as e:
        print(f"❌ Error: Could not read configuration from config.py: {e}")
        return False

    # 2. Check if already installed in the TARGET interpreter
    try:
        # Run a small script using the target python to check if matlab.engine is available
        check_cmd = [python_exe, "-c", "import matlab.engine; print('SUCCESS')"]
        result = subprocess.run(check_cmd, capture_output=True, text=True)
        if "SUCCESS" in result.stdout:
            print(f"✅ MATLAB Python Engine is already installed in {python_exe}.")
            return True
        else:
            print(f"❌ MATLAB Python Engine not found in {python_exe}. Starting installation...")
    except Exception as e:
        print(f"⚠️ Could not verify installation in target interpreter: {e}")

    # 3. Verify source path exists
    if not os.path.exists(MATLAB_PYTHON_ENGINE_PATH):
        print(f"❌ Error: The path '{MATLAB_PYTHON_ENGINE_PATH}' does not exist.")
        return False

    # 4. Get Python version of the target interpreter and check compatibility
    try:
        ver_cmd = [python_exe, "-c", "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')"]
        py_version_dot = subprocess.check_output(ver_cmd, text=True).strip()
        print(f"ℹ️ Target Python version: {py_version_dot}")
    except Exception as e:
        print(f"❌ Error getting target Python version: {e}")
        return False

    supported_dot = ["2.7", "3.8", "3.9", "3.10"]
    
    if py_version_dot not in supported_dot:
        print(f"❌ Error: Python version {py_version_dot} is not supported by MATLAB 2022b.")
        print(f"💡 Officially supported versions: {', '.join(supported_dot)}")
        print("🛑 Installation aborted due to version mismatch.")
        return False

    # 5. Run installation
    print(f"🚀 Installing into {python_exe}...")
    try:
        # IMPORTANT: Use the specified python_exe here
        cmd = [python_exe, "setup.py", "install", "--user"]
        result = subprocess.run(cmd, cwd=MATLAB_PYTHON_ENGINE_PATH, capture_output=True, text=True, check=True)
        print("✅ Installation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed:\n{e.stderr}")
        return False


if __name__ == "__main__":
    if setup_matlab_engine():
        print("\n✨ Setup completed successfully.")
    else:
        print("\n❌ Setup failed.")
        sys.exit(1)

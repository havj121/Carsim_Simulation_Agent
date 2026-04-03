import os
import sys
import subprocess
import shutil
import tempfile
import re

def setup_matlab_engine(force=True, python_exe=None):
    """
    Check if MATLAB Python Engine is installed. If not, install it.
    If 'force' is True, it will attempt to bypass Python version checks by patching a temp copy of setup.py and matlab/__init__.py.
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

    # 4. Get Python version of the target interpreter
    try:
        ver_cmd = [python_exe, "-c", "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')"]
        py_version_dot = subprocess.check_output(ver_cmd, text=True).strip()
        py_version_underscore = py_version_dot.replace(".", "_")
        print(f"ℹ️ Target Python version: {py_version_dot}")
    except Exception as e:
        print(f"❌ Error getting target Python version: {e}")
        return False

    supported_dot = ["2.7", "3.8", "3.9", "3.10"]
    supported_underscore = ["2_7", "3_8", "3_9", "3_10"]
    
    install_path = MATLAB_PYTHON_ENGINE_PATH
    temp_dir = None

    # Patch if version not supported
    if py_version_dot not in supported_dot:
        if force:
            print(f"⚠️ Warning: {py_version_dot} is not officially supported. Patching...")
            try:
                temp_dir = tempfile.mkdtemp(prefix="matlab_engine_patch_")
                shutil.copytree(MATLAB_PYTHON_ENGINE_PATH, temp_dir, dirs_exist_ok=True)
                
                # Patch setup.py
                setup_file = os.path.join(temp_dir, "setup.py")
                if os.path.exists(setup_file):
                    with open(setup_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    new_supported = supported_dot + [py_version_dot]
                    content = re.sub(r"(_supported_versions\s*=\s*\[)(.*?)(\])", 
                                    rf"\1{', '.join([repr(v) for v in new_supported])}\3", content)
                    content = content.replace("_cwd = os.getcwd()", f"_cwd = r'{MATLAB_PYTHON_ENGINE_PATH}'")
                    with open(setup_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print("✅ Patched setup.py")
                
                # Patch matlab/__init__.py
                init_file = os.path.join(temp_dir, "dist", "matlab", "__init__.py")
                if os.path.exists(init_file):
                    with open(init_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    new_supported_u = supported_underscore + [py_version_underscore]
                    content = re.sub(r"(_supported_versions\s*=\s*\[)(.*?)(\])", 
                                    rf"\1{', '.join([repr(v) for v in new_supported_u])}\3", content)
                    with open(init_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print("✅ Patched matlab/__init__.py")
                
                install_path = temp_dir
            except Exception as e:
                print(f"❌ Patching failed: {e}")
                return False
        else:
            print("❌ Version mismatch. Installation aborted.")
            return False

    # 5. Run installation
    print(f"🚀 Installing into {python_exe}...")
    try:
        # IMPORTANT: Use the specified python_exe here
        cmd = [python_exe, "setup.py", "install", "--user"]
        result = subprocess.run(cmd, cwd=install_path, capture_output=True, text=True, check=True)
        print("✅ Installation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed:\n{e.stderr}")
        return False
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    if setup_matlab_engine(force=True):
        print("\n✨ Setup completed successfully.")
    else:
        print("\n❌ Setup failed.")
        sys.exit(1)

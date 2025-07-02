import os
import platform
import subprocess
import shutil

def install_dependencies():
    """Install required system and Python packages."""
    print("Installing required dependencies...")
    
    # Install system packages if on Linux
    if platform.system() == "Linux":
        try:
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "install", "-y", "python3-dev"], check=True)
        except subprocess.CalledProcessError:
            print("Warning: Could not install system packages. You may need to run 'sudo apt-get install python3-dev' manually.")
    
    # Install Python packages
    subprocess.run(["pip", "install", "pyinstaller"], check=True)

def build_executable():
    """Build executable for the current platform."""
    print("Building Side by Side Comparison Tool...")
    
    # Install dependencies
    install_dependencies()
    
    # Determine platform-specific settings
    is_windows = platform.system() == "Windows"
    executable_name = "comparison_tool.exe" if is_windows else "comparison_tool"
    icon_param = ["--icon=assets/icon.ico"] if is_windows else []
    
    # Use correct path separator based on platform
    separator = ";" if is_windows else ":"
    
    # Create build command
    cmd = [
        "pyinstaller",
        f"--name={executable_name}",
        "--onefile",
        "--noconsole",
        f"--add-data=sample_data{separator}sample_data",  # Include sample data
        f"--add-data=reports{separator}reports",          # Include reports directory
        f"--add-data=utils{separator}utils",             # Include utils
        "--hidden-import=streamlit",
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--hidden-import=openpyxl",
        "--hidden-import=xlsxwriter",
        *icon_param,
        "app.py"
    ]
    
    # Print command for debugging
    print("Running PyInstaller with command:", " ".join(cmd))
    
    # Run PyInstaller
    subprocess.run(cmd, check=True)
    
    # Copy necessary files to dist directory
    dist_dir = "dist"
    files_to_copy = [
        "requirements.txt",
        ".env.example",
        "README.md"
    ]
    
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(dist_dir, file))
    
    # Create empty reports directory in dist
    os.makedirs(os.path.join(dist_dir, "reports"), exist_ok=True)
    
    print(f"\nBuild complete! Executable created in {dist_dir}/{executable_name}")
    print("Copy the entire 'dist' directory to share with team members.")

if __name__ == "__main__":
    build_executable()

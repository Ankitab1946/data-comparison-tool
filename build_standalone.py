"""Build script for creating standalone executable."""
import os
import platform
import subprocess
import shutil
from pathlib import Path

def create_standalone():
    """Create standalone executable with all dependencies bundled."""
    print("Creating standalone executable...")
    
    # Install required packages
    print("Installing required packages...")
    subprocess.run(["pip", "install", "pyinstaller"], check=True)
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
    
    # Determine platform-specific settings
    is_windows = platform.system() == "Windows"
    exe_name = "comparison_tool.exe" if is_windows else "comparison_tool"
    separator = ";" if is_windows else ":"
    
    # Create dist directory
    dist_dir = Path("dist")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir()
    
    # Create the executable
    print("Building executable...")
    cmd = [
        "pyinstaller",
        "--name", exe_name,
        "--onefile",  # Create a single executable
        "--noconsole",  # Don't show console window
        f"--add-data=sample_data{separator}sample_data",
        f"--add-data=reports{separator}reports",
        f"--add-data=utils{separator}utils",
        "--hidden-import=streamlit",
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--hidden-import=openpyxl",
        "--hidden-import=xlsxwriter",
        "--collect-all=streamlit",
        "app.py"
    ]
    
    subprocess.run(cmd, check=True)
    
    # Create distribution package
    package_dir = dist_dir / "comparison_tool_package"
    package_dir.mkdir(exist_ok=True)
    
    # Copy executable and support files
    shutil.copy2(dist_dir / exe_name, package_dir)
    shutil.copy2(".env.example", package_dir / ".env.example")
    
    # Create a simplified README
    with open(package_dir / "README.txt", "w") as f:
        f.write("""Side by Side Comparison Tool
=========================

Quick Start Guide:
1. Extract all files from this ZIP
2. For SQL Server connectivity:
   - Copy .env.example to .env
   - Edit .env with your database settings

Windows Users:
- Double-click launch.bat
- The application will open in your browser

Linux/Mac Users:
- Open terminal in this directory
- Run: chmod +x launch.sh
- Run: ./launch.sh
- The application will open in your browser

No Python installation needed! Everything is included in the executable.

Troubleshooting:
- Make sure port 8000 is not in use
- Check .env file if using SQL Server
- Wait 30 seconds for first startup
""")
    
    # Create launcher scripts
    if is_windows:
        with open(package_dir / "launch.bat", "w") as f:
            f.write(f"""@echo off
echo Starting Comparison Tool...
start {exe_name}
echo Waiting for application to start...
timeout /t 5
start http://localhost:8000
""")
    else:
        with open(package_dir / "launch.sh", "w") as f:
            f.write(f"""#!/bin/bash
echo "Starting Comparison Tool..."
./{exe_name} &
echo "Waiting for application to start..."
sleep 5
xdg-open http://localhost:8000 2>/dev/null || open http://localhost:8000 2>/dev/null || echo "Please open http://localhost:8000 in your browser"
""")
        os.chmod(package_dir / "launch.sh", 0o755)
    
    # Create ZIP archive
    shutil.make_archive(dist_dir / "comparison_tool_standalone", "zip", package_dir)
    
    print(f"""
Standalone executable package created successfully!

Distribution package is available at:
1. ZIP file: {dist_dir}/comparison_tool_standalone.zip
2. Directory: {dist_dir}/comparison_tool_package

Instructions for team members:
1. Download and extract comparison_tool_standalone.zip
2. No Python installation needed!
3. Windows users:
   - Double-click launch.bat
4. Linux/Mac users:
   - Open terminal in the extracted directory
   - Run: chmod +x launch.sh
   - Run: ./launch.sh
5. The application will open in your default web browser

For SQL Server connectivity:
1. Copy .env.example to .env
2. Edit .env with your database settings
""")

if __name__ == "__main__":
    create_standalone()

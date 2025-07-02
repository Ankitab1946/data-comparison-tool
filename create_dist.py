import os
import shutil
import platform
import subprocess
from pathlib import Path

def create_distribution():
    """Create a distributable package with virtual environment."""
    print("Creating distributable package...")
    
    # Create dist directory
    dist_dir = Path("dist")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir()
    
    # Create app directory structure
    app_dir = dist_dir / "comparison_tool"
    app_dir.mkdir()
    
    # Copy necessary files and directories
    files_to_copy = [
        "app.py",
        "requirements.txt",
        ".env.example",
        "README.md"
    ]
    
    dirs_to_copy = [
        "reports",
        "sample_data",
        "utils"
    ]
    
    # Copy files
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, app_dir)
            
    # Copy directories
    for dir_name in dirs_to_copy:
        if os.path.exists(dir_name):
            shutil.copytree(dir_name, app_dir / dir_name)
    
    # Create platform-specific launch scripts
    if platform.system() == "Windows":
        # Windows batch file
        with open(app_dir / "run.bat", "w") as f:
            f.write("""@echo off
echo Creating virtual environment...
python -m venv venv
call venv\\Scripts\\activate.bat
echo Installing requirements...
pip install -r requirements.txt
echo Starting application...
streamlit run app.py --server.port 8000
pause
""")
    else:
        # Unix shell script
        with open(app_dir / "run.sh", "w") as f:
            f.write("""#!/bin/bash
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "Installing requirements..."
pip install -r requirements.txt
echo "Starting application..."
streamlit run app.py --server.port 8000
""")
        # Make shell script executable
        os.chmod(app_dir / "run.sh", 0o755)
    
    # Create ZIP archive
    shutil.make_archive(dist_dir / "comparison_tool", 'zip', dist_dir, "comparison_tool")
    
    print(f"""
Distribution package created successfully!

The package is available in two formats:
1. ZIP archive: {dist_dir}/comparison_tool.zip
2. Directory: {dist_dir}/comparison_tool

To share with team members:
1. Share the ZIP file
2. Recipients should:
   - Extract the ZIP file
   - Run 'run.bat' on Windows or './run.sh' on Linux/Mac
   - The application will open in their default web browser

Note: Each team member will need Python 3.10+ installed on their system.
""")

if __name__ == "__main__":
    create_distribution()

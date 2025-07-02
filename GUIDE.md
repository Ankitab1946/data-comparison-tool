# Side by Side Comparison Tool - Setup Guide

## For Developers (Creating the Executable)

### Prerequisites
1. Install Python 3.10 or later
2. Clone this repository

### Steps to Create Executable
1. Open terminal/command prompt in the project directory
2. Run the build script:
   ```bash
   # On Windows
   python build_standalone.py

   # On Linux/Mac
   python3 build_standalone.py
   ```
3. The executable package will be created in:
   - ZIP file: `dist/comparison_tool_standalone.zip`
   - Directory: `dist/comparison_tool_package`

### Distribution
Share the `comparison_tool_standalone.zip` file with team members. This package contains everything needed to run the application, including:
- Standalone executable
- Launch scripts
- Configuration templates
- Documentation

## For End Users (Running the Application)

### No Python Required!
This is a standalone application - you don't need to install Python or any other dependencies.

### Windows Users
1. Download `comparison_tool_standalone.zip`
2. Extract the ZIP file
3. Double-click `launch.bat`
4. The application will open in your default web browser

### Linux/Mac Users
1. Download `comparison_tool_standalone.zip`
2. Extract the ZIP file
3. Open terminal in the extracted directory
4. Make the launch script executable:
   ```bash
   chmod +x launch.sh
   ```
5. Run the application:
   ```bash
   ./launch.sh
   ```
6. The application will open in your default web browser

### SQL Server Connection Setup
If you need to connect to SQL Server:

1. Copy `.env.example` to `.env`
2. Edit `.env` with your database settings:
   ```env
   # For SQL Authentication
   SQL_SERVER=your_server_name
   SQL_DATABASE=your_database_name
   SQL_USERNAME=your_username
   SQL_PASSWORD=your_password

   # For Windows Authentication
   SQL_USE_WINDOWS_AUTH=true
   SQL_SERVER=your_server_name
   SQL_DATABASE=your_database_name
   ```

### Troubleshooting

#### Application Won't Start
1. Make sure port 8000 is not in use
2. Try running the executable directly:
   - Windows: `comparison_tool.exe`
   - Linux/Mac: `./comparison_tool`
3. Check if any antivirus is blocking the executable

#### SQL Server Connection Issues
1. Verify your `.env` file settings
2. Ensure you have network access to the SQL Server
3. For Windows Authentication:
   - Make sure you're logged into your domain account
   - Verify you have access to the database

#### Browser Issues
1. The application runs on http://localhost:8000
2. If the browser doesn't open automatically:
   - Open your browser manually
   - Navigate to http://localhost:8000

### Support
If you encounter any issues:
1. Check the console output for error messages
2. Verify your SQL Server credentials if using database connections
3. Contact your team's technical support

## Features

### Data Sources
- CSV files
- SQL Server databases (with Windows Authentication support)
- Sample data included for testing

### Report Types
1. Summary Report
   - Source and target record counts
   - Join columns used
   - Overall comparison status

2. Difference Report
   - Side-by-side comparison
   - Highlighted differences
   - Status indicators (Left Only, Right Only, Both)

### Security Notes
- Database credentials are stored locally in `.env`
- Windows Authentication is recommended for SQL Server
- No data is sent to external servers

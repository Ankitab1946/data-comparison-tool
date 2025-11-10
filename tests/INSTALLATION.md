# Installation Guide

Complete installation guide for the Selenium+Python+Pytest+Allure testing framework.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **pip**: Latest version
- **Browser**: Chrome or Firefox
- **Database**: SQL Server (for database tests)

### Check Prerequisites

```bash
# Check Python version
python --version  # or python3 --version

# Check pip version
pip --version  # or pip3 --version

# Check if Chrome is installed
google-chrome --version  # Linux
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --version  # macOS
```

## 🚀 Installation Steps

### Step 1: Navigate to Tests Directory

```bash
cd /vercel/sandbox/tests
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- selenium (WebDriver automation)
- pytest (Testing framework)
- allure-pytest (Allure reporting)
- webdriver-manager (Automatic driver management)
- And all other dependencies

### Step 5: Install Allure Command-Line Tool

#### On Amazon Linux 2023 / RHEL / CentOS

```bash
# Download Allure
wget https://github.com/allure-framework/allure2/releases/download/2.24.1/allure-2.24.1.tgz

# Extract
tar -zxvf allure-2.24.1.tgz

# Move to /opt
sudo mv allure-2.24.1 /opt/allure

# Create symlink
sudo ln -s /opt/allure/bin/allure /usr/bin/allure

# Verify installation
allure --version
```

#### On Ubuntu/Debian

```bash
# Add repository
sudo apt-add-repository ppa:qameta/allure
sudo apt-get update

# Install
sudo apt-get install allure

# Verify
allure --version
```

#### On macOS

```bash
# Using Homebrew
brew install allure

# Verify
allure --version
```

#### On Windows

```bash
# Using Scoop
scoop install allure

# Or download from:
# https://github.com/allure-framework/allure2/releases
```

### Step 6: Install SQL Server ODBC Driver (For Database Tests)

#### On Amazon Linux 2023 / RHEL

```bash
# Download Microsoft repository config
curl https://packages.microsoft.com/config/rhel/8/prod.repo | sudo tee /etc/yum.repos.d/mssql-release.repo

# Install ODBC Driver
sudo dnf remove unixODBC-utf16 unixODBC-utf16-devel
sudo ACCEPT_EULA=Y dnf install -y msodbcsql17

# Install development headers (optional)
sudo ACCEPT_EULA=Y dnf install -y mssql-tools
echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc
source ~/.bashrc

# Verify
odbcinst -j
```

#### On Ubuntu/Debian

```bash
# Add Microsoft repository
curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list

# Update and install
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get install -y msodbcsql17

# Verify
odbcinst -j
```

#### On macOS

```bash
# Using Homebrew
brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
brew update
brew install msodbcsql17 mssql-tools

# Verify
odbcinst -j
```

### Step 7: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
nano .env  # or use vim, vi, or any text editor
```

Update the following in `.env`:

```env
# Autosys Portal
AUTOSYS_URL=https://your-autosys-portal.com
AUTOSYS_USERNAME=your_username
AUTOSYS_PASSWORD=your_password

# Database
DB_SERVER=your_db_server
DB_NAME=your_db_name
DB_USERNAME=your_db_username
DB_PASSWORD=your_db_password
```

### Step 8: Make Scripts Executable

```bash
chmod +x run_tests.sh
```

### Step 9: Verify Installation

```bash
# Check pytest
pytest --version

# Check Allure
allure --version

# Run a simple test
pytest tests/ --collect-only
```

## 🔧 Post-Installation Configuration

### Configure Test Data

Edit `config/test_data.yaml` with your test data:

```bash
nano config/test_data.yaml
```

Update box names, job names, and other test-specific data.

### Configure Browser Settings

Edit `config/settings.py` or use environment variables:

```bash
# In .env file
BROWSER=chrome
HEADLESS=false
IMPLICIT_WAIT=10
EXPLICIT_WAIT=20
```

### Create Reports Directories

```bash
mkdir -p reports/{html,logs,screenshots,allure-results}
```

## ✅ Verification

### Test Installation

```bash
# Run smoke tests
pytest tests/ -m smoke -v

# Generate Allure report
allure serve reports/allure-results
```

### Expected Output

```
======================== test session starts =========================
platform linux -- Python 3.9.x, pytest-7.4.3, pluggy-1.3.0
rootdir: /vercel/sandbox/tests
plugins: allure-pytest-2.13.2, html-4.1.1, xdist-3.5.0
collected X items

tests/test_autosys.py::TestAutosysBoxConfiguration::test_verify_date_condition[TEST_BOX_001] PASSED
tests/test_autosys.py::TestAutosysBoxConfiguration::test_verify_date_condition[TEST_BOX_002] PASSED
...

======================== X passed in X.XXs ==========================
```

## 🐛 Troubleshooting

### Issue: pip install fails

```bash
# Upgrade pip
pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v
```

### Issue: WebDriver not found

```bash
# Reinstall webdriver-manager
pip uninstall webdriver-manager
pip install webdriver-manager

# Clear cache
rm -rf ~/.wdm
```

### Issue: Permission denied

```bash
# Make scripts executable
chmod +x run_tests.sh

# Fix directory permissions
chmod -R 755 tests/
```

### Issue: ODBC Driver not found

```bash
# List installed drivers
odbcinst -q -d

# If not listed, reinstall ODBC driver
# See Step 6 above
```

### Issue: Allure not found

```bash
# Check if allure is in PATH
which allure

# If not, add to PATH
export PATH=$PATH:/opt/allure/bin

# Or reinstall Allure
# See Step 5 above
```

### Issue: Import errors

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: Database connection fails

```bash
# Test database connectivity
python -c "import pyodbc; print(pyodbc.drivers())"

# Check connection string
# Verify credentials in .env file
```

## 📚 Next Steps

After successful installation:

1. **Read Documentation**
   - `README.md` - Complete documentation
   - `QUICKSTART.md` - Quick start guide
   - `FRAMEWORK_OVERVIEW.md` - Framework architecture

2. **Configure Test Data**
   - Update `config/test_data.yaml`
   - Update `.env` with your settings

3. **Run Tests**
   - Start with smoke tests: `./run_tests.sh smoke`
   - View reports: `./run_tests.sh allure`

4. **Customize**
   - Add new test cases
   - Modify page objects
   - Update configuration

## 🎓 Training Resources

- **Pytest**: https://docs.pytest.org/
- **Selenium**: https://www.selenium.dev/documentation/
- **Allure**: https://docs.qameta.io/allure/
- **Page Object Model**: https://www.selenium.dev/documentation/test_practices/encouraged/page_object_models/

## 📞 Support

If you encounter issues:

1. Check logs: `reports/logs/test_execution.log`
2. Review error messages in console
3. Check Allure reports for details
4. Verify all prerequisites are installed
5. Ensure configuration is correct

## ✨ Success!

You're now ready to run automated tests! 🎉

Try running your first test:

```bash
./run_tests.sh smoke
```

Happy Testing! 🚀

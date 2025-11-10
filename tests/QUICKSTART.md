# Quick Start Guide

Get started with the Selenium+Python+Pytest+Allure testing framework in 5 minutes!

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
cd tests
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env  # or use your preferred editor
```

Update these key settings:
```env
AUTOSYS_URL=https://your-autosys-portal.com
AUTOSYS_USERNAME=your_username
AUTOSYS_PASSWORD=your_password
```

### 3. Run Your First Test

```bash
# Run smoke tests
pytest tests/ -m smoke -v
```

## 📊 View Reports

### HTML Report
```bash
# Report is auto-generated at:
open reports/html/report.html
```

### Allure Report
```bash
# Install Allure (one-time)
# macOS: brew install allure
# Linux: See README.md for instructions

# Generate and view report
allure serve reports/allure-results
```

## 🎯 Common Commands

### Run Tests

```bash
# All tests
pytest tests/

# Smoke tests only
pytest tests/ -m smoke

# Parallel execution
pytest tests/ -n auto

# Headless mode
HEADLESS=true pytest tests/

# Specific test
pytest tests/test_autosys.py::TestAutosysBoxConfiguration::test_verify_date_condition
```

### Using the Helper Script

```bash
# Make script executable (first time only)
chmod +x run_tests.sh

# Run smoke tests
./run_tests.sh smoke

# Run in parallel
./run_tests.sh parallel

# View Allure report
./run_tests.sh allure

# Clean reports
./run_tests.sh clean
```

## 📝 Test Markers

Use markers to run specific test groups:

| Marker | Description | Command |
|--------|-------------|---------|
| `smoke` | Quick smoke tests | `pytest -m smoke` |
| `regression` | Full regression suite | `pytest -m regression` |
| `critical` | Critical tests | `pytest -m critical` |
| `database` | Database tests | `pytest -m database` |
| `ui` | UI tests | `pytest -m ui` |

## 🔧 Troubleshooting

### Issue: WebDriver not found
```bash
pip install --upgrade webdriver-manager
```

### Issue: Database connection failed
- Check SQL Server is running
- Verify credentials in `.env`
- Ensure ODBC Driver 17 is installed

### Issue: Tests hanging
```bash
# Set timeout
pytest tests/ --timeout=300
```

### Issue: Permission denied on run_tests.sh
```bash
chmod +x run_tests.sh
```

## 📚 Next Steps

1. **Customize Test Data**: Edit `config/test_data.yaml`
2. **Add New Tests**: See `tests/test_autosys.py` for examples
3. **Configure CI/CD**: See README.md for Jenkins/GitHub Actions examples
4. **Read Full Documentation**: See `README.md`

## 💡 Tips

- Use `-v` flag for verbose output
- Use `-s` flag to see print statements
- Use `--lf` to run last failed tests
- Use `--sw` to stop on first failure
- Use `-k "test_name"` to run tests matching pattern

## 🎓 Example Workflow

```bash
# 1. Setup (first time only)
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your settings

# 2. Run smoke tests
pytest tests/ -m smoke -v

# 3. View results
allure serve reports/allure-results

# 4. Run full regression
pytest tests/ -m regression -n auto

# 5. Clean up
./run_tests.sh clean
```

## 📞 Need Help?

- Check `README.md` for detailed documentation
- Review test logs in `reports/logs/`
- Check Allure reports for detailed test results
- Review pytest output for error messages

Happy Testing! 🎉

# Selenium + Python + Pytest + Allure Testing Framework

A comprehensive hybrid testing framework for Autosys UI portal automation with support for multiple execution modes, detailed reporting, and extensive logging.

## 🎯 Features

- ✅ **Multiple Test Execution Modes**
  - Serial execution
  - Parallel execution (pytest-xdist)
  - Selective execution (markers)
  - Batch execution
  
- ✅ **Test Management**
  - Run multiple test cases
  - Skip tests conditionally
  - Repeat tests multiple times
  - Parameterized tests
  
- ✅ **Comprehensive Reporting**
  - Allure reports with screenshots
  - HTML reports
  - Detailed logs with rotation
  - Test execution summary
  
- ✅ **Page Object Model (POM)**
  - Reusable page objects
  - Clean separation of concerns
  - Easy maintenance
  
- ✅ **Robust Framework**
  - Automatic screenshot on failure
  - Retry mechanism
  - Explicit and implicit waits
  - Cross-browser support (Chrome, Firefox)

## 📁 Project Structure

```
tests/
├── conftest.py                 # Pytest fixtures and hooks
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── README.md                  # This file
│
├── config/                    # Configuration files
│   ├── __init__.py
│   ├── settings.py           # Test settings
│   └── test_data.yaml        # Test data
│
├── pages/                     # Page Object Model
│   ├── __init__.py
│   ├── base_page.py          # Base page class
│   └── autosys_page.py       # Autosys portal page objects
│
├── tests/                     # Test cases
│   ├── __init__.py
│   └── test_autosys.py       # Autosys test cases
│
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── logger.py             # Custom logging
│   ├── screenshot.py         # Screenshot utilities
│   └── helpers.py            # Helper functions
│
└── reports/                   # Generated reports
    ├── html/                 # HTML reports
    ├── logs/                 # Test logs
    ├── screenshots/          # Test screenshots
    └── allure-results/       # Allure results
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Chrome or Firefox browser
- SQL Server (for database tests)

### Installation

1. **Clone or navigate to the tests directory:**
   ```bash
   cd tests
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install Allure command-line tool:**
   
   **On Linux:**
   ```bash
   # Using apt (Debian/Ubuntu)
   sudo apt-add-repository ppa:qameta/allure
   sudo apt-get update
   sudo apt-get install allure
   
   # Or download manually
   wget https://github.com/allure-framework/allure2/releases/download/2.24.1/allure-2.24.1.tgz
   tar -zxvf allure-2.24.1.tgz
   sudo mv allure-2.24.1 /opt/allure
   sudo ln -s /opt/allure/bin/allure /usr/bin/allure
   ```
   
   **On macOS:**
   ```bash
   brew install allure
   ```
   
   **On Windows:**
   ```bash
   scoop install allure
   ```

5. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

6. **Install browser drivers (automatic via webdriver-manager):**
   - Chrome driver will be downloaded automatically
   - Firefox driver will be downloaded automatically

## 🧪 Test Cases

### Autosys Portal Tests

1. **TC001: Verify BOX Autosys Date Condition must be 1**
   - Validates that the Date Condition parameter is set to 1
   - Marker: `smoke`, `critical`

2. **TC002: Verify BOX Autosys Parameter 'alarm_if_fail' must be 1**
   - Validates that alarm_if_fail parameter is set to 1
   - Marker: `smoke`, `critical`

3. **TC003: Verify BOX Autosys Parameter 'alarm_if_terminated' must be 0**
   - Validates that alarm_if_terminated parameter is set to 0
   - Marker: `smoke`, `critical`

4. **TC004: Verify Filewatcher job exists for Load Job**
   - Validates that a Filewatcher job exists for Load jobs
   - Marker: `regression`, `critical`

5. **TC005: Verify job load on SQL Server**
   - Validates that job loads are recorded in SQL Server
   - Marker: `regression`, `database`

## 🎮 Running Tests

### Basic Execution

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_autosys.py

# Run specific test
pytest tests/test_autosys.py::TestAutosysBoxConfiguration::test_verify_date_condition
```

### Execution by Markers

```bash
# Run smoke tests only
pytest tests/ -m smoke

# Run critical tests only
pytest tests/ -m critical

# Run regression tests
pytest tests/ -m regression

# Run database tests
pytest tests/ -m database

# Run UI tests
pytest tests/ -m ui

# Exclude specific markers
pytest tests/ -m "not slow"

# Combine markers
pytest tests/ -m "smoke and critical"
```

### Parallel Execution

```bash
# Run tests in parallel (auto-detect CPU cores)
pytest tests/ -n auto

# Run tests with 4 workers
pytest tests/ -n 4

# Parallel execution with specific marker
pytest tests/ -m smoke -n 4
```

### Multiple Runs

```bash
# Run tests 3 times
pytest tests/ --count=3

# Run with retry on failure
pytest tests/ --reruns 3 --reruns-delay 2
```

### Skip Tests

```bash
# Skip tests with 'skip' marker
pytest tests/ -m "not skip"

# Run with skip summary
pytest tests/ -rs
```

### Headless Mode

```bash
# Run in headless mode
HEADLESS=true pytest tests/

# Or set in .env file
# HEADLESS=true
```

### Browser Selection

```bash
# Run with Chrome (default)
BROWSER=chrome pytest tests/

# Run with Firefox
BROWSER=firefox pytest tests/
```

## 📊 Reports

### HTML Report

HTML reports are automatically generated after each test run:

```bash
# Run tests and generate HTML report
pytest tests/

# View report
open reports/html/report.html  # macOS
xdg-open reports/html/report.html  # Linux
start reports/html/report.html  # Windows
```

### Allure Report

Generate and view Allure reports:

```bash
# Run tests with Allure
pytest tests/ --alluredir=reports/allure-results

# Generate and open Allure report
allure serve reports/allure-results

# Or generate static report
allure generate reports/allure-results -o reports/allure-report --clean
allure open reports/allure-report
```

### Logs

Test execution logs are stored in:
- `reports/logs/test_execution.log` - Main test log
- `reports/logs/TestFramework_YYYYMMDD.log` - Daily log with rotation

View logs:
```bash
tail -f reports/logs/test_execution.log
```

### Screenshots

Screenshots are automatically captured:
- On test failure (automatic)
- On demand (programmatic)

Location: `reports/screenshots/`

## 🔧 Configuration

### Environment Variables

Edit `.env` file to configure:

```env
# Browser settings
BROWSER=chrome
HEADLESS=false

# Autosys portal
AUTOSYS_URL=https://your-autosys-portal.com
AUTOSYS_USERNAME=your_username
AUTOSYS_PASSWORD=your_password

# Database
DB_SERVER=your_db_server
DB_NAME=your_db_name
DB_USERNAME=your_db_username
DB_PASSWORD=your_db_password
```

### Test Data

Edit `config/test_data.yaml` to configure test data:

```yaml
boxes:
  - box_name: "YOUR_BOX_NAME"
    date_condition: 1
    alarm_if_fail: 1
    alarm_if_terminated: 0

jobs:
  - job_name: "YOUR_JOB_NAME"
    job_type: "Load"
```

### Pytest Configuration

Edit `pytest.ini` to customize pytest behavior:

```ini
[pytest]
markers =
    smoke: Quick smoke tests
    regression: Full regression suite
    critical: Critical test cases
```

## 📝 Writing New Tests

### Example Test

```python
import pytest
import allure
from pages.autosys_page import AutosysPage

@allure.feature("Autosys Portal")
@allure.story("Your Story")
class TestYourFeature:
    
    @allure.title("Your Test Title")
    @allure.description("Your test description")
    @pytest.mark.smoke
    def test_your_test(self, autosys_page):
        """Your test docstring."""
        
        with allure.step("Step 1"):
            # Your test code
            pass
        
        with allure.step("Step 2"):
            # Your assertion
            assert True
```

### Using Page Objects

```python
# In your test
def test_example(self, autosys_page):
    # Navigate
    autosys_page.navigate_to("https://example.com")
    
    # Interact
    autosys_page.click((By.ID, "button"))
    autosys_page.enter_text((By.ID, "input"), "text")
    
    # Assert
    text = autosys_page.get_text((By.ID, "result"))
    assert text == "expected"
```

## 🐛 Troubleshooting

### Common Issues

1. **WebDriver not found:**
   ```bash
   # Reinstall webdriver-manager
   pip install --upgrade webdriver-manager
   ```

2. **Database connection failed:**
   - Verify SQL Server is running
   - Check connection credentials in `.env`
   - Ensure ODBC Driver 17 is installed

3. **Allure command not found:**
   ```bash
   # Install Allure
   # See installation instructions above
   ```

4. **Tests hanging:**
   - Check timeout settings in `config/settings.py`
   - Use `pytest --timeout=300` to set global timeout

5. **Screenshot not captured:**
   - Ensure `SCREENSHOT_ON_FAILURE=True` in settings
   - Check `reports/screenshots/` directory permissions

## 📚 Best Practices

1. **Use Page Object Model (POM)**
   - Keep page objects in `pages/` directory
   - Separate locators from test logic

2. **Use Allure Steps**
   - Wrap test steps with `with allure.step()`
   - Makes reports more readable

3. **Use Markers**
   - Tag tests appropriately (`@pytest.mark.smoke`)
   - Enables selective execution

4. **Use Fixtures**
   - Reuse common setup/teardown code
   - Define in `conftest.py`

5. **Use Logging**
   - Log important actions and assertions
   - Helps with debugging

6. **Use Assertions**
   - Use descriptive assertion messages
   - Include expected and actual values

## 🤝 Contributing

1. Follow PEP 8 style guide
2. Add docstrings to all functions
3. Write meaningful test names
4. Update documentation

## 📄 License

This project is licensed under the MIT License.

## 📞 Support

For issues or questions:
- Check logs in `reports/logs/`
- Review Allure reports
- Check pytest output

## 🔄 CI/CD Integration

### Jenkins Example

```groovy
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Test') {
            steps {
                sh 'pytest tests/ -n auto --alluredir=reports/allure-results'
            }
        }
        
        stage('Report') {
            steps {
                allure includeProperties: false,
                       jdk: '',
                       results: [[path: 'reports/allure-results']]
            }
        }
    }
}
```

### GitHub Actions Example

```yaml
name: Test Automation

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/ -n auto --alluredir=reports/allure-results
    
    - name: Generate Allure Report
      uses: simple-elf/allure-report-action@master
      if: always()
      with:
        allure_results: reports/allure-results
```

## 📈 Metrics

The framework provides:
- Test execution time
- Pass/fail rates
- Test coverage
- Failure trends
- Screenshot evidence
- Detailed logs

All available in Allure reports!

# Selenium + Python + Pytest + Allure Framework Overview

## 🎯 Framework Objectives

This hybrid testing framework is designed to meet all the specified requirements for Autosys UI portal testing:

### ✅ Core Requirements Met

1. **Run Multiple TestCases** ✓
   - Framework supports unlimited test cases
   - Organized using Page Object Model
   - Easy to add new tests

2. **Run Testcases in Batch - Serial/Parallel** ✓
   - Serial: `pytest tests/`
   - Parallel: `pytest tests/ -n auto`
   - Configurable worker count

3. **Selectively/Optionally Running Test Cases** ✓
   - Markers: `@pytest.mark.smoke`, `@pytest.mark.regression`, etc.
   - Run by marker: `pytest -m smoke`
   - Run by name: `pytest -k "test_name"`
   - Run specific file: `pytest tests/test_autosys.py`

4. **Running Test Cases Multiple Times** ✓
   - Repeat: `pytest --count=3`
   - Retry on failure: `pytest --reruns 3`
   - Parameterized tests for data-driven execution

5. **Skip Test Cases** ✓
   - `@pytest.mark.skip` decorator
   - `@pytest.mark.skipif` for conditional skip
   - Skip by marker: `pytest -m "not skip"`

6. **Generate Reports: Logs, Screenshots, Headings** ✓
   - **Allure Reports**: Rich HTML reports with test steps, screenshots, logs
   - **HTML Reports**: pytest-html for quick viewing
   - **Logs**: Detailed logs with rotation and color coding
   - **Screenshots**: Automatic capture on failure, manual capture on demand

## 📋 Test Cases Implemented

### 1. TC001: Verify BOX Autosys Date Condition = 1
- **Marker**: `smoke`, `critical`
- **Parameterized**: Yes (multiple boxes)
- **Validates**: Date Condition parameter value

### 2. TC002: Verify alarm_if_fail = 1
- **Marker**: `smoke`, `critical`
- **Parameterized**: Yes (multiple boxes)
- **Validates**: alarm_if_fail parameter value

### 3. TC003: Verify alarm_if_terminated = 0
- **Marker**: `smoke`, `critical`
- **Parameterized**: Yes (multiple boxes)
- **Validates**: alarm_if_terminated parameter value

### 4. TC004: Verify Filewatcher Job Exists for Load Job
- **Marker**: `regression`, `critical`
- **Parameterized**: Yes (multiple jobs)
- **Validates**: Filewatcher job existence for Load jobs

### 5. TC005: Verify Job Load on SQL Server
- **Marker**: `regression`, `database`
- **Parameterized**: Yes (multiple jobs)
- **Validates**: Job load records in SQL Server database

## 🏗️ Framework Architecture

### Layer Structure

```
┌─────────────────────────────────────────┐
│         Test Cases Layer                │
│  (test_autosys.py)                     │
│  - Business logic tests                 │
│  - Assertions and validations          │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│      Page Object Layer                  │
│  (autosys_page.py, base_page.py)      │
│  - UI interactions                      │
│  - Element locators                     │
│  - Page-specific methods                │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│       Utilities Layer                   │
│  (logger, screenshot, helpers)          │
│  - Common functions                     │
│  - Helper methods                       │
│  - Database operations                  │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│      Configuration Layer                │
│  (settings.py, test_data.yaml)         │
│  - Environment settings                 │
│  - Test data                            │
│  - Constants                            │
└─────────────────────────────────────────┘
```

### Key Components

#### 1. **conftest.py** - Pytest Configuration
- WebDriver fixture with auto setup/teardown
- Browser configuration (Chrome/Firefox)
- Headless mode support
- Screenshot capture on failure
- Database connection fixture
- Test logging hooks

#### 2. **Page Objects** - UI Abstraction
- `BasePage`: Common page operations
- `AutosysPage`: Autosys-specific operations
- Reusable locators and methods
- Allure step integration

#### 3. **Utilities** - Helper Functions
- `logger.py`: Custom logging with colors and rotation
- `screenshot.py`: Screenshot capture and Allure attachment
- `helpers.py`: Wait helpers, retry logic, data loading
- `db_helper.py`: Database operations and validations

#### 4. **Configuration** - Settings Management
- `settings.py`: Environment variables and constants
- `test_data.yaml`: Test data in YAML format
- `.env`: Sensitive configuration (not in git)

#### 5. **Test Cases** - Business Logic
- Organized by feature/story
- Allure annotations for rich reporting
- Parameterized for data-driven testing
- Markers for selective execution

## 🎨 Reporting Features

### Allure Reports
- **Test Steps**: Detailed step-by-step execution
- **Screenshots**: Attached to failed tests
- **Logs**: Test execution logs
- **Parameters**: Test data and parameters
- **Timing**: Execution time for each test
- **Trends**: Historical test results
- **Categories**: Test categorization
- **Severity**: Test priority levels

### HTML Reports
- Quick overview of test results
- Pass/fail statistics
- Execution time
- Error messages
- Self-contained (single file)

### Logs
- **Console Logs**: Color-coded, real-time
- **File Logs**: Detailed, with rotation
- **Test Logs**: Per-test execution details
- **Framework Logs**: Framework-level information

### Screenshots
- **Automatic**: On test failure
- **Manual**: On-demand capture
- **Full Page**: Scrolling screenshots
- **Element**: Specific element capture
- **Allure Integration**: Attached to reports

## 🔧 Execution Modes

### 1. Serial Execution
```bash
pytest tests/
```
- Tests run one after another
- Easier debugging
- Slower execution

### 2. Parallel Execution
```bash
pytest tests/ -n auto
pytest tests/ -n 4
```
- Tests run concurrently
- Faster execution
- Requires thread-safe tests

### 3. Selective Execution
```bash
pytest tests/ -m smoke
pytest tests/ -m "smoke and critical"
pytest tests/ -k "date_condition"
```
- Run specific test groups
- Filter by markers or names
- Efficient test execution

### 4. Repeated Execution
```bash
pytest tests/ --count=3
pytest tests/ --reruns 3
```
- Run tests multiple times
- Retry on failure
- Stability testing

### 5. Headless Execution
```bash
HEADLESS=true pytest tests/
```
- No browser UI
- Faster execution
- CI/CD friendly

## 📊 Test Markers

| Marker | Purpose | Usage |
|--------|---------|-------|
| `smoke` | Quick validation tests | `pytest -m smoke` |
| `regression` | Full test suite | `pytest -m regression` |
| `critical` | Critical functionality | `pytest -m critical` |
| `database` | Database validation | `pytest -m database` |
| `ui` | UI interaction tests | `pytest -m ui` |
| `slow` | Long-running tests | `pytest -m "not slow"` |
| `skip` | Tests to skip | `pytest -m "not skip"` |

## 🚀 Quick Commands Reference

### Setup
```bash
pip install -r requirements.txt
cp .env.example .env
chmod +x run_tests.sh
```

### Run Tests
```bash
# All tests
pytest tests/

# Smoke tests
pytest tests/ -m smoke

# Parallel
pytest tests/ -n auto

# Headless
HEADLESS=true pytest tests/

# Specific test
pytest tests/test_autosys.py::TestAutosysBoxConfiguration::test_verify_date_condition
```

### Reports
```bash
# Allure
allure serve reports/allure-results

# HTML
open reports/html/report.html

# Logs
tail -f reports/logs/test_execution.log
```

### Helper Script
```bash
./run_tests.sh smoke
./run_tests.sh parallel
./run_tests.sh allure
./run_tests.sh clean
```

## 🎓 Best Practices Implemented

1. **Page Object Model (POM)**
   - Separation of concerns
   - Reusable components
   - Easy maintenance

2. **DRY Principle**
   - No code duplication
   - Reusable fixtures
   - Common utilities

3. **Explicit Waits**
   - Reliable element interactions
   - Configurable timeouts
   - Retry mechanisms

4. **Logging**
   - Detailed execution logs
   - Color-coded console output
   - Log rotation

5. **Error Handling**
   - Graceful failure handling
   - Descriptive error messages
   - Screenshot on failure

6. **Configuration Management**
   - Environment-based config
   - Externalized test data
   - Secure credential handling

7. **Reporting**
   - Rich Allure reports
   - Screenshot evidence
   - Test step documentation

## 🔐 Security Considerations

- Credentials stored in `.env` (not in git)
- `.gitignore` configured properly
- Password masking in logs
- Secure database connections

## 📈 Scalability

- Easy to add new test cases
- Supports parallel execution
- Modular architecture
- Reusable components
- Data-driven testing support

## 🛠️ Maintenance

- Clear code structure
- Comprehensive documentation
- Consistent naming conventions
- Version control friendly
- Easy to update

## 📞 Support

- **Documentation**: README.md, QUICKSTART.md
- **Examples**: Sample tests included
- **Logs**: Detailed execution logs
- **Reports**: Visual test results

## 🎉 Summary

This framework provides a **production-ready**, **scalable**, and **maintainable** solution for Autosys UI portal testing with:

✅ All required features implemented
✅ 5 test cases for Autosys validation
✅ Multiple execution modes
✅ Comprehensive reporting
✅ Detailed logging
✅ Screenshot capture
✅ Easy to use and extend
✅ Well documented
✅ CI/CD ready

**Ready to use out of the box!** 🚀

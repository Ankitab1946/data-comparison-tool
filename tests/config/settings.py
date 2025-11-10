"""Test configuration settings."""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
SCREENSHOTS_DIR = REPORTS_DIR / "screenshots"
LOGS_DIR = REPORTS_DIR / "logs"
ALLURE_RESULTS_DIR = REPORTS_DIR / "allure-results"

# Create directories if they don't exist
for directory in [REPORTS_DIR, SCREENSHOTS_DIR, LOGS_DIR, ALLURE_RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Browser settings
BROWSER = os.getenv("BROWSER", "chrome")  # chrome, firefox, edge
HEADLESS = os.getenv("HEADLESS", "false").lower() == "true"
IMPLICIT_WAIT = int(os.getenv("IMPLICIT_WAIT", "10"))
EXPLICIT_WAIT = int(os.getenv("EXPLICIT_WAIT", "20"))
PAGE_LOAD_TIMEOUT = int(os.getenv("PAGE_LOAD_TIMEOUT", "30"))

# Browser options
WINDOW_SIZE = os.getenv("WINDOW_SIZE", "1920,1080")
DOWNLOAD_DIR = str(BASE_DIR / "downloads")

# Autosys portal settings
AUTOSYS_URL = os.getenv("AUTOSYS_URL", "https://autosys-portal.example.com")
AUTOSYS_USERNAME = os.getenv("AUTOSYS_USERNAME", "test_user")
AUTOSYS_PASSWORD = os.getenv("AUTOSYS_PASSWORD", "test_password")

# Database settings (for SQL Server validation)
DB_SERVER = os.getenv("DB_SERVER", "localhost")
DB_NAME = os.getenv("DB_NAME", "autosys_db")
DB_USERNAME = os.getenv("DB_USERNAME", "db_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "db_password")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")

# Test data
EXPECTED_DATE_CONDITION = 1
EXPECTED_ALARM_IF_FAIL = 1
EXPECTED_ALARM_IF_TERMINATED = 0

# Retry settings
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2"))

# Screenshot settings
SCREENSHOT_ON_FAILURE = True
SCREENSHOT_ON_SUCCESS = False

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(levelname)8s] [%(name)s] [%(filename)s:%(lineno)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

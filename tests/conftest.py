"""Pytest configuration and fixtures."""
import pytest
import allure
from datetime import datetime
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager

from config.settings import (
    BROWSER, HEADLESS, IMPLICIT_WAIT, PAGE_LOAD_TIMEOUT,
    WINDOW_SIZE, DOWNLOAD_DIR, SCREENSHOTS_DIR, SCREENSHOT_ON_FAILURE,
    AUTOSYS_URL
)
from utils.logger import TestLogger
from utils.screenshot import ScreenshotCapture

logger = TestLogger.get_logger("Conftest")


def pytest_configure(config):
    """Pytest configuration hook."""
    # Create reports directories
    Path("reports/html").mkdir(parents=True, exist_ok=True)
    Path("reports/logs").mkdir(parents=True, exist_ok=True)
    Path("reports/screenshots").mkdir(parents=True, exist_ok=True)
    Path("reports/allure-results").mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("TEST EXECUTION STARTED")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Browser: {BROWSER}")
    logger.info(f"Headless: {HEADLESS}")
    logger.info("=" * 80)


def pytest_unconfigure(config):
    """Pytest cleanup hook."""
    logger.info("=" * 80)
    logger.info("TEST EXECUTION COMPLETED")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


@pytest.fixture(scope="session")
def browser_config():
    """Browser configuration fixture."""
    return {
        "browser": BROWSER,
        "headless": HEADLESS,
        "implicit_wait": IMPLICIT_WAIT,
        "page_load_timeout": PAGE_LOAD_TIMEOUT,
        "window_size": WINDOW_SIZE,
        "download_dir": DOWNLOAD_DIR
    }


@pytest.fixture(scope="function")
def driver(browser_config):
    """
    WebDriver fixture with automatic setup and teardown.
    
    Yields:
        WebDriver instance
    """
    logger.info(f"Setting up {browser_config['browser']} driver")
    
    driver_instance = None
    
    try:
        if browser_config["browser"].lower() == "chrome":
            options = webdriver.ChromeOptions()
            
            if browser_config["headless"]:
                options.add_argument("--headless=new")
            
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument(f"--window-size={browser_config['window_size']}")
            options.add_argument("--start-maximized")
            options.add_argument("--disable-blink-features=AutomationControlled")
            
            # Download preferences
            prefs = {
                "download.default_directory": browser_config["download_dir"],
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True
            }
            options.add_experimental_option("prefs", prefs)
            options.add_experimental_option("excludeSwitches", ["enable-logging"])
            
            service = ChromeService(ChromeDriverManager().install())
            driver_instance = webdriver.Chrome(service=service, options=options)
            
        elif browser_config["browser"].lower() == "firefox":
            options = webdriver.FirefoxOptions()
            
            if browser_config["headless"]:
                options.add_argument("--headless")
            
            options.set_preference("browser.download.folderList", 2)
            options.set_preference("browser.download.dir", browser_config["download_dir"])
            options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/octet-stream")
            
            service = FirefoxService(GeckoDriverManager().install())
            driver_instance = webdriver.Firefox(service=service, options=options)
            
        else:
            raise ValueError(f"Unsupported browser: {browser_config['browser']}")
        
        # Set timeouts
        driver_instance.implicitly_wait(browser_config["implicit_wait"])
        driver_instance.set_page_load_timeout(browser_config["page_load_timeout"])
        
        logger.info("Driver setup completed successfully")
        
        yield driver_instance
        
    except Exception as e:
        logger.error(f"Error setting up driver: {str(e)}")
        raise
        
    finally:
        if driver_instance:
            logger.info("Tearing down driver")
            driver_instance.quit()


@pytest.fixture(scope="function")
def autosys_page(driver):
    """
    Autosys page fixture.
    
    Args:
        driver: WebDriver fixture
        
    Returns:
        AutosysPage instance
    """
    from pages.autosys_page import AutosysPage
    
    logger.info("Initializing Autosys page")
    page = AutosysPage(driver)
    page.navigate_to(AUTOSYS_URL)
    
    return page


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Pytest hook to capture test results and take screenshots on failure.
    
    Args:
        item: Test item
        call: Test call
    """
    outcome = yield
    report = outcome.get_result()
    
    # Set test result as attribute for use in fixtures
    setattr(item, f"report_{report.when}", report)
    
    if report.when == "call":
        # Log test result
        test_name = item.nodeid
        
        if report.passed:
            TestLogger.log_test_end(test_name, "PASSED")
        elif report.failed:
            TestLogger.log_test_end(test_name, "FAILED")
        elif report.skipped:
            TestLogger.log_test_end(test_name, "SKIPPED")
        
        # Capture screenshot on failure
        if report.failed and SCREENSHOT_ON_FAILURE:
            driver = item.funcargs.get("driver")
            if driver:
                try:
                    screenshot_name = test_name.replace("::", "_").replace("/", "_")
                    screenshot_path = ScreenshotCapture.capture(
                        driver, screenshot_name, "failure"
                    )
                    
                    if screenshot_path:
                        # Attach to Allure report
                        with open(screenshot_path, 'rb') as f:
                            allure.attach(
                                f.read(),
                                name="Failure Screenshot",
                                attachment_type=allure.attachment_type.PNG
                            )
                        
                        logger.info(f"Failure screenshot captured: {screenshot_path}")
                        
                except Exception as e:
                    logger.error(f"Failed to capture failure screenshot: {str(e)}")


@pytest.fixture(scope="function", autouse=True)
def log_test_info(request):
    """
    Automatically log test information before and after each test.
    
    Args:
        request: Pytest request object
    """
    test_name = request.node.nodeid
    TestLogger.log_test_start(test_name)
    
    yield
    
    # Test result logging is handled in pytest_runtest_makereport


@pytest.fixture(scope="function")
def test_data():
    """
    Load test data from YAML file.
    
    Returns:
        Dictionary with test data
    """
    from utils.helpers import DataHelper
    
    return DataHelper.load_yaml("test_data.yaml")


@pytest.fixture(scope="function")
def db_connection():
    """
    Database connection fixture for SQL Server validation.
    
    Yields:
        SQLAlchemy engine
    """
    from sqlalchemy import create_engine
    from urllib.parse import quote_plus
    from config.settings import DB_SERVER, DB_NAME, DB_USERNAME, DB_PASSWORD, DB_DRIVER
    
    logger.info("Creating database connection")
    
    try:
        connection_string = (
            f"mssql+pyodbc://{DB_USERNAME}:{quote_plus(DB_PASSWORD)}"
            f"@{DB_SERVER}/{DB_NAME}?driver={quote_plus(DB_DRIVER)}"
        )
        
        engine = create_engine(connection_string, pool_pre_ping=True)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        
        logger.info("Database connection established")
        
        yield engine
        
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        pytest.skip(f"Database connection failed: {str(e)}")
    
    finally:
        if 'engine' in locals():
            engine.dispose()
            logger.info("Database connection closed")


# Pytest markers
def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers dynamically.
    
    Args:
        config: Pytest config
        items: Test items
    """
    for item in items:
        # Add autosys marker to all tests in test_autosys.py
        if "test_autosys" in item.nodeid:
            item.add_marker(pytest.mark.autosys)
        
        # Add ui marker to all tests using driver fixture
        if "driver" in item.fixturenames:
            item.add_marker(pytest.mark.ui)
        
        # Add database marker to tests using db_connection fixture
        if "db_connection" in item.fixturenames:
            item.add_marker(pytest.mark.database)

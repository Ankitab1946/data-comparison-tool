"""Helper utilities for test framework."""
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from config.settings import BASE_DIR, EXPLICIT_WAIT, RETRY_DELAY, MAX_RETRIES
from utils.logger import TestLogger

logger = TestLogger.get_logger("Helpers")


class WaitHelper:
    """Helper class for explicit waits."""
    
    @staticmethod
    def wait_for_element(driver, locator: tuple, timeout: int = EXPLICIT_WAIT):
        """
        Wait for element to be present.
        
        Args:
            driver: WebDriver instance
            locator: Tuple of (By, value)
            timeout: Maximum wait time in seconds
            
        Returns:
            WebElement if found
            
        Raises:
            TimeoutException if element not found
        """
        try:
            element = WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located(locator)
            )
            logger.debug(f"Element found: {locator}")
            return element
        except TimeoutException:
            logger.error(f"Element not found within {timeout}s: {locator}")
            raise
    
    @staticmethod
    def wait_for_element_clickable(driver, locator: tuple, timeout: int = EXPLICIT_WAIT):
        """
        Wait for element to be clickable.
        
        Args:
            driver: WebDriver instance
            locator: Tuple of (By, value)
            timeout: Maximum wait time in seconds
            
        Returns:
            WebElement if clickable
            
        Raises:
            TimeoutException if element not clickable
        """
        try:
            element = WebDriverWait(driver, timeout).until(
                EC.element_to_be_clickable(locator)
            )
            logger.debug(f"Element clickable: {locator}")
            return element
        except TimeoutException:
            logger.error(f"Element not clickable within {timeout}s: {locator}")
            raise
    
    @staticmethod
    def wait_for_element_visible(driver, locator: tuple, timeout: int = EXPLICIT_WAIT):
        """
        Wait for element to be visible.
        
        Args:
            driver: WebDriver instance
            locator: Tuple of (By, value)
            timeout: Maximum wait time in seconds
            
        Returns:
            WebElement if visible
            
        Raises:
            TimeoutException if element not visible
        """
        try:
            element = WebDriverWait(driver, timeout).until(
                EC.visibility_of_element_located(locator)
            )
            logger.debug(f"Element visible: {locator}")
            return element
        except TimeoutException:
            logger.error(f"Element not visible within {timeout}s: {locator}")
            raise
    
    @staticmethod
    def wait_for_text_in_element(driver, locator: tuple, text: str, 
                                 timeout: int = EXPLICIT_WAIT):
        """
        Wait for specific text to appear in element.
        
        Args:
            driver: WebDriver instance
            locator: Tuple of (By, value)
            text: Expected text
            timeout: Maximum wait time in seconds
            
        Returns:
            True if text found
            
        Raises:
            TimeoutException if text not found
        """
        try:
            result = WebDriverWait(driver, timeout).until(
                EC.text_to_be_present_in_element(locator, text)
            )
            logger.debug(f"Text '{text}' found in element: {locator}")
            return result
        except TimeoutException:
            logger.error(f"Text '{text}' not found in element within {timeout}s: {locator}")
            raise


class RetryHelper:
    """Helper class for retry logic."""
    
    @staticmethod
    def retry_on_exception(func, max_retries: int = MAX_RETRIES, 
                          delay: int = RETRY_DELAY, exceptions: tuple = (Exception,)):
        """
        Retry function on exception.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retries
            delay: Delay between retries in seconds
            exceptions: Tuple of exceptions to catch
            
        Returns:
            Function result if successful
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries}")
                return func()
            except exceptions as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
        
        logger.error(f"All {max_retries} attempts failed")
        raise last_exception


class DataHelper:
    """Helper class for test data management."""
    
    @staticmethod
    def load_yaml(filename: str) -> Dict[str, Any]:
        """
        Load YAML file.
        
        Args:
            filename: Name of YAML file
            
        Returns:
            Dictionary with YAML content
        """
        try:
            filepath = BASE_DIR / "config" / filename
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            logger.debug(f"Loaded YAML file: {filepath}")
            return data
        except Exception as e:
            logger.error(f"Failed to load YAML file {filename}: {str(e)}")
            raise
    
    @staticmethod
    def get_test_data(data_key: str) -> Any:
        """
        Get test data from test_data.yaml.
        
        Args:
            data_key: Key to retrieve from test data
            
        Returns:
            Test data value
        """
        try:
            data = DataHelper.load_yaml("test_data.yaml")
            return data.get(data_key)
        except Exception as e:
            logger.error(f"Failed to get test data for key '{data_key}': {str(e)}")
            raise


class StringHelper:
    """Helper class for string operations."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Remove extra whitespace and newlines."""
        return ' '.join(text.split())
    
    @staticmethod
    def extract_number(text: str) -> Optional[int]:
        """Extract first number from text."""
        import re
        match = re.search(r'\d+', text)
        return int(match.group()) if match else None
    
    @staticmethod
    def format_test_name(name: str) -> str:
        """Format test name for file naming."""
        return name.replace(" ", "_").replace("::", "_").lower()

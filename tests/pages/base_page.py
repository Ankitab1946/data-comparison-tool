"""Base page object with common functionality."""
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from typing import List, Optional, Tuple
import allure

from config.settings import EXPLICIT_WAIT, IMPLICIT_WAIT
from utils.logger import TestLogger
from utils.screenshot import ScreenshotCapture
from utils.helpers import WaitHelper

logger = TestLogger.get_logger("BasePage")


class BasePage:
    """Base page object class with common methods."""
    
    def __init__(self, driver: WebDriver):
        """
        Initialize base page.
        
        Args:
            driver: Selenium WebDriver instance
        """
        self.driver = driver
        self.wait = WebDriverWait(driver, EXPLICIT_WAIT)
        self.wait_helper = WaitHelper()
        self.screenshot = ScreenshotCapture()
    
    @allure.step("Navigate to URL: {url}")
    def navigate_to(self, url: str):
        """
        Navigate to URL.
        
        Args:
            url: URL to navigate to
        """
        logger.info(f"Navigating to: {url}")
        self.driver.get(url)
    
    @allure.step("Find element: {locator}")
    def find_element(self, locator: Tuple[By, str]) -> WebElement:
        """
        Find element by locator.
        
        Args:
            locator: Tuple of (By, value)
            
        Returns:
            WebElement
        """
        logger.debug(f"Finding element: {locator}")
        return self.wait_helper.wait_for_element(self.driver, locator)
    
    @allure.step("Find elements: {locator}")
    def find_elements(self, locator: Tuple[By, str]) -> List[WebElement]:
        """
        Find multiple elements by locator.
        
        Args:
            locator: Tuple of (By, value)
            
        Returns:
            List of WebElements
        """
        logger.debug(f"Finding elements: {locator}")
        return self.driver.find_elements(*locator)
    
    @allure.step("Click element: {locator}")
    def click(self, locator: Tuple[By, str]):
        """
        Click element.
        
        Args:
            locator: Tuple of (By, value)
        """
        logger.info(f"Clicking element: {locator}")
        element = self.wait_helper.wait_for_element_clickable(self.driver, locator)
        element.click()
    
    @allure.step("Enter text: {text}")
    def enter_text(self, locator: Tuple[By, str], text: str, clear_first: bool = True):
        """
        Enter text into element.
        
        Args:
            locator: Tuple of (By, value)
            text: Text to enter
            clear_first: Clear field before entering text
        """
        logger.info(f"Entering text into {locator}: {text}")
        element = self.find_element(locator)
        
        if clear_first:
            element.clear()
        
        element.send_keys(text)
    
    @allure.step("Get text from element: {locator}")
    def get_text(self, locator: Tuple[By, str]) -> str:
        """
        Get text from element.
        
        Args:
            locator: Tuple of (By, value)
            
        Returns:
            Element text
        """
        logger.debug(f"Getting text from: {locator}")
        element = self.find_element(locator)
        return element.text
    
    @allure.step("Get attribute: {attribute}")
    def get_attribute(self, locator: Tuple[By, str], attribute: str) -> str:
        """
        Get attribute value from element.
        
        Args:
            locator: Tuple of (By, value)
            attribute: Attribute name
            
        Returns:
            Attribute value
        """
        logger.debug(f"Getting attribute '{attribute}' from: {locator}")
        element = self.find_element(locator)
        return element.get_attribute(attribute)
    
    @allure.step("Check if element is displayed: {locator}")
    def is_displayed(self, locator: Tuple[By, str]) -> bool:
        """
        Check if element is displayed.
        
        Args:
            locator: Tuple of (By, value)
            
        Returns:
            True if displayed, False otherwise
        """
        try:
            element = self.find_element(locator)
            result = element.is_displayed()
            logger.debug(f"Element {locator} displayed: {result}")
            return result
        except Exception:
            logger.debug(f"Element {locator} not displayed")
            return False
    
    @allure.step("Check if element is enabled: {locator}")
    def is_enabled(self, locator: Tuple[By, str]) -> bool:
        """
        Check if element is enabled.
        
        Args:
            locator: Tuple of (By, value)
            
        Returns:
            True if enabled, False otherwise
        """
        try:
            element = self.find_element(locator)
            result = element.is_enabled()
            logger.debug(f"Element {locator} enabled: {result}")
            return result
        except Exception:
            logger.debug(f"Element {locator} not enabled")
            return False
    
    @allure.step("Wait for element to be visible: {locator}")
    def wait_for_visible(self, locator: Tuple[By, str], timeout: int = EXPLICIT_WAIT):
        """
        Wait for element to be visible.
        
        Args:
            locator: Tuple of (By, value)
            timeout: Maximum wait time
        """
        logger.debug(f"Waiting for element to be visible: {locator}")
        self.wait_helper.wait_for_element_visible(self.driver, locator, timeout)
    
    @allure.step("Wait for text in element: {text}")
    def wait_for_text(self, locator: Tuple[By, str], text: str, timeout: int = EXPLICIT_WAIT):
        """
        Wait for text to appear in element.
        
        Args:
            locator: Tuple of (By, value)
            text: Expected text
            timeout: Maximum wait time
        """
        logger.debug(f"Waiting for text '{text}' in element: {locator}")
        self.wait_helper.wait_for_text_in_element(self.driver, locator, text, timeout)
    
    @allure.step("Scroll to element: {locator}")
    def scroll_to_element(self, locator: Tuple[By, str]):
        """
        Scroll to element.
        
        Args:
            locator: Tuple of (By, value)
        """
        logger.debug(f"Scrolling to element: {locator}")
        element = self.find_element(locator)
        self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
    
    @allure.step("Hover over element: {locator}")
    def hover(self, locator: Tuple[By, str]):
        """
        Hover over element.
        
        Args:
            locator: Tuple of (By, value)
        """
        logger.debug(f"Hovering over element: {locator}")
        element = self.find_element(locator)
        ActionChains(self.driver).move_to_element(element).perform()
    
    @allure.step("Select dropdown option by text: {text}")
    def select_dropdown_by_text(self, locator: Tuple[By, str], text: str):
        """
        Select dropdown option by visible text.
        
        Args:
            locator: Tuple of (By, value)
            text: Option text
        """
        from selenium.webdriver.support.select import Select
        
        logger.info(f"Selecting dropdown option '{text}' from: {locator}")
        element = self.find_element(locator)
        select = Select(element)
        select.select_by_visible_text(text)
    
    @allure.step("Execute JavaScript: {script}")
    def execute_script(self, script: str, *args):
        """
        Execute JavaScript.
        
        Args:
            script: JavaScript code
            *args: Arguments to pass to script
            
        Returns:
            Script result
        """
        logger.debug(f"Executing JavaScript: {script}")
        return self.driver.execute_script(script, *args)
    
    @allure.step("Get page title")
    def get_title(self) -> str:
        """
        Get page title.
        
        Returns:
            Page title
        """
        title = self.driver.title
        logger.debug(f"Page title: {title}")
        return title
    
    @allure.step("Get current URL")
    def get_current_url(self) -> str:
        """
        Get current URL.
        
        Returns:
            Current URL
        """
        url = self.driver.current_url
        logger.debug(f"Current URL: {url}")
        return url
    
    @allure.step("Refresh page")
    def refresh(self):
        """Refresh current page."""
        logger.info("Refreshing page")
        self.driver.refresh()
    
    @allure.step("Switch to frame: {locator}")
    def switch_to_frame(self, locator: Tuple[By, str]):
        """
        Switch to iframe.
        
        Args:
            locator: Tuple of (By, value)
        """
        logger.info(f"Switching to frame: {locator}")
        frame = self.find_element(locator)
        self.driver.switch_to.frame(frame)
    
    @allure.step("Switch to default content")
    def switch_to_default_content(self):
        """Switch back to default content."""
        logger.info("Switching to default content")
        self.driver.switch_to.default_content()
    
    def take_screenshot(self, name: str) -> str:
        """
        Take screenshot.
        
        Args:
            name: Screenshot name
            
        Returns:
            Path to screenshot
        """
        return self.screenshot.capture(self.driver, name, "page")

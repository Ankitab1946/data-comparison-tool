"""Autosys portal page object."""
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from typing import Dict, Optional
import allure

from pages.base_page import BasePage
from utils.logger import TestLogger

logger = TestLogger.get_logger("AutosysPage")


class AutosysPage(BasePage):
    """Page object for Autosys portal."""
    
    # Locators
    USERNAME_INPUT = (By.ID, "username")
    PASSWORD_INPUT = (By.ID, "password")
    LOGIN_BUTTON = (By.ID, "login-button")
    
    SEARCH_BOX = (By.ID, "search-box")
    SEARCH_BUTTON = (By.ID, "search-button")
    
    BOX_NAME_LABEL = (By.XPATH, "//label[text()='Box Name:']/following-sibling::span")
    DATE_CONDITION_FIELD = (By.XPATH, "//label[text()='Date Condition:']/following-sibling::input")
    ALARM_IF_FAIL_FIELD = (By.XPATH, "//label[text()='Alarm If Fail:']/following-sibling::input")
    ALARM_IF_TERMINATED_FIELD = (By.XPATH, "//label[text()='Alarm If Terminated:']/following-sibling::input")
    
    JOB_TYPE_FIELD = (By.XPATH, "//label[text()='Job Type:']/following-sibling::span")
    FILEWATCHER_SECTION = (By.ID, "filewatcher-section")
    FILEWATCHER_JOB_NAME = (By.XPATH, "//div[@id='filewatcher-section']//span[@class='job-name']")
    
    JOBS_TAB = (By.ID, "jobs-tab")
    JOBS_TABLE = (By.ID, "jobs-table")
    JOB_ROW = (By.XPATH, "//table[@id='jobs-table']//tr[@data-job-name='{job_name}']")
    
    PARAMETERS_TAB = (By.ID, "parameters-tab")
    PARAMETER_ROW = (By.XPATH, "//table[@id='parameters-table']//tr[td[text()='{param_name}']]")
    PARAMETER_VALUE = (By.XPATH, "//table[@id='parameters-table']//tr[td[text()='{param_name}']]/td[2]")
    
    ERROR_MESSAGE = (By.CLASS_NAME, "error-message")
    SUCCESS_MESSAGE = (By.CLASS_NAME, "success-message")
    LOADING_SPINNER = (By.CLASS_NAME, "loading-spinner")
    
    def __init__(self, driver: WebDriver):
        """
        Initialize Autosys page.
        
        Args:
            driver: Selenium WebDriver instance
        """
        super().__init__(driver)
    
    @allure.step("Login to Autosys portal")
    def login(self, username: str, password: str):
        """
        Login to Autosys portal.
        
        Args:
            username: Username
            password: Password
        """
        logger.info(f"Logging in as: {username}")
        
        self.enter_text(self.USERNAME_INPUT, username)
        self.enter_text(self.PASSWORD_INPUT, password)
        self.click(self.LOGIN_BUTTON)
        
        # Wait for login to complete
        self.wait_for_page_load()
        logger.info("Login successful")
    
    @allure.step("Search for box: {box_name}")
    def search_box(self, box_name: str):
        """
        Search for a box by name.
        
        Args:
            box_name: Name of the box to search
        """
        logger.info(f"Searching for box: {box_name}")
        
        self.enter_text(self.SEARCH_BOX, box_name)
        self.click(self.SEARCH_BUTTON)
        
        # Wait for search results
        self.wait_for_page_load()
        logger.info(f"Search completed for: {box_name}")
    
    @allure.step("Get date condition value")
    def get_date_condition(self) -> int:
        """
        Get date condition value.
        
        Returns:
            Date condition value as integer
        """
        logger.info("Getting date condition value")
        value = self.get_attribute(self.DATE_CONDITION_FIELD, "value")
        logger.info(f"Date condition value: {value}")
        return int(value)
    
    @allure.step("Get alarm if fail parameter")
    def get_alarm_if_fail(self) -> int:
        """
        Get alarm_if_fail parameter value.
        
        Returns:
            alarm_if_fail value as integer
        """
        logger.info("Getting alarm_if_fail parameter")
        
        # Navigate to parameters tab
        self.click(self.PARAMETERS_TAB)
        self.wait_for_page_load()
        
        # Get parameter value
        locator = (
            By.XPATH,
            self.PARAMETER_VALUE[1].format(param_name="alarm_if_fail")
        )
        value = self.get_text(locator)
        logger.info(f"alarm_if_fail value: {value}")
        return int(value)
    
    @allure.step("Get alarm if terminated parameter")
    def get_alarm_if_terminated(self) -> int:
        """
        Get alarm_if_terminated parameter value.
        
        Returns:
            alarm_if_terminated value as integer
        """
        logger.info("Getting alarm_if_terminated parameter")
        
        # Navigate to parameters tab if not already there
        if not self.is_displayed(self.PARAMETER_ROW):
            self.click(self.PARAMETERS_TAB)
            self.wait_for_page_load()
        
        # Get parameter value
        locator = (
            By.XPATH,
            self.PARAMETER_VALUE[1].format(param_name="alarm_if_terminated")
        )
        value = self.get_text(locator)
        logger.info(f"alarm_if_terminated value: {value}")
        return int(value)
    
    @allure.step("Check if filewatcher exists for job: {job_name}")
    def has_filewatcher_for_job(self, job_name: str) -> bool:
        """
        Check if filewatcher job exists for a given job.
        
        Args:
            job_name: Name of the job
            
        Returns:
            True if filewatcher exists, False otherwise
        """
        logger.info(f"Checking filewatcher for job: {job_name}")
        
        # Navigate to jobs tab
        self.click(self.JOBS_TAB)
        self.wait_for_page_load()
        
        # Search for the job
        job_locator = (
            By.XPATH,
            self.JOB_ROW[1].format(job_name=job_name)
        )
        
        if not self.is_displayed(job_locator):
            logger.warning(f"Job not found: {job_name}")
            return False
        
        # Click on the job to view details
        self.click(job_locator)
        self.wait_for_page_load()
        
        # Check if filewatcher section exists
        has_fw = self.is_displayed(self.FILEWATCHER_SECTION)
        logger.info(f"Filewatcher exists for {job_name}: {has_fw}")
        return has_fw
    
    @allure.step("Get filewatcher job name")
    def get_filewatcher_job_name(self) -> Optional[str]:
        """
        Get filewatcher job name.
        
        Returns:
            Filewatcher job name or None if not found
        """
        logger.info("Getting filewatcher job name")
        
        if not self.is_displayed(self.FILEWATCHER_SECTION):
            logger.warning("Filewatcher section not found")
            return None
        
        fw_name = self.get_text(self.FILEWATCHER_JOB_NAME)
        logger.info(f"Filewatcher job name: {fw_name}")
        return fw_name
    
    @allure.step("Get job type")
    def get_job_type(self) -> str:
        """
        Get job type.
        
        Returns:
            Job type
        """
        logger.info("Getting job type")
        job_type = self.get_text(self.JOB_TYPE_FIELD)
        logger.info(f"Job type: {job_type}")
        return job_type
    
    @allure.step("Get all box parameters")
    def get_box_parameters(self) -> Dict[str, str]:
        """
        Get all box parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
        logger.info("Getting all box parameters")
        
        # Navigate to parameters tab
        self.click(self.PARAMETERS_TAB)
        self.wait_for_page_load()
        
        # Get all parameter rows
        rows = self.find_elements((By.XPATH, "//table[@id='parameters-table']//tr"))
        
        parameters = {}
        for row in rows[1:]:  # Skip header row
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) >= 2:
                param_name = cells[0].text
                param_value = cells[1].text
                parameters[param_name] = param_value
        
        logger.info(f"Retrieved {len(parameters)} parameters")
        return parameters
    
    @allure.step("Wait for page to load")
    def wait_for_page_load(self):
        """Wait for page to finish loading."""
        # Wait for loading spinner to disappear
        try:
            self.wait.until_not(
                lambda driver: self.is_displayed(self.LOADING_SPINNER)
            )
        except Exception:
            pass  # Spinner might not be present
        
        logger.debug("Page loaded")
    
    @allure.step("Check if error message is displayed")
    def has_error(self) -> bool:
        """
        Check if error message is displayed.
        
        Returns:
            True if error exists, False otherwise
        """
        return self.is_displayed(self.ERROR_MESSAGE)
    
    @allure.step("Get error message")
    def get_error_message(self) -> Optional[str]:
        """
        Get error message text.
        
        Returns:
            Error message or None
        """
        if self.has_error():
            return self.get_text(self.ERROR_MESSAGE)
        return None

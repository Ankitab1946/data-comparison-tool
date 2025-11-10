"""Screenshot utility for capturing test evidence."""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from selenium.webdriver.remote.webdriver import WebDriver
from PIL import Image
import io

from config.settings import SCREENSHOTS_DIR
from utils.logger import TestLogger

logger = TestLogger.get_logger("Screenshot")


class ScreenshotCapture:
    """Utility class for capturing and managing screenshots."""
    
    @staticmethod
    def capture(driver: WebDriver, test_name: str, step: str = "default") -> Optional[str]:
        """
        Capture screenshot and save to file.
        
        Args:
            driver: Selenium WebDriver instance
            test_name: Name of the test
            step: Step description or identifier
            
        Returns:
            Path to saved screenshot or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{test_name}_{step}_{timestamp}.png"
            filepath = SCREENSHOTS_DIR / filename
            
            # Capture screenshot
            screenshot = driver.get_screenshot_as_png()
            
            # Save screenshot
            with open(filepath, 'wb') as f:
                f.write(screenshot)
            
            logger.info(f"Screenshot captured: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {str(e)}")
            return None
    
    @staticmethod
    def capture_element(driver: WebDriver, element, test_name: str, 
                       element_name: str = "element") -> Optional[str]:
        """
        Capture screenshot of specific element.
        
        Args:
            driver: Selenium WebDriver instance
            element: WebElement to capture
            test_name: Name of the test
            element_name: Name/description of the element
            
        Returns:
            Path to saved screenshot or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{test_name}_{element_name}_{timestamp}.png"
            filepath = SCREENSHOTS_DIR / filename
            
            # Capture element screenshot
            screenshot = element.screenshot_as_png
            
            # Save screenshot
            with open(filepath, 'wb') as f:
                f.write(screenshot)
            
            logger.info(f"Element screenshot captured: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to capture element screenshot: {str(e)}")
            return None
    
    @staticmethod
    def capture_full_page(driver: WebDriver, test_name: str) -> Optional[str]:
        """
        Capture full page screenshot (scrolling if needed).
        
        Args:
            driver: Selenium WebDriver instance
            test_name: Name of the test
            
        Returns:
            Path to saved screenshot or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{test_name}_fullpage_{timestamp}.png"
            filepath = SCREENSHOTS_DIR / filename
            
            # Get page dimensions
            total_height = driver.execute_script("return document.body.scrollHeight")
            viewport_height = driver.execute_script("return window.innerHeight")
            total_width = driver.execute_script("return document.body.scrollWidth")
            
            # Create list to store screenshot parts
            screenshots = []
            offset = 0
            
            while offset < total_height:
                # Scroll to position
                driver.execute_script(f"window.scrollTo(0, {offset});")
                
                # Wait for scroll
                driver.implicitly_wait(0.5)
                
                # Capture screenshot
                screenshot = driver.get_screenshot_as_png()
                screenshots.append(Image.open(io.BytesIO(screenshot)))
                
                offset += viewport_height
            
            # Stitch screenshots together
            stitched_image = Image.new('RGB', (total_width, total_height))
            offset = 0
            
            for screenshot in screenshots:
                stitched_image.paste(screenshot, (0, offset))
                offset += screenshot.size[1]
            
            # Save stitched screenshot
            stitched_image.save(filepath)
            
            # Scroll back to top
            driver.execute_script("window.scrollTo(0, 0);")
            
            logger.info(f"Full page screenshot captured: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to capture full page screenshot: {str(e)}")
            return None
    
    @staticmethod
    def attach_to_allure(screenshot_path: str, name: str = "Screenshot"):
        """
        Attach screenshot to Allure report.
        
        Args:
            screenshot_path: Path to screenshot file
            name: Name for the attachment
        """
        try:
            import allure
            
            with open(screenshot_path, 'rb') as f:
                allure.attach(
                    f.read(),
                    name=name,
                    attachment_type=allure.attachment_type.PNG
                )
            logger.debug(f"Screenshot attached to Allure: {name}")
            
        except ImportError:
            logger.warning("Allure not available, skipping attachment")
        except Exception as e:
            logger.error(f"Failed to attach screenshot to Allure: {str(e)}")

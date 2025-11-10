"""Autosys portal test cases."""
import pytest
import allure
from sqlalchemy import text

from config.settings import (
    AUTOSYS_USERNAME, AUTOSYS_PASSWORD,
    EXPECTED_DATE_CONDITION, EXPECTED_ALARM_IF_FAIL, EXPECTED_ALARM_IF_TERMINATED
)
from utils.logger import TestLogger

logger = TestLogger.get_logger("AutosysTests")


@allure.feature("Autosys Portal")
@allure.story("Box Configuration Validation")
class TestAutosysBoxConfiguration:
    """Test cases for Autosys box configuration validation."""
    
    @allure.title("TC001: Verify BOX Autosys Date Condition must be 1")
    @allure.description("Verify that the BOX Autosys Date Condition parameter is set to 1")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.smoke
    @pytest.mark.critical
    @pytest.mark.parametrize("box_name", [
        "TEST_BOX_001",
        "TEST_BOX_002"
    ])
    def test_verify_date_condition(self, autosys_page, box_name):
        """
        Test to verify BOX Autosys Date Condition is 1.
        
        Steps:
        1. Login to Autosys portal
        2. Search for the box
        3. Verify Date Condition value is 1
        """
        with allure.step(f"Login to Autosys portal as {AUTOSYS_USERNAME}"):
            autosys_page.login(AUTOSYS_USERNAME, AUTOSYS_PASSWORD)
            logger.info("Login successful")
        
        with allure.step(f"Search for box: {box_name}"):
            autosys_page.search_box(box_name)
            logger.info(f"Box search completed: {box_name}")
        
        with allure.step("Get Date Condition value"):
            date_condition = autosys_page.get_date_condition()
            logger.info(f"Date Condition value: {date_condition}")
            
            allure.attach(
                f"Box Name: {box_name}\nDate Condition: {date_condition}",
                name="Date Condition Details",
                attachment_type=allure.attachment_type.TEXT
            )
        
        with allure.step(f"Verify Date Condition is {EXPECTED_DATE_CONDITION}"):
            assert date_condition == EXPECTED_DATE_CONDITION, \
                f"Date Condition mismatch! Expected: {EXPECTED_DATE_CONDITION}, Actual: {date_condition}"
            
            logger.info(f"✓ Date Condition verified: {date_condition}")
            TestLogger.log_assertion(
                f"Date Condition == {EXPECTED_DATE_CONDITION}",
                True
            )
    
    @allure.title("TC002: Verify BOX Autosys Parameter 'alarm_if_fail' must be 1")
    @allure.description("Verify that the BOX Autosys parameter 'alarm_if_fail' is set to 1")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.smoke
    @pytest.mark.critical
    @pytest.mark.parametrize("box_name", [
        "TEST_BOX_001",
        "TEST_BOX_002"
    ])
    def test_verify_alarm_if_fail(self, autosys_page, box_name):
        """
        Test to verify alarm_if_fail parameter is 1.
        
        Steps:
        1. Login to Autosys portal
        2. Search for the box
        3. Navigate to Parameters tab
        4. Verify alarm_if_fail value is 1
        """
        with allure.step(f"Login to Autosys portal as {AUTOSYS_USERNAME}"):
            autosys_page.login(AUTOSYS_USERNAME, AUTOSYS_PASSWORD)
            logger.info("Login successful")
        
        with allure.step(f"Search for box: {box_name}"):
            autosys_page.search_box(box_name)
            logger.info(f"Box search completed: {box_name}")
        
        with allure.step("Get alarm_if_fail parameter value"):
            alarm_if_fail = autosys_page.get_alarm_if_fail()
            logger.info(f"alarm_if_fail value: {alarm_if_fail}")
            
            allure.attach(
                f"Box Name: {box_name}\nalarm_if_fail: {alarm_if_fail}",
                name="alarm_if_fail Details",
                attachment_type=allure.attachment_type.TEXT
            )
        
        with allure.step(f"Verify alarm_if_fail is {EXPECTED_ALARM_IF_FAIL}"):
            assert alarm_if_fail == EXPECTED_ALARM_IF_FAIL, \
                f"alarm_if_fail mismatch! Expected: {EXPECTED_ALARM_IF_FAIL}, Actual: {alarm_if_fail}"
            
            logger.info(f"✓ alarm_if_fail verified: {alarm_if_fail}")
            TestLogger.log_assertion(
                f"alarm_if_fail == {EXPECTED_ALARM_IF_FAIL}",
                True
            )
    
    @allure.title("TC003: Verify BOX Autosys Parameter 'alarm_if_terminated' must be 0")
    @allure.description("Verify that the BOX Autosys parameter 'alarm_if_terminated' is set to 0")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.smoke
    @pytest.mark.critical
    @pytest.mark.parametrize("box_name", [
        "TEST_BOX_001",
        "TEST_BOX_002"
    ])
    def test_verify_alarm_if_terminated(self, autosys_page, box_name):
        """
        Test to verify alarm_if_terminated parameter is 0.
        
        Steps:
        1. Login to Autosys portal
        2. Search for the box
        3. Navigate to Parameters tab
        4. Verify alarm_if_terminated value is 0
        """
        with allure.step(f"Login to Autosys portal as {AUTOSYS_USERNAME}"):
            autosys_page.login(AUTOSYS_USERNAME, AUTOSYS_PASSWORD)
            logger.info("Login successful")
        
        with allure.step(f"Search for box: {box_name}"):
            autosys_page.search_box(box_name)
            logger.info(f"Box search completed: {box_name}")
        
        with allure.step("Get alarm_if_terminated parameter value"):
            alarm_if_terminated = autosys_page.get_alarm_if_terminated()
            logger.info(f"alarm_if_terminated value: {alarm_if_terminated}")
            
            allure.attach(
                f"Box Name: {box_name}\nalarm_if_terminated: {alarm_if_terminated}",
                name="alarm_if_terminated Details",
                attachment_type=allure.attachment_type.TEXT
            )
        
        with allure.step(f"Verify alarm_if_terminated is {EXPECTED_ALARM_IF_TERMINATED}"):
            assert alarm_if_terminated == EXPECTED_ALARM_IF_TERMINATED, \
                f"alarm_if_terminated mismatch! Expected: {EXPECTED_ALARM_IF_TERMINATED}, Actual: {alarm_if_terminated}"
            
            logger.info(f"✓ alarm_if_terminated verified: {alarm_if_terminated}")
            TestLogger.log_assertion(
                f"alarm_if_terminated == {EXPECTED_ALARM_IF_TERMINATED}",
                True
            )
    
    @allure.title("TC004: Verify Filewatcher job exists for Load Job")
    @allure.description("Verify that a Filewatcher job exists for any Load Job")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.regression
    @pytest.mark.critical
    @pytest.mark.parametrize("job_data", [
        {"box_name": "TEST_BOX_001", "job_name": "LOAD_JOB_001", "job_type": "Load"},
        {"box_name": "TEST_BOX_002", "job_name": "LOAD_JOB_002", "job_type": "Load"}
    ])
    def test_verify_filewatcher_for_load_job(self, autosys_page, job_data):
        """
        Test to verify Filewatcher job exists for Load jobs.
        
        Steps:
        1. Login to Autosys portal
        2. Search for the box
        3. Navigate to Jobs tab
        4. Search for the Load job
        5. Verify Filewatcher job exists
        """
        box_name = job_data["box_name"]
        job_name = job_data["job_name"]
        job_type = job_data["job_type"]
        
        with allure.step(f"Login to Autosys portal as {AUTOSYS_USERNAME}"):
            autosys_page.login(AUTOSYS_USERNAME, AUTOSYS_PASSWORD)
            logger.info("Login successful")
        
        with allure.step(f"Search for box: {box_name}"):
            autosys_page.search_box(box_name)
            logger.info(f"Box search completed: {box_name}")
        
        with allure.step(f"Verify job type is {job_type}"):
            actual_job_type = autosys_page.get_job_type()
            logger.info(f"Job type: {actual_job_type}")
            
            assert actual_job_type == job_type, \
                f"Job type mismatch! Expected: {job_type}, Actual: {actual_job_type}"
        
        with allure.step(f"Check if Filewatcher exists for job: {job_name}"):
            has_filewatcher = autosys_page.has_filewatcher_for_job(job_name)
            logger.info(f"Filewatcher exists: {has_filewatcher}")
            
            if has_filewatcher:
                fw_job_name = autosys_page.get_filewatcher_job_name()
                logger.info(f"Filewatcher job name: {fw_job_name}")
                
                allure.attach(
                    f"Box Name: {box_name}\nJob Name: {job_name}\n"
                    f"Job Type: {job_type}\nFilewatcher: {fw_job_name}",
                    name="Filewatcher Details",
                    attachment_type=allure.attachment_type.TEXT
                )
        
        with allure.step("Verify Filewatcher exists"):
            assert has_filewatcher, \
                f"Filewatcher job not found for Load job: {job_name}"
            
            logger.info(f"✓ Filewatcher verified for job: {job_name}")
            TestLogger.log_assertion(
                f"Filewatcher exists for {job_name}",
                True
            )
    
    @allure.title("TC005: Verify job load on SQL Server")
    @allure.description("Verify that the job load is successfully recorded in SQL Server")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.regression
    @pytest.mark.database
    @pytest.mark.parametrize("job_name,expected_status", [
        ("LOAD_JOB_001", "SUCCESS"),
        ("LOAD_JOB_002", "SUCCESS")
    ])
    def test_verify_job_load_on_sql_server(self, db_connection, job_name, expected_status):
        """
        Test to verify job load on SQL Server.
        
        Steps:
        1. Connect to SQL Server
        2. Query LoadStatus table for the job
        3. Verify job status is SUCCESS
        4. Verify load count
        """
        with allure.step("Connect to SQL Server"):
            logger.info("Database connection established")
        
        with allure.step(f"Query LoadStatus for job: {job_name}"):
            query = text(
                "SELECT COUNT(*) as count FROM dbo.LoadStatus "
                "WHERE job_name = :job_name AND status = :status"
            )
            
            with db_connection.connect() as conn:
                result = conn.execute(
                    query,
                    {"job_name": job_name, "status": expected_status}
                )
                row = result.fetchone()
                load_count = row[0] if row else 0
            
            logger.info(f"Load count for {job_name}: {load_count}")
            
            allure.attach(
                f"Job Name: {job_name}\nExpected Status: {expected_status}\n"
                f"Load Count: {load_count}",
                name="SQL Server Load Details",
                attachment_type=allure.attachment_type.TEXT
            )
        
        with allure.step("Verify job load exists"):
            assert load_count > 0, \
                f"No {expected_status} load found for job: {job_name}"
            
            logger.info(f"✓ Job load verified: {job_name} - {load_count} records")
            TestLogger.log_assertion(
                f"Job load exists for {job_name}",
                True
            )
        
        with allure.step("Get latest job status"):
            status_query = text(
                "SELECT TOP 1 status, load_date FROM dbo.LoadStatus "
                "WHERE job_name = :job_name ORDER BY load_date DESC"
            )
            
            with db_connection.connect() as conn:
                result = conn.execute(status_query, {"job_name": job_name})
                row = result.fetchone()
                
                if row:
                    latest_status = row[0]
                    load_date = row[1]
                    
                    logger.info(f"Latest status: {latest_status}, Load date: {load_date}")
                    
                    allure.attach(
                        f"Latest Status: {latest_status}\nLoad Date: {load_date}",
                        name="Latest Load Status",
                        attachment_type=allure.attachment_type.TEXT
                    )
                    
                    assert latest_status == expected_status, \
                        f"Latest status mismatch! Expected: {expected_status}, Actual: {latest_status}"


@allure.feature("Autosys Portal")
@allure.story("Test Execution Modes")
class TestExecutionModes:
    """Test cases demonstrating different execution modes."""
    
    @allure.title("Demo: Test that can be skipped")
    @allure.description("Demonstration of test skip functionality")
    @pytest.mark.skip(reason="Demonstration of skip marker")
    def test_skip_demo(self, autosys_page):
        """This test will be skipped."""
        logger.info("This test should be skipped")
        assert False, "This should not execute"
    
    @allure.title("Demo: Test with conditional skip")
    @allure.description("Demonstration of conditional skip")
    @pytest.mark.skipif(
        EXPECTED_DATE_CONDITION != 1,
        reason="Date condition not configured for testing"
    )
    def test_conditional_skip_demo(self, autosys_page):
        """This test will be skipped conditionally."""
        logger.info("This test runs only if date condition is 1")
        assert True
    
    @allure.title("Demo: Slow running test")
    @allure.description("Demonstration of slow test marker")
    @pytest.mark.slow
    def test_slow_demo(self):
        """This test is marked as slow."""
        import time
        logger.info("Running slow test")
        time.sleep(2)
        assert True
    
    @allure.title("Demo: Parameterized test with multiple runs")
    @allure.description("Demonstration of parameterized test execution")
    @pytest.mark.parametrize("iteration", range(1, 4))
    def test_multiple_runs_demo(self, iteration):
        """This test runs multiple times with different parameters."""
        logger.info(f"Running iteration: {iteration}")
        assert iteration > 0

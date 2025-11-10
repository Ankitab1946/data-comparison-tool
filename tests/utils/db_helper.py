"""Database helper utilities for SQL Server validation."""
from typing import Any, Dict, List, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from urllib.parse import quote_plus
import pandas as pd

from config.settings import DB_SERVER, DB_NAME, DB_USERNAME, DB_PASSWORD, DB_DRIVER
from utils.logger import TestLogger

logger = TestLogger.get_logger("DBHelper")


class DatabaseHelper:
    """Helper class for database operations."""
    
    def __init__(self, server: str = None, database: str = None, 
                 username: str = None, password: str = None, driver: str = None):
        """
        Initialize database helper.
        
        Args:
            server: Database server
            database: Database name
            username: Database username
            password: Database password
            driver: ODBC driver name
        """
        self.server = server or DB_SERVER
        self.database = database or DB_NAME
        self.username = username or DB_USERNAME
        self.password = password or DB_PASSWORD
        self.driver = driver or DB_DRIVER
        self.engine: Optional[Engine] = None
    
    def connect(self) -> Engine:
        """
        Create database connection.
        
        Returns:
            SQLAlchemy engine
        """
        try:
            connection_string = (
                f"mssql+pyodbc://{self.username}:{quote_plus(self.password)}"
                f"@{self.server}/{self.database}?driver={quote_plus(self.driver)}"
            )
            
            self.engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(f"Connected to database: {self.server}/{self.database}")
            return self.engine
            
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute SQL query and return results.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            List of dictionaries with query results
        """
        try:
            if not self.engine:
                self.connect()
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                
                # Convert to list of dictionaries
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                
                logger.info(f"Query executed successfully. Rows returned: {len(rows)}")
                return rows
                
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def execute_query_df(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Pandas DataFrame with query results
        """
        try:
            if not self.engine:
                self.connect()
            
            df = pd.read_sql(text(query), self.engine, params=params or {})
            logger.info(f"Query executed successfully. Rows returned: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def execute_scalar(self, query: str, params: Dict[str, Any] = None) -> Any:
        """
        Execute SQL query and return single value.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Single value from query result
        """
        try:
            if not self.engine:
                self.connect()
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                value = result.scalar()
                
                logger.info(f"Scalar query executed. Value: {value}")
                return value
                
        except Exception as e:
            logger.error(f"Scalar query execution failed: {str(e)}")
            raise
    
    def check_job_load(self, job_name: str, status: str = "SUCCESS") -> int:
        """
        Check job load count in LoadStatus table.
        
        Args:
            job_name: Job name
            status: Expected status
            
        Returns:
            Count of matching records
        """
        query = """
            SELECT COUNT(*) as count 
            FROM dbo.LoadStatus 
            WHERE job_name = :job_name AND status = :status
        """
        
        result = self.execute_scalar(query, {"job_name": job_name, "status": status})
        logger.info(f"Job load count for {job_name}: {result}")
        return result or 0
    
    def get_latest_job_status(self, job_name: str) -> Optional[Dict[str, Any]]:
        """
        Get latest job status from LoadStatus table.
        
        Args:
            job_name: Job name
            
        Returns:
            Dictionary with job status details
        """
        query = """
            SELECT TOP 1 
                job_name, 
                status, 
                load_date,
                record_count
            FROM dbo.LoadStatus 
            WHERE job_name = :job_name 
            ORDER BY load_date DESC
        """
        
        results = self.execute_query(query, {"job_name": job_name})
        
        if results:
            logger.info(f"Latest status for {job_name}: {results[0]}")
            return results[0]
        
        logger.warning(f"No status found for job: {job_name}")
        return None
    
    def verify_table_exists(self, table_name: str, schema: str = "dbo") -> bool:
        """
        Verify if table exists in database.
        
        Args:
            table_name: Table name
            schema: Schema name
            
        Returns:
            True if table exists, False otherwise
        """
        query = """
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = :schema 
            AND TABLE_NAME = :table_name
        """
        
        count = self.execute_scalar(query, {"schema": schema, "table_name": table_name})
        exists = count > 0
        
        logger.info(f"Table {schema}.{table_name} exists: {exists}")
        return exists
    
    def get_table_row_count(self, table_name: str, schema: str = "dbo") -> int:
        """
        Get row count for a table.
        
        Args:
            table_name: Table name
            schema: Schema name
            
        Returns:
            Row count
        """
        query = f"SELECT COUNT(*) FROM {schema}.{table_name}"
        count = self.execute_scalar(query)
        
        logger.info(f"Row count for {schema}.{table_name}: {count}")
        return count or 0
    
    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

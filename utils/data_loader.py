"""Utility functions for loading data from various sources."""
import os
import pandas as pd
import zipfile
import sqlalchemy
import requests
import logging
from typing import Union, Dict, Any, Optional
from pathlib import Path
from urllib.parse import quote_plus

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import configurations
LARGE_FILE_THRESHOLD = 3 * 1024 * 1024 * 1024  # 3GB
CHUNK_SIZE = 10 ** 6  # 1 million rows per chunk
TEMP_DIR = "temp"

class DataLoader:
    @staticmethod
    def read_chunked_file(file_path: str, delimiter: str = ',', **kwargs) -> pd.DataFrame:
        """
        Read large files in chunks to handle files > 3GB.
        
        Args:
            file_path: Path to the file
            delimiter: File delimiter (for CSV/DAT files)
            **kwargs: Additional parameters for pd.read_csv
            
        Returns:
            DataFrame containing the file contents
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            logger.info(f"Starting to read file: {file_path}")
            file_size = os.path.getsize(file_path)
            logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
            
            if file_size > LARGE_FILE_THRESHOLD:
                logger.info("Large file detected, reading in chunks...")
                chunks = []
                chunk_count = 0
                for chunk in pd.read_csv(file_path, delimiter=delimiter, chunksize=CHUNK_SIZE, **kwargs):
                    chunks.append(chunk)
                    chunk_count += 1
                    logger.info(f"Processed chunk {chunk_count}")
                
                result = pd.concat(chunks, ignore_index=True)
                logger.info(f"File reading completed. Total rows: {len(result)}")
                return result
            else:
                result = pd.read_csv(file_path, delimiter=delimiter, **kwargs)
                logger.info(f"File reading completed. Total rows: {len(result)}")
                return result
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

    @staticmethod
    def read_parquet(file_path: str) -> pd.DataFrame:
        """Read a parquet file."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Parquet file not found: {file_path}")
            
            logger.info(f"Reading parquet file: {file_path}")
            result = pd.read_parquet(file_path)
            logger.info(f"Parquet file reading completed. Total rows: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Error reading parquet file {file_path}: {str(e)}")
            raise

    @staticmethod
    def extract_zip(zip_path: str) -> list:
        """
        Extract files from a zip archive.
        
        Args:
            zip_path: Path to the zip file
            
        Returns:
            List of paths to extracted files
        """
        try:
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"ZIP file not found: {zip_path}")
            
            extract_dir = Path(TEMP_DIR) / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Extracting ZIP file: {zip_path}")
            extracted_files = []
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                extracted_files = [str(extract_dir / f) for f in zip_ref.namelist() if not f.endswith('/')]
            
            logger.info(f"ZIP extraction completed. Files extracted: {len(extracted_files)}")
            return extracted_files
            
        except Exception as e:
            logger.error(f"Error extracting ZIP file {zip_path}: {str(e)}")
            raise

    @staticmethod
    def connect_database(connection_params: Dict[str, Any]) -> sqlalchemy.engine.Engine:
        """
        Create database connection using SQLAlchemy with Windows SSO support.
        
        Args:
            connection_params: Dictionary containing connection parameters
                Required keys: 'type' (sql_server/teradata), other connection details
                Optional: 'trusted_connection' for Windows SSO
                
        Returns:
            SQLAlchemy engine object
        """
        try:
            # Validate required parameters
            if not connection_params or 'type' not in connection_params:
                raise ValueError("Connection parameters must include 'type'")
            
            db_type = connection_params['type'].lower()
            server = connection_params.get('server')
            database = connection_params.get('database')
            
            if not server or not database:
                raise ValueError("Server and database are required parameters")
            
            logger.info(f"Connecting to {db_type} database: {server}/{database}")
            
            if db_type == 'sql_server':
                # Check for Windows Authentication (SSO)
                if connection_params.get('trusted_connection') or connection_params.get('use_windows_auth'):
                    # Windows Authentication
                    conn_str = (
                        f"mssql+pyodbc://@{server}/{database}"
                        f"?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
                    )
                    logger.info("Using Windows Authentication (SSO)")
                else:
                    # SQL Server Authentication
                    username = connection_params.get('username')
                    password = connection_params.get('password')
                    
                    if not username or not password:
                        raise ValueError("Username and password are required for SQL Server authentication")
                    
                    # URL encode the password to handle special characters
                    encoded_password = quote_plus(password)
                    conn_str = (
                        f"mssql+pyodbc://{username}:{encoded_password}"
                        f"@{server}/{database}"
                        f"?driver=ODBC+Driver+17+for+SQL+Server"
                    )
                    logger.info("Using SQL Server Authentication")
                    
            elif db_type == 'teradata':
                username = connection_params.get('username')
                password = connection_params.get('password')
                
                if not username or not password:
                    raise ValueError("Username and password are required for Teradata")
                
                encoded_password = quote_plus(password)
                # Use teradata dialect with pyodbc
                conn_str = (
                    f"teradata://{username}:{encoded_password}"
                    f"@{server}/?database={database}"
                )
                logger.info("Using Teradata connection")
            else:
                raise ValueError(f"Unsupported database type: {db_type}")

            # Create engine with connection pooling and timeout settings
            engine = sqlalchemy.create_engine(
                conn_str,
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={"timeout": 30}
            )
            
            # Test the connection
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            
            logger.info("Database connection successful")
            return engine
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise ConnectionError(f"Failed to connect to {connection_params.get('type', 'unknown')}: {str(e)}")

    @staticmethod
    def execute_stored_proc(engine: sqlalchemy.engine.Engine, proc_name: str, 
                          params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a stored procedure and return results as DataFrame.
        
        Args:
            engine: SQLAlchemy engine object
            proc_name: Name of the stored procedure
            params: Dictionary of procedure parameters
            
        Returns:
            DataFrame containing the procedure results
        """
        try:
            if not engine:
                raise ValueError("Database engine is required")
            
            if not proc_name:
                raise ValueError("Stored procedure name is required")
            
            logger.info(f"Executing stored procedure: {proc_name}")
            
            with engine.connect() as conn:
                if params and len(params) > 0:
                    # Safely format parameters
                    param_list = []
                    for k, v in params.items():
                        if isinstance(v, str):
                            param_list.append(f"@{k}='{v}'")
                        else:
                            param_list.append(f"@{k}={v}")
                    param_str = ','.join(param_list)
                    query = f"EXEC {proc_name} {param_str}"
                else:
                    query = f"EXEC {proc_name}"
                
                logger.info(f"Executing query: {query}")
                result = pd.read_sql(sqlalchemy.text(query), conn)
                logger.info(f"Stored procedure executed successfully. Rows returned: {len(result)}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to execute stored procedure {proc_name}: {str(e)}")
            raise RuntimeError(f"Failed to execute stored procedure {proc_name}: {str(e)}")

    @staticmethod
    def call_api(url: str, method: str = 'GET', headers: Optional[Dict[str, str]] = None, 
                 params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Call an API endpoint and convert response to DataFrame.
        
        Args:
            url: API endpoint URL
            method: HTTP method (GET/POST)
            headers: Request headers
            params: Query parameters
            data: Request body for POST
            
        Returns:
            DataFrame containing the API response
        """
        try:
            if not url:
                raise ValueError("API URL is required")
            
            logger.info(f"Making {method} request to: {url}")
            
            # Set default headers
            if headers is None:
                headers = {}
            
            if 'User-Agent' not in headers:
                headers['User-Agent'] = 'DataComparisonTool/1.0'
            
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            logger.info(f"API call successful. Status code: {response.status_code}")
            
            # Try to parse JSON response
            try:
                json_data = response.json()
                if isinstance(json_data, list):
                    result = pd.DataFrame(json_data)
                elif isinstance(json_data, dict):
                    # If it's a single object, wrap it in a list
                    result = pd.DataFrame([json_data])
                else:
                    raise ValueError("API response is not in a format that can be converted to DataFrame")
                
                logger.info(f"API data converted to DataFrame. Rows: {len(result)}")
                return result
                
            except ValueError as json_error:
                logger.error(f"Failed to parse JSON response: {str(json_error)}")
                raise ValueError(f"API response is not valid JSON: {str(json_error)}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise RuntimeError(f"API call failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in API call: {str(e)}")
            raise RuntimeError(f"API call failed: {str(e)}")

    @staticmethod
    def cleanup_temp_files():
        """Remove temporary files and directories."""
        try:
            if os.path.exists(TEMP_DIR):
                logger.info(f"Cleaning up temporary directory: {TEMP_DIR}")
                
                for root, dirs, files in os.walk(TEMP_DIR, topdown=False):
                    for name in files:
                        try:
                            file_path = os.path.join(root, name)
                            os.remove(file_path)
                            logger.debug(f"Removed file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove file {file_path}: {str(e)}")
                    
                    for name in dirs:
                        try:
                            dir_path = os.path.join(root, name)
                            os.rmdir(dir_path)
                            logger.debug(f"Removed directory: {dir_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove directory {dir_path}: {str(e)}")
                
                try:
                    os.rmdir(TEMP_DIR)
                    logger.info("Temporary directory cleanup completed")
                except Exception as e:
                    logger.warning(f"Failed to remove temp directory: {str(e)}")
            else:
                logger.info("No temporary directory to clean up")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

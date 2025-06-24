"""Utility functions for loading data from various sources."""
import pandas as pd
import sqlalchemy
import logging
from typing import Dict, Any
from urllib.parse import quote_plus

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    @staticmethod
    def connect_database(connection_params: Dict[str, Any]) -> sqlalchemy.engine.Engine:
        """
        Create database connection using SQLAlchemy.
        
        Args:
            connection_params: Dictionary containing connection parameters
                Required keys: 'type' (sql_server/teradata), other connection details
                
        Returns:
            SQLAlchemy engine object
        """
        try:
            # Validate required parameters
            if not connection_params or 'type' not in connection_params:
                raise ValueError("Connection parameters must include 'type'")
            
            db_type = connection_params['type'].lower()
            server = connection_params.get('server')
            
            if not server:
                raise ValueError("Server/Hostname is required")
                
            # Only check for database if not Teradata
            if db_type != 'teradata' and not connection_params.get('database'):
                raise ValueError("Database name is required for non-Teradata connections")
            
            database = connection_params.get('database', '')
            logger.info(f"Connecting to {db_type} database: {server}{f'/{database}' if database else ''}")
            
            if db_type == 'sql_server':
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
                server = connection_params.get('server')
                
                if not username or not password or not server:
                    raise ValueError("Username, password, and hostname are required for Teradata")
                
                try:
                    import teradatasql
                    conn = teradatasql.connect(host=server, user=username, password=password)
                    
                    # Create a mock engine object that works with pandas.read_sql
                    class TeradataEngine:
                        def __init__(self, conn):
                            self._conn = conn
                        
                        def connect(self):
                            return self
                            
                        def cursor(self):
                            return self._conn.cursor()
                            
                        def execute(self, sql):
                            cursor = self._conn.cursor()
                            cursor.execute(str(sql))
                            columns = [desc[0] for desc in cursor.description]
                            data = cursor.fetchall()
                            return pd.DataFrame(data, columns=columns)
                    
                    engine = TeradataEngine(conn)
                    logger.info("Teradata connection successful")
                    return engine
                    
                except ImportError:
                    raise ImportError("teradatasql package is required for Teradata connections")
                except Exception as e:
                    raise ConnectionError(f"Failed to connect to Teradata: {str(e)}")
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

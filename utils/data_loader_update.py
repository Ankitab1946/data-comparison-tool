import os
import pandas as pd
import sqlalchemy
import logging
from urllib.parse import quote_plus
from typing import Dict, Any
from sqlalchemy.engine import Engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    @staticmethod
    def connect_database(connection_params: Dict[str, Any]) -> Engine:
        try:
            if not connection_params or 'type' not in connection_params:
                raise ValueError("Connection parameters must include 'type'")

            db_type = connection_params['type'].lower()
            logger.info(f"Connecting to {db_type} database...")

            if db_type == 'snowflake':
                # Required Snowflake parameters
                user = connection_params.get('user')
                password = connection_params.get('password')
                account = connection_params.get('account')
                warehouse = connection_params.get('warehouse')
                database = connection_params.get('database')
                schema = connection_params.get('schema')
                role = connection_params.get('role', None)

                if not all([user, password, account, warehouse, database, schema]):
                    raise ValueError("Missing Snowflake connection parameters")

                from snowflake.sqlalchemy import URL
                conn_url = URL(
                    user=user,
                    password=password,
                    account=account,
                    warehouse=warehouse,
                    database=database,
                    schema=schema,
                    role=role
                )

                engine = sqlalchemy.create_engine(conn_url)

            elif db_type == 'sql_server':
                server = connection_params.get('server')
                database = connection_params.get('database')
                use_windows_auth = connection_params.get('trusted_connection', False)

                if use_windows_auth:
                    conn_str = (
                        f"mssql+pyodbc://@{server}/{database}"
                        f"?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
                    )
                else:
                    username = connection_params.get('username')
                    password = quote_plus(connection_params.get('password'))
                    conn_str = (
                        f"mssql+pyodbc://{username}:{password}@{server}/{database}"
                        f"?driver=ODBC+Driver+17+for+SQL+Server"
                    )
                engine = sqlalchemy.create_engine(conn_str)

            else:
                raise ValueError(f"Unsupported database type: {db_type}")

            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))

            logger.info(f"Connected to {db_type} successfully.")
            return engine

        except Exception as e:
            logger.error(f"Failed to connect to {db_type}: {e}")
            raise

    @staticmethod
    def load_data(engine: Engine, sql_query: str) -> pd.DataFrame:
        try:
            logger.info("Loading data using SQL query...")
            with engine.connect() as conn:
                df = pd.read_sql(sql_query, conn)
            logger.info(f"Data loaded. Rows: {len(df)} Columns: {len(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

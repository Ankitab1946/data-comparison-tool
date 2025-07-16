import os
import pandas as pd
import snowflake.connector
import logging
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from urllib.parse import quote_plus
from typing import Dict, Any
from sqlalchemy.engine import Engine
import sqlalchemy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    @staticmethod
    def connect_database(connection_params: Dict[str, Any]):
        try:
            if not connection_params or 'type' not in connection_params:
                raise ValueError("Connection parameters must include 'type'")

            db_type = connection_params['type'].lower()
            logger.info(f"Connecting to {db_type} database...")

            if db_type == 'snowflake':
                account = connection_params.get('account')
                warehouse = connection_params.get('warehouse')
                database = connection_params.get('database')
                schema = connection_params.get('schema')
                user = connection_params.get('user')
                private_key_file = connection_params.get('private_key_file')
                private_key_passphrase = connection_params.get('private_key_passphrase', None)

                if not all([user, account, warehouse, database, schema, private_key_file]):
                    raise ValueError("Missing Snowflake connection parameters for private key authentication.")

                with open(private_key_file, 'rb') as key_file:
                    p_key = serialization.load_pem_private_key(
                        key_file.read(),
                        password=private_key_passphrase.encode() if private_key_passphrase else None,
                        backend=default_backend()
                    )
                    private_key = p_key.private_bytes(
                        encoding=serialization.Encoding.DER,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    )

                conn = snowflake.connector.connect(
                    user=user,
                    account=account,
                    private_key=private_key,
                    warehouse=warehouse,
                    database=database,
                    schema=schema
                )

                logger.info("Connected to Snowflake using private key authentication.")
                return conn

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

                with engine.connect() as conn:
                    conn.execute(sqlalchemy.text("SELECT 1"))

                logger.info(f"Connected to {db_type} successfully.")
                return engine

            else:
                raise ValueError(f"Unsupported database type: {db_type}")

        except Exception as e:
            logger.error(f"Failed to connect to {db_type}: {e}")
            raise

    @staticmethod
    def load_data(connection, sql_query: str, db_type: str) -> pd.DataFrame:
        try:
            logger.info("Loading data using SQL query...")
            if db_type == 'snowflake':
                df = pd.read_sql(sql_query, connection)
            else:
                with connection.connect() as conn:
                    df = pd.read_sql(sql_query, conn)
            logger.info(f"Data loaded. Rows: {len(df)} Columns: {len(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

import streamlit as st
import pandas as pd
from utils.data_loader import DataLoader

def get_connection_inputs(source_type: str, prefix: str) -> dict:
    params = {}

    if source_type == 'Snowflake':
        st.subheader(f"Snowflake {prefix.title()} Connection (Private Key)")

        params['type'] = 'snowflake'
        params['account'] = st.text_input("Account", key=f"{prefix}_account")
        params['user'] = st.text_input("User", key=f"{prefix}_user")
        params['warehouse'] = st.text_input("Warehouse", key=f"{prefix}_warehouse")
        params['database'] = st.text_input("Database", key=f"{prefix}_database")
        params['schema'] = st.text_input("Schema", key=f"{prefix}_schema")

        key_file = st.file_uploader("Upload Private Key (.p8)", type="p8", key=f"{prefix}_keyfile")
        if key_file:
            temp_key_path = f"/tmp/{prefix}_private_key.p8"
            with open(temp_key_path, "wb") as f:
                f.write(key_file.getvalue())
            params['private_key_file'] = temp_key_path

        params['private_key_passphrase'] = st.text_input(
            "Private Key Passphrase (Optional)",
            type="password",
            key=f"{prefix}_keypass"
        )

        params['query'] = st.text_area("SQL Query", key=f"{prefix}_query")

    elif source_type == 'SQL Server':
        st.subheader(f"SQL Server {prefix.title()} Connection")

        params['type'] = 'sql_server'
        params['server'] = st.text_input(f"{prefix.title()} Server", key=f"{prefix}_server")
        params['database'] = st.text_input("Database Name", key=f"{prefix}_database")
        params['trusted_connection'] = st.checkbox(
            "Use Windows Authentication",
            value=False,
            key=f"{prefix}_trusted_connection"
        )

        if not params['trusted_connection']:
            params['username'] = st.text_input("Username", key=f"{prefix}_username")
            params['password'] = st.text_input("Password", type="password", key=f"{prefix}_password")

        params['query'] = st.text_area("SQL Query", key=f"{prefix}_query")

    return params

@staticmethod
    def load_data(connection: Union[snowflake.connector.SnowflakeConnection, Engine], sql_query: str, db_type: str) -> pd.DataFrame:
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

def main():
    st.title("Database Data Viewer")

    source_type = st.selectbox("Select Source Type", ['Snowflake', 'SQL Server', 'Teradata'], key='source_type')

    st.header("Source Connection Setup")
    source_params = get_connection_inputs(source_type, "source")

    if st.button("Load Source Data"):
        try:
            conn = DataLoader.connect_database(source_params)
            st.success(f"✅ Connected to {source_type} Database")
            sql_query = source_params.get('query')
            if sql_query and sql_query.strip():
                df = DataLoader.load_data(conn, sql_query, source_params['type'].lower())
                st.success(f"{source_type} Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                st.dataframe(df.head(100))

        except Exception as e:
            st.error(f"❌ Failed to load source data: {e}")

if __name__ == "__main__":
    main()

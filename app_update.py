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

def main():
    st.title("Database Comparison Tool")

    source_type = st.selectbox("Select Source Type", ['Snowflake', 'SQL Server'], key='source_type')
    target_type = st.selectbox("Select Target Type", ['Snowflake', 'SQL Server'], key='target_type')

    st.header("Source Connection Setup")
    source_params = get_connection_inputs(source_type, "source")

    st.header("Target Connection Setup")
    target_params = get_connection_inputs(target_type, "target")

    if st.button("Test Source Connection"):
        try:
            conn = DataLoader.connect_database(source_params)
            st.success("✅ Connected to Source Database")
        except Exception as e:
            st.error(f"❌ Failed to connect to Source: {e}")

    if st.button("Test Target Connection"):
        try:
            conn = DataLoader.connect_database(target_params)
            st.success("✅ Connected to Target Database")
        except Exception as e:
            st.error(f"❌ Failed to connect to Target: {e}")

if __name__ == "__main__":
    main()

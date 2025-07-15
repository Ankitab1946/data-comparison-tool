import streamlit as st
from utils.data_loader import DataLoader

def get_connection_inputs(source_type: str, prefix: str) -> dict:
    params = {}

    if source_type == 'Snowflake':
        st.subheader(f"Snowflake {prefix.title()} Connection")
        params['type'] = 'snowflake'
        params['account'] = st.text_input("Account", key=f"{prefix}_account")
        params['warehouse'] = st.text_input("Warehouse", key=f"{prefix}_warehouse")
        params['database'] = st.text_input("Database", key=f"{prefix}_database")
        params['schema'] = st.text_input("Schema", key=f"{prefix}_schema")
        params['role'] = st.text_input("Role (Optional)", key=f"{prefix}_role")

        params['use_externalbrowser'] = st.checkbox(
            "Use Azure AD (External Browser Authentication)?",
            value=False,
            key=f"{prefix}_use_externalbrowser",
            help="Enable this to authenticate using Azure AD via browser popup."
        )

        if not params['use_externalbrowser']:
            params['user'] = st.text_input("User", key=f"{prefix}_user")
            params['password'] = st.text_input("Password", type="password", key=f"{prefix}_password")

        params['query'] = st.text_area("SQL Query", key=f"{prefix}_query")

    elif source_type == 'SQL Server':
        params['type'] = 'sql_server'
        params['server'] = st.text_input(f"{prefix.title()} Server", key=f"{prefix}_server")
        params['database'] = st.text_input("Database Name", key=f"{prefix}_database")

        params['trusted_connection'] = st.checkbox(
            "Use Windows Authentication",
            value=False,
            key=f"{prefix}_trusted_connection",
            help="Enable this for Windows SSO."
        )

        if not params['trusted_connection']:
            params['username'] = st.text_input("Username", key=f"{prefix}_username")
            params['password'] = st.text_input("Password", type="password", key=f"{prefix}_password")

        params['query'] = st.text_area("SQL Query", key=f"{prefix}_query")

    return params

def main():
    st.title("Data Comparison Tool")

    source_type = st.selectbox("Select Source Type", ['Snowflake', 'SQL Server'], key='source_type')
    target_type = st.selectbox("Select Target Type", ['Snowflake', 'SQL Server'], key='target_type')

    st.subheader("Source Connection")
    source_params = get_connection_inputs(source_type, "source")

    st.subheader("Target Connection")
    target_params = get_connection_inputs(target_type, "target")

    if st.button("Test Source Connection"):
        try:
            engine = DataLoader.connect_database(source_params)
            st.success("Source database connection successful!")
        except Exception as e:
            st.error(f"Failed to connect to source: {e}")

    if st.button("Test Target Connection"):
        try:
            engine = DataLoader.connect_database(target_params)
            st.success("Target database connection successful!")
        except Exception as e:
            st.error(f"Failed to connect to target: {e}")

if __name__ == "__main__":
    main()

import streamlit as st

def get_connection_inputs(source_type: str, prefix: str) -> dict:
    params = {}

    if source_type == 'Snowflake':
        st.subheader(f"Snowflake {prefix.title()} Connection")
        params['type'] = 'snowflake'
        params['account'] = st.text_input("Account", key=f"{prefix}_account")
        params['user'] = st.text_input("User", key=f"{prefix}_user")
        params['password'] = st.text_input("Password", type="password", key=f"{prefix}_password")
        params['warehouse'] = st.text_input("Warehouse", key=f"{prefix}_warehouse")
        params['database'] = st.text_input("Database", key=f"{prefix}_database")
        params['schema'] = st.text_input("Schema", key=f"{prefix}_schema")
        params['role'] = st.text_input("Role (Optional)", key=f"{prefix}_role")
        params['query'] = st.text_area("SQL Query", key=f"{prefix}_query")

    elif source_type == 'SQL Server':
        # Existing logic
        pass

    # Add other data sources as needed

    return params

# Update the SUPPORTED_SOURCES list
SUPPORTED_SOURCES = [
    'CSV file',
    'SQL Server',
    'Snowflake',
    'API',
    'Parquet file',
    'Flat files inside zipped folder'
]

# In your main UI logic, invoke get_connection_inputs with 'Snowflake' option where appropriate.

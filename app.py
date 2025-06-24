"""Main Streamlit application for the Data Comparison Tool."""
import streamlit as st
import pandas as pd
from typing import Dict, Any

from utils.data_loader import DataLoader
from utils.config import SUPPORTED_SOURCES, TYPE_MAPPING

def get_connection_inputs(source_type: str, prefix: str) -> Dict[str, Any]:
    """Get connection parameters based on source type."""
    params = {}
    
    if source_type in ['SQL Server', 'Teradata']:
        # Convert source type to the format expected by DataLoader
        db_type = source_type.lower().replace(' ', '_')
        
        col1, col2 = st.columns(2)
        with col1:
            params['server'] = st.text_input(
                "Hostname" if source_type == "Teradata" else f"{source_type} Server",
                key=f"{prefix}_server"
            )
            # Only show database field for non-Teradata connections
            if source_type != "Teradata":
                params['database'] = st.text_input(
                    "Database Name",
                    key=f"{prefix}_database"
                )
        with col2:
            params['username'] = st.text_input(
                "Username",
                key=f"{prefix}_username"
            )
            params['password'] = st.text_input(
                "Password",
                type="password",
                key=f"{prefix}_password"
            )
        
        params['type'] = db_type
        
        # Option to use query or table
        query_type = st.radio(
            "Select Input Type",
            ["Table", "Query"],
            key=f"{prefix}_query_type"
        )
        if query_type == "Table":
            params['table'] = st.text_input(
                "Table Name",
                key=f"{prefix}_table"
            )
        else:
            params['query'] = st.text_area(
                "SQL Query",
                key=f"{prefix}_query"
            )
            
    return params

def main():
    # Page configuration
    st.set_page_config(
        page_title="Data Comparison Tool",
        page_icon="📊",
        layout="wide"
    )

    # Header
    st.markdown("""
        <div style="padding: 2rem; background: linear-gradient(90deg, #1E3D59 0%, #1E3D59 100%); color: white; border-radius: 0.5rem; margin-bottom: 2rem; text-align: center;">
            <h1>Data Comparison Tool</h1>
            <p>Compare data across SQL Server and Teradata databases</p>
        </div>
    """, unsafe_allow_html=True)

    # Source Selection
    st.subheader("Select Source and Target")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Source")
        source_type = st.selectbox("Select Source Type", SUPPORTED_SOURCES, key="source_type")
        source_params = get_connection_inputs(source_type, "source")
            
    with col2:
        st.markdown("### Target")
        target_type = st.selectbox("Select Target Type", SUPPORTED_SOURCES, key="target_type")
        target_params = get_connection_inputs(target_type, "target")

    # Load Data button
    if st.button("Test Connections"):
        try:
            # Validate source input
            if not source_params or not source_params.get('server'):
                st.error("Please provide hostname for source")
                return
            if source_type != 'Teradata' and not source_params.get('database'):
                st.error("Please provide database name for source")
                return

            # Validate target input
            if not target_params or not target_params.get('server'):
                st.error("Please provide hostname for target")
                return
            if target_type != 'Teradata' and not target_params.get('database'):
                st.error("Please provide database name for target")
                return

            # Test source connection
            with st.spinner("Testing source connection..."):
                try:
                    loader = DataLoader()
                    source_engine = loader.connect_database(source_params)
                    st.success("Source connection successful!")
                except Exception as e:
                    st.error(f"Error connecting to source: {str(e)}")
                    return
                    
            # Test target connection
            with st.spinner("Testing target connection..."):
                try:
                    target_engine = loader.connect_database(target_params)
                    st.success("Target connection successful!")
                except Exception as e:
                    st.error(f"Error connecting to target: {str(e)}")
                    return
                    
            st.success("Both connections tested successfully!")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return

if __name__ == "__main__":
    main()

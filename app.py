
"""Main Streamlit application for the Comparison Tool."""
import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import os
from typing import Dict, Any, Tuple
from datetime import datetime

import io
from utils.data_loader import DataLoader
from utils.comparison_engine import ComparisonEngine
from reports.report_generator import ReportGenerator

def update_column_mapping(edited_mapping, engine):
    """Helper function to safely update column mapping in session state."""
    if edited_mapping is not None:
        try:
            # Convert to DataFrame if not already
            mapping_df = edited_mapping if isinstance(edited_mapping, pd.DataFrame) else pd.DataFrame(edited_mapping)
            
            # Ensure join column is boolean
            mapping_df['join'] = mapping_df['join'].fillna(False).astype(bool)
            
            # Extract join columns
            join_columns = list(mapping_df[mapping_df['join']]['source'])
            
            # Convert to records for processing
            mapping_records = []
            for _, row in mapping_df.iterrows():
                record = row.to_dict()
                
                # Handle type conversions
                if record['source_type'] == 'object':
                    record['source_type'] = 'string'
                if record['target_type'] == 'object':
                    record['target_type'] = 'string'
                
                # Ensure float types are consistent
                if any('float' in str(t).lower() for t in [record['source_type'], record['target_type']]):
                    record['data_type'] = 'float64'
                    record['source_type'] = 'float64'
                    record['target_type'] = 'float64'
                
                # Ensure all required fields exist
                if 'data_type' not in record or not record['data_type']:
                    record['data_type'] = record['source_type']
                
                # Update engine's type mapping
                engine.update_column_types(
                    record['source'],
                    source_type=record['source_type'],
                    target_type=record['target_type']
                )
                
                mapping_records.append(record)
            
            # Set mapping in engine with preserved join columns
            engine.set_mapping(mapping_records, join_columns)
            
            # Store updated mapping in session state
            st.session_state.column_mapping = mapping_records
            
            # Log join columns for debugging
            st.session_state['debug_join_columns'] = join_columns
            logger.info(f"Number of join columns after update: {len(join_columns)}")
            return True
            
        except Exception as e:
            st.error(f"Error updating column mapping: {str(e)}")
            st.session_state.column_mapping = engine.auto_map_columns()
            return False
    else:
        st.warning("No changes made to column mapping")
        if 'column_mapping' not in st.session_state:
            st.session_state.column_mapping = engine.auto_map_columns()
        return False

def create_mapping_editor(mapping_data, target_columns, key_suffix=""):
    """Create a data editor for column mapping with error handling."""
    try:
        # Convert join column to boolean and preserve status
        if 'join' in mapping_data.columns:
            mapping_data['join'] = mapping_data['join'].fillna(False).astype(bool)
        else:
            mapping_data['join'] = False
            
        # Store original join status
        join_status = mapping_data['join'].copy()
        
        # Automatically detect and normalize data types
        for row in mapping_data.itertuples():
            # Handle source type
            if pd.isna(row.source_type) or row.source_type == '':
                mapping_data.at[row.Index, 'source_type'] = TYPE_MAPPING.get(str(row.source_type).lower(), 'string')
            elif 'float' in str(row.source_type).lower():
                mapping_data.at[row.Index, 'source_type'] = 'float64'
            
            # Handle target type
            if pd.isna(row.target_type) or row.target_type == '':
                mapping_data.at[row.Index, 'target_type'] = TYPE_MAPPING.get(str(row.target_type).lower(), 'string')
            elif 'float' in str(row.target_type).lower():
                mapping_data.at[row.Index, 'target_type'] = 'float64'
            
            # Handle comparison type (data_type)
            if pd.isna(row.data_type) or row.data_type == '':
                # If either source or target is float, use float64
                if ('float' in str(row.source_type).lower() or 
                    'float' in str(row.target_type).lower() or 
                    'decimal' in str(row.source_type).lower() or 
                    'decimal' in str(row.target_type).lower()):
                    mapping_data.at[row.Index, 'data_type'] = 'float64'
                else:
                    mapping_data.at[row.Index, 'data_type'] = TYPE_MAPPING.get(str(row.source_type).lower(), 'string')
        
        # Restore join status
        mapping_data['join'] = join_status

        # Create data editor with enhanced configuration
        edited_mapping = st.data_editor(
            mapping_data,
            column_config={
                "source": st.column_config.TextColumn(
                    "Source Column",
                    disabled=True,
                    help="Original column name from source data"
                ),
                "target": st.column_config.SelectboxColumn(
                    "Target Column",
                    options=[""] + list(target_columns),
                    help="Select matching column from target data"
                ),
                "join": st.column_config.CheckboxColumn(
                    "Join Column",
                    default=False,
                    help="⚠️ Select columns to match records between source and target (at least one required)"
                ),
                "source_type": st.column_config.SelectboxColumn(
                    "Source Type",
                    options=list(TYPE_MAPPING.values()),
                    help="Data type for source column"
                ),
                "target_type": st.column_config.SelectboxColumn(
                    "Target Type",
                    options=list(TYPE_MAPPING.values()),
                    help="Data type for target column"
                ),
                "data_type": st.column_config.SelectboxColumn(
                    "Comparison Type",
                    options=list(TYPE_MAPPING.values()),
                    help="Data type for comparison"
                ),
                "exclude": st.column_config.CheckboxColumn(
                    "Exclude",
                    help="Exclude this column from comparison"
                )
            },
            hide_index=True,
            key=f"mapping_editor_{key_suffix}"
        )
        return edited_mapping
    except Exception as e:
        st.error(f"Error creating mapping editor: {str(e)}")
        return None

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
            # Add Windows Authentication option
            use_windows_auth = st.checkbox(
                "Use Windows Authentication",
                key=f"{prefix}_windows_auth"
            )
        with col2:
            if not use_windows_auth:
                params['username'] = st.text_input(
                    "Username",
                    key=f"{prefix}_username"
                )
                params['password'] = st.text_input(
                    "Password",
                    type="password",
                    key=f"{prefix}_password"
                )
            else:
                params['trusted_connection'] = True
        
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
            
    elif source_type == 'Stored Procs':
        col1, col2 = st.columns(2)
        with col1:
            params['server'] = st.text_input("Database Server")
            params['database'] = st.text_input("Database Name")
        with col2:
            params['username'] = st.text_input("Username")
            params['password'] = st.text_input("Password", type="password")
        
        params['type'] = 'sql_server'  # Default to SQL Server for stored procs
        params['proc_name'] = st.text_input("Stored Procedure Name")
        
        # Optional procedure parameters
        if st.checkbox("Add Procedure Parameters"):
            param_count = st.number_input("Number of Parameters", min_value=1, value=1)
            params['params'] = {}
            for i in range(param_count):
                col1, col2 = st.columns(2)
                with col1:
                    param_name = st.text_input(f"Parameter {i+1} Name")
                with col2:
                    param_value = st.text_input(f"Parameter {i+1} Value")
                if param_name:
                    params['params'][param_name] = param_value
                    
    elif source_type == 'API':
        params['url'] = st.text_input("API URL")
        params['method'] = st.selectbox("HTTP Method", ["GET", "POST"])
        
        if st.checkbox("Add Headers"):
            params['headers'] = {}
            header_count = st.number_input("Number of Headers", min_value=1, value=1)
            for i in range(header_count):
                col1, col2 = st.columns(2)
                with col1:
                    header_name = st.text_input(f"Header {i+1} Name")
                with col2:
                    header_value = st.text_input(f"Header {i+1} Value")
                if header_name:
                    params['headers'][header_name] = header_value
        
        if params['method'] == "POST":
            params['data'] = st.text_area("Request Body (JSON)")
            
    return params

# Constants
SUPPORTED_SOURCES = [
    'CSV file',
    'DAT file',
    'SQL Server',
    'Stored Procs',
    'Teradata',
    'API',
    'Parquet file',
    'Flat files inside zipped folder'
]

TYPE_MAPPING = {
    'int': 'int32',
    'int64': 'int64',
    'numeric': 'int64',
    'bigint': 'int64',
    'smallint': 'int64',
    'varchar': 'string',
    'nvarchar': 'string',
    'char': 'string',
    'date': 'datetime64[ns]',
    'datetime': 'datetime64[ns]',
    'decimal': 'float64',
    'float': 'float64',
    'float64': 'float64',
    'double': 'float64',
    'bit': 'bool',
    'nchar': 'char',
    'boolean': 'bool'
}

MAX_PREVIEW_ROWS = 1000

# Page configuration
st.set_page_config(
    page_title="Data Comparison Tool",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
    }
    .report-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .header-banner {
        padding: 2rem;
        background: linear-gradient(90deg, #1E3D59 0%, #1E3D59 100%);
        color: white;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'report_paths' not in st.session_state:
    st.session_state.report_paths = None

def load_data(source_type: str, file_upload, connection_params: Dict[str, Any] = None, 
              delimiter: str = ',') -> pd.DataFrame:
    """Load data based on the selected source type."""
    loader = DataLoader()
    
    if source_type in ['CSV file', 'DAT file']:
        if file_upload is None:
            raise ValueError(f"Please upload a {source_type}")
        
        try:
            # Validate file extension
            if not file_upload.name.lower().endswith(('.csv', '.dat')):
                raise ValueError(f"Invalid file type. Please upload a {source_type}")
            
            # Debug info
            st.write(f"Debug: File name: {file_upload.name}")
            st.write(f"Debug: File size: {file_upload.size} bytes")
            
            # Read file content into BytesIO
            import io
            file_upload.seek(0)
            file_bytes = file_upload.read()
            file_buffer = io.BytesIO(file_bytes)
            
            # Debug preview of content
            preview_text = file_bytes[:200].decode('utf-8', errors='ignore')
            st.write("Debug: File content preview:")
            st.code(preview_text)
            
            # Reset buffer pointer
            file_buffer.seek(0)
            
            try:
                # Try different encodings
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        file_buffer.seek(0)  # Reset buffer position
                        df = pd.read_csv(
                            file_buffer,
                            delimiter=delimiter,
                            encoding=encoding,
                            on_bad_lines='skip'
                        )
                        st.write(f"Debug: Successfully read file with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        st.write(f"Debug: Error with {encoding} encoding: {str(e)}")
                        continue
                else:
                    raise ValueError("Could not read file with any supported encoding")
                
                if df.empty:
                    raise ValueError("No data found in the file")
                
                # Show successful read info
                st.write(f"Debug: Successfully read CSV. Shape: {df.shape}")
                st.write(f"Debug: Columns: {list(df.columns)}")
                return df
                
            except Exception as e:
                st.write(f"Debug: Error reading CSV: {str(e)}")
                raise ValueError(f"Error reading CSV file: {str(e)}")
                
        except Exception as e:
            st.write(f"Debug: Error details: {str(e)}")
            raise ValueError(f"Error reading {source_type}: {str(e)}")
            
    elif source_type == 'Parquet file':
        if file_upload is None:
            raise ValueError("Please upload a Parquet file")
            
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp_file:
                tmp_file.write(file_upload.getvalue())
                df = loader.read_parquet(tmp_file.name)
                if df.empty:
                    raise ValueError("No data found in the Parquet file")
                return df
        except Exception as e:
            raise ValueError(f"Error reading Parquet file: {str(e)}")
            
    elif source_type == 'Flat files inside zipped folder':
        if file_upload is None:
            raise ValueError("Please upload a ZIP file")
            
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_file.write(file_upload.getvalue())
                extracted_files = loader.extract_zip(tmp_file.name)
                
                if not extracted_files:
                    raise ValueError("No files found in ZIP archive")
                
                # Combine all extracted files
                dfs = []
                for file_path in extracted_files:
                    if file_path.endswith(('.csv', '.dat')):
                        try:
                            # Try reading with pandas directly first
                            df = pd.read_csv(file_path, delimiter=delimiter)
                            if not df.empty:
                                dfs.append(df)
                        except Exception:
                            # If direct reading fails, try chunked reading
                            df = loader.read_chunked_file(file_path, delimiter=delimiter)
                            if not df.empty:
                                dfs.append(df)
                
                if not dfs:
                    raise ValueError("No valid data files found in ZIP archive")
                    
                combined_df = pd.concat(dfs, ignore_index=True)
                if combined_df.empty:
                    raise ValueError("No data found in any of the files")
                    
                return combined_df
        except Exception as e:
            raise ValueError(f"Error processing ZIP file: {str(e)}")
            
    elif source_type in ['SQL Server', 'Teradata']:
        if not connection_params:
            raise ValueError(f"Please provide connection parameters for {source_type}")
            
        try:
            engine = loader.connect_database(connection_params)
            
            try:
                if 'query' in connection_params and connection_params['query'].strip():
                    df = pd.read_sql(connection_params['query'], engine)
                elif 'table' in connection_params and connection_params['table'].strip():
                    df = pd.read_sql(connection_params['table'], engine)
                else:
                    raise ValueError("Please provide either a query or table name")
                
                if df.empty:
                    raise ValueError("No data returned from the database")
                return df
                
            except Exception as e:
                raise ValueError(f"Error executing query: {str(e)}")
                
        except Exception as e:
            raise ValueError(f"Database connection error: {str(e)}")
            
    elif source_type == 'Stored Procs':
        if not connection_params:
            raise ValueError("Please provide stored procedure details")
            
        if not connection_params.get('proc_name'):
            raise ValueError("Stored procedure name is required")
            
        try:
            engine = loader.connect_database(connection_params)
            df = loader.execute_stored_proc(
                engine,
                connection_params['proc_name'],
                connection_params.get('params')
            )
            
            if df is None or df.empty:
                raise ValueError("No data returned from stored procedure")
            return df
            
        except Exception as e:
            raise ValueError(f"Error executing stored procedure: {str(e)}")
        
    elif source_type == 'API':
        if not connection_params:
            raise ValueError("Please provide API details")
            
        if not connection_params.get('url'):
            raise ValueError("API URL is required")
            
        try:
            df = loader.call_api(
                connection_params['url'],
                method=connection_params.get('method', 'GET'),
                headers=connection_params.get('headers'),
                params=connection_params.get('params'),
                data=connection_params.get('data')
            )
            
            if df is None or df.empty:
                raise ValueError("No data returned from API")
            return df
            
        except Exception as e:
            raise ValueError(f"API call failed: {str(e)}")
        
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

def main():
    # Header
    st.markdown("""
        <div class="header-banner">
            <h1>Data Comparison Tool</h1>
            <p>Compare data across multiple sources with detailed analysis and reporting</p>
        </div>
    """, unsafe_allow_html=True)

    # Source Selection
    st.subheader("1. Select Source and Target")
    
    # Add sample data option with clear UI
    st.markdown("""
        <style>
        .sample-data-section {
            padding: 1rem;
            margin-bottom: 2rem;
            border-radius: 0.5rem;
            background-color: #f8f9fa;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sample data section
    with st.container():
        st.markdown('<div class="sample-data-section">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            use_sample = st.checkbox("📊 Use Sample Data for Testing", 
                                   value=False,
                                   help="Load pre-configured sample data to test the comparison functionality")
            
            # Track mode switch for data clearing
            current_mode = "sample" if use_sample else "manual"
            previous_mode = "sample" if st.session_state.get('previous_use_sample', False) else "manual"
            
            # Clear previous data when switching between sample and manual
            if current_mode != previous_mode:
                with st.spinner(f"🔄 Switching to {current_mode} data mode..."):
                    import time
                    
                    # Clear all relevant session state
                    keys_to_clear = [
                        'column_mapping',
                        'comparison_results',
                        'source_data',
                        'target_data',
                        'report_paths',
                        'zip_path',
                        'filtered_source',
                        'filtered_target'
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Update mode tracking
                    st.session_state.previous_use_sample = use_sample
                    
                    # Show transition message
                    st.success(f"✨ Switched to {current_mode} data mode. Previous data cleared.")
                    time.sleep(0.5)  # Short delay for smooth transition
                    st.rerun()  # Ensure clean UI state
                
        with col2:
            if st.button("Load Sample", disabled=not use_sample):
                try:
                    # Get the current directory
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    
                    # Construct paths to sample data files
                    source_path = os.path.join(current_dir, "sample_data", "source.csv")
                    target_path = os.path.join(current_dir, "sample_data", "target.csv")
                    
                    # Load the data
                    source_data = pd.read_csv(source_path)
                    target_data = pd.read_csv(target_path)
                    
                    # Store in session state
                    st.session_state.source_data = source_data
                    st.session_state.target_data = target_data
                    
                    # Initialize comparison engine
                    engine = ComparisonEngine(source_data, target_data)
                    
                    # Initialize mapping if not already present
                    if 'column_mapping' not in st.session_state:
                        st.session_state.column_mapping = engine.auto_map_columns()
                    
                    st.success("✅ Sample data loaded successfully!")
                    
                    # Show initial mapping status
                    st.info("🔄 Auto-mapping columns...")
                    mapped_cols = sum(1 for m in st.session_state.column_mapping if m['target'])
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Auto-mapped Columns", f"{mapped_cols}/{len(st.session_state.column_mapping)}")
                    with col2:
                        join_cols = sum(1 for m in st.session_state.column_mapping if m.get('join', False))
                        st.metric("Join Columns", join_cols)
                    with col3:
                        excluded_cols = sum(1 for m in st.session_state.column_mapping if m.get('exclude', False))
                        st.metric("Excluded Columns", excluded_cols)
                    
                    # Show preview of sample data
                    st.markdown("### Data Preview")
                    preview_col1, preview_col2 = st.columns(2)
                    with preview_col1:
                        st.markdown("### Source Data")
                        st.dataframe(source_data.head(MAX_PREVIEW_ROWS), use_container_width=True)
                    with preview_col2:
                        st.markdown("### Target Data")
                        st.dataframe(target_data.head(MAX_PREVIEW_ROWS), use_container_width=True)
                    
                    # Show next steps guidance
                    st.info("✨ Proceed to Column Mapping section below to configure comparison settings.")
                    
                    # Show mapping section
                    st.subheader("2. Column Mapping")
                    st.markdown("Configure how columns should be compared between source and target data.")
                    
                    # Create mapping editor with enhanced UI
                    mapping_data = pd.DataFrame(st.session_state.column_mapping)
                    edited_mapping = create_mapping_editor(mapping_data, target_data.columns, "sample")
                    
                    # Update session state with edited mapping
                    update_column_mapping(edited_mapping, engine)
                    
                    # Show mapping summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        mapped_cols = sum(1 for m in st.session_state.column_mapping if m['target'])
                        st.metric("Mapped Columns", f"{mapped_cols}/{len(st.session_state.column_mapping)}")
                    with col2:
                        join_cols = sum(1 for m in st.session_state.column_mapping if m['join'])
                        st.metric("Join Columns", join_cols)
                    with col3:
                        excluded_cols = sum(1 for m in st.session_state.column_mapping if m['exclude'])
                        st.metric("Excluded Columns", excluded_cols)
                    
                    # Convert mapping to DataFrame for validation
                    mapping_df = pd.DataFrame(st.session_state.column_mapping)
                    
                    # Ensure join column is boolean
                    mapping_df['join'] = mapping_df['join'].fillna(False).astype(bool)
                    
                    # Get selected join columns
                    join_columns = list(mapping_df[mapping_df['join']]['source'])
                    
                    # Store join columns in session state for debugging
                    st.session_state['current_join_columns'] = join_columns
                    
                    # Show join column status
                    if not join_columns:
                        st.warning("⚠️ Please select at least one join column for comparison", icon="⚠️")
                        st.markdown("""
                            <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>
                                <p><strong>Why do I need join columns?</strong></p>
                                <p>Join columns are used to match records between source and target data. 
                                Select columns that uniquely identify records (e.g., ID fields, primary keys).</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success(f"✅ Selected join columns: {', '.join(join_columns)}")
                        # Debug information
                        st.info(f"Debug: Found {len(join_columns)} join column(s): {join_columns}")
                        # Compare button
                        if st.button("Compare Data", key="sample_compare"):
                            try:
                                # Validate join columns
                                if not join_columns:
                                    st.error("⚠️ No join columns selected. Please select at least one join column.")
                                    return
                                
                                # Log current join column status
                                st.info(f"Starting comparison with join columns: {', '.join(join_columns)}")
                                
                                # Verify join columns exist in both datasets
                                missing_in_source = [col for col in join_columns if col not in mapping_df['source'].values]
                                missing_in_target = [col for col in join_columns if col not in st.session_state.target_data.columns]
                                
                                if missing_in_source or missing_in_target:
                                    error_msg = []
                                    if missing_in_source:
                                        error_msg.append(f"Columns missing in source: {', '.join(missing_in_source)}")
                                    if missing_in_target:
                                        error_msg.append(f"Columns missing in target: {', '.join(missing_in_target)}")
                                    st.error("⚠️ " + " | ".join(error_msg))
                                    return
                                
                                # Convert mapping to records and set in engine
                                mapping_records = mapping_df.to_dict('records')
                                engine.set_mapping(mapping_records, join_columns)
                                
                                # Store updated mapping and join columns in session state
                                st.session_state.column_mapping = mapping_records
                                st.session_state['active_join_columns'] = join_columns
                                
                                # Show detailed join column analysis
                                with st.expander("🔍 Join Column Analysis"):
                                    st.markdown("### Selected Join Columns")
                                    for col in join_columns:
                                        source_unique = st.session_state.source_data[col].nunique()
                                        target_unique = st.session_state.target_data[col].nunique()
                                        source_nulls = st.session_state.source_data[col].isnull().sum()
                                        target_nulls = st.session_state.target_data[col].isnull().sum()
                                        
                                        try:
                                            # Calculate match ratio safely
                                            max_unique = max(source_unique, target_unique)
                                            match_ratio = (min(source_unique, target_unique) / max_unique) if max_unique > 0 else 0
                                            
                                            # Get sample values safely
                                            source_sample = st.session_state.source_data[col].dropna().unique()[:3]
                                            target_sample = st.session_state.target_data[col].dropna().unique()[:3]
                                            
                                            # Format sample values safely
                                            source_samples_str = ', '.join(str(x)[:50] for x in source_sample)  # Truncate long values
                                            target_samples_str = ', '.join(str(x)[:50] for x in target_sample)  # Truncate long values
                                            
                                            st.markdown(f"""
                                                **{col}**
                                                - Unique values: Source ({source_unique}) | Target ({target_unique})
                                                - Null values: Source ({source_nulls}) | Target ({target_nulls})
                                                - Match ratio: {match_ratio:.2%}
                                                - Sample values:
                                                  - Source: {source_samples_str}
                                                  - Target: {target_samples_str}
                                            """)
                                            
                                            # Show warnings inside try block to ensure variables are defined
                                            if source_unique != target_unique:
                                                st.warning(f"⚠️ Different number of unique values in {col}")
                                            if source_nulls > 0 or target_nulls > 0:
                                                st.warning(f"⚠️ Found null values in {col}")
                                            if match_ratio < 0.9:  # Less than 90% match
                                                st.warning(f"⚠️ Low match ratio ({match_ratio:.2%}) for {col}")
                                                
                                        except Exception as e:
                                            st.error(f"⚠️ Error analyzing join column {col}: {str(e)}")
                                            continue
                                
                                # Log confirmation of mapping update
                                st.success(f"✅ Mapping updated with {len(join_columns)} join column(s)")
                                
                                # Perform comparison with updated mapping
                                comparison_results = engine.compare()
                                st.session_state.comparison_results = comparison_results
                                
                                # Show results
                                st.subheader("3. Comparison Results")
                                
                                # Summary statistics
                                result_col1, result_col2, result_col3 = st.columns(3)
                                with result_col1:
                                    st.metric("Rows Match", "Yes" if comparison_results['rows_match'] else "No")
                                with result_col2:
                                    st.metric("Columns Match", "Yes" if comparison_results['columns_match'] else "No")
                                with result_col3:
                                    st.metric("Overall Match", "Yes" if comparison_results['match_status'] else "No")
                                
                                # Detailed report
                                with st.expander("View Detailed Report"):
                                    st.text(comparison_results['datacompy_report'])
                                    
                                    if len(comparison_results['source_unmatched_rows']) > 0:
                                        st.markdown("### Unmatched Rows in Source")
                                        st.dataframe(comparison_results['source_unmatched_rows'])
                                        
                                    if len(comparison_results['target_unmatched_rows']) > 0:
                                        st.markdown("### Unmatched Rows in Target")
                                        st.dataframe(comparison_results['target_unmatched_rows'])
                                
                            except Exception as e:
                                st.error(f"Comparison failed: {str(e)}")
                    else:
                        st.warning("Please select at least one join column for comparison")
                except Exception as e:
                    st.error(f"Error loading sample data: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Manual data upload section
    if not use_sample:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Source")
            source_type = st.selectbox("Select Source Type", SUPPORTED_SOURCES, key="source_type")
            
            # File upload or connection parameters for source
            source_data = None
            if source_type in ['CSV file', 'DAT file', 'Parquet file', 'Flat files inside zipped folder']:
                source_file = st.file_uploader(f"Upload {source_type}", key="source_file")
                if source_type in ['CSV file', 'DAT file', 'Flat files inside zipped folder']:
                    source_delimiter = st.text_input("Source Delimiter", ",", key="source_delimiter")
                else:
                    source_delimiter = ","
            else:
                source_params = get_connection_inputs(source_type, "source")
                source_delimiter = ","
                
        with col2:
            st.markdown("### Target")
            target_type = st.selectbox("Select Target Type", SUPPORTED_SOURCES, key="target_type")
            
            # File upload or connection parameters for target
            target_data = None
            if target_type in ['CSV file', 'DAT file', 'Parquet file', 'Flat files inside zipped folder']:
                target_file = st.file_uploader(f"Upload {target_type}", key="target_file")
                if target_type in ['CSV file', 'DAT file', 'Flat files inside zipped folder']:
                    target_delimiter = st.text_input("Target Delimiter", ",", key="target_delimiter")
                else:
                    target_delimiter = ","
            else:
                target_params = get_connection_inputs(target_type, "target")
                target_delimiter = ","

    # Load Data button
    if st.button("Load Data"):
        try:

            # Validate source input
            if source_type in ['CSV file', 'DAT file', 'Parquet file', 'Flat files inside zipped folder']:
                if 'source_file' not in st.session_state or st.session_state.source_file is None:
                    st.error(f"Please upload a {source_type} for source data")
                    return
            else:
                if not source_params or not source_params.get('server'):
                    st.error("Please provide hostname for source")
                    return
                if source_type != 'Teradata' and not source_params.get('database'):
                    st.error("Please provide database name for source")
                    return

            # Validate target input
            if target_type in ['CSV file', 'DAT file', 'Parquet file', 'Flat files inside zipped folder']:
                if 'target_file' not in st.session_state or st.session_state.target_file is None:
                    st.error(f"Please upload a {target_type} for target data")
                    return
            else:
                if not target_params or not target_params.get('server'):
                    st.error("Please provide hostname for target")
                    return
                if target_type != 'Teradata' and not target_params.get('database'):
                    st.error("Please provide database name for target")
                    return

            # Load source data
            with st.spinner("Loading source data..."):
                try:
                    if source_type in ['CSV file', 'DAT file', 'Parquet file', 'Flat files inside zipped folder']:
                        source_data = load_data(source_type, st.session_state.source_file, None, source_delimiter)
                    else:
                        source_data = load_data(source_type, None, source_params)
                    
                    if source_data is None or source_data.empty:
                        st.error("No data loaded from source. Please check your source configuration.")
                        return
                except Exception as e:
                    st.error(f"Error loading source data: {str(e)}")
                    return
                    
            # Load target data
            with st.spinner("Loading target data..."):
                try:
                    if target_type in ['CSV file', 'DAT file', 'Parquet file', 'Flat files inside zipped folder']:
                        target_data = load_data(target_type, st.session_state.target_file, None, target_delimiter)
                    else:
                        target_data = load_data(target_type, None, target_params)
                    
                    if target_data is None or target_data.empty:
                        st.error("No data loaded from target. Please check your target configuration.")
                        return
                except Exception as e:
                    st.error(f"Error loading target data: {str(e)}")
                    return
                    
            # Store in session state
            st.session_state.source_data = source_data
            st.session_state.target_data = target_data
            
            # Initialize comparison engine
            engine = ComparisonEngine(source_data, target_data)
            
            # Initialize mapping if not already present
            if 'column_mapping' not in st.session_state:
                st.session_state.column_mapping = engine.auto_map_columns()
            
            st.success("✅ Data loaded successfully!")
            
            # Show preview with mapping status
            st.subheader("Data Preview")
            
            # Show initial mapping status
            st.info("🔄 Auto-mapping columns...")
            mapped_cols = sum(1 for m in st.session_state.column_mapping if m['target'])
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Auto-mapped Columns", f"{mapped_cols}/{len(st.session_state.column_mapping)}")
            with col2:
                join_cols = sum(1 for m in st.session_state.column_mapping if m.get('join', False))
                st.metric("Join Columns", join_cols)
            with col3:
                excluded_cols = sum(1 for m in st.session_state.column_mapping if m.get('exclude', False))
                st.metric("Excluded Columns", excluded_cols)
            
            # Show data preview
            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                st.markdown("### Source Data")
                st.dataframe(source_data.head(MAX_PREVIEW_ROWS))
            with preview_col2:
                st.markdown("### Target Data")
                st.dataframe(target_data.head(MAX_PREVIEW_ROWS))
            
            # Show next steps guidance
            st.info("✨ Data loaded successfully! Proceed to Column Mapping section below to configure comparison settings.")
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return

    # Column Mapping
    if 'source_data' in st.session_state and 'target_data' in st.session_state:
        st.subheader("2. Column Mapping")
        
        # Initialize comparison engine
        engine = ComparisonEngine(st.session_state.source_data, st.session_state.target_data)
        
        # Get automatic mapping
        if 'column_mapping' not in st.session_state:
            st.session_state.column_mapping = engine.auto_map_columns()
        
        # Column Mapping Section
        st.markdown("### Column Mapping Configuration")
        
        # Add buttons for quick mapping actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Auto-Map All"):
                mapping_data = pd.DataFrame(engine.auto_map_columns())
                st.session_state.column_mapping = mapping_data.to_dict('records')
                st.success("Columns auto-mapped successfully!")
                
        with col2:
            if st.button("Clear All Mappings"):
                for mapping in st.session_state.column_mapping:
                    mapping['target'] = ''
                    mapping['join'] = False
                    mapping['exclude'] = False
                st.success("All mappings cleared!")
                
        with col3:
            if st.button("Reset to Default"):
                st.session_state.column_mapping = engine.auto_map_columns()
                st.success("Mappings reset to default!")

        # Create mapping editor with enhanced UI
        st.markdown("#### Edit Column Mappings")
        st.markdown("- Select target columns from dropdown")
        st.markdown("- Check 'Join Column' for columns to use in matching records")
        st.markdown("- Check 'Exclude' to ignore columns in comparison")
        
        mapping_data = pd.DataFrame(st.session_state.column_mapping)
        edited_mapping = create_mapping_editor(mapping_data, st.session_state.target_data.columns)
        
        # Update session state with edited mapping
        if edited_mapping is not None:
            update_column_mapping(edited_mapping, engine)
        
        # Show mapping summary
        st.markdown("#### Mapping Summary")
        mapped_cols = sum(1 for m in st.session_state.column_mapping if m['target'])
        join_cols = sum(1 for m in st.session_state.column_mapping if m['join'])
        excluded_cols = sum(1 for m in st.session_state.column_mapping if m['exclude'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mapped Columns", f"{mapped_cols}/{len(st.session_state.column_mapping)}")
        with col2:
            st.metric("Join Columns", join_cols)
        with col3:
            st.metric("Excluded Columns", excluded_cols)
        
        # Convert mapping to DataFrame for validation
        mapping_df = pd.DataFrame(st.session_state.column_mapping)
        
        # Ensure join column is boolean
        mapping_df['join'] = mapping_df['join'].fillna(False).astype(bool)
        
        # Get selected join columns
        join_columns = list(mapping_df[mapping_df['join']]['source'])
        
        # Store join columns in session state for debugging
        st.session_state['current_join_columns'] = join_columns
        
        # Show join column status with clear visibility
        if not join_columns:
            st.warning("⚠️ Please select at least one join column for comparison", icon="⚠️")
            # Add helper text with more details
            st.markdown("""
                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>
                    <p><strong>Why do I need join columns?</strong></p>
                    <p>Join columns are used to match records between source and target data. 
                    Select columns that uniquely identify records (e.g., ID fields, primary keys).</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.success(f"✅ Selected join columns: {', '.join(join_columns)}")
            # Debug information
            st.info(f"Debug: Found {len(join_columns)} join column(s): {join_columns}")
            # Query Section (before Compare button)
            st.markdown("### Data Filtering (Optional)")
            enable_query = st.checkbox("Enable Data Filtering", help="Filter source or target data before comparison")
            
            source_query = None
            target_query = None
            
            if enable_query:
                query_col1, query_col2 = st.columns(2)
                with query_col1:
                    st.markdown("#### Source Data Filter")
                    source_query = st.text_area(
                        "SQL Query for Source Data",
                        placeholder="Example: SELECT * FROM data WHERE amount > 1000",
                        help="Leave empty to use all data"
                    )
                    if source_query:
                        st.info("✓ Query will be applied to source data")
                        
                with query_col2:
                    st.markdown("#### Target Data Filter")
                    target_query = st.text_area(
                        "SQL Query for Target Data",
                        placeholder="Example: SELECT * FROM data WHERE status = 'active'",
                        help="Leave empty to use all data"
                    )
                    if target_query:
                        st.info("✓ Query will be applied to target data")
                
                # Query Examples
                with st.expander("Show Query Examples"):
                    st.code("""
# Select all records
SELECT * FROM data

# Filter by column value
SELECT * FROM data WHERE column_name = 'value'

# Select specific columns
SELECT column1, column2 FROM data

# Multiple conditions
SELECT * FROM data WHERE amount > 1000 AND status = 'active'
                    """)
            
            # Show filtered data preview if queries are applied
            if enable_query and (source_query or target_query):
                st.subheader("Filtered Data Preview")
                preview_col1, preview_col2 = st.columns(2)
                
                with preview_col1:
                    st.markdown("#### Source Data")
                    if source_query:
                        if 'filtered_source' in st.session_state:
                            filtered_df = st.session_state['filtered_source']
                            if not filtered_df.empty:
                                st.write(f"Showing {len(filtered_df)} rows")
                                st.dataframe(filtered_df.head(10))
                            else:
                                st.warning("Query returned no results")
                        else:
                            st.info("Execute comparison to see filtered data")
                    else:
                        st.write("Original source data")
                        st.dataframe(st.session_state.source_data.head(10))
                
                with preview_col2:
                    st.markdown("#### Target Data")
                    if target_query:
                        if 'filtered_target' in st.session_state:
                            filtered_df = st.session_state['filtered_target']
                            if not filtered_df.empty:
                                st.write(f"Showing {len(filtered_df)} rows")
                                st.dataframe(filtered_df.head(10))
                            else:
                                st.warning("Query returned no results")
                        else:
                            st.info("Execute comparison to see filtered data")
                    else:
                        st.write("Original target data")
                        st.dataframe(st.session_state.target_data.head(10))
            
            # Compare button with consistent key
            if st.button("Compare Data", key="main_compare"):
                try:
                    with st.spinner("Comparing data..."):
                        # Validate join columns
                        if not join_columns:
                            st.error("⚠️ No join columns selected. Please select at least one join column.")
                            return
                        
                        # Log current join column status
                        st.info(f"Starting comparison with join columns: {', '.join(join_columns)}")
                        
                        # Verify join columns exist in both datasets
                        missing_in_source = [col for col in join_columns if col not in mapping_df['source'].values]
                        missing_in_target = [col for col in join_columns if col not in st.session_state.target_data.columns]
                        
                        if missing_in_source or missing_in_target:
                            error_msg = []
                            if missing_in_source:
                                error_msg.append(f"Columns missing in source: {', '.join(missing_in_source)}")
                            if missing_in_target:
                                error_msg.append(f"Columns missing in target: {', '.join(missing_in_target)}")
                            st.error("⚠️ " + " | ".join(error_msg))
                            return
                        
                        # Convert mapping to records and set in engine
                        mapping_records = mapping_df.to_dict('records')
                        engine.set_mapping(mapping_records, join_columns)
                        
                        # Store updated mapping and join columns in session state
                        st.session_state.column_mapping = mapping_records
                        st.session_state['active_join_columns'] = join_columns
                        
                        # Show detailed join column analysis
                        with st.expander("🔍 Join Column Analysis"):
                            st.markdown("### Selected Join Columns")
                            for col in join_columns:
                                source_unique = st.session_state.source_data[col].nunique()
                                target_unique = st.session_state.target_data[col].nunique()
                                source_nulls = st.session_state.source_data[col].isnull().sum()
                                target_nulls = st.session_state.target_data[col].isnull().sum()
                                
                                try:
                                    # Get sample values and calculate statistics safely
                                    source_sample = st.session_state.source_data[col].dropna().unique()[:3]
                                    target_sample = st.session_state.target_data[col].dropna().unique()[:3]
                                    
                                    # Calculate match ratio
                                    max_unique = max(source_unique, target_unique)
                                    match_ratio = (min(source_unique, target_unique) / max_unique) if max_unique > 0 else 0
                                        
                                        # Format sample values safely
                                        source_samples_str = ', '.join(str(x)[:50] for x in source_sample)  # Truncate long values
                                        target_samples_str = ', '.join(str(x)[:50] for x in target_sample)  # Truncate long values
                                        
                                        # Display analysis results
                                        st.markdown(f"""
                                            **{col}**
                                            - Unique values: Source ({source_unique}) | Target ({target_unique})
                                            - Null values: Source ({source_nulls}) | Target ({target_nulls})
                                            - Match ratio: {match_ratio:.2%}
                                            - Sample values:
                                              - Source: {source_samples_str}
                                              - Target: {target_samples_str}
                                        """)
                                        
                                        # Show warnings for potential issues
                                        if source_unique != target_unique:
                                            st.warning(f"⚠️ Different number of unique values in {col}")
                                        if source_nulls > 0 or target_nulls > 0:
                                            st.warning(f"⚠️ Found null values in {col}")
                                        if match_ratio < 0.9:  # Less than 90% match
                                            st.warning(f"⚠️ Low match ratio ({match_ratio:.2%}) for {col}")
                                            
                                    except Exception as e:
                                        st.error(f"⚠️ Error analyzing join column {col}: {str(e)}")
                                        continue
                        
                        # Log confirmation of mapping update
                        st.success(f"✅ Mapping updated with {len(join_columns)} join column(s)")
                        
                        try:
                            # Perform comparison with queries
                            comparison_results = engine.compare(
                                source_query=source_query,
                                target_query=target_query
                            )
                            
                            # Initialize default comparison results
                            default_results = {
                                'rows_match': False,
                                'columns_match': False,
                                'match_status': False,
                                'datacompy_report': '',
                                'source_unmatched_rows': pd.DataFrame(),
                                'target_unmatched_rows': pd.DataFrame(),
                                'column_summary': {},
                                'distinct_values': {},
                                'row_counts': {
                                    'source_name': 'Source',
                                    'target_name': 'Target',
                                    'source_count': 0,
                                    'target_count': 0
                                }
                            }
                            
                            # Update default results with actual results
                            if comparison_results is not None:
                                for key in default_results:
                                    if key in comparison_results:
                                        default_results[key] = comparison_results[key]
                            
                            # Ensure boolean values are properly set
                            default_results['rows_match'] = bool(default_results['rows_match'])
                            default_results['columns_match'] = bool(default_results['columns_match'])
                            default_results['match_status'] = bool(default_results['match_status'])
                            
                            # Use the properly initialized results
                            comparison_results = default_results
                                
                        except Exception as e:
                            st.error(f"Comparison failed: {str(e)}")
                            return
                        
                        # Generate reports
                        report_gen = ReportGenerator("reports")
                        
                        # Generate and save reports
                        report_paths = {}
                        
                        with st.spinner("Generating DataCompy report..."):
                            # Save DataCompy report
                            datacompy_path = report_gen.generate_datacompy_report(comparison_results)
                            report_paths['datacompy'] = datacompy_path
                            
                        with st.spinner("Generating Y-DataProfiling report..."):
                            # Generate profile reports
                            profile_paths = engine.generate_profiling_reports("reports")
                            report_paths.update(profile_paths)
                        
                        with st.spinner("Generating Regression report..."):
                            # Enhanced Regression report with multiple checks
                            report_paths['regression'] = report_gen.generate_regression_report(
                                comparison_results,
                                st.session_state.source_data,
                                st.session_state.target_data
                            )
                        
                        with st.spinner("Generating Side-by-side Difference report..."):
                            # Enhanced difference report
                            diff_report = report_gen.generate_difference_report(
                                st.session_state.source_data,
                                st.session_state.target_data,
                                join_columns
                            )
                            if diff_report:
                                report_paths['differences'] = diff_report
                        
                        with st.spinner("Creating report archive..."):
                            # Create ZIP archive with all reports
                            zip_path = report_gen.create_report_archive(report_paths)
                        
                        # Store results in session state
                        st.session_state.comparison_results = comparison_results
                        st.session_state.report_paths = report_paths
                        st.session_state.zip_path = zip_path
                        
                        st.success("All reports generated successfully!")
                        
                except Exception as e:
                    st.error(f"Comparison failed: {str(e)}")
                    return


    # Display Results
    if st.session_state.comparison_results:
        st.subheader("3. Comparison Results")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows Match", "Yes" if st.session_state.comparison_results['rows_match'] else "No")
        with col2:
            st.metric("Columns Match", "Yes" if st.session_state.comparison_results['columns_match'] else "No")
        with col3:
            st.metric("Overall Match", "Yes" if st.session_state.comparison_results['match_status'] else "No")
        
        # DataCompy Report
        with st.expander("View DataCompy Report"):
            st.text(st.session_state.comparison_results['datacompy_report'])
        
        # Download Reports Section
        st.subheader("4. Download Reports")
        
        # Create tabs for different report categories
        report_tabs = st.tabs([
            "Analysis Reports", 
            "Difference Reports", 
            "Profile Reports",
            "Download All"
        ])
        
        with report_tabs[0]:
            st.markdown("### Analysis Reports")
            if 'regression' in st.session_state.report_paths:
                with open(st.session_state.report_paths['regression'], 'rb') as f:
                    st.download_button(
                        "📊 Download Regression Report (Excel)",
                        f,
                        file_name=os.path.basename(st.session_state.report_paths['regression']),
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        help="Contains Count Check, Aggregation Check, and Distinct Value Analysis"
                    )
            
            # DataCompy Report
            st.markdown("#### DataCompy Report")
            st.text(st.session_state.comparison_results['datacompy_report'])
            
        with report_tabs[1]:
            st.markdown("### Difference Reports")
            if 'differences' in st.session_state.report_paths:
                with open(st.session_state.report_paths['differences'], 'rb') as f:
                    st.download_button(
                        "📋 Download Side-by-Side Difference Report (Excel)",
                        f,
                        file_name=os.path.basename(st.session_state.report_paths['differences']),
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        help="Shows source and target data side by side with highlighted differences"
                    )
            else:
                st.info("No differences found between source and target datasets.")
            
        with report_tabs[2]:
            st.markdown("### Profile Reports")
            # Add custom CSS for profile container
            st.markdown("""
                <style>
                    .profile-container {
                        height: 600px;
                        overflow-y: auto;
                        border: 1px solid #e0e0e0;
                        border-radius: 4px;
                        padding: 10px;
                        margin-top: 10px;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Display profile reports with tabs
            if any(key.startswith('profile') for key in st.session_state.report_paths.keys()):
                profile_tabs = st.tabs(["Source Profile", "Target Profile", "Comparison Profile"])
                
                with profile_tabs[0]:
                    if 'source_profile' in st.session_state.report_paths:
                        with open(st.session_state.report_paths['source_profile'], 'rb') as f:
                            st.download_button(
                                "📈 Download Source Profile",
                                f,
                                file_name=os.path.basename(st.session_state.report_paths['source_profile']),
                                mime='text/html'
                            )
                        with open(st.session_state.report_paths['source_profile'], 'r') as f:
                            st.components.v1.html(f.read(), height=600, scrolling=True)
                
                with profile_tabs[1]:
                    if 'target_profile' in st.session_state.report_paths:
                        with open(st.session_state.report_paths['target_profile'], 'rb') as f:
                            st.download_button(
                                "📈 Download Target Profile",
                                f,
                                file_name=os.path.basename(st.session_state.report_paths['target_profile']),
                                mime='text/html'
                            )
                        with open(st.session_state.report_paths['target_profile'], 'r') as f:
                            st.components.v1.html(f.read(), height=600, scrolling=True)
                
                with profile_tabs[2]:
                    if 'comparison_profile' in st.session_state.report_paths:
                        with open(st.session_state.report_paths['comparison_profile'], 'rb') as f:
                            st.download_button(
                                "📈 Download Comparison Profile",
                                f,
                                file_name=os.path.basename(st.session_state.report_paths['comparison_profile']),
                                mime='text/html'
                            )
                        with open(st.session_state.report_paths['comparison_profile'], 'r') as f:
                            st.components.v1.html(f.read(), height=600, scrolling=True)
        
        with report_tabs[3]:
            st.markdown("### Download All Reports")
            with open(st.session_state.zip_path, 'rb') as f:
                st.download_button(
                    "📦 Download All Reports (ZIP)",
                    f,
                    file_name=os.path.basename(st.session_state.zip_path),
                    mime='application/zip',
                    help="Contains all generated reports in a single ZIP file"
                )
            st.info("The ZIP archive contains all reports including regression analysis, differences, and data profiles.")

if __name__ == "__main__":
    main()

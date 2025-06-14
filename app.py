"""Main Streamlit application for the Comparison Tool."""
import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import os
from typing import Dict, Any, Tuple
from datetime import datetime

# Import local modules
from utils.data_loader import DataLoader
from utils.comparison_engine import ComparisonEngine
from reports.report_generator import ReportGenerator

def update_column_mapping(edited_mapping, engine):
    """Helper function to safely update column mapping in session state."""
    if edited_mapping is not None:
        try:
            st.session_state.column_mapping = edited_mapping.to_dict('records')
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
                    help="Use this column to match records between source and target"
                ),
                "data_type": st.column_config.SelectboxColumn(
                    "Data Type",
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
                f"{source_type} Server",
                key=f"{prefix}_server"
            )
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
    'decimal': 'float',
    'float': 'float',
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
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_upload.name) as tmp_file:
                tmp_file.write(file_upload.getvalue())
                
                # Try reading with pandas directly first
                try:
                    df = pd.read_csv(tmp_file.name, delimiter=delimiter)
                    if df.empty:
                        raise ValueError("No data found in the file")
                    return df
                except pd.errors.EmptyDataError:
                    raise ValueError("The file appears to be empty")
                except Exception as e:
                    # If direct reading fails, try chunked reading
                    return loader.read_chunked_file(tmp_file.name, delimiter=delimiter)
        except Exception as e:
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
                    
                    st.success("✅ Sample data loaded successfully!")
                    
                    # Show preview of sample data
                    st.markdown("### Data Preview")
                    tab1, tab2 = st.tabs(["Source Data", "Target Data"])
                    with tab1:
                        st.dataframe(source_data.head(), use_container_width=True)
                    with tab2:
                        st.dataframe(target_data.head(), use_container_width=True)
                    
                    # Initialize comparison engine
                    engine = ComparisonEngine(source_data, target_data)
                    
                    # Get automatic mapping
                    if 'column_mapping' not in st.session_state:
                        st.session_state.column_mapping = engine.auto_map_columns()
                    
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
                    
                    # Compare button
                    if join_cols > 0:
                        if st.button("Compare Data"):
                            try:
                                # Get selected join columns
                                join_columns = [m['source'] for m in st.session_state.column_mapping if m['join']]
                                
                                # Set mapping in comparison engine
                                engine.set_mapping(st.session_state.column_mapping, join_columns)
                                
                                # Perform comparison
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
                source_params = get_connection_inputs(source_type, "source")
                
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
                target_params = get_connection_inputs(target_type, "target")

    # Load Data button
    if st.button("Load Data"):
        try:
            # Validate inputs before loading
            if source_type in ['CSV file', 'DAT file', 'Parquet file', 'Flat files inside zipped folder']:
                if 'source_file' not in st.session_state or st.session_state.source_file is None:
                    st.error(f"Please upload a {source_type} for source data")
                    return
            else:
                if not source_params or not source_params.get('server') or not source_params.get('database'):
                    st.error("Please provide all required connection parameters for source")
                    return

            if target_type in ['CSV file', 'DAT file', 'Parquet file', 'Flat files inside zipped folder']:
                if 'target_file' not in st.session_state or st.session_state.target_file is None:
                    st.error(f"Please upload a {target_type} for target data")
                    return
            else:
                if not target_params or not target_params.get('server') or not target_params.get('database'):
                    st.error("Please provide all required connection parameters for target")
                    return

            # Load source data
            with st.spinner("Loading source data..."):
                try:
                    if source_type in ['CSV file', 'DAT file', 'Parquet file', 'Flat files inside zipped folder']:
                        source_data = load_data(source_type, st.session_state.source_file,
                                             delimiter=st.session_state.get('source_delimiter', ','))
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
                        target_data = load_data(target_type, st.session_state.target_file,
                                             delimiter=st.session_state.get('target_delimiter', ','))
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
            
            # Show preview
            st.subheader("Data Preview")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Source Data")
                st.dataframe(source_data.head(MAX_PREVIEW_ROWS))
            with col2:
                st.markdown("### Target Data")
                st.dataframe(target_data.head(MAX_PREVIEW_ROWS))
                
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
        
        # Get selected join columns
        join_columns = [m['source'] for m in st.session_state.column_mapping if m['join']]
        
        if not join_columns:
            st.warning("Please select at least one join column")
        else:
            # Compare button
            if st.button("Compare"):
                try:
                    with st.spinner("Performing comparison..."):
                        # Set mapping in comparison engine
                        engine.set_mapping(st.session_state.column_mapping, join_columns)
                        
                        # Perform comparison
                        comparison_results = engine.compare()
                        
                        # Generate reports
                        report_gen = ReportGenerator("reports")
                        
                        # Generate and save reports
                        report_paths = {}
                        
                        # Regression report
                        report_paths['regression'] = report_gen.generate_regression_report(comparison_results)
                        
                        # Difference report
                        if 'source_unmatched_rows' in comparison_results:
                            differences = pd.concat([
                                comparison_results['source_unmatched_rows'],
                                comparison_results['target_unmatched_rows']
                            ])
                            report_paths['differences'] = report_gen.generate_difference_report(differences)
                        
                        # Profile reports
                        profile_paths = engine.generate_profiling_reports("reports")
                        report_paths.update(profile_paths)
                        
                        # Create ZIP archive
                        zip_path = report_gen.create_report_archive(report_paths)
                        
                        # Store results in session state
                        st.session_state.comparison_results = comparison_results
                        st.session_state.report_paths = report_paths
                        st.session_state.zip_path = zip_path
                        
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
        
        # Download buttons
        st.subheader("4. Download Reports")
        
        col1, col2 = st.columns(2)
        with col1:
            # Individual report downloads
            for report_type, path in st.session_state.report_paths.items():
                with open(path, 'rb') as f:
                    st.download_button(
                        f"Download {report_type.replace('_', ' ').title()} Report",
                        f,
                        file_name=os.path.basename(path),
                        mime='application/octet-stream'
                    )
        
        with col2:
            # ZIP archive download
            with open(st.session_state.zip_path, 'rb') as f:
                st.download_button(
                    "Download All Reports (ZIP)",
                    f,
                    file_name=os.path.basename(st.session_state.zip_path),
                    mime='application/zip'
                )

if __name__ == "__main__":
    main()

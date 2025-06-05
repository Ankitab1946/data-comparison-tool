"""Core comparison engine for data comparison operations."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from ydata_profiling import ProfileReport
from pathlib import Path
from flask import request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComparisonEngine:
    # SQL Server to pandas type mapping
    SQL_TYPE_MAPPING = {
        'varchar': 'string',
        'nvarchar': 'string',
        'char': 'string',
        'nchar': 'string',
        'text': 'string',
        'ntext': 'string',
        'int': 'int32',
        'bigint': 'int64',
        'smallint': 'int32',
        'tinyint': 'int32',
        'numeric': 'float64',
        'decimal': 'float64',
        'float': 'float64',
        'real': 'float32',
        'bit': 'bool',
        'date': 'datetime64[ns]',
        'datetime': 'datetime64[ns]',
        'datetime2': 'datetime64[ns]',
        'datetimeoffset': 'datetime64[ns]',
        'time': 'datetime64[ns]',
        'binary': 'string',
        'varbinary': 'string',
        'uniqueidentifier': 'string',
        'xml': 'string',
        'unsupported': 'string'
    }

    # Pandas type mapping
    PANDAS_TYPE_MAPPING = {
        'object': 'string',
        'string': 'string',
        'int64': 'int64',
        'int32': 'int32',
        'float64': 'float64',
        'float32': 'float32',
        'bool': 'bool',
        'datetime64[ns]': 'datetime64[ns]',
        'category': 'string',
        'unsupported': 'string'
    }

    def __init__(self, source_df: pd.DataFrame, target_df: pd.DataFrame):
        """
        Initialize the comparison engine with source and target dataframes.
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
        """
        self.source_df = source_df
        self.target_df = target_df
        self.mapping = None
        self.join_columns = None
        self.excluded_columns = []
        self.type_overrides = {}  # Store user-defined type overrides

    def set_column_type(self, column: str, dtype: str):
        """
        Set a specific data type for a column.
        
        Args:
            column: Column name
            dtype: Data type to use for the column
        """
        self.type_overrides[column] = dtype

    def _get_mapped_type(self, column: str, current_type: str, is_sql_type: bool = False) -> str:
        """
        Get the mapped data type for a column.
        
        Args:
            column: Column name
            current_type: Current data type
            is_sql_type: Whether the type is from SQL Server
            
        Returns:
            Mapped data type string
        """
        # Check for user override first
        if column in self.type_overrides:
            return self.type_overrides[column]
        
        # Special handling for ColumnNameID
        if column == 'ColumnNameID':
            return 'string'
            
        # Convert current type to lowercase and clean it
        current_type = str(current_type).lower().strip()
        # Remove length specifications like varchar(50)
        current_type = current_type.split('(')[0]
        
        # Handle SQL Server types
        if is_sql_type:
            # Try exact match first
            if current_type in self.SQL_TYPE_MAPPING:
                return self.SQL_TYPE_MAPPING[current_type]
            # Try partial match
            for sql_type, pandas_type in self.SQL_TYPE_MAPPING.items():
                if sql_type in current_type:
                    return pandas_type
        
        # Handle Pandas types
        else:
            # Remove [ns] from datetime types
            base_type = current_type.split('[')[0]
            # Try exact match first
            if base_type in self.PANDAS_TYPE_MAPPING:
                return self.PANDAS_TYPE_MAPPING[base_type]
            # Try partial match
            for pandas_type, mapped_type in self.PANDAS_TYPE_MAPPING.items():
                if pandas_type in base_type:
                    return mapped_type
        
        # If type contains 'char' or 'text', treat as string
        if 'char' in current_type or 'text' in current_type:
            return 'string'
            
        # Default to string for unknown types
        logger.warning(f"Unknown type '{current_type}' for column '{column}', defaulting to string")
        return 'string'

    def auto_map_columns(self) -> List[Dict[str, Any]]:
        """
        Automatically map columns between source and target based on names and data types.
        
        Returns:
            List of dictionaries containing column mappings
        """
        source_cols = list(self.source_df.columns)
        target_cols = list(self.target_df.columns)
        mapping = []

        for s_col in source_cols:
            # Try exact match first
            t_col = s_col if s_col in target_cols else None
            
            # If no exact match, try case-insensitive match
            if not t_col:
                t_col = next((col for col in target_cols 
                            if col.lower() == s_col.lower()), None)
            
            # If still no match, try removing special characters
            if not t_col:
                s_clean = ''.join(e.lower() for e in s_col if e.isalnum())
                t_col = next((col for col in target_cols 
                            if ''.join(e.lower() for e in col if e.isalnum()) == s_clean), None)

            # Get source and target types
            source_type = str(self.source_df[s_col].dtype)
            target_type = str(self.target_df[t_col].dtype) if t_col else 'unknown'
            
            # Convert 'object' type to 'string'
            if source_type == 'object':
                source_type = 'string'
            if target_type == 'object':
                target_type = 'string'
            
            # Check if either type is from SQL Server
            is_sql_source = any(sql_type in source_type.lower() for sql_type in self.SQL_TYPE_MAPPING.keys())
            is_sql_target = any(sql_type in target_type.lower() for sql_type in self.SQL_TYPE_MAPPING.keys())
            
            # Get mapped types
            source_mapped = self._get_mapped_type(s_col, source_type, is_sql_source)
            target_mapped = self._get_mapped_type(s_col, target_type, is_sql_target) if t_col else source_mapped
            
            # Store original types
            original_source_type = source_type
            original_target_type = target_type
            
            # Determine final type
            if source_mapped == target_mapped:
                mapped_type = source_mapped
            else:
                # For mismatched types, prefer string for text-like columns
                if any(t in str(source_type).lower() or t in str(target_type).lower() 
                      for t in ['char', 'text', 'string', 'object']):
                    mapped_type = 'string'
                    original_source_type = 'string'
                    original_target_type = 'string'
                else:
                    # For numeric types, use the wider type
                    if all(t in ['int32', 'int64', 'float32', 'float64'] for t in [source_mapped, target_mapped]):
                        mapped_type = 'float64'  # widest numeric type
                    else:
                        mapped_type = 'string'  # fallback for incompatible types
                        original_source_type = 'string'
                        original_target_type = 'string'

            # Create mapping with editable types
            mapping_entry = {
                'source': s_col,
                'target': t_col or '',
                'join': False,
                'data_type': mapped_type,
                'exclude': False,
                'source_type': original_source_type,
                'target_type': original_target_type,
                'editable': True,  # Flag to indicate type is editable
                'original_source_type': original_source_type,  # Keep original for reference
                'original_target_type': original_target_type   # Keep original for reference
            }
            
            # Special handling for memory-intensive numeric columns
            if mapped_type in ['float64', 'int64']:
                # Check if column has too many unique values
                unique_count = min(
                    self.source_df[s_col].nunique() if s_col in self.source_df else 0,
                    self.target_df[t_col].nunique() if t_col in self.target_df else 0
                )
                if unique_count > 1000000:  # Threshold for large columns
                    mapping_entry['data_type'] = 'string'
                    mapping_entry['source_type'] = 'string'
                    mapping_entry['target_type'] = 'string'
                    logger.warning(f"Converting {s_col} to string due to high cardinality")
            
            mapping.append(mapping_entry)

        return mapping

    def set_mapping(self, mapping: List[Dict[str, Any]], join_columns: List[str]):
        """
        Set the column mapping and join columns for comparison.
        
        Args:
            mapping: List of mapping dictionaries
            join_columns: List of columns to use for joining
        """
        self.mapping = mapping
        self.join_columns = join_columns
        self.excluded_columns = [m['source'] for m in mapping if m['exclude']]
        
        # Store original data types
        self.source_types = {m['source']: m.get('source_type', '') for m in mapping}
        self.target_types = {m['source']: m.get('target_type', '') for m in mapping}

    def update_column_types(self, column: str, source_type: str = None, target_type: str = None):
        """
        Update the data types for a column.
        
        Args:
            column: Column name
            source_type: New source data type
            target_type: New target data type
        """
        if not self.mapping:
            raise ValueError("Mapping must be set before updating types")
            
        for m in self.mapping:
            if m['source'] == column and m.get('editable', True):
                # Convert 'object' to 'string'
                if source_type == 'object':
                    source_type = 'string'
                if target_type == 'object':
                    target_type = 'string'
                
                # Update types
                if source_type:
                    m['source_type'] = source_type
                    self.source_types[column] = source_type
                if target_type:
                    m['target_type'] = target_type
                    self.target_types[column] = target_type
                
                # Determine new mapped type
                new_source_type = source_type or m['source_type']
                new_target_type = target_type or m['target_type']
                
                # Check for memory-intensive numeric types
                if new_source_type in ['float64', 'int64'] or new_target_type in ['float64', 'int64']:
                    unique_count = min(
                        self.source_df[column].nunique() if column in self.source_df else 0,
                        self.target_df[column].nunique() if column in self.target_df else 0
                    )
                    if unique_count > 1000000:  # Threshold for large columns
                        logger.warning(f"Converting {column} to string due to high cardinality")
                        m['data_type'] = 'string'
                        m['source_type'] = 'string'
                        m['target_type'] = 'string'
                        return
                
                # Update the mapped type
                m['data_type'] = self._get_mapped_type(column, new_source_type, True)
                break

    def _process_in_chunks(self, source_df: pd.DataFrame, target_df: pd.DataFrame, 
                          join_columns: List[str], chunk_size: int = 100000) -> Dict[str, pd.DataFrame]:
        """
        Process large dataframes in chunks to avoid memory issues.
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            join_columns: Columns to join on
            chunk_size: Size of each chunk
            
        Returns:
            Dictionary containing unmatched rows
        """
        source_unmatched = []
        target_unmatched = []
        
        # Process in chunks
        for i in range(0, len(source_df), chunk_size):
            source_chunk = source_df.iloc[i:i + chunk_size]
            
            # Merge chunk with target
            merged = pd.merge(source_chunk, target_df, on=join_columns, how='outer', indicator=True)
            
            # Collect unmatched rows
            source_unmatched.append(merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1))
            target_unmatched.append(merged[merged['_merge'] == 'right_only'].drop('_merge', axis=1))
            
            # Clear memory
            del merged
            
        return {
            'source_unmatched': pd.concat(source_unmatched) if source_unmatched else pd.DataFrame(),
            'target_unmatched': pd.concat(target_unmatched) if target_unmatched else pd.DataFrame()
        }

    def _prepare_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare dataframes for comparison by applying mappings and type conversions.
        Memory-optimized version that processes columns individually.
        
        Returns:
            Tuple of (prepared source DataFrame, prepared target DataFrame)
        """
        if not self.mapping:
            raise ValueError("Mapping must be set before comparison")

        # Create empty dataframes with only required columns
        source = pd.DataFrame()
        target = pd.DataFrame()

        # Process one column at a time
        for mapping in self.mapping:
            if mapping['exclude'] or not mapping['target']:
                continue

            source_col = mapping['source']
            target_col = mapping['target']
            
            try:
                # Get single columns from original dataframes
                source[source_col] = self.source_df[source_col].copy()
                target[source_col] = self.target_df[target_col].copy()  # Use source_col as new name

                # Get mapped type from mapping configuration
                mapped_type = mapping.get('data_type', 'string')

                # Convert types with memory optimization
                try:
                    if mapped_type == 'string' or 'char' in str(source[source_col].dtype).lower():
                        source[source_col] = source[source_col].fillna('').astype('string')  # Use pandas string type
                        target[source_col] = target[source_col].fillna('').astype('string')
                    elif mapped_type in ['int32', 'int64']:
                        # Check cardinality before converting
                        unique_count = min(
                            source[source_col].nunique(),
                            target[source_col].nunique()
                        )
                        if unique_count > 1000000:  # High cardinality
                            source[source_col] = source[source_col].fillna('').astype('string')
                            target[source_col] = target[source_col].fillna('').astype('string')
                            logger.warning(f"Converting {source_col} to string due to high cardinality")
                        else:
                            # Use smaller integer types when possible
                            source[source_col] = pd.to_numeric(source[source_col], errors='coerce', downcast='integer')
                            target[source_col] = pd.to_numeric(target[source_col], errors='coerce', downcast='integer')
                    elif mapped_type in ['float32', 'float64']:
                        # Check cardinality before converting
                        unique_count = min(
                            source[source_col].nunique(),
                            target[source_col].nunique()
                        )
                        if unique_count > 1000000:  # High cardinality
                            source[source_col] = source[source_col].fillna('').astype('string')
                            target[source_col] = target[source_col].fillna('').astype('string')
                            logger.warning(f"Converting {source_col} to string due to high cardinality")
                        else:
                            # Use float32 instead of float64 to save memory
                            source[source_col] = pd.to_numeric(source[source_col], errors='coerce', downcast='float')
                            target[source_col] = pd.to_numeric(target[source_col], errors='coerce', downcast='float')
                    elif mapped_type == 'datetime64[ns]':
                        source[source_col] = pd.to_datetime(source[source_col], errors='coerce')
                        target[source_col] = pd.to_datetime(target[source_col], errors='coerce')
                    elif mapped_type == 'bool':
                        source[source_col] = source[source_col].fillna(False).astype('boolean')  # Use pandas boolean type
                        target[source_col] = target[source_col].fillna(False).astype('boolean')
                    else:
                        # Default to string for unknown types
                        source[source_col] = source[source_col].fillna('').astype('string')
                        target[source_col] = target[source_col].fillna('').astype('string')

                    # Free memory after processing each column
                    import gc
                    gc.collect()

                except Exception as e:
                    logger.warning(f"Type conversion failed for column {source_col}: {str(e)}. Converting to string.")
                    source[source_col] = source[source_col].fillna('').astype('string')
                    target[source_col] = target[source_col].fillna('').astype('string')

            except Exception as e:
                logger.error(f"Error processing column {source_col}: {str(e)}")
                # Skip problematic column
                if source_col in source.columns:
                    source.drop(columns=[source_col], inplace=True)
                if source_col in target.columns:
                    target.drop(columns=[source_col], inplace=True)

        # Return the prepared dataframes
        return source, target

    def compare(self, chunk_size: int = 100000) -> Dict[str, Any]:
        """
        Perform the comparison between source and target data.
        
        Args:
            chunk_size: Size of chunks for processing large datasets
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            source, target = self._prepare_dataframes()
            
            # Initialize comparison results with optimized data handling
            try:
                # Use efficient row counting
                source_count = source.shape[0]  # Faster than len()
                target_count = target.shape[0]
                
                # Initialize results with proper structure
                results = {
                    'match_status': False,
                    'rows_match': False,
                    'columns_match': False,
                    'datacompy_report': '',
                    'source_unmatched_rows': pd.DataFrame(),
                    'target_unmatched_rows': pd.DataFrame(),
                    'column_summary': {},  # Will be populated later
                    'row_counts': {
                        'source_name': 'Source',
                        'target_name': 'Target',
                        'source_count': source_count,
                        'target_count': target_count
                    },
                    'distinct_values': {}
                }
                
                # Generate column summary in chunks if needed
                if source_count > 100000 or target_count > 100000:
                    results['column_summary'] = self._generate_column_summary_chunked(source, target)
                else:
                    results['column_summary'] = self._generate_column_summary(source, target)
                
                # Optimized comparison checks
                results['columns_match'] = set(source.columns) == set(target.columns)
                results['rows_match'] = source_count == target_count
                
            except Exception as e:
                logger.error(f"Error initializing results: {str(e)}")
                return {
                    'match_status': False,
                    'error': str(e),
                    'datacompy_report': f"Error initializing comparison: {str(e)}",
                    'row_counts': {
                        'source_name': 'Source',
                        'target_name': 'Target',
                        'source_count': 0,
                        'target_count': 0
                    }
                }

            # Get distinct values for non-numeric columns
            try:
                results['distinct_values'] = self.get_distinct_values()
            except Exception as e:
                logger.warning(f"Error getting distinct values: {str(e)}")
                results['distinct_values'] = {}

            # Detailed comparison
            if self.join_columns:
                try:
                    # Process large datasets in chunks
                    unmatched = self._process_in_chunks(source, target, self.join_columns, chunk_size)
                    results['source_unmatched_rows'] = unmatched['source_unmatched']
                    results['target_unmatched_rows'] = unmatched['target_unmatched']
                    
                    # Generate detailed comparison report
                    report_lines = []
                    report_lines.append("DataCompy Comparison Report")
                    report_lines.append("=" * 50)
                    report_lines.append("\nSummary:")
                    report_lines.append("-" * 20)
                    report_lines.append(f"Source rows: {len(source)}")
                    report_lines.append(f"Target rows: {len(target)}")
                    report_lines.append(f"Unmatched in source: {len(results['source_unmatched_rows'])}")
                    report_lines.append(f"Unmatched in target: {len(results['target_unmatched_rows'])}")
                    
                    # Add column comparison
                    report_lines.append("\nColumn Analysis:")
                    report_lines.append("-" * 20)
                    for col in source.columns:
                        report_lines.append(f"\nColumn: {col}")
                        if col in results['column_summary']:
                            summary = results['column_summary'][col]
                            report_lines.append(f"Source null count: {summary['source_null_count']}")
                            report_lines.append(f"Target null count: {summary['target_null_count']}")
                            report_lines.append(f"Source unique count: {summary['source_unique_count']}")
                            report_lines.append(f"Target unique count: {summary['target_unique_count']}")
                            if 'source_sum' in summary:  # Numeric columns
                                report_lines.append(f"Source sum: {summary['source_sum']}")
                                report_lines.append(f"Target sum: {summary['target_sum']}")
                                report_lines.append(f"Source mean: {summary['source_mean']}")
                                report_lines.append(f"Target mean: {summary['target_mean']}")
                    
                    # Add value distribution for join columns
                    report_lines.append("\nJoin Columns Analysis:")
                    report_lines.append("-" * 20)
                    if results['distinct_values']:
                        for col in self.join_columns:
                            if col in results['distinct_values']:
                                report_lines.append(f"\nJoin Column: {col}")
                                report_lines.append(f"Source unique values: {results['distinct_values'][col]['source_count']}")
                                report_lines.append(f"Target unique values: {results['distinct_values'][col]['target_count']}")
                                report_lines.append("Sample values comparison:")
                                s_vals = list(results['distinct_values'][col]['source_values'].items())[:5]
                                t_vals = list(results['distinct_values'][col]['target_values'].items())[:5]
                                report_lines.append("Source top 5: " + ", ".join(f"{v}({c})" for v, c in s_vals))
                                report_lines.append("Target top 5: " + ", ".join(f"{v}({c})" for v, c in t_vals))
                    
                    # Add sample of unmatched rows
                    if len(results['source_unmatched_rows']) > 0:
                        report_lines.append("\nSample Unmatched Rows in Source:")
                        report_lines.append("-" * 20)
                        sample = results['source_unmatched_rows'].head(5).to_string()
                        report_lines.append(sample)
                    
                    if len(results['target_unmatched_rows']) > 0:
                        report_lines.append("\nSample Unmatched Rows in Target:")
                        report_lines.append("-" * 20)
                        sample = results['target_unmatched_rows'].head(5).to_string()
                        report_lines.append(sample)
                    
                    results['datacompy_report'] = "\n".join(report_lines)
                    
                    # Overall match status
                    results['match_status'] = (
                        results['columns_match'] and 
                        results['rows_match'] and
                        len(results['source_unmatched_rows']) == 0 and
                        len(results['target_unmatched_rows']) == 0
                    )
                except Exception as e:
                    logger.error(f"Error in detailed comparison: {str(e)}")
                    results['datacompy_report'] = f"Error in comparison: {str(e)}"
                    results['match_status'] = False

            return results
        except Exception as e:
            logger.error(f"Error in compare method: {str(e)}")
            return {
                'match_status': False,
                'error': str(e),
                'datacompy_report': f"Comparison failed: {str(e)}"
            }

    def _generate_column_summary(self, source: pd.DataFrame, 
                               target: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Generate detailed column-level comparison summary.
        
        Args:
            source: Prepared source DataFrame
            target: Prepared target DataFrame
            
        Returns:
            Dictionary containing column-level statistics
        """
        summary = {}
        
        for col in source.columns:
            if col in self.join_columns:
                continue
                
            try:
                summary[col] = {
                    'source_null_count': int(source[col].isnull().sum()),
                    'target_null_count': int(target[col].isnull().sum()),
                    'source_unique_count': int(source[col].nunique()),
                    'target_unique_count': int(target[col].nunique()),
                }
                
                # For numeric columns, add statistical comparisons
                if np.issubdtype(source[col].dtype, np.number):
                    summary[col].update({
                        'source_sum': float(source[col].sum()),
                        'target_sum': float(target[col].sum()),
                        'source_mean': float(source[col].mean()),
                        'target_mean': float(target[col].mean()),
                        'source_std': float(source[col].std()),
                        'target_std': float(target[col].std()),
                    })
            except Exception as e:
                logger.warning(f"Error processing column {col}: {str(e)}")
                summary[col] = {
                    'source_null_count': 0,
                    'target_null_count': 0,
                    'source_unique_count': 0,
                    'target_unique_count': 0
                }

        return summary

    def _generate_column_summary_chunked(self, source: pd.DataFrame, 
                                       target: pd.DataFrame, 
                                       chunk_size: int = 50000) -> Dict[str, Dict[str, Any]]:
        """
        Generate detailed column-level comparison summary using chunked processing.
        
        Args:
            source: Prepared source DataFrame
            target: Prepared target DataFrame
            chunk_size: Size of chunks for processing
            
        Returns:
            Dictionary containing column-level statistics
        """
        summary = {}
        
        for col in source.columns:
            if col in self.join_columns:
                continue
            
            try:
                # Initialize summary for this column
                col_summary = {
                    'source_null_count': 0,
                    'target_null_count': 0,
                    'source_unique_count': 0,
                    'target_unique_count': 0
                }
                
                # Process source data in chunks
                unique_values_source = set()
                source_sum = 0.0
                source_values = []
                
                for i in range(0, len(source), chunk_size):
                    chunk = source.iloc[i:i + chunk_size]
                    col_summary['source_null_count'] += int(chunk[col].isnull().sum())
                    
                    # Handle unique values based on dtype
                    if np.issubdtype(chunk[col].dtype, np.number):
                        non_null = chunk[col].dropna()
                        unique_values_source.update(non_null.astype('float64').tolist())
                        source_sum += float(non_null.sum())
                        source_values.extend(non_null.astype('float64').tolist())
                    else:
                        unique_values_source.update(chunk[col].dropna().astype(str).tolist())
                
                # Process target data in chunks
                unique_values_target = set()
                target_sum = 0.0
                target_values = []
                
                for i in range(0, len(target), chunk_size):
                    chunk = target.iloc[i:i + chunk_size]
                    col_summary['target_null_count'] += int(chunk[col].isnull().sum())
                    
                    # Handle unique values based on dtype
                    if np.issubdtype(chunk[col].dtype, np.number):
                        non_null = chunk[col].dropna()
                        unique_values_target.update(non_null.astype('float64').tolist())
                        target_sum += float(non_null.sum())
                        target_values.extend(non_null.astype('float64').tolist())
                    else:
                        unique_values_target.update(chunk[col].dropna().astype(str).tolist())
                
                col_summary['source_unique_count'] = len(unique_values_source)
                col_summary['target_unique_count'] = len(unique_values_target)
                
                # For numeric columns, calculate statistics
                if np.issubdtype(source[col].dtype, np.number):
                    try:
                        # Convert to numpy arrays with explicit dtype
                        source_values = np.array(source_values, dtype=np.float64)
                        target_values = np.array(target_values, dtype=np.float64)
                        
                        col_summary.update({
                            'source_sum': float(source_sum),
                            'target_sum': float(target_sum),
                            'source_mean': float(np.mean(source_values)) if len(source_values) > 0 else 0.0,
                            'target_mean': float(np.mean(target_values)) if len(target_values) > 0 else 0.0,
                            'source_std': float(np.std(source_values)) if len(source_values) > 0 else 0.0,
                            'target_std': float(np.std(target_values)) if len(target_values) > 0 else 0.0
                        })
                    except Exception as e:
                        logger.warning(f"Error calculating statistics for column {col}: {str(e)}")
                        col_summary.update({
                            'source_sum': 0.0,
                            'target_sum': 0.0,
                            'source_mean': 0.0,
                            'target_mean': 0.0,
                            'source_std': 0.0,
                            'target_std': 0.0
                        })
                
                summary[col] = col_summary
                
            except Exception as e:
                logger.warning(f"Error processing column {col}: {str(e)}")
                summary[col] = {
                    'source_null_count': 0,
                    'target_null_count': 0,
                    'source_unique_count': 0,
                    'target_unique_count': 0
                }
        
        return summary

    def generate_profiling_reports(self, output_dir: str) -> Dict[str, str]:
        """
        Generate YData Profiling reports for source and target data.
        
        Args:
            output_dir: Directory to save the reports
            
        Returns:
            Dictionary containing paths to generated reports
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Create copies of dataframes to avoid modifying originals
            source_df = self.source_df.copy()
            target_df = self.target_df.copy()

            # Convert problematic columns to string
            for df in [source_df, target_df]:
                for col in df.columns:
                    # Check if column type is problematic
                    col_type = str(df[col].dtype).lower()
                    if ('object' in col_type or 
                        'unsupported' in col_type or 
                        col == 'ColumnNameID' or
                        any(t in col_type for t in ['char', 'text'])):
                        df[col] = df[col].fillna('').astype(str)

            # Generate individual profiles with memory optimization
            profile_kwargs = {
                'progress_bar': False,
                'explorative': True,
                'minimal': True,  # Reduce memory usage
                'pool_size': 1,   # Reduce parallel processing
                'samples': None   # Disable sample generation
            }

            try:
                source_profile = ProfileReport(
                    source_df, 
                    title="Source Data Profile",
                    **profile_kwargs
                )
            except Exception as e:
                logger.error(f"Error generating source profile: {str(e)}")
                # Try with more aggressive memory optimization
                source_profile = ProfileReport(
                    source_df.astype(str),
                    title="Source Data Profile",
                    minimal=True,
                    pool_size=1,
                    samples=None,
                    progress_bar=False
                )

            try:
                target_profile = ProfileReport(
                    target_df,
                    title="Target Data Profile",
                    **profile_kwargs
                )
            except Exception as e:
                logger.error(f"Error generating target profile: {str(e)}")
                # Try with more aggressive memory optimization
                target_profile = ProfileReport(
                    target_df.astype(str),
                    title="Target Data Profile",
                    minimal=True,
                    pool_size=1,
                    samples=None,
                    progress_bar=False
                )

            # Save reports
            source_path = output_path / "source_profile.html"
            target_path = output_path / "target_profile.html"
            comparison_path = output_path / "comparison_profile.html"

            source_profile.to_file(str(source_path))
            target_profile.to_file(str(target_path))
            
            # Generate comparison report with error handling
            try:
                comparison_report = source_profile.compare(target_profile)
                comparison_report.to_file(str(comparison_path))
            except Exception as e:
                logger.error(f"Error generating comparison report: {str(e)}")
                # Create a basic comparison report
                with open(str(comparison_path), 'w') as f:
                    f.write("""
                    <html>
                    <head><title>Data Comparison Report</title></head>
                    <body>
                        <h1>Data Comparison Report</h1>
                        <p>Error generating detailed comparison report. Please check individual profiles.</p>
                        <ul>
                            <li><a href="source_profile.html">Source Profile</a></li>
                            <li><a href="target_profile.html">Target Profile</a></li>
                        </ul>
                    </body>
                    </html>
                    """)

            return {
                'source_profile': str(source_path),
                'target_profile': str(target_path),
                'comparison_profile': str(comparison_path)
            }
            
        except Exception as e:
            logger.error(f"Error in generate_profiling_reports: {str(e)}")
            raise

    def get_distinct_values(self, columns: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get distinct values and their counts for specified columns.
        
        Args:
            columns: List of columns to analyze. If None, analyze all non-numeric columns.
            
        Returns:
            Dictionary containing distinct values and counts for each column
        """
        try:
            source, target = self._prepare_dataframes()
            
            if not columns:
                # Get all columns that exist in both dataframes
                columns = [col for col in source.columns 
                        if col in target.columns and not np.issubdtype(source[col].dtype, np.number)]
            
            if not columns:  # If still no columns, return empty dict
                return {}

            distinct_values = {}
            for col in columns:
                try:
                    if col in source.columns and col in target.columns:
                        source_distinct = source[col].value_counts().to_dict()
                        target_distinct = target[col].value_counts().to_dict()
                        
                        distinct_values[col] = {
                            'source_values': source_distinct,
                            'target_values': target_distinct,
                            'source_count': len(source_distinct),
                            'target_count': len(target_distinct),
                            'matching': set(source_distinct.keys()) == set(target_distinct.keys())
                        }
                except Exception as e:
                    logger.warning(f"Error processing column {col}: {str(e)}")
                    continue

            return distinct_values
        except Exception as e:
            logger.error(f"Error in get_distinct_values: {str(e)}")
            return {}

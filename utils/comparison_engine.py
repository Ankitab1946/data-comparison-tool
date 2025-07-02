"""Core comparison engine for data comparison operations."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from ydata_profiling import ProfileReport
from pathlib import Path
from flask import request
import re

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
        """Initialize comparison engine with source and target dataframes."""
        self.source_df = source_df
        self.target_df = target_df
        self.mapping = None
        self.join_columns = None
        self.excluded_columns = []
        self.type_overrides = {}

    def _apply_type_conversion(self, source: pd.DataFrame, target: pd.DataFrame, 
                             col: str, mapped_type: str) -> None:
        """Apply type conversion to a column in both dataframes."""
        try:
            source_len = len(source)
            target_len = len(target)
            
            source_temp = source[col].copy()
            target_temp = target[col].copy()
            
            if mapped_type == 'string' or 'char' in str(source[col].dtype).lower():
                try:
                    # Use pandas string type with error handling
                    source_temp = pd.Series(source_temp, dtype=pd.StringDtype())
                    target_temp = pd.Series(target_temp, dtype=pd.StringDtype())
                except Exception as e:
                    logger.warning(f"StringDtype conversion failed for {col}: {str(e)}. Using object type.")
                    source_temp = source_temp.astype(object).fillna('')
                    target_temp = target_temp.astype(object).fillna('')
                
            elif mapped_type in ['int32', 'int64']:
                try:
                    source_temp = pd.to_numeric(source_temp, errors='coerce')
                    target_temp = pd.to_numeric(target_temp, errors='coerce')
                    source_temp = source_temp.fillna(0).astype(np.int64)
                    target_temp = target_temp.fillna(0).astype(np.int64)
                except Exception as e:
                    logger.warning(f"Integer conversion failed for {col}: {str(e)}. Converting to string.")
                    source_temp = pd.Series(source_temp, dtype=pd.StringDtype())
                    target_temp = pd.Series(target_temp, dtype=pd.StringDtype())
                    
            elif mapped_type in ['float32', 'float64']:
                try:
                    source_temp = pd.to_numeric(source_temp, errors='coerce')
                    target_temp = pd.to_numeric(target_temp, errors='coerce')
                    source_temp = source_temp.fillna(0).astype(np.float64)
                    target_temp = target_temp.fillna(0).astype(np.float64)
                except Exception as e:
                    logger.warning(f"Float conversion failed for {col}: {str(e)}. Converting to string.")
                    source_temp = pd.Series(source_temp, dtype=pd.StringDtype())
                    target_temp = pd.Series(target_temp, dtype=pd.StringDtype())
                    
            elif mapped_type == 'datetime64[ns]':
                try:
                    source_temp = pd.to_datetime(source_temp, errors='coerce')
                    target_temp = pd.to_datetime(target_temp, errors='coerce')
                except Exception as e:
                    logger.warning(f"Datetime conversion failed for {col}: {str(e)}. Converting to string.")
                    source_temp = pd.Series(source_temp, dtype=pd.StringDtype())
                    target_temp = pd.Series(target_temp, dtype=pd.StringDtype())
                
            elif mapped_type == 'bool':
                try:
                    source_temp = source_temp.map({'True': True, 'False': False, True: True, False: False, 
                                               1: True, 0: False}).fillna(False)
                    target_temp = target_temp.map({'True': True, 'False': False, True: True, False: False, 
                                               1: True, 0: False}).fillna(False)
                    source_temp = source_temp.astype(np.bool_)
                    target_temp = target_temp.astype(np.bool_)
                except Exception as e:
                    logger.warning(f"Boolean conversion failed for {col}: {str(e)}. Converting to string.")
                    source_temp = pd.Series(source_temp, dtype=pd.StringDtype())
                    target_temp = pd.Series(target_temp, dtype=pd.StringDtype())
            else:
                # Default to string for unknown types
                source_temp = pd.Series(source_temp, dtype=pd.StringDtype())
                target_temp = pd.Series(target_temp, dtype=pd.StringDtype())
            
            # Verify row counts are preserved
            if len(source_temp) != source_len or len(target_temp) != target_len:
                raise ValueError(f"Row count changed during conversion of {col}")
            
            # Only update if conversion was successful
            source[col] = source_temp
            target[col] = target_temp
            
            # Final verification
            if len(source[col]) != source_len or len(target[col]) != target_len:
                raise ValueError(f"Final row count mismatch for {col}")
                
        except Exception as e:
            logger.error(f"Error in type conversion for column {col}: {str(e)}")
            if col in source.columns:
                source.drop(columns=[col], inplace=True)
            if col in target.columns:
                target.drop(columns=[col], inplace=True)
            raise

    def _prepare_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare dataframes for comparison.
        Memory-optimized version for handling large datasets.
        """
        if not self.mapping:
            raise ValueError("Mapping must be set before comparison")

        try:
            logger.info("Starting dataframe preparation...")
            
            # Get required columns
            required_cols = {m['source']: m['target'] for m in self.mapping 
                           if not m['exclude'] and m['target']}
            
            if not required_cols:
                raise ValueError("No valid columns found in mapping")
            
            # Initialize empty dataframes
            source = pd.DataFrame()
            target = pd.DataFrame()
            
            # Process columns in batches to manage memory
            batch_size = 5  # Process 5 columns at a time
            column_batches = [list(required_cols.items())[i:i + batch_size] 
                            for i in range(0, len(required_cols), batch_size)]
            
            for batch_num, batch in enumerate(column_batches, 1):
                logger.info(f"Processing column batch {batch_num}/{len(column_batches)}")
                
                batch_source = pd.DataFrame()
                batch_target = pd.DataFrame()
                
                for source_col, target_col in batch:
                    try:
                        mapping = next(m for m in self.mapping if m['source'] == source_col)
                        mapped_type = mapping.get('data_type', 'string')
                        
                        # Copy columns with memory optimization
                        batch_source[source_col] = self.source_df[source_col].copy()
                        batch_target[source_col] = self.target_df[target_col].copy()
                        
                        # Convert numeric columns to smaller dtypes where possible
                        if pd.api.types.is_float_dtype(batch_source[source_col]):
                            batch_source[source_col] = pd.to_numeric(batch_source[source_col], downcast='float')
                            batch_target[source_col] = pd.to_numeric(batch_target[source_col], downcast='float')
                        elif pd.api.types.is_integer_dtype(batch_source[source_col]):
                            batch_source[source_col] = pd.to_numeric(batch_source[source_col], downcast='integer')
                            batch_target[source_col] = pd.to_numeric(batch_target[source_col], downcast='integer')
                        
                        # Apply type conversion
                        self._apply_type_conversion(batch_source, batch_target, source_col, mapped_type)
                        
                    except Exception as e:
                        logger.error(f"Error processing column {source_col}: {str(e)}")
                        if source_col in batch_source.columns:
                            batch_source.drop(columns=[source_col], inplace=True)
                        if source_col in batch_target.columns:
                            batch_target.drop(columns=[source_col], inplace=True)
                
                # Merge batch results into main dataframes
                if not batch_source.empty:
                    for col in batch_source.columns:
                        source[col] = batch_source[col]
                        target[col] = batch_target[col]
                
                # Clear memory
                del batch_source, batch_target
                gc.collect()
            
            return source, target
            
        except Exception as e:
            logger.error(f"Error in prepare_dataframes: {str(e)}")
            raise

    def compare(self, chunk_size: int = 50000, source_query: str = None, 
                target_query: str = None) -> Dict[str, Any]:
        """
        Perform the comparison between source and target data.
        
        Args:
            chunk_size: Size of chunks for processing large datasets
            source_query: Optional SQL-like query to filter source data
            target_query: Optional SQL-like query to filter target data
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            # Get initial prepared dataframes
            source, target = self._prepare_dataframes()
            
            # Apply queries if provided
            if source_query:
                try:
                    # Handle TOP/LIMIT in query
                    query = source_query.replace('top', 'limit', flags=re.IGNORECASE)
                    filtered_source = source.query(query, engine='python')
                    if not filtered_source.empty:
                        source = filtered_source
                    else:
                        logger.warning("Source query returned no results")
                except Exception as e:
                    logger.error(f"Error executing source query: {str(e)}")
                    return {
                        'match_status': False,
                        'error': f"Source query failed: {str(e)}",
                        'datacompy_report': f"Comparison failed: Source query error - {str(e)}"
                    }
            
            if target_query:
                try:
                    # Handle TOP/LIMIT in query
                    query = target_query.replace('top', 'limit', flags=re.IGNORECASE)
                    filtered_target = target.query(query, engine='python')
                    if not filtered_target.empty:
                        target = filtered_target
                    else:
                        logger.warning("Target query returned no results")
                except Exception as e:
                    logger.error(f"Error executing target query: {str(e)}")
                    return {
                        'match_status': False,
                        'error': f"Target query failed: {str(e)}",
                        'datacompy_report': f"Comparison failed: Target query error - {str(e)}"
                    }
            
            results = {
                'match_status': False,
                'rows_match': False,
                'columns_match': False,
                'datacompy_report': '',
                'source_unmatched_rows': pd.DataFrame(),
                'target_unmatched_rows': pd.DataFrame(),
                'column_summary': {},
                'row_counts': {
                    'source_name': 'Source',
                    'target_name': 'Target',
                    'source_count': len(source),
                    'target_count': len(target)
                }
            }
            
            # Basic comparisons
            results['columns_match'] = set(source.columns) == set(target.columns)
            results['rows_match'] = len(source) == len(target)
            
            # Process unmatched rows if join columns exist
            if self.join_columns:
                try:
                    unmatched = self._process_in_chunks(source, target, self.join_columns, chunk_size)
                    results['source_unmatched_rows'] = unmatched.get('source_unmatched', pd.DataFrame())
                    results['target_unmatched_rows'] = unmatched.get('target_unmatched', pd.DataFrame())
                except Exception as e:
                    logger.error(f"Error processing unmatched rows: {str(e)}")
            
            # Generate column summary
            results['column_summary'] = self._generate_column_summary(source, target)
            
            # Set final match status
            results['match_status'] = (
                results['columns_match'] and 
                results['rows_match'] and
                len(results['source_unmatched_rows']) == 0 and
                len(results['target_unmatched_rows']) == 0
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in compare method: {str(e)}")
            return {
                'match_status': False,
                'error': str(e),
                'datacompy_report': f"Comparison failed: {str(e)}"
            }

    def _process_in_chunks(self, source_df: pd.DataFrame, target_df: pd.DataFrame, 
                          join_columns: List[str], chunk_size: int = 10000) -> Dict[str, pd.DataFrame]:
        """
        Process large dataframes in chunks with memory optimization.
        Uses smaller chunk size and efficient data types.
        """
        source_unmatched = []
        target_unmatched = []
        processed_chunks = 0
        total_chunks = (len(source_df) + chunk_size - 1) // chunk_size

        # Convert numeric columns to smaller dtypes where possible
        for df in [source_df, target_df]:
            for col in df.columns:
                if pd.api.types.is_float_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast='float')
                elif pd.api.types.is_integer_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast='integer')

        # Process source in chunks
        for i in range(0, len(source_df), chunk_size):
            processed_chunks += 1
            logger.info(f"Processing chunk {processed_chunks}/{total_chunks}")
            
            try:
                # Get chunk of source data
                source_chunk = source_df.iloc[i:i + chunk_size]
                
                # Create hash of join columns for faster matching
                source_keys = set(map(tuple, source_chunk[join_columns].values))
                target_keys = set(map(tuple, target_df[join_columns].values))
                
                # Find unmatched in source
                source_unmatched_keys = source_keys - target_keys
                if source_unmatched_keys:
                    unmatched = source_chunk[source_chunk[join_columns].apply(tuple, axis=1).isin(source_unmatched_keys)]
                    source_unmatched.append(unmatched)
                
                # Find unmatched in target (only for the relevant subset)
                target_unmatched_keys = target_keys - source_keys
                if target_unmatched_keys:
                    unmatched = target_df[target_df[join_columns].apply(tuple, axis=1).isin(target_unmatched_keys)]
                    target_unmatched.append(unmatched)
                
                # Clear memory
                del source_chunk, source_keys, target_keys
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing chunk {processed_chunks}: {str(e)}")
                continue
        
        # Combine results efficiently
        try:
            result = {
                'source_unmatched': (pd.concat(source_unmatched, ignore_index=True, copy=False)
                                   if source_unmatched else pd.DataFrame()),
                'target_unmatched': (pd.concat(target_unmatched, ignore_index=True, copy=False)
                                   if target_unmatched else pd.DataFrame())
            }
            
            # Clear memory
            del source_unmatched, target_unmatched
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Error combining results: {str(e)}")
            return {
                'source_unmatched': pd.DataFrame(),
                'target_unmatched': pd.DataFrame()
            }

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
        Includes all columns from both source and target, even if unmapped.
        
        Returns:
            List of dictionaries containing column mappings
        """
        source_cols = list(self.source_df.columns)
        target_cols = list(self.target_df.columns)
        mapping = []
        mapped_sources = set()
        mapped_targets = set()

        # First pass: Map source columns
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
            
            # Get mapped types
            is_sql_source = any(sql_type in source_type.lower() for sql_type in self.SQL_TYPE_MAPPING.keys())
            is_sql_target = any(sql_type in target_type.lower() for sql_type in self.SQL_TYPE_MAPPING.keys())
            source_mapped = self._get_mapped_type(s_col, source_type, is_sql_source)
            target_mapped = self._get_mapped_type(s_col, target_type, is_sql_target) if t_col else source_mapped
            
            # Determine final type
            if source_mapped == target_mapped:
                mapped_type = source_mapped
            else:
                mapped_type = 'string'  # Default to string for mismatched types

            # Create mapping entry for source column
            mapping_entry = {
                'source': s_col,
                'target': t_col or '',  # Empty string if no target match
                'join': False,
                'data_type': mapped_type,
                'exclude': False,
                'source_type': source_type,
                'target_type': target_type,
                'editable': True,
                'original_source_type': source_type,
                'original_target_type': target_type
            }
            
            # Handle memory-intensive columns
            if mapped_type in ['float64', 'int64']:
                unique_count = min(
                    self.source_df[s_col].nunique() if s_col in self.source_df else 0,
                    self.target_df[t_col].nunique() if t_col in self.target_df else 0
                )
                if unique_count > 1000000:
                    mapping_entry['data_type'] = 'string'
                    mapping_entry['source_type'] = 'string'
                    mapping_entry['target_type'] = 'string'
                    logger.warning(f"Converting {s_col} to string due to high cardinality")
            
            mapping.append(mapping_entry)
            mapped_sources.add(s_col)
            if t_col:
                mapped_targets.add(t_col)

        # Second pass: Add unmapped target columns
        for t_col in target_cols:
            if t_col not in mapped_targets:
                target_type = str(self.target_df[t_col].dtype)
                if target_type == 'object':
                    target_type = 'string'
                
                mapping_entry = {
                    'source': '',  # Empty string for unmapped source
                    'target': t_col,
                    'join': False,
                    'data_type': self._get_mapped_type(t_col, target_type, False),
                    'exclude': False,
                    'source_type': '',
                    'target_type': target_type,
                    'editable': True,
                    'original_source_type': '',
                    'original_target_type': target_type
                }
                mapping.append(mapping_entry)

        return mapping

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
                # Keep existing types if they're float
                if m['source_type'] == 'float' and not source_type:
                    source_type = 'float'
                if m['target_type'] == 'float' and not target_type:
                    target_type = 'float'
                
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
                
                # Preserve float type if either source or target is float
                if new_source_type == 'float' or new_target_type == 'float':
                    m['data_type'] = 'float'
                    return
                
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

    def set_mapping(self, mapping: List[Dict[str, Any]], join_columns: List[str]):
        """
        Set the column mapping and join columns for comparison.
        
        Args:
            mapping: List of mapping dictionaries
            join_columns: List of columns to use for joining
        """
        # Store current types before updating mapping
        current_types = {}
        if self.mapping:
            for m in self.mapping:
                if m['source']:
                    current_types[m['source']] = {
                        'source_type': m.get('source_type', ''),
                        'target_type': m.get('target_type', ''),
                        'data_type': m.get('data_type', '')
                    }

        # Update mapping
        self.mapping = mapping
        self.join_columns = join_columns
        self.excluded_columns = [m['source'] for m in mapping if m['exclude']]
        
        # Preserve float types from previous mapping
        for m in self.mapping:
            if m['source'] in current_types:
                prev_types = current_types[m['source']]
                if prev_types['data_type'] == 'float' or 'float' in [prev_types['source_type'], prev_types['target_type']]:
                    m['data_type'] = 'float'
                    m['source_type'] = 'float'
                    m['target_type'] = 'float'
        
        # Store original data types
        self.source_types = {m['source']: m.get('source_type', '') for m in mapping}
        self.target_types = {m['source']: m.get('target_type', '') for m in mapping}

    def generate_profiling_reports(self, output_dir: str) -> Dict[str, str]:
        """
        Generate YData Profiling reports for source and target data.
        
        Args:
            output_dir: Directory to save the reports
            
        Returns:
            Dictionary containing paths to generated reports
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize report paths
        source_path = output_path / "source_profile.html"
        target_path = output_path / "target_profile.html"
        comparison_path = output_path / "comparison_profile.html"
        regression_path = output_path / "regression_report.html"
        
        result_paths = {
            'source_profile': str(source_path),
            'target_profile': str(target_path),
            'comparison_profile': str(comparison_path),
            'regression_report': str(regression_path)
        }

        try:
            # Create copies of dataframes and apply mapping types
            source_df = self.source_df.copy()
            target_df = self.target_df.copy()

            # Apply type conversions
            self._prepare_dataframes_for_profiling(source_df, target_df)

            # Generate profiles with error handling
            profile_kwargs = {
                'progress_bar': False,
                'explorative': True,
                'minimal': True,
                'pool_size': 1,
                'samples': None,
                'words_to_plot': 100  # Ensure word cloud has enough words
            }

            # Generate source profile
            try:
                source_profile = ProfileReport(
                    source_df, 
                    title="Source Data Profile",
                    **profile_kwargs
                )
                source_profile.to_file(str(source_path))
            except Exception as e:
                logger.error(f"Error generating source profile: {str(e)}")
                self._generate_basic_report(source_path, "Source Data Profile", str(e))

            # Generate target profile
            try:
                target_profile = ProfileReport(
                    target_df,
                    title="Target Data Profile",
                    **profile_kwargs
                )
                target_profile.to_file(str(target_path))
            except Exception as e:
                logger.error(f"Error generating target profile: {str(e)}")
                self._generate_basic_report(target_path, "Target Data Profile", str(e))

            # Generate comparison report
            try:
                if 'source_profile' in locals() and 'target_profile' in locals():
                    comparison_report = source_profile.compare(target_profile)
                    comparison_report.to_file(str(comparison_path))
            except Exception as e:
                logger.error(f"Error generating comparison report: {str(e)}")
                self._generate_basic_report(comparison_path, "Comparison Report", str(e))

            # Generate regression report (continues even if profiling failed)
            try:
                self._generate_regression_report(source_df, target_df, regression_path)
            except Exception as e:
                logger.error(f"Error generating regression report: {str(e)}")
                self._generate_basic_report(regression_path, "Regression Report", str(e))

        except Exception as e:
            logger.error(f"Error in generate_profiling_reports: {str(e)}")
            # Generate basic reports for all failed reports
            for path, title in [
                (source_path, "Source Data Profile"),
                (target_path, "Target Data Profile"),
                (comparison_path, "Comparison Report"),
                (regression_path, "Regression Report")
            ]:
                if not path.exists():
                    self._generate_basic_report(path, title, str(e))

        return result_paths

    def _prepare_dataframes_for_profiling(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> None:
        """Prepare dataframes for profiling by applying type conversions."""
        if self.mapping:
            logger.info("Applying mapping types before generating profiles...")
            for mapping in self.mapping:
                if mapping['exclude'] or not mapping['target']:
                    continue

                source_col = mapping['source']
                target_col = mapping['target']
                mapped_type = mapping.get('data_type', 'string')

                try:
                    # Apply consistent type conversion based on mapping
                    if mapped_type == 'string' or mapped_type == 'datetime64[ns]':
                        source_df[source_col] = source_df[source_col].fillna('').astype(str)
                        target_df[target_col] = target_df[target_col].fillna('').astype(str)
                    elif mapped_type in ['int32', 'int64']:
                        source_df[source_col] = pd.to_numeric(source_df[source_col], errors='coerce').fillna(0).astype(np.int64)
                        target_df[target_col] = pd.to_numeric(target_df[target_col], errors='coerce').fillna(0).astype(np.int64)
                    elif mapped_type in ['float32', 'float64']:
                        source_df[source_col] = pd.to_numeric(source_df[source_col], errors='coerce').fillna(0).astype(np.float64)
                        target_df[target_col] = pd.to_numeric(target_df[target_col], errors='coerce').fillna(0).astype(np.float64)
                except Exception as e:
                    logger.warning(f"Error converting types for column {source_col}: {str(e)}")
                    source_df[source_col] = source_df[source_col].fillna('').astype(str)
                    target_df[target_col] = target_df[target_col].fillna('').astype(str)
        else:
            # If no mapping, convert all to string
            for df in [source_df, target_df]:
                for col in df.columns:
                    df[col] = df[col].fillna('').astype(str)

    def _generate_basic_report(self, path: Path, title: str, error_msg: str) -> None:
        """Generate a basic HTML report when profiling fails."""
        with open(str(path), 'w', encoding='utf-8') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 2em; line-height: 1.6; }}
                    h1, h2 {{ color: #333; }}
                    .error {{ color: #721c24; background-color: #f8d7da; padding: 1em; border-radius: 4px; }}
                    .info {{ color: #0c5460; background-color: #d1ecf1; padding: 1em; border-radius: 4px; }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
                <div class="error">
                    <h2>Error Generating Report</h2>
                    <p>An error occurred while generating this report: {error_msg}</p>
                </div>
                <div class="info">
                    <h2>Next Steps</h2>
                    <p>While this report generation failed, other comparison results are still available.</p>
                    <p>Please check the regression report and other available reports for analysis.</p>
                </div>
            </body>
            </html>
            """)

    def _generate_regression_report(self, source_df: pd.DataFrame, target_df: pd.DataFrame, path: Path) -> None:
        """
        Generate a regression analysis report comparing source and target data.
        Memory-optimized version for large datasets.
        """
        try:
            # Calculate basic statistics in chunks
            stats = {}
            chunk_size = 10000
            
            for col in source_df.columns:
                if col in target_df.columns and pd.api.types.is_numeric_dtype(source_df[col]):
                    # Initialize accumulators
                    source_sum = 0
                    target_sum = 0
                    source_sq_sum = 0
                    target_sq_sum = 0
                    cross_prod_sum = 0
                    count = 0
                    
                    # Process in chunks
                    for i in range(0, len(source_df), chunk_size):
                        source_chunk = source_df[col].iloc[i:i + chunk_size]
                        target_chunk = target_df[col].iloc[i:i + chunk_size]
                        
                        # Remove NaN values
                        mask = ~(source_chunk.isna() | target_chunk.isna())
                        source_chunk = source_chunk[mask]
                        target_chunk = target_chunk[mask]
                        
                        # Update accumulators
                        source_sum += source_chunk.sum()
                        target_sum += target_chunk.sum()
                        source_sq_sum += (source_chunk ** 2).sum()
                        target_sq_sum += (target_chunk ** 2).sum()
                        cross_prod_sum += (source_chunk * target_chunk).sum()
                        count += len(source_chunk)
                    
                    if count > 0:
                        # Calculate statistics
                        source_mean = source_sum / count
                        target_mean = target_sum / count
                        source_var = (source_sq_sum / count) - (source_mean ** 2)
                        target_var = (target_sq_sum / count) - (target_mean ** 2)
                        source_std = np.sqrt(source_var) if source_var > 0 else 0
                        target_std = np.sqrt(target_var) if target_var > 0 else 0
                        
                        # Calculate correlation
                        covariance = (cross_prod_sum / count) - (source_mean * target_mean)
                        correlation = (covariance / (source_std * target_std)) if source_std * target_std > 0 else 0
                        
                        stats[col] = {
                            'correlation': correlation,
                            'source_mean': source_mean,
                            'target_mean': target_mean,
                            'source_std': source_std,
                            'target_std': target_std,
                            'diff_mean': abs(source_mean - target_mean),
                            'diff_std': abs(source_std - target_std)
                        }

            # Generate HTML report
            with open(str(path), 'w', encoding='utf-8') as f:
                f.write("""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>Regression Analysis Report</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 2em; line-height: 1.6; }
                        table { border-collapse: collapse; width: 100%; margin: 1em 0; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        th { background-color: #f5f5f5; }
                        tr:nth-child(even) { background-color: #f9f9f9; }
                        .high-correlation { background-color: #d4edda; }
                        .low-correlation { background-color: #f8d7da; }
                    </style>
                </head>
                <body>
                    <h1>Regression Analysis Report</h1>
                    <table>
                        <tr>
                            <th>Column</th>
                            <th>Correlation</th>
                            <th>Source Mean</th>
                            <th>Target Mean</th>
                            <th>Mean Difference</th>
                            <th>Source Std</th>
                            <th>Target Std</th>
                            <th>Std Difference</th>
                        </tr>
                """)

                for col, stat in stats.items():
                    correlation_class = 'high-correlation' if stat['correlation'] > 0.9 else (
                        'low-correlation' if stat['correlation'] < 0.5 else '')
                    f.write(f"""
                        <tr class="{correlation_class}">
                            <td>{col}</td>
                            <td>{stat['correlation']:.4f}</td>
                            <td>{stat['source_mean']:.4f}</td>
                            <td>{stat['target_mean']:.4f}</td>
                            <td>{stat['diff_mean']:.4f}</td>
                            <td>{stat['source_std']:.4f}</td>
                            <td>{stat['target_std']:.4f}</td>
                            <td>{stat['diff_std']:.4f}</td>
                        </tr>
                    """)

                f.write("""
                    </table>
                </body>
                </html>
                """)

        except Exception as e:
            logger.error(f"Error generating regression report: {str(e)}")
            self._generate_basic_report(path, "Regression Analysis Report", str(e))

    def _generate_column_summary(self, source: pd.DataFrame, target: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Generate column-level summary statistics.
        Memory-optimized version for large datasets.
        """
        summary = {}
        chunk_size = 10000
        
        for col in source.columns:
            if col in self.join_columns:
                continue
                
            try:
                # Initialize accumulators
                source_null_count = 0
                target_null_count = 0
                source_unique_values = set()
                target_unique_values = set()
                source_sum = 0
                target_sum = 0
                source_count = 0
                target_count = 0
                
                # Process in chunks
                for i in range(0, len(source), chunk_size):
                    # Get chunks
                    source_chunk = source[col].iloc[i:i + chunk_size]
                    target_chunk = target[col].iloc[i:i + chunk_size]
                    
                    # Update null counts
                    source_null_count += int(source_chunk.isnull().sum())
                    target_null_count += int(target_chunk.isnull().sum())
                    
                    # Update unique values (for non-null values only)
                    source_unique_values.update(source_chunk.dropna().unique())
                    target_unique_values.update(target_chunk.dropna().unique())
                    
                    # For numeric columns, update sums
                    if np.issubdtype(source[col].dtype, np.number):
                        source_sum += float(source_chunk.fillna(0).sum())
                        target_sum += float(target_chunk.fillna(0).sum())
                        source_count += len(source_chunk.dropna())
                        target_count += len(target_chunk.dropna())
                
                # Create summary
                summary[col] = {
                    'source_null_count': source_null_count,
                    'target_null_count': target_null_count,
                    'source_unique_count': len(source_unique_values),
                    'target_unique_count': len(target_unique_values)
                }
                
                # Add numeric statistics if applicable
                if np.issubdtype(source[col].dtype, np.number) and source_count > 0:
                    summary[col].update({
                        'source_sum': source_sum,
                        'target_sum': target_sum,
                        'source_mean': source_sum / source_count if source_count > 0 else 0,
                        'target_mean': target_sum / target_count if target_count > 0 else 0
                    })
                
                # Clear memory
                del source_unique_values, target_unique_values
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Error processing column {col}: {str(e)}")
                summary[col] = {
                    'source_null_count': 0,
                    'target_null_count': 0,
                    'source_unique_count': 0,
                    'target_unique_count': 0
                }
        
        return summary

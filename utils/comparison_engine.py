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
        """Prepare dataframes for comparison."""
        if not self.mapping:
            raise ValueError("Mapping must be set before comparison")

        try:
            logger.info("Starting dataframe preparation...")
            
            required_cols = {m['source']: m['target'] for m in self.mapping 
                           if not m['exclude'] and m['target']}
            
            if not required_cols:
                raise ValueError("No valid columns found in mapping")
            
            source = pd.DataFrame()
            target = pd.DataFrame()
            
            for source_col, target_col in required_cols.items():
                try:
                    mapping = next(m for m in self.mapping if m['source'] == source_col)
                    mapped_type = mapping.get('data_type', 'string')
                    
                    source[source_col] = self.source_df[source_col].copy()
                    target[source_col] = self.target_df[target_col].copy()
                    
                    self._apply_type_conversion(source, target, source_col, mapped_type)
                    
                except Exception as e:
                    logger.error(f"Error processing column {source_col}: {str(e)}")
                    if source_col in source.columns:
                        source.drop(columns=[source_col], inplace=True)
                    if source_col in target.columns:
                        target.drop(columns=[source_col], inplace=True)
            
            return source, target
            
        except Exception as e:
            logger.error(f"Error in prepare_dataframes: {str(e)}")
            raise

    def compare(self, chunk_size: int = 50000) -> Dict[str, Any]:
        """Perform the comparison between source and target data."""
        try:
            source, target = self._prepare_dataframes()
            
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
                          join_columns: List[str], chunk_size: int = 50000) -> Dict[str, pd.DataFrame]:
        """Process large dataframes in chunks."""
        source_unmatched = []
        target_unmatched = []
        
        for i in range(0, len(source_df), chunk_size):
            source_chunk = source_df.iloc[i:i + chunk_size]
            merged = pd.merge(source_chunk, target_df, on=join_columns, how='outer', indicator=True)
            source_unmatched.append(merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1))
            target_unmatched.append(merged[merged['_merge'] == 'right_only'].drop('_merge', axis=1))
            del merged
            
        return {
            'source_unmatched': pd.concat(source_unmatched) if source_unmatched else pd.DataFrame(),
            'target_unmatched': pd.concat(target_unmatched) if target_unmatched else pd.DataFrame()
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
        
        Returns:
            List of dictionaries containing column mappings
        """
        source_cols = list(self.source_df.columns)
        target_cols = list(self.target_df.columns)
        mapping = []

        # Map source columns to target columns
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
                'editable': True,
                'original_source_type': original_source_type,
                'original_target_type': original_target_type
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

    def _generate_column_summary(self, source: pd.DataFrame, target: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Generate column-level summary statistics."""
        summary = {}
        
        for col in source.columns:
            if col in self.join_columns:
                continue
                
            try:
                summary[col] = {
                    'source_null_count': int(source[col].isnull().sum()),
                    'target_null_count': int(target[col].isnull().sum()),
                    'source_unique_count': int(source[col].nunique()),
                    'target_unique_count': int(target[col].nunique())
                }
                
                if np.issubdtype(source[col].dtype, np.number):
                    summary[col].update({
                        'source_sum': float(source[col].sum()),
                        'target_sum': float(target[col].sum()),
                        'source_mean': float(source[col].mean()),
                        'target_mean': float(target[col].mean())
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

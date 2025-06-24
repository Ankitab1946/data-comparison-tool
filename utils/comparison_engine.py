# """Core comparison engine for data comparison operations."""
# import pandas as pd
# import numpy as np
# from typing import Dict, List, Tuple, Any, Optional
# import logging
# from ydata_profiling import ProfileReport
# from pathlib import Path
# from flask import request

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ComparisonEngine:
#     # SQL Server to pandas type mapping
#     SQL_TYPE_MAPPING = {
#         'varchar': 'string',
#         'nvarchar': 'string',
#         'char': 'string',
#         'nchar': 'string',
#         'text': 'string',
#         'ntext': 'string',
#         'int': 'int32',
#         'bigint': 'int64',
#         'smallint': 'int32',
#         'tinyint': 'int32',
#         'numeric': 'float64',
#         'decimal': 'float64',
#         'float': 'float64',
#         'real': 'float32',
#         'bit': 'bool',
#         'date': 'datetime64[ns]',
#         'datetime': 'datetime64[ns]',
#         'datetime2': 'datetime64[ns]',
#         'datetimeoffset': 'datetime64[ns]',
#         'time': 'datetime64[ns]',
#         'binary': 'string',
#         'varbinary': 'string',
#         'uniqueidentifier': 'string',
#         'xml': 'string',
#         'unsupported': 'string'
#     }

#     # Pandas type mapping
#     PANDAS_TYPE_MAPPING = {
#         'object': 'string',
#         'string': 'string',
#         'int64': 'int64',
#         'int32': 'int32',
#         'float64': 'float64',
#         'float32': 'float32',
#         'bool': 'bool',
#         'datetime64[ns]': 'datetime64[ns]',
#         'category': 'string',
#         'unsupported': 'string'
#     }

#     def __init__(self, source_df: pd.DataFrame, target_df: pd.DataFrame):
#         """
#         Initialize the comparison engine with source and target dataframes.
        
#         Args:
#             source_df: Source DataFrame
#             target_df: Target DataFrame
#         """
#         self.source_df = source_df
#         self.target_df = target_df
#         self.mapping = None
#         self.join_columns = None
#         self.excluded_columns = []
#         self.type_overrides = {}  # Store user-defined type overrides

#     def set_column_type(self, column: str, dtype: str):
#         """
#         Set a specific data type for a column.
        
#         Args:
#             column: Column name
#             dtype: Data type to use for the column
#         """
#         self.type_overrides[column] = dtype

#     def _get_mapped_type(self, column: str, current_type: str, is_sql_type: bool = False) -> str:
#         """
#         Get the mapped data type for a column.
        
#         Args:
#             column: Column name
#             current_type: Current data type
#             is_sql_type: Whether the type is from SQL Server
            
#         Returns:
#             Mapped data type string
#         """
#         # Check for user override first
#         if column in self.type_overrides:
#             return self.type_overrides[column]
        
#         # Special handling for ColumnNameID
#         if column == 'ColumnNameID':
#             return 'string'
            
#         # Convert current type to lowercase and clean it
#         current_type = str(current_type).lower().strip()
#         # Remove length specifications like varchar(50)
#         current_type = current_type.split('(')[0]
        
#         # Handle SQL Server types
#         if is_sql_type:
#             # Try exact match first
#             if current_type in self.SQL_TYPE_MAPPING:
#                 return self.SQL_TYPE_MAPPING[current_type]
#             # Try partial match
#             for sql_type, pandas_type in self.SQL_TYPE_MAPPING.items():
#                 if sql_type in current_type:
#                     return pandas_type
        
#         # Handle Pandas types
#         else:
#             # Remove [ns] from datetime types
#             base_type = current_type.split('[')[0]
#             # Try exact match first
#             if base_type in self.PANDAS_TYPE_MAPPING:
#                 return self.PANDAS_TYPE_MAPPING[base_type]
#             # Try partial match
#             for pandas_type, mapped_type in self.PANDAS_TYPE_MAPPING.items():
#                 if pandas_type in base_type:
#                     return mapped_type
        
#         # If type contains 'char' or 'text', treat as string
#         if 'char' in current_type or 'text' in current_type:
#             return 'string'
            
#         # Default to string for unknown types
#         logger.warning(f"Unknown type '{current_type}' for column '{column}', defaulting to string")
#         return 'string'

#     def auto_map_columns(self) -> List[Dict[str, Any]]:
#         """
#         Automatically map columns between source and target based on names and data types.
        
#         Returns:
#             List of dictionaries containing column mappings
#         """
#         source_cols = list(self.source_df.columns)
#         target_cols = list(self.target_df.columns)
#         mapping = []

#         for s_col in source_cols:
#             # Try exact match first
#             t_col = s_col if s_col in target_cols else None
            
#             # If no exact match, try case-insensitive match
#             if not t_col:
#                 t_col = next((col for col in target_cols 
#                             if col.lower() == s_col.lower()), None)
            
#             # If still no match, try removing special characters
#             if not t_col:
#                 s_clean = ''.join(e.lower() for e in s_col if e.isalnum())
#                 t_col = next((col for col in target_cols 
#                             if ''.join(e.lower() for e in col if e.isalnum()) == s_clean), None)

#             # Get source and target types
#             source_type = str(self.source_df[s_col].dtype)
#             target_type = str(self.target_df[t_col].dtype) if t_col else 'unknown'
            
#             # Convert 'object' type to 'string'
#             if source_type == 'object':
#                 source_type = 'string'
#             if target_type == 'object':
#                 target_type = 'string'
            
#             # Check if either type is from SQL Server
#             is_sql_source = any(sql_type in source_type.lower() for sql_type in self.SQL_TYPE_MAPPING.keys())
#             is_sql_target = any(sql_type in target_type.lower() for sql_type in self.SQL_TYPE_MAPPING.keys())
            
#             # Get mapped types
#             source_mapped = self._get_mapped_type(s_col, source_type, is_sql_source)
#             target_mapped = self._get_mapped_type(s_col, target_type, is_sql_target) if t_col else source_mapped
            
#             # Store original types
#             original_source_type = source_type
#             original_target_type = target_type
            
#             # Determine final type
#             if source_mapped == target_mapped:
#                 mapped_type = source_mapped
#             else:
#                 # For mismatched types, prefer string for text-like columns
#                 if any(t in str(source_type).lower() or t in str(target_type).lower() 
#                       for t in ['char', 'text', 'string', 'object']):
#                     mapped_type = 'string'
#                     original_source_type = 'string'
#                     original_target_type = 'string'
#                 else:
#                     # For numeric types, use the wider type
#                     if all(t in ['int32', 'int64', 'float32', 'float64'] for t in [source_mapped, target_mapped]):
#                         mapped_type = 'float64'  # widest numeric type
#                     else:
#                         mapped_type = 'string'  # fallback for incompatible types
#                         original_source_type = 'string'
#                         original_target_type = 'string'

#             # Create mapping with editable types
#             mapping_entry = {
#                 'source': s_col,
#                 'target': t_col or '',
#                 'join': False,
#                 'data_type': mapped_type,
#                 'exclude': False,
#                 'source_type': original_source_type,
#                 'target_type': original_target_type,
#                 'editable': True,  # Flag to indicate type is editable
#                 'original_source_type': original_source_type,  # Keep original for reference
#                 'original_target_type': original_target_type   # Keep original for reference
#             }
            
#             # Special handling for memory-intensive numeric columns
#             if mapped_type in ['float64', 'int64']:
#                 # Check if column has too many unique values
#                 unique_count = min(
#                     self.source_df[s_col].nunique() if s_col in self.source_df else 0,
#                     self.target_df[t_col].nunique() if t_col in self.target_df else 0
#                 )
#                 if unique_count > 1000000:  # Threshold for large columns
#                     mapping_entry['data_type'] = 'string'
#                     mapping_entry['source_type'] = 'string'
#                     mapping_entry['target_type'] = 'string'
#                     logger.warning(f"Converting {s_col} to string due to high cardinality")
            
#             mapping.append(mapping_entry)

#         return mapping

#     def set_mapping(self, mapping: List[Dict[str, Any]], join_columns: List[str]):
#         """
#         Set the column mapping and join columns for comparison.
        
#         Args:
#             mapping: List of mapping dictionaries
#             join_columns: List of columns to use for joining
#         """
#         self.mapping = mapping
#         self.join_columns = join_columns
#         self.excluded_columns = [m['source'] for m in mapping if m['exclude']]
        
#         # Store original data types
#         self.source_types = {m['source']: m.get('source_type', '') for m in mapping}
#         self.target_types = {m['source']: m.get('target_type', '') for m in mapping}

#     def update_column_types(self, column: str, source_type: str = None, target_type: str = None):
#         """
#         Update the data types for a column.
        
#         Args:
#             column: Column name
#             source_type: New source data type
#             target_type: New target data type
#         """
#         if not self.mapping:
#             raise ValueError("Mapping must be set before updating types")
            
#         for m in self.mapping:
#             if m['source'] == column and m.get('editable', True):
#                 # Convert 'object' to 'string'
#                 if source_type == 'object':
#                     source_type = 'string'
#                 if target_type == 'object':
#                     target_type = 'string'
                
#                 # Update types
#                 if source_type:
#                     m['source_type'] = source_type
#                     self.source_types[column] = source_type
#                 if target_type:
#                     m['target_type'] = target_type
#                     self.target_types[column] = target_type
                
#                 # Determine new mapped type
#                 new_source_type = source_type or m['source_type']
#                 new_target_type = target_type or m['target_type']
                
#                 # Check for memory-intensive numeric types
#                 if new_source_type in ['float64', 'int64'] or new_target_type in ['float64', 'int64']:
#                     unique_count = min(
#                         self.source_df[column].nunique() if column in self.source_df else 0,
#                         self.target_df[column].nunique() if column in self.target_df else 0
#                     )
#                     if unique_count > 1000000:  # Threshold for large columns
#                         logger.warning(f"Converting {column} to string due to high cardinality")
#                         m['data_type'] = 'string'
#                         m['source_type'] = 'string'
#                         m['target_type'] = 'string'
#                         return
                
#                 # Update the mapped type
#                 m['data_type'] = self._get_mapped_type(column, new_source_type, True)
#                 break

#     def _process_in_chunks(self, source_df: pd.DataFrame, target_df: pd.DataFrame, 
#                           join_columns: List[str], chunk_size: int = 25000) -> Dict[str, pd.DataFrame]:
#         """
#         Process large dataframes in chunks to avoid memory issues.
#         Optimized for very large datasets (3GB+).
        
#         Args:
#             source_df: Source DataFrame
#             target_df: Target DataFrame
#             join_columns: Columns to join on
#             chunk_size: Size of each chunk
            
#         Returns:
#             Dictionary containing unmatched rows
#         """
#         try:
#             logger.info("Starting chunked comparison process...")
#             source_unmatched = []
#             target_unmatched = []
#             processed_target_chunks = set()
            
#             # Calculate total chunks for progress tracking
#             total_source_chunks = (len(source_df) + chunk_size - 1) // chunk_size
#             total_target_chunks = (len(target_df) + chunk_size - 1) // chunk_size
            
#             # Process source in chunks
#             for source_chunk_num in range(total_source_chunks):
#                 try:
#                     logger.info(f"Processing source chunk {source_chunk_num + 1}/{total_source_chunks}")
                    
#                     # Get chunk of source data
#                     start_idx = source_chunk_num * chunk_size
#                     end_idx = min(start_idx + chunk_size, len(source_df))
#                     source_chunk = source_df.iloc[start_idx:end_idx]
                    
#                     # Create a hash of join column values for this chunk
#                     source_keys = set(map(tuple, source_chunk[join_columns].values))
                    
#                     # Process target in chunks
#                     for target_chunk_num in range(total_target_chunks):
#                         if target_chunk_num in processed_target_chunks:
#                             continue
                            
#                         # Get chunk of target data
#                         start_idx = target_chunk_num * chunk_size
#                         end_idx = min(start_idx + chunk_size, len(target_df))
#                         target_chunk = target_df.iloc[start_idx:end_idx]
                        
#                         # Create a hash of join column values for target chunk
#                         target_keys = set(map(tuple, target_chunk[join_columns].values))
                        
#                         # Find unmatched in source
#                         source_unmatched_keys = source_keys - target_keys
#                         if source_unmatched_keys:
#                             mask = source_chunk[join_columns].apply(tuple, axis=1).isin(source_unmatched_keys)
#                             source_unmatched.append(source_chunk[mask])
                        
#                         # Find unmatched in target
#                         target_unmatched_keys = target_keys - source_keys
#                         if target_unmatched_keys:
#                             mask = target_chunk[join_columns].apply(tuple, axis=1).isin(target_unmatched_keys)
#                             target_unmatched.append(target_chunk[mask])
                        
#                         processed_target_chunks.add(target_chunk_num)
                        
#                         # Clear memory
#                         del target_chunk, target_keys
#                         import gc
#                         gc.collect()
                    
#                     # Clear memory after processing each source chunk
#                     del source_chunk, source_keys
#                     gc.collect()
                    
#                 except Exception as chunk_error:
#                     logger.error(f"Error processing chunks: {str(chunk_error)}")
#                     continue
            
#             # Combine results efficiently
#             logger.info("Combining results...")
#             result = {
#                 'source_unmatched': (pd.concat(source_unmatched, ignore_index=True, copy=False)
#                                    if source_unmatched else pd.DataFrame()),
#                 'target_unmatched': (pd.concat(target_unmatched, ignore_index=True, copy=False)
#                                    if target_unmatched else pd.DataFrame())
#             }
            
#             # Clear memory
#             del source_unmatched, target_unmatched, processed_target_chunks
#             gc.collect()
            
#             logger.info("Chunk processing completed successfully")
#             return result
            
#         except Exception as e:
#             logger.error(f"Error in chunk processing: {str(e)}")
#             return {
#                 'source_unmatched': pd.DataFrame(),
#                 'target_unmatched': pd.DataFrame(),
#                 'error': str(e)
#             }

#     def _prepare_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         """
#         Prepare dataframes for comparison by applying mappings and type conversions.
#         Memory-optimized version for handling large datasets (3GB+).
        
#         Returns:
#             Tuple of (prepared source DataFrame, prepared target DataFrame)
#         """
#         if not self.mapping:
#             raise ValueError("Mapping must be set before comparison")

#         try:
#             logger.info("Starting dataframe preparation...")
            
#             # Get required columns only
#             required_cols = {m['source']: m['target'] for m in self.mapping 
#                            if not m['exclude'] and m['target']}
            
#             if not required_cols:
#                 raise ValueError("No valid columns found in mapping")
            
#             # Get row counts and indices
#             source_len = len(self.source_df.index)
#             target_len = len(self.target_df.index)
#             logger.info(f"Initial row counts - Source: {source_len}, Target: {target_len}")
            
#             # Initialize dataframes with RangeIndex for consistent alignment
#             source = pd.DataFrame(index=pd.RangeIndex(source_len))
#             target = pd.DataFrame(index=pd.RangeIndex(target_len))
            
#             # Store original indices and log information
#             source._original_index = self.source_df.index.copy()
#             target._original_index = self.target_df.index.copy()
#             logger.info(f"Initialized dataframes - Source rows: {source_len}, Target rows: {target_len}")
            
#             # Process columns in smaller batches for better memory management
#             batch_size = 3  # Reduced batch size for large datasets
#             column_batches = [list(required_cols.items())[i:i + batch_size] 
#                             for i in range(0, len(required_cols), batch_size)]
            
#             # Process columns in batches
#             for batch_num, batch in enumerate(column_batches, 1):
#                 try:
#                     logger.info(f"Processing batch {batch_num}/{len(column_batches)}: {[col for col, _ in batch]}")
                    
#                     for source_col, target_col in batch:
#                         try:
#                             # Get mapping configuration
#                             mapping = next(m for m in self.mapping if m['source'] == source_col)
#                             mapped_type = mapping.get('data_type', 'string')
                            
#                             # Read and verify column lengths
#                             source_data = self.source_df[source_col].copy()
#                             target_data = self.target_df[target_col].copy()
                            
#                             if len(source_data) != source_len or len(target_data) != target_len:
#                                 raise ValueError(f"Column length mismatch for {source_col}")
                            
#                             # Assign data with index alignment
#                             source[source_col] = source_data
#                             target[source_col] = target_data
                            
#                             # Apply type conversions
#                             self._apply_type_conversion(source, target, source_col, mapped_type)
                            
#                             # Verify row counts after conversion
#                             if len(source[source_col]) != source_len or len(target[source_col]) != target_len:
#                                 raise ValueError(f"Row count changed after processing {source_col}")
                            
#                         except Exception as col_error:
#                             logger.error(f"Error processing column {source_col}: {str(col_error)}")
#                             # Remove problematic columns
#                             if source_col in source.columns:
#                                 source.drop(columns=[source_col], inplace=True)
#                             if source_col in target.columns:
#                                 target.drop(columns=[source_col], inplace=True)
                    
#                     # Force garbage collection after each batch
#                     import gc
#                     gc.collect()
                    
#                 except Exception as batch_error:
#                     logger.error(f"Error processing batch: {str(batch_error)}")
#                     continue
            
#             logger.info("Dataframe preparation completed successfully")
#             return source, target
            
#         except Exception as e:
#             logger.error(f"Error in prepare_dataframes: {str(e)}")
#             raise

#     def _apply_type_conversion(self, source: pd.DataFrame, target: pd.DataFrame, 
#                              col: str, mapped_type: str) -> None:
#         """
#         Apply type conversion to a column in both dataframes.
#         Optimized for memory usage and large datasets.
#         Ensures row counts are preserved during conversion.
        
#         Args:
#             source: Source DataFrame
#             target: Target DataFrame
#             col: Column name
#             mapped_type: Target data type
#         """
#         try:
#             # Store original row counts
#             source_len = len(source)
#             target_len = len(target)
            
#             # Create temporary Series to avoid modifying original data
#             source_temp = source[col].copy()
#             target_temp = target[col].copy()
            
#             if mapped_type == 'string' or 'char' in str(source[col].dtype).lower():
#                 source_temp = source_temp.fillna('').astype('string')
#                 target_temp = target_temp.fillna('').astype('string')
                
#             elif mapped_type in ['int32', 'int64']:
#                 try:
#                     # Convert to numeric while preserving NaN
#                     source_temp = pd.to_numeric(source_temp, errors='coerce')
#                     target_temp = pd.to_numeric(target_temp, errors='coerce')
                    
#                     # Fill NaN with 0 and convert to int64
#                     source_temp = source_temp.fillna(0).astype(np.int64)
#                     target_temp = target_temp.fillna(0).astype(np.int64)
#                 except Exception as e:
#                     logger.warning(f"Integer conversion failed for {col}: {str(e)}. Converting to string.")
#                     source_temp = source_temp.fillna('').astype('string')
#                     target_temp = target_temp.fillna('').astype('string')
                    
#             elif mapped_type in ['float32', 'float64']:
#                 try:
#                     source_temp = pd.to_numeric(source_temp, errors='coerce')
#                     target_temp = pd.to_numeric(target_temp, errors='coerce')
#                     source_temp = source_temp.fillna(0).astype(np.float64)
#                     target_temp = target_temp.fillna(0).astype(np.float64)
#                 except Exception as e:
#                     logger.warning(f"Float conversion failed for {col}: {str(e)}. Converting to string.")
#                     source_temp = source_temp.fillna('').astype('string')
#                     target_temp = target_temp.fillna('').astype('string')
                    
#             elif mapped_type == 'datetime64[ns]':
#                 source_temp = pd.to_datetime(source_temp, errors='coerce')
#                 target_temp = pd.to_datetime(target_temp, errors='coerce')
                
#             elif mapped_type == 'bool':
#                 try:
#                     # Convert to boolean safely
#                     source_temp = source_temp.map({'True': True, 'False': False, True: True, False: False, 
#                                                1: True, 0: False}).fillna(False)
#                     target_temp = target_temp.map({'True': True, 'False': False, True: True, False: False, 
#                                                1: True, 0: False}).fillna(False)
                    
#                     # Ensure numpy boolean type
#                     source_temp = source_temp.astype(np.bool_)
#                     target_temp = target_temp.astype(np.bool_)
#                 except Exception as e:
#                     logger.warning(f"Boolean conversion failed for {col}: {str(e)}. Converting to string.")
#                     source_temp = source_temp.fillna('').astype('string')
#                     target_temp = target_temp.fillna('').astype('string')
#             else:
#                 # Default to string for unknown types
#                 source_temp = source_temp.fillna('').astype('string')
#                 target_temp = target_temp.fillna('').astype('string')
            
#             # Verify row counts are preserved
#             if len(source_temp) != source_len or len(target_temp) != target_len:
#                 raise ValueError(f"Row count changed during conversion of {col}")
            
#             # Only update the original dataframes if conversion was successful
#             source[col] = source_temp
#             target[col] = target_temp
            
#             # Verify final row counts
#             if len(source[col]) != source_len or len(target[col]) != target_len:
#                 raise ValueError(f"Final row count mismatch for {col}")
                
#         except Exception as e:
#             logger.error(f"Error in type conversion for column {col}: {str(e)}")
#             # Ensure columns are removed if conversion fails
#             if col in source.columns:
#                 source.drop(columns=[col], inplace=True)
#             if col in target.columns:
#                 target.drop(columns=[col], inplace=True)
#             raise  # Re-raise the exception to handle it in the calling method
            

#     def compare(self, chunk_size: int = 50000) -> Dict[str, Any]:
#         """
#         Perform the comparison between source and target data.
        
#         Args:
#             chunk_size: Size of chunks for processing large datasets
            
#         Returns:
#             Dictionary containing comparison results
#         """
#         try:
#             source, target = self._prepare_dataframes()
            
#             # Initialize comparison results with optimized data handling
#             try:
#                 # Initialize comparison with robust row counting
#                 logger.info("Initializing comparison...")
                
#                 try:
#                     # Get row counts using multiple methods for verification
#                     source_len_index = len(source.index)
#                     target_len_index = len(target.index)
#                     source_len_shape = source.shape[0]
#                     target_len_shape = target.shape[0]
                    
#                     # Verify counts match between methods
#                     if source_len_index != source_len_shape or target_len_index != target_len_shape:
#                         logger.warning("Row count mismatch between index and shape methods")
#                         # Use the larger count to be safe
#                         source_len = max(source_len_index, source_len_shape)
#                         target_len = max(target_len_index, target_len_shape)
#                     else:
#                         source_len = source_len_index
#                         target_len = target_len_index
                    
#                     logger.info(f"Source row count methods - index: {source_len_index}, shape: {source_len_shape}")
#                     logger.info(f"Target row count methods - index: {target_len_index}, shape: {target_len_shape}")
                    
#                     # Initialize results with verified counts
#                     results = {
#                         'match_status': False,
#                         'rows_match': False,
#                         'columns_match': False,
#                         'datacompy_report': '',
#                         'source_unmatched_rows': pd.DataFrame(),
#                         'target_unmatched_rows': pd.DataFrame(),
#                         'column_summary': {},  # Will be populated later
#                         'row_counts': {
#                             'source_name': 'Source',
#                             'target_name': 'Target',
#                             'source_count': int(np.int64(source_len)),
#                             'target_count': int(np.int64(target_len))
#                         },
#                         'distinct_values': {}
#                     }
                    
#                     # Verify row alignment and counts
#                     source_aligned = source.index.equals(source._original_index)
#                     target_aligned = target.index.equals(target._original_index)
                    
#                     if not source_aligned or not target_aligned:
#                         logger.error("Index alignment check failed")
#                         if not source_aligned:
#                             logger.error("Source index mismatch")
#                         if not target_aligned:
#                             logger.error("Target index mismatch")
#                         results['rows_match'] = False
#                     else:
#                         # Compare row counts only if indices are aligned
#                         results['rows_match'] = (source_len == target_len)
#                         logger.info(f"Row counts - Source: {source_len}, Target: {target_len}, Match: {results['rows_match']}")
#                         logger.info("Index alignment verified")
                    
#                 except Exception as e:
#                     logger.error(f"Error in row counting: {str(e)}")
#                     raise ValueError(f"Failed to verify row counts: {str(e)}")
                
#                 # Free memory
#                 import gc
#                 gc.collect()
                
#                 try:
#                     # Reset index to ensure consistent row alignment
#                     source = source.reset_index(drop=True)
#                     target = target.reset_index(drop=True)
                    
#                     logger.info("Starting comparison with reset indices...")
                    
#                     # Generate column summary with smaller chunks
#                     results['column_summary'] = self._generate_column_summary_chunked(source, target, chunk_size=25000)
                    
#                     # Free memory after column summary
#                     gc.collect()
                    
#                     # Get and verify row counts
#                     source_len = len(source)
#                     target_len = len(target)
#                     source_shape = source.shape[0]
#                     target_shape = target.shape[0]
                    
#                     if source_len != source_shape or target_len != target_shape:
#                         raise ValueError(f"Row count mismatch - Source: len={source_len}, shape={source_shape}; Target: len={target_len}, shape={target_shape}")
                    
#                     # Store verified row counts
#                     results['row_counts'] = {
#                         'source_name': 'Source',
#                         'target_name': 'Target',
#                         'source_count': int(np.int64(source_len)),
#                         'target_count': int(np.int64(target_len))
#                     }
                    
#                     logger.info(f"Row counts verified - Source: {source_len}, Target: {target_len}")
                    
#                     # Set match status based on verified counts
#                     results['rows_match'] = (source_len == target_len)
#                     results['columns_match'] = set(source.columns) == set(target.columns)
                    
#                     # Log comparison status
#                     logger.info(f"Comparison status - Rows match: {results['rows_match']}, Columns match: {results['columns_match']}")
                    
#                 except Exception as e:
#                     logger.error(f"Error in comparison initialization: {str(e)}")
#                     raise ValueError(f"Comparison initialization failed: {str(e)}")
                
#                 # Process unmatched rows with index verification
#                 if join_columns:
#                     try:
#                         logger.info("Processing unmatched rows...")
                        
#                         # Verify indices are still aligned
#                         if not source.index.equals(pd.RangeIndex(len(source))) or not target.index.equals(pd.RangeIndex(len(target))):
#                             logger.warning("Index alignment lost, resetting indices...")
#                             source = source.reset_index(drop=True)
#                             target = target.reset_index(drop=True)
                        
#                         # Process chunks with verified indices
#                         unmatched = self._process_in_chunks(source, target, join_columns, chunk_size=25000)
                        
#                         if isinstance(unmatched, dict):
#                             results['source_unmatched_rows'] = unmatched.get('source_unmatched', pd.DataFrame())
#                             results['target_unmatched_rows'] = unmatched.get('target_unmatched', pd.DataFrame())
                            
#                             # Log unmatched counts
#                             logger.info(f"Found {len(results['source_unmatched_rows'])} unmatched source rows")
#                             logger.info(f"Found {len(results['target_unmatched_rows'])} unmatched target rows")
                            
#                             # Verify unmatched results have proper indices
#                             if not results['source_unmatched_rows'].empty and not results['source_unmatched_rows'].index.is_monotonic_increasing:
#                                 results['source_unmatched_rows'] = results['source_unmatched_rows'].sort_index()
#                             if not results['target_unmatched_rows'].empty and not results['target_unmatched_rows'].index.is_monotonic_increasing:
#                                 results['target_unmatched_rows'] = results['target_unmatched_rows'].sort_index()
#                         else:
#                             logger.error("Invalid unmatched results structure")
#                             results['source_unmatched_rows'] = pd.DataFrame()
#                             results['target_unmatched_rows'] = pd.DataFrame()
                            
#                     except Exception as e:
#                         logger.error(f"Error processing unmatched rows: {str(e)}")
#                         results['source_unmatched_rows'] = pd.DataFrame()
#                         results['target_unmatched_rows'] = pd.DataFrame()
                    
#                     # Final verification of results
#                     logger.info("Verifying final comparison results...")
                    
#                     # Ensure all required keys exist
#                     if not all(key in results for key in ['rows_match', 'columns_match', 'source_unmatched_rows', 'target_unmatched_rows']):
#                         raise KeyError("Missing required keys in results dictionary")
                    
#                     # Verify row matching
#                     if not isinstance(results['rows_match'], bool):
#                         results['rows_match'] = bool(results['rows_match'])
                    
#                     # Verify column matching
#                     if not isinstance(results['columns_match'], bool):
#                         results['columns_match'] = bool(results['columns_match'])
                    
#                     # Verify unmatched rows are DataFrames
#                     if not isinstance(results['source_unmatched_rows'], pd.DataFrame):
#                         results['source_unmatched_rows'] = pd.DataFrame()
#                     if not isinstance(results['target_unmatched_rows'], pd.DataFrame):
#                         results['target_unmatched_rows'] = pd.DataFrame()
                    
#                     # Set final match status
#                     results['match_status'] = bool(
#                         results['rows_match'] and 
#                         results['columns_match'] and 
#                         results['source_unmatched_rows'].empty and 
#                         results['target_unmatched_rows'].empty
#                     )
                    
#                     logger.info(f"Final comparison status: {results['match_status']}")
                    
#                 except Exception as e:
#                     logger.error(f"Error in comparison verification: {str(e)}")
#                     results.update({
#                         'match_status': False,
#                         'rows_match': False,
#                         'error': f"Comparison verification failed: {str(e)}"
#                     })
                
#             except Exception as e:
#                 error_msg = f"Error initializing comparison: {str(e)}"
#                 logger.error(error_msg)
                
#                 # Initialize error results with proper structure
#                 results = {
#                     'match_status': False,
#                     'rows_match': False,
#                     'columns_match': False,
#                     'datacompy_report': error_msg,
#                     'source_unmatched_rows': pd.DataFrame(),
#                     'target_unmatched_rows': pd.DataFrame(),
#                     'column_summary': {},
#                     'row_counts': {
#                         'source_name': 'Source',
#                         'target_name': 'Target',
#                         'source_count': 0,
#                         'target_count': 0
#                     },
#                     'distinct_values': {},
#                     'error': error_msg
#                 }
#                 return results  # Return error results

#             # Get distinct values
#             results['distinct_values'] = {}  # Initialize empty dict
#             try:
#                 distinct_values = self.get_distinct_values()
#                 if isinstance(distinct_values, dict):
#                     results['distinct_values'] = distinct_values
#             except Exception as e:
#                 logger.error(f"Error getting distinct values: {str(e)}")

#             # Process unmatched rows if join columns exist
#             if self.join_columns:
#                 # Initialize unmatched results
#                 results['source_unmatched_rows'] = pd.DataFrame()
#                 results['target_unmatched_rows'] = pd.DataFrame()
                
#                 try:
#                     unmatched = self._process_in_chunks(source, target, self.join_columns, chunk_size)
#                     if isinstance(unmatched, dict):
#                         results['source_unmatched_rows'] = unmatched.get('source_unmatched', pd.DataFrame())
#                         results['target_unmatched_rows'] = unmatched.get('target_unmatched', pd.DataFrame())
#                 except Exception as e:
#                     logger.error(f"Error processing unmatched rows: {str(e)}")
                    
#                 # Generate comparison report with proper sections
#                 try:
#                     # Initialize report with header and summary
#                     report_lines = [
#                         "DataCompy Comparison Report",
#                         "=" * 50,
#                         "\nSummary:",
#                         "-" * 20,
#                         f"Source rows: {len(source)}",
#                         f"Target rows: {len(target)}",
#                         f"Unmatched in source: {len(results['source_unmatched_rows'])}",
#                         f"Unmatched in target: {len(results['target_unmatched_rows'])}",
#                         "\nColumn Analysis:",
#                         "-" * 20
#                     ]
#                     # Add column-level analysis
#                     for col in source.columns:
#                         if col in results['column_summary']:
#                             summary = results['column_summary'][col]
#                             col_lines = [
#                                 f"\nColumn: {col}",
#                                 f"Source null count: {summary['source_null_count']}",
#                                 f"Target null count: {summary['target_null_count']}",
#                                 f"Source unique count: {summary['source_unique_count']}",
#                                 f"Target unique count: {summary['target_unique_count']}"
#                             ]
                            
#                             # Add numeric column statistics if available
#                             if 'source_sum' in summary:
#                                 col_lines.extend([
#                                     f"Source sum: {summary['source_sum']}",
#                                     f"Target sum: {summary['target_sum']}",
#                                     f"Source mean: {summary['source_mean']}",
#                                     f"Target mean: {summary['target_mean']}"
#                                 ])
                            
#                             report_lines.extend(col_lines)
                    
#                     # Add join columns analysis section
#                     report_lines.extend([
#                         "\nJoin Columns Analysis:",
#                         "-" * 20
#                     ])
                    
#                     if results['distinct_values'] and self.join_columns:
#                         for col in self.join_columns:
#                             if col in results['distinct_values']:
#                                 col_data = results['distinct_values'][col]
#                                 s_vals = list(col_data['source_values'].items())[:5]
#                                 t_vals = list(col_data['target_values'].items())[:5]
                                
#                                 join_col_lines = [
#                                     f"\nJoin Column: {col}",
#                                     f"Source unique values: {col_data['source_count']}",
#                                     f"Target unique values: {col_data['target_count']}",
#                                     "Sample values comparison:",
#                                     "Source top 5: " + ", ".join(f"{v}({c})" for v, c in s_vals),
#                                     "Target top 5: " + ", ".join(f"{v}({c})" for v, c in t_vals)
#                                 ]
                                
#                                 report_lines.extend(join_col_lines)
                    
#                     # Add unmatched rows samples
#                     if not results['source_unmatched_rows'].empty:
#                         report_lines.extend([
#                             "\nSample Unmatched Rows in Source:",
#                             "-" * 20,
#                             results['source_unmatched_rows'].head(5).to_string()
#                         ])
                    
#                     if not results['target_unmatched_rows'].empty:
#                         report_lines.extend([
#                             "\nSample Unmatched Rows in Target:",
#                             "-" * 20,
#                             results['target_unmatched_rows'].head(5).to_string()
#                         ])
                    
#                     # Finalize report and set match status
#                     try:
#                         # Join report lines
#                         results['datacompy_report'] = "\n".join(report_lines)
                        
#                         # Set overall match status
#                         results['match_status'] = bool(
#                             results['columns_match'] and 
#                             results['rows_match'] and
#                             results['source_unmatched_rows'].empty and
#                             results['target_unmatched_rows'].empty
#                         )
                        
#                         logger.info(f"Comparison completed - Match status: {results['match_status']}")
#                     except Exception as e:
#                         logger.error(f"Error finalizing comparison report: {str(e)}")
#                         results.update({
#                             'datacompy_report': f"Error in comparison: {str(e)}",
#                             'match_status': False
#                         })

#             return results
#         except Exception as e:
#             error_msg = f"Comparison failed: {str(e)}"
#             logger.error(error_msg)
#             return {
#                 'match_status': False,
#                 'rows_match': False,
#                 'columns_match': False,
#                 'source_unmatched_rows': pd.DataFrame(),
#                 'target_unmatched_rows': pd.DataFrame(),
#                 'column_summary': {},
#                 'distinct_values': {},
#                 'row_counts': {
#                     'source_name': 'Source',
#                     'target_name': 'Target',
#                     'source_count': 0,
#                     'target_count': 0
#                 },
#                 'datacompy_report': error_msg,
#                 'error': error_msg
#             }

#     def _generate_column_summary(self, source: pd.DataFrame, target: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
#         """
#         Generate detailed column-level comparison summary with improved error handling.
        
#         Args:
#             source: Prepared source DataFrame
#             target: Prepared target DataFrame
            
#         Returns:
#             Dictionary containing column-level statistics
#         """
#         summary = {}
        
#         try:
#             for col in source.columns:
#                 if col in self.join_columns:
#                     continue
                    
#                 try:
#                     # Initialize basic column statistics
#                     col_stats = {
#                         'source_null_count': int(source[col].isnull().sum()),
#                         'target_null_count': int(target[col].isnull().sum()),
#                         'source_unique_count': int(source[col].nunique()),
#                         'target_unique_count': int(target[col].nunique())
#                     }
                    
#                     # Add numeric statistics if applicable
#                     if np.issubdtype(source[col].dtype, np.number):
#                         numeric_stats = {
#                             'source_sum': float(source[col].sum()),
#                             'target_sum': float(target[col].sum()),
#                             'source_mean': float(source[col].mean()),
#                             'target_mean': float(target[col].mean()),
#                             'source_std': float(source[col].std()),
#                             'target_std': float(target[col].std())
#                         }
#                         col_stats.update(numeric_stats)
                    
#                     summary[col] = col_stats
                    
#                 except Exception as col_error:
#                     logger.warning(f"Error processing column {col}: {str(col_error)}")
#                     summary[col] = {
#                         'source_null_count': 0,
#                         'target_null_count': 0,
#                         'source_unique_count': 0,
#                         'target_unique_count': 0,
#                         'error': str(col_error)
#                     }
            
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error generating column summary: {str(e)}")
#             return {}

#     def _generate_column_summary_chunked(self, source: pd.DataFrame, target: pd.DataFrame, 
#                                        chunk_size: int = 50000) -> Dict[str, Dict[str, Any]]:
#         """
#         Generate detailed column-level comparison summary using chunked processing.
#         Memory-optimized version for large datasets.
        
#         Args:
#             source: Prepared source DataFrame
#             target: Prepared target DataFrame
#             chunk_size: Size of chunks for processing
            
#         Returns:
#             Dictionary containing column-level statistics
#         """
#         try:
#             summary = {}
#             logger.info("Starting chunked column summary generation...")
#             total_columns = len([col for col in source.columns if col not in self.join_columns])
            
#             for col_idx, col in enumerate(source.columns, 1):
#                 if col in self.join_columns:
#                     continue
                
#                 logger.info(f"Processing column {col_idx}/{total_columns}: {col}")
            
#                 # Initialize column summary with proper types
#                 try:
#                     col_summary = {
#                         'source_null_count': np.int64(0),
#                         'target_null_count': np.int64(0),
#                         'source_unique_count': np.int64(0),
#                         'target_unique_count': np.int64(0)
#                     }
                    
#                     # Add numeric statistics if applicable
#                     if np.issubdtype(source[col].dtype, np.number):
#                         col_summary.update({
#                             'source_sum': np.float64(0),
#                             'target_sum': np.float64(0),
#                             'source_mean': np.float64(0),
#                             'target_mean': np.float64(0),
#                             'source_std': np.float64(0),
#                             'target_std': np.float64(0)
#                         })
#                         logger.debug(f"Added numeric statistics for column {col}")
#                 except Exception as e:
#                     logger.error(f"Error initializing summary for column {col}: {str(e)}")
#                     continue

#                 # Process column in chunks
#                 try:
#                     unique_values_source = set()
#                     unique_values_target = set()
#                     source_values = []
#                     target_values = []
                    
#                     # Calculate total chunks
#                     total_source_chunks = (len(source) + chunk_size - 1) // chunk_size
#                     logger.info(f"Processing {total_source_chunks} chunks for column {col}")
                    
#                     for chunk_start in range(0, len(source), chunk_size):
#                         chunk_end = min(chunk_start + chunk_size, len(source))
#                         source_chunk = source.iloc[chunk_start:chunk_end]
#                         target_chunk = target.iloc[chunk_start:chunk_end]
                        
#                         # Update statistics
#                         col_summary['source_null_count'] += np.int64(source_chunk[col].isnull().sum())
#                         col_summary['target_null_count'] += np.int64(target_chunk[col].isnull().sum())
                        
#                         # Update unique values
#                         unique_values_source.update(source_chunk[col].dropna().unique())
#                         unique_values_target.update(target_chunk[col].dropna().unique())
                        
#                         # Update numeric statistics
#                         if np.issubdtype(source[col].dtype, np.number):
#                             source_values.extend(source_chunk[col].dropna().tolist())
#                             target_values.extend(target_chunk[col].dropna().tolist())
                    
#                     # Finalize statistics
#                     col_summary['source_unique_count'] = np.int64(len(unique_values_source))
#                     col_summary['target_unique_count'] = np.int64(len(unique_values_target))
                    
#                     if np.issubdtype(source[col].dtype, np.number) and source_values:
#                         source_arr = np.array(source_values, dtype=np.float64)
#                         target_arr = np.array(target_values, dtype=np.float64)
                        
#                         col_summary.update({
#                             'source_sum': float(np.sum(source_arr)),
#                             'target_sum': float(np.sum(target_arr)),
#                             'source_mean': float(np.mean(source_arr)),
#                             'target_mean': float(np.mean(target_arr)),
#                             'source_std': float(np.std(source_arr)),
#                             'target_std': float(np.std(target_arr))
#                         })
                    
#                     summary[col] = col_summary
#                     logger.info(f"Completed processing column {col}")
                    
#                 except Exception as e:
#                     logger.error(f"Error processing chunks for column {col}: {str(e)}")
#                     # Initialize error state with basic counters
#                     error_summary = {
#                         'source_null_count': np.int64(0),
#                         'target_null_count': np.int64(0),
#                         'source_unique_count': np.int64(0),
#                         'target_unique_count': np.int64(0),
#                         'error': str(e)
#                     }
                    
#                     # Add numeric stats if it was a numeric column
#                     if np.issubdtype(source[col].dtype, np.number):
#                         error_summary.update({
#                             'source_sum': np.float64(0),
#                             'target_sum': np.float64(0),
#                             'source_mean': np.float64(0),
#                             'target_mean': np.float64(0),
#                             'source_std': np.float64(0),
#                             'target_std': np.float64(0)
#                         })
                    
#                     summary[col] = error_summary
                
#                 # Force garbage collection after each column
#                 gc.collect()
                
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                
#             # Return the final summary
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
#             # Return the final summary
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                        
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                        
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                        
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                        
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                            
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                    
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                    
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                            
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                            
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                    
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                    
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                    
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
                
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}
#             # Return the final summary with proper error handling
#             logger.info("Completed chunked column summary generation")
#             return summary
            
#         except Exception as e:
#             logger.error(f"Error in chunked column summary generation: {str(e)}")
#             return {}

#     def generate_profiling_reports(self, output_dir: str) -> Dict[str, str]:
#         """
#         Generate YData Profiling reports for source and target data.
        
#         Args:
#             output_dir: Directory to save the reports
            
#         Returns:
#             Dictionary containing paths to generated reports
#         """
#         try:
#             output_path = Path(output_dir)
#             output_path.mkdir(parents=True, exist_ok=True)

#             # Create copies of dataframes to avoid modifying originals
#             source_df = self.source_df.copy()
#             target_df = self.target_df.copy()

#             # Convert problematic columns to string
#             for df in [source_df, target_df]:
#                 for col in df.columns:
#                     # Check if column type is problematic
#                     col_type = str(df[col].dtype).lower()
#                     if ('object' in col_type or 
#                         'unsupported' in col_type or 
#                         col == 'ColumnNameID' or
#                         any(t in col_type for t in ['char', 'text'])):
#                         df[col] = df[col].fillna('').astype(str)

#             # Generate individual profiles with memory optimization
#             profile_kwargs = {
#                 'progress_bar': False,
#                 'explorative': True,
#                 'minimal': True,  # Reduce memory usage
#                 'pool_size': 1,   # Reduce parallel processing
#                 'samples': None   # Disable sample generation
#             }

#             try:
#                 source_profile = ProfileReport(
#                     source_df, 
#                     title="Source Data Profile",
#                     **profile_kwargs
#                 )
#             except Exception as e:
#                 logger.error(f"Error generating source profile: {str(e)}")
#                 # Try with more aggressive memory optimization
#                 source_profile = ProfileReport(
#                     source_df.astype(str),
#                     title="Source Data Profile",
#                     minimal=True,
#                     pool_size=1,
#                     samples=None,
#                     progress_bar=False
#                 )

#             try:
#                 target_profile = ProfileReport(
#                     target_df,
#                     title="Target Data Profile",
#                     **profile_kwargs
#                 )
#             except Exception as e:
#                 logger.error(f"Error generating target profile: {str(e)}")
#                 # Try with more aggressive memory optimization
#                 target_profile = ProfileReport(
#                     target_df.astype(str),
#                     title="Target Data Profile",
#                     minimal=True,
#                     pool_size=1,
#                     samples=None,
#                     progress_bar=False
#                 )

#             # Save reports
#             source_path = output_path / "source_profile.html"
#             target_path = output_path / "target_profile.html"
#             comparison_path = output_path / "comparison_profile.html"

#             source_profile.to_file(str(source_path))
#             target_profile.to_file(str(target_path))
            
#             # Generate comparison report with error handling
#             try:
#                 comparison_report = source_profile.compare(target_profile)
#                 comparison_report.to_file(str(comparison_path))
#             except Exception as e:
#                 logger.error(f"Error generating comparison report: {str(e)}")
#                 # Create a basic comparison report
#                 with open(str(comparison_path), 'w') as f:
#                     f.write("""
#                     <html>
#                     <head><title>Data Comparison Report</title></head>
#                     <body>
#                         <h1>Data Comparison Report</h1>
#                         <p>Error generating detailed comparison report. Please check individual profiles.</p>
#                         <ul>
#                             <li><a href="source_profile.html">Source Profile</a></li>
#                             <li><a href="target_profile.html">Target Profile</a></li>
#                         </ul>
#                     </body>
#                     </html>
#                     """)

#             return {
#                 'source_profile': str(source_path),
#                 'target_profile': str(target_path),
#                 'comparison_profile': str(comparison_path)
#             }
            
#         except Exception as e:
#             logger.error(f"Error in generate_profiling_reports: {str(e)}")
#             raise

#     def get_distinct_values(self, columns: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
#         """
#         Get distinct values and their counts for specified columns.
        
#         Args:
#             columns: List of columns to analyze. If None, analyze all non-numeric columns.
            
#         Returns:
#             Dictionary containing distinct values and counts for each column
#         """
#         try:
#             source, target = self._prepare_dataframes()
            
#             if not columns:
#                 # Get all columns that exist in both dataframes
#                 columns = [col for col in source.columns 
#                         if col in target.columns and not np.issubdtype(source[col].dtype, np.number)]
            
#             if not columns:  # If still no columns, return empty dict
#                 return {}

#             distinct_values = {}
#             for col in columns:
#                 try:
#                     if col in source.columns and col in target.columns:
#                         source_distinct = source[col].value_counts().to_dict()
#                         target_distinct = target[col].value_counts().to_dict()
                        
#                         distinct_values[col] = {
#                             'source_values': source_distinct,
#                             'target_values': target_distinct,
#                             'source_count': len(source_distinct),
#                             'target_count': len(target_distinct),
#                             'matching': set(source_distinct.keys()) == set(target_distinct.keys())
#                         }
#                 except Exception as e:
#                     logger.warning(f"Error processing column {col}: {str(e)}")
#                     continue

#             return distinct_values
#         except Exception as e:
#             logger.error(f"Error in get_distinct_values: {str(e)}")
#             return {}
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
        Includes unmapped source and target columns with blank mappings.
        
        Returns:
            List of dictionaries containing column mappings
        """
        source_cols = list(self.source_df.columns)
        target_cols = list(self.target_df.columns)
        mapping = []

        mapped_targets = set()

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

            if t_col:
                mapped_targets.add(t_col)

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

        # Add unmapped target columns with blank source
        for t_col in target_cols:
            if t_col not in mapped_targets:
                mapping_entry = {
                    'source': '',
                    'target': t_col,
                    'join': False,
                    'data_type': 'string',
                    'exclude': False,
                    'source_type': '',
                    'target_type': str(self.target_df[t_col].dtype),
                    'editable': True,
                    'original_source_type': '',
                    'original_target_type': str(self.target_df[t_col].dtype)
                }
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
                    # Special handling for Feed ID column - convert blank to <NA>
                    if source_col == 'Feed_ID':
                        # Convert blank and whitespace-only cells to <NA>
                        source[source_col] = source[source_col].replace(r'^\s*$', '<NA>', regex=True)
                        target[source_col] = target[source_col].replace(r'^\s*$', '<NA>', regex=True)
                        # Also convert NaN and None to <NA>
                        source[source_col] = source[source_col].fillna('<NA>').astype('string')
                        target[source_col] = target[source_col].fillna('<NA>').astype('string')
                    # Handle date columns - standardize format with better parsing
                    elif mapped_type == 'datetime64[ns]':
                        # First convert to datetime with flexible parsing
                        source[source_col] = pd.to_datetime(source[source_col], errors='coerce', infer_datetime_format=True)
                        target[source_col] = pd.to_datetime(target[source_col], errors='coerce', infer_datetime_format=True)
                        
                        # Then convert to string in a consistent format
                        source[source_col] = source[source_col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else '')
                        target[source_col] = target[source_col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else '')
                    elif mapped_type == 'string' or 'char' in str(source[source_col].dtype).lower():
                        source[source_col] = source[source_col].fillna('').astype('string')  # Use pandas string type
                        target[source_col] = target[source_col].fillna('').astype('string')
                    elif mapped_type in ['int32', 'int64']:
                        # Always use int64 for integer columns to avoid dtype mismatches
                        try:
                            # First convert to float to handle NaN values
                            source[source_col] = pd.to_numeric(source[source_col], errors='coerce')
                            target[source_col] = pd.to_numeric(target[source_col], errors='coerce')
                            
                            # Check cardinality
                            unique_count = min(
                                source[source_col].nunique(),
                                target[source_col].nunique()
                            )
                            
                            if unique_count > 1000000:  # High cardinality
                                source[source_col] = source[source_col].fillna('').astype('string')
                                target[source_col] = target[source_col].fillna('').astype('string')
                                logger.warning(f"Converting {source_col} to string due to high cardinality")
                            else:
                                # Fill NaN with 0 and convert to int64
                                source[source_col] = source[source_col].fillna(0).astype(np.int64)
                                target[source_col] = target[source_col].fillna(0).astype(np.int64)
                        except Exception as e:
                            logger.warning(f"Integer conversion failed for {source_col}: {str(e)}. Converting to string.")
                            source[source_col] = source[source_col].fillna('').astype('string')
                            target[source_col] = target[source_col].fillna('').astype('string')
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
                
                # Initialize results with proper structure and type conversion
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
                        'source_count': int(np.int64(source_count)),
                        'target_count': int(np.int64(target_count))
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
                # Initialize error results with proper type conversion
                error_results = {
                    'match_status': False,
                    'error': str(e),
                    'datacompy_report': f"Error initializing comparison: {str(e)}",
                    'row_counts': {
                        'source_name': 'Source',
                        'target_name': 'Target',
                        'source_count': int(np.int64(0)),
                        'target_count': int(np.int64(0))
                    },
                    'column_summary': {},
                    'distinct_values': {},
                    'source_unmatched_rows': pd.DataFrame(),
                    'target_unmatched_rows': pd.DataFrame()
                }
                return error_results

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
                # Initialize summary for this column with np.int64
                col_summary = {
                    'source_null_count': np.int64(0),
                    'target_null_count': np.int64(0),
                    'source_unique_count': np.int64(0),
                    'target_unique_count': np.int64(0)
                }
                
                # Initialize numeric statistics with np.float64
                if np.issubdtype(source[col].dtype, np.number):
                    col_summary.update({
                        'source_sum': np.float64(0),
                        'target_sum': np.float64(0),
                        'source_mean': np.float64(0),
                        'target_mean': np.float64(0),
                        'source_std': np.float64(0),
                        'target_std': np.float64(0)
                    })
                
                # Process source data in chunks
                unique_values_source = set()
                source_sum = 0.0
                source_values = []
                
                # Determine if column is integer type
                is_integer = np.issubdtype(source[col].dtype, np.integer)
                
                for i in range(0, len(source), chunk_size):
                    chunk = source.iloc[i:i + chunk_size]
                    col_summary['source_null_count'] += np.int64(chunk[col].isnull().sum())
                    
                    # Handle unique values based on dtype
                    if np.issubdtype(chunk[col].dtype, np.number):
                        non_null = chunk[col].dropna()
                        if is_integer:
                            values = non_null.astype(np.int64)
                            unique_values_source.update(values.tolist())
                            source_values.extend(values.tolist())
                        else:
                            values = non_null.astype(np.float64)
                            unique_values_source.update(values.tolist())
                            source_values.extend(values.tolist())
                        source_sum += np.float64(non_null.sum())
                    else:
                        unique_values_source.update(chunk[col].dropna().astype(str).tolist())
                
                # Process target data in chunks
                unique_values_target = set()
                target_sum = 0.0
                target_values = []
                
                # Determine if column is integer type in target
                is_integer = np.issubdtype(target[col].dtype, np.integer)
                
                for i in range(0, len(target), chunk_size):
                    chunk = target.iloc[i:i + chunk_size]
                    col_summary['target_null_count'] += np.int64(chunk[col].isnull().sum())
                    
                    # Handle unique values based on dtype
                    if np.issubdtype(chunk[col].dtype, np.number):
                        non_null = chunk[col].dropna()
                        if is_integer:
                            values = non_null.astype(np.int64)
                            unique_values_target.update(values.tolist())
                            target_values.extend(values.tolist())
                        else:
                            values = non_null.astype(np.float64)
                            unique_values_target.update(values.tolist())
                            target_values.extend(values.tolist())
                        target_sum += np.float64(non_null.sum())
                    else:
                        unique_values_target.update(chunk[col].dropna().astype(str).tolist())
                
                # Convert counts to np.int64
                col_summary['source_unique_count'] = np.int64(len(unique_values_source))
                col_summary['target_unique_count'] = np.int64(len(unique_values_target))
                
                # Convert null counts to np.int64 if they aren't already
                col_summary['source_null_count'] = np.int64(col_summary['source_null_count'])
                col_summary['target_null_count'] = np.int64(col_summary['target_null_count'])
                
                # Convert all integer values to Python int for JSON serialization
                for key in ['source_unique_count', 'target_unique_count', 'source_null_count', 'target_null_count']:
                    col_summary[key] = int(col_summary[key])
                
                # For numeric columns, calculate statistics
                if np.issubdtype(source[col].dtype, np.number):
                    try:
                        # Convert to numpy arrays with explicit dtype based on data type
                        if np.issubdtype(source[col].dtype, np.integer):
                            source_values = np.array(source_values, dtype=np.int64)
                            target_values = np.array(target_values, dtype=np.int64)
                            # Convert to float64 for calculations
                            source_values = source_values.astype(np.float64)
                            target_values = target_values.astype(np.float64)
                        else:
                            source_values = np.array(source_values, dtype=np.float64)
                            target_values = np.array(target_values, dtype=np.float64)
                        
                        # Calculate statistics using numpy's methods
                        stats = {
                            'source_sum': np.float64(source_sum),
                            'target_sum': np.float64(target_sum),
                            'source_mean': np.float64(np.mean(source_values)) if len(source_values) > 0 else np.float64(0),
                            'target_mean': np.float64(np.mean(target_values)) if len(target_values) > 0 else np.float64(0),
                            'source_std': np.float64(np.std(source_values)) if len(source_values) > 0 else np.float64(0),
                            'target_std': np.float64(np.std(target_values)) if len(target_values) > 0 else np.float64(0)
                        }
                        
                        # Convert numpy types to Python types for JSON serialization
                        col_summary.update({k: float(v) for k, v in stats.items()})
                        
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
                # Initialize with proper numpy types even in error case
                summary[col] = {
                    'source_null_count': int(np.int64(0)),
                    'target_null_count': int(np.int64(0)),
                    'source_unique_count': int(np.int64(0)),
                    'target_unique_count': int(np.int64(0))
                }
                
                # Add numeric stats with proper types if it's a numeric column
                if np.issubdtype(source[col].dtype, np.number):
                    summary[col].update({
                        'source_sum': float(np.float64(0)),
                        'target_sum': float(np.float64(0)),
                        'source_mean': float(np.float64(0)),
                        'target_mean': float(np.float64(0)),
                        'source_std': float(np.float64(0)),
                        'target_std': float(np.float64(0))
                    })
        
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
            
            # Generate comparison report with enhanced error handling and memory optimization
            try:
                logger.info("Generating comparison report...")
                
                # Clear any existing report at the path
                if comparison_path.exists():
                    comparison_path.unlink()
                
                # Generate comparison with correct method signature
                comparison_report = source_profile.compare(target_profile)
                
                # Save comparison report
                logger.info("Saving comparison report...")
                comparison_report.to_file(str(comparison_path))
                logger.info("Comparison report saved successfully")
                
            except Exception as e:
                logger.error(f"Error generating comparison report: {str(e)}")
                # Create a more detailed error report
                with open(str(comparison_path), 'w', encoding='utf-8') as f:
                    f.write(f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="utf-8">
                        <title>Data Profile Comparison Report</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 2em; line-height: 1.6; }}
                            h1, h2 {{ color: #333; }}
                            .error {{ color: #721c24; background-color: #f8d7da; padding: 1em; border-radius: 4px; }}
                            .links {{ margin-top: 2em; }}
                            .links a {{ color: #007bff; text-decoration: none; }}
                            .links a:hover {{ text-decoration: underline; }}
                        </style>
                    </head>
                    <body>
                        <h1>Data Profile Comparison Report</h1>
                        <div class="error">
                            <h2>Error Generating Detailed Comparison</h2>
                            <p>An error occurred while generating the comparison report: {str(e)}</p>
                        </div>
                        <div class="links">
                            <h2>Individual Profile Reports</h2>
                            <p>Please check the individual profile reports for detailed analysis:</p>
                            <ul>
                                <li><a href="source_profile.html">Source Data Profile</a></li>
                                <li><a href="target_profile.html">Target Data Profile</a></li>
                            </ul>
                        </div>
                        <div>
                            <h2>Troubleshooting Steps</h2>
                            <ul>
                                <li>Check if both source and target profiles were generated successfully</li>
                                <li>Verify that the data types are compatible between source and target</li>
                                <li>Consider reducing the dataset size if memory issues occur</li>
                            </ul>
                        </div>
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

# """Core comparison engine for data comparison operations."""
# import pandas as pd
# import numpy as np
# from typing import Dict, List, Tuple, Any, Optional
# import logging
# from ydata_profiling import ProfileReport
# from pathlib import Path

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ComparisonEngine:
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

#     def auto_map_columns(self) -> List[Dict[str, Any]]:
#         """
#         Automatically map columns between source and target based on names.
        
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

#             mapping.append({
#                 'source': s_col,
#                 'target': t_col or '',
#                 'join': False,
#                 'data_type': str(self.source_df[s_col].dtype),
#                 'exclude': False
#             })

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

#     def _prepare_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         """
#         Prepare dataframes for comparison by applying mappings and type conversions.
        
#         Returns:
#             Tuple of (prepared source DataFrame, prepared target DataFrame)
#         """
#         if not self.mapping:
#             raise ValueError("Mapping must be set before comparison")

#         # Create copies to avoid modifying original dataframes
#         source = self.source_df.copy()
#         target = self.target_df.copy()

#         # Remove excluded columns
#         source = source[[m['source'] for m in self.mapping if not m['exclude']]]
#         target = target[[m['target'] for m in self.mapping if not m['exclude']]]

#         # Rename target columns to match source for comparison
#         rename_dict = {m['target']: m['source'] 
#                       for m in self.mapping 
#                       if not m['exclude'] and m['target']}
#         target = target.rename(columns=rename_dict)

#         # Convert data types for compatibility
#         for mapping in self.mapping:
#             if mapping['exclude'] or not mapping['target']:
#                 continue

#             source_col = mapping['source']
#             try:
#                 # Force string conversion for problematic columns
#                 if source_col == 'ColumnNameID' or (
#                     not np.issubdtype(source[source_col].dtype, np.number) and
#                     not np.issubdtype(source[source_col].dtype, np.datetime64)):
#                     source[source_col] = source[source_col].fillna('').astype(str)
#                     target[source_col] = target[source_col].fillna('').astype(str)
#                 # Handle numeric columns
#                 elif np.issubdtype(source[source_col].dtype, np.number):
#                     source[source_col] = pd.to_numeric(source[source_col], errors='coerce')
#                     target[source_col] = pd.to_numeric(target[source_col], errors='coerce')
#                 # Handle datetime columns
#                 elif np.issubdtype(source[source_col].dtype, np.datetime64):
#                     source[source_col] = pd.to_datetime(source[source_col], errors='coerce')
#                     target[source_col] = pd.to_datetime(target[source_col], errors='coerce')
#             except Exception as e:
#                 logger.warning(f"Type conversion failed for column {source_col}: {str(e)}. Converting to string.")
#                 source[source_col] = source[source_col].astype(str)
#                 target[source_col] = target[source_col].astype(str)

#         return source, target

#     def compare(self) -> Dict[str, Any]:
#         """
#         Perform the comparison between source and target data.
        
#         Returns:
#             Dictionary containing comparison results
#         """
#         try:
#             source, target = self._prepare_dataframes()
            
#             # Initialize comparison results
#             results = {
#                 'match_status': False,
#                 'rows_match': False,
#                 'columns_match': False,
#                 'datacompy_report': '',
#                 'source_unmatched_rows': pd.DataFrame(),
#                 'target_unmatched_rows': pd.DataFrame(),
#                 'column_summary': self._generate_column_summary(source, target),
#                 'row_counts': {
#                     'source_name': 'Source',
#                     'target_name': 'Target',
#                     'source_count': len(source),
#                     'target_count': len(target)
#                 },
#                 'distinct_values': {}  # Initialize distinct_values in results
#             }

#             # Basic comparison checks
#             results['columns_match'] = set(source.columns) == set(target.columns)
#             results['rows_match'] = len(source) == len(target)

#             # Get distinct values for non-numeric columns
#             try:
#                 results['distinct_values'] = self.get_distinct_values()
#             except Exception as e:
#                 logger.warning(f"Error getting distinct values: {str(e)}")
#                 results['distinct_values'] = {}

#             # Detailed comparison
#             if self.join_columns:
#                 try:
#                     # Find unmatched rows
#                     merged = pd.merge(source, target, on=self.join_columns, how='outer', indicator=True)
#                     results['source_unmatched_rows'] = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)
#                     results['target_unmatched_rows'] = merged[merged['_merge'] == 'right_only'].drop('_merge', axis=1)
                    
#                     # Generate detailed comparison report
#                     report_lines = []
#                     report_lines.append("DataCompy Comparison Report")
#                     report_lines.append("=" * 50)
#                     report_lines.append("\nSummary:")
#                     report_lines.append("-" * 20)
#                     report_lines.append(f"Source rows: {len(source)}")
#                     report_lines.append(f"Target rows: {len(target)}")
#                     report_lines.append(f"Unmatched in source: {len(results['source_unmatched_rows'])}")
#                     report_lines.append(f"Unmatched in target: {len(results['target_unmatched_rows'])}")
                    
#                     # Add column comparison
#                     report_lines.append("\nColumn Analysis:")
#                     report_lines.append("-" * 20)
#                     for col in source.columns:
#                         report_lines.append(f"\nColumn: {col}")
#                         if col in results['column_summary']:
#                             summary = results['column_summary'][col]
#                             report_lines.append(f"Source null count: {summary['source_null_count']}")
#                             report_lines.append(f"Target null count: {summary['target_null_count']}")
#                             report_lines.append(f"Source unique count: {summary['source_unique_count']}")
#                             report_lines.append(f"Target unique count: {summary['target_unique_count']}")
#                             if 'source_sum' in summary:  # Numeric columns
#                                 report_lines.append(f"Source sum: {summary['source_sum']}")
#                                 report_lines.append(f"Target sum: {summary['target_sum']}")
#                                 report_lines.append(f"Source mean: {summary['source_mean']}")
#                                 report_lines.append(f"Target mean: {summary['target_mean']}")
                    
#                     # Add value distribution for join columns
#                     report_lines.append("\nJoin Columns Analysis:")
#                     report_lines.append("-" * 20)
#                     if results['distinct_values']:
#                         for col in self.join_columns:
#                             if col in results['distinct_values']:
#                                 report_lines.append(f"\nJoin Column: {col}")
#                                 report_lines.append(f"Source unique values: {results['distinct_values'][col]['source_count']}")
#                                 report_lines.append(f"Target unique values: {results['distinct_values'][col]['target_count']}")
#                                 report_lines.append("Sample values comparison:")
#                                 s_vals = list(results['distinct_values'][col]['source_values'].items())[:5]
#                                 t_vals = list(results['distinct_values'][col]['target_values'].items())[:5]
#                                 report_lines.append("Source top 5: " + ", ".join(f"{v}({c})" for v, c in s_vals))
#                                 report_lines.append("Target top 5: " + ", ".join(f"{v}({c})" for v, c in t_vals))
                    
#                     # Add sample of unmatched rows
#                     if len(results['source_unmatched_rows']) > 0:
#                         report_lines.append("\nSample Unmatched Rows in Source:")
#                         report_lines.append("-" * 20)
#                         sample = results['source_unmatched_rows'].head(5).to_string()
#                         report_lines.append(sample)
                    
#                     if len(results['target_unmatched_rows']) > 0:
#                         report_lines.append("\nSample Unmatched Rows in Target:")
#                         report_lines.append("-" * 20)
#                         sample = results['target_unmatched_rows'].head(5).to_string()
#                         report_lines.append(sample)
                    
#                     results['datacompy_report'] = "\n".join(report_lines)
                    
#                     # Overall match status
#                     results['match_status'] = (
#                         results['columns_match'] and 
#                         results['rows_match'] and
#                         len(results['source_unmatched_rows']) == 0 and
#                         len(results['target_unmatched_rows']) == 0
#                     )
#                 except Exception as e:
#                     logger.error(f"Error in detailed comparison: {str(e)}")
#                     results['datacompy_report'] = f"Error in comparison: {str(e)}"
#                     results['match_status'] = False

#             return results
#         except Exception as e:
#             logger.error(f"Error in compare method: {str(e)}")
#             return {
#                 'match_status': False,
#                 'error': str(e),
#                 'datacompy_report': f"Comparison failed: {str(e)}"
#             }

#     def _generate_column_summary(self, source: pd.DataFrame, 
#                                target: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
#         """
#         Generate detailed column-level comparison summary.
        
#         Args:
#             source: Prepared source DataFrame
#             target: Prepared target DataFrame
            
#         Returns:
#             Dictionary containing column-level statistics
#         """
#         summary = {}
        
#         for col in source.columns:
#             if col in self.join_columns:
#                 continue
                
#             summary[col] = {
#                 'source_null_count': source[col].isnull().sum(),
#                 'target_null_count': target[col].isnull().sum(),
#                 'source_unique_count': source[col].nunique(),
#                 'target_unique_count': target[col].nunique(),
#             }
            
#             # For numeric columns, add statistical comparisons
#             if np.issubdtype(source[col].dtype, np.number):
#                 summary[col].update({
#                     'source_sum': source[col].sum(),
#                     'target_sum': target[col].sum(),
#                     'source_mean': source[col].mean(),
#                     'target_mean': target[col].mean(),
#                     'source_std': source[col].std(),
#                     'target_std': target[col].std(),
#                 })

#         return summary

#     def generate_profiling_reports(self, output_dir: str) -> Dict[str, str]:
#         """
#         Generate YData Profiling reports for source and target data.
        
#         Args:
#             output_dir: Directory to save the reports
            
#         Returns:
#             Dictionary containing paths to generated reports
#         """
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)

#         # Generate individual profiles
#         source_profile = ProfileReport(self.source_df, title="Source Data Profile")
#         target_profile = ProfileReport(self.target_df, title="Target Data Profile")

#         # Save reports
#         source_path = output_path / "source_profile.html"
#         target_path = output_path / "target_profile.html"
#         comparison_path = output_path / "comparison_profile.html"

#         source_profile.to_file(str(source_path))
#         target_profile.to_file(str(target_path))
        
#         # Generate comparison report
#         comparison_report = source_profile.compare(target_profile)
#         comparison_report.to_file(str(comparison_path))

#         return {
#             'source_profile': str(source_path),
#             'target_profile': str(target_path),
#             'comparison_profile': str(comparison_path)
#         }

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComparisonEngine:
    # Type mapping dictionary for data type conversion
    TYPE_MAPPING = {
        'int': 'int32',
        'int64': 'int64',
        'numeric': 'int64',
        'bigint': 'int64',
        'smallint': 'int64',
        'varchar': 'string',
        'nvarchar': 'string',
        'char': 'string',
        'nchar': 'string',
        'text': 'string',
        'date': 'datetime64[ns]',
        'datetime': 'datetime64[ns]',
        'decimal': 'float64',
        'float': 'float64',
        'bit': 'bool',
        'boolean': 'bool',
        'unsupported': 'string'  # Default handling for unsupported types
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

    def _get_mapped_type(self, column: str, current_type: str) -> str:
        """
        Get the mapped data type for a column.
        
        Args:
            column: Column name
            current_type: Current data type
            
        Returns:
            Mapped data type string
        """
        # Check for user override first
        if column in self.type_overrides:
            return self.type_overrides[column]
            
        # Convert current type to lowercase for matching
        current_type = str(current_type).lower().split('[')[0]  # Handle cases like 'datetime64[ns]'
        
        # Check if type exists in mapping
        for key, value in self.TYPE_MAPPING.items():
            if key in current_type:
                return value
                
        # Default to string for unknown types
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

            # Get source column type
            source_type = str(self.source_df[s_col].dtype)
            
            # Get target column type if there's a match
            target_type = str(self.target_df[t_col].dtype) if t_col else 'unknown'
            
            # Determine the best data type to use
            if t_col:
                # If types match exactly, use that type
                if source_type == target_type:
                    mapped_type = source_type
                else:
                    # Try to find compatible type from TYPE_MAPPING
                    source_mapped = self._get_mapped_type(s_col, source_type)
                    target_mapped = self._get_mapped_type(t_col, target_type)
                    
                    if source_mapped == target_mapped:
                        mapped_type = source_mapped
                    else:
                        # If types are incompatible, default to string
                        mapped_type = 'string'
            else:
                # If no target match, use source type
                mapped_type = self._get_mapped_type(s_col, source_type)

            mapping.append({
                'source': s_col,
                'target': t_col or '',
                'join': False,
                'data_type': mapped_type,
                'exclude': False,
                'source_type': source_type,  # Original source type for reference
                'target_type': target_type   # Original target type for reference
            })

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

    def _prepare_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare dataframes for comparison by applying mappings and type conversions.
        
        Returns:
            Tuple of (prepared source DataFrame, prepared target DataFrame)
        """
        if not self.mapping:
            raise ValueError("Mapping must be set before comparison")

        # Create copies to avoid modifying original dataframes
        source = self.source_df.copy()
        target = self.target_df.copy()

        # Remove excluded columns
        source = source[[m['source'] for m in self.mapping if not m['exclude']]]
        target = target[[m['target'] for m in self.mapping if not m['exclude']]]

        # Rename target columns to match source for comparison
        rename_dict = {m['target']: m['source'] 
                      for m in self.mapping 
                      if not m['exclude'] and m['target']}
        target = target.rename(columns=rename_dict)

        # Convert data types for compatibility
        for mapping in self.mapping:
            if mapping['exclude'] or not mapping['target']:
                continue

            source_col = mapping['source']
            try:
                # Get the mapped type for this column
                mapped_type = self._get_mapped_type(source_col, mapping.get('data_type', str(source[source_col].dtype)))
                
                # Convert based on mapped type
                if mapped_type == 'string':
                    source[source_col] = source[source_col].fillna('').astype(str)
                    target[source_col] = target[source_col].fillna('').astype(str)
                elif mapped_type in ['int32', 'int64']:
                    source[source_col] = pd.to_numeric(source[source_col], errors='coerce').astype(mapped_type)
                    target[source_col] = pd.to_numeric(target[source_col], errors='coerce').astype(mapped_type)
                elif mapped_type == 'float64':
                    source[source_col] = pd.to_numeric(source[source_col], errors='coerce').astype(np.float64)
                    target[source_col] = pd.to_numeric(target[source_col], errors='coerce').astype(np.float64)
                elif mapped_type == 'datetime64[ns]':
                    source[source_col] = pd.to_datetime(source[source_col], errors='coerce')
                    target[source_col] = pd.to_datetime(target[source_col], errors='coerce')
                elif mapped_type == 'bool':
                    source[source_col] = source[source_col].fillna(False).astype(bool)
                    target[source_col] = target[source_col].fillna(False).astype(bool)
                else:
                    # Default to string for unknown types
                    source[source_col] = source[source_col].fillna('').astype(str)
                    target[source_col] = target[source_col].fillna('').astype(str)
            except Exception as e:
                logger.warning(f"Type conversion failed for column {source_col}: {str(e)}. Converting to string.")
                source[source_col] = source[source_col].astype(str)
                target[source_col] = target[source_col].astype(str)

        return source, target

    def compare(self) -> Dict[str, Any]:
        """
        Perform the comparison between source and target data.
        
        Returns:
            Dictionary containing comparison results
        """
        try:
            source, target = self._prepare_dataframes()
            
            # Initialize comparison results
            results = {
                'match_status': False,
                'rows_match': False,
                'columns_match': False,
                'datacompy_report': '',
                'source_unmatched_rows': pd.DataFrame(),
                'target_unmatched_rows': pd.DataFrame(),
                'column_summary': self._generate_column_summary(source, target),
                'row_counts': {
                    'source_name': 'Source',
                    'target_name': 'Target',
                    'source_count': len(source),
                    'target_count': len(target)
                },
                'distinct_values': {}  # Initialize distinct_values in results
            }

            # Basic comparison checks
            results['columns_match'] = set(source.columns) == set(target.columns)
            results['rows_match'] = len(source) == len(target)

            # Get distinct values for non-numeric columns
            try:
                results['distinct_values'] = self.get_distinct_values()
            except Exception as e:
                logger.warning(f"Error getting distinct values: {str(e)}")
                results['distinct_values'] = {}

            # Detailed comparison
            if self.join_columns:
                try:
                    # Find unmatched rows
                    merged = pd.merge(source, target, on=self.join_columns, how='outer', indicator=True)
                    results['source_unmatched_rows'] = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)
                    results['target_unmatched_rows'] = merged[merged['_merge'] == 'right_only'].drop('_merge', axis=1)
                    
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
                
            summary[col] = {
                'source_null_count': source[col].isnull().sum(),
                'target_null_count': target[col].isnull().sum(),
                'source_unique_count': source[col].nunique(),
                'target_unique_count': target[col].nunique(),
            }
            
            # For numeric columns, add statistical comparisons
            if np.issubdtype(source[col].dtype, np.number):
                summary[col].update({
                    'source_sum': source[col].sum(),
                    'target_sum': target[col].sum(),
                    'source_mean': source[col].mean(),
                    'target_mean': target[col].mean(),
                    'source_std': source[col].std(),
                    'target_std': target[col].std(),
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
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate individual profiles
        source_profile = ProfileReport(self.source_df, title="Source Data Profile")
        target_profile = ProfileReport(self.target_df, title="Target Data Profile")

        # Save reports
        source_path = output_path / "source_profile.html"
        target_path = output_path / "target_profile.html"
        comparison_path = output_path / "comparison_profile.html"

        source_profile.to_file(str(source_path))
        target_profile.to_file(str(target_path))
        
        # Generate comparison report
        comparison_report = source_profile.compare(target_profile)
        comparison_report.to_file(str(comparison_path))

        return {
            'source_profile': str(source_path),
            'target_profile': str(target_path),
            'comparison_profile': str(comparison_path)
        }

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

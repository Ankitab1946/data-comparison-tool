"""Report generation utilities for the Data Comparison Tool."""
import pandas as pd
import numpy as np
import os
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import logging
from openpyxl.styles import PatternFill, Font, Color
from openpyxl.utils import get_column_letter

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        """Initialize the report generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def execute_query(self, df: pd.DataFrame, query: str = None) -> pd.DataFrame:
        """Execute a SQL-like query on the DataFrame.
        
        Args:
            df: The DataFrame to query
            query: SQL-like query string (e.g., "SELECT * FROM data WHERE Column > 5")
                  If None, returns the original DataFrame
        
        Returns:
            Filtered DataFrame based on the query
        """
        if not query:
            return df
            
        try:
            # Parse the query
            query = query.strip().lower()
            if not query.startswith("select"):
                raise ValueError("Query must start with SELECT")
            
            # Handle FROM clause
            if "from" not in query:
                query = query.replace("where", "FROM data WHERE")
                if "where" not in query:
                    query += " FROM data"
            
            # Split query parts
            parts = query.split("from")
            if len(parts) != 2:
                raise ValueError("Invalid query format")
                
            select_part = parts[0].replace("select", "").strip()
            from_part = parts[1].strip()
            
            # Extract WHERE clause if it exists
            where_clause = None
            if "where" in from_part:
                from_parts = from_part.split("where")
                if len(from_parts) != 2:
                    raise ValueError("Invalid WHERE clause")
                where_clause = from_parts[1].strip()
            
            # Extract column names
            if select_part == "*":
                selected_cols = df.columns.tolist()
            else:
                selected_cols = [col.strip() for col in select_part.split(",")]
                
            # Validate columns exist
            missing_cols = [col for col in selected_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found: {', '.join(missing_cols)}")
            
            result_df = df[selected_cols]
            
            # Apply WHERE clause if it exists
            if where_clause:
                # Replace column names with df[] notation
                for col in df.columns:
                    where_clause = where_clause.replace(col, f"df['{col}']")
                try:
                    result_df = result_df[eval(where_clause)]
                except Exception as e:
                    raise ValueError(f"Invalid WHERE clause: {str(e)}")
            
            if result_df.empty:
                logger.warning("Query returned no results")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise ValueError(f"Query execution failed: {str(e)}")
        
    def _style_excel_cell(self, cell, is_pass: bool):
        """Apply styling to Excel cell based on pass/fail status."""
        if is_pass:
            cell.fill = PatternFill(start_color='90EE90', end_color='90EE90', fill_type='solid')  # Light green
            cell.font = Font(color='006400')  # Dark green
        else:
            cell.fill = PatternFill(start_color='FFB6C1', end_color='FFB6C1', fill_type='solid')  # Light pink
            cell.font = Font(color='8B0000')  # Dark red

    def _calculate_aggregations(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Calculate aggregations for numeric columns."""
        aggs = []
        for col in numeric_cols:
            try:
                aggs.append({
                    'Column': col,
                    'Sum': df[col].sum(),
                    'Mean': df[col].mean(),
                    'Min': df[col].min(),
                    'Max': df[col].max(),
                    'StdDev': df[col].std()
                })
            except Exception as e:
                logger.error(f"Error calculating aggregations for column {col}: {str(e)}")
                # Add a placeholder with error indicators
                aggs.append({
                    'Column': col,
                    'Sum': np.nan,
                    'Mean': np.nan,
                    'Min': np.nan,
                    'Max': np.nan,
                    'StdDev': np.nan
                })
        return pd.DataFrame(aggs)

    def generate_regression_report(self, comparison_results: Dict[str, Any], 
                                 source_df: pd.DataFrame, 
                                 target_df: pd.DataFrame) -> str:
        """Generate enhanced regression report with multiple checks."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_dir / f"RegressionReport_{timestamp}.xlsx"
            
            with pd.ExcelWriter(str(report_path), engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Create formats
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#D3D3D3',
                    'border': 1,
                    'align': 'center'
                })
                
                pass_format = workbook.add_format({
                    'bg_color': '90EE90',  # Light green
                    'font_color': '006400',  # Dark green
                    'border': 1
                })
                
                fail_format = workbook.add_format({
                    'bg_color': 'FFB6C1',  # Light pink
                    'font_color': '8B0000',  # Dark red
                    'border': 1
                })
                
                cell_format = workbook.add_format({
                    'border': 1
                })
                
                # 1. Null Check Tab
                null_sheet = workbook.add_worksheet('NullCheck')
                
                # Write headers
                headers = ['Column', 'Source Null Count', 'Target Null Count', 'Difference', 'Status']
                for col, header in enumerate(headers):
                    null_sheet.write(0, col, header, header_format)
                
                row = 1
                for column in source_df.columns:
                    source_nulls = source_df[column].isnull().sum()
                    target_nulls = target_df[column].isnull().sum() if column in target_df.columns else 0
                    difference = abs(source_nulls - target_nulls)
                    status = 'Pass' if difference == 0 else 'Fail'
                    
                    null_sheet.write(row, 0, column, cell_format)
                    null_sheet.write(row, 1, source_nulls, cell_format)
                    null_sheet.write(row, 2, target_nulls, cell_format)
                    null_sheet.write(row, 3, difference, cell_format)
                    null_sheet.write(row, 4, status, pass_format if status == 'Pass' else fail_format)
                    row += 1
                
                # Auto-adjust column widths
                for col in range(len(headers)):
                    null_sheet.set_column(col, col, 15)
                
                # 2. String Hash Check Tab
                hash_sheet = workbook.add_worksheet('StringHashCheck')
                
                # Write headers
                headers = ['Column', 'Source Hash', 'Target Hash', 'Match']
                for col, header in enumerate(headers):
                    hash_sheet.write(0, col, header, header_format)
                
                row = 1
                for column in source_df.select_dtypes(include=['object']).columns:
                    if column in target_df.columns:
                        source_hash = pd.util.hash_pandas_object(source_df[column]).sum()
                        target_hash = pd.util.hash_pandas_object(target_df[column]).sum()
                        match = 'Pass' if source_hash == target_hash else 'Fail'
                        
                        hash_sheet.write(row, 0, column, cell_format)
                        hash_sheet.write(row, 1, str(source_hash), cell_format)
                        hash_sheet.write(row, 2, str(target_hash), cell_format)
                        hash_sheet.write(row, 3, match, pass_format if match == 'Pass' else fail_format)
                        row += 1
                
                # Auto-adjust column widths
                for col in range(len(headers)):
                    hash_sheet.set_column(col, col, 20)
                
                # 3. Count Check Tab
                count_sheet = workbook.add_worksheet('CountCheck')
                
                # Write headers
                headers = ['Metric', 'Value', 'Result']
                for col, header in enumerate(headers):
                    count_sheet.write(0, col, header, header_format)
                
                # Write data
                source_count = len(source_df)
                target_count = len(target_df)
                match = source_count == target_count
                
                count_sheet.write(1, 0, 'Source Count', cell_format)
                count_sheet.write(1, 1, source_count, cell_format)
                count_sheet.write(1, 2, 'Pass' if match else 'Fail', pass_format if match else fail_format)
                
                count_sheet.write(2, 0, 'Target Count', cell_format)
                count_sheet.write(2, 1, target_count, cell_format)
                count_sheet.write(2, 2, 'Pass' if match else 'Fail', pass_format if match else fail_format)
                
                # Auto-adjust column widths
                for col in range(len(headers)):
                    count_sheet.set_column(col, col, 15)

                # 4. Aggregation Check Tab
                agg_sheet = workbook.add_worksheet('AggregationCheck')
                
                # Get numeric columns from both dataframes
                source_numeric = set(source_df.select_dtypes(include=[np.number]).columns)
                target_numeric = set(target_df.select_dtypes(include=[np.number]).columns)
                numeric_cols = list(source_numeric.intersection(target_numeric))

                if not numeric_cols:
                    # Write message if no common numeric columns
                    agg_sheet.write(0, 0, 'No common numeric columns found for aggregation check', cell_format)
                else:
                    # Write headers
                    headers = ['Column', 'Metric', 'Source', 'Target', 'Result']
                    for col, header in enumerate(headers):
                        agg_sheet.write(0, col, header, header_format)
                    
                    row = 1
                    for col in numeric_cols:
                        try:
                            # Calculate metrics for source
                            source_stats = source_df[col].agg(['sum', 'mean', 'min', 'max', 'std']).to_dict()
                            target_stats = target_df[col].agg(['sum', 'mean', 'min', 'max', 'std']).to_dict()
                            
                            for metric in ['sum', 'mean', 'min', 'max', 'std']:
                                source_val = source_stats[metric]
                                target_val = target_stats[metric]
                                
                                # Check if values match (considering floating point precision)
                                if pd.isna(source_val) or pd.isna(target_val):
                                    matches = False
                                else:
                                    matches = np.isclose(source_val, target_val, rtol=1e-05)
                                
                                # Write row
                                agg_sheet.write(row, 0, col, cell_format)
                                agg_sheet.write(row, 1, metric.upper(), cell_format)
                                agg_sheet.write(row, 2, str(source_val), cell_format)
                                agg_sheet.write(row, 3, str(target_val), cell_format)
                                agg_sheet.write(row, 4, 'Pass' if matches else 'Fail',
                                              pass_format if matches else fail_format)
                                row += 1
                                
                        except Exception as e:
                            logger.error(f"Error comparing column {col}: {str(e)}")
                            # Write error row
                            agg_sheet.write(row, 0, col, cell_format)
                            agg_sheet.write(row, 1, 'ERROR', cell_format)
                            agg_sheet.write(row, 2, 'ERROR', cell_format)
                            agg_sheet.write(row, 3, 'ERROR', cell_format)
                            agg_sheet.write(row, 4, 'Fail', fail_format)
                            row += 1
                    
                    # Auto-adjust column widths
                    for col in range(len(headers)):
                        agg_sheet.set_column(col, col, 15)

                # 5. Distinct Check Tab
                distinct_sheet = workbook.add_worksheet('DistinctCheck')
                
                # Write headers
                headers = ['Column', 'Source Distinct Count', 'Target Distinct Count', 'Count Match', 'Values Match', 'Source Values', 'Target Values']
                for col, header in enumerate(headers):
                    distinct_sheet.write(0, col, header, header_format)
                
                # Get non-numeric columns
                non_numeric_cols = source_df.select_dtypes(exclude=[np.number]).columns
                
                row = 1
                for col in non_numeric_cols:
                    try:
                        # Get source values
                        source_vals = sorted(source_df[col].fillna('NULL').astype(str).unique())
                        source_count = len(source_vals)
                        source_display = ', '.join(source_vals[:10])
                        if source_count > 10:
                            source_display += '...'
                        
                        # Get target values
                        if col in target_df.columns:
                            target_vals = sorted(target_df[col].fillna('NULL').astype(str).unique())
                            target_count = len(target_vals)
                            target_display = ', '.join(target_vals[:10])
                            if target_count > 10:
                                target_display += '...'
                        else:
                            target_vals = []
                            target_count = 0
                            target_display = 'Column not found'
                        
                        # Compare values
                        count_match = 'Pass' if source_count == target_count else 'Fail'
                        values_match = 'Pass' if set(source_vals) == set(target_vals) else 'Fail'
                        
                        # Write row
                        distinct_sheet.write(row, 0, str(col), cell_format)
                        distinct_sheet.write(row, 1, source_count, cell_format)
                        distinct_sheet.write(row, 2, target_count, cell_format)
                        distinct_sheet.write(row, 3, count_match, pass_format if count_match == 'Pass' else fail_format)
                        distinct_sheet.write(row, 4, values_match, pass_format if values_match == 'Pass' else fail_format)
                        distinct_sheet.write(row, 5, source_display, cell_format)
                        distinct_sheet.write(row, 6, target_display, cell_format)
                        row += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing column {col}: {str(e)}")
                        continue
                
                # Write message if no columns processed
                if row == 1:
                    distinct_sheet.write(1, 0, 'No non-numeric columns found', cell_format)
                
                # Auto-adjust column widths
                for col in range(len(headers)):
                    max_length = 20  # Default width
                    distinct_sheet.set_column(col, col, max_length)
                
            logger.info(f"Enhanced regression report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating regression report: {str(e)}")
            raise

    def _normalize_column_name(self, col_name):
        """Normalize column name by removing spaces, underscores, and special chars"""
        # Remove all spaces, underscores, hyphens and convert to lowercase
        normalized = col_name.lower()
        normalized = normalized.replace(' ', '').replace('_', '').replace('-', '')
        # Keep only alphanumeric characters
        normalized = ''.join(c for c in normalized if c.isalnum())
        return normalized

    def _map_join_columns(self, source_df: pd.DataFrame, target_df: pd.DataFrame, join_columns: List[str]) -> List[str]:
        """Map join columns using the same normalization logic as comparison engine"""
        logger.info(f"Original join columns: {join_columns}")
        logger.info(f"Source columns: {list(source_df.columns)}")
        logger.info(f"Target columns: {list(target_df.columns)}")
        
        # Create normalized lookup dictionaries
        source_normalized = {self._normalize_column_name(col): col for col in source_df.columns}
        target_normalized = {self._normalize_column_name(col): col for col in target_df.columns}
        
        mapped_join_columns = []
        missing_columns = []
        
        for join_col in join_columns:
            # Try exact match first
            if join_col in source_df.columns and join_col in target_df.columns:
                mapped_join_columns.append(join_col)
                logger.info(f"Exact match found for join column: {join_col}")
                continue
            
            # Try case-insensitive match
            join_col_lower = join_col.lower()
            source_match = None
            target_match = None
            
            for s_col in source_df.columns:
                if s_col.lower() == join_col_lower:
                    source_match = s_col
                    break
            
            for t_col in target_df.columns:
                if t_col.lower() == join_col_lower:
                    target_match = t_col
                    break
            
            if source_match and target_match:
                # Use the source column name as the canonical name
                mapped_join_columns.append(source_match)
                logger.info(f"Case-insensitive match found: {join_col} -> {source_match}")
                continue
            
            # Try normalized matching
            join_col_normalized = self._normalize_column_name(join_col)
            
            if join_col_normalized in source_normalized and join_col_normalized in target_normalized:
                source_col = source_normalized[join_col_normalized]
                target_col = target_normalized[join_col_normalized]
                mapped_join_columns.append(source_col)
                logger.info(f"Normalized match found: {join_col} -> {source_col} (source) and {target_col} (target)")
                continue
            
            # No match found
            missing_columns.append(join_col)
            logger.warning(f"No match found for join column: {join_col}")
        
        if missing_columns:
            error_msg = f"Failed to generate difference report: Join columns missing: {', '.join(missing_columns)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Final mapped join columns: {mapped_join_columns}")
        return mapped_join_columns

    def generate_difference_report(self, source_df: pd.DataFrame, target_df: pd.DataFrame, 
                                 join_columns: List[str]) -> str:
        """Generate enhanced side-by-side difference report with normalized column mapping."""
        try:
            if source_df.empty or target_df.empty:
                logger.info("No data to compare in difference report")
                return None

            # Apply the same column normalization logic as comparison engine
            try:
                mapped_join_columns = self._map_join_columns(source_df, target_df, join_columns)
            except ValueError as e:
                logger.error(f"Column mapping failed: {str(e)}")
                raise

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_dir / f"DifferenceReport_{timestamp}.xlsx"

            # Constants for Excel limitations - reduced for better performance
            MAX_ROWS = 100000  # Significantly reduced from Excel's limit for stability
            CHUNK_SIZE = 90000  # Slightly less than max for headers and formatting
            
            # Create normalized target dataframe for merging
            target_df_normalized = target_df.copy()
            
            # Create column mapping for renaming target columns to match source
            column_mapping = {}
            source_normalized = {self._normalize_column_name(col): col for col in source_df.columns}
            target_normalized = {self._normalize_column_name(col): col for col in target_df.columns}
            
            # Map target columns to source column names where they match
            for target_col in target_df.columns:
                target_norm = self._normalize_column_name(target_col)
                if target_norm in source_normalized:
                    source_col = source_normalized[target_norm]
                    if source_col != target_col:
                        column_mapping[target_col] = source_col
                        logger.info(f"Renaming target column '{target_col}' to '{source_col}' for merge")
            
            # Apply column renaming to target dataframe
            if column_mapping:
                target_df_normalized = target_df_normalized.rename(columns=column_mapping)
            
            # Process data in chunks to avoid memory issues
            dfs_to_process = []
            for start_idx in range(0, len(source_df), CHUNK_SIZE):
                # Get chunks of both dataframes
                source_chunk = source_df.iloc[start_idx:start_idx + CHUNK_SIZE]
                
                try:
                    # Find corresponding rows in target using mapped join columns
                    chunk_merged = source_chunk.merge(
                        target_df_normalized, 
                        on=mapped_join_columns, 
                        how='outer', 
                        indicator=True,
                        suffixes=('_source', '_target')
                    )
                    
                    # Create comparison status column
                    chunk_merged['Status'] = chunk_merged['_merge'].map({
                        'left_only': 'Deleted',
                        'right_only': 'Inserted',
                        'both': 'Updated'
                    })
                    
                    # Remove the merge indicator column
                    chunk_merged = chunk_merged.drop('_merge', axis=1)
                    
                    # Only keep rows with differences
                    diff_rows = chunk_merged[chunk_merged['Status'] != 'Updated']
                    if not diff_rows.empty:
                        dfs_to_process.append(diff_rows)
                        
                except Exception as merge_error:
                    logger.error(f"Error during merge operation: {str(merge_error)}")
                    # Try with string conversion for join columns
                    try:
                        logger.info("Attempting merge with string conversion for join columns")
                        source_chunk_str = source_chunk.copy()
                        target_df_str = target_df_normalized.copy()
                        
                        for col in mapped_join_columns:
                            if col in source_chunk_str.columns:
                                source_chunk_str[col] = source_chunk_str[col].astype(str)
                            if col in target_df_str.columns:
                                target_df_str[col] = target_df_str[col].astype(str)
                        
                        chunk_merged = source_chunk_str.merge(
                            target_df_str, 
                            on=mapped_join_columns, 
                            how='outer', 
                            indicator=True,
                            suffixes=('_source', '_target')
                        )
                        
                        # Create comparison status column
                        chunk_merged['Status'] = chunk_merged['_merge'].map({
                            'left_only': 'Deleted',
                            'right_only': 'Inserted',
                            'both': 'Updated'
                        })
                        
                        # Remove the merge indicator column
                        chunk_merged = chunk_merged.drop('_merge', axis=1)
                        
                        # Only keep rows with differences
                        diff_rows = chunk_merged[chunk_merged['Status'] != 'Updated']
                        if not diff_rows.empty:
                            dfs_to_process.append(diff_rows)
                            
                    except Exception as retry_error:
                        logger.error(f"Retry merge also failed: {str(retry_error)}")
                        raise ValueError(f"Failed to merge data even after string conversion: {str(retry_error)}")
            
            # If there are no differences
            if not dfs_to_process:
                logger.info("No differences found between source and target")
                
                # Create Excel file with summary only
                with pd.ExcelWriter(str(report_path), engine='xlsxwriter') as writer:
                    # Create summary sheet
                    summary_sheet = writer.book.add_worksheet('Summary')
                    
                    # Create formats
                    header_format = writer.book.add_format({
                        'bold': True,
                        'font_size': 12,
                        'bg_color': '#D3D3D3',
                        'border': 1,
                        'align': 'center',
                        'valign': 'vcenter'
                    })
                    
                    cell_format = writer.book.add_format({
                        'font_size': 11,
                        'border': 1,
                        'align': 'left',
                        'valign': 'vcenter'
                    })
                    
                    # Write summary information
                    summary_sheet.merge_range('A1:B1', 'Comparison Results', header_format)
                    summary_sheet.write(2, 0, 'Status:', cell_format)
                    summary_sheet.write(2, 1, 'No differences found between source and target datasets.', cell_format)
                    
                    summary_sheet.write(4, 0, 'Source Records:', cell_format)
                    summary_sheet.write(4, 1, len(source_df), cell_format)
                    
                    summary_sheet.write(5, 0, 'Target Records:', cell_format)
                    summary_sheet.write(5, 1, len(target_df), cell_format)
                    
                    summary_sheet.write(7, 0, 'Join Columns:', cell_format)
                    summary_sheet.write(7, 1, ', '.join(mapped_join_columns), cell_format)
                    
                    # Set column widths
                    summary_sheet.set_column(0, 0, 20)
                    summary_sheet.set_column(1, 1, 60)
                
                return str(report_path)
            
            # Combine all difference chunks
            merged = pd.concat(dfs_to_process, ignore_index=True)
            
            # Define status colors
            status_colors = {
                'Deleted': 'FFB6C1',     # Light pink
                'Left Only': 'FFB6C1',   # Light pink
                'Inserted': '90EE90',    # Light green
                'Right Only': '90EE90',  # Light green
                'Updated': 'FFD700'      # Gold
            }
            
            # Calculate number of chunks needed
            total_rows = len(merged)
            num_chunks = (total_rows - 1) // CHUNK_SIZE + 1
            
            # Save to Excel with formatting, splitting into multiple sheets if necessary
            with pd.ExcelWriter(str(report_path), engine='openpyxl') as writer:
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * CHUNK_SIZE
                    end_idx = min((chunk_idx + 1) * CHUNK_SIZE, total_rows)
                    
                    # Get chunk of data
                    chunk = merged.iloc[start_idx:end_idx]
                    
                    # Create sheet name
                    sheet_name = 'Differences' if num_chunks == 1 else f'Differences_{chunk_idx + 1}'
                    
                    # Write chunk to Excel
                    chunk.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Apply conditional formatting to chunk
                    worksheet = writer.sheets[sheet_name]
                    
                    # Apply formatting based on status
                    for idx, status in enumerate(chunk['Status'], start=2):
                        cell = worksheet.cell(row=idx, column=chunk.columns.get_loc('Status') + 1)
                        cell.fill = PatternFill(
                            start_color=status_colors.get(status, 'FFFFFF'),
                            end_color=status_colors.get(status, 'FFFFFF'),
                            fill_type='solid'
                        )
                    
                    # Add summary at top of each sheet
                    summary_data = chunk['Status'].value_counts().to_dict()
                    worksheet.cell(row=1, column=len(chunk.columns) + 2, value="Summary:")
                    for i, (status, count) in enumerate(summary_data.items(), start=2):
                        worksheet.cell(row=i, column=len(chunk.columns) + 2, value=f"{status}: {count}")
                
                # Add summary sheet
                summary_df = pd.DataFrame({
                    'Description': [
                        'Total Rows',
                        'Number of Sheets',
                        'Rows per Sheet',
                        'Deleted Records',
                        'Inserted Records',
                        'Updated Records'
                    ],
                    'Value': [
                        total_rows,
                        num_chunks,
                        CHUNK_SIZE,
                        sum(merged['Status'] == 'Deleted'),
                        sum(merged['Status'] == 'Inserted'),
                        sum(merged['Status'] == 'Updated')
                    ]
                })
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            logger.info(f"Difference report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating difference report: {str(e)}")
            raise

    def generate_datacompy_report(self, comparison_results: Dict[str, Any]) -> str:
        """Generate detailed DataCompy report with proper formatting."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_dir / f"DataCompy_{timestamp}.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                # Write header with timestamp
                f.write("=" * 80 + "\n")
                f.write("DataCompy Comparison Report\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")

                # Write summary section
                f.write("SUMMARY\n")
                f.write("-" * 80 + "\n")
                f.write(f"Source Rows: {comparison_results.get('row_counts', {}).get('source_count', 'N/A')}\n")
                f.write(f"Target Rows: {comparison_results.get('row_counts', {}).get('target_count', 'N/A')}\n")
                
                # Write column details
                f.write("\nCOLUMN ANALYSIS\n")
                f.write("-" * 80 + "\n")
                for col, details in comparison_results.get('column_summary', {}).items():
                    f.write(f"\nColumn: {col}\n")
                    f.write("  Source:\n")
                    f.write(f"    Null Count: {details.get('source_null_count', 'N/A')}\n")
                    f.write(f"    Unique Values: {details.get('source_unique_count', 'N/A')}\n")
                    if 'source_mean' in details:
                        f.write(f"    Mean: {details['source_mean']}\n")
                        f.write(f"    Sum: {details.get('source_sum', 'N/A')}\n")
                    
                    f.write("  Target:\n")
                    f.write(f"    Null Count: {details.get('target_null_count', 'N/A')}\n")
                    f.write(f"    Unique Values: {details.get('target_unique_count', 'N/A')}\n")
                    if 'target_mean' in details:
                        f.write(f"    Mean: {details['target_mean']}\n")
                        f.write(f"    Sum: {details.get('target_sum', 'N/A')}\n")
                    f.write("\n")
                
                # Write unmatched rows summary
                f.write("\nUNMATCHED ROWS SUMMARY\n")
                f.write("-" * 80 + "\n")
                
                source_unmatched = comparison_results.get('source_unmatched_rows', pd.DataFrame())
                target_unmatched = comparison_results.get('target_unmatched_rows', pd.DataFrame())
                
                f.write(f"Source Unmatched Count: {len(source_unmatched)}\n")
                f.write(f"Target Unmatched Count: {len(target_unmatched)}\n\n")

                if not source_unmatched.empty:
                    f.write("Sample of Source Unmatched Rows (first 5):\n")
                    f.write("-" * 40 + "\n")
                    sample = source_unmatched.head(5).to_string()
                    f.write(sample + "\n\n")

                if not target_unmatched.empty:
                    f.write("Sample of Target Unmatched Rows (first 5):\n")
                    f.write("-" * 40 + "\n")
                    sample = target_unmatched.head(5).to_string()
                    f.write(sample + "\n\n")

                # Write match status summary
                f.write("\nMATCH STATUS SUMMARY\n")
                f.write("-" * 80 + "\n")
                f.write(f"Rows Match: {'Yes' if comparison_results.get('rows_match', False) else 'No'}\n")
                f.write(f"Columns Match: {'Yes' if comparison_results.get('columns_match', False) else 'No'}\n")
                f.write(f"Data Matches: {'Yes' if comparison_results.get('match_status', False) else 'No'}\n")

                # Write any additional details from datacompy
                if 'datacompy_report' in comparison_results:
                    f.write("\nDETAILED COMPARISON\n")
                    f.write("-" * 80 + "\n")
                    f.write(comparison_results['datacompy_report'])
            
            logger.info(f"DataCompy report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating DataCompy report: {str(e)}")
            raise

    def generate_side_by_side_report(self, source_df: pd.DataFrame, target_df: pd.DataFrame, join_columns: List[str]) -> str:
        """
        Generate an Excel report with two worksheets:
        1. Summary: Shows the heading, record count for source & target, join columns, and overall status.
        2. Diff_Report: Shows a side-by-side difference of source and target records.
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            join_columns: List of columns to join on
            
        Returns:
            Path to the generated Excel report
        """
        try:
            # Validate inputs
            if source_df.empty or target_df.empty:
                raise ValueError("Source and Target DataFrames must contain data")

            # Create timestamp for unique file naming
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_dir / f"SideBySideReport_{timestamp}.xlsx"

            # Create Excel writer
            with pd.ExcelWriter(str(report_path), engine='xlsxwriter') as writer:
                workbook = writer.book

                # Create formats
                header_format = workbook.add_format({
                    'bold': True,
                    'font_size': 12,
                    'bg_color': '#D3D3D3',
                    'border': 1,
                    'align': 'center'
                })
                cell_format = workbook.add_format({
                    'font_size': 11,
                    'border': 1,
                    'align': 'left'
                })
                pass_format = workbook.add_format({
                    'bg_color': '90EE90',  # Light green
                    'font_color': '006400',  # Dark green
                    'border': 1
                })
                fail_format = workbook.add_format({
                    'bg_color': 'FFB6C1',  # Light pink
                    'font_color': '8B0000',  # Dark red
                    'border': 1
                })
                yellow_format = workbook.add_format({
                    'bg_color': 'FFFF00',  # Yellow
                    'border': 1
                })

                # Create Summary worksheet
                summary_sheet = workbook.add_worksheet('Summary')

                # Write Summary sheet
                summary_sheet.merge_range('A1:B1', 'Comparison Results', header_format)
                
                # Write record counts
                source_count = len(source_df)
                target_count = len(target_df)
                
                summary_sheet.write(2, 0, 'Source Record Count:', cell_format)
                summary_sheet.write(2, 1, source_count, cell_format)
                
                summary_sheet.write(3, 0, 'Target Record Count:', cell_format)
                summary_sheet.write(3, 1, target_count, cell_format)
                
                # Write join columns
                summary_sheet.write(4, 0, 'Join Columns:', cell_format)
                summary_sheet.write(4, 1, ', '.join(join_columns), cell_format)
                
                # Write status
                status = 'Pass' if source_count == target_count else 'Fail'
                summary_sheet.write(5, 0, 'Status:', cell_format)
                summary_sheet.write(5, 1, status, pass_format if status == 'Pass' else fail_format)

                # Set column widths for Summary sheet
                summary_sheet.set_column(0, 0, 20)
                summary_sheet.set_column(1, 1, 40)

                # Create Diff_Report worksheet
                diff_sheet = workbook.add_worksheet('Diff_Report')

                # Perform the comparison
                # Normalize column names to lowercase for case-insensitive comparison
                source_df.columns = source_df.columns.str.lower()
                target_df.columns = target_df.columns.str.lower()
                join_columns = [col.lower() for col in join_columns]

                # Merge dataframes
                merged_df = source_df.merge(
                    target_df,
                    on=join_columns,
                    how='outer',
                    indicator=True,
                    suffixes=('_source', '_target')
                )

                # Create comparison status column
                merged_df['Status'] = merged_df['_merge'].map({
                    'left_only': 'Left Only',
                    'right_only': 'Right Only',
                    'both': 'Both'
                })

                # For rows present in both, check if there are any differences
                both_mask = merged_df['Status'] == 'Both'
                if both_mask.any():
                    for col in source_df.columns:
                        if col not in join_columns:
                            source_col = col
                            target_col = f"{col}_target"
                            if target_col in merged_df.columns:
                                # Mark as updated if values are different
                                diff_mask = (merged_df[source_col] != merged_df[target_col]) & both_mask
                                merged_df.loc[diff_mask, 'Status'] = 'Both - Update'

                # Write Diff_Report headers
                headers = ['Column', 'Source Value', 'Target Value', 'Status']
                for col, header in enumerate(headers):
                    diff_sheet.write(0, col, header, header_format)

                # Write Diff_Report data
                row = 1
                for _, record in merged_df.iterrows():
                    status = record['Status']
                    
                    for col in source_df.columns:
                        if col not in join_columns:
                            source_val = record.get(col, '')
                            target_val = record.get(f"{col}_target", '')
                            
                            # Only write if there's a difference
                            if source_val != target_val or status in ['Left Only', 'Right Only']:
                                diff_sheet.write(row, 0, col, cell_format)
                                
                                # Format cells based on status
                                if status == 'Left Only':
                                    diff_sheet.write(row, 1, str(source_val), yellow_format)
                                    diff_sheet.write(row, 2, '', cell_format)
                                elif status == 'Right Only':
                                    diff_sheet.write(row, 1, '', cell_format)
                                    diff_sheet.write(row, 2, str(target_val), yellow_format)
                                else:  # Both - Update
                                    diff_sheet.write(row, 1, str(source_val), yellow_format)
                                    diff_sheet.write(row, 2, str(target_val), yellow_format)
                                
                                diff_sheet.write(row, 3, status, cell_format)
                                row += 1

                # Set column widths for Diff_Report sheet
                for col in range(len(headers)):
                    diff_sheet.set_column(col, col, 20)

            logger.info(f"Side by side comparison report generated: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Error generating side by side comparison report: {str(e)}")
            raise

    def generate_ydata_report(self, comparison_results: Dict[str, Any]) -> str:
        """Generate Y-Data Profiling comparison report."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_dir / f"Comparison_profile_{timestamp}.html"
            
            # Create resources directory
            resources_dir = self.output_dir / 'resources'
            resources_dir.mkdir(exist_ok=True)
            
            # Get profile data
            source_profile = comparison_results.get('source_profile', {})
            target_profile = comparison_results.get('target_profile', {})
            
            # Generate comparison HTML
            with open(report_path, 'w', encoding='utf-8') as f:
                html_content = '''
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>Data Profile Comparison Report</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 2em; line-height: 1.6; }
                        h1, h2 { color: #333; }
                        .container { max-width: 1200px; margin: 0 auto; }
                        .section { margin-bottom: 2em; padding: 1em; border: 1px solid #ddd; border-radius: 4px; }
                        .diff { background-color: #fff3cd; padding: 0.5em; }
                        table { width: 100%; border-collapse: collapse; margin: 1em 0; }
                        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                        th { background-color: #f5f5f5; }
                        .highlight { background-color: #ffe3e3; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Data Profile Comparison Report</h1>
                '''
                
                # Add timestamp
                html_content += f'<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>'
                
                # Basic Statistics Section
                html_content += '''
                    <div class="section">
                        <h2>Basic Statistics</h2>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Source</th>
                                <th>Target</th>
                                <th>Difference</th>
                            </tr>
                '''
                
                # Add basic statistics comparison
                metrics = ['row_count', 'column_count', 'duplicate_rows', 'missing_cells']
                for metric in metrics:
                    source_val = source_profile.get(metric, 'N/A')
                    target_val = target_profile.get(metric, 'N/A')
                    diff = ''
                    if isinstance(source_val, (int, float)) and isinstance(target_val, (int, float)):
                        diff = target_val - source_val
                    
                    html_content += f'''
                        <tr>
                            <td>{metric.replace('_', ' ').title()}</td>
                            <td>{source_val}</td>
                            <td>{target_val}</td>
                            <td>{diff if diff != '' else 'N/A'}</td>
                        </tr>
                    '''
                
                html_content += '</table></div>'
                
                # Column Analysis Section
                html_content += '''
                    <div class="section">
                        <h2>Column Analysis</h2>
                '''
                
                # Get all columns from both profiles
                all_columns = set(source_profile.get('columns', {}).keys()) | set(target_profile.get('columns', {}).keys())
                
                for column in sorted(all_columns):
                    source_col = source_profile.get('columns', {}).get(column, {})
                    target_col = target_profile.get('columns', {}).get(column, {})
                    
                    html_content += f'''
                        <div class="section">
                            <h3>Column: {column}</h3>
                            <table>
                                <tr>
                                    <th>Metric</th>
                                    <th>Source</th>
                                    <th>Target</th>
                                </tr>
                    '''
                    
                    # Compare column metrics
                    col_metrics = ['type', 'unique_count', 'missing_count', 'min', 'max', 'mean', 'std']
                    for metric in col_metrics:
                        source_val = source_col.get(metric, 'N/A')
                        target_val = target_col.get(metric, 'N/A')
                        highlight = ' class="highlight"' if source_val != target_val else ''
                        
                        html_content += f'''
                            <tr{highlight}>
                                <td>{metric.replace('_', ' ').title()}</td>
                                <td>{source_val}</td>
                                <td>{target_val}</td>
                            </tr>
                        '''
                    
                    html_content += '</table></div>'
                
                # Close main container and body
                html_content += '''
                    </div>
                </body>
                </html>
                '''
                
                f.write(html_content)
            
            logger.info(f"Y-Data Profile report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating Y-Data Profile report: {str(e)}")
            raise

    def create_report_archive(self, report_paths: Dict[str, str]) -> str:
        """Create a ZIP archive containing all reports."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            zip_path = self.output_dir / f"all_reports_{timestamp}.zip"
            
            with zipfile.ZipFile(str(zip_path), 'w') as zipf:
                for report_type, path in report_paths.items():
                    if path and os.path.exists(path):
                        # Add file to zip with its original name
                        zipf.write(path, os.path.basename(path))
                        
                        # For HTML reports, also add any associated resources
                        if path.endswith('.html'):
                            resources_dir = Path(path).parent / 'resources'
                            if resources_dir.exists():
                                for resource in resources_dir.rglob('*'):
                                    if resource.is_file():
                                        zipf.write(
                                            resource,
                                            os.path.join('resources', resource.relative_to(resources_dir))
                                        )
            
            logger.info(f"Report archive created: {zip_path}")
            return str(zip_path)
            
        except Exception as e:
            logger.error(f"Error creating report archive: {str(e)}")
            raise

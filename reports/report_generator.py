# """Report generation utilities for the Data Comparison Tool."""
# import pandas as pd
# import numpy as np
# import os
# import zipfile
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, Any, List, Tuple
# import logging
# from openpyxl.styles import PatternFill, Font, Color
# from openpyxl.utils import get_column_letter

# logger = logging.getLogger(__name__)

# class ReportGenerator:
#     def __init__(self, output_dir: str = "reports"):
#         """Initialize the report generator."""
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#     def _style_excel_cell(self, cell, is_pass: bool):
#         """Apply styling to Excel cell based on pass/fail status."""
#         if is_pass:
#             cell.fill = PatternFill(start_color='90EE90', end_color='90EE90', fill_type='solid')  # Light green
#             cell.font = Font(color='006400')  # Dark green
#         else:
#             cell.fill = PatternFill(start_color='FFB6C1', end_color='FFB6C1', fill_type='solid')  # Light pink
#             cell.font = Font(color='8B0000')  # Dark red

#     def _calculate_aggregations(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
#         """Calculate aggregations for numeric columns."""
#         aggs = []
#         for col in numeric_cols:
#             aggs.append({
#                 'Column': col,
#                 'Sum': df[col].sum(),
#                 'Mean': df[col].mean(),
#                 'Min': df[col].min(),
#                 'Max': df[col].max(),
#                 'StdDev': df[col].std()
#             })
#         return pd.DataFrame(aggs)


#     def generate_regression_report(self, comparison_results: Dict[str, Any], 
#                                  source_df: pd.DataFrame, 
#                                  target_df: pd.DataFrame) -> str:
#         """Generate enhanced regression report with multiple checks."""
#         try:
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             report_path = self.output_dir / f"regression_report_{timestamp}.xlsx"
            
#             with pd.ExcelWriter(str(report_path), engine='openpyxl') as writer:
#                 # 1. Count Check Tab
#                 count_data = {
#                     'Metric': ['Source Count', 'Target Count'],
#                     'Value': [len(source_df), len(target_df)],
#                     'Result': ['BASELINE', 'PASS' if len(source_df) == len(target_df) else 'FAIL']
#                 }
#                 count_df = pd.DataFrame(count_data)
#                 count_df.to_excel(writer, sheet_name='CountCheck', index=False)
                
#                 # Apply conditional formatting
#                 count_sheet = writer.sheets['CountCheck']
#                 for idx, result in enumerate(count_df['Result'], start=2):
#                     if result != 'BASELINE':
#                         cell = count_sheet.cell(row=idx, column=3)
#                         self._style_excel_cell(cell, result == 'PASS')

#                 # 2. Aggregation Check Tab
#                 numeric_cols = source_df.select_dtypes(include=[np.number]).columns
#                 source_aggs = self._calculate_aggregations(source_df, numeric_cols)
#                 target_aggs = self._calculate_aggregations(target_df, numeric_cols)
                
#                 agg_comparison = []
#                 for col in numeric_cols:
#                     source_row = source_aggs[source_aggs['Column'] == col].iloc[0]
#                     target_row = target_aggs[target_aggs['Column'] == col].iloc[0]
                    
#                     for metric in ['Sum', 'Mean', 'Min', 'Max', 'StdDev']:
#                         matches = np.isclose(source_row[metric], target_row[metric], rtol=1e-05)
#                         agg_comparison.append({
#                             'Column': col,
#                             'Metric': metric,
#                             'Source': source_row[metric],
#                             'Target': target_row[metric],
#                             'Result': 'PASS' if matches else 'FAIL'
#                         })
                
#                 agg_df = pd.DataFrame(agg_comparison)
#                 agg_df.to_excel(writer, sheet_name='AggregationCheck', index=False)
                
#                 # Apply conditional formatting
#                 agg_sheet = writer.sheets['AggregationCheck']
#                 for idx, result in enumerate(agg_df['Result'], start=2):
#                     cell = agg_sheet.cell(row=idx, column=5)
#                     self._style_excel_cell(cell, result == 'PASS')

#                 # 3. Distinct Check Tab
#                 # Create worksheet
#                 distinct_sheet = writer.book.create_sheet('DistinctCheck')
                
#                 # Write headers
#                 headers = ['Column', 'Source Distinct Count', 'Target Distinct Count', 'Count Match', 'Values Match', 'Source Values', 'Target Values']
#                 for col_idx, header in enumerate(headers, start=1):
#                     cell = distinct_sheet.cell(row=1, column=col_idx, value=header)
#                     cell.font = Font(bold=True)
                
#                 # Get non-numeric columns
#                 non_numeric_cols = source_df.select_dtypes(exclude=['number']).columns
                
#                 # Process each column
#                 row_idx = 2
#                 for col in non_numeric_cols:
#                     # Get source values
#                     source_vals = sorted(source_df[col].fillna('NULL').astype(str).unique())
#                     source_count = len(source_vals)
#                     source_display = ', '.join(source_vals[:10])
#                     if source_count > 10:
#                         source_display += '...'
                    
#                     # Get target values
#                     if col in target_df.columns:
#                         target_vals = sorted(target_df[col].fillna('NULL').astype(str).unique())
#                         target_count = len(target_vals)
#                         target_display = ', '.join(target_vals[:10])
#                         if target_count > 10:
#                             target_display += '...'
#                     else:
#                         target_vals = []
#                         target_count = 0
#                         target_display = 'Column not found'
                    
#                     # Compare values
#                     count_match = 'PASS' if source_count == target_count else 'FAIL'
#                     values_match = 'PASS' if set(source_vals) == set(target_vals) else 'FAIL'
                    
#                     # Write row
#                     values = [str(col), source_count, target_count, count_match, values_match, source_display, target_display]
#                     for col_idx, value in enumerate(values, start=1):
#                         cell = distinct_sheet.cell(row=row_idx, column=col_idx, value=value)
#                         if col_idx in [4, 5]:  # Format match columns
#                             fill_color = PatternFill(start_color='90EE90' if value == 'PASS' else 'FFB6C1', 
#                                                    end_color='90EE90' if value == 'PASS' else 'FFB6C1',
#                                                    fill_type='solid')
#                             cell.fill = fill_color
#                     row_idx += 1
                
#                 # Write message if no columns processed
#                 if row_idx == 2:
#                     distinct_sheet.cell(row=2, column=1, value='No non-numeric columns found')
                
#                 # Adjust column widths
#                 for col_idx in range(1, len(headers) + 1):
#                     distinct_sheet.column_dimensions[get_column_letter(col_idx)].width = 20
                
#             logger.info(f"Enhanced regression report generated: {report_path}")
#             return str(report_path)
            
#         except Exception as e:
#             logger.error(f"Error generating regression report: {str(e)}")
#             raise

#     def generate_difference_report(self, source_df: pd.DataFrame, target_df: pd.DataFrame, 
#                                  join_columns: List[str]) -> str:
#         """Generate enhanced side-by-side difference report."""
#         try:
#             if source_df.empty or target_df.empty:
#                 logger.info("No data to compare in difference report")
#                 return None

#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             report_path = self.output_dir / f"DifferenceReport_{timestamp}.xlsx"

#             # Constants for Excel limitations
#             MAX_ROWS = 1000000  # Excel's limit is 1,048,576 rows
#             CHUNK_SIZE = 900000  # Slightly less than max to account for headers and formatting
            
#             # Merge dataframes to identify differences
#             merged = source_df.merge(target_df, on=join_columns, how='outer', 
#                                    indicator=True, suffixes=('_source', '_target'))
            
#             # Create comparison status column
#             merged['Status'] = merged['_merge'].map({
#                 'left_only': 'Deleted',
#                 'right_only': 'Inserted',
#                 'both': 'Updated'
#             })
            
#             # Remove the merge indicator column
#             merged = merged.drop('_merge', axis=1)
            
#             # If there are no differences
#             if merged['Status'].isin(['Updated']).all():
#                 logger.info("No differences found between source and target")
#                 with open(report_path, 'w') as f:
#                     f.write("No differences found between source and target datasets.")
#                 return str(report_path)
            
#             # Define status colors
#             status_colors = {
#                 'Deleted': 'FFB6C1',     # Light pink
#                 'Left Only': 'FFB6C1',   # Light pink
#                 'Inserted': '90EE90',    # Light green
#                 'Right Only': '90EE90',  # Light green
#                 'Updated': 'FFD700'      # Gold
#             }
            
#             # Calculate number of chunks needed
#             total_rows = len(merged)
#             num_chunks = (total_rows - 1) // CHUNK_SIZE + 1
            
#             # Save to Excel with formatting, splitting into multiple sheets if necessary
#             with pd.ExcelWriter(str(report_path), engine='openpyxl') as writer:
#                 for chunk_idx in range(num_chunks):
#                     start_idx = chunk_idx * CHUNK_SIZE
#                     end_idx = min((chunk_idx + 1) * CHUNK_SIZE, total_rows)
                    
#                     # Get chunk of data
#                     chunk = merged.iloc[start_idx:end_idx]
                    
#                     # Create sheet name
#                     sheet_name = 'Differences' if num_chunks == 1 else f'Differences_{chunk_idx + 1}'
                    
#                     # Write chunk to Excel
#                     chunk.to_excel(writer, sheet_name=sheet_name, index=False)
                    
#                     # Apply conditional formatting to chunk
#                     worksheet = writer.sheets[sheet_name]
                    
#                     # Apply formatting based on status
#                     for idx, status in enumerate(chunk['Status'], start=2):
#                         cell = worksheet.cell(row=idx, column=chunk.columns.get_loc('Status') + 1)
#                         cell.fill = PatternFill(
#                             start_color=status_colors.get(status, 'FFFFFF'),
#                             end_color=status_colors.get(status, 'FFFFFF'),
#                             fill_type='solid'
#                         )
                    
#                     # Add summary at top of each sheet
#                     summary_data = chunk['Status'].value_counts().to_dict()
#                     worksheet.cell(row=1, column=len(chunk.columns) + 2, value="Summary:")
#                     for i, (status, count) in enumerate(summary_data.items(), start=2):
#                         worksheet.cell(row=i, column=len(chunk.columns) + 2, value=f"{status}: {count}")
                
#                 # Add summary sheet
#                 summary_df = pd.DataFrame({
#                     'Description': [
#                         'Total Rows',
#                         'Number of Sheets',
#                         'Rows per Sheet',
#                         'Deleted Records',
#                         'Inserted Records',
#                         'Updated Records'
#                     ],
#                     'Value': [
#                         total_rows,
#                         num_chunks,
#                         CHUNK_SIZE,
#                         sum(merged['Status'] == 'Deleted'),
#                         sum(merged['Status'] == 'Inserted'),
#                         sum(merged['Status'] == 'Updated')
#                     ]
#                 })
#                 summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
#             logger.info(f"Difference report generated: {report_path}")
#             return str(report_path)
            
#         except Exception as e:
#             logger.error(f"Error generating difference report: {str(e)}")
#             raise

#     def generate_datacompy_report(self, comparison_results: Dict[str, Any]) -> str:
#         """Generate DataCompy report."""
#         try:
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             report_path = self.output_dir / f"DataCompy_{timestamp}.txt"
            
#             with open(report_path, 'w') as f:
#                 f.write(comparison_results.get('datacompy_report', 'No DataCompy report available'))
            
#             logger.info(f"DataCompy report generated: {report_path}")
#             return str(report_path)
            
#         except Exception as e:
#             logger.error(f"Error generating DataCompy report: {str(e)}")
#             raise

#     def generate_ydata_report(self, comparison_results: Dict[str, Any]) -> str:
#         """Generate Y-Data Profiling report."""
#         try:
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             report_path = self.output_dir / f"YDataProfile_{timestamp}.html"
            
#             # Save the Y-Data Profile HTML report
#             with open(report_path, 'w') as f:
#                 f.write(comparison_results.get('ydata_report', '<h1>No Y-Data Profile report available</h1>'))
            
#             logger.info(f"Y-Data Profile report generated: {report_path}")
#             return str(report_path)
            
#         except Exception as e:
#             logger.error(f"Error generating Y-Data Profile report: {str(e)}")
#             raise

#     def create_report_archive(self, report_paths: Dict[str, str]) -> str:
#         """Create a ZIP archive containing all reports."""
#         try:
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             zip_path = self.output_dir / f"all_reports_{timestamp}.zip"
            
#             with zipfile.ZipFile(str(zip_path), 'w') as zipf:
#                 for report_type, path in report_paths.items():
#                     if path and os.path.exists(path):
#                         # Add file to zip with its original name
#                         zipf.write(path, os.path.basename(path))
                        
#                         # For HTML reports, also add any associated resources
#                         if path.endswith('.html'):
#                             resources_dir = Path(path).parent / 'resources'
#                             if resources_dir.exists():
#                                 for resource in resources_dir.rglob('*'):
#                                     if resource.is_file():
#                                         zipf.write(
#                                             resource,
#                                             os.path.join('resources', resource.relative_to(resources_dir))
#                                         )
            
#             logger.info(f"Report archive created: {zip_path}")
#             return str(zip_path)
            
#         except Exception as e:
#             logger.error(f"Error creating report archive: {str(e)}")
#             raise

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
            aggs.append({
                'Column': col,
                'Sum': df[col].sum(),
                'Mean': df[col].mean(),
                'Min': df[col].min(),
                'Max': df[col].max(),
                'StdDev': df[col].std()
            })
        return pd.DataFrame(aggs)


    def generate_regression_report(self, comparison_results: Dict[str, Any], 
                                 source_df: pd.DataFrame, 
                                 target_df: pd.DataFrame) -> str:
        """Generate enhanced regression report with multiple checks."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_dir / f"regression_report_{timestamp}.xlsx"
            
            with pd.ExcelWriter(str(report_path), engine='openpyxl') as writer:
                # 1. Count Check Tab
                count_data = {
                    'Metric': ['Source Count', 'Target Count'],
                    'Value': [len(source_df), len(target_df)],
                    'Result': ['BASELINE', 'PASS' if len(source_df) == len(target_df) else 'FAIL']
                }
                count_df = pd.DataFrame(count_data)
                count_df.to_excel(writer, sheet_name='CountCheck', index=False)
                
                # Apply conditional formatting
                count_sheet = writer.sheets['CountCheck']
                for idx, result in enumerate(count_df['Result'], start=2):
                    if result != 'BASELINE':
                        cell = count_sheet.cell(row=idx, column=3)
                        self._style_excel_cell(cell, result == 'PASS')

                # 2. Aggregation Check Tab
                numeric_cols = source_df.select_dtypes(include=[np.number]).columns
                source_aggs = self._calculate_aggregations(source_df, numeric_cols)
                target_aggs = self._calculate_aggregations(target_df, numeric_cols)
                
                agg_comparison = []
                for col in numeric_cols:
                    source_row = source_aggs[source_aggs['Column'] == col].iloc[0]
                    target_row = target_aggs[target_aggs['Column'] == col].iloc[0]
                    
                    for metric in ['Sum', 'Mean', 'Min', 'Max', 'StdDev']:
                        matches = np.isclose(source_row[metric], target_row[metric], rtol=1e-05)
                        agg_comparison.append({
                            'Column': col,
                            'Metric': metric,
                            'Source': source_row[metric],
                            'Target': target_row[metric],
                            'Result': 'PASS' if matches else 'FAIL'
                        })
                
                agg_df = pd.DataFrame(agg_comparison)
                agg_df.to_excel(writer, sheet_name='AggregationCheck', index=False)
                
                # Apply conditional formatting
                agg_sheet = writer.sheets['AggregationCheck']
                for idx, result in enumerate(agg_df['Result'], start=2):
                    cell = agg_sheet.cell(row=idx, column=5)
                    self._style_excel_cell(cell, result == 'PASS')

                # 3. Distinct Check Tab
                # Create worksheet
                distinct_sheet = writer.book.create_sheet('DistinctCheck')
                
                # Write headers
                headers = ['Column', 'Source Distinct Count', 'Target Distinct Count', 'Count Match', 'Values Match', 'Source Values', 'Target Values']
                for col_idx, header in enumerate(headers, start=1):
                    cell = distinct_sheet.cell(row=1, column=col_idx, value=header)
                    cell.font = Font(bold=True)
                
                # Get non-numeric columns
                non_numeric_cols = source_df.select_dtypes(exclude=['number']).columns
                
                # Process each column
                row_idx = 2
                for col in non_numeric_cols:
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
                    count_match = 'PASS' if source_count == target_count else 'FAIL'
                    values_match = 'PASS' if set(source_vals) == set(target_vals) else 'FAIL'
                    
                    # Write row
                    values = [str(col), source_count, target_count, count_match, values_match, source_display, target_display]
                    for col_idx, value in enumerate(values, start=1):
                        cell = distinct_sheet.cell(row=row_idx, column=col_idx, value=value)
                        if col_idx in [4, 5]:  # Format match columns
                            fill_color = PatternFill(start_color='90EE90' if value == 'PASS' else 'FFB6C1', 
                                                   end_color='90EE90' if value == 'PASS' else 'FFB6C1',
                                                   fill_type='solid')
                            cell.fill = fill_color
                    row_idx += 1
                
                # Write message if no columns processed
                if row_idx == 2:
                    distinct_sheet.cell(row=2, column=1, value='No non-numeric columns found')
                
                # Adjust column widths
                for col_idx in range(1, len(headers) + 1):
                    distinct_sheet.column_dimensions[get_column_letter(col_idx)].width = 20
                
            logger.info(f"Enhanced regression report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating regression report: {str(e)}")
            raise

    def generate_difference_report(self, source_df: pd.DataFrame, target_df: pd.DataFrame, 
                                 join_columns: List[str]) -> str:
        """Generate enhanced side-by-side difference report."""
        try:
            if source_df.empty or target_df.empty:
                logger.info("No data to compare in difference report")
                return None

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_dir / f"DifferenceReport_{timestamp}.xlsx"

            # Constants for Excel limitations and chunking
            MAX_ROWS = 500000  # Half of Excel's limit for better performance
            CHUNK_SIZE = 450000  # Slightly less than max for headers and formatting
            
            # Process data in chunks to avoid memory issues
            dfs_to_process = []
            for start_idx in range(0, len(source_df), CHUNK_SIZE):
                # Get chunks of both dataframes
                source_chunk = source_df.iloc[start_idx:start_idx + CHUNK_SIZE]
                
                # Find corresponding rows in target using join columns
                chunk_merged = source_chunk.merge(
                    target_df, 
                    on=join_columns, 
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
            
            # If there are no differences
            if not dfs_to_process:
                logger.info("No differences found between source and target")
                with open(report_path, 'w') as f:
                    f.write("No differences found between source and target datasets.")
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
        """Generate DataCompy report with chunking for large datasets."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_dir / f"DataCompy_{timestamp}.txt"
            
            # Constants for chunking
            CHUNK_SIZE = 100000  # Process 100k rows at a time
            MAX_SAMPLE_ROWS = 5  # Maximum number of sample rows to show
            
            with open(report_path, 'w') as f:
                # Write header
                f.write("DataCompy Comparison Report\n")
                f.write("=" * 50 + "\n\n")
                
                # Write summary section
                f.write("Summary:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Source rows: {comparison_results['row_counts']['source_count']}\n")
                f.write(f"Target rows: {comparison_results['row_counts']['target_count']}\n")
                
                # Write column analysis
                f.write("\nColumn Analysis:\n")
                f.write("-" * 20 + "\n")
                for col, summary in comparison_results['column_summary'].items():
                    f.write(f"\nColumn: {col}\n")
                    f.write(f"Source null count: {summary['source_null_count']}\n")
                    f.write(f"Target null count: {summary['target_null_count']}\n")
                    f.write(f"Source unique count: {summary['source_unique_count']}\n")
                    f.write(f"Target unique count: {summary['target_unique_count']}\n")
                    if 'source_sum' in summary:
                        f.write(f"Source sum: {summary['source_sum']}\n")
                        f.write(f"Target sum: {summary['target_sum']}\n")
                        f.write(f"Source mean: {summary['source_mean']}\n")
                        f.write(f"Target mean: {summary['target_mean']}\n")
                
                # Write unmatched rows in chunks
                if len(comparison_results['source_unmatched_rows']) > 0:
                    f.write("\nUnmatched Rows in Source:\n")
                    f.write("-" * 20 + "\n")
                    
                    # Process source unmatched rows in chunks
                    source_unmatched = comparison_results['source_unmatched_rows']
                    for start_idx in range(0, len(source_unmatched), CHUNK_SIZE):
                        chunk = source_unmatched.iloc[start_idx:start_idx + CHUNK_SIZE]
                        if start_idx == 0:  # Only show sample from first chunk
                            sample = chunk.head(MAX_SAMPLE_ROWS).to_string()
                            f.write(f"Sample rows (showing {MAX_SAMPLE_ROWS} of {len(source_unmatched)}):\n")
                            f.write(sample + "\n")
                
                if len(comparison_results['target_unmatched_rows']) > 0:
                    f.write("\nUnmatched Rows in Target:\n")
                    f.write("-" * 20 + "\n")
                    
                    # Process target unmatched rows in chunks
                    target_unmatched = comparison_results['target_unmatched_rows']
                    for start_idx in range(0, len(target_unmatched), CHUNK_SIZE):
                        chunk = target_unmatched.iloc[start_idx:start_idx + CHUNK_SIZE]
                        if start_idx == 0:  # Only show sample from first chunk
                            sample = chunk.head(MAX_SAMPLE_ROWS).to_string()
                            f.write(f"Sample rows (showing {MAX_SAMPLE_ROWS} of {len(target_unmatched)}):\n")
                            f.write(sample + "\n")
                
                # Write overall status
                f.write("\nOverall Status:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Rows match: {'Yes' if comparison_results['rows_match'] else 'No'}\n")
                f.write(f"Columns match: {'Yes' if comparison_results['columns_match'] else 'No'}\n")
                f.write(f"Data matches: {'Yes' if comparison_results['match_status'] else 'No'}\n")
            
            logger.info(f"DataCompy report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating DataCompy report: {str(e)}")
            raise

    def generate_ydata_report(self, comparison_results: Dict[str, Any]) -> str:
        """Generate Y-Data Profiling report."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_dir / f"YDataProfile_{timestamp}.html"
            
            # Save the Y-Data Profile HTML report
            with open(report_path, 'w') as f:
                f.write(comparison_results.get('ydata_report', '<h1>No Y-Data Profile report available</h1>'))
            
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

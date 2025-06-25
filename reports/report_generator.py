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
                    'Result': ['PASS', 'PASS' if len(source_df) == len(target_df) else 'FAIL']
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

            # Constants for Excel limitations - reduced for better performance
            MAX_ROWS = 100000  # Significantly reduced from Excel's limit for stability
            CHUNK_SIZE = 90000  # Slightly less than max for headers and formatting
            
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

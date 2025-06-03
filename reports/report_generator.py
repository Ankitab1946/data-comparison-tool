"""Report generation utilities for the Data Comparison Tool."""
import pandas as pd
import os
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_regression_report(self, comparison_results: Dict[str, Any]) -> str:
        """
        Generate a regression report from comparison results.
        
        Args:
            comparison_results: Results from comparison engine
            
        Returns:
            Path to the generated report file
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_dir / f"regression_report_{timestamp}.xlsx"
            
            with pd.ExcelWriter(str(report_path), engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    'Metric': ['Rows Match', 'Columns Match', 'Overall Match', 'Source Row Count', 'Target Row Count'],
                    'Value': [
                        comparison_results.get('rows_match', False),
                        comparison_results.get('columns_match', False),
                        comparison_results.get('match_status', False),
                        comparison_results.get('row_counts', {}).get('source_count', 0),
                        comparison_results.get('row_counts', {}).get('target_count', 0)
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Column summary sheet
                if 'column_summary' in comparison_results:
                    column_df = pd.DataFrame(comparison_results['column_summary']).T
                    column_df.to_excel(writer, sheet_name='Column_Summary')
                
                # Distinct values sheet
                if 'distinct_values' in comparison_results:
                    distinct_data = []
                    for col, data in comparison_results['distinct_values'].items():
                        distinct_data.append({
                            'Column': col,
                            'Source_Distinct_Count': data.get('source_count', 0),
                            'Target_Distinct_Count': data.get('target_count', 0),
                            'Values_Match': data.get('matching', False)
                        })
                    if distinct_data:
                        pd.DataFrame(distinct_data).to_excel(writer, sheet_name='Distinct_Values', index=False)
            
            logger.info(f"Regression report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating regression report: {str(e)}")
            raise
    
    def generate_difference_report(self, differences: pd.DataFrame) -> str:
        """
        Generate a difference report.
        
        Args:
            differences: DataFrame containing differences
            
        Returns:
            Path to the generated report file
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_dir / f"differences_report_{timestamp}.xlsx"
            
            differences.to_excel(str(report_path), index=False)
            
            logger.info(f"Difference report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating difference report: {str(e)}")
            raise
    
    def create_report_archive(self, report_paths: Dict[str, str]) -> str:
        """
        Create a ZIP archive containing all reports.
        
        Args:
            report_paths: Dictionary of report types and their file paths
            
        Returns:
            Path to the ZIP archive
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            zip_path = self.output_dir / f"all_reports_{timestamp}.zip"
            
            with zipfile.ZipFile(str(zip_path), 'w') as zipf:
                for report_type, path in report_paths.items():
                    if os.path.exists(path):
                        zipf.write(path, os.path.basename(path))
            
            logger.info(f"Report archive created: {zip_path}")
            return str(zip_path)
            
        except Exception as e:
            logger.error(f"Error creating report archive: {str(e)}")
            raise

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute_query(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        try:
            return df.query(query)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def zip_report_file(self, report_file_path: str) -> str:
        zip_path = str(Path(report_file_path).with_suffix('.zip'))
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(report_file_path, arcname=Path(report_file_path).name)
        return zip_path

    def generate_difference_report(self, source_df: pd.DataFrame, target_df: pd.DataFrame,
                                   join_columns: List[str], source_query: str = None,
                                   target_query: str = None) -> str:
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_dir / f"DifferenceReport_{timestamp}.xlsx"

            if source_query:
                source_df = self.execute_query(source_df, source_query)
            if target_query:
                target_df = self.execute_query(target_df, target_query)

            source_df.columns = [col.lower() for col in source_df.columns]
            target_df.columns = [col.lower() for col in target_df.columns]
            join_columns = [col.lower() for col in join_columns]

            missing_cols = [col for col in join_columns if col not in source_df.columns or col not in target_df.columns]
            if missing_cols:
                raise ValueError(f"Join columns missing: {', '.join(missing_cols)}")

            target_df = target_df.rename(columns={col: col for col in join_columns})

            chunk_size = 100000
            total_rows = len(source_df)
            chunk_diffs = []

            for start in range(0, total_rows, chunk_size):
                s_chunk = source_df.iloc[start:start+chunk_size]
                merged = s_chunk.merge(target_df, on=join_columns, how='outer', indicator=True, suffixes=('_src', '_tgt'))
                merged['__status'] = merged['_merge'].map({
                    'left_only': 'Only in Source',
                    'right_only': 'Only in Target',
                    'both': 'Match'
                })
                diff_chunk = merged[merged['_merge'] != 'both']
                chunk_diffs.append(diff_chunk.drop(columns=['_merge']))

            with pd.ExcelWriter(str(report_path), engine='xlsxwriter') as writer:
                for i, chunk in enumerate(chunk_diffs):
                    if chunk.empty:
                        continue
                    max_rows = 100000
                    for j in range(0, len(chunk), max_rows):
                        sheet_chunk = chunk.iloc[j:j+max_rows]
                        sheet_name = f'diff_{i+1}_{j//max_rows+1}'
                        sheet_chunk.to_excel(writer, sheet_name=sheet_name[:31], index=False)

                summary = {
                    "Total Source Records": len(source_df),
                    "Total Target Records": len(target_df),
                    "Join Columns": ', '.join(join_columns),
                    "Difference Chunks": len(chunk_diffs)
                }
                summary_df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

            zipped_report_path = self.zip_report_file(str(report_path))
            return zipped_report_path

        except Exception as e:
            logger.error(f"Failed to generate difference report: {e}")
            raise

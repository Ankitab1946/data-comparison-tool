from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
import zipfile
import logging
import hashlib

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

    def _hash_row(self, row_values) -> str:
        joined = '|'.join(str(v) for v in row_values)
        return hashlib.md5(joined.encode()).hexdigest()

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

            logger.info("Hashing join columns for source...")
            source_hashes = source_df[join_columns].apply(lambda row: self._hash_row(row.values), axis=1)
            source_df['__row_hash'] = source_hashes
            source_keys = set(source_hashes)

            logger.info("Hashing join columns for target...")
            target_hashes = target_df[join_columns].apply(lambda row: self._hash_row(row.values), axis=1)
            target_df['__row_hash'] = target_hashes
            target_keys = set(target_hashes)

            only_in_source_keys = source_keys - target_keys
            only_in_target_keys = target_keys - source_keys

            logger.info(f"Rows only in source: {len(only_in_source_keys)}")
            logger.info(f"Rows only in target: {len(only_in_target_keys)}")

            only_in_source_rows = source_df[source_df['__row_hash'].isin(only_in_source_keys)].drop(columns=['__row_hash'])
            only_in_target_rows = target_df[target_df['__row_hash'].isin(only_in_target_keys)].drop(columns=['__row_hash'])

            with pd.ExcelWriter(str(report_path), engine='xlsxwriter') as writer:
                max_rows = 100_000

                if not only_in_source_rows.empty:
                    for i in range(0, len(only_in_source_rows), max_rows):
                        chunk = only_in_source_rows.iloc[i:i+max_rows]
                        sheet_name = f'OnlyInSource_{i//max_rows+1}'
                        chunk.to_excel(writer, sheet_name=sheet_name[:31], index=False)

                if not only_in_target_rows.empty:
                    for i in range(0, len(only_in_target_rows), max_rows):
                        chunk = only_in_target_rows.iloc[i:i+max_rows]
                        sheet_name = f'OnlyInTarget_{i//max_rows+1}'
                        chunk.to_excel(writer, sheet_name=sheet_name[:31], index=False)

                summary = {
                    "Total Source Records": len(source_df),
                    "Total Target Records": len(target_df),
                    "Join Columns": ', '.join(join_columns),
                    "Rows Only In Source": len(only_in_source_rows),
                    "Rows Only In Target": len(only_in_target_rows)
                }
                summary_df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

            zipped_report_path = self.zip_report_file(str(report_path))
            return zipped_report_path

        except Exception as e:
            logger.error(f"Failed to generate difference report: {e}")
            raise

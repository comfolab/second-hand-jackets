from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_dataset(path: str | Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    """Select the consolidated analysis sheet, never silently use the first sheet.

    An explicit sheet is allowed for already consolidated workbooks. Automatic
    selection requires the canonical visual columns and regional rain scores.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    from garment_analysis.cleaning import standardize_column_names

    with pd.ExcelFile(path) as workbook:
        if sheet_name is None:
            candidates = []
            required = {"JN", "Face_Neck", "Membrane_Neck", "Seams_Neck", "Rain_Neck"}
            for sheet in workbook.sheet_names:
                columns = standardize_column_names(pd.read_excel(workbook, sheet, nrows=0)).columns
                if required.issubset(columns):
                    candidates.append(sheet)
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected one consolidated sheet, found {candidates}. "
                    "Select the analysis sheet explicitly with --sheet."
                )
            sheet_name = candidates[0]
        df = pd.read_excel(workbook, sheet_name=sheet_name)
    df.attrs["source_sheet"] = sheet_name
    return df

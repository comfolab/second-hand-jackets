from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_dataset(path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """
    Load the dataset from an Excel file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name)
    return df
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the credit card dataset from disk."""
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop identifier columns, keep numeric fields, and impute missing values."""
    cleaned_df = df.copy()
    id_like_columns = [
        column
        for column in cleaned_df.columns
        if column.upper() == "CUST_ID" or column.upper().endswith("_ID")
    ]
    cleaned_df = cleaned_df.drop(columns=id_like_columns, errors="ignore")

    numeric_df = cleaned_df.select_dtypes(include=["number"]).copy()
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
    return cleaned_df, numeric_df


def scale_features(df_numeric: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize numeric features for clustering."""
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_numeric)
    scaled_df = pd.DataFrame(
        scaled_array,
        columns=df_numeric.columns,
        index=df_numeric.index,
    )
    return scaled_df, scaler

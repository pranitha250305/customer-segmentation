"""Data validation and feature preparation for K-Means."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_features(
    data: pd.DataFrame, selected_columns: Sequence[str]
) -> tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """Validate, impute, and standardize the chosen columns.

    Missing and infinite values are treated as missing and replaced with each
    feature's median. The returned cleaned frame is in original units; the
    returned array is the standardized input suitable for K-Means.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("The dataset is empty or is not a valid table.")
    columns = list(selected_columns)
    if not columns:
        raise ValueError("Select at least one numerical feature for clustering.")
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Selected column(s) not found: {', '.join(missing_columns)}.")
    if len(set(columns)) != len(columns):
        raise ValueError("Each feature can only be selected once.")

    cleaned = pd.DataFrame(index=data.index)
    for column in columns:
        # Numeric-looking text (for example, values imported from a CSV) is
        # accepted. Non-numeric entries become missing and are median-imputed.
        values = pd.to_numeric(data[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        median = values.median()
        if pd.isna(median):
            raise ValueError(
                f"'{column}' has no usable numeric values. Choose another feature."
            )
        cleaned[column] = values.fillna(median).astype(float)

    if len(cleaned) < 2:
        raise ValueError("At least two customer rows are needed for clustering.")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cleaned)
    if not np.isfinite(scaled).all():
        raise ValueError("The selected features could not be standardized safely.")
    return scaled, scaler, cleaned


def default_feature_columns(data: pd.DataFrame) -> list[str]:
    """Choose numeric defaults while excluding likely identifier columns."""
    features: list[str] = []
    for column in data.select_dtypes(include=[np.number]).columns:
        normalized = str(column).strip().lower().replace("_", " ").replace("-", " ")
        words = normalized.split()
        looks_like_id = (
            normalized in {"id", "customerid", "customer id"}
            or normalized.endswith(" id")
            or normalized.endswith(" id number")
            or "identifier" in words
        )
        if not looks_like_id:
            features.append(column)
    return features

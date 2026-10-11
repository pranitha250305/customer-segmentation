"""K-Means fitting, validation, and elbow-method calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_customers(
    original_data: pd.DataFrame,
    scaled_features: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[KMeans, pd.DataFrame, dict[str, float | str | None]]:
    """Fit K-Means and return the model, a labelled copy, and evaluation info."""
    values = np.asarray(scaled_features)
    if values.ndim != 2 or values.shape[0] != len(original_data) or not len(values):
        raise ValueError("Feature data must have one valid row per customer.")
    if "Cluster" in original_data.columns:
        raise ValueError(
            "The dataset already has a 'Cluster' column. Rename it before clustering "
            "so the new labels do not replace existing data."
        )
    if not isinstance(n_clusters, (int, np.integer)) or isinstance(n_clusters, bool):
        raise ValueError("The number of clusters must be a whole number.")
    unique_points = np.unique(values, axis=0).shape[0]
    if n_clusters < 2:
        raise ValueError("Choose at least 2 clusters.")
    if n_clusters >= len(values):
        raise ValueError("The number of clusters must be smaller than the row count.")
    if n_clusters > unique_points:
        raise ValueError(
            f"Choose no more than {unique_points} clusters; the data has only "
            "that many distinct feature rows."
        )

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(values)
    labelled = original_data.copy()
    labelled["Cluster"] = labels

    evaluation: dict[str, float | str | None] = {
        "silhouette_score": None,
        "silhouette_note": "",
        "inertia": float(model.inertia_),
    }
    if 1 < len(np.unique(labels)) < len(values):
        try:
            evaluation["silhouette_score"] = float(silhouette_score(values, labels))
        except (ValueError, FloatingPointError) as error:
            evaluation["silhouette_note"] = f"The score could not be calculated: {error}"
    else:
        evaluation["silhouette_note"] = "A silhouette score is not defined for these clusters."
    return model, labelled, evaluation


def calculate_elbow(
    scaled_features: np.ndarray, max_k: int = 10, random_state: int = 42
) -> pd.DataFrame:
    """Calculate inertia for every valid K from 1 through the safe upper bound."""
    values = np.asarray(scaled_features)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("At least two rows of prepared features are needed for the elbow plot.")
    unique_points = np.unique(values, axis=0).shape[0]
    upper = min(int(max_k), len(values) - 1, unique_points)
    if upper < 1:
        raise ValueError("There are not enough distinct rows to calculate an elbow curve.")
    records = []
    for k in range(1, upper + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        model.fit(values)
        records.append({"K": k, "Inertia": float(model.inertia_)})
    return pd.DataFrame(records)

import numpy as np
import pandas as pd
import pytest

from src.clustering import calculate_elbow, cluster_customers
from src.preprocessing import prepare_features


def sample_data():
    return pd.DataFrame({"Age": [20, 21, 22, 60, 61, 62], "Income": [20, 22, 19, 80, 82, 78]})


def test_clustering_adds_labels_without_modifying_original_and_is_reproducible():
    original = sample_data()
    scaled, _, _ = prepare_features(original, original.columns.tolist())
    model_a, result_a, info_a = cluster_customers(original, scaled, 2)
    model_b, result_b, _ = cluster_customers(original, scaled, 2)
    assert "Cluster" not in original.columns
    assert "Cluster" in result_a.columns
    assert result_a["Cluster"].tolist() == result_b["Cluster"].tolist()
    assert np.array_equal(model_a.cluster_centers_, model_b.cluster_centers_)
    assert len(result_a["Cluster"].unique()) == 2
    assert "inertia" in info_a


@pytest.mark.parametrize("k", [1, 6, 9, 2.5])
def test_invalid_cluster_counts_are_rejected(k):
    data = sample_data()
    scaled, _, _ = prepare_features(data, data.columns.tolist())
    with pytest.raises(ValueError):
        cluster_customers(data, scaled, k)


def test_elbow_calculates_only_valid_k_values():
    data = sample_data()
    scaled, _, _ = prepare_features(data, data.columns.tolist())
    elbow = calculate_elbow(scaled, max_k=10)
    assert elbow["K"].tolist() == [1, 2, 3, 4, 5]
    assert (elbow["Inertia"] >= 0).all()

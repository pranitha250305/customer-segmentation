import numpy as np
import pandas as pd
import pytest

from src.preprocessing import default_feature_columns, prepare_features


def test_prepare_features_imputes_and_standardizes():
    data = pd.DataFrame({"CustomerID": [1, 2, 3], "Age": [20, np.nan, 40], "Income": [10, 20, 30]})
    scaled, scaler, cleaned = prepare_features(data, ["Age", "Income"])
    assert cleaned.loc[1, "Age"] == 30
    assert np.allclose(scaled.mean(axis=0), [0, 0])
    assert np.allclose(scaled.std(axis=0), [1, 1])
    assert scaler.n_features_in_ == 2


def test_numeric_text_and_infinity_are_handled():
    data = pd.DataFrame({"score": ["1", "bad", "inf", "5"]})
    scaled, _, cleaned = prepare_features(data, ["score"])
    assert cleaned["score"].tolist() == [1.0, 3.0, 3.0, 5.0]
    assert np.isfinite(scaled).all()


@pytest.mark.parametrize("columns", [[], ["missing"]])
def test_invalid_feature_selection_has_clear_error(columns):
    with pytest.raises(ValueError):
        prepare_features(pd.DataFrame({"Age": [20, 30]}), columns)


def test_all_non_numeric_feature_is_rejected():
    with pytest.raises(ValueError, match="no usable numeric values"):
        prepare_features(pd.DataFrame({"Gender": ["F", "M"]}), ["Gender"])


def test_defaults_exclude_identifiers():
    data = pd.DataFrame({"CustomerID": [1, 2], "Age": [20, 30], "Annual Income": [10, 20]})
    assert default_feature_columns(data) == ["Age", "Annual Income"]

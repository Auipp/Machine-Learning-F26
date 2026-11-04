import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.transformers import FeatureEngineer, TopNGrouper


def _synthetic_bookings():
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    return pd.DataFrame({
        "ArrivalDateMonth": months,
        "Children": [0, 1, np.nan, 0, 2, 0, 1, 0, 0, 1, 0, 0],
        "Country": ["PRT"] * 11 + [np.nan],
        "Agent": [9, 9, np.nan, 10, 10, 10, np.nan, 11, 11, 11, 12, np.nan],
        "Company": [np.nan] * 10 + [123, np.nan],
        "MarketSegment": ["Online TA"] * 11 + ["Undefined"],
        "DistributionChannel": ["TA/TO"] * 12,
        "StaysInWeekendNights": [1] * 12,
        "StaysInWeekNights": [2] * 12,
        "Adults": [2] * 12,
        "Babies": [0] * 12,
    })


def test_feature_engineer_fit_returns_self():
    transformer = FeatureEngineer()
    assert transformer.fit(_synthetic_bookings()) is transformer


def test_feature_engineer_months_are_mapped_to_one_through_twelve():
    result = FeatureEngineer().fit_transform(_synthetic_bookings())
    assert result["ArrivalDateMonth"].tolist() == list(range(1, 13))
    assert pd.api.types.is_numeric_dtype(result["ArrivalDateMonth"])


def test_feature_engineer_accepts_numeric_months():
    data = _synthetic_bookings()
    data["ArrivalDateMonth"] = list(range(1, 13))
    result = FeatureEngineer().fit_transform(data)
    assert result["ArrivalDateMonth"].tolist() == list(range(1, 13))


def test_feature_engineer_fills_children_missing_with_zero():
    result = FeatureEngineer().fit_transform(_synthetic_bookings())
    assert result["Children"].isna().sum() == 0
    assert result.loc[2, "Children"] == 0


def test_feature_engineer_fills_missing_country_and_unknown_tokens():
    data = _synthetic_bookings()
    data.loc[0, "Country"] = np.nan
    data.loc[1, "Country"] = "NULL"
    result = FeatureEngineer().fit_transform(data)
    assert result.loc[0, "Country"] == "Unknown"
    assert result.loc[1, "Country"] == "Unknown"


def test_feature_engineer_fills_missing_agent_with_noagent():
    data = _synthetic_bookings()
    data["Agent"] = np.nan
    result = FeatureEngineer().fit_transform(data)
    assert (result["Agent"] == "NoAgent").all()


def test_feature_engineer_agent_ids_are_strings_without_decimal_suffix():
    result = FeatureEngineer().fit_transform(_synthetic_bookings())
    assert result["Agent"].map(lambda value: isinstance(value, str)).all()
    numeric_agents = result.loc[result["Agent"] != "NoAgent", "Agent"]
    assert not numeric_agents.str.endswith(".0").any()


def test_feature_engineer_creates_has_agent_flag():
    result = FeatureEngineer().fit_transform(_synthetic_bookings())
    assert set(result["HasAgent"].unique()).issubset({0, 1})
    expected = (result["Agent"] != "NoAgent").astype(int)
    assert result["HasAgent"].equals(expected)


def test_feature_engineer_creates_has_company_flag_and_drops_company():
    result = FeatureEngineer().fit_transform(_synthetic_bookings())
    assert "Company" not in result.columns
    assert "HasCompany" in result.columns
    assert set(result["HasCompany"].unique()).issubset({0, 1})


def test_feature_engineer_creates_total_nights():
    data = _synthetic_bookings()
    result = FeatureEngineer().fit_transform(data)
    expected = data["StaysInWeekendNights"] + data["StaysInWeekNights"]
    np.testing.assert_array_equal(result["TotalNights"], expected)


def test_feature_engineer_creates_total_guests():
    data = _synthetic_bookings()
    result = FeatureEngineer().fit_transform(data)
    expected = data["Adults"] + data["Children"].fillna(0) + data["Babies"]
    np.testing.assert_array_equal(result["TotalGuests"], expected)


def test_feature_engineer_creates_optional_columns_when_absent():
    data = _synthetic_bookings().drop(
        columns=["Company", "Agent", "Children", "Country",
                 "MarketSegment", "DistributionChannel"]
    )
    result = FeatureEngineer().fit_transform(data)
    for col in ["HasCompany", "HasAgent", "Children", "Country",
                "Agent", "MarketSegment", "DistributionChannel"]:
        assert col in result.columns


def test_feature_engineer_returns_dataframe_and_feature_names():
    data = _synthetic_bookings()
    transformer = FeatureEngineer().fit(data)
    result = transformer.transform(data)
    assert isinstance(result, pd.DataFrame)
    assert list(transformer.get_feature_names_out()) == list(result.columns)


def _frequency_data():
    return pd.DataFrame({"category": ["A"] * 100 + ["B"] * 60 + ["C"] * 30 + ["D"] * 10})


def test_top_n_grouper_fit_returns_self():
    grouper = TopNGrouper(n=3)
    assert grouper.fit(_frequency_data()) is grouper


def test_top_n_grouper_retains_most_frequent_categories():
    grouper = TopNGrouper(n=3).fit(_frequency_data())
    assert grouper.top_categories_ == {"A", "B", "C"}


def test_top_n_grouper_respects_n():
    data = pd.DataFrame({"category": list("ABCDEFGHIJ") * 10})
    grouper = TopNGrouper(n=5).fit(data)
    assert len(grouper.top_categories_) == 5


def test_top_n_grouper_maps_rare_categories_to_other():
    data = _frequency_data()
    grouper = TopNGrouper(n=3).fit(data)
    result = grouper.transform(data).ravel()
    assert "D" not in result
    assert "Other" in result


def test_top_n_grouper_preserves_top_categories():
    data = _frequency_data()
    grouper = TopNGrouper(n=3).fit(data)
    result = set(grouper.transform(data).ravel())
    assert {"A", "B", "C"}.issubset(result)


def test_top_n_grouper_maps_unseen_inference_values_to_other():
    train = pd.DataFrame({"category": ["A"] * 50 + ["B"] * 30})
    test = pd.DataFrame({"category": ["A", "B", "BRAND_NEW", "NEVER_SEEN"]})
    grouper = TopNGrouper(n=2).fit(train)
    result = grouper.transform(test).ravel().tolist()
    assert result == ["A", "B", "Other", "Other"]


def test_top_n_grouper_output_is_two_dimensional_numpy():
    data = pd.DataFrame({"category": ["A", "B", "C"]})
    result = TopNGrouper(n=2).fit(data).transform(data)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 1)


def test_top_n_grouper_accepts_series():
    data = pd.Series(["A"] * 50 + ["B"] * 30 + ["C"] * 10)
    grouper = TopNGrouper(n=2).fit(data)
    assert grouper.transform(data).shape == (90, 1)


def test_top_n_grouper_supports_custom_other_label():
    data = pd.DataFrame({"category": ["A"] * 50 + ["Z"] * 5})
    result = TopNGrouper(n=1, other_label="RARE").fit(data).transform(data).ravel()
    assert "RARE" in result
    assert "Z" not in result


def test_top_n_grouper_tie_break_is_deterministic():
    data = pd.DataFrame({"category": ["x", "y", "z", "x", "y", "z"]})
    first = TopNGrouper(n=2).fit(data).top_categories_
    shuffled = TopNGrouper(n=2).fit(data.sample(frac=1, random_state=7)).top_categories_
    assert first == shuffled == {"x", "y"}


def test_top_n_grouper_transform_before_fit_raises():
    grouper = TopNGrouper(n=2)
    with pytest.raises(Exception):
        grouper.transform(pd.DataFrame({"category": ["A"]}))

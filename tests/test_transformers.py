import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from src.data.loader import load_raw, clean
from src.preprocessing.transformers import FeatureEngineer, TopNGrouper


def _sample():
    return clean(load_raw()).head(500).copy()


def test_fe_fit_returns_self():
    fe = FeatureEngineer()
    assert fe.fit(_sample()) is fe

def test_fe_month_is_numeric_1_to_12():
    out = FeatureEngineer().transform(_sample())
    assert pd.api.types.is_numeric_dtype(out["ArrivalDateMonth"])
    assert out["ArrivalDateMonth"].between(1, 12).all()

def test_fe_all_12_months_mapped():
    months = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"]
    df = pd.DataFrame({"ArrivalDateMonth": months,
                       "Children": [0]*12, "Country": ["PRT"]*12,
                       "Agent": [np.nan]*12, "Company": [np.nan]*12,
                       "StaysInWeekendNights": [1]*12, "StaysInWeekNights": [2]*12,
                       "Adults": [2]*12, "Babies": [0]*12})
    out = FeatureEngineer().transform(df)
    assert list(out["ArrivalDateMonth"]) == list(range(1, 13))

def test_fe_children_no_nan():
    assert FeatureEngineer().transform(_sample())["Children"].isna().sum() == 0

def test_fe_country_no_nan():
    assert FeatureEngineer().transform(_sample())["Country"].isna().sum() == 0

def test_fe_country_unknown_fills():
    df = _sample().copy()
    df.loc[df.index[:5], "Country"] = np.nan
    out = FeatureEngineer().transform(df)
    assert (out["Country"] == "Unknown").sum() == 5

def test_fe_agent_is_string():
    out = FeatureEngineer().transform(_sample())
    assert isinstance(out["Agent"].iloc[0], str)

def test_fe_agent_no_nan():
    assert FeatureEngineer().transform(_sample())["Agent"].isna().sum() == 0

def test_fe_agent_na_becomes_noagent():
    df = _sample().copy()
    df["Agent"] = np.nan
    out = FeatureEngineer().transform(df)
    assert (out["Agent"] == "NoAgent").all()

def test_fe_agent_numeric_ids_are_strings():
    out = FeatureEngineer().transform(_sample())
    non_noagent = out.loc[out["Agent"] != "NoAgent", "Agent"]
    assert not non_noagent.str.endswith(".0").any()

def test_fe_has_company_binary():
    out = FeatureEngineer().transform(_sample())
    assert "HasCompany" in out.columns
    assert set(out["HasCompany"].unique()).issubset({0, 1})

def test_fe_company_column_dropped():
    assert "Company" not in FeatureEngineer().transform(_sample()).columns

def test_fe_has_agent_binary():
    out = FeatureEngineer().transform(_sample())
    assert "HasAgent" in out.columns
    assert set(out["HasAgent"].unique()).issubset({0, 1})

def test_fe_has_agent_consistent_with_agent():
    out = FeatureEngineer().transform(_sample())
    assert (out["HasAgent"] == (out["Agent"] != "NoAgent").astype(int)).all()

def test_fe_total_nights_correct():
    df = _sample()
    out = FeatureEngineer().transform(df)
    expected = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    assert (out["TotalNights"].values == expected.values).all()

def test_fe_total_guests_correct():
    df = _sample()
    out = FeatureEngineer().transform(df)
    expected = df["Adults"] + df["Children"].fillna(0) + df["Babies"]
    assert np.allclose(out["TotalGuests"].values, expected.values)

def test_fe_transform_returns_dataframe():
    assert isinstance(FeatureEngineer().transform(_sample()), pd.DataFrame)


def _freq_data():
    return pd.DataFrame({"x": ["A"]*100 + ["B"]*60 + ["C"]*30 + ["D"]*10})

def test_tng_fit_returns_self():
    assert TopNGrouper(n=3).fit(_freq_data()[["x"]]) is not None

def test_tng_top_n_retained():
    g = TopNGrouper(n=3)
    g.fit(_freq_data()[["x"]])
    assert g.top_categories_ == {"A", "B", "C"}

def test_tng_n_respected():
    data = pd.DataFrame({"x": list("ABCDEFGHIJ") * 10})
    g = TopNGrouper(n=5).fit(data[["x"]])
    assert len(g.top_categories_) == 5

def test_tng_rare_becomes_other():
    g = TopNGrouper(n=3).fit(_freq_data()[["x"]])
    out = g.transform(_freq_data()[["x"]]).flatten()
    assert "D" not in out
    assert "Other" in out

def test_tng_top_cats_preserved():
    g = TopNGrouper(n=3).fit(_freq_data()[["x"]])
    out = g.transform(_freq_data()[["x"]]).flatten()
    assert "A" in out and "B" in out and "C" in out

def test_tng_unseen_at_inference():
    train = pd.DataFrame({"x": ["A"]*50 + ["B"]*30})
    test  = pd.DataFrame({"x": ["A", "B", "BRAND_NEW", "NEVER_SEEN"]})
    g = TopNGrouper(n=2).fit(train[["x"]])
    out = g.transform(test[["x"]]).flatten()
    assert "BRAND_NEW" not in out
    assert "NEVER_SEEN" not in out
    assert out[2] == "Other" and out[3] == "Other"

def test_tng_output_shape_2d():
    data = pd.DataFrame({"x": ["A", "B", "C"]})
    g = TopNGrouper(n=2).fit(data[["x"]])
    assert g.transform(data[["x"]]).shape == (3, 1)

def test_tng_output_is_numpy():
    data = pd.DataFrame({"x": ["A", "B"]})
    g = TopNGrouper(n=1).fit(data[["x"]])
    assert isinstance(g.transform(data[["x"]]), np.ndarray)

def test_tng_custom_other_label():
    data = pd.DataFrame({"x": ["A"]*50 + ["Z"]*5})
    g = TopNGrouper(n=1, other_label="RARE").fit(data[["x"]])
    out = g.transform(data[["x"]]).flatten()
    assert "RARE" in out and "Z" not in out

def test_tng_accepts_series():
    data = pd.Series(["A"]*50 + ["B"]*30 + ["C"]*10)
    g = TopNGrouper(n=2).fit(data)
    assert g.transform(data).shape[1] == 1


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    print(f"Running {len(tests)} transformer tests...\n")
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)

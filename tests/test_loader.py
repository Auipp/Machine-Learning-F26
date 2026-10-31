import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

import pandas as pd
from src.data.loader import load_raw, clean, split, load_and_split


def _raw():
    return load_raw()

def _cleaned():
    return clean(_raw())


def test_load_raw_shape():
    df = _raw()
    assert df.shape == (79330, 31), f"Expected (79330, 31), got {df.shape}"

def test_load_raw_column_names():
    df = _raw()
    expected = {"IsCanceled", "LeadTime", "ADR", "ArrivalDateMonth",
                "Agent", "Company", "Country", "ReservationStatus"}
    assert expected.issubset(set(df.columns))

def test_load_raw_null_sentinels_converted():
    df = _raw()
    for col in ["Agent", "Company"]:
        null_strings = (df[col].astype(str).str.strip().str.upper() == "NULL").sum()
        assert null_strings == 0, f"{col} still has 'NULL' strings"

def test_load_raw_no_leading_trailing_whitespace():
    df = _raw()
    for col in df.select_dtypes("object").columns:
        has_ws = (df[col].str.startswith(" ") | df[col].str.endswith(" ")).any()
        assert not has_ws, f"Column '{col}' has leading/trailing whitespace"

def test_load_raw_agent_is_nullable():
    df = _raw()
    assert df["Agent"].isna().sum() > 0

def test_load_raw_company_mostly_null():
    df = _raw()
    assert df["Company"].isna().mean() > 0.90

def test_load_raw_children_small_nulls():
    df = _raw()
    assert df["Children"].isna().sum() == 4

def test_load_raw_country_small_nulls():
    df = _raw()
    assert df["Country"].isna().sum() == 24


def test_clean_reduces_rows():
    assert len(_cleaned()) < len(_raw())

def test_clean_no_duplicates():
    assert _cleaned().duplicated().sum() == 0

def test_clean_no_zero_guests():
    df = _cleaned()
    children = df["Children"].fillna(0)
    assert ((df["Adults"] + children + df["Babies"]) == 0).sum() == 0

def test_clean_no_zero_nights():
    df = _cleaned()
    assert ((df["StaysInWeekendNights"] + df["StaysInWeekNights"]) == 0).sum() == 0

def test_clean_preserves_columns():
    assert _cleaned().shape[1] == 31

def test_clean_resets_index():
    df = _cleaned()
    assert list(df.index) == list(range(len(df)))


def test_split_sizes_sum_to_total():
    df = _cleaned()
    train, val, test = split(df)
    assert len(train) + len(val) + len(test) == len(df)

def test_split_test_proportion():
    df = _cleaned()
    _, _, test = split(df)
    assert abs(len(test) / len(df) - 0.15) < 0.01

def test_split_val_proportion():
    df = _cleaned()
    _, val, _ = split(df)
    assert abs(len(val) / len(df) - 0.15) < 0.01

def test_split_stratification():
    df = _cleaned()
    train, val, test = split(df)
    overall = df["IsCanceled"].mean()
    for name, part in [("train", train), ("val", val), ("test", test)]:
        assert abs(part["IsCanceled"].mean() - overall) < 0.01, \
            f"{name} cancellation rate deviates from overall"

def test_split_different_random_seeds():
    df = _cleaned()
    t1, _, _ = split(df, random_state=42)
    t2, _, _ = split(df, random_state=99)
    assert not t1.equals(t2)

def test_load_and_split_convenience():
    train, val, test = load_and_split()
    assert len(train) > 0 and len(val) > 0 and len(test) > 0


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    print(f"Running {len(tests)} loader tests...\n")
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

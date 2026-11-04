import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import (
    DATA_PATH, DUP_COL, clean, drop_aux, load_and_split, load_raw,
    sample_weights, split,
)

pytestmark = pytest.mark.skipif(not DATA_PATH.exists(), reason="H2.csv not found")


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def cleaned(raw):
    return clean(raw)


def test_load_raw_returns_dataframe(raw):
    assert isinstance(raw, pd.DataFrame)
    assert len(raw) > 0


def test_load_raw_has_required_columns(raw):
    required = {
        "IsCanceled", "ADR", "LeadTime", "ArrivalDateMonth",
        "Adults", "Children", "Babies", "StaysInWeekendNights",
        "StaysInWeekNights", "Country", "Agent", "Company",
    }
    assert required.issubset(raw.columns)


def test_load_raw_has_no_untrimmed_object_values(raw):
    for col in raw.select_dtypes(include="object"):
        values = raw[col].dropna().astype(str)
        assert not values.str.startswith(" ").any(), col
        assert not values.str.endswith(" ").any(), col


def test_clean_adds_duplicate_count(cleaned):
    assert DUP_COL in cleaned.columns
    assert pd.api.types.is_integer_dtype(cleaned[DUP_COL])
    assert (cleaned[DUP_COL] >= 1).all()


def test_clean_keeps_one_row_per_duplicate_group(cleaned):
    original_columns = [c for c in cleaned.columns if c != DUP_COL]
    assert not cleaned.duplicated(subset=original_columns).any()


def test_clean_preserves_source_columns(cleaned, raw):
    assert set(raw.columns).issubset(cleaned.columns)


def test_clean_removes_zero_guest_rows(cleaned):
    children = cleaned["Children"].fillna(0)
    guests = cleaned["Adults"] + children + cleaned["Babies"]
    assert (guests > 0).all()


def test_clean_removes_zero_night_rows(cleaned):
    nights = cleaned["StaysInWeekendNights"] + cleaned["StaysInWeekNights"]
    assert (nights > 0).all()


def test_clean_resets_index(cleaned):
    assert cleaned.index.equals(pd.RangeIndex(len(cleaned)))


def test_split_sizes_sum_to_total(cleaned):
    train, val, test = split(cleaned)
    assert len(train) + len(val) + len(test) == len(cleaned)


def test_split_proportions(cleaned):
    train, val, test = split(cleaned)
    total = len(cleaned)
    assert abs(len(test) / total - 0.15) < 0.01
    assert abs(len(val) / total - 0.15) < 0.01


def test_split_stratifies_cancellation_target(cleaned):
    train, val, test = split(cleaned)
    overall = cleaned["IsCanceled"].mean()
    for name, part in (("train", train), ("validation", val), ("test", test)):
        assert abs(part["IsCanceled"].mean() - overall) < 0.02, name


def test_split_is_reproducible(cleaned):
    first = split(cleaned, random_state=42)
    second = split(cleaned, random_state=42)
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)


def test_split_changes_with_random_seed(cleaned):
    first_train = split(cleaned, random_state=42)[0]
    second_train = split(cleaned, random_state=99)[0]
    assert not first_train.equals(second_train)


def test_sample_weights_formula(cleaned):
    weights_0 = sample_weights(cleaned, 0)
    weights_50 = sample_weights(cleaned, 50)
    weights_100 = sample_weights(cleaned, 100)

    assert (weights_0 == 1).all()
    assert (weights_50 == 1 + (cleaned[DUP_COL] - 1) * 0.5).all()
    assert (weights_100 == cleaned[DUP_COL]).all()


@pytest.mark.parametrize("bad_value", [-1, 100.1, 101, None])
def test_sample_weights_rejects_invalid_percentage(cleaned, bad_value):
    with pytest.raises((ValueError, TypeError)):
        sample_weights(cleaned, bad_value)


def test_drop_aux_removes_duplicate_helper(cleaned):
    result = drop_aux(cleaned)
    assert DUP_COL not in result.columns
    assert len(result) == len(cleaned)


def test_load_and_split_returns_three_nonempty_splits():
    train, val, test = load_and_split()
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    assert all(DUP_COL in part.columns for part in (train, val, test))

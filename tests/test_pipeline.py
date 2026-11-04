import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import DATA_PATH, load_and_split

from src.preprocessing.pipeline import (
    ADR_MAX,
    ADR_MIN,
    AUX_COLUMNS,
    CLUSTERING_FEATURES,
    JUDGMENT_COLUMNS,
    TASK_TARGETS,
    TASKS,
    build_preprocessor,
    filter_adr,
    filter_for_task,
    make_xy,
)

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="H2.csv not found"
)


@pytest.fixture(scope="module")
def splits():
    return load_and_split()


def _feature_contains_column(feature_names, column):
    """
    Check whether a transformed feature name belongs to an original column.

    Examples:
        num__ADR                  -> ADR
        low_cat__DepositType_X    -> DepositType
    """
    return any(
        name.endswith(f"__{column}")
        or f"__{column}_" in name
        for name in feature_names
    )


def test_filter_adr_keeps_only_inclusive_valid_range(splits):
    train = splits[0]

    result = filter_adr(train)

    assert result["ADR"].between(
        ADR_MIN,
        ADR_MAX,
        inclusive="both"
    ).all()


def test_filter_adr_does_not_modify_input(splits):
    train = splits[0]

    before = train.copy(deep=True)

    filter_adr(train)

    pd.testing.assert_frame_equal(train, before)


def test_filter_for_task_regression_filters_adr(splits):
    result = filter_for_task(splits[0], "regression")

    assert result["ADR"].between(ADR_MIN, ADR_MAX).all()


def test_filter_for_task_clustering_filters_adr(splits):
    result = filter_for_task(splits[0], "clustering")

    assert result["ADR"].between(ADR_MIN, ADR_MAX).all()


def test_filter_for_task_classification_keeps_all_rows(splits):
    train = splits[0]

    result = filter_for_task(train, "classification")

    assert len(result) == len(train)

    pd.testing.assert_frame_equal(result, train)



@pytest.mark.parametrize("task", ["invalid", "", None])
def test_invalid_task_raises_value_error(task):
    with pytest.raises(ValueError):
        build_preprocessor(task)



def test_make_xy_removes_target_and_auxiliary_columns(splits):
    train = splits[0]

    for task, target in TASK_TARGETS.items():
        X, y = make_xy(train, task)

        assert target not in X.columns

        assert all(
            col not in X.columns
            for col in AUX_COLUMNS
        )

        assert y.name == target
        assert len(X) == len(y)


def test_make_xy_clustering_has_no_target_and_returns_none_y(splits):
    X, y = make_xy(splits[0], "clustering")

    assert y is None

    assert all(
        col not in X.columns
        for col in AUX_COLUMNS
    )



@pytest.mark.parametrize("task", TASKS)
def test_preprocessor_output_is_2d_finite_and_stable_width(
    splits, task
):
    train, val, test = splits

    train = filter_for_task(train, task)
    val = filter_for_task(val, task)
    test = filter_for_task(test, task)

    X_train, _ = make_xy(train, task)
    X_val, _ = make_xy(val, task)
    X_test, _ = make_xy(test, task)

    preprocessor = build_preprocessor(task)

    train_out = preprocessor.fit_transform(X_train)
    val_out = preprocessor.transform(X_val)
    test_out = preprocessor.transform(X_test)

    assert isinstance(train_out, np.ndarray)

    assert train_out.ndim == 2

    assert train_out.shape[1] == val_out.shape[1] == test_out.shape[1]

    assert np.isfinite(train_out).all()
    assert np.isfinite(val_out).all()
    assert np.isfinite(test_out).all()



def test_clustering_uses_configured_feature_list(splits):
    X, _ = make_xy(
        filter_for_task(splits[0], "clustering"),
        "clustering"
    )

    preprocessor = build_preprocessor("clustering").fit(X)

    names = preprocessor.get_feature_names_out()

    assert len(names) == len(CLUSTERING_FEATURES)

    for feature in CLUSTERING_FEATURES:
        assert _feature_contains_column(names, feature)



def test_regression_target_not_in_features(splits):
    X, _ = make_xy(
        filter_for_task(splits[0], "regression"),
        "regression"
    )

    names = build_preprocessor("regression").fit(X).get_feature_names_out()

    assert not _feature_contains_column(names, "ADR")


def test_classification_target_not_in_features(splits):
    X, _ = make_xy(splits[0], "classification")

    names = (
        build_preprocessor("classification")
        .fit(X)
        .get_feature_names_out()
    )

    assert not _feature_contains_column(names, "IsCanceled")



def test_judgment_columns_excluded_by_default(splits):
    X, _ = make_xy(splits[0], "classification")

    names = (
        build_preprocessor("classification")
        .fit(X)
        .get_feature_names_out()
    )

    for col in JUDGMENT_COLUMNS["classification"]:
        assert not _feature_contains_column(names, col)


def test_judgment_columns_can_be_enabled(splits):
    X, _ = make_xy(splits[0], "classification")

    requested = tuple(JUDGMENT_COLUMNS["classification"])

    names = (
        build_preprocessor(
            "classification",
            include_judgment=requested
        )
        .fit(X)
        .get_feature_names_out()
    )

    for col in requested:
        assert _feature_contains_column(names, col), col


def test_regression_judgment_columns_can_be_enabled(splits):
    X, _ = make_xy(
        filter_for_task(splits[0], "regression"),
        "regression"
    )

    requested = tuple(JUDGMENT_COLUMNS["regression"])

    names = (
        build_preprocessor(
            "regression",
            include_judgment=requested
        )
        .fit(X)
        .get_feature_names_out()
    )

    for col in requested:
        assert _feature_contains_column(names, col), col


@pytest.mark.parametrize(
    "column",
    [
        "BookingChanges",
        "AssignedRoomType",
        "ReservationStatus"
    ]
)
def test_non_judgment_or_leakage_columns_cannot_be_enabled(column):
    with pytest.raises(ValueError):
        build_preprocessor(
            "classification",
            include_judgment=(column,)
        )



def test_extra_columns_are_ignored_by_whitelist(splits):
    train = splits[0]

    X, _ = make_xy(train, "classification")

    clean_pre = build_preprocessor("classification").fit(X)

    clean_output = clean_pre.transform(X)

    dirty = X.copy()

    dirty["dup_count"] = train["dup_count"].to_numpy()
    dirty["IsCanceled"] = train["IsCanceled"].to_numpy()
    dirty["ReservationStatus"] = "Check-Out"

    dirty_output = clean_pre.transform(dirty)

    assert np.allclose(clean_output, dirty_output)


def test_dup_count_is_not_a_configured_feature():
    assert "dup_count" in AUX_COLUMNS

    assert "dup_count" not in CLUSTERING_FEATURES

    for columns in JUDGMENT_COLUMNS.values():
        assert "dup_count" not in columns



def test_preprocessor_is_unfitted_when_built():
    preprocessor = build_preprocessor("classification")

    assert not hasattr(
        preprocessor.named_steps["engineer"],
        "feature_names_out_"
    )


def test_preprocessor_rejects_unknown_judgment_column():
    with pytest.raises(ValueError, match="not judgment columns"):
        build_preprocessor(
            "regression",
            include_judgment=("Company",)
        )
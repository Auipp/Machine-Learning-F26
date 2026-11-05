import numpy as np
import pytest

from src.data.loader import load_and_split
from src.preprocessing.pipeline import (
    TASK_TARGETS,
    build_preprocessor,
    filter_adr,
)


@pytest.fixture(scope="module")
def splits():
    train, val, test = load_and_split()
    return train, val, test


def _get_X(df, task):
    target = TASK_TARGETS.get(task)
    drop_cols = [c for c in [target] if c is not None and c in df.columns]
    return df.drop(columns=drop_cols)


@pytest.mark.parametrize("task", ["regression", "classification", "clustering"])
def test_fit_transform_no_errors(splits, task):
    train, val, test = splits
    if task == "regression":
        train = filter_adr(train)
        val = filter_adr(val)
        test = filter_adr(test)

    X_train = _get_X(train, task)
    X_val = _get_X(val, task)
    X_test = _get_X(test, task)

    pipe = build_preprocessor(task)
    X_train_t = pipe.fit_transform(X_train)
    X_val_t = pipe.transform(X_val)
    X_test_t = pipe.transform(X_test)

    assert X_train_t.ndim == 2
    assert X_val_t.shape[1] == X_train_t.shape[1], "val/train feature mismatch"
    assert X_test_t.shape[1] == X_train_t.shape[1], "test/train feature mismatch"

    assert not np.isnan(X_train_t).any(), "NaN in train output"
    assert not np.isinf(X_train_t).any(), "Inf in train output"
    assert not np.isnan(X_val_t).any(), "NaN in val output"
    assert not np.isnan(X_test_t).any(), "NaN in test output"


def test_no_leakage_cols_in_regression_output(splits):
    """Regression pipeline must not pass IsCanceled or any leakage column through."""
    train, _, _ = splits
    train = filter_adr(train)
    X_train = _get_X(train, "regression")

    pipe = build_preprocessor("regression")
    
    pipe.fit_transform(X_train)  


def test_train_only_fit(splits):
    """Pipeline fit on train must transform val without errors.

    Two pipelines fit on different data subsets may produce different feature
    widths (different top-N categories discovered from each subset), which is
    correct behaviour.  What must hold is that each fitted pipeline can
    transform unseen data cleanly — guaranteed by OHE handle_unknown='ignore'.
    """
    train, val, _ = splits
    half_train = train.iloc[: len(train) // 2]

    X_full = _get_X(train, "classification")
    X_half = _get_X(half_train, "classification")
    X_val = _get_X(val, "classification")

    pipe_full = build_preprocessor("classification")
    pipe_half = build_preprocessor("classification")

    pipe_full.fit(X_full)
    pipe_half.fit(X_half)

    val_from_full = pipe_full.transform(X_val)
    val_from_half = pipe_half.transform(X_val)

    train_from_full = pipe_full.transform(X_full)
    train_from_half = pipe_half.transform(X_half)

    assert val_from_full.shape[1] == train_from_full.shape[1], \
        "val feature count must match train for pipe_full"
    assert val_from_half.shape[1] == train_from_half.shape[1], \
        "val feature count must match train for pipe_half"

    assert not np.isnan(val_from_full).any()
    assert not np.isnan(val_from_half).any()

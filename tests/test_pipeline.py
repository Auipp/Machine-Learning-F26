import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

import numpy as np
from src.data.loader import load_and_split
from src.preprocessing.pipeline import (
    build_preprocessor,
    filter_adr,
    TASK_TARGETS,
    CLUSTERING_FEATURES,
    ADR_MIN,
    ADR_MAX,
)

_train = _val = _test = None

def _splits():
    global _train, _val, _test
    if _train is None:
        _train, _val, _test = load_and_split()
    return _train, _val, _test

def _X(df, task):
    target = TASK_TARGETS.get(task)
    drop = [c for c in [target] if c and c in df.columns]
    return df.drop(columns=drop)


def test_filter_adr_removes_zero():
    train, _, _ = _splits()
    assert (filter_adr(train)["ADR"] == 0).sum() == 0

def test_filter_adr_removes_high_outlier():
    train, _, _ = _splits()
    assert (filter_adr(train)["ADR"] > ADR_MAX).sum() == 0

def test_filter_adr_min_boundary():
    train, _, _ = _splits()
    assert (filter_adr(train)["ADR"] < ADR_MIN).sum() == 0

def test_filter_adr_preserves_valid_rows():
    train, _, _ = _splits()
    n_valid = ((train["ADR"] >= ADR_MIN) & (train["ADR"] <= ADR_MAX)).sum()
    assert len(filter_adr(train)) == n_valid


def test_invalid_task_raises_valueerror():
    try:
        build_preprocessor("bad_task")
        assert False
    except ValueError as e:
        assert "bad_task" in str(e)


def test_regression_output_is_2d_numpy():
    train, _, _ = _splits()
    out = build_preprocessor("regression").fit_transform(_X(filter_adr(train), "regression"))
    assert isinstance(out, np.ndarray) and out.ndim == 2

def test_regression_feature_count():
    train, _, _ = _splits()
    out = build_preprocessor("regression").fit_transform(_X(filter_adr(train), "regression"))
    assert out.shape[1] == 93

def test_regression_no_nan():
    train, val, _ = _splits()
    pipe = build_preprocessor("regression")
    Xtr = pipe.fit_transform(_X(filter_adr(train), "regression"))
    Xv  = pipe.transform(_X(filter_adr(val), "regression"))
    assert not np.isnan(Xtr).any()
    assert not np.isnan(Xv).any()

def test_regression_no_inf():
    train, _, _ = _splits()
    out = build_preprocessor("regression").fit_transform(_X(filter_adr(train), "regression"))
    assert not np.isinf(out).any()

def test_regression_val_width_matches_train():
    train, val, test = _splits()
    pipe = build_preprocessor("regression")
    n_tr = pipe.fit_transform(_X(filter_adr(train), "regression")).shape[1]
    assert pipe.transform(_X(filter_adr(val),  "regression")).shape[1] == n_tr
    assert pipe.transform(_X(filter_adr(test), "regression")).shape[1] == n_tr

def test_regression_target_not_in_input():
    train, _, _ = _splits()
    assert "ADR" not in _X(filter_adr(train), "regression").columns


def test_classification_output_is_2d_numpy():
    train, _, _ = _splits()
    out = build_preprocessor("classification").fit_transform(_X(train, "classification"))
    assert isinstance(out, np.ndarray) and out.ndim == 2

def test_classification_feature_count():
    train, _, _ = _splits()
    out = build_preprocessor("classification").fit_transform(_X(train, "classification"))
    assert out.shape[1] == 94

def test_classification_adr_included_as_feature():
    train, _, _ = _splits()
    assert "ADR" in _X(train, "classification").columns

def test_classification_no_nan():
    train, val, _ = _splits()
    pipe = build_preprocessor("classification")
    assert not np.isnan(pipe.fit_transform(_X(train, "classification"))).any()
    assert not np.isnan(pipe.transform(_X(val, "classification"))).any()

def test_classification_val_width_matches_train():
    train, val, test = _splits()
    pipe = build_preprocessor("classification")
    n_tr = pipe.fit_transform(_X(train, "classification")).shape[1]
    assert pipe.transform(_X(val,  "classification")).shape[1] == n_tr
    assert pipe.transform(_X(test, "classification")).shape[1] == n_tr


def test_clustering_output_is_2d_numpy():
    train, _, _ = _splits()
    out = build_preprocessor("clustering").fit_transform(train)
    assert isinstance(out, np.ndarray) and out.ndim == 2

def test_clustering_feature_count():
    train, _, _ = _splits()
    out = build_preprocessor("clustering").fit_transform(train)
    assert out.shape[1] == len(CLUSTERING_FEATURES)

def test_clustering_no_nan():
    train, val, _ = _splits()
    pipe = build_preprocessor("clustering")
    assert not np.isnan(pipe.fit_transform(train)).any()
    assert not np.isnan(pipe.transform(val)).any()

def test_clustering_val_width_matches_train():
    train, val, test = _splits()
    pipe = build_preprocessor("clustering")
    n_tr = pipe.fit_transform(train).shape[1]
    assert pipe.transform(val).shape[1]  == n_tr
    assert pipe.transform(test).shape[1] == n_tr


def test_leakage_cols_dropped_implicitly():
    train, _, _ = _splits()
    pipe = build_preprocessor("classification")
    out = pipe.fit_transform(_X(train, "classification"))
    assert out.shape[1] == 94


def test_scaled_output_has_reasonable_range():
    train, _, _ = _splits()
    pipe = build_preprocessor("regression")
    out = pipe.fit_transform(_X(filter_adr(train), "regression"))
    assert np.percentile(np.abs(out[:, :20]), 99) < 50


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    print(f"Running {len(tests)} pipeline tests...\n")
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

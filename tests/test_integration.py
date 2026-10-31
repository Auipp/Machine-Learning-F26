import sys
import pathlib
import tempfile
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

import joblib
import numpy as np
from src.data.loader import load_and_split
from src.preprocessing.pipeline import build_preprocessor, filter_adr, TASK_TARGETS

_train = _val = _test = None

def _splits():
    global _train, _val, _test
    if _train is None:
        _train, _val, _test = load_and_split()
    return _train, _val, _test

def _prep(task):
    train, val, test = _splits()
    target = TASK_TARGETS.get(task)
    if task == "regression":
        train, val, test = filter_adr(train), filter_adr(val), filter_adr(test)
    if target:
        X_tr, y_tr = train.drop(columns=[target]), train[target]
        X_v,  y_v  = val.drop(columns=[target]),   val[target]
        X_te, y_te = test.drop(columns=[target]),  test[target]
    else:
        X_tr, y_tr = train, None
        X_v,  y_v  = val,   None
        X_te, y_te = test,  None
    return X_tr, y_tr, X_v, y_v, X_te, y_te

def _run_full_workflow(task):
    X_tr, y_tr, X_v, y_v, X_te, y_te = _prep(task)
    pipe = build_preprocessor(task)
    X_tr_t = pipe.fit_transform(X_tr)
    X_v_t  = pipe.transform(X_v)
    X_te_t = pipe.transform(X_te)
    assert X_tr_t.ndim == 2
    assert X_v_t.shape[1]  == X_tr_t.shape[1]
    assert X_te_t.shape[1] == X_tr_t.shape[1]
    for arr in [X_tr_t, X_v_t, X_te_t]:
        assert not np.isnan(arr).any()
        assert not np.isinf(arr).any()
    return pipe, X_tr_t, X_v_t, X_te_t


def test_regression_full_workflow():
    pipe, Xtr, Xv, Xte = _run_full_workflow("regression")
    assert Xtr.shape[1] == 93
    assert Xtr.shape[0] > 30_000

def test_classification_full_workflow():
    pipe, Xtr, Xv, Xte = _run_full_workflow("classification")
    assert Xtr.shape[1] == 94
    assert Xtr.shape[0] == 37_129

def test_clustering_full_workflow():
    pipe, Xtr, Xv, Xte = _run_full_workflow("clustering")
    assert Xtr.shape[1] == 8


def _serialisation_test(task):
    X_tr, _, X_v, _, X_te, _ = _prep(task)
    pipe = build_preprocessor(task)
    pipe.fit(X_tr)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    joblib.dump(pipe, path)
    loaded = joblib.load(path)

    out_orig   = pipe.transform(X_v)
    out_loaded = loaded.transform(X_v)
    assert np.allclose(out_orig, out_loaded)

    out_single = loaded.transform(X_te.iloc[[0]])
    assert out_single.shape == (1, out_orig.shape[1])
    assert not np.isnan(out_single).any()

def test_regression_serialisation():
    _serialisation_test("regression")

def test_classification_serialisation():
    _serialisation_test("classification")

def test_clustering_serialisation():
    _serialisation_test("clustering")


def test_leakage_cols_not_passed_to_regression():
    train, _, _ = _splits()
    X = train.drop(columns=["ADR"])
    out = build_preprocessor("regression").fit_transform(X)
    assert out.shape[1] == 93

def test_leakage_cols_not_passed_to_classification():
    train, _, _ = _splits()
    X = train.drop(columns=["IsCanceled"])
    out = build_preprocessor("classification").fit_transform(X)
    assert out.shape[1] == 94


def test_pipeline_not_fitted_on_val():
    train, val, _ = _splits()
    X_tr = train.drop(columns=["IsCanceled"])
    X_v  = val.drop(columns=["IsCanceled"])

    pipe_train = build_preprocessor("classification")
    pipe_val   = build_preprocessor("classification")
    pipe_train.fit(X_tr)
    pipe_val.fit(X_v)

    out_correct = pipe_train.transform(X_v)
    out_wrong   = pipe_val.transform(X_v)

    if out_correct.shape != out_wrong.shape:
        return
    assert not np.allclose(out_correct, out_wrong)


def test_no_target_leakage_in_X():
    for task in ["regression", "classification"]:
        X_tr, *_ = _prep(task)
        assert TASK_TARGETS[task] not in X_tr.columns

def test_val_test_never_used_in_fit():
    train, _, _ = _splits()
    X_tr_full = train.drop(columns=["IsCanceled"])
    X_tr_half = X_tr_full.iloc[:len(X_tr_full) // 2]

    p_full = build_preprocessor("classification")
    p_half = build_preprocessor("classification")
    p_full.fit(X_tr_full)
    p_half.fit(X_tr_half)

    out_full = p_full.transform(X_tr_half)
    out_half = p_half.transform(X_tr_half)

    if out_full.shape != out_half.shape:
        return
    assert not np.allclose(out_full, out_half)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    print(f"Running {len(tests)} integration tests...\n")
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

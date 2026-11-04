import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import DATA_PATH, load_and_split
from src.preprocessing.pipeline import (
    TASKS,
    filter_for_task,
    make_xy,
    build_preprocessor,
)

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="H2.csv not found"
)


@pytest.fixture(scope="module")
def splits():
    return load_and_split()


def _task_data(splits, task):
    train, val, test = (
        filter_for_task(part, task)
        for part in splits
    )

    X_train, y_train = make_xy(train, task)
    X_val, y_val = make_xy(val, task)
    X_test, y_test = make_xy(test, task)

    return X_train, y_train, X_val, y_val, X_test, y_test


@pytest.mark.parametrize("task", TASKS)
def test_full_workflow_train_validation_test(splits, task):
    X_train, _, X_val, _, X_test, _ = _task_data(splits, task)

    preprocessor = build_preprocessor(task)

    train_out = preprocessor.fit_transform(X_train)
    val_out = preprocessor.transform(X_val)
    test_out = preprocessor.transform(X_test)

    assert train_out.ndim == val_out.ndim == test_out.ndim == 2

    assert train_out.shape[1] == val_out.shape[1] == test_out.shape[1]

    assert np.isfinite(train_out).all()
    assert np.isfinite(val_out).all()
    assert np.isfinite(test_out).all()


@pytest.mark.parametrize("task", TASKS)
def test_pipeline_serialization_round_trip(splits, task):
    X_train, _, X_val, _, X_test, _ = _task_data(splits, task)

    preprocessor = build_preprocessor(task).fit(X_train)
    expected = preprocessor.transform(X_val)

    # TemporaryDirectory avoids Windows file-locking issues
    # caused by reopening an active NamedTemporaryFile.
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "preprocessor.joblib"

        joblib.dump(preprocessor, file_path)
        loaded = joblib.load(file_path)

        actual = loaded.transform(X_val)

        assert np.allclose(expected, actual)

        one_row = loaded.transform(X_test.iloc[[0]])

        assert one_row.shape == (1, expected.shape[1])
        assert np.isfinite(one_row).all()


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_target_is_not_in_model_input(splits, task):
    X_train, _, *_ = _task_data(splits, task)

    target = "ADR" if task == "regression" else "IsCanceled"

    assert target not in X_train.columns


@pytest.mark.parametrize("task", TASKS)
def test_duplicate_count_never_appears_in_model_input_or_feature_names(
    splits, task
):
    X_train, _, X_val, _, _, _ = _task_data(splits, task)

    # dup_count must not be an input feature.
    assert "dup_count" not in X_train.columns

    preprocessor = build_preprocessor(task).fit(X_train)

    # It must not appear in the transformed feature names either.
    names = preprocessor.get_feature_names_out()

    assert not any("dup_count" in str(name).lower() for name in names)

    # Even if accidentally supplied, the ColumnTransformer whitelist
    # must ignore dup_count.
    dirty = X_val.assign(dup_count=999)

    clean_output = preprocessor.transform(X_val)
    dirty_output = preprocessor.transform(dirty)

    assert np.allclose(clean_output, dirty_output)


def test_validation_data_is_transformed_without_refitting(splits):
    X_train, _, X_val, _, _, _ = _task_data(
        splits, "classification"
    )

    preprocessor = build_preprocessor("classification").fit(X_train)

    before = preprocessor.transform(X_val)

    # Save the categories learned from training data.
    country_transformer = (
        preprocessor.named_steps["preprocessor"]
        .named_transformers_["country"]
        .named_steps["topngrouper"]
    )

    agent_transformer = (
        preprocessor.named_steps["preprocessor"]
        .named_transformers_["agent"]
        .named_steps["topngrouper"]
    )

    learned_before = {
        "country": set(country_transformer.top_categories_),
        "agent": set(agent_transformer.top_categories_),
    }

    # Transform validation data again.
    preprocessor.transform(X_val)
    after = preprocessor.transform(X_val)

    learned_after = {
        "country": set(country_transformer.top_categories_),
        "agent": set(agent_transformer.top_categories_),
    }

    assert np.allclose(before, after)
    assert learned_before == learned_after


def test_pipeline_fit_uses_only_the_data_passed_to_fit(splits):
    X_train, _, X_val, _, _, _ = _task_data(
        splits, "classification"
    )

    full_fit = build_preprocessor("classification").fit(X_train)

    half_train = X_train.iloc[:len(X_train) // 2]

    half_fit = build_preprocessor("classification").fit(half_train)

    # Get the categories learned by each pipeline.
    full_country = (
        full_fit.named_steps["preprocessor"]
        .named_transformers_["country"]
    )

    half_country = (
        half_fit.named_steps["preprocessor"]
        .named_transformers_["country"]
    )

    full_top = full_country.named_steps["topngrouper"].top_categories_
    half_top = half_country.named_steps["topngrouper"].top_categories_

    # Each pipeline must learn categories only from its own training data.
    assert full_top.issubset(set(X_train["Country"].astype(str)))

    assert half_top.issubset(
        set(half_train["Country"].astype(str))
    )

    # Validation transformations must not change fitted categories.
    full_fit.transform(X_val)
    half_fit.transform(X_val)

    assert full_top == (
        full_country.named_steps["topngrouper"].top_categories_
    )

    assert half_top == (
        half_country.named_steps["topngrouper"].top_categories_
    )
"""Shared data preparation for the prediction models.

The models consume the preprocessing branch directly:

    raw H2.csv -> load_and_split -> filter_for_task -> make_xy
               -> build_preprocessor -> estimator

The preprocessor is kept inside an sklearn Pipeline so it is fitted on the
training data only (including inside cross-validation) and saved together
with the model. Prediction data therefore goes through exactly the same
feature engineering as the data used for training.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

from src.data.loader import load_and_split, load_raw, sample_weights
from src.preprocessing.pipeline import (
    AUX_COLUMNS,
    TASK_TARGETS,
    TASKS,
    build_preprocessor,
    filter_for_task,
    make_xy,
)

SPLIT_NAMES = ("train", "val", "test")


@lru_cache(maxsize=1)
def default_splits():
    """Load, clean and split the raw dataset once per Python process."""
    return load_and_split()


def prepare_splits(task, duplicate_weight=0.0, splits=None):
    """Return ``(train, val, test)`` tuples of ``(X, y, sample_weight)``.

    Rows are filtered for the task first (the valid ADR range for regression
    and clustering), then ``make_xy`` removes the target and helper columns.
    Weights come from ``dup_count`` recorded by the loader; the default
    ``duplicate_weight=0`` means every unique booking counts once.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if splits is None:
        splits = default_splits()

    prepared = []
    for part in splits:
        filtered = filter_for_task(part, task)
        X, y = make_xy(filtered, task)
        weights = sample_weights(filtered, duplicate_weight)
        prepared.append((X, y, weights))
    return tuple(prepared)


def build_model_pipeline(task, estimator):
    """Attach the task's preprocessing pipeline to a model."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(task)),
            ("model", estimator),
        ]
    )


def fit_model(pipeline, X, y=None, sample_weight=None):
    """Fit a model pipeline, optionally using per-row training weights."""
    if sample_weight is None:
        pipeline.fit(X, y)
    else:
        pipeline.fit(X, y, model__sample_weight=sample_weight)
    return pipeline


def prediction_features(pred_set, task, data_dir):
    """Build model input for a split name, a DataFrame or a raw CSV path.

    ``pred_set`` may be ``"train"``, ``"val"`` or ``"test"`` to use the
    project split, or a DataFrame / CSV path containing raw booking columns.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")

    if isinstance(pred_set, pd.DataFrame):
        frame = pred_set.copy()
    elif isinstance(pred_set, str) and pred_set in SPLIT_NAMES:
        frame = default_splits()[SPLIT_NAMES.index(pred_set)].copy()
    else:
        frame = load_raw(_resolve_path(pred_set, data_dir))

    # Classification keeps every row; regression/clustering need a valid ADR
    # if the column is available. A prediction file may omit the target.
    if "ADR" in frame.columns or task == "classification":
        frame = filter_for_task(frame, task)

    target = TASK_TARGETS.get(task)
    if target is not None and target in frame.columns:
        X, _ = make_xy(frame, task)
    else:
        X = frame.drop(columns=[c for c in AUX_COLUMNS if c in frame.columns])
    return X


def load_model(path):
    """Load a saved model and make sure it contains its preprocessor."""
    model = load(path)
    if not isinstance(model, Pipeline) or "preprocessor" not in model.named_steps:
        raise RuntimeError(
            f"{path} was saved before the preprocessing integration. "
            "Retrain the model so the preprocessing step is included."
        )
    return model


def _resolve_path(pred_set, data_dir):
    path = Path(pred_set)
    if not path.is_absolute() and not path.exists():
        path = Path(data_dir) / path
    if path.suffix == "":
        path = path.with_suffix(".csv")
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    return path

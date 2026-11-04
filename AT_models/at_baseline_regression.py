"""Baseline regression models using the shared preprocessing pipeline."""

import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_data import build_model_pipeline, prepare_splits  # noqa: E402


def main():
    # Shared splits already apply the regression ADR filter and preprocessing.
    train, _, _ = prepare_splits("regression")
    X_train, y_train, _ = train

    # Five-fold cross-validation on training data only
    cross_validation = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    models = {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
    }

    for name, estimator in models.items():
        # The preprocessing pipeline is cloned and refitted inside every fold.
        model = build_model_pipeline("regression", estimator)

        # Cross-validation
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cross_validation,
            scoring=scoring,
        )

        print(f"\n{name}")
        print(f"CV MAE:  {-scores['test_MAE'].mean():.2f}")
        print(f"CV RMSE: {-scores['test_RMSE'].mean():.2f}")
        print(f"CV R^2:   {scores['test_R2'].mean():.4f}")

        # Train the original model on all development data
        model.fit(X_train, y_train)

        # Predict on the development data
        predictions = model.predict(X_train)

        # Calculate training metrics
        mae = mean_absolute_error(y_train, predictions)
        rmse = np.sqrt(mean_squared_error(y_train, predictions))
        r2 = r2_score(y_train, predictions)

        print(f"Training MAE:  {mae:.2f}")
        print(f"Training RMSE: {rmse:.2f}")
        print(f"Training R^2:   {r2:.4f}")


if __name__ == "__main__":
    main()

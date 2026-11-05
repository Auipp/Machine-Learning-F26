"""Extended regression experiment using the shared preprocessing pipeline."""

import sys
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_data import build_model_pipeline, prepare_splits  # noqa: E402
from src.preprocessing.pipeline import JUDGMENT_COLUMNS  # noqa: E402


def main():
    # The shared loader removes invalid ADR rows and creates the splits.
    train, _, _ = prepare_splits("regression")
    X_train, y_train, _ = train

    # Configure five-fold cross-validation
    cross_validation = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # "neg" is required because scikit-learn assumes that higher scores are better
    # So, use negative for errors, positive for R^2 (coefficient of determination).
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    # Preprocessing (imputation, one-hot encoding, scaling) is part of the
    # shared pipeline and is refitted on each training fold. The extended
    # experiment also opts into the extra judgment columns.
    model = build_model_pipeline(
        "regression",
        LinearRegression(),
        include_judgment=tuple(JUDGMENT_COLUMNS["regression"]),
    )

    # Evaluate on development data using cross-validation
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cross_validation,
        scoring=scoring,
    )
    print("Linear Regression (shared preprocessing)")

    for i in range(cross_validation.n_splits):
        print(
            f"Fold {i + 1}: "
            f"MAE={-scores['test_MAE'][i]:.2f}, "
            f"RMSE={-scores['test_RMSE'][i]:.2f}, "
            f"R^2={scores['test_R2'][i]:.4f}"
        )

    print("\nADR distribution across validation folds:")

    for fold_number, (_, validation_indices) in enumerate(
        cross_validation.split(X_train), start=1
    ):
        fold_adr = y_train.iloc[validation_indices]

        print(
            f"Fold {fold_number}: "
            f"Max ADR={fold_adr.max():.2f}, "
            f"ADR > 1000: {(fold_adr > 1000).sum()}"
        )

    # Train on all development observations
    model.fit(X_train, y_train)


if __name__ == "__main__":
    main()

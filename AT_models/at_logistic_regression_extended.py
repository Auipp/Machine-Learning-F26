"""Extended classification experiment using the shared preprocessing pipeline."""

import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_data import build_model_pipeline, prepare_splits  # noqa: E402
from src.preprocessing.pipeline import JUDGMENT_COLUMNS  # noqa: E402


def main():
    # Shared loader: removes duplicate/empty bookings and creates the splits.
    train, _, _ = prepare_splits("classification")
    X_train, y_train, _ = train

    # 42% of the bookings were cancelled. 58% were not.
    # StratifiedKFold attempts to preserve that distribution, unlike normal KFold
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # Preprocessing (feature engineering, encoding, scaling) is part of the
    # shared pipeline and is refitted on every training fold. The extended
    # experiment also opts into the extra judgment columns.
    model = build_model_pipeline(
        "classification",
        LogisticRegression(max_iter=1000),
        include_judgment=tuple(JUDGMENT_COLUMNS["classification"]),
    )

    # Accuracy is overall correctness rate
    # Precision is how often "true" assertions (predicts "cancelled") were right
    # Recall is how often "true" assertions are made when one should've been made
    # ROC-AUC is about the separation of classes.
    # F1 is the harmonic mean between precision and recall
    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC_AUC": "roc_auc",
        "PR_AUC": "average_precision",
    }

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cross_validation,
        scoring=scoring,
    )

    print("Extended Logistic Regression (shared preprocessing)")

    for metric in scoring:
        values = scores[f"test_{metric}"]
        print(f"CV {metric}: {values.mean():.4f}")

    print("\nIndividual folds:")

    for i in range(cross_validation.n_splits):
        print(
            f"Fold {i + 1}: "
            f"F1={scores['test_F1'][i]:.4f}, "
            f"ROC-AUC={scores['test_ROC_AUC'][i]:.4f}"
        )

    # Obtain predictions from held-out validation folds
    validation_predictions = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cross_validation,
    )

    print("\nCross-validated confusion matrix:")
    print(confusion_matrix(y_train, validation_predictions))

    print("\nAs percentages:")
    print(
        100
        * confusion_matrix(y_train, validation_predictions)
        / len(X_train)
    )

    # Fit one model on all development observations
    model.fit(X_train, y_train)


if __name__ == "__main__":
    main()

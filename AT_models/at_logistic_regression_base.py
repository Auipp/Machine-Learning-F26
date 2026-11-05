"""Baseline classification models using the shared preprocessing pipeline."""

import sys
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_data import build_model_pipeline, prepare_splits  # noqa: E402


def main():
    # Shared splits already apply the classification preprocessing.
    train, _, _ = prepare_splits("classification")
    X_train, y_train, _ = train

    print("Development set cancellation distribution:")
    print(y_train.value_counts())
    print(f"Cancellation rate: {y_train.mean():.2%}")

    # 42% of the bookings were cancelled. 58% were not.
    # StratifiedKFold attempts to preserve that distribution, unlike normal KFold
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
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
    }

    # Precision, Recall and F1 are all zero cause the most frequent is "not cancelled"
    # The model has no "true" assertions (true positives), therefore, it's all zeros.
    models = {
        "Dummy (majority class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    for name, estimator in models.items():
        # Preprocessing is refitted on each training fold by cross_validate.
        model = build_model_pipeline("classification", estimator)

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cross_validation,
            scoring=scoring,
        )

        print(f"\n{name}")
        print(f"CV Accuracy:  {scores['test_Accuracy'].mean():.4f}")
        print(f"CV Precision: {scores['test_Precision'].mean():.4f}")
        print(f"CV Recall:    {scores['test_Recall'].mean():.4f}")
        print(f"CV F1:        {scores['test_F1'].mean():.4f}")
        print(f"CV ROC-AUC:   {scores['test_ROC_AUC'].mean():.4f}")


if __name__ == "__main__":
    main()

"""K-Nearest Neighbours classifier using the shared preprocessing pipeline."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from joblib import dump
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_data import (  # noqa: E402
    build_model_pipeline,
    fit_model,
    load_model,
    prediction_features,
    prepare_splits,
)

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.resolve().parents[1] / "data"
artifact_dir = current_dir.resolve().parents[1] / "artifacts"


# Train a K-Nearest Neighbors Classifier with the Scikit-learn default parameters
def train_base_KNN(name, duplicate_weight=0.0):
    return _train_knn(name, KNeighborsClassifier(), duplicate_weight)


# Train a K-Nearest Neighbors Classifier with custom parameters
def train_parameterized_KNN(
    name,
    neighbors=5,
    weights="uniform",
    distance_metric="euclidean",
    power_parameter=2,
    duplicate_weight=0.0,
):
    knn = KNeighborsClassifier(
        n_neighbors=neighbors,
        weights=weights,
        metric=distance_metric,
        p=power_parameter,
    )
    return _train_knn(name, knn, duplicate_weight)


def _train_knn(name, knn, duplicate_weight):
    # The loader returns the raw splits; the preprocessing pipeline is fitted
    # on the training split only and then applied to validation/test data.
    train, _, test = prepare_splits("classification", duplicate_weight)
    X_train, y_train, w_train = train
    X_test, y_test, _ = test

    pipeline = build_model_pipeline("classification", knn)
    fit_model(
        pipeline,
        X_train,
        y_train,
        w_train if duplicate_weight else None,
    )

    # Get trained model predictions and probabilities on the test set
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Calculate classification metrics and display them
    show_test_results(
        pipeline, y_test, y_pred, y_prob, X_train, y_train, cv=5
    )

    # Save the preprocessing pipeline together with the trained model
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dump(pipeline, artifact_dir / f"{name}_knn.joblib")
    return pipeline


# Load an artifact and make predictions on a new dataset
def predict_with_KNN(name, pred_set="test"):
    """Predict on a project split ("train"/"val"/"test") or a raw CSV path."""
    pipeline = load_model(artifact_dir / f"{name}_knn.joblib")
    features = prediction_features(pred_set, "classification", data_dir)

    y_pred = pipeline.predict(features)
    print(y_pred)
    return y_pred


# Display the results of the model on the test set
def show_test_results(pipeline, y_test, y_pred, y_prob, X_train, y_train, cv):
    # Classification Report including several metrics like:
    # Accuracy: The percentage of correctly predicted samples out of the total samples
    # Precision: The percentage of samples predicted as positive that are actually positive
    # Recall: The percentage of actual positive samples that are correctly predicted as positive
    # F1-Score: Harmonic mean of precision and recall, providing a balance between the two metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # ROC-AUC: area under the true-positive/false-positive rate curve
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}\n")

    # K-fold Cross-Validation: the pipeline is cloned and the preprocessor is
    # refitted on each training fold, so no validation fold leaks into it.
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="accuracy"
    )

    print(f"Cross-Validation Results ({cv}-Fold)")
    print(f"CV Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")

    # Confusion Matrix: true positives, true negatives, false positives, false negatives
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("KNN Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    train_base_KNN("B")

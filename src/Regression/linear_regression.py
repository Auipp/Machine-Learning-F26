"""Linear regression using the shared preprocessing pipeline."""

import sys
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

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

# Directory paths for data and artifacts
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.resolve().parents[1] / "data"
artifact_dir = current_dir.resolve().parents[1] / "artifacts"


# Train a Linear Regression model with the Scikit-learn default parameters
def train_base_Linear_Regression(name, duplicate_weight=0.0):
    return _train_linear_regression(name, SGDRegressor(), duplicate_weight)


# Train a Linear Regression model with custom parameters
def train_parameterized_Linear_Regression(
    name,
    loss="squared_error",
    penalty="l2",
    learning_rate="constant",
    eta0=0.01,
    alpha=0.0001,
    max_iter=1000,
    tol=1e-3,
    duplicate_weight=0.0,
):
    linear_regress = SGDRegressor(
        loss=loss,
        penalty=penalty,
        learning_rate=learning_rate,
        eta0=eta0,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
    )
    return _train_linear_regression(
        name, linear_regress, duplicate_weight
    )


def _train_linear_regression(name, linear_regress, duplicate_weight):
    # Rows with an invalid ADR are removed for regression by the loader helpers;
    # the preprocessor is then fitted on the training split only.
    train, _, test = prepare_splits("regression", duplicate_weight)
    X_train, y_train, w_train = train
    X_test, y_test, _ = test

    pipeline = build_model_pipeline("regression", linear_regress)
    fit_model(
        pipeline,
        X_train,
        y_train,
        w_train if duplicate_weight else None,
    )

    # Get trained model predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Calculate regression metrics and display them
    show_test_results(pipeline, y_test, y_pred, X_train, y_train, 5)

    # Save the preprocessing pipeline together with the trained model
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dump(pipeline, artifact_dir / f"{name}_lr.joblib")
    return pipeline


# Load an artifact and make predictions on a new dataset
def predict_with_Linear_Regression(name, pred_set="test"):
    """Predict on a project split ("train"/"val"/"test") or a raw CSV path."""
    pipeline = load_model(artifact_dir / f"{name}_lr.joblib")
    features = prediction_features(pred_set, "regression", data_dir)

    y_pred = pipeline.predict(features)
    print(y_pred)
    return y_pred


# Display the results of the model on the test set
def show_test_results(pipeline, y_test, y_pred, X_train, y_train, cv):
    # Mean Absolute Error: average absolute difference between predictions and actual values
    mae = mean_absolute_error(y_test, y_pred)

    # Mean Squared Error: average squared difference between predictions and actual values
    mse = mean_squared_error(y_test, y_pred)

    # Root Mean Squared Error: the square root of the MSE
    rmse = np.sqrt(mse)

    # Coefficient of Determination (R²): proportion of target variation explained by the model
    r2 = r2_score(y_test, y_pred)

    # Print results
    print("Test Set Regression Metrics: ")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}\n")

    # K-fold Cross-Validation: the preprocessor is refitted on every training
    # fold, so the validation fold never influences feature scaling/encoding.
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="r2"
    )

    # Print cross-validation results
    print(f"Cross-Validation Results ({cv}-Fold)")
    print(f"CV Accuracy Scores: {cv_scores}")
    print(
        f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n"
    )


if __name__ == "__main__":
    train_base_Linear_Regression("A")

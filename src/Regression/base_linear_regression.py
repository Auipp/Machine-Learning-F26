import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from pathlib import Path

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.resolve().parents[1] / "data"
artifact_dir = current_dir.resolve().parents[1] / "artifacts"

def train_base_Linear_Regression(name, train_set, test_set):
    train_filename = f"{train_set}.csv"
    test_filename = f"{test_set}.csv"
    model_filename =  f"{name}_lr.joblib"

    train_set = pd.read_csv(data_dir / train_filename)
    test_set = pd.read_csv(data_dir / test_filename)

    # Step 1.1: Seperate the last column from both data sets since it is the target

    x_train = train_set.drop(columns=["ADR"])
    y_train = train_set["ADR"]

    X_test = test_set.drop(columns=["ADR"])
    y_test = test_set["ADR"]

    # Step 2: Run model with data
    linear_regress = SGDRegressor()

    linear_regress.fit(x_train, y_train)
    
    y_pred = linear_regress.predict(X_test)

    show_test_results(linear_regress, y_test, y_pred, x_train, y_train, 5)

    dump(linear_regress, artifact_dir / model_filename)

def train_parameterized_Linear_Regression(name, train_set, test_set, loss='squared_error', penalty='l2', learning_rate="constant", eta0=0.01, alpha=0.0001, max_iter=1000, tol=1e-3):
    train_filename = f"{train_set}.csv"
    test_filename = f"{test_set}.csv"
    model_filename =  f"{name}_lr.joblib"

    train_set = pd.read_csv(data_dir / train_filename)
    test_set = pd.read_csv(data_dir / test_filename)

    # Step 1.1: Seperate the last column from both data sets since it is the target

    x_train = train_set.drop(columns=["ADR"])
    y_train = train_set["ADR"]

    x_test = test_set.drop(columns=["ADR"])
    y_test = test_set["ADR"]

    # Step 2: Run model with data
    linear_regress = SGDRegressor(loss=loss, penalty=penalty, learning_rate=learning_rate, eta0=eta0, alpha=alpha, max_iter=max_iter, tol=tol)

    linear_regress.fit(x_train, y_train)
    
    y_pred = linear_regress.predict(x_test)

    show_test_results(linear_regress, y_test, y_pred, x_train, y_train, 5)

    dump(linear_regress, artifact_dir / model_filename)


def predict_with_Linear_Regression(name, pred_set):
    pred_filename = f"{pred_set}.csv"
    model_filename =  f"{name}_lr.joblib"

    pred_set = pd.read_csv(data_dir / pred_filename)
    linear_regress = load(artifact_dir / model_filename)

    y_pred = linear_regress.predict(pred_set)

    print(y_pred)


def show_test_results(linear_regress, y_test, y_pred, X_train, y_train, cv):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  
    r2 = r2_score(y_test, y_pred)

    print("Test Set Regression Metrics: ")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}\n")
    
    cv_scores = cross_val_score(linear_regress, X_train, y_train, cv=cv, scoring="r2")

    print(f"Cross-Validation Results ({cv}-Fold)")
    print(f"CV Accuracy Scores: {cv_scores}")
    print(
        f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n"
    )

        






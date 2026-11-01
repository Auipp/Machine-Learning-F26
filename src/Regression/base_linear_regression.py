import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from pathlib import Path

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.resolve().parents[1] / "data"
artifact_dir = current_dir.resolve().parents[1] / "artifacts"

def train_Linear_Regression(name, train_set):
    filename = f"{train_set}.csv"
    model_filename =  f"{name}_lr.joblib"

    train_set = pd.read_csv(data_dir / filename)
    # Step 1.1: Seperate the last column from both data sets since it is the target

    x_train = train_set.drop(columns=["ADR"])
    y_train = train_set["ADR"]

    # Step 2: Run model with data
    linear_regress = LinearRegression()

    linear_regress.fit(x_train, y_train)
    show_cross_validation(linear_regress, x_train, y_train, 5)

    dump(linear_regress, artifact_dir / model_filename)

def test_Linear_Regression(name, test_set):
    test_filename = f"{test_set}.csv"
    model_filename =  f"{name}_lr.joblib"
    test_set = pd.read_csv(data_dir / test_filename)
    linear_regress = load(artifact_dir / model_filename)

    X_test = test_set.drop(columns=["ADR"])
    y_test = test_set["ADR"]

    y_pred = linear_regress.predict(X_test)

    show_test_results(y_test, y_pred)

def show_test_results(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  
    r2 = r2_score(y_test, y_pred)

    print("Test Set Regression Metrics: ")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

def show_cross_validation(linear_regress, X_train, y_train, cv):
    cv_scores = cross_val_score(linear_regress, X_train, y_train, cv=cv, scoring="r2")

    print(f"Cross-Validation Results ({cv}-Fold)")
    print(f"CV Accuracy Scores: {cv_scores}")
    print(
        f"Mean CV Accuracy:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n"
    )
        

train_Linear_Regression("A", "hotel_train")
test_Linear_Regression("A", "hotel_test")






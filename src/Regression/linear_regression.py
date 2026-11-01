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


# Directory paths for data and artifacts
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.resolve().parents[1] / "data"
artifact_dir = current_dir.resolve().parents[1] / "artifacts"

# Function to train a Linear Regression model with the Scikit-learn default parameters
def train_base_Linear_Regression(name, train_set, test_set):
    # File Names
    train_filename = f"{train_set}.csv"
    test_filename = f"{test_set}.csv"
    model_filename =  f"{name}_lr.joblib"

    # Load Training and Test Datasets
    train_set = pd.read_csv(data_dir / train_filename)
    test_set = pd.read_csv(data_dir / test_filename)

    # Remove Target Column from Training Set and Store it in another dataset
    x_train = train_set.drop(columns=["ADR"])
    y_train = train_set["ADR"]

    # Remove Target Column from Test Set and Store it in another dataset
    X_test = test_set.drop(columns=["ADR"])
    y_test = test_set["ADR"]

    # Initialize the Linear Regression model without parameters
    linear_regress = SGDRegressor()

    # Train
    linear_regress.fit(x_train, y_train)
    
    # Get Trained Model Predictions on the Test Set
    y_pred = linear_regress.predict(X_test)

    # Calculate regression metrics and display them
    show_test_results(linear_regress, y_test, y_pred, x_train, y_train, 5)

    # Save trained model as an artifact
    dump(linear_regress, artifact_dir / model_filename)

# Function to train a Linear Regression model with the Scikit-learn custom parameters
def train_parameterized_Linear_Regression(name, train_set, test_set, loss='squared_error', penalty='l2', learning_rate="constant", eta0=0.01, alpha=0.0001, max_iter=1000, tol=1e-3):
    # File Names
    train_filename = f"{train_set}.csv"
    test_filename = f"{test_set}.csv"
    model_filename =  f"{name}_lr.joblib"

    # Load Training and Test Datasets
    train_set = pd.read_csv(data_dir / train_filename)
    test_set = pd.read_csv(data_dir / test_filename)

    # Remove Target Column from Training Set and Store it in another dataset
    x_train = train_set.drop(columns=["ADR"])
    y_train = train_set["ADR"]

    # Remove Target Column from Test Set and Store it in another dataset
    x_test = test_set.drop(columns=["ADR"])
    y_test = test_set["ADR"]

    # Initialize the Linear Regression model with parameters
    linear_regress = SGDRegressor(loss=loss, penalty=penalty, learning_rate=learning_rate, eta0=eta0, alpha=alpha, max_iter=max_iter, tol=tol)

    # Train
    linear_regress.fit(x_train, y_train)
    
    # Get Trained Model Predictions on the Test Set
    y_pred = linear_regress.predict(x_test)

    # Calculate regression metrics and display them
    show_test_results(linear_regress, y_test, y_pred, x_train, y_train, 5)

    # Save trained model as an artifact
    dump(linear_regress, artifact_dir / model_filename)


# Function to load an artifact and make predictions on a new dataset
def predict_with_Linear_Regression(name, pred_set):
    # File Names
    pred_filename = f"{pred_set}.csv"
    model_filename =  f"{name}_lr.joblib"

    # Load the prediction dataset and the trained model
    pred_set = pd.read_csv(data_dir / pred_filename)
    linear_regress = load(artifact_dir / model_filename)

    # Make predictions using the loaded model
    y_pred = linear_regress.predict(pred_set)

    # Print results
    print(y_pred)


# Function to display the results of the model on the test set
def show_test_results(linear_regress, y_test, y_pred, X_train, y_train, cv):

    # Mean Absolute Error: The average of the differences between predicted and actual values in absolute terms
    mae = mean_absolute_error(y_test, y_pred)

    # Mean Squared Error: The average of the differences between predicted and actual values squared 
    mse = mean_squared_error(y_test, y_pred)

    # Root Mean Squared Error: The square root of the MSE
    rmse = np.sqrt(mse)  

    # Coefficient of Determination (R²):  is the proportion of the variation in the dependent variable that is predictable from the independent variables
    r2 = r2_score(y_test, y_pred)

    # Print results
    print("Test Set Regression Metrics: ")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}\n")

    # K-fold Cross-Validation: The process of partitioning the training data into k subsets, training the model on k-1 of those subsets, 
    # and validating it on the remaining subset. This is done repeatadly a K number of times.
    cv_scores = cross_val_score(linear_regress, X_train, y_train, cv=cv, scoring="r2")

    # Print cross-validation results
    print(f"Cross-Validation Results ({cv}-Fold)")
    print(f"CV Accuracy Scores: {cv_scores}")
    print(
        f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n"
    )

        






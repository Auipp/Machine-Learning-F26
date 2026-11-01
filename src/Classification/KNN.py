import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from pathlib import Path

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_auc_score,
    classification_report,
)

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.resolve().parents[1] / "data"
artifact_dir = current_dir.resolve().parents[1] / "artifacts"

# Function to train a K-Nearest Neighbors Classifier with the Scikit-learn default parameters
def train_base_KNN(name, train_set, test_set):
    # File Names
    train_filename = f"{train_set}.csv"
    test_filename = f"{test_set}.csv"
    model_filename =  f"{name}_knn.joblib"

    # Load Training and Test Datasets
    train_set = pd.read_csv(data_dir / train_filename)
    test_set = pd.read_csv(data_dir / test_filename)

    # Remove Target Column from Training Set and Store it in another dataset
    x_train = train_set.drop(columns=["IsCanceled"])
    y_train = train_set["IsCanceled"]
    
    # Remove Target Column from Test Set and Store it in another dataset
    x_test = test_set.drop(columns=["IsCanceled"])
    y_test = test_set["IsCanceled"]

    # Initialize the Linear Regression model without parameters
    knn = KNeighborsClassifier()

    # Train
    knn.fit(x_train, y_train)
    
    # Get Trained Model Predictions on the Test Set
    y_pred = knn.predict(x_test)

    # Get the predicted probabilities for the positive class for each test sample
    y_prob = knn.predict_proba(x_test)[:, 1]

    # Calculate regression metrics and display them
    show_test_results(knn, y_test, y_pred, y_prob, x_train, y_train, cv=5)

    # Save trained model as an artifact
    dump(knn, artifact_dir / model_filename)
    
# Function to train a K-Nearest Neighbors Classifier with the Scikit-learn custom parameters
def train_parameterized_KNN(name, train_set, test_set, neighbors=5, weights='uniform', distance_metric='euclidean', power_parameter=2):
    # File Names
    train_filename = f"{train_set}.csv"
    test_filename = f"{test_set}.csv"
    model_filename =  f"{name}_knn.joblib"

    # Load Training and Test Datasets
    train_set = pd.read_csv(data_dir / train_filename)
    test_set = pd.read_csv(data_dir / test_filename)

    # Remove Target Column from Training Set and Store it in another dataset
    x_train = train_set.drop(columns=["IsCanceled"])
    y_train = train_set["IsCanceled"]

    # Remove Target Column from Test Set and Store it in another dataset
    X_test = test_set.drop(columns=["IsCanceled"])
    y_test = test_set["IsCanceled"]

    # Initialize the Linear Regression model with parameters
    knn = KNeighborsClassifier(
        n_neighbors=neighbors,
        weights=weights,
        metric=distance_metric,
        p=power_parameter
    )

    # Train
    knn.fit(x_train, y_train)
    
    # Get Trained Model Predictions on the Test Set
    y_pred = knn.predict(X_test)

    # Get the predicted probabilities for the positive class for each test sample
    y_prob = knn.predict_proba(X_test)[:, 1]

    # Calculate regression metrics and display them
    show_test_results(knn, y_test, y_pred, y_prob, x_train, y_train, cv=5)

    # Save trained model as an artifact
    dump(knn, artifact_dir / model_filename)

# Function to load an artifact and make predictions on a new dataset
def predict_with_KNN(name, pred_set):
    # File Names
    pred_filename = f"{pred_set}.csv"
    model_filename =  f"{name}_knn.joblib"

    # Load the prediction dataset and the trained model
    pred_set = pd.read_csv(data_dir / pred_filename)
    knn = load(artifact_dir / model_filename)

    # Make predictions using the loaded model
    y_pred = knn.predict(pred_set)

    # Print Results
    print(y_pred)

# Function to display the results of the model on the test set
def show_test_results(knn, y_test, y_pred, y_prob, X_train, y_train, cv):
    # Classification Report including several metrics like:
    # Accuracy: The percentage of correctly predicted samples out of the total samples
    # Precision: The percentage of samples predicted as positive that are actually positive
    # Recall: The percentage of actual positive samples that are correctly predicted as positive
    # F1-Score: Harmonic mean of precision and recall, providing a balance between the two metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Receiver Operating Characteristic - Area Under the Curve (ROC-AUC) Score: ROC curve is the plot of the true positive rate against the false positive rate at each threshold setting.
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}\n")


    # K-fold Cross-Validation: The process of partitioning the training data into k subsets, training the model on k-1 of those subsets, 
    # and validating it on the remaining subset. This is done repeatadly a K number of times.
    cv_scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring="accuracy")

    print(f"Cross-Validation Results ({cv}-Fold)")
    print(f"CV Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")

    # Confusion Matrix: A table with four quadrants that shows the number of true positives, true negatives, false positives, and false negatives
    print("Confusion Matrix:\n", cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("KNN Confusion Matrix")
    plt.show()

train_base_KNN("B", "classification_train", "classification_test")





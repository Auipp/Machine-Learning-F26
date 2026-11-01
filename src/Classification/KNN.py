import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from pathlib import Path

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    classification_report,
)

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.resolve().parents[1] / "data"
artifact_dir = current_dir.resolve().parents[1] / "artifacts"

def train_base_KNN(name, train_set, test_set):
    train_filename = f"{train_set}.csv"
    test_filename = f"{test_set}.csv"
    model_filename =  f"{name}_knn.joblib"

    train_set = pd.read_csv(data_dir / train_filename)
    test_set = pd.read_csv(data_dir / test_filename)

    x_train = train_set.drop(columns=["IsCanceled"])
    y_train = train_set["IsCanceled"]
    x_test = test_set.drop(columns=["IsCanceled"])
    y_test = test_set["IsCanceled"]

    knn = KNeighborsClassifier()

    knn.fit(x_train, y_train)
    
    y_pred = knn.predict(x_test)

    y_prob = knn.predict_proba(x_test)[:, 1]

    show_test_results(knn, y_test, y_pred, y_prob, x_train, y_train, cv=5)
    dump(knn, artifact_dir / model_filename)
    
def train_parameterized_KNN(name, train_set, test_set, neighbors=5, weights='uniform', distance_metric='euclidean', power_parameter=2):
    train_filename = f"{train_set}.csv"
    test_filename = f"{test_set}.csv"
    model_filename =  f"{name}_knn.joblib"

    train_set = pd.read_csv(data_dir / train_filename)
    test_set = pd.read_csv(data_dir / test_filename)

    x_train = train_set.drop(columns=["IsCanceled"])
    y_train = train_set["IsCanceled"]
    X_test = test_set.drop(columns=["IsCanceled"])
    y_test = test_set["IsCanceled"]

    knn = KNeighborsClassifier(
        n_neighbors=neighbors,
        weights=weights,
        metric=distance_metric,
        p=power_parameter
    )

    knn.fit(x_train, y_train)
    
    y_pred = knn.predict(X_test)

    y_prob = knn.predict_proba(X_test)[:, 1]

    show_test_results(knn, y_test, y_pred, y_prob, x_train, y_train, cv=5)
    dump(knn, artifact_dir / model_filename)

def predict_with_KNN(name, pred_set):
    pred_filename = f"{pred_set}.csv"
    model_filename =  f"{name}_knn.joblib"

    pred_set = pd.read_csv(data_dir / pred_filename)
    knn = load(artifact_dir / model_filename)

    y_pred = knn.predict(pred_set)

    print(y_pred)

def show_test_results(knn, y_test, y_pred, y_prob, X_train, y_train, cv):
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}\n")

    cv_scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring="accuracy")

    print(f"Cross-Validation Results ({cv}-Fold)")
    print(f"CV Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("KNN Confusion Matrix")
    plt.show()






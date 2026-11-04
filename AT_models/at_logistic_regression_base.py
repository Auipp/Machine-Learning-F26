import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/raw/H2.csv")

numerical_features = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
]

X = df[numerical_features]
y = df["IsCanceled"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
)

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
# ROC-AUC is about the seperation of of classes.
# F1 is the harmonic mean between precision and recall
scoring = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "ROC_AUC": "roc_auc",
}

logistic_regression = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# Precision, Recall and F1 are all zero cause the most frequent is "not cancelled"
# The model has no "true" assertions (true positives), therefore, it's all zeros.
models = {
    "Dummy (majority class)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": logistic_regression,
}

for name, model in models.items():
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
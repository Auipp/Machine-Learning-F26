
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
)
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix


df = pd.read_csv("data/raw/H2.csv")

numerical_features = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "ArrivalDateYear",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
]

categorical_features = [
    "ArrivalDateMonth",
    "ReservedRoomType",
    "Meal",
    "MarketSegment",
    "CustomerType",
    "DistributionChannel",
]

all_features = numerical_features + categorical_features

X = df[all_features].copy()
y = df["IsCanceled"]

# Trim categorical features
for column in categorical_features:
    X[column] = X[column].str.strip()

# Preserve the same development/test partition as the baseline
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
)

# 42% of the bookings were cancelled. 58% were not.
# StratifiedKFold attempts to preserve that distribution, unlike normal KFold
cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

numerical_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_preprocessor, numerical_features),
    ("categorical", categorical_preprocessor, categorical_features),
])

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])


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
    "PR_AUC": "average_precision",
}

scores = cross_validate(
    model,
    X_train,
    y_train,
    cv=cross_validation,
    scoring=scoring,
)

print("Extended Logistic Regression (16 features)")

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
print(100 * confusion_matrix(y_train, validation_predictions) / len(X_train))

# Fit one model on all development observations
model.fit(X_train, y_train)
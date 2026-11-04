import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("data/raw/H2.csv")

numerical_features = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
]

X = df[numerical_features]
y = df["ADR"]

# Reserve 30% for final testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# Five-fold cross-validation on training data only
cross_validation = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}

models = {
    "Dummy (mean)": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
}

for name, model in models.items():

    # Cross-validation
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cross_validation,
        scoring=scoring
    )

    print(f"\n{name}")
    print(f"CV MAE:  {-scores['test_MAE'].mean():.2f}")
    print(f"CV RMSE: {-scores['test_RMSE'].mean():.2f}")
    print(f"CV R^2:   {scores['test_R2'].mean():.4f}")

    # Train the original model on all development data
    model.fit(X_train, y_train)

    # Predict on the development data
    predictions = model.predict(X_train)

    # Calculate training metrics
    mae = mean_absolute_error(y_train, predictions)
    rmse = np.sqrt(mean_squared_error(y_train, predictions))
    r2 = r2_score(y_train, predictions)

    print(f"Training MAE:  {mae:.2f}")
    print(f"Training RMSE: {rmse:.2f}")
    print(f"Training R^2:   {r2:.4f}")
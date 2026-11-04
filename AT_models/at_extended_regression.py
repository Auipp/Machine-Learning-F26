import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


df = pd.read_csv("data/raw/H2.csv")

numerical_features = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
]

categorical_features = [
    "ArrivalDateMonth",
    "ReservedRoomType",
    "Meal",
    "MarketSegment",
]

all_features = numerical_features + categorical_features

X = df[all_features].copy()
y = df["ADR"]

# Remove whitespace from categorical values
for column in categorical_features:
    X[column] = X[column].str.strip()

# Reserve 30% of the dataset for final testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# Configure five-fold cross-validation
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

# Scale numerical features and one-hot encode categorical features
preprocessor = ColumnTransformer([
    ("numerical", StandardScaler(), numerical_features),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# Combine preprocessing and regression into one pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("regression", LinearRegression()),
])

# Evaluate on development data using cross-validation
scores = cross_validate(
    model,
    X_train,
    y_train,
    cv=cross_validation,
    scoring=scoring
)

print("Extended Linear Regression (8 features)")
print(f"CV MAE:  {-scores['test_MAE'].mean():.2f}")
print(f"CV RMSE: {-scores['test_RMSE'].mean():.2f}")
print(f"CV R²:   {scores['test_R2'].mean():.4f}")

# Train on all development observations
model.fit(X_train, y_train)

print(f"\nTraining observations: {len(X_train)}")
print(f"Reserved test observations: {len(X_test)}")
print(f"Encoded feature count: {len(model.named_steps['preprocessing'].get_feature_names_out())}")
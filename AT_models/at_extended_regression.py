import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


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

# "neg" is required because scikit-learn assumes that higher scores are better
# So, use negative for errors, positive for R^2 (coefficient of determination).
scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}

numerical_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), # For "NaN" values
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
print("Linear Regression (14 features)")

for i in range(cross_validation.n_splits):
    print(
        f"Fold {i + 1}: "
        f"MAE={-scores['test_MAE'][i]:.2f}, "
        f"RMSE={-scores['test_RMSE'][i]:.2f}, "
        f"R^2={scores['test_R2'][i]:.4f}"
    )

print("\nADR distribution across validation folds:")

# Fold 5 appears to have an outlier vlaue, which is affecting overall RMSE and R^2 for fold 5
for fold_number, (_, validation_indices) in enumerate(cross_validation.split(X_train), start=1):
    fold_adr = y_train.iloc[validation_indices]

    print(
        f"Fold {fold_number}: "
        f"Max ADR={fold_adr.max():.2f}, "
        f"ADR > 1000: {(fold_adr > 1000).sum()}"
    )

# Train on all development observations
model.fit(X_train, y_train)
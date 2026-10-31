import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_MONTH_TO_NUM: dict[str, int] = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()

        df["Children"] = df["Children"].fillna(0)
        df["Country"] = df["Country"].fillna("Unknown")

        df["HasCompany"] = df["Company"].notna().astype(int)
        df = df.drop(columns=["Company"])

        df["Agent"] = df["Agent"].apply(
            lambda x: "NoAgent" if pd.isna(x) else str(int(float(x)))
        )

        df["HasAgent"] = (df["Agent"] != "NoAgent").astype(int)
        df["ArrivalDateMonth"] = df["ArrivalDateMonth"].map(_MONTH_TO_NUM)
        df["TotalNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
        df["TotalGuests"] = df["Adults"] + df["Children"] + df["Babies"]

        return df

    def get_feature_names_out(self, input_features=None):
        raise NotImplementedError("Use ColumnTransformer.get_feature_names_out after fit.")


class TopNGrouper(BaseEstimator, TransformerMixin):

    def __init__(self, n: int = 15, other_label: str = "Other"):
        self.n = n
        self.other_label = other_label

    def fit(self, X, y=None):
        ser = self._to_series(X)
        self.top_categories_: set[str] = set(
            ser.value_counts().nlargest(self.n).index
        )
        return self

    def transform(self, X, y=None):
        ser = self._to_series(X)
        result = ser.where(ser.isin(self.top_categories_), other=self.other_label)
        return result.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.array(["x0"], dtype=object)

    @staticmethod
    def _to_series(X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].astype(str)
        if isinstance(X, pd.Series):
            return X.astype(str)
        arr = np.asarray(X)
        return pd.Series(arr[:, 0] if arr.ndim > 1 else arr, dtype=str)

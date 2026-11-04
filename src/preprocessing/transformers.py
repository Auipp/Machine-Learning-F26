import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

_MONTH_TO_NUM: dict[str, int] = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

_MISSING_TOKENS = {"", "NULL", "NAN", "NONE", "<NA>"}

_OPTIONAL_COLUMNS = ("Company", "Agent", "Children", "Country",
                     "MarketSegment", "DistributionChannel")


def _is_missing(series: pd.Series) -> pd.Series:
    """True for real NaN and for placeholder strings such as 'NULL'."""
    as_text = series.astype(object).map(lambda v: str(v).strip().upper())
    return series.isna() | as_text.isin(_MISSING_TOKENS)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Deterministic cleaning and feature construction (nothing is learned).

    Because it learns nothing from the data, applying it before the
    train/test split cannot leak information. It only stores the output
    column names so the whole pipeline can report its feature names.

    Missing-value decisions:
      * Children NaN (4 rows)        -> 0 (the overwhelming mode)
      * Country NaN                  -> "Unknown"
      * MarketSegment/Distribution   -> "Unknown" ("Undefined" is loaded as NaN)
      * Agent NaN                    -> "NoAgent" (direct booking, informative)
      * Company (95% NULL)           -> binary HasCompany flag, raw ID dropped
    """

    def fit(self, X, y=None):
        self.feature_names_out_ = np.asarray(self._engineer(X).columns, dtype=object)
        return self

    def transform(self, X, y=None):
        return self._engineer(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    @staticmethod
    def _engineer(X) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        for col in _OPTIONAL_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        df["Children"] = pd.to_numeric(df["Children"], errors="coerce").fillna(0)

        for col, fill in (("Country", "Unknown"),
                          ("MarketSegment", "Unknown"),
                          ("DistributionChannel", "Unknown")):
            col_obj = df[col].astype(object)
            df[col] = col_obj.where(~_is_missing(df[col]), fill)

        df["HasCompany"] = (~_is_missing(df["Company"])).astype(int)
        df = df.drop(columns=["Company"])

        agent_num = pd.to_numeric(df["Agent"], errors="coerce")
        df["HasAgent"] = agent_num.notna().astype(int)
        df["Agent"] = np.where(
            agent_num.notna(), agent_num.fillna(0).astype(int).astype(str), "NoAgent"
        )

        month = df["ArrivalDateMonth"]
        mapped = month.map(_MONTH_TO_NUM)
        df["ArrivalDateMonth"] = mapped.where(mapped.notna(), pd.to_numeric(month, errors="coerce"))

        df["TotalNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
        df["TotalGuests"] = df["Adults"] + df["Children"] + df["Babies"]

        return df


class TopNGrouper(BaseEstimator, TransformerMixin):
    """Keep the N most frequent levels (learned on the training data), map the rest to 'Other'."""

    def __init__(self, n: int = 15, other_label: str = "Other"):
        self.n = n
        self.other_label = other_label

    def fit(self, X, y=None):
        counts = self._to_series(X).value_counts()
        # Sort by count, then alphabetically, so ties are resolved the same way every run.
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        self.top_categories_: set[str] = {level for level, _ in ordered[: self.n]}
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "top_categories_")
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

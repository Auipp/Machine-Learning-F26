from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .transformers import FeatureEngineer, TopNGrouper

TASKS = ("regression", "classification", "clustering")

TASK_TARGETS = {
    "regression": "ADR",
    "classification": "IsCanceled",
}

AUX_COLUMNS = ("dup_count",)

_NUMERIC_BASE = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "TotalNights",
    "TotalGuests",
    "HasAgent",
    "HasCompany",
]

_CATEGORICAL_BASE = [
    "Meal",
    "MarketSegment",
    "DistributionChannel",
    "CustomerType",
    "ReservedRoomType",
]

_CATEGORICAL_EXTRA = {
    "regression": ["DepositType"],
    "classification": [],
}

JUDGMENT_COLUMNS = {
    "classification": {
        "DepositType": "cat",
        "ADR": "num",
        "TotalOfSpecialRequests": "num",
        "DaysInWaitingList": "num",
    },
    "regression": {
        "TotalOfSpecialRequests": "num",
        "DaysInWaitingList": "num",
    },
}

CLUSTERING_FEATURES = [
    "LeadTime",
    "ADR",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "IsRepeatedGuest",
    "TotalOfSpecialRequests",
]

ADR_MIN = 1.0
ADR_MAX = 510.0


def filter_adr(df, min_adr: float = ADR_MIN, max_adr: float = ADR_MAX):
    """Rule-based removal of zero-rate rows and the 5,400 EUR data error."""
    return df[(df["ADR"] >= min_adr) & (df["ADR"] <= max_adr)].copy()


def filter_for_task(df, task: str):
    """Task-specific row filter. Apply to train, val and test separately.

    Classification keeps every row (dropping zero-ADR rows would distort the
    cancellation rate). Regression and clustering need a valid ADR.
    """
    _check_task(task)
    return filter_adr(df) if task in ("regression", "clustering") else df.copy()


def make_xy(df, task: str):
    """Split a (filtered) dataframe into (X, y). y is None for clustering.

    Targets and helper columns are removed from X. Keep `df` around to compute
    sample weights from the same rows: sample_weights(df, x).
    """
    _check_task(task)
    drop = [c for c in AUX_COLUMNS if c in df.columns]
    if task == "clustering":
        return df.drop(columns=drop), None
    target = TASK_TARGETS[task]
    return df.drop(columns=drop + [target]), df[target]

def _check_task(task: str) -> None:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")


def _numeric_pipe() -> Pipeline:
    return make_pipeline(SimpleImputer(strategy="median"), RobustScaler())


def _onehot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_preprocessor(task: str = "regression", include_judgment=()) -> Pipeline:
    """Build an UNFITTED preprocessing pipeline.

    Fit it on the training split only. Inside GridSearchCV/cross_val_score,
    sklearn refits it on each training fold, so no fold leaks into another.
    """
    _check_task(task)
    engineer = FeatureEngineer()

    if task == "clustering":
        ct = ColumnTransformer(
            [("num", _numeric_pipe(), CLUSTERING_FEATURES)], remainder="drop"
        )
        return Pipeline([("engineer", engineer), ("preprocessor", ct)])

    allowed = JUDGMENT_COLUMNS[task]
    unknown = set(include_judgment) - set(allowed)
    if unknown:
        raise ValueError(
            f"{sorted(unknown)} are not judgment columns for {task}. "
            f"Choose from {sorted(allowed)}."
        )

    numeric_cols = list(_NUMERIC_BASE)
    categorical_cols = _CATEGORICAL_BASE + _CATEGORICAL_EXTRA[task]
    for col in include_judgment:
        (numeric_cols if allowed[col] == "num" else categorical_cols).append(col)

    ct = ColumnTransformer(
        [
            ("num", _numeric_pipe(), numeric_cols),
            ("country", make_pipeline(TopNGrouper(n=15), _onehot()), ["Country"]),
            ("agent", make_pipeline(TopNGrouper(n=25), _onehot()), ["Agent"]),
            ("low_cat", _onehot(), categorical_cols),
        ],
        remainder="drop",
    )
    return Pipeline([("engineer", engineer), ("preprocessor", ct)])

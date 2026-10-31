from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .transformers import FeatureEngineer, TopNGrouper

_NUMERICAL_BASE = [
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
    "BookingChanges",
    "DaysInWaitingList",
    "TotalOfSpecialRequests",
    "TotalNights",
    "TotalGuests",
    "HasAgent",
    "HasCompany",
]

_LOW_CARD_CAT = [
    "Meal",
    "MarketSegment",
    "DistributionChannel",
    "DepositType",
    "CustomerType",
    "ReservedRoomType",
]

CLUSTERING_FEATURES = [
    "LeadTime",
    "ADR",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "BookingChanges",
    "TotalOfSpecialRequests",
]

TASK_TARGETS = {
    "regression": "ADR",
    "classification": "IsCanceled",
}

ADR_MIN = 1.0
ADR_MAX = 510.0


def filter_adr(df, min_adr: float = ADR_MIN, max_adr: float = ADR_MAX):
    return df[(df["ADR"] >= min_adr) & (df["ADR"] <= max_adr)].copy()


def build_preprocessor(task: str = "regression") -> Pipeline:
    if task not in ("regression", "classification", "clustering"):
        raise ValueError(f"task must be one of regression/classification/clustering, got {task!r}")

    engineer = FeatureEngineer()

    if task == "clustering":
        ct = ColumnTransformer(
            [("num", RobustScaler(), CLUSTERING_FEATURES)],
            remainder="drop",
        )
        return Pipeline([("engineer", engineer), ("preprocessor", ct)])

    numerical_cols = (
        _NUMERICAL_BASE + ["ADR"] if task == "classification" else _NUMERICAL_BASE
    )

    country_pipe = make_pipeline(
        TopNGrouper(n=15),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )

    agent_pipe = make_pipeline(
        TopNGrouper(n=25),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )

    ct = ColumnTransformer(
        [
            ("num", RobustScaler(), numerical_cols),
            ("country", country_pipe, ["Country"]),
            ("agent", agent_pipe, ["Agent"]),
            ("low_cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _LOW_CARD_CAT),
        ],
        remainder="drop",
    )

    return Pipeline([("engineer", engineer), ("preprocessor", ct)])

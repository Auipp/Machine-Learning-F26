from io import StringIO
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "H2.csv"
RANDOM_STATE = 42

DUP_COL = "dup_count"

def load_raw(path: str | Path = DATA_PATH) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, skipinitialspace=True)
    raw = raw.apply(lambda col: col.str.strip())
    df = pd.read_csv(StringIO(raw.to_csv(index=False)), na_values=["Undefined"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:

    group_id = df.groupby(list(df.columns), dropna=False, sort=False).ngroup()
    dup_count = group_id.map(group_id.value_counts())

    df = df.assign(**{DUP_COL: dup_count.astype(int)})
   
    df = df.drop_duplicates(subset=[c for c in df.columns if c != DUP_COL]).copy()

    children_filled = df["Children"].fillna(0)
    valid_guests = (df["Adults"] + children_filled + df["Babies"]) > 0
    df = df[valid_guests]

    valid_nights = (df["StaysInWeekendNights"] + df["StaysInWeekNights"]) > 0
    df = df[valid_nights]

    return df.reset_index(drop=True)


def split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    stratify_col: str = "IsCanceled",
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
   
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=random_state,
    )
    adjusted_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val,
        stratify=train_val[stratify_col],
        random_state=random_state,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def sample_weights(df: pd.DataFrame, x: float = 0.0) -> pd.Series:
    """Training weight per row: 1 + (dup_count - 1) * x / 100.

    x = 0   -> every unique booking counts once (plain deduplication)
    x = 100 -> a booking counts as many times as it appeared in the raw file
    Use for TRAINING rows only. Evaluate on validation/test unweighted.
    """
    if not 0 <= x <= 100:
        raise ValueError(f"x must be between 0 and 100, got {x}")
    return 1.0 + (df[DUP_COL] - 1.0) * x / 100.0


def drop_aux(df: pd.DataFrame) -> pd.DataFrame:
    """Remove helper columns so dup_count can never become a feature."""
    return df.drop(columns=[DUP_COL], errors="ignore")


def load_and_split(
    path: str | Path = DATA_PATH,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_raw(path)
    df = clean(df)
    return split(df, test_size=test_size, val_size=val_size)
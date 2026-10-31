from io import StringIO
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "H2.csv"
RANDOM_STATE = 42


def load_raw(path: str | Path = DATA_PATH) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, skipinitialspace=True)
    raw = raw.apply(lambda col: col.str.strip())
    df = pd.read_csv(StringIO(raw.to_csv(index=False)), na_values=["Undefined"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()

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


def load_and_split(
    path: str | Path = DATA_PATH,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_raw(path)
    df = clean(df)
    return split(df, test_size=test_size, val_size=val_size)

from typing import Literal, overload

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

SplitData = tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]


def label_encode() -> LabelEncoder:
    return LabelEncoder()


def build_pipeline(
    model: BaseEstimator,
    tex_cols: str = "message_text",
    cat_cols: list = ["channel", "language_hint", "customer_tier"],
) -> Pipeline:
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
    encoder = OneHotEncoder(handle_unknown="ignore")

    column_transform = ColumnTransformer(
        [("vectorizer", vectorizer, tex_cols), ("encode", encoder, cat_cols)],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[("column_transformer", column_transform), ("model", model)]
    )
    return pipeline


def column_transformer(
    tex_cols: str = "message_text",
    cat_cols: list = ["channel", "language_hint", "customer_tier"],
) -> ColumnTransformer:
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
    encoder = OneHotEncoder(handle_unknown="ignore")

    column_transform = ColumnTransformer(
        [("vectorizer", vectorizer, tex_cols), ("encode", encoder, cat_cols)],
        remainder="drop",
    )

    return column_transform


def _apply_split(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    val_fraction: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x = df.drop("intent", axis=1)
    y = df["intent"]

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=1 - train_fraction,
        random_state=54,
        stratify=y,
    )

    x_test, x_val, y_test, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=val_fraction,
        random_state=54,
        stratify=y_temp,
    )
    return x_train, x_test, x_val, y_train, y_test, y_val


def split_default() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
):
    """Return the baseline stratified 70/15/15 split."""
    df = pd.read_csv("src/brief3/data/raw/support_messages.csv")
    return _apply_split(df, train_fraction=0.7, val_fraction=0.5)


def split_balanced() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
):
    """Return a stratified split with a larger training share."""
    df = pd.read_csv("src/brief3/data/raw/support_messages.csv")
    return _apply_split(df, train_fraction=0.75, val_fraction=0.5)


def split_drop_duplicates() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
):
    """Return a split after removing duplicate message text rows."""
    df = pd.read_csv("src/brief3/data/raw/support_messages.csv")
    df = df.drop_duplicates(subset=["message_text"]).reset_index(drop=True)
    return _apply_split(df, train_fraction=0.7, val_fraction=0.5)


def split_mixed_language() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
):
    """Return a split after removing mixed-language rows."""
    df = pd.read_csv("src/brief3/data/raw/support_messages.csv")
    df = df[df["language_hint"] != "mixed"].reset_index(drop=True)
    return _apply_split(df, train_fraction=0.7, val_fraction=0.5)


def split_imbalance_aware() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
):
    """Return a stratified split with an 80 percent training share."""
    df = pd.read_csv("src/brief3/data/raw/support_messages.csv")
    return _apply_split(df, train_fraction=0.8, val_fraction=0.5)


@overload
def split(
    split_name: str = "default",
    *,
    return_name: Literal[False] = False,
) -> SplitData: ...


@overload
def split(
    split_name: str = "default",
    *,
    return_name: Literal[True],
) -> tuple[SplitData, str]: ...


def split(
    split_name: str = "default",
    return_name: bool = False,
) -> SplitData | tuple[SplitData, str]:
    """Select a split strategy, optionally returning its name."""
    strategy_map = {
        "default": split_default,
        "balanced": split_balanced,
        "drop_duplicates": split_drop_duplicates,
        "mixed_language": split_mixed_language,
        "imbalance_aware": split_imbalance_aware,
    }

    strategy = strategy_map.get(split_name, split_default)
    result = strategy()

    if return_name:
        return result, split_name
    return result

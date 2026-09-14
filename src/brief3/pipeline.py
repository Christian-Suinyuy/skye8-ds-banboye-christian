import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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


def split() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
):
    df = pd.read_csv("src/brief3/data/raw/support_messages.csv")

    x = df.drop("intent", axis=1)
    y = df["intent"]

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.3, random_state=54, stratify=y
    )

    x_test, x_val, y_test, y_val = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=54, stratify=y_temp
    )
    return x_train, x_test, x_val, y_train, y_test, y_val

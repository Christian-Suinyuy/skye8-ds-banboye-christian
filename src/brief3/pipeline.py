from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
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

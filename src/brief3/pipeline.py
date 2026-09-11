from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(
    model: object,
    tex_cols: str = "message_text",
    cat_cols: list = ["channel", "language_hint", "customer_tier"],
) -> object:
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
    encoder = OneHotEncoder(handle_unknown="ignore")

    column_transform = ColumnTransformer(
        [("vectorizer", vectorizer, "message_text"), ("encode", encoder, cat_cols)],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[("column_transformer", column_transform), ("model", model)]
    )
    return pipeline

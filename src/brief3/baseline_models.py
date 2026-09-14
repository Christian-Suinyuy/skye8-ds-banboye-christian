import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split

from .pipeline import build_pipeline

df = pd.read_csv("src/brief3/data/raw/support_messages.csv")

x = df.drop("intent", axis=1)
y = df["intent"]

x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3, random_state=54, stratify=y
)

x_test, x_val, y_test, y_val = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=54, stratify=y_temp
)

mlflow.set_experiment("message_classifier")

with mlflow.start_run(run_name="Dummy_classifier"):
    model = DummyClassifier(strategy="most_frequent")
    pipeline = build_pipeline(model=model)
    pipeline.fit(x_train, y_train)

    prediction = pipeline.predict(x_test)
    metrics = {
        "recall": recall_score(y_test, prediction, average="macro", zero_division=0),
        "precision": precision_score(
            y_test, prediction, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(prediction, y_test, average="macro"),
        "f1_weighted": f1_score(y_test, prediction, average="weighted"),
    }

    mlflow.log_param("model", "DummyClassifier")
    mlflow.log_metrics(metrics=metrics)
    mlflow.log_text(str(confusion_matrix(y_test, prediction)), "confusion_matrix.txt")
    mlflow.sklearn.log_model(pipeline, "Dummy_majority_classifier")

    print(f"Dummy classifier accuracyy = {pipeline.score(x_test, y_test)}")


with mlflow.start_run(run_name="Logistic_Regression"):
    base_model = LogisticRegression(max_iter=1000)
    pipeline = build_pipeline(base_model)

    param_grid = {
        "model__C": [0.1, 1.0, 10.0],
        "model__penalty": ["l2"],
        "column_transformer__vectorizer__ngram_range": [(1, 2), (1, 3)],
    }
    grid = GridSearchCV(
        pipeline, param_grid=param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1
    )

    grid.fit(x_val, y_val)

    best = grid.best_estimator_
    best.fit(x_train, y_train)

    prediction = best.predict(x_test)
    metrics = {
        "recall": recall_score(y_test, prediction, average="macro", zero_division=0),
        "precision": precision_score(
            y_test, prediction, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_test, prediction, average="macro"),
        "f1_weighted": f1_score(y_test, prediction, average="weighted"),
    }

    mlflow.log_params(params=grid.best_params_)
    mlflow.log_metrics(metrics=metrics)
    mlflow.log_text(str(confusion_matrix(y_test, prediction)), "confusion_matrix.txt")
    mlflow.sklearn.log_model(best, "Tuned_logistic_regression")

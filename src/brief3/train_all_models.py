from collections.abc import Callable

import mlflow

from .baseline_models import train_dummy_classifier, train_logistic_regression
from .nn_model import train_nn_model
from .transformer import train_transformer

mlflow.set_experiment("message_classifier")


def run_model_training(
    model_name: str,
    training_fn: Callable[[str], None],
    split_name: str = "default",
) -> None:
    try:
        training_fn(split_name)
        with mlflow.start_run(run_name=f"{model_name}_status"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("split_name", split_name)
            mlflow.log_param("status", "success")
    except Exception as exc:
        with mlflow.start_run(run_name=f"{model_name}_status"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("split_name", split_name)
            mlflow.log_param("status", "failed")
            mlflow.log_param("error_type", type(exc).__name__)
            mlflow.log_text(str(exc), f"{model_name}_error.txt")
        print(f"{model_name} failed for split '{split_name}': {exc}")


def train_all_models(split_name: str = "default") -> None:
    models = [
        ("dummy_classifier", train_dummy_classifier),
        ("logistic_regression", train_logistic_regression),
        ("neural_network", train_nn_model),
        ("transformer", train_transformer),
        # ("llm", evaluate_llm),
    ]

    for model_name, training_fn in models:
        run_model_training(
            model_name=model_name, training_fn=training_fn, split_name=split_name
        )


if __name__ == "__main__":
    train_all_models()

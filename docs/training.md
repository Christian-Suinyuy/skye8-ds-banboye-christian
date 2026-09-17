# Training and MLflow

All Brief 3 training functions accept a `split_name` argument. The available values are documented in [split.md](split.md). Each model writes metrics, parameters, and model artifacts to the MLflow experiment `message_classifier`.

For local tracking, use the SQLite backend once in the active PowerShell session:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow_local.db"
```

## Run one model

Run these commands from the repository root after completing the environment setup in [README.md](../README.md):

```powershell
python -c "from src.brief3.baseline_models import train_dummy_classifier; train_dummy_classifier()"
python -c "from src.brief3.baseline_models import train_logistic_regression; train_logistic_regression()"
python -c "from src.brief3.nn_model import train_nn_model; train_nn_model()"
python -c "from src.brief3.transformer import train_transformer; train_transformer()"
```

To run one model with another split, pass its name explicitly:

```powershell
python -c "from src.brief3.transformer import train_transformer; train_transformer(split_name='drop_duplicates')"
```

The optional hosted LLM evaluation requires `GROQ_API_KEY` and can be run separately:

```powershell
python -c "from src.brief3.llm import evaluate_llm; evaluate_llm()"
```

## Run all enabled models

The aggregate entry point runs the dummy classifier, logistic regression, neural network, and transformer in sequence using the default split:

```powershell
python -m src.brief3.train_all_models
```

Run the same four models with a named split:

```powershell
python -c "from src.brief3.train_all_models import train_all_models; train_all_models('imbalance_aware')"
```

The aggregate runner records a success status run for each completed model. If a model fails, the exception is printed to the console and the next model is attempted.

## View MLflow runs

Start the local MLflow server from the repository root in a separate terminal:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow_local.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000>, select the `message_classifier` experiment, and compare runs by `model` and `split_name`. Stop the server with `Ctrl+C` when finished.

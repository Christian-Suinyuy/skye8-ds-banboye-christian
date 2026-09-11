from datetime import datetime

import mlflow

today = datetime.now().strftime("%A")

experiment_name = f"Skye8_{today}"

mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    mlflow.set_tag("project", "Skye8")
    mlflow.set_tag("day", today)

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

from .pipeline import column_transformer, label_encode, split

mlflow.set_experiment("message_classifier")

device = "cuda" if torch.cuda.is_available() else "cpu"

# load dataset and convert that to pytorch tensors
x_train, x_test, x_val, y_train, y_test, y_val = split()

transformer = column_transformer()
x_train = transformer.fit_transform(x_train)

x_train_tensors = torch.from_numpy(x_train.toarray()).float()
y_encoder = label_encode()
y_train = y_encoder.fit_transform(y_train)

y_train_tensors = torch.from_numpy(y_train).long()
input_dim = x_train_tensors.shape[1]
output_dim = len(y_train_tensors)
print(f"y_train: {y_train_tensors}")

train_dataset = TensorDataset(x_train_tensors, y_train_tensors)
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)


class BCNet(nn.Module):
    def __init__(self) -> None:
        super(BCNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.bcn1 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 100)
        self.bcn2 = nn.Linear(100, 50)
        self.fc6 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.bcn1(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.bcn2(x))
        x = self.dropout(x)
        x = self.fc6(x)

        return x


model = BCNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 15

with mlflow.start_run(run_name="nn_model"):
    mlflow.log_param("model", "nn_model")
    mlflow.log_params(
        params={
            "epochcs": epochs,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "batch_size": train_loader.batch_size,
            "criterion": type(criterion).__name__,
            "optimizer": type(optimizer).__name__,
        }
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"epoch {epoch}: loss {running_loss/len(train_loader)}")

    x_test = transformer.transform(x_test)
    x_test_tensors = torch.from_numpy(x_test.toarray()).float()

    y_test = y_encoder.fit_transform(y_test)
    y_test_tensors = torch.from_numpy(y_test).long()

    model.eval()
    with torch.no_grad():
        preds = model(x_test_tensors)
        loss = criterion(preds, y_test_tensors).item()

        predictions = torch.argmax(preds, dim=1)
        accuracy = accuracy_score(predictions.numpy(), y_test_tensors)
        confusion = confusion_matrix(predictions.numpy(), y_test_tensors)
        metrics = {
            "recall": recall_score(
                y_test_tensors, predictions, average="macro", zero_division=0
            ),
            "precision": precision_score(
                y_test_tensors, predictions, average="macro", zero_division=0
            ),
            "f1_macro": float(f1_score(predictions, y_test_tensors, average="macro")),
            "f1_weighted": f1_score(y_test_tensors, predictions, average="weighted"),
        }
        mlflow.log_metrics(
            metrics=metrics,
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion, display_labels=y_encoder.classes_
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, xticks_rotation=90, colorbar=False)

        mlflow.log_figure(fig, f"confusion_epoch_{epoch}.png")

    mlflow.pytorch.log_model(model, name="nn_model", serialization_format="pickle")

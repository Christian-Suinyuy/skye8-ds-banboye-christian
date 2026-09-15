import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

from .pipeline import label_encode, split
from .utils import encode_text, tokenizer

x_train, x_test, x_val, y_train, y_test, y_val = split()

vocab = tokenizer(x_train.message_text)
x_train_tensor = encode_text(vocab=vocab, texts=x_train.message_text, max_lenght=20)
x_test_tensor = encode_text(vocab=vocab, texts=x_test.message_text, max_lenght=20)

y_encoder = label_encode()
y_train = y_encoder.fit_transform(y_train)
y_test = y_encoder.transform(y_test)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_train_tensor, y_train_tensor)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def transformer_classifier() -> None:
    class TransformerClassifier(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 128,
            num_heads: int = 4,
            hidden_dim: int = 256,
            num_classes: int = 10,
            num_layers: int = 1,
        ) -> None:
            super().__init__()

            self.embedding = nn.Embedding(vocab_size, embedding_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
            )

            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )

            self.classifier = nn.Linear(embedding_dim, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embedding(x)  # token embedding

            x = self.transformer(x)  # transformer
            x = x.mean(dim=1)
            x = self.classifier(x)

            return x

    # model configurtion

    embedding_dim = 128
    num_heads = 4
    hidden_dim = 256
    num_layers = 1
    num_classes = 10
    vocab_size = len(vocab)

    model = TransformerClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    epochs = 10

    mlflow.set_experiment("message_classifier")
    with mlflow.start_run(run_name="auto_transformer"):
        mlflow.log_params(
            {
                "model": type(model).__name__,
                "embedding_dim": embedding_dim,
                "num_heads": num_heads,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_classes": num_classes,
                # "vocab_size": vocab_size,
                "epochs": epochs,
                "criterion": type(criterion).__name__,
                "optimizer": type(optimizer).__name__,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "batch_size": train_loader.batch_size,
            }
        )

        # total_params = sum(
        #     p.numel()
        #     fort p in model.parameters()
        #     if p.requires_grad
        # )

        for epoch in range(epochs):
            model.train()

            running_loss = 0.0

            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = model(x_batch)

                loss = criterion(predictions, y_batch)

                loss.backward()

                optimizer.step()
                running_loss += loss.item()

            print(f"epoch {epoch}: loss-> {running_loss/len(train_loader)}")

        model.eval()

        with torch.no_grad():
            predictions = model(x_test_tensor)
            predictions = torch.argmax(predictions, dim=1)

        y_true = y_test_tensor.numpy()
        y_pred = predictions.numpy()

        precision = precision_score(y_true, y_pred, zero_division=0)

        recall = recall_score(y_true, y_pred, zero_division=0)

        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Log metrics
        mlflow.log_metrics(
            {
                "precision_macro": float(precision),
                "recall_macro": float(recall),
                "f1_macro": float(f1_macro),
                "f1_weighted": float(f1_weighted),
            }
        )

        confusion = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion, display_labels=y_encoder.classes_
        )

        fig, ax = plt.subplots(figsize=(8, 8))

        disp.plot(ax=ax, xticks_rotation=90, colorbar=False)

        mlflow.log_figure(fig, "confusion_matrix.png")

        plt.close(fig)

        mlflow.pytorch.log_model(
            model, name="transformer_model", serialization_format="pickle"
        )


transformer_classifier()

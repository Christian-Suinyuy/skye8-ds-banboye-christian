from collections import Counter

import pandas as pd
import torch

from .pipeline import split

splits = split()

x_train = splits[0].message_text


def tokenizer(x_train: pd.Series) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in x_train:
        tokens = text.lower().split()
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word in counter:
        vocab[word] = len(vocab)

    return vocab


text = "send money but ei no come  this is very stupid"


def text_to_ids(text: str, vocab: dict[str, int]) -> list:
    tokens = text.lower().split()
    results = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    return results


def encode_text(
    texts: pd.Series, vocab: dict[str, int], max_lenght: int
) -> torch.Tensor:
    encoded = []

    for text in texts:
        ids = text_to_ids(text, vocab)

        ids = ids[:max_lenght]
        ids += [vocab["<PAD>"]] * (max_lenght - len(ids))

        encoded.append(ids)
    return torch.tensor(encoded, dtype=torch.long)


vocab = tokenizer(x_train)
print(text_to_ids(text, vocab))

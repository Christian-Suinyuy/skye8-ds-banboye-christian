import os
from decimal import Decimal

import pandas as pd

# Get the directory of this file and construct the path to the data
current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, "data", "raw", "model_pricing.csv"))

model_pricing = {
    "mlg-translate-sm": {"prompt_price": "1.5e-07", "completion_price": "6e-07"},
    "mlg-translate-lg": {"prompt_price": "8e-07", "completion_price": "3.2e-06"},
    "mlg-tutor-sm": {"prompt_price": "2e-07", "completion_price": "8e-07"},
    "mlg-tutor-lg": {"prompt_price": "1.1e-06", "completion_price": "4.4e-06"},
    "mlg-asr-align": {"prompt_price": "9e-08", "completion_price": "3e-07"},
}  # computed this from model_pricing


def get_model_completion_cost(model: str) -> dict:
    completion_price, promt_price = (
        model_pricing[model]["completion_price"],
        model_pricing[model]["prompt_price"],
    )
    return {"completion_price": completion_price, "prompt_price": promt_price}


def calculate_actual_cost(
    model: str,
    completion_tokens: Decimal,
    input_tokens: Decimal,
    cache_status: bool = False,
) -> Decimal:
    token_price = get_model_completion_cost(model)
    if cache_status:
        return Decimal(0)

    completion_cost = Decimal(completion_tokens) * Decimal(
        token_price["completion_price"]
    )

    input_cost = Decimal(input_tokens) * Decimal(token_price["prompt_price"])
    return completion_cost + input_cost

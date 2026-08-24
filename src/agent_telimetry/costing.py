import os
from decimal import Decimal

import pandas as pd

# Get the directory of this file and construct the path to the data
current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, "data", "raw", "model_pricing.csv"))


def get_model_completion_cost(model: str) -> dict:
    completion_price, promt_price = df.query(f"model == '{model}'")[
        ["completion_usd_per_token", "prompt_usd_per_token"]
    ].iloc[0]
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

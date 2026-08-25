import os
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv()
connection_string = os.getenv("DATABASE_STRING")

connection = psycopg.connect(connection_string or "")


def load_calls(raw_dir: str) -> None:
    print("loading calls...")
    raw_dir_path = Path(raw_dir)
    calls_file = raw_dir_path / "calls.csv"

    df = pd.read_csv(calls_file)

    print(df.head)


load_calls("ffe")

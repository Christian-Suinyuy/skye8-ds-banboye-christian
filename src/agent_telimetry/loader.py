import os
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv()
connection_string = os.getenv("DATABASE_STRING")

connection = psycopg.connect(connection_string or "")


def parse_timestamp(value: object) -> object:
    if pd.isna(value):
        return None

    value_text = str(value).strip()
    if value_text.isdigit():
        return pd.to_datetime(int(value_text), unit="s").to_pydatetime()

    return pd.to_datetime(value_text, format="mixed").to_pydatetime()


def load_model_pricing(raw_dir: str) -> None:
    print("loading model pricing...")
    df = pd.read_csv(Path(raw_dir) / "model_pricing.csv")
    df["model"] = df["model"].str.lower()

    with connection.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO model_pricing (
                model, prompt_usd_per_token, completion_usd_per_token,
                effective_from
            ) VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            df.itertuples(index=False, name=None),
        )


def load_sessions(raw_dir: str) -> None:
    print("loading sessions...")
    df = pd.read_csv(Path(raw_dir) / "sessions.csv")
    df["started_at"] = df["started_at"].map(parse_timestamp)

    with connection.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO sessions (
                session_id, user_id, app_version, device, locale, plan,
                started_at, network
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            df.itertuples(index=False, name=None),
        )


def load_calls(raw_dir: str) -> None:
    """handles insertions line by line so we can catch a contraint arror"""
    print("loading calls...")
    df = pd.read_csv(Path(raw_dir) / "llm_calls.csv", index_col=0)
    df["model"] = df["model"].str.lower()
    df["called_at"] = df["called_at"].map(parse_timestamp)

    foreign_key_voilations = 0

    print(f"loaded {len(df)} calls")

    with connection.cursor() as cur:
        for row in df.itertuples(index=False):
            try:
                with connection.transaction():
                    cur.execute(
                        """
                        INSERT INTO llm_calls (
                            call_id, session_id, idempotency_key, model, tool_name,
                            prompt_tokens, completion_tokens, cache_hit, latency_ms,
                            status, logged_cost_usd, called_at, attempt_no
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        tuple(row),
                    )
            except Exception as e:
                print(e)
                foreign_key_voilations += 1

    print(
        f"""number of rows that voilated forein key
        for llm_calls where constraints : {foreign_key_voilations}"""
    )  #


def load_evaluations(raw_dir: str) -> None:
    """Has call_id wich does not exist in llm_call
    table and voilate foreign key constraints"""

    print("loading evaluations...")
    df = pd.read_csv(Path(raw_dir) / "evaluations.csv")
    df["evaluated_at"] = df["evaluated_at"].map(parse_timestamp)

    forein_key_voilations = 0

    with connection.cursor() as cur:
        for row in df.itertuples(index=False):
            try:
                with connection.transaction():
                    cur.execute(
                        """
                        INSERT INTO evaluation (
                            eval_id, call_id, rubric, verdict, score, evaluator,
                            evaluated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        tuple(row),
                    )
            except Exception:
                forein_key_voilations += 1

    print(
        f"""number of rows that voilated forein key constraints
          for evaluations tabel are: {forein_key_voilations} """
    )


def load_all() -> None:
    """Load tables in foreign-key dependency order."""
    load_model_pricing(raw_dir="./src/agent_telimetry/data/raw")
    load_sessions(raw_dir="./src/agent_telimetry/data/raw")
    load_calls(raw_dir="./src/agent_telimetry/data/cleaned")
    load_evaluations(raw_dir="./src/agent_telimetry/data/raw")
    connection.commit()


load_all()

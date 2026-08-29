import os

import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv()
connection_string = os.getenv("DATABASE_STRING")

connection = psycopg.connect(connection_string or "")


def parse_timestamp(value: object) -> object:
    # if pd.isna(value):
    #     return None

    value_text = str(value).strip()
    if value_text.isdigit():
        return pd.to_datetime(int(value_text), unit="s").to_pydatetime()

    return pd.to_datetime(value_text, format="mixed").to_pydatetime()


def load_model_pricing(model_pricing: pd.DataFrame) -> int:
    with connection.cursor() as cur:
        cur.execute("TRUNCATE TABLE model_pricing CASCADE;")
    connection.commit()

    print("loading model pricing...")
    df = model_pricing.copy()
    df["model"] = df["model"].str.lower()

    rows_affected = 0

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

        rows_affected += cur.rowcount

    connection.commit()

    return rows_affected


def load_sessions(sessions: pd.DataFrame) -> int:
    # clear table first before loading
    with connection.cursor() as cur:
        cur.execute("TRUNCATE TABLE sessions CASCADE;")
    connection.commit()

    print("loading sessions...")
    df = sessions.copy()
    df["started_at"] = df["started_at"].map(parse_timestamp)

    rows_affected = 0

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

        rows_affected += cur.rowcount

    return rows_affected


def load_calls(calls: pd.DataFrame) -> int:
    """handles insertions line by line so we can catch a contraint arror"""

    with connection.cursor() as cur:
        cur.execute("TRUNCATE TABLE llm_calls CASCADE;")
    connection.commit()
    print("loading calls...")
    df = calls.copy()
    df["model"] = df["model"].str.lower()
    df["called_at"] = df["called_at"].map(parse_timestamp)

    foreign_key_voilations = 0
    rows_affected = 0

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
                rows_affected += cur.rowcount
            except Exception as e:
                print(e)
                foreign_key_voilations += 1

    print(
        f"""number of rows that voilated forein key
        for llm_calls where constraints : {foreign_key_voilations}"""
    )  #

    return rows_affected


def load_evaluations(evaluations: pd.DataFrame) -> int:
    """Has call_id wich does not exist in llm_call
    table and voilate foreign key constraints"""

    with connection.cursor() as cur:
        cur.execute("TRUNCATE TABLE evaluation CASCADE;")
    connection.commit()

    print("loading evaluations...")
    df = evaluations.copy()
    df["evaluated_at"] = df["evaluated_at"].map(parse_timestamp)

    forein_key_voilations = 0
    rows_affected = 0

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
                rows_affected += cur.rowcount
            except Exception:
                forein_key_voilations += 1

    print(
        f"""number of rows that voilated forein key constraints
          for evaluations tabel are: {forein_key_voilations} """
    )

    return rows_affected


def load_all() -> None:
    model_pricing = pd.read_csv("./src/agent_telimetry/data/raw/model_pricing.csv")
    sessions = pd.read_csv("./src/agent_telimetry/data/raw/sessions.csv")
    calls = pd.read_csv("./src/agent_telimetry/data/cleaned/llm_calls.csv", index_col=0)
    evaluations = pd.read_csv("./src/agent_telimetry/data/raw/evaluations.csv")

    """Load tables in foreign-key dependency order."""
    if connection is None:
        raise RuntimeError("DATABASE_STRING must be set before loading data")
    load_model_pricing(model_pricing)
    load_sessions(sessions)
    load_calls(calls)
    load_evaluations(evaluations)
    connection.commit()


if __name__ == "__main__":
    load_all()

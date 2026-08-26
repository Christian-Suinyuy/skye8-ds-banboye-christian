import pandas as pd

from ..loader import load_calls, load_evaluations, load_model_pricing, load_sessions

evaluation_data = [
    {
        "eval_id": "EV-TEST-001",
        "call_id": "CL-TEST-0016",
        "rubric": "accuracy",
        "verdict": "pass",
        "score": "0.9500",
        "evaluator": "system",
        "evaluated_at": "2026-06-01 09:05:00",
    },
    {
        "eval_id": "EV-TEST-002",
        "call_id": "CL-TEST-002",
        "rubric": "translation",
        "verdict": "pass",
        "score": "0.9000",
        "evaluator": "human",
        "evaluated_at": "2026-06-01 09:06:00",
    },
    {
        "eval_id": "EV-TEST-003",
        "call_id": "CL-TEST-003",
        "rubric": "relevance",
        "verdict": "pass",
        "score": "0.8800",
        "evaluator": "system",
        "evaluated_at": "2026-06-01 10:20:00",
    },
    {
        "eval_id": "EV-TEST-004",
        "call_id": "CL-TEST-005",
        "rubric": "accuracy",
        "verdict": "fail",
        "score": "0.4200",
        "evaluator": "human",
        "evaluated_at": "2026-06-01 10:25:00",
    },
    {
        "eval_id": "EV-TEST-005",
        "call_id": "CL-TEST-007",
        "rubric": "accuracy",
        "verdict": "pass",
        "score": "0.9700",
        "evaluator": "system",
        "evaluated_at": "2026-06-02 14:40:00",
    },
]

llm_calls_data = [
    {
        "call_id": "CL-TEST-001",
        "session_id": "SS-TEST-001",
        "idempotency_key": "IK-TEST-001",
        "model": "mlg-tutor-sm",
        "tool_name": "none",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cache_hit": False,
        "latency_ms": 500,
        "status": "success",
        "logged_cost_usd": "0.00004500",
        "called_at": "2026-06-01 09:01:00",
        "attempt_no": 1,
    },
    {
        "call_id": "CL-TEST-002",
        "session_id": "SS-TEST-001",
        "idempotency_key": "IK-TEST-002",
        "model": "mlg-translate-sm",
        "tool_name": "lookup_dictionary",
        "prompt_tokens": 200,
        "completion_tokens": 80,
        "cache_hit": False,
        "latency_ms": 700,
        "status": "success",
        "logged_cost_usd": "0.00008000",
        "called_at": "2026-06-01 09:02:00",
        "attempt_no": 1,
    },
    {
        "call_id": "CL-TEST-003",
        "session_id": "SS-TEST-002",
        "idempotency_key": "IK-TEST-003",
        "model": "mlg-tutor-lg",
        "tool_name": "none",
        "prompt_tokens": 500,
        "completion_tokens": 200,
        "cache_hit": True,
        "latency_ms": 300,
        "status": "success",
        "logged_cost_usd": "0.00000000",
        "called_at": "2026-06-01 10:16:00",
        "attempt_no": 1,
    },
    {
        "call_id": "CL-TEST-004",
        "session_id": "SS-TEST-050",
        "idempotency_key": "IK-TEST-004",
        "model": "mlg-tutor-sm",
        "tool_name": "none",
        "prompt_tokens": 300,
        "completion_tokens": 0,
        "cache_hit": False,
        "latency_ms": 100,
        "status": "rate_limited",
        "logged_cost_usd": "0.00000000",
        "called_at": "2026-06-01 10:17:00",
        "attempt_no": 1,
    },
    {
        "call_id": "CL-TEST-020",
        "session_id": "SS-TEST-002",
        "idempotency_key": "IK-TEST-004",
        "model": "mlg-tutor-sm",
        "tool_name": "none",
        "prompt_tokens": 300,
        "completion_tokens": 120,
        "cache_hit": False,
        "latency_ms": 2500,
        "status": "success",
        "logged_cost_usd": "0.00011700",
        "called_at": "2026-06-01 10:18:00",
        "attempt_no": 2,
    },
    {
        "call_id": "CL-TEST-006",
        "session_id": "SS-TEST-003",
        "idempotency_key": "IK-TEST-005",
        "model": "mlg-translate-lg",
        "tool_name": "none",
        "prompt_tokens": 400,
        "completion_tokens": 150,
        "cache_hit": False,
        "latency_ms": 5000,
        "status": "timeout",
        "logged_cost_usd": None,
        "called_at": "2026-06-02 14:31:00",
        "attempt_no": 1,
    },
    {
        "call_id": "CL-TEST-007",
        "session_id": "SS-TEST-003",
        "idempotency_key": "IK-TEST-006",
        "model": "mlg-asr-align",
        "tool_name": "none",
        "prompt_tokens": 1000,
        "completion_tokens": 300,
        "cache_hit": False,
        "latency_ms": 900,
        "status": "success",
        "logged_cost_usd": "0.00021000",
        "called_at": "2026-06-02 14:32:00",
        "attempt_no": 1,
    },
]

sessions_data = [
    {
        "session_id": "SS-TEST-001",
        "user_id": "U-001",
        "app_version": "1.4.0",
        "device": "android",
        "locale": "en-US",
        "plan": "free",
        "started_at": "2026-06-01 09:00:00",
        "network": "wifi",
    },
    {
        "session_id": "SS-TEST-002",
        "user_id": "U-002",
        "app_version": "1.4.0",
        "device": "ios",
        "locale": "fr-FR",
        "plan": "pro",
        "started_at": "2026-06-01 10:15:00",
        "network": "4G",
    },
    {
        "session_id": "SS-TEST-003",
        "user_id": "U-003",
        "app_version": "1.5.0",
        "device": "android",
        "locale": "en-US",
        "plan": "free",
        "started_at": "2026-06-02 14:30:00",
        "network": "5G",
    },
]

model_pricing_data = [
    {
        "model": "mlg-asr-align",
        "prompt_usd_per_token": "0.00000012",
        "completion_usd_per_token": "0.00000030",
        "effective_from": "2026-01-01",
    },
    {
        "model": "mlg-tutor-sm",
        "prompt_usd_per_token": "0.00000015",
        "completion_usd_per_token": "0.00000060",
        "effective_from": "2026-01-01",
    },
    {
        "model": "mlg-translate-sm",
        "prompt_usd_per_token": "0.00000018",
        "completion_usd_per_token": "0.00000055",
        "effective_from": "2026-01-01",
    },
    {
        "model": "mlg-tutor-lg",
        "prompt_usd_per_token": "0.00000030",
        "completion_usd_per_token": "0.00000120",
        "effective_from": "2026-01-01",
    },
    {
        "model": "mlg-translate-lg",
        "prompt_usd_per_token": "0.00000035",
        "completion_usd_per_token": "0.00000140",
        "effective_from": "2026-01-01",
    },
]


model_pricing_csv = pd.DataFrame(model_pricing_data)
sessions_csv = pd.DataFrame(sessions_data)
llm_calls_csv = pd.DataFrame(llm_calls_data)
evaluation_csv = pd.DataFrame(evaluation_data)

print(model_pricing_csv)


def test_loads() -> None:
    results = load_model_pricing(model_pricing_csv)
    assert results == 5

    results = load_sessions(sessions_csv)
    assert results == 3

    results = load_calls(llm_calls_csv)
    assert results == 6

    results = load_evaluations(evaluation_csv)
    assert results == 3


test_loads()

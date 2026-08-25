from decimal import Decimal

from ..costing import calculate_actual_cost


def test_cost_calculation() -> None:
    result = calculate_actual_cost(
        model="mlg-asr-align",
        input_tokens=Decimal(1328),
        completion_tokens=Decimal(413),
        cache_status=False,
    )

    assert result == Decimal("0.00024342")

    result = calculate_actual_cost(
        model="mlg-tutor-lg",
        input_tokens=Decimal(1338),
        completion_tokens=Decimal(239),
        cache_status=True,
    )

    assert result == Decimal(0)

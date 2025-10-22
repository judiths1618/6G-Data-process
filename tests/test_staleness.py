import datetime as dt
import math
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from staleness import staleness_score


def test_staleness_score_with_datetime_inputs() -> None:
    base = dt.datetime(2024, 3, 1, tzinfo=dt.timezone.utc)
    input_time = base + dt.timedelta(hours=1)
    delivery_time = input_time + dt.timedelta(hours=2)

    score = staleness_score(
        age=dt.timedelta(hours=3),
        delivery_time=delivery_time,
        input_time=input_time,
        validity_duration=dt.timedelta(hours=12),
    )

    assert math.isclose(score, 1 - (3 + 2) / 12)


def test_staleness_score_with_numeric_inputs() -> None:
    score = staleness_score(age=3600, delivery_time=7200, input_time=1800, validity_duration=10800)
    # numerator = 3600 + (7200 - 1800) = 9000 seconds => score = 1 - 9000/10800 = 1 - 5/6 = 1/6
    assert math.isclose(score, 1 / 6)


def test_staleness_score_clamped_to_bounds() -> None:
    score_high = staleness_score(
        age=0,
        delivery_time=0,
        input_time=10,
        validity_duration=100,
    )
    assert score_high == 1.0

    score_low = staleness_score(
        age=100,
        delivery_time=200,
        input_time=0,
        validity_duration=150,
    )
    assert score_low == 0.0


def test_staleness_requires_positive_validity_duration() -> None:
    with pytest.raises(ValueError):
        staleness_score(
            age=1,
            delivery_time=2,
            input_time=1,
            validity_duration=0,
        )

"""Utility helpers for quantifying dataset staleness."""

from __future__ import annotations

import datetime as _dt
from typing import Union

Number = Union[int, float]
DurationLike = Union[_dt.timedelta, Number]
TimeLike = Union[_dt.datetime, Number]


def _duration_to_seconds(value: DurationLike) -> float:
    """Return ``value`` in seconds.

    Accepts numeric values that are assumed to already be expressed in seconds
    as well as :class:`datetime.timedelta` instances.  A ``TypeError`` is raised
    for unsupported values to surface programming errors early.
    """

    if isinstance(value, _dt.timedelta):
        return value.total_seconds()
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(
        "Duration values must be numeric seconds or datetime.timedelta instances; "
        f"received {type(value)!r}."
    )


def _time_difference_seconds(delivery_time: TimeLike, input_time: TimeLike) -> float:
    """Return the delivery minus input time expressed in seconds."""

    if isinstance(delivery_time, _dt.datetime) and isinstance(input_time, _dt.datetime):
        return (delivery_time - input_time).total_seconds()

    if isinstance(delivery_time, (int, float)) and isinstance(input_time, (int, float)):
        return float(delivery_time) - float(input_time)

    raise TypeError(
        "Delivery and input times must both be datetimes or numeric seconds; "
        f"received {type(delivery_time)!r} and {type(input_time)!r}."
    )


def staleness_score(
    *,
    age: DurationLike,
    delivery_time: TimeLike,
    input_time: TimeLike,
    validity_duration: DurationLike,
) -> float:
    """Compute the normalized freshness score for a data point.

    The score implements the formula::

        max(0, 1 - (age + delivery_time - input_time) / validity_duration)

    The arguments accept numeric values expressed in seconds or the
    corresponding :mod:`datetime` objects.  The result is always clipped to the
    inclusive range ``[0, 1]``.  A :class:`ValueError` is raised if the
    ``validity_duration`` is not strictly positive.
    """

    validity_seconds = _duration_to_seconds(validity_duration)
    if validity_seconds <= 0:
        raise ValueError("validity_duration must be a positive interval")

    age_seconds = _duration_to_seconds(age)
    propagation_delay = _time_difference_seconds(delivery_time, input_time)

    numerator = age_seconds + propagation_delay
    raw_score = 1.0 - numerator / validity_seconds
    # Guard against propagation_delay being negative (e.g., prefetching) and
    # floating-point imprecision by clamping to [0, 1].
    return max(0.0, min(1.0, raw_score))


__all__ = ["staleness_score"]

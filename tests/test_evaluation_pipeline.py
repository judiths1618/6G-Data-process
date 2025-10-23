import datetime as dt
import math

import pytest

from methods.evaluation_pipeline import (
    evaluate_model_improvement,
    evaluate_time_series_augmentation,
)


def _generate_time_series(rows: int) -> list[dict[str, object]]:
    data: list[dict[str, object]] = []
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    for index in range(rows):
        timestamp = start + dt.timedelta(minutes=index)
        load = float(index)
        binary = float(index % 2)
        target = 3.0 * load + 5.0 * binary
        data.append({"time": timestamp, "load": load, "binary": binary, "target": target})
    return data


def test_evaluate_model_improvement_prefers_augmented_features():
    base_rows = _generate_time_series(20)
    augmented_rows: list[dict[str, object]] = []

    for row in base_rows:
        augmented_rows.append({
            "load": row["load"],
            "binary": row["binary"],
            "interaction": row["load"] * row["binary"],
            "target": row["target"],
        })

    baseline_rows = [{"load": row["load"], "target": row["target"]} for row in base_rows]

    results = evaluate_model_improvement(
        baseline_rows,
        augmented_rows,
        target_feature="target",
        baseline_features=["load"],
        augmented_features=["load", "binary", "interaction"],
        test_ratio=0.25,
    )

    assert results["metric"] == "rmse"
    assert results["baseline"] > results["augmented"]
    assert results["improvement"] > 0.0


def test_evaluate_time_series_augmentation_reports_improvement(tmp_path):
    rows = []
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    for hour in range(24):
        timestamp = start + dt.timedelta(hours=hour)
        signal = math.sin(2 * math.pi * hour / 24.0)
        rows.append((timestamp, signal))

    metrics_csv = "time,load,target\n" + "\n".join(
        f"{timestamp.isoformat()},{(hour % 6) + hour * 0.1},{signal}"
        for hour, (timestamp, signal) in enumerate(rows)
    )
    secondary_csv = "time,noise\n" + "\n".join(
        f"{timestamp.isoformat()},{0.0}" for timestamp, _ in rows
    )

    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text(metrics_csv)
    secondary_path = tmp_path / "secondary.csv"
    secondary_path.write_text(secondary_csv)

    results = evaluate_time_series_augmentation(
        [str(metrics_path), str(secondary_path)],
        target_feature="metrics_target",
        test_ratio=0.25,
    )

    assert results["baseline"] >= results["augmented"]
    assert results["improvement"] >= 0.0


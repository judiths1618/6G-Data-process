import csv
import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dq_local_beam import DEFAULT_RULE
from augmentation import generate_augmented_dataset


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


def test_temporal_jitter_creates_augmented_file(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    _write_csv(
        source,
        [
            ["time", "cpu_usage", "n"],
            ["1700000000", "50", "10"],
        ],
    )

    rule = dict(DEFAULT_RULE)
    rule["patterns"] = [".*\\.csv$"]
    rule["event_time_col"] = "time"
    rule["event_time_format"] = "epoch_s"
    rule["numeric_cols"] = ["cpu_usage", "n"]
    rule["freshness_slo_hours"] = 24

    output_dir = tmp_path / "aug"
    generated = generate_augmented_dataset(
        [str(source)],
        [rule],
        str(output_dir),
        strategy="temporal_jitter",
        repeat=1,
        seed=42,
    )

    assert len(generated) == 1
    augmented_path = Path(generated[0])
    assert augmented_path.exists()

    with augmented_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert len(rows) == 1
    record = rows[0]
    assert "cpu_usage" in record
    assert 0 <= float(record["cpu_usage"]) <= 100
    assert int(float(record["n"])) >= 0


def test_load_scaling_replicates_rows(tmp_path: Path) -> None:
    source = tmp_path / "svc.csv"
    _write_csv(
        source,
        [
            ["time", "cpu_usage", "lat99", "n"],
            ["1700000000", "40", "200", "5"],
        ],
    )

    rule = dict(DEFAULT_RULE)
    rule["patterns"] = [".*svc\\.csv$"]
    rule["event_time_col"] = "time"
    rule["event_time_format"] = "epoch_s"
    rule["numeric_cols"] = ["cpu_usage", "lat99", "n"]
    rule["freshness_slo_hours"] = 24

    output_dir = tmp_path / "scaled"
    generated = generate_augmented_dataset(
        [str(source)],
        [rule],
        str(output_dir),
        strategy="load_scaling",
        repeat=2,
        seed=7,
    )

    assert len(generated) == 1
    augmented_path = Path(generated[0])
    with augmented_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert len(rows) == 2
    for row in rows:
        assert 0 <= float(row["cpu_usage"]) <= 100
        assert float(row["lat99"]) >= 0
        assert int(float(row["n"])) >= 5

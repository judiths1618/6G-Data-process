"""Summarize completeness characteristics of the KUL 6GDALI datasets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


DATASET_DIRS = {
    "antennas_as_features": Path("6GDALI_Datasets/KUL/nomadic_dataset_ULA_static/antennas_as_features"),
    "csi_as_features": Path("6GDALI_Datasets/KUL/nomadic_dataset_ULA_static/csi_as_features"),
    "subcarriers_as_features_complex": Path(
        "6GDALI_Datasets/KUL/nomadic_dataset_ULA_static/subcarriers_as_features_complex"
    ),
    "subcarriers_as_features_real": Path(
        "6GDALI_Datasets/KUL/nomadic_dataset_ULA_static/subcarriers_as_features_real"
    ),
}


@dataclass
class DatasetSummary:
    """Compact representation of completeness metrics for a dataset."""

    files_scanned: int
    rows_scanned: int
    rows_with_missing: int
    files_with_missing: int
    columns_with_missing: Dict[str, int]
    header_variants: Counter[Tuple[str, ...]]

    def as_dict(self) -> Dict[str, object]:
        """Serialize the summary to a JSON-compatible mapping."""

        column_overview = {
            column: {"missing": missing}
            for column, missing in sorted(self.columns_with_missing.items())
        }

        variant_items = sorted(
            self.header_variants.items(), key=lambda item: (-item[1], item[0])
        )
        variant_examples = [
            {
                "count": count,
                "first_columns": list(variant[:5]),
                "last_columns": list(variant[-5:]) if variant else [],
            }
            for variant, count in variant_items[:10]
        ]

        return {
            "files_scanned": self.files_scanned,
            "rows_scanned": self.rows_scanned,
            "rows_with_missing": self.rows_with_missing,
            "files_with_missing": self.files_with_missing,
            "columns_with_missing": column_overview,
            "header_variant_count": len(self.header_variants),
            "header_variants": variant_examples,
        }


def scan_dataset(path: Path, limit: int | None = None) -> DatasetSummary:
    """Scan a dataset directory and build a completeness summary."""

    csv_paths = sorted(path.glob("*.csv"))
    if limit is not None:
        csv_paths = csv_paths[:limit]

    rows_scanned = 0
    rows_with_missing = 0
    files_with_missing = 0
    columns_with_missing: Counter[str] = Counter()
    header_variants: Counter[Tuple[str, ...]] = Counter()

    for csv_path in csv_paths:
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            header = tuple(reader.fieldnames or [])
            header_variants[header] += 1

            file_has_missing = False
            for row in reader:
                rows_scanned += 1
                row_has_missing = False
                for column in header:
                    value = row.get(column)
                    if value is None or value == "":
                        columns_with_missing[column] += 1
                        row_has_missing = True
                if row_has_missing:
                    rows_with_missing += 1
                    file_has_missing = True

            if file_has_missing:
                files_with_missing += 1

    return DatasetSummary(
        files_scanned=len(csv_paths),
        rows_scanned=rows_scanned,
        rows_with_missing=rows_with_missing,
        files_with_missing=files_with_missing,
        columns_with_missing=dict(columns_with_missing),
        header_variants=header_variants,
    )


def build_report(limit: int | None = None) -> Dict[str, object]:
    """Generate completeness summaries for every known dataset directory."""

    report: Dict[str, object] = {}
    for name, directory in DATASET_DIRS.items():
        if not directory.exists():
            report[name] = {"error": f"directory {directory} does not exist"}
            continue

        summary = scan_dataset(directory, limit=limit)
        report[name] = summary.as_dict()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize missing values in the KUL dataset")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally restrict the number of CSV files scanned per dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to save the report to",
    )
    args = parser.parse_args()

    report = build_report(limit=args.limit)
    formatted = json.dumps(report, indent=2)
    print(formatted)

    if args.output is not None:
        args.output.write_text(formatted + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


"""Benchmark helper for comparing pipeline engines.

This module wraps ``dq_local_beam.run`` and measures how long the pipeline
needs to finish with each execution backend.  The generated summary table
includes the runtime together with metrics extracted from the quality report
so analysts can quickly determine which engine offers the best trade-off for
their dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dq_local_beam import _HAVE_BEAM  # type: ignore
from dq_local_beam import run as run_pipeline  # type: ignore


QUALITY_REPORT_PREFIX = "quality_report"
QUALITY_REPORT_SUFFIX = ".json"


@dataclass
class BenchmarkResult:
    """Container for the outcome of a single engine benchmark run."""

    engine: str
    status: str
    elapsed_seconds: Optional[float] = None
    files_processed: Optional[int] = None
    good_rows: Optional[int] = None
    issue_rows: Optional[int] = None
    output_root: Optional[str] = None
    note: Optional[str] = None

    def as_row(self) -> List[str]:
        """Render the result as a list of formatted strings for table output."""

        def fmt(value: Optional[float], precision: int = 2) -> str:
            if value is None:
                return "-"
            return f"{value:.{precision}f}"

        def fmt_int(value: Optional[int]) -> str:
            if value is None:
                return "-"
            return f"{value:,}"

        runtime = fmt(self.elapsed_seconds) if self.elapsed_seconds is not None else "-"
        return [
            self.engine,
            self.status,
            runtime,
            fmt_int(self.files_processed),
            fmt_int(self.good_rows),
            fmt_int(self.issue_rows),
            self.output_root or "-",
            (self.note or "").replace("\n", " "),
        ]


def _single_shard_path(prefix: str, suffix: str) -> str:
    return f"{prefix}-00000-of-00001{suffix}"


def _load_quality_report(dq_out: str) -> Dict[str, object]:
    path = _single_shard_path(os.path.join(dq_out, QUALITY_REPORT_PREFIX), QUALITY_REPORT_SUFFIX)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Quality report not found at {path}")
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _clean_engine_root(engine_root: str) -> None:
    if os.path.exists(engine_root):
        shutil.rmtree(engine_root)


def _benchmark_engine(
    engine: str,
    input_pattern: str,
    config_path: str,
    engine_root: str,
) -> BenchmarkResult:
    if engine == "beam" and not _HAVE_BEAM:
        return BenchmarkResult(engine=engine, status="skipped", note="apache_beam is not installed")

    good_out = os.path.join(engine_root, "good", "rows")
    bad_out = os.path.join(engine_root, "bad", "rows")
    dq_out = os.path.join(engine_root, "dq")

    _clean_engine_root(engine_root)

    os.makedirs(engine_root, exist_ok=True)

    start = time.perf_counter()
    try:
        run_pipeline(
            input_pattern=input_pattern,
            good_out=good_out,
            bad_out=bad_out,
            dq_out=dq_out,
            config_path=config_path,
            engine=engine,
        )
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        return BenchmarkResult(
            engine=engine,
            status="error",
            note=str(exc),
        )
    elapsed = time.perf_counter() - start

    try:
        report = _load_quality_report(dq_out)
    except FileNotFoundError as exc:  # pragma: no cover - surfaced to CLI
        return BenchmarkResult(
            engine=engine,
            status="error",
            elapsed_seconds=elapsed,
            output_root=engine_root,
            note=str(exc),
        )

    files_processed = int(report.get("files_processed", 0))
    good_rows = int(report.get("total_good_rows", 0))
    issue_rows = int(report.get("total_issue_records", 0))

    return BenchmarkResult(
        engine=engine,
        status="ok",
        elapsed_seconds=elapsed,
        files_processed=files_processed,
        good_rows=good_rows,
        issue_rows=issue_rows,
        output_root=engine_root,
    )


def _format_table(rows: List[List[str]], headers: Iterable[str]) -> str:
    columns = list(headers)
    widths = [len(col) for col in columns]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(row: Iterable[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    lines = [format_row(columns), "-+-".join("-" * width for width in widths)]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the sequential and Beam engines")
    parser.add_argument("--input_pattern", required=True, help="Glob expression for input CSV files")
    parser.add_argument("--config", required=True, help="Path to the YAML rule configuration")
    parser.add_argument(
        "--output_root",
        required=True,
        help="Directory where engine-specific outputs will be written",
    )
    parser.add_argument(
        "--engines",
        choices=["sequential", "beam"],
        nargs="+",
        help="Subset of engines to benchmark (default: sequential and beam if available)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    engines = args.engines
    if not engines:
        engines = ["sequential"]
        if _HAVE_BEAM:
            engines.append("beam")

    results: List[BenchmarkResult] = []
    for engine in engines:
        engine_root = os.path.join(args.output_root, engine)
        result = _benchmark_engine(engine, args.input_pattern, args.config, engine_root)
        results.append(result)

    table_rows = [result.as_row() for result in results]
    headers = [
        "Engine",
        "Status",
        "Runtime (s)",
        "Files",
        "Good rows",
        "Issue rows",
        "Output root",
        "Notes",
    ]
    print(_format_table(table_rows, headers))

    if any(result.status == "error" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

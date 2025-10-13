"""Interactive quality-report dashboard builder.

This helper runs the data-quality pipeline and assembles a self-contained
interactive HTML dashboard summarizing the results. The dashboard embeds the
combined outlier visualizations that are generated for each dataset.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import textwrap
import webbrowser
from typing import Any, Dict, List

import dq_local_beam


def _single_shard(prefix: str, suffix: str) -> str:
    """Return the canonical shard file name used by the pipeline."""

    return f"{prefix}-00000-of-00001{suffix}"


def _load_quality_report(dq_out: str) -> Dict[str, Any]:
    report_path = _single_shard(os.path.join(dq_out, "quality_report"), ".json")
    if not os.path.exists(report_path):
        raise FileNotFoundError(
            "Quality report not found. Ensure the data-quality pipeline completed successfully."
        )
    with open(report_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _build_issue_summary(issue_summary: List[Dict[str, Any]]) -> str:
    if not issue_summary:
        return "<p>No data-quality issues were reported.</p>"

    rows: List[str] = [
        "<table class=\"issues\">",
        "  <thead><tr><th>Dimension</th><th>Issues</th><th>Scenarios</th></tr></thead>",
        "  <tbody>",
    ]
    for entry in issue_summary:
        scenarios = []
        for sc in entry.get("scenarios", []):
            label = html.escape(str(sc.get("scenario", "Unknown")))
            scenarios.append(f"{label} ({sc.get('count', 0)})")
        dimension = html.escape(str(entry.get("dimension", "Unknown")))
        count = entry.get("issue_count", 0)
        scenario_text = ", ".join(scenarios) or "—"
        rows.append(f"    <tr><th>{dimension}</th><td>{count}</td><td>{scenario_text}</td></tr>")
    rows.extend(["  </tbody>", "</table>"])
    return "\n".join(rows)


def _summarize_numeric_columns(numeric_columns: Dict[str, Any]) -> str:
    if not numeric_columns:
        return "<p>No numeric features detected.</p>"

    rows: List[str] = [
        "<table class=\"features\">",
        "  <thead><tr><th>Feature</th><th>Stats</th><th>Outliers</th></tr></thead>",
        "  <tbody>",
    ]
    for name, stats in sorted(numeric_columns.items()):
        desc = stats.get("description")
        name_html = html.escape(str(name))
        desc_html = html.escape(str(desc)) if desc else ""
        header = name_html if not desc else f"{name_html}<div class=\"feature-desc\">{desc_html}</div>"
        quantiles = stats.get("quantiles") or []
        q1 = quantiles[1] if len(quantiles) >= 4 else None
        median = quantiles[2] if len(quantiles) >= 3 else None
        q3 = quantiles[3] if len(quantiles) >= 4 else None
        profile = [
            f"Count: {stats.get('count', 0)}",
            f"Min: {stats.get('min')}",
            f"Max: {stats.get('max')}",
            f"Mean: {stats.get('mean')}",
            f"Std dev: {stats.get('stddev')}",
            f"Q1: {q1}",
            f"Median: {median}",
            f"Q3: {q3}",
        ]
        outliers = stats.get("outliers") or {}
        fences = textwrap.dedent(
            f"""
            Lower fence: {outliers.get('lower_fence')}
            Upper fence: {outliers.get('upper_fence')}
            IQR: {outliers.get('iqr')}
            Total outliers: {outliers.get('count', 0)}
            """
        ).strip()
        rows.append(
            "    <tr><th>{header}</th><td><pre>{profile}</pre></td><td><pre>{fences}</pre></td></tr>".format(
                header=header,
                profile=html.escape("\n".join(profile)),
                fences=html.escape(fences),
            )
        )
    rows.extend(["  </tbody>", "</table>"])
    return "\n".join(rows)


def _relpath(from_path: str, to_path: str) -> str:
    return os.path.relpath(to_path, os.path.dirname(from_path))


def _build_dashboard(report: Dict[str, Any], dq_out: str) -> str:
    index_path = os.path.join(dq_out, "interactive_report.html")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    lines: List[str] = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        "  <meta charset=\"utf-8\" />",
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
        "  <title>Data quality dashboard</title>",
        "  <style>",
        "    body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; background: #f6f8fa; color: #24292f; }",
        "    header { background: #1f2933; color: #fff; padding: 1.5rem 2rem; }",
        "    main { padding: 2rem; max-width: 1100px; margin: 0 auto; }",
        "    h1 { margin: 0 0 0.5rem 0; font-size: 2rem; }",
        "    .summary { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top: 1rem; }",
        "    .card { background: #fff; border-radius: 10px; padding: 1rem; box-shadow: 0 1px 3px rgba(15,23,42,0.15); }",
        "    section { margin-top: 2.5rem; }",
        "    section h2 { margin-bottom: 0.75rem; }",
        "    .issues, .features { width: 100%; border-collapse: collapse; margin-top: 1rem; background: #fff; box-shadow: 0 1px 3px rgba(15,23,42,0.1); }",
        "    .issues th, .issues td, .features th, .features td { padding: 0.6rem 0.75rem; border-bottom: 1px solid #d8dee4; text-align: left; vertical-align: top; }",
        "    .features th { width: 20%; }",
        "    .features pre { background: #f8fafc; border-radius: 6px; padding: 0.5rem; margin: 0; font-size: 0.85rem; }",
        "    .feature-desc { font-size: 0.8rem; color: #475569; font-weight: normal; }",
        "    iframe { width: 100%; min-height: 420px; border: none; box-shadow: 0 1px 3px rgba(15,23,42,0.1); border-radius: 10px; margin-top: 1rem; background: #fff; }",
        "    .viz-links { margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.75rem; }",
        "    .viz-links a { background: #e9f5f2; border: 1px solid #b7e4d9; border-radius: 6px; padding: 0.4rem 0.6rem; text-decoration: none; color: #1f2933; font-size: 0.9rem; }",
        "    footer { padding: 1.5rem 2rem; background: #0f172a; color: #94a3b8; margin-top: 3rem; font-size: 0.85rem; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <header>",
        "    <h1>Data quality dashboard</h1>",
        "    <p>Interactive summary generated from pattern: <code>{pattern}</code></p>".format(
            pattern=html.escape(str(report.get("input_pattern", "")))
        ),
        "    <div class=\"summary\">",
        "      <div class=\"card\"><strong>Files processed</strong><div>{}</div></div>".format(
            report.get("files_processed", 0)
        ),
        "      <div class=\"card\"><strong>Total good rows</strong><div>{}</div></div>".format(
            report.get("total_good_rows", 0)
        ),
        "      <div class=\"card\"><strong>Total issue records</strong><div>{}</div></div>".format(
            report.get("total_issue_records", 0)
        ),
        "    </div>",
        "  </header>",
        "  <main>",
        "    <section>",
        "      <h2>Issue summary</h2>",
        _build_issue_summary(report.get("issue_summary", [])),
        "    </section>",
    ]

    for file_entry in report.get("per_file", []):
        file_name = file_entry.get("file", "(unknown)")
        visualizations = file_entry.get("visualizations") or {}
        numeric_columns = file_entry.get("numeric_columns") or {}
        combined_dashboard = visualizations.get("combined_outliers_dashboard")
        lines.extend(
            [
                "    <section>",
                "      <h2>{}</h2>".format(html.escape(str(file_name))),
                "      <p>Good rows: {}</p>".format(file_entry.get("good_rows", 0)),
                _summarize_numeric_columns(numeric_columns),
            ]
        )
        if combined_dashboard:
            combined_path = os.path.join(dq_out, combined_dashboard)
            rel = _relpath(index_path, combined_path)
            lines.append("      <iframe src=\"{}\" title=\"Combined outlier view\"></iframe>".format(rel))
        other_links: List[str] = []
        for feature_name, rel_path in sorted(visualizations.items()):
            if feature_name == "combined_outliers_dashboard":
                continue
            other_links.append(
                "<a href=\"{href}\" target=\"_blank\">{label}</a>".format(
                    href=_relpath(index_path, os.path.join(dq_out, rel_path)),
                    label=html.escape(str(feature_name)),
                )
            )
        if other_links:
            lines.append("      <div class=\"viz-links\">{}</div>".format("".join(other_links)))
        lines.append("    </section>")

    lines.extend(
        [
            "  </main>",
            "  <footer>",
            "    Generated by <code>quality_dashboard.py</code>. Re-run the script with a new dataset to refresh this report.",
            "  </footer>",
            "</body>",
            "</html>",
        ]
    )

    with open(index_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    return index_path


def run_dashboard(input_pattern: str, config: str, output_root: str, engine: str = "sequential", open_browser: bool = False) -> str:
    good_out = os.path.join(output_root, "good")
    bad_out = os.path.join(output_root, "bad")
    dq_out = os.path.join(output_root, "dq")
    os.makedirs(output_root, exist_ok=True)

    dq_local_beam.run(
        input_pattern=input_pattern,
        good_out=good_out,
        bad_out=bad_out,
        dq_out=dq_out,
        config_path=config,
        engine=engine,
    )

    report = _load_quality_report(dq_out)
    index_path = _build_dashboard(report, dq_out)

    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(index_path)}", new=2)

    return index_path


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an interactive data-quality dashboard.")
    parser.add_argument("--input_pattern", required=True, help="Glob pattern selecting the CSV files to profile")
    parser.add_argument("--config", required=True, help="Path to the YAML rule configuration")
    parser.add_argument("--output_root", required=True, help="Directory where outputs and dashboard will be written")
    parser.add_argument(
        "--engine",
        choices=["auto", "beam", "sequential"],
        default="sequential",
        help="Execution engine to use when running the pipeline (default: sequential)",
    )
    parser.add_argument("--open-browser", action="store_true", help="Open the generated dashboard in the default browser")

    args = parser.parse_args(argv)

    index_path = run_dashboard(
        input_pattern=args.input_pattern,
        config=args.config,
        output_root=args.output_root,
        engine=args.engine,
        open_browser=args.open_browser,
    )

    print(f"Dashboard generated at {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

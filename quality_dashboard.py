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


def _summarize_numeric_columns(
    numeric_columns: Dict[str, Any],
    visualization_links: Dict[str, str] | None = None,
) -> str:
    if not numeric_columns:
        return "<p>No numeric features detected.</p>"

    visualization_links = visualization_links or {}
    rows: List[str] = [
        "<table class=\"features\">",
        "  <thead><tr><th>Feature</th><th>Stats</th><th>Outliers</th><th>Visualization</th></tr></thead>",
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
            "    <tr><th>{header}</th><td><pre>{profile}</pre></td><td><pre>{fences}</pre></td><td>{viz}</td></tr>".format(
                header=header,
                profile=html.escape("\n".join(profile)),
                fences=html.escape(fences),
                viz=_render_distribution_viz(name, stats, visualization_links),
            )
        )
    rows.extend(["  </tbody>", "</table>"])
    return "\n".join(rows)


def _render_distribution_viz(
    feature_name: str, stats: Dict[str, Any], visualization_links: Dict[str, str]
) -> str:
    min_val = stats.get("min")
    max_val = stats.get("max")
    if min_val is None or max_val is None:
        return "<div class=\"sparkline-wrapper sparkline-wrapper--empty\">Not available</div>"

    quantiles = stats.get("quantiles") or []
    q1 = quantiles[1] if len(quantiles) >= 4 else min_val
    median = quantiles[2] if len(quantiles) >= 3 else (min_val + max_val) / 2 if min_val is not None and max_val is not None else None
    q3 = quantiles[3] if len(quantiles) >= 4 else max_val
    outliers = stats.get("outliers") or {}
    distribution = stats.get("distribution") or stats.get("histogram") or {}
    lower_fence = outliers.get("lower_fence")
    upper_fence = outliers.get("upper_fence")
    outlier_count = outliers.get("count")
    hist_edges = distribution.get("edges") or []
    hist_counts = distribution.get("counts") or []

    width = 220
    height = 36
    padding = 12
    span = max(max_val - min_val, 1e-9)

    def _scale(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return padding + (float(value) - float(min_val)) / span * (width - 2 * padding)
        except (TypeError, ValueError):
            return None

    x_min = _scale(min_val)
    x_max = _scale(max_val)
    x_q1 = _scale(q1)
    x_median = _scale(median)
    x_q3 = _scale(q3)
    x_lower_fence = _scale(lower_fence)
    x_upper_fence = _scale(upper_fence)

    histogram_elements: List[str] = []
    if len(hist_edges) == len(hist_counts) + 1 and hist_counts:
        max_count = max(hist_counts)
        if max_count:
            histogram_elements.append('<div class="sparkline-title">Histogram</div>')
            histogram_elements.append(
                f"<svg class=\"sparkline sparkline--hist\" viewBox=\"0 0 {width} {height}\" preserveAspectRatio=\"none\">"
            )
            histogram_baseline = height - 6
            histogram_height = max(histogram_baseline - padding, 1)
            for idx, count in enumerate(hist_counts):
                if count <= 0:
                    continue
                left = _scale(hist_edges[idx])
                right = _scale(hist_edges[idx + 1])
                if left is None or right is None:
                    continue
                if right <= left:
                    right = left + 1.0
                normalized = count / max_count
                bar_height = max(normalized * histogram_height, 1.0)
                top = max(histogram_baseline - bar_height, 0.0)
                histogram_elements.append(
                    "  <rect class=\"sparkline-hist-bar\" x=\"{x:.2f}\" y=\"{y:.2f}\" width=\"{w:.2f}\" height=\"{h:.2f}\" />".format(
                        x=left,
                        y=top,
                        w=max(right - left, 1.0),
                        h=bar_height,
                    )
                )
            histogram_elements.append("</svg>")

    elements: List[str] = [
        f"<svg class=\"sparkline sparkline--box\" viewBox=\"0 0 {width} {height}\" preserveAspectRatio=\"none\">",
    ]
    center = height / 2
    whisker_top = center - 6
    whisker_bottom = center + 6
    histogram_baseline = height - 6
    histogram_height = max(histogram_baseline - padding, 1)

    if x_lower_fence is not None:
        elements.append(
            "  <line class=\"sparkline-fence\" x1=\"{x}\" y1=\"{top}\" x2=\"{x}\" y2=\"{bottom}\" />".format(
                x=x_lower_fence,
                top=whisker_top,
                bottom=whisker_bottom,
            )
        )
    if x_upper_fence is not None:
        elements.append(
            "  <line class=\"sparkline-fence\" x1=\"{x}\" y1=\"{top}\" x2=\"{x}\" y2=\"{bottom}\" />".format(
                x=x_upper_fence,
                top=whisker_top,
                bottom=whisker_bottom,
            )
        )

    if x_min is not None and x_max is not None:
        elements.append(
            "  <line class=\"sparkline-whisker\" x1=\"{x1}\" y1=\"{center}\" x2=\"{x2}\" y2=\"{center}\" />".format(
                x1=x_min,
                x2=x_max,
                center=center,
            )
        )
        elements.append(
            "  <line class=\"sparkline-cap\" x1=\"{x}\" y1=\"{top}\" x2=\"{x}\" y2=\"{bottom}\" />".format(
                x=x_min,
                top=whisker_top,
                bottom=whisker_bottom,
            )
        )
        elements.append(
            "  <line class=\"sparkline-cap\" x1=\"{x}\" y1=\"{top}\" x2=\"{x}\" y2=\"{bottom}\" />".format(
                x=x_max,
                top=whisker_top,
                bottom=whisker_bottom,
            )
        )

    if x_q1 is not None and x_q3 is not None:
        box_left = min(x_q1, x_q3)
        box_width = abs(x_q3 - x_q1)
        elements.append(
            "  <rect class=\"sparkline-iqr\" x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"12\" />".format(
                x=box_left,
                y=center - 6,
                w=max(box_width, 2),
            )
        )

    if x_median is not None:
        elements.append(
            "  <line class=\"sparkline-median\" x1=\"{x}\" y1=\"{top}\" x2=\"{x}\" y2=\"{bottom}\" />".format(
                x=x_median,
                top=whisker_top,
                bottom=whisker_bottom,
            )
        )

    elements.append("</svg>")

    badge = ""
    if outlier_count is not None:
        badge = "<div class=\"sparkline-meta\">Outliers: {}</div>".format(html.escape(str(outlier_count)))

    classes = ["sparkline-wrapper"]
    if histogram_elements:
        classes.append("sparkline-wrapper--split")

    distribution_href = visualization_links.get(feature_name)
    outlier_href = visualization_links.get(f"{feature_name}_outliers")

    hist_block = "".join(histogram_elements)
    if hist_block and distribution_href:
        hist_block = "<a class=\"sparkline-link\" href=\"{href}\" target=\"_blank\" rel=\"noopener\" title=\"Open interactive distribution\">{content}</a>".format(
            href=html.escape(distribution_href, quote=True),
            content=hist_block,
        )

    box_block = "".join(elements)
    if box_block and outlier_href:
        box_block = "<a class=\"sparkline-link\" href=\"{href}\" target=\"_blank\" rel=\"noopener\" title=\"Open interactive outlier view\">{content}</a>".format(
            href=html.escape(outlier_href, quote=True),
            content=box_block,
        )

    return "<div class=\"{classes}\">{hist}{box}{badge}</div>".format(
        classes=" ".join(classes),
        hist=hist_block,
        box=box_block,
        badge=badge,
    )


def _relpath(from_path: str, to_path: str) -> str:
    return os.path.relpath(to_path, os.path.dirname(from_path))


def _build_dashboard(report: Dict[str, Any], dq_out: str) -> str:
    index_path = os.path.join(dq_out, "interactive_report.html")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    feature_registry: List[Dict[str, Any]] = []

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
        "    .sparkline-wrapper { display: flex; flex-direction: column; gap: 0.35rem; }",
        "    .sparkline-wrapper--split { gap: 0.6rem; }",
        "    .sparkline-wrapper--empty { color: #64748b; font-size: 0.85rem; }",
        "    .sparkline-title { font-size: 0.75rem; font-weight: 600; color: #475569; }",
        "    .sparkline { width: 100%; height: 40px; }",
        "    .sparkline--hist { height: 48px; }",
        "    .sparkline--box { height: 40px; }",
        "    .sparkline-whisker { stroke: #94a3b8; stroke-width: 2; }",
        "    .sparkline-cap { stroke: #94a3b8; stroke-width: 2; }",
        "    .sparkline-iqr { fill: #bfdbfe; opacity: 0.9; }",
        "    .sparkline-hist-bar { fill: rgba(59,130,246,0.35); stroke: rgba(37,99,235,0.45); stroke-width: 0.5; }",
        "    .sparkline-median { stroke: #1d4ed8; stroke-width: 2; }",
        "    .sparkline-fence { stroke: #f97316; stroke-width: 1.5; stroke-dasharray: 4 3; }",
        "    .sparkline-meta { font-size: 0.8rem; color: #475569; }",
        "    .sparkline-link { display: block; border-radius: 8px; padding: 0.2rem 0.25rem; transition: box-shadow 0.15s ease, transform 0.15s ease; text-decoration: none; color: inherit; }",
        "    .sparkline-link:hover { box-shadow: 0 0 0 3px rgba(37,99,235,0.2); transform: translateY(-1px); }",
        "    .sparkline-link:focus { outline: none; box-shadow: 0 0 0 3px rgba(37,99,235,0.35); }",
        "    .overview { margin-top: 2.5rem; }",
        "    .overview-card { background: #fff; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(15,23,42,0.1); }",
        "    .overview-card h2 { margin-top: 0; }",
        "    .overview-intro { margin: 0 0 1rem 0; color: #475569; }",
        "    .overview-layout { display: grid; gap: 1.5rem; grid-template-columns: minmax(240px, 320px) 1fr; align-items: stretch; }",
        "    .feature-library { border: 1px solid #d8dee4; border-radius: 10px; background: #f8fafc; padding: 1rem; display: flex; flex-direction: column; gap: 0.75rem; max-height: 420px; overflow: auto; }",
        "    .feature-library h3 { margin: 0; font-size: 1rem; color: #1f2937; }",
        "    .feature-library p { margin: 0; font-size: 0.85rem; color: #475569; }",
        "    .feature-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.6rem; }",
        "    .feature-item { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.55rem 0.75rem; cursor: grab; display: flex; flex-direction: column; gap: 0.2rem; transition: border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease; }",
        "    .feature-item strong { font-size: 0.95rem; color: #1f2937; }",
        "    .feature-item span { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; color: #475569; }",
        "    .feature-item__desc { font-size: 0.75rem; color: #64748b; }",
        "    .feature-item:hover { border-color: #94a3b8; box-shadow: 0 6px 14px rgba(148,163,184,0.25); transform: translateY(-1px); }",
        "    .feature-item--dragging { opacity: 0.6; }",
        "    .comparison-dropzone { border: 2px dashed #cbd5f5; border-radius: 12px; background: #f1f5f9; padding: 1rem 1.25rem; display: flex; flex-direction: column; gap: 0.85rem; min-height: 420px; transition: border-color 0.2s ease, background 0.2s ease; }",
        "    .comparison-dropzone h3 { margin: 0; font-size: 1.05rem; }",
        "    .comparison-dropzone p { margin: 0; font-size: 0.85rem; color: #475569; }",
        "    .comparison-dropzone--hover { border-color: #2563eb; background: #e0ecff; }",
        "    .comparison-dropzone--active { border-style: solid; border-color: #2563eb; background: #eff6ff; }",
        "    .comparison-selection { display: flex; flex-wrap: wrap; gap: 0.5rem; }",
        "    .feature-chip { background: #2563eb; color: #fff; border: none; border-radius: 999px; padding: 0.35rem 0.85rem; font-size: 0.8rem; display: inline-flex; align-items: center; gap: 0.4rem; cursor: pointer; box-shadow: 0 4px 10px rgba(37,99,235,0.25); }",
        "    .feature-chip__remove { font-weight: 600; font-size: 1.05rem; line-height: 1; }",
        "    .feature-chip:hover { background: #1d4ed8; }",
        "    .comparison-chart { flex: 1; min-height: 320px; border-radius: 10px; background: #fff; box-shadow: inset 0 0 0 1px #e2e8f0; overflow: hidden; }",
        "    .comparison-placeholder { color: #64748b; display: flex; align-items: center; justify-content: center; height: 100%; font-size: 0.95rem; text-align: center; padding: 1rem; }",
        "    @media (max-width: 900px) { .overview-layout { grid-template-columns: 1fr; } .feature-library { max-height: none; } .comparison-dropzone { min-height: 360px; } }",
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
        "    <section class=\"overview\">",
        "      <div class=\"overview-card\">",
        "        <h2>Overview comparison lab</h2>",
        "        <p class=\"overview-intro\">Drag important numeric features into the comparison board to explore multi-feature distribution overlays. Click a chip to remove it.</p>",
        "        <div class=\"overview-layout\">",
        "          <div class=\"feature-library\">",
        "            <h3>Feature library</h3>",
        "            <p>Select and drag features into the board.</p>",
        "            <ul class=\"feature-list\" id=\"feature-library-list\"></ul>",
        "          </div>",
        "          <div class=\"comparison-dropzone\" id=\"comparison-dropzone\">",
        "            <div>",
        "              <h3>Comparison board</h3>",
        "              <p>Drop features here to build interactive comparisons.</p>",
        "            </div>",
        "            <div class=\"comparison-selection\" id=\"comparison-selection\"></div>",
        "            <div class=\"comparison-chart\" id=\"comparison-chart\">",
        "              <div class=\"comparison-placeholder\">Drag features into the board to compare their distributions.</div>",
        "            </div>",
        "          </div>",
        "        </div>",
        "      </div>",
        "    </section>",
        "    <section>",
        "      <h2>Issue summary</h2>",
        _build_issue_summary(report.get("issue_summary", [])),
        "    </section>",
    ]

    for file_entry in report.get("per_file", []):
        file_name = file_entry.get("file", "(unknown)")
        raw_visualizations = file_entry.get("visualizations") or {}
        resolved_visualizations: Dict[str, str] = {}
        for key, rel_path in raw_visualizations.items():
            resolved_visualizations[key] = _relpath(index_path, os.path.join(dq_out, rel_path))
        numeric_columns = file_entry.get("numeric_columns") or {}
        combined_dashboard = resolved_visualizations.get("combined_outliers_dashboard")
        lines.extend(
            [
                "    <section>",
                "      <h2>{}</h2>".format(html.escape(str(file_name))),
                "      <p>Good rows: {}</p>".format(file_entry.get("good_rows", 0)),
                _summarize_numeric_columns(numeric_columns, resolved_visualizations),
            ]
        )
        if combined_dashboard:
            lines.append(
                "      <iframe src=\"{}\" title=\"Combined outlier view\"></iframe>".format(
                    combined_dashboard
                )
            )
        other_links: List[str] = []
        for feature_name, rel_path in sorted(resolved_visualizations.items()):
            if feature_name == "combined_outliers_dashboard":
                continue
            other_links.append(
                "<a href=\"{href}\" target=\"_blank\">{label}</a>".format(
                    href=rel_path,
                    label=html.escape(str(feature_name)),
                )
            )
        if other_links:
            lines.append("      <div class=\"viz-links\">{}</div>".format("".join(other_links)))
        lines.append("    </section>")

        for feature_name, stats in sorted(numeric_columns.items()):
            distribution = stats.get("distribution") or stats.get("histogram") or {}
            edges = distribution.get("edges") or []
            counts = distribution.get("counts") or []
            if not (len(edges) == len(counts) + 1 and counts):
                continue
            quantiles = stats.get("quantiles") or []
            feature_registry.append(
                {
                    "id": f"{file_name}::{feature_name}",
                    "file": file_name,
                    "file_label": os.path.basename(file_name) or file_name,
                    "feature": feature_name,
                    "description": stats.get("description"),
                    "distribution": distribution,
                    "stats": {
                        "count": stats.get("count"),
                        "min": stats.get("min"),
                        "max": stats.get("max"),
                        "mean": stats.get("mean"),
                        "stddev": stats.get("stddev"),
                        "q1": quantiles[1] if len(quantiles) >= 4 else None,
                        "median": quantiles[2] if len(quantiles) >= 3 else None,
                        "q3": quantiles[3] if len(quantiles) >= 4 else None,
                        "outliers": (stats.get("outliers") or {}).get("count"),
                    },
                    "links": {
                        "distribution": resolved_visualizations.get(feature_name),
                        "outliers": resolved_visualizations.get(f"{feature_name}_outliers"),
                    },
                }
            )

    lines.extend(
        [
            "  </main>",
            "  <footer>",
            "    Generated by <code>quality_dashboard.py</code>. Re-run the script with a new dataset to refresh this report.",
            "  </footer>",
        ]
    )

    feature_registry_json = html.escape(json.dumps(feature_registry), quote=False)
    comparison_script = textwrap.dedent(
        """
        (function() {
          const registryEl = document.getElementById('feature-registry');
          if (!registryEl) {
            return;
          }
          let features = [];
          try {
            features = JSON.parse(registryEl.textContent || '[]');
          } catch (error) {
            console.warn('Unable to parse feature registry', error);
            return;
          }
          if (!Array.isArray(features) || !features.length) {
            return;
          }
          if (typeof Plotly === 'undefined') {
            console.warn('Plotly library is not available. Feature comparisons are disabled.');
            return;
          }

          const libraryList = document.getElementById('feature-library-list');
          const dropzone = document.getElementById('comparison-dropzone');
          const selection = document.getElementById('comparison-selection');
          const chartId = 'comparison-chart';
          const chartContainer = document.getElementById(chartId);
          const emptyHtml = '<div class="comparison-placeholder">Drag features into the board to compare their distributions.</div>';
          const palette = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#f97316', '#0f766e', '#1d4ed8', '#a16207', '#0891b2'];
          const active = new Map();

          function renderLibraryItem(feature) {
            const item = document.createElement('li');
            item.className = 'feature-item';
            item.draggable = true;
            item.dataset.featureId = feature.id;
            const title = document.createElement('strong');
            title.textContent = feature.feature;
            const origin = document.createElement('span');
            origin.textContent = feature.file_label;
            item.appendChild(title);
            item.appendChild(origin);
            if (feature.description) {
              const desc = document.createElement('div');
              desc.className = 'feature-item__desc';
              desc.textContent = feature.description;
              item.appendChild(desc);
            }
            item.addEventListener('dragstart', (event) => {
              item.classList.add('feature-item--dragging');
              if (event.dataTransfer) {
                event.dataTransfer.effectAllowed = 'copy';
                event.dataTransfer.setData('application/json', JSON.stringify(feature));
                event.dataTransfer.setData('text/plain', feature.id);
              }
            });
            item.addEventListener('dragend', () => {
              item.classList.remove('feature-item--dragging');
            });
            return item;
          }

          function ensurePlaceholder() {
            if (!chartContainer) {
              return;
            }
            chartContainer.innerHTML = emptyHtml;
          }

          function renderSelection() {
            if (!selection) {
              return;
            }
            selection.innerHTML = '';
            active.forEach((feature) => {
              const chip = document.createElement('button');
              chip.type = 'button';
              chip.className = 'feature-chip';
              chip.dataset.featureId = feature.id;
              chip.textContent = feature.feature + ' • ' + feature.file_label;
              const remove = document.createElement('span');
              remove.className = 'feature-chip__remove';
              remove.setAttribute('aria-hidden', 'true');
              remove.textContent = '×';
              chip.appendChild(remove);
              chip.addEventListener('click', () => {
                active.delete(feature.id);
                renderSelection();
                updateChart();
              });
              chip.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  active.delete(feature.id);
                  renderSelection();
                  updateChart();
                }
              });
              selection.appendChild(chip);
            });
          }

          function updateChart() {
            if (!chartContainer) {
              return;
            }
            if (!active.size) {
              Plotly.purge(chartId);
              ensurePlaceholder();
              if (dropzone) {
                dropzone.classList.remove('comparison-dropzone--active');
              }
              return;
            }

            const traces = [];
            let idx = 0;
            active.forEach((feature) => {
              const distribution = feature.distribution || {};
              const edges = Array.isArray(distribution.edges) ? distribution.edges : [];
              const counts = Array.isArray(distribution.counts) ? distribution.counts : [];
              if (!edges.length || edges.length !== counts.length + 1) {
                return;
              }
              const midpoints = [];
              const normalized = [];
              let maxCount = 0;
              counts.forEach((val) => {
                const parsed = Number(val);
                if (!Number.isNaN(parsed)) {
                  maxCount = Math.max(maxCount, parsed);
                }
              });
              if (!maxCount) {
                return;
              }
              for (let i = 0; i < counts.length; i += 1) {
                const left = Number(edges[i]);
                const right = Number(edges[i + 1]);
                if (Number.isNaN(left) || Number.isNaN(right)) {
                  continue;
                }
                midpoints.push((left + right) / 2);
                normalized.push(Number(counts[i]) / maxCount);
              }
              if (!midpoints.length) {
                return;
              }
              traces.push({
                x: midpoints,
                y: normalized,
                mode: 'lines',
                name: feature.feature + ' (' + feature.file_label + ')',
                line: { shape: 'spline', width: 2.5, color: palette[idx % palette.length] },
                hovertemplate: '<b>' + feature.feature + '</b><br>Value=%{x}<br>Normalized count=%{y:.2f}<extra></extra>',
              });
              idx += 1;
            });

            if (!traces.length) {
              Plotly.purge(chartId);
              ensurePlaceholder();
              if (dropzone) {
                dropzone.classList.remove('comparison-dropzone--active');
              }
              return;
            }

            Plotly.react(
              chartId,
              traces,
              {
                margin: { l: 60, r: 40, t: 40, b: 60 },
                hovermode: 'closest',
                template: 'plotly_white',
                legend: { orientation: 'h', y: -0.2 },
                xaxis: { title: 'Value', zeroline: false },
                yaxis: { title: 'Normalized frequency', rangemode: 'tozero' },
                paper_bgcolor: '#ffffff',
                plot_bgcolor: '#ffffff',
              },
              { displaylogo: false, responsive: true }
            );
            if (dropzone) {
              dropzone.classList.add('comparison-dropzone--active');
            }
          }

          features.forEach((feature) => {
            if (!libraryList) {
              return;
            }
            libraryList.appendChild(renderLibraryItem(feature));
          });

          if (dropzone) {
            dropzone.addEventListener('dragover', (event) => {
              event.preventDefault();
              dropzone.classList.add('comparison-dropzone--hover');
            });
            dropzone.addEventListener('dragleave', () => {
              dropzone.classList.remove('comparison-dropzone--hover');
            });
            dropzone.addEventListener('drop', (event) => {
              event.preventDefault();
              dropzone.classList.remove('comparison-dropzone--hover');
              let payload = null;
              if (event.dataTransfer) {
                payload = event.dataTransfer.getData('application/json');
              }
              if (!payload) {
                return;
              }
              try {
                const feature = JSON.parse(payload);
                if (!feature || !feature.id || active.has(feature.id)) {
                  return;
                }
                active.set(feature.id, feature);
                renderSelection();
                updateChart();
              } catch (error) {
                console.warn('Unable to activate feature', error);
              }
            });
          }

          ensurePlaceholder();
        })();
        """
    ).strip()

    lines.extend(
        [
            "  <script src=\"https://cdn.plot.ly/plotly-2.27.0.min.js\"></script>",
            "  <script id=\"feature-registry\" type=\"application/json\">{}</script>".format(
                feature_registry_json
            ),
            "  <script>{}</script>".format(comparison_script),
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

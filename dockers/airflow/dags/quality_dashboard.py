# quality_dashboard.py
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import tempfile
import textwrap
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
# Optional S3/MinIO (only if available at runtime)
try:
    import s3fs  # type: ignore
    HAVE_S3FS = True
except Exception:
    s3fs = None
    HAVE_S3FS = False

try:
    import boto3  # type: ignore
    HAVE_BOTO3 = True
except Exception:
    boto3 = None
    HAVE_BOTO3 = False

# Local pipeline (the minimal sequential implementation)
import dq_local_beam

DEFAULT_ENDPOINT = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
DEFAULT_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
DEFAULT_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
DEFAULT_REGION = os.getenv("AWS_REGION", "us-east-1")


# =========================
# S3 / MinIO helpers
# =========================
def is_s3(path: str | None) -> bool:
    return bool(path) and str(path).startswith("s3://")


def get_s3fs():
    if not HAVE_S3FS:
        raise RuntimeError("s3fs is not installed. Please install s3fs to use s3:// paths.")
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": DEFAULT_ENDPOINT, "region_name": DEFAULT_REGION},
        key=DEFAULT_KEY,
        secret=DEFAULT_SECRET,
    )


def split_s3_url(url: str) -> Tuple[str, str]:
    assert url.startswith("s3://")
    no_scheme = url[5:]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def presigned_url(bucket: str, key: str, expires: int = 24 * 3600) -> str:
    if not HAVE_BOTO3:
        raise RuntimeError("boto3 is not installed; cannot generate presigned URLs.")
    s3 = boto3.client(
        "s3",
        endpoint_url=DEFAULT_ENDPOINT,
        aws_access_key_id=DEFAULT_KEY,
        aws_secret_access_key=DEFAULT_SECRET,
        region_name=DEFAULT_REGION,
    )
    return s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)


def upload_dir(local_dir: Path, dest_root: str) -> None:
    if not is_s3(dest_root):
        return
    fs = get_s3fs()
    bucket, prefix = split_s3_url(dest_root.rstrip("/"))
    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            remote_key = f"{prefix}/{rel}" if prefix else rel
            fs.put(p.as_posix(), f"{bucket}/{remote_key}")


def glob_s3(pattern: str) -> List[str]:
    fs = get_s3fs()
    bucket, key = split_s3_url(pattern)
    matches = fs.glob(f"{bucket}/{key}")
    return [f"s3://{m}" for m in matches]


def stage_inputs_if_s3(input_pattern: str) -> Tuple[str, str | None]:
    if not is_s3(input_pattern):
        return "", None
    matches = glob_s3(input_pattern)
    if not matches:
        raise FileNotFoundError(f"No files matched input pattern on S3: {input_pattern}")
    tmp_root = Path(tempfile.mkdtemp(prefix="dq_inputs_"))
    fs = get_s3fs()
    for src in matches:
        b, k = split_s3_url(src)
        local_path = tmp_root / Path(k).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        fs.get(f"{b}/{k}", local_path.as_posix())
    local_glob = (tmp_root / "*").as_posix()
    return tmp_root.as_posix(), local_glob


def materialize_metadata_if_s3(metadata_path: str | None) -> str | None:
    if not metadata_path or not is_s3(metadata_path):
        return metadata_path
    fs = get_s3fs()
    b, k = split_s3_url(metadata_path)
    tmp_dir = Path(tempfile.mkdtemp(prefix="dq_meta_"))
    local_meta = tmp_dir / Path(k).name
    local_meta.parent.mkdir(parents=True, exist_ok=True)
    fs.get(f"{b}/{k}", local_meta.as_posix())
    return local_meta.as_posix()


# =========================
# Dashboard helpers
# =========================
def _inject_timeseries_from_csvs(per_file: List[Dict[str, Any]],
                                 time_column: str,
                                 ts_limit: int = 2000) -> None:
    """
    为 report['per_file'] 中的每个文件，若能读取到 CSV，
    则为其 numeric_columns[*] 注入 time_series = {time: [...], values: [...]}
    仅当该特征尚无 time_series 时注入；最多读取 ts_limit 行以控制大小。
    """
    if not time_column:
        return

    for file_entry in per_file:
        csv_path = file_entry.get("file")
        if not csv_path or not os.path.exists(csv_path):
            # 如果是相对路径，尝试做一次绝对化
            try:
                csv_path_abs = Path(csv_path).resolve()
                if not csv_path_abs.exists():
                    continue
                csv_path = csv_path_abs.as_posix()
            except Exception:
                continue

        numeric_columns: Dict[str, Any] = file_entry.get("numeric_columns") or {}
        if not numeric_columns:
            continue

        # 仅抽样读取：时间列 + 所有数值特征列
        wanted_cols = [time_column] + [c for c in numeric_columns.keys() if c != time_column]
        # 去重并保持顺序
        seen = set()
        wanted_cols = [c for c in wanted_cols if not (c in seen or seen.add(c))]

        try:
            # 这里不 parse_dates，先读原样；后面交给 JS 正则把 epoch秒/毫秒处理
            df = pd.read_csv(csv_path, usecols=lambda c: c in wanted_cols, nrows=ts_limit)
        except Exception:
            continue

        if time_column not in df.columns:
            # 文件里没有时间列，跳过
            continue

        # 丢掉空时间并重置索引
        ts = df[time_column]
        valid_mask = ts.notna()
        if not valid_mask.any():
            continue
        df = df.loc[valid_mask].reset_index(drop=True)

        times = df[time_column].tolist()

        # 为每个特征注入 time_series（如果还没有）
        for feat_name, stats in numeric_columns.items():
            if feat_name == time_column:
                # 时间列自己不画
                continue
            if isinstance(stats, dict) and stats.get("time_series"):
                # 已存在就不覆盖
                continue
            if feat_name not in df.columns:
                # CSV 没有该列（可能被前置流程过滤/重命名），跳过
                continue
            values = df[feat_name].tolist()
            # 长度对齐
            n = min(len(times), len(values))
            if n <= 1:
                continue
            stats.setdefault("time_series", {"time": times[:n], "values": values[:n]})


def _single_shard(prefix: str, suffix: str) -> str:
    return f"{prefix}-00000-of-00001{suffix}"


def _load_quality_report(dq_out: str) -> Dict[str, Any]:
    report_path = _single_shard(os.path.join(dq_out, "quality_report"), ".json")
    if not os.path.exists(report_path):
        raise FileNotFoundError("Quality report not found. Ensure the data-quality pipeline completed successfully.")
    with open(report_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _build_issue_summary(issue_summary: List[Dict[str, Any]]) -> str:
    if not issue_summary:
        return "<p>No data-quality issues were reported.</p>"
    rows: List[str] = [
        '<table class="issues">',
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


def _render_distribution_viz(
    feature_name: str, stats: Dict[str, Any], visualization_links: Dict[str, str]
) -> str:
    min_val = stats.get("min")
    max_val = stats.get("max")
    if min_val is None or max_val is None:
        return '<div class="sparkline-wrapper sparkline-wrapper--empty">Not available</div>'

    quantiles = stats.get("quantiles") or []
    q1 = quantiles[1] if len(quantiles) >= 4 else min_val
    median = quantiles[2] if len(quantiles) >= 3 else (
        (min_val + max_val) / 2 if (min_val is not None and max_val is not None) else None
    )
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
                f'<svg class="sparkline sparkline--hist" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
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
                    '  <rect class="sparkline-hist-bar" x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" />'.format(
                        x=left, y=top, w=max(right - left, 1.0), h=bar_height
                    )
                )
            histogram_elements.append("</svg>")

    elements: List[str] = [
        f'<svg class="sparkline sparkline--box" viewBox="0 0 {width} {height}" preserveAspectRatio="none">',
    ]
    center = height / 2
    whisker_top = center - 6
    whisker_bottom = center + 6

    if x_lower_fence is not None:
        elements.append(
            '  <line class="sparkline-fence" x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" />'.format(
                x=x_lower_fence, top=whisker_top, bottom=whisker_bottom
            )
        )
    if x_upper_fence is not None:
        elements.append(
            '  <line class="sparkline-fence" x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" />'.format(
                x=x_upper_fence, top=whisker_top, bottom=whisker_bottom
            )
        )

    if x_min is not None and x_max is not None:
        elements.append(
            '  <line class="sparkline-whisker" x1="{x1}" y1="{center}" x2="{x2}" y2="{center}" />'.format(
                x1=x_min, x2=x_max, center=center
            )
        )
        elements.append('  <line class="sparkline-cap" x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" />'.format(
            x=x_min, top=whisker_top, bottom=whisker_bottom
        ))
        elements.append('  <line class="sparkline-cap" x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" />'.format(
            x=x_max, top=whisker_top, bottom=whisker_bottom
        ))

    if x_q1 is not None and x_q3 is not None:
        box_left = min(x_q1, x_q3)
        box_width = abs(x_q3 - x_q1)
        elements.append(
            '  <rect class="sparkline-iqr" x="{x}" y="{y}" width="{w}" height="12" />'.format(
                x=box_left, y=height / 2 - 6, w=max(box_width, 2)
            )
        )

    if x_median is not None:
        elements.append('  <line class="sparkline-median" x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" />'.format(
            x=x_median, top=whisker_top, bottom=whisker_bottom
        ))

    elements.append("</svg>")

    badge = f'<div class="sparkline-meta">Outliers: {html.escape(str(outlier_count))}</div>' if outlier_count is not None else ""
    classes = ["sparkline-wrapper"]
    if histogram_elements:
        classes.append("sparkline-wrapper--split")

    distribution_href = visualization_links.get(feature_name)
    outlier_href = visualization_links.get(f"{feature_name}_outliers")

    hist_block = "".join(histogram_elements)
    if hist_block and distribution_href:
        hist_block = (
            f'<a class="sparkline-link" href="{html.escape(distribution_href, True)}" '
            f'target="_blank" rel="noopener" title="Open interactive distribution">{hist_block}</a>'
        )
    box_block = "".join(elements)
    if box_block and outlier_href:
        box_block = (
            f'<a class="sparkline-link" href="{html.escape(outlier_href, True)}" '
            f'target="_blank" rel="noopener" title="Open interactive outlier view">{box_block}</a>'
        )

    return f'<div class="{" ".join(classes)}">{hist_block}{box_block}{badge}</div>'


def _summarize_numeric_columns(
    numeric_columns: Dict[str, Any],
    visualization_links: Dict[str, str] | None = None,
) -> str:
    if not numeric_columns:
        return "<p>No numeric features detected.</p>"

    visualization_links = visualization_links or {}
    rows: List[str] = [
        '<table class="features">',
        "  <thead><tr><th>Feature</th><th>Stats</th><th>Outliers</th><th>Visualization</th></tr></thead>",
        "  <tbody>",
    ]
    for name, stats in sorted(numeric_columns.items()):
        desc = stats.get("description")
        name_html = html.escape(str(name))
        desc_html = html.escape(str(desc)) if desc else ""
        header = name_html if not desc else f'{name_html}<div class="feature-desc">{desc_html}</div>'
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
            '    <tr><th>{header}</th><td><pre>{profile}</pre></td><td><pre>{fences}</pre></td><td>{viz}</td></tr>'.format(
                header=header,
                profile=html.escape("\n".join(profile)),
                fences=html.escape(fences),
                viz=_render_distribution_viz(name, stats, visualization_links),
            )
        )
    rows.extend(["  </tbody>", "</table>"])
    return "\n".join(rows)


def _relpath(from_path: str, to_path: str) -> str:
    return os.path.relpath(to_path, os.path.dirname(from_path))


def _build_dashboard(report: Dict[str, Any], dq_out: str, *, embed_viz: bool, iframe_height: int) -> str:
    index_path = os.path.join(dq_out, "interactive_report.html")
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    feature_registry: List[Dict[str, Any]] = []

    # ---------- head / style ----------
    lines: List[str] = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8' />",
        "  <meta name='viewport' content='width=device-width, initial-scale=1' />",
        "  <title>Data quality dashboard</title>",
        "  <style>",
        "    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin:0; background:#f6f8fa; color:#24292f;}",
        "    header { background:#1f2937; color:#fff; padding:1.5rem 2rem;}",
        "    main { padding:2rem; max-width:1100px; margin:0 auto;}",
        "    h1 { margin:0 0 .5rem 0; font-size:2rem;}",
        "    .summary { display:grid; gap:1rem; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top:1rem;}",
        "    .card { background:#fff; border-radius:10px; padding:1rem; box-shadow:0 1px 3px rgba(15,23,42,.15);}",
        "    section { margin-top:2.5rem; }",
        "    table { width:100%; border-collapse:collapse; background:#fff; box-shadow:0 1px 3px rgba(15,23,42,.1);}",
        "    th, td { padding:.6rem .75rem; border-bottom:1px solid #d8dee4; text-align:left; vertical-align:top;}",
        "    .features th { width: 20%; }",
        "    .feature-desc { font-size:.85rem; color:#475569; font-weight:normal;}",
        "    pre { background:#f8fafc; border-radius:6px; padding:.5rem; margin:0; font-size:.85rem;}",
        "    .sparkline-wrapper { display:flex; flex-direction:column; gap:.35rem; }",
        "    .sparkline-wrapper--split { gap:.6rem; }",
        "    .sparkline-wrapper--empty { color:#64748b; font-size:.85rem; }",
        "    .sparkline-title { font-size:.75rem; font-weight:600; color:#475569; }",
        "    .sparkline { width:100%; height:40px; }",
        "    .sparkline--hist { height:48px; }",
        "    .sparkline-whisker { stroke:#94a3b8; stroke-width:2; }",
        "    .sparkline-cap { stroke:#94a3b8; stroke-width:2; }",
        "    .sparkline-iqr { fill:#bfdbfe; opacity:.9; }",
        "    .sparkline-hist-bar { fill:rgba(59,130,246,.35); stroke:rgba(37,99,235,.45); stroke-width:.5; }",
        "    .sparkline-median { stroke:#1d4ed8; stroke-width:2; }",
        "    .sparkline-fence { stroke:#f97316; stroke-width:1.5; stroke-dasharray:4 3; }",
        "    .sparkline-meta { font-size:.8rem; color:#475569; }",
        "    .sparkline-link { display:block; border-radius:8px; padding:.2rem .25rem; text-decoration:none; color:inherit; }",
        "    .viz-links { margin-top:.75rem; display:flex; flex-wrap:wrap; gap:.75rem; }",
        "    .viz-links a { background:#e9f5f2; border:1px solid #b7e4d9; border-radius:6px; padding:.4rem .6rem; text-decoration:none; color:#1f2933; font-size:.9rem; }",
        "    .overview-card { background:#fff; border-radius:12px; padding:1.5rem; box-shadow:0 1px 3px rgba(15,23,42,.1); }",
        "    .overview-layout { display:grid; gap:1.5rem; grid-template-columns: minmax(240px, 320px) 1fr; align-items:stretch; }",
        "    .feature-library { border:1px solid #d8dee4; border-radius:10px; background:#f8fafc; padding:1rem; display:flex; flex-direction:column; gap:.75rem; max-height:420px; overflow:auto; }",
        "    .feature-list { list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:.6rem; }",
        "    .feature-item { background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:.55rem .75rem; cursor:grab; display:flex; flex-direction:column; gap:.2rem; }",
        "    .comparison-dropzone { border:2px dashed #cbd5f5; border-radius:12px; background:#f1f5f9; padding:1rem 1.25rem; display:flex; flex-direction:column; gap:.85rem; min-height:420px; }",
        "    .comparison-selection { display:flex; flex-wrap:wrap; gap:.5rem; }",
        "    .feature-chip { background:#2563eb; color:#fff; border:none; border-radius:999px; padding:.35rem .85rem; font-size:.8rem; display:inline-flex; align-items:center; gap:.4rem; cursor:pointer; }",
        "    .feature-chip__remove { font-weight:600; font-size:1.05rem; line-height:1; }",
        "    .comparison-chart { flex:1; min-height:320px; border-radius:10px; background:#fff; box-shadow: inset 0 0 0 1px #e2e8f0; overflow:hidden; }",
        "    footer { padding:1.5rem 2rem; background:#0f172a; color:#94a3b8; margin-top:3rem; font-size:.85rem;}",
        "    @media (max-width: 900px) { .overview-layout { grid-template-columns: 1fr; } .feature-library { max-height:none; } .comparison-dropzone { min-height:360px; } }",
        "    iframe { width:100%; min-height: 420px; border:none; box-shadow:0 1px 3px rgba(15,23,42,.1); border-radius:10px; margin-top:1rem; background:#fff; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <header>",
        "    <h1>Data quality dashboard</h1>",
        f"    <p>Generated from pattern: <code>{html.escape(str(report.get('input_pattern','')))}</code></p>",
        "    <div class='summary'>",
        f"      <div class='card'><strong>Files processed</strong><div>{report.get('files_processed',0)}</div></div>",
        f"      <div class='card'><strong>Total good rows</strong><div>{report.get('total_good_rows',0)}</div></div>",
        f"      <div class='card'><strong>Total issue records</strong><div>{report.get('total_issue_records',0)}</div></div>",
        "    </div>",
        "  </header>",
        "  <main>",
        "    <section>",
        "      <h2>Issue summary</h2>",
        _build_issue_summary(report.get("issue_summary", [])),
        "    </section>",
    ]

    # ---------- per-file blocks ----------
    for file_entry in report.get("per_file", []):
        file_name = file_entry.get("file", "(unknown)")
        numeric_columns = file_entry.get("numeric_columns") or {}
        # resolve visualization links (relative to index_path)
        raw_visualizations = file_entry.get("visualizations") or {}
        resolved_visualizations: Dict[str, str] = {}
        for key, rel_path in raw_visualizations.items():
            resolved_visualizations[key] = _relpath(index_path, os.path.join(dq_out, rel_path))

        lines.extend(
            [
                "    <section>",
                f"      <h2>{html.escape(str(file_name))}</h2>",
                f"      <p>Good rows: {file_entry.get('good_rows', 0)}</p>",
                _summarize_numeric_columns(numeric_columns, resolved_visualizations),
            ]
        )

        # other viz links (skip outlier pages & combined_outliers_dashboard)
        other_links: List[str] = []
        for feature_name, rel_path in sorted(resolved_visualizations.items()):
            if feature_name == "combined_outliers_dashboard" or "outlier" in feature_name.lower():
                continue
            other_links.append(
                f'<a href="{html.escape(rel_path, True)}" target="_blank" rel="noopener">{html.escape(str(feature_name))}</a>'
            )
        if other_links:
            lines.append(f'      <div class="viz-links">{"".join(other_links)}</div>')

        # optional embed: combined outliers dashboard
        combined_key = resolved_visualizations.get("combined_outliers_dashboard")
        if embed_viz and combined_key:
            lines.append(f'      <iframe src="{html.escape(combined_key, True)}" title="Combined outliers dashboard" style="min-height:{iframe_height}px"></iframe>')

        lines.append("    </section>")

        # register features for comparison lab
        for feature_name, stats in sorted(numeric_columns.items()):
            distribution = stats.get("distribution") or stats.get("histogram") or {}
            edges = distribution.get("edges") or []
            counts = distribution.get("counts") or []
            if not (len(edges) == len(counts) + 1 and counts):
                # allow time_series-only features
                series = stats.get("time_series") or {}
                if not (isinstance(series, dict) and series.get("time") and series.get("values")):
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
                    "time_series": stats.get("time_series"),
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

    # ---------- overview comparison lab ----------
    feature_registry_json = html.escape(json.dumps(feature_registry), quote=False)
    comparison_script = textwrap.dedent(
    r"""
    (function() {
      const registryEl = document.getElementById('feature-registry');
      let features = [];
      try { features = JSON.parse(registryEl.textContent || '[]'); } catch (e) { return; }
      if (!Array.isArray(features) || !features.length) return;
      if (typeof Plotly === 'undefined') return;

      const dropzone = document.getElementById('comparison-dropzone');
      const selection = document.getElementById('comparison-selection');
      const libraryList = document.getElementById('feature-library-list');
      const chartId = 'comparison-chart';
      const chartContainer = document.getElementById(chartId);
      const emptyHtml = '<div class="comparison-placeholder">Drag features into the board to compare their distributions.</div>';
      const palette = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#f97316', '#0f766e', '#1d4ed8', '#a16207', '#0891b2'];
      const active = new Map();

      function renderLibraryItem(feature) {
        const li = document.createElement('li');
        li.className = 'feature-item';
        li.draggable = true;
        const title = document.createElement('strong'); title.textContent = feature.feature;
        const origin = document.createElement('span'); origin.textContent = feature.file_label;
        li.appendChild(title); li.appendChild(origin);
        if (feature.description) {
          const d = document.createElement('div'); d.className = 'feature-item__desc'; d.textContent = feature.description; li.appendChild(d);
        }
        li.addEventListener('dragstart', (ev)=> {
          li.classList.add('feature-item--dragging');
          if (ev.dataTransfer) {
            ev.dataTransfer.effectAllowed = 'copy';
            ev.dataTransfer.setData('application/json', JSON.stringify(feature));
          }
        });
        li.addEventListener('dragend', ()=> li.classList.remove('feature-item--dragging'));
        return li;
      }

      function ensurePlaceholder(){ if (chartContainer) chartContainer.innerHTML = emptyHtml; }
      function renderSelection(){
        if (!selection) return;
        selection.innerHTML = '';
        active.forEach((f)=> {
          const chip = document.createElement('button'); chip.type='button'; chip.className='feature-chip'; chip.textContent = f.feature + ' • ' + f.file_label;
          const x = document.createElement('span'); x.className='feature-chip__remove'; x.textContent='×'; chip.appendChild(x);
          chip.addEventListener('click', ()=> { active.delete(f.id); renderSelection(); updateChart(); });
          selection.appendChild(chip);
        });
      }

      // 将 epoch 秒时间戳转换为 JS 可识别的毫秒时间戳；字符串或可被 Date 解析的值原样返回
      function normalizeTimes(arr) {
        if (!Array.isArray(arr) || !arr.length) return arr || [];
        // 所有元素都是数字？
        const allNumeric = arr.every(v => typeof v === 'number' && isFinite(v));
        if (allNumeric) {
          // 简单启发式：小于 10^12 视为秒级；否则视为毫秒
          const scale = Math.max(...arr) < 1e12 ? 1000 : 1;
          return arr.map(v => v * scale);
        }
        // 否则让 Plotly 走 ISO 字符串/可解析日期
        return arr;
      }

      function updateChart(){
        if (!chartContainer) return;

        // 1) 优先收集 time-series 特征
        const tsTraces = [];
        let idx = 0;
        active.forEach((f)=> {
          const series = f.time_series || {};
          const times = Array.isArray(series.time) ? series.time : [];
          const values = Array.isArray(series.values) ? series.values : [];
          if (times.length && times.length === values.length) {
            const x = normalizeTimes(times);
            tsTraces.push({
              x,
              y: values,
              mode: 'lines',
              name: f.feature + ' (' + f.file_label + ')',
              line: { shape: 'linear', width: 2.5, color: palette[idx++ % palette.length] },
              hovertemplate: '<b>' + f.feature + '</b><br>%{x}<br>Value=%{y:.3f}<extra></extra>',
            });
          }
        });

        if (tsTraces.length) {
          Plotly.react(
            chartId,
            tsTraces,
            {
              margin: { l: 60, r: 40, t: 40, b: 60 },
              hovermode: 'x unified',
              template: 'plotly_white',
              legend: { orientation: 'h', y: -0.2 },
              xaxis: {
                title: 'Time',
                type: 'date',
                zeroline: false,
                tickformat: '%Y-%m-%d %H:%M',
                hoverformat: '%Y-%m-%d %H:%M:%S',
              },
              yaxis: { title: 'Value', zeroline: false },
              paper_bgcolor: '#fff',
              plot_bgcolor: '#fff',
            },
            { displaylogo: false, responsive: true }
          );
          dropzone && dropzone.classList.add('comparison-dropzone--active');
          return;
        }

        // 2) 如果没有任何 time-series，回退到直方图对比
        const histTraces = [];
        idx = 0;
        active.forEach((f)=> {
          const dist = f.distribution || {}; const edges = Array.isArray(dist.edges)? dist.edges: []; const counts = Array.isArray(dist.counts)? dist.counts: [];
          if (!(edges.length === counts.length + 1 && counts.length)) return;
          let maxC = 0; counts.forEach(v=> { v = Number(v); if (!Number.isNaN(v)) maxC = Math.max(maxC, v); });
          if (!maxC) return;
          const mid=[], norm=[];
          for(let i=0;i<counts.length;i++){ const l=Number(edges[i]), r=Number(edges[i+1]); if (Number.isNaN(l)||Number.isNaN(r)) continue; mid.push((l+r)/2); norm.push(Number(counts[i])/maxC); }
          histTraces.push({
            x: mid,
            y: norm,
            mode:'lines',
            name: f.feature + ' ('+f.file_label+')',
            line:{shape:'spline', width:2.5, color: palette[idx++ % palette.length]},
            hovertemplate: '<b>'+f.feature+'</b><br>Value=%{x}<br>Normalized count=%{y:.2f}<extra></extra>'
          });
        });

        if (histTraces.length) {
          Plotly.react(
            chartId,
            histTraces,
            {
              margin:{l:60,r:40,t:40,b:60},
              hovermode:'closest',
              template:'plotly_white',
              legend:{orientation:'h', y:-0.2},
              xaxis:{title:'Value', zeroline:false},
              yaxis:{title:'Normalized frequency', rangemode:'tozero'},
              paper_bgcolor:'#fff',
              plot_bgcolor:'#fff'
            },
            {displaylogo:false, responsive:true}
          );
          dropzone && dropzone.classList.add('comparison-dropzone--active');
          return;
        }

        // 3) 两类都没有 → 清空
        Plotly.purge(chartId);
        ensurePlaceholder();
        dropzone && dropzone.classList.remove('comparison-dropzone--active');
      }

      // 初始化库与拖拽
      features.forEach(f=> libraryList && libraryList.appendChild(renderLibraryItem(f)));
      const dz = document.getElementById('comparison-dropzone');
      dz && dz.addEventListener('dragover', (ev)=> { ev.preventDefault(); dz.classList.add('comparison-dropzone--hover'); });
      dz && dz.addEventListener('dragleave', ()=> dz.classList.remove('comparison-dropzone--hover'));
      dz && dz.addEventListener('drop', (ev)=> {
        ev.preventDefault(); dz.classList.remove('comparison-dropzone--hover');
        let payload = ev.dataTransfer ? ev.dataTransfer.getData('application/json') : null; if (!payload) return;
        try { const f = JSON.parse(payload); if (!f || !f.id || active.has(f.id)) return; active.set(f.id, f); renderSelection(); updateChart(); } catch(e){}
      });

      ensurePlaceholder();
    })();
    """
).strip()

    lines.extend(
        [
            '    <section class="overview">',
            '      <div class="overview-card">',
            "        <h2>Overview comparison lab</h2>",
            '        <p class="overview-intro">Drag important numeric features into the comparison board to explore multi-feature distribution overlays.</p>',
            '        <div class="overview-layout">',
            '          <div class="feature-library">',
            "            <h3>Feature library</h3>",
            "            <p>Select and drag features into the board.</p>",
            '            <ul class="feature-list" id="feature-library-list"></ul>',
            "          </div>",
            '          <div class="comparison-dropzone" id="comparison-dropzone">',
            "            <div>",
            "              <h3>Comparison board</h3>",
            "              <p>Drop features here to build interactive comparisons.</p>",
            "            </div>",
            '            <div class="comparison-selection" id="comparison-selection"></div>',
            '            <div class="comparison-chart" id="comparison-chart">',
            '              <div class="comparison-placeholder">Drag features into the board to compare their distributions.</div>',
            "            </div>",
            "          </div>",
            "        </div>",
            "      </div>",
            "    </section>",
            "  </main>",
            "  <footer>Generated by <code>quality_dashboard.py</code>.</footer>",
            '  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>',
            f'  <script id="feature-registry" type="application/json">{feature_registry_json}</script>',
            f"  <script>{comparison_script}</script>",
            "</body>",
            "</html>",
        ]
    )

    Path(index_path).write_text("\n".join(lines), encoding="utf-8")
    return index_path


# =========================
# Runner
# =========================
def run_dashboard(
    input_pattern: str,
    config: str | None,
    output_root: str,
    engine: str = "sequential",
    open_browser: bool = False,
    presign_expires: int = 24 * 3600,
    embed_viz: bool = True,
    iframe_height: int = 560,
) -> str:
    # Stage S3 inputs if needed
    staged_root, local_glob = stage_inputs_if_s3(input_pattern) if is_s3(input_pattern) else ("", None)
    effective_input_pattern = local_glob or input_pattern
    display_pattern = input_pattern

    # metadata passthrough via env (CLI may set --metadata)
    meta_env = os.getenv("DQ_METADATA_PATH")
    if meta_env and is_s3(meta_env):
        os.environ["DQ_METADATA_PATH"] = materialize_metadata_if_s3(meta_env) or ""

    # Ensure local output root (or temp if s3)
    if is_s3(output_root):
        temp_root = Path(tempfile.mkdtemp(prefix="dq_dash_out_"))
        out_root_local = temp_root
    else:
        out_root_local = Path(output_root)
        out_root_local.mkdir(parents=True, exist_ok=True)

    good_out = str(out_root_local / "good")
    bad_out = str(out_root_local / "bad")
    dq_out = str(out_root_local / "dq")

    dq_local_beam.run(
        input_pattern=effective_input_pattern,
        good_out=good_out,
        bad_out=bad_out,
        dq_out=dq_out,
        config_path=config,  # None or "AUTO" → auto-rules inside dq_local_beam
        engine=engine,
    )

    report = _load_quality_report(dq_out)
    if isinstance(report, dict) and "input_pattern" in report:
        report["input_pattern"] = display_pattern

    # 如果用户提供了 --time-column，就尝试为所有数值特征注入 time_series
    time_column = os.getenv("DQ_TIME_COLUMN", "")
    ts_limit = int(os.getenv("DQ_TS_LIMIT", "2000"))
    try:
        _inject_timeseries_from_csvs(report.get("per_file", []), time_column=time_column, ts_limit=ts_limit)
    except Exception as _e:
        # 安静失败，不影响 dashboard 生成
        pass


    index_path = _build_dashboard(report, dq_out, embed_viz=embed_viz, iframe_height=iframe_height)

    if is_s3(output_root):
        upload_dir(out_root_local, output_root.rstrip("/") + "/")
        # index is under "<output_root>/dq/interactive_report.html"
        bucket, prefix = split_s3_url(output_root.rstrip("/"))
        index_key = f"{prefix}/dq/interactive_report.html" if prefix else "dq/interactive_report.html"
        try:
            url = presigned_url(bucket, index_key, expires=presign_expires)
            print(f"[DQ REPORT URL] {url}")
        except Exception as e:
            print(f"[WARN] Failed to create presigned URL: {e}")
    else:
        print(f"[DQ REPORT PATH] file://{Path(index_path).resolve()}")
        if open_browser:
            webbrowser.open(f"file://{Path(index_path).resolve()}", new=2)

    return index_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pattern", required=True, help="CSV glob (local or s3://bucket/prefix/*.csv)")
    parser.add_argument("--config", help="Rules file (local/s3). Omit or set AUTO for auto-rules.")
    parser.add_argument("--metadata", help="Optional metadata file (local or s3). Will be set to env DQ_METADATA_PATH.")
    parser.add_argument("--output_root", required=True, help="Output root (local or s3).")
    parser.add_argument("--engine", choices=["auto", "beam", "sequential"], default="sequential")
    def _b(v: str) -> bool: return str(v).lower() in ("1","true","t","yes","y")
    parser.add_argument("--open-browser", type=_b, default=False)
    parser.add_argument("--expires", type=int, default=int(os.getenv("DQ_REPORT_EXPIRES", "86400")))
    parser.add_argument("--embed-viz", type=_b, default=True, help="Embed combined_outliers_dashboard in an iframe when available.")
    parser.add_argument("--iframe-height", type=int, default=560, help="Height for embedded viz iframe.")
    parser.add_argument("--time-column", help="Name of the column used as time axis for comparison lab.")
    parser.add_argument("--ts-limit", type=int, default=2000, help="Max rows to sample for time-series injection.")


    args = parser.parse_args()

    # materialize config if s3
    config_local = args.config
    if args.time_column:
        os.environ["DQ_TIME_COLUMN"] = args.time_column
        os.environ["DQ_TS_LIMIT"] = str(args.ts_limit)
    if config_local and is_s3(config_local):
        fs = get_s3fs()
        b, k = split_s3_url(config_local)
        tmp_dir = Path(tempfile.mkdtemp(prefix="dq_cfg_"))
        tmp_cfg = tmp_dir / Path(k).name
        tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
        fs.get(f"{b}/{k}", tmp_cfg.as_posix())
        config_local = tmp_cfg.as_posix()

    # pass metadata via env (materialize if s3)
    if args.metadata:
        meta_local = args.metadata
        if is_s3(meta_local):
            meta_local = materialize_metadata_if_s3(meta_local)
        if meta_local:
            os.environ["DQ_METADATA_PATH"] = meta_local

    run_dashboard(
        input_pattern=args.input_pattern,
        config=config_local,
        output_root=args.output_root,
        engine=args.engine,
        open_browser=args.open_browser,
        presign_expires=args.expires,
        embed_viz=args.embed_viz,
        iframe_height=args.iframe_height,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)

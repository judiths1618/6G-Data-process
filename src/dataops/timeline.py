"""timeline — run segmentation, timestamp-collision resolution, cadence estimation.

Separates three concerns the pipeline previously collapsed into a single
``drop_duplicates(subset=[ts_col], keep="last")`` followed by ``sort_values``:

* **Runs.** A backward jump in the timestamp column marks a new acquisition run,
  not corruption. Sorting across a run boundary interleaves independent
  experiments into one fictitious series.
* **Collisions.** Rows sharing a timestamp are only duplicates when nothing else
  distinguishes them. In parameter-sweep datasets the swept factors do: keying on
  ``(time, ram_limit)`` takes the EUR golang subset from 7172 colliding rows to 13.
* **Cadence.** The modal inter-arrival interval is unstable when collision
  residue leaves short steps behind. The median matches what the regularizer
  actually uses (``_preprocess_impl.infer_base_dt``), so the report and the
  transform can no longer disagree by two orders of magnitude.

Everything here is pure: functions take a DataFrame and return a new frame plus a
JSON-serializable report. Nothing mutates its input.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "diffs_in_seconds",
    "enforce_monotonic",
    "DISORDER_POLICIES",
    "detect_runs",
    "estimate_cadence",
    "infer_key_columns",
    "resolve_collisions",
    "COLLISION_POLICIES",
    "METADATA",
]

COLLISION_POLICIES = ("aggregate", "keep_last", "keep_first", "none")

# Columns with more distinct values than this are never treated as sweep factors.
_MAX_KEY_CARDINALITY = 128
# At most this many columns are combined into an inferred composite key.
_MAX_KEY_COLUMNS = 4


def diffs_in_seconds(ts: pd.Series) -> pd.Series:
    """diff(ts) in seconds whether ``ts`` is datetime64, epoch s, or epoch ms.

    The returned Series is indexed by the *right-hand* row of each pair, so a
    negative value at index ``i`` means row ``i`` starts a new run.
    """
    if pd.api.types.is_datetime64_any_dtype(ts):
        return ts.diff().dropna().dt.total_seconds()

    s = pd.to_numeric(ts, errors="coerce").dropna()
    diffs = s.diff().dropna()
    if len(diffs) == 0:
        return diffs
    med = float(s.median())
    if 1e12 <= med < 1e14:  # milliseconds since epoch
        return diffs / 1000.0
    return diffs


def _as_seconds(ts: pd.Series) -> pd.Series:
    """Numeric seconds view of a timestamp column, for arithmetic comparisons."""
    if pd.api.types.is_datetime64_any_dtype(ts):
        return (ts - pd.Timestamp("1970-01-01")).dt.total_seconds()
    s = pd.to_numeric(ts, errors="coerce")
    med = float(s.median()) if s.notna().any() else 0.0
    return s / 1000.0 if 1e12 <= med < 1e14 else s


# --------------------------------------------------------------------------- #
# Run segmentation
# --------------------------------------------------------------------------- #

def detect_runs(
    df: pd.DataFrame,
    ts_col: str,
    *,
    min_backward_seconds: float = 0.0,
    min_overlap_rows: int = 8,
) -> dict:
    """Split a frame into acquisition runs at backward timestamp jumps.

    A backward jump can mean two different things, and conflating them is what
    made the rabbitmq subset look like a single corrupt cell:

    * **A stray out-of-order row** — one row lands early, the series resumes
      immediately. Sorting it into place is correct.
    * **A restarted acquisition run** — a whole block of rows re-covers wall-clock
      time the previous run already spanned. Sorting *this* interleaves two
      independent experiments into one fictitious series.

    The discriminator is how many rows follow the jump before the timeline climbs
    back past the previous run's peak. Blocks of at least ``min_overlap_rows``
    are treated as runs; anything shorter is left for ordinary sorting. Backward
    steps are reported either way, so nothing is hidden by the threshold.

    Returns a dict with a ``run_id`` array aligned to ``df`` positions.
    """
    n = len(df)
    empty = {
        "num_runs": 1 if n else 0,
        "run_id": np.zeros(n, dtype=int),
        "run_sizes": [n] if n else [],
        "boundaries": [],
        "ignored_backward_steps": 0,
        "rows_out_of_order": 0,
        "is_monotonic_increasing": True,
        "has_overlapping_runs": False,
        "min_overlap_rows": int(min_overlap_rows),
    }
    if ts_col not in df.columns or n < 2:
        return empty

    ts = df[ts_col]
    secs = _as_seconds(ts).to_numpy(dtype=float)
    if np.isnan(secs).all():
        return empty

    step = np.diff(secs)
    # NaN timestamps must not manufacture a boundary.
    candidates = np.where(step < -abs(min_backward_seconds))[0] + 1
    candidates = candidates[~np.isnan(step[candidates - 1])]

    boundaries = []
    accepted = []
    rows_out_of_order = 0
    for pos in candidates:
        prev_peak = np.nanmax(secs[:pos])
        tail = secs[pos:]
        beyond = np.where(tail > prev_peak)[0]
        overlap_rows = int(beyond[0]) if len(beyond) else int(len(tail))
        prev_seg = secs[:pos]
        overlap_prev = int(
            np.sum((prev_seg >= secs[pos]) & (prev_seg <= prev_peak))
        )
        if overlap_rows < min_overlap_rows:
            continue  # a stray row, not a run restart — let sorting handle it
        accepted.append(pos)
        rows_out_of_order += overlap_rows
        boundaries.append({
            "row_index": int(pos),
            "prev_timestamp": _scalar(ts.iloc[pos - 1]),
            "next_timestamp": _scalar(ts.iloc[pos]),
            "backward_seconds": float(secs[pos] - secs[pos - 1]),
            "overlap_seconds": float(max(0.0, prev_peak - secs[pos])),
            "overlap_rows_new_run": overlap_rows,
            "overlap_rows_prev_run": overlap_prev,
        })

    run_id = np.zeros(n, dtype=int)
    if accepted:
        run_id[np.array(accepted)] = 1
        run_id = np.cumsum(run_id)

    sizes = np.bincount(run_id, minlength=run_id.max() + 1).tolist() if n else []
    return {
        "num_runs": int(run_id.max()) + 1,
        "run_id": run_id,
        "run_sizes": [int(s) for s in sizes],
        "boundaries": boundaries,
        "ignored_backward_steps": int(len(candidates) - len(accepted)),
        "rows_out_of_order": int(rows_out_of_order),
        "is_monotonic_increasing": bool(ts.is_monotonic_increasing),
        "has_overlapping_runs": any(b["overlap_rows_new_run"] > 0 for b in boundaries),
        "min_overlap_rows": int(min_overlap_rows),
    }


def _scalar(value):
    """JSON-safe scalar for report payloads."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


# --------------------------------------------------------------------------- #
# Cadence
# --------------------------------------------------------------------------- #

def estimate_cadence(
    df: pd.DataFrame,
    ts_col: str,
    *,
    run_id: Optional[np.ndarray] = None,
) -> dict:
    """Estimate the sampling interval, reporting both candidate estimators.

    ``expected_dt_seconds`` is the **median** positive step, matching
    ``transform.preprocess``'s ``infer_base_dt``. The modal step is reported
    alongside it — with the fraction of steps that actually support it — because
    the mode silently wins on a small plurality whenever timestamp collisions
    leave 1-second residue behind.
    """
    info = {
        "expected_dt_seconds": None,
        "median_dt_seconds": None,
        "modal_dt_seconds": None,
        "modal_support": 0.0,
        "estimator": "median",
        "estimators_disagree": False,
        "disagreement_ratio": None,
        "notes": [],
    }
    if ts_col not in df.columns or len(df) < 2:
        info["notes"].append("fewer than 2 timestamps — cannot infer cadence")
        return info

    diffs = _within_run_diffs(df, ts_col, run_id)
    diffs = diffs[diffs > 0]
    if diffs.empty:
        info["notes"].append("no positive-duration intervals")
        return info

    median_dt = float(diffs.median())
    modes = diffs.mode()
    modal_dt = float(modes.iloc[0]) if len(modes) else median_dt
    support = float((diffs == modal_dt).mean())

    info["median_dt_seconds"] = median_dt
    info["modal_dt_seconds"] = modal_dt
    info["modal_support"] = round(support, 4)
    info["expected_dt_seconds"] = median_dt if median_dt > 0 else modal_dt

    if modal_dt > 0 and median_dt > 0:
        ratio = max(median_dt, modal_dt) / min(median_dt, modal_dt)
        info["disagreement_ratio"] = round(float(ratio), 3)
        if ratio >= 2.0:
            info["estimators_disagree"] = True
            info["notes"].append(
                f"modal cadence {modal_dt:g}s (support {support:.1%}) disagrees with the "
                f"median {median_dt:g}s by {ratio:.1f}x; using the median so this report "
                f"matches transform.preprocess:infer_base_dt"
            )
    return info


def _within_run_diffs(
    df: pd.DataFrame, ts_col: str, run_id: Optional[np.ndarray]
) -> pd.Series:
    """Positive timestamp steps, never crossing a run boundary."""
    if run_id is None or len(set(run_id.tolist())) <= 1:
        return diffs_in_seconds(df[ts_col].dropna().sort_values())
    parts = []
    for rid in sorted(set(run_id.tolist())):
        seg = df.loc[run_id == rid, ts_col].dropna().sort_values()
        if len(seg) >= 2:
            parts.append(diffs_in_seconds(seg))
    return pd.concat(parts) if parts else pd.Series(dtype=float)


# --------------------------------------------------------------------------- #
# Collision keys
# --------------------------------------------------------------------------- #

def infer_key_columns(
    df: pd.DataFrame,
    ts_col: str,
    *,
    max_cardinality: int = _MAX_KEY_CARDINALITY,
    max_columns: int = _MAX_KEY_COLUMNS,
) -> dict:
    """Find low-cardinality columns that disambiguate colliding timestamps.

    Greedy: repeatedly add whichever candidate column removes the most remaining
    collisions, stopping when collisions hit zero, no column helps, or
    ``max_columns`` is reached. Returns the chosen key plus the per-candidate
    scores so the dashboard can show why it was chosen.
    """
    result = {
        "key_columns": [],
        "collisions_on_timestamp": 0,
        "residual_collisions": 0,
        "candidates": [],
    }
    if ts_col not in df.columns or df.empty:
        return result

    base = int(df[ts_col].duplicated().sum())
    result["collisions_on_timestamp"] = base
    result["residual_collisions"] = base
    if base == 0:
        return result

    candidates = [
        c for c in df.columns
        if c != ts_col and 1 < df[c].nunique(dropna=False) <= max_cardinality
    ]
    scored = []
    for col in candidates:
        residual = int(df.duplicated(subset=[ts_col, col]).sum())
        scored.append({
            "column": col,
            "cardinality": int(df[col].nunique(dropna=False)),
            "residual_collisions": residual,
            "collisions_resolved": base - residual,
        })
    scored.sort(key=lambda d: (d["residual_collisions"], d["cardinality"]))
    result["candidates"] = scored

    chosen: list[str] = []
    residual = base
    while len(chosen) < max_columns and residual > 0:
        best, best_residual = None, residual
        for cand in scored:
            col = cand["column"]
            if col in chosen:
                continue
            trial = int(df.duplicated(subset=[ts_col, *chosen, col]).sum())
            if trial < best_residual:
                best, best_residual = col, trial
        if best is None:
            break
        chosen.append(best)
        residual = best_residual

    result["key_columns"] = chosen
    result["residual_collisions"] = residual
    return result


# --------------------------------------------------------------------------- #
# Collision resolution
# --------------------------------------------------------------------------- #

def resolve_collisions(
    df: pd.DataFrame,
    ts_col: str,
    *,
    key_columns: Optional[Sequence[str]] = None,
    policy: str = "aggregate",
    run_id: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, dict]:
    """Collapse only rows that are genuinely indistinguishable.

    Rows sharing ``ts_col`` but differing in ``key_columns`` (and/or belonging to
    different runs) are *preserved*: in a parameter sweep they are separate
    experimental conditions, not duplicates. Only rows identical across the full
    key are reduced, using ``policy``:

    ``aggregate``  mean for numerics, last for everything else (default)
    ``keep_last``  legacy behaviour, retained for reproducing older runs
    ``keep_first`` keep the earliest observation of each key
    ``none``       preserve every row and only report the collisions

    Returns ``(frame, report)``. The frame keeps the input row order.
    """
    if policy not in COLLISION_POLICIES:
        raise ValueError(
            f"unknown collision policy {policy!r}; expected one of {COLLISION_POLICIES}"
        )

    key_columns = [c for c in (key_columns or []) if c in df.columns]
    report = {
        "policy": policy,
        "key_columns": list(key_columns),
        "rows_before": int(len(df)),
        "rows_after": int(len(df)),
        "collisions_on_timestamp": 0,
        "collisions_after_key": 0,
        "distinct_conditions_preserved": 0,
        "rows_removed": 0,
        "groups_aggregated": 0,
        "split_by_run": False,
    }
    if ts_col not in df.columns or df.empty:
        return df, report

    report["collisions_on_timestamp"] = int(df[ts_col].duplicated().sum())

    full_key = [ts_col, *key_columns]
    work = df
    if run_id is not None and len(set(run_id.tolist())) > 1:
        work = df.assign(**{"__run__": run_id})
        full_key = ["__run__", *full_key]
        report["split_by_run"] = True

    residual = int(work.duplicated(subset=full_key).sum())
    report["collisions_after_key"] = residual
    report["distinct_conditions_preserved"] = (
        report["collisions_on_timestamp"] - residual
    )

    if policy == "none" or residual == 0:
        report["rows_after"] = int(len(df))
        return df, report

    if policy in ("keep_last", "keep_first"):
        keep = "last" if policy == "keep_last" else "first"
        mask = ~work.duplicated(subset=full_key, keep=keep)
        out = df.loc[mask.to_numpy()].reset_index(drop=True)
    else:
        out = _aggregate_groups(df, work, full_key, report)

    report["rows_after"] = int(len(out))
    report["rows_removed"] = report["rows_before"] - report["rows_after"]
    return out, report


def _aggregate_groups(
    df: pd.DataFrame,
    work: pd.DataFrame,
    full_key: list[str],
    report: dict,
) -> pd.DataFrame:
    """Mean-reduce numeric columns within each colliding group, last otherwise.

    Non-colliding rows are passed through untouched so the common path stays
    exact; only genuinely tied groups are averaged.
    """
    dup_mask = work.duplicated(subset=full_key, keep=False).to_numpy()
    if not dup_mask.any():
        return df.reset_index(drop=True)

    untouched = df.loc[~dup_mask]
    colliding = work.loc[dup_mask]

    numeric = [
        c for c in df.columns
        if c not in full_key and pd.api.types.is_numeric_dtype(df[c])
    ]
    agg = {c: ("mean" if c in numeric else "last")
           for c in df.columns if c not in full_key}
    # Preserve original position so the output can be restored to input order.
    colliding = colliding.assign(__pos__=np.flatnonzero(dup_mask))
    agg["__pos__"] = "first"

    reduced = colliding.groupby(full_key, sort=False, dropna=False).agg(agg)
    report["groups_aggregated"] = int(len(reduced))
    reduced = reduced.reset_index()

    untouched = untouched.assign(__pos__=np.flatnonzero(~dup_mask))
    out = pd.concat([untouched, reduced], ignore_index=True, sort=False)
    out = out.sort_values("__pos__", kind="mergesort")
    out = out.drop(columns=[c for c in ("__pos__", "__run__") if c in out.columns])
    return out[list(df.columns)].reset_index(drop=True)


DISORDER_POLICIES = ("drop", "sort", "none")


def enforce_monotonic(
    df: pd.DataFrame,
    ts_col: str,
    *,
    policy: str = "drop",
    run_id: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, dict]:
    """Make the timeline non-decreasing.

    ``drop`` (default, the conventional treatment) keeps a forward scan: a row
    survives only when its timestamp is strictly greater than the highest one
    kept so far. Rows that jump backwards — including a whole block that
    re-covers an earlier span — are removed rather than shuffled into the middle
    of the series, so the surviving timeline is a genuine prefix-consistent
    ordering rather than a merge of two acquisitions.

    ``sort`` restores the previous behaviour (stable sort, within runs when
    ``run_id`` is given). ``none`` reports without changing the frame.
    """
    if policy not in DISORDER_POLICIES:
        raise ValueError(
            f"unknown disorder policy {policy!r}; expected one of {DISORDER_POLICIES}"
        )

    report = {
        "policy": policy,
        "rows_before": int(len(df)),
        "rows_after": int(len(df)),
        "rows_dropped": 0,
        "backward_steps": 0,
        "was_monotonic": True,
        "sample_dropped_indices": [],
    }
    if ts_col not in df.columns or len(df) < 2:
        return df, report

    secs = _as_seconds(df[ts_col]).to_numpy(dtype=float)
    step = np.diff(secs)
    report["backward_steps"] = int(np.nansum(step < 0))
    report["was_monotonic"] = bool(df[ts_col].is_monotonic_increasing)
    if report["was_monotonic"] or policy == "none":
        return df, report

    if policy == "sort":
        if run_id is not None and len(set(run_id.tolist())) > 1:
            out = (df.assign(__run__=run_id)
                     .sort_values(by=["__run__", ts_col], kind="mergesort",
                                  na_position="last")
                     .drop(columns="__run__"))
        else:
            out = df.sort_values(by=ts_col, kind="mergesort", na_position="last")
        return out.reset_index(drop=True), report

    # drop: forward scan against the running maximum
    keep = np.ones(len(df), dtype=bool)
    running = -np.inf
    for i, t in enumerate(secs):
        if np.isnan(t) or t <= running:
            keep[i] = False
        else:
            running = t
    out = df.loc[keep].reset_index(drop=True)
    dropped = np.flatnonzero(~keep)
    report["rows_after"] = int(len(out))
    report["rows_dropped"] = int(len(dropped))
    report["sample_dropped_indices"] = [int(i) for i in dropped[:5]]
    return out, report


METADATA = {
    "name": "timeline",
    "version": "0.1.0",
    "category": "quality_check",
    "summary": (
        "Run segmentation, sweep-aware timestamp collision resolution, and "
        "median-based cadence estimation."
    ),
    "entrypoint": "dataops.timeline:resolve_collisions",
    "gpu": False,
    "dependencies": ["pandas", "numpy"],
    "inputs": {
        "df": {"type": "DataFrame", "required": True},
        "ts_col": {"type": "str", "required": True},
        "key_columns": {"type": "list[str]", "default": None},
        "policy": {"type": "str", "default": "aggregate"},
    },
    "outputs": {
        "frame": {"type": "DataFrame"},
        "report": {"type": "dict", "schema": "timeline_collision_report"},
    },
    "artifacts": [],
}

"""
Optimized preprocessing for time series data (v2)

Improvements over v1:
1. Column-level missingness tracking (not just row-level)
2. Per-column gap structure features
3. Robust scaler stats (median/IQR) saved for downstream choice
4. Outlier detection & reporting before saving
5. Auto-registration of cond cols (no fragile hardcoded list)
6. Multi-segment support (concatenate all segments, not just longest)
7. Aligned observed_row_mask with reindex (no mask/data disagreement)
8. Safer latency column detection
9. Normalization metadata for downstream soft-clipping
"""
from __future__ import annotations  # ADD THIS — enables PEP 604
import json
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# =============================================================================
# Unit conversion helpers
# =============================================================================

_RAM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGTP]?)\s*$", re.IGNORECASE)


def ram_to_mb(x) -> float:
    """Convert ram_limit string (e.g., '2048M') to MB."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = _RAM_RE.match(s)
    if not m:
        raise ValueError(f"Unsupported ram_limit format: {x}")
    val = float(m.group(1))
    unit = (m.group(2) or "").upper()
    scale = {
        "": 1.0 / 1e6, "K": 1.0 / 1e3, "M": 1.0,
        "G": 1e3, "T": 1e6, "P": 1e9,
    }
    return val * scale.get(unit, 1.0)


def bytes_to_mb(x) -> float:
    if pd.isna(x):
        return np.nan
    try:
        return float(x) / 1e6
    except (ValueError, TypeError):
        return np.nan


def microseconds_to_ms(x) -> float:
    if pd.isna(x):
        return np.nan
    try:
        return float(x) / 1000.0
    except (ValueError, TypeError):
        return np.nan


# =============================================================================
# Time inference & diagnostics
# =============================================================================

def infer_base_dt(time_arr: np.ndarray) -> float:
    """
    Infer base time interval from data.
    Uses median of positive diffs for robustness.
    """
    dt = np.diff(time_arr)
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise ValueError("Not enough timestamps to infer base_dt.")
    return float(np.median(dt))


def diagnose_time_range(times: np.ndarray, verbose: bool = True):
    """Diagnose time range, gaps, and anomalies."""
    t_min, t_max = times.min(), times.max()
    duration_sec = t_max - t_min
    duration_days = duration_sec / 86400

    intervals = np.diff(times)
    large_gaps = intervals > 1000
    n_large_gaps = int(large_gaps.sum())

    info = {
        "duration_sec": float(duration_sec),
        "duration_days": float(duration_days),
        "n_points": len(times),
        "n_large_gaps": n_large_gaps,
    }

    if not verbose:
        return info

    print(f"\n{'='*70}")
    print(f"TIME RANGE DIAGNOSTIC")
    print(f"{'='*70}")
    print(f"  Data points: {len(times)}")
    print(f"  Time range:  {duration_sec:.0f}s ({duration_days:.1f} days)")

    try:
        print(f"  Start: {datetime.fromtimestamp(t_min)}")
        print(f"  End:   {datetime.fromtimestamp(t_max)}")
    except Exception:
        print(f"  Start timestamp: {t_min}")
        print(f"  End timestamp:   {t_max}")

    if n_large_gaps > 0:
        gap_sizes = intervals[large_gaps]
        print(f"\n  ⚠️  {n_large_gaps} large gaps (>1000 s)")
        top = np.sort(gap_sizes / 3600)[:5]
        print(f"     Largest gaps (hours): {top}")

    if duration_days > 2:
        print(f"\n  ⚠️  Data spans {duration_days:.1f} days — "
              f"regularisation may create >80 % gaps")

    return info


# =============================================================================
# Segment extraction (multi-segment aware)
# =============================================================================

def find_segments(df: pd.DataFrame, time_col: str,
                  gap_threshold: float = 1000.0):
    """Return list of (start_idx, end_idx) for contiguous segments."""
    times = df[time_col].values
    intervals = np.diff(times)
    large_gaps = intervals > gap_threshold

    starts = [0] + (np.where(large_gaps)[0] + 1).tolist()
    ends = np.where(large_gaps)[0].tolist() + [len(df)]
    segments = list(zip(starts, ends))
    return segments


def extract_longest_segment(df: pd.DataFrame, time_col: str,
                            gap_threshold: float = 1000.0):
    """Extract the single longest contiguous segment."""
    segments = find_segments(df, time_col, gap_threshold)
    lengths = [e - s for s, e in segments]
    best = int(np.argmax(lengths))
    s, e = segments[best]

    times = df[time_col].values
    dur_h = (times[e - 1] - times[s]) / 3600

    print(f"\n[SEGMENT] Found {len(segments)} segments")
    print(f"[SEGMENT] Keeping longest: rows {s}–{e} "
          f"({lengths[best]} pts, {dur_h:.1f} h)")

    return df.iloc[s:e].copy()


def extract_all_segments(df: pd.DataFrame, time_col: str,
                         gap_threshold: float = 1000.0,
                         min_segment_length: int = 50):
    """
    Extract all segments that meet a minimum length.
    Each segment gets its own time reset so they can be concatenated
    into a single timeline without huge gaps.
    """
    segments = find_segments(df, time_col, gap_threshold)
    kept = []
    total_kept = 0

    print(f"\n[SEGMENTS] Found {len(segments)} segments")

    for i, (s, e) in enumerate(segments):
        length = e - s
        if length < min_segment_length:
            print(f"  Segment {i}: {length} pts — SKIPPED (< {min_segment_length})")
            continue
        seg = df.iloc[s:e].copy()
        dur_h = (seg[time_col].iloc[-1] - seg[time_col].iloc[0]) / 3600
        print(f"  Segment {i}: {length} pts, {dur_h:.1f} h — KEPT")
        kept.append(seg)
        total_kept += length

    if not kept:
        print("[WARNING] No segments meet minimum length, keeping all data")
        return df.copy()

    # Concatenate with time resets: each segment starts where the
    # previous one ended, separated by exactly one base_dt gap.
    base_dt = infer_base_dt(kept[0][time_col].values)
    combined_parts = []
    time_offset = 0.0

    for seg in kept:
        seg = seg.copy()
        seg_times = seg[time_col].values
        seg[time_col] = seg_times - seg_times[0] + time_offset
        time_offset = seg[time_col].iloc[-1] + base_dt
        combined_parts.append(seg)

    combined = pd.concat(combined_parts, ignore_index=True)
    print(f"[SEGMENTS] Total kept: {total_kept} pts across "
          f"{len(kept)} segments")
    return combined


# =============================================================================
# Regularization
# =============================================================================

def regularize(df: pd.DataFrame, time_col="time", base_dt=None,
               skip_if_sparse=True):
    """
    Regularize timeline to uniform intervals.

    Returns (df_regularized, observed_col_mask, base_dt) where
    observed_col_mask is a dict of {col_name: np.ndarray[bool]}.
    """
    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.drop_duplicates(subset=[time_col], keep="last")

    t = df[time_col].to_numpy()

    if base_dt is None:
        base_dt = infer_base_dt(t)
        print(f"[INFO] Auto-inferred base_dt: {base_dt} s")
    else:
        print(f"[INFO] Using specified base_dt: {base_dt} s")

    t_min, t_max = t.min(), t.max()
    n_grid = int((t_max - t_min) / base_dt) + 1
    expected_gap_pct = 100 * (1 - len(t) / n_grid)

    if skip_if_sparse and expected_gap_pct > 80:
        print(f"[WARNING] Regularization would create {expected_gap_pct:.1f}% gaps — skipping")
        # Build per-column observed masks from the raw data
        data_cols = [c for c in df.columns if c != time_col]
        observed_col_mask = {}
        for c in data_cols:
            observed_col_mask[c] = df[c].notna().to_numpy()
        # Row-level mask (all-True since we're keeping originals)
        observed_row_mask = np.ones(len(df), dtype=bool)
        return df, observed_row_mask, observed_col_mask, base_dt

    full = np.linspace(t_min, t_min + (n_grid - 1) * base_dt, n_grid)
    tolerance = base_dt / 2.0

    # ── Aligned mask + reindex ─────────────────────────────────────
    # We use the SAME nearest-neighbor logic for both the mask and the
    # actual reindex so they can never disagree.
    idx_right = np.searchsorted(t, full)
    idx_right = np.clip(idx_right, 0, len(t) - 1)
    idx_left = np.maximum(idx_right - 1, 0)

    dist_right = np.abs(t[idx_right] - full)
    dist_left = np.abs(t[idx_left] - full)

    # Pick the closer neighbour
    use_left = dist_left < dist_right
    best_idx = np.where(use_left, idx_left, idx_right)
    best_dist = np.where(use_left, dist_left, dist_right)

    observed_row_mask = best_dist < tolerance

    # Build the reindexed DataFrame directly from best_idx to guarantee
    # alignment with observed_row_mask.
    df_values = df.reset_index(drop=True)
    data_cols = [c for c in df_values.columns if c != time_col]

    new_data = {time_col: full}
    for c in data_cols:
        col_vals = df_values[c].to_numpy()
        filled = col_vals[best_idx].astype(float)
        # Set non-observed grid points to NaN
        filled[~observed_row_mask] = np.nan
        new_data[c] = filled

    df_reg = pd.DataFrame(new_data)

    # Per-column observed mask: a grid point is observed for column c
    # if (a) the row is observed AND (b) the original value wasn't NaN.
    observed_col_mask = {}
    for c in data_cols:
        orig_notna = df_values[c].notna().to_numpy()
        mapped_notna = orig_notna[best_idx]
        observed_col_mask[c] = observed_row_mask & mapped_notna

    # Stats
    obs_count = observed_row_mask.sum()
    gap_pct = 100 * (1 - obs_count / len(full))
    print(f"[INFO] Grid: {len(full)} pts, Observed: {obs_count} "
          f"({100 - gap_pct:.1f}%), Gaps: {gap_pct:.1f}%")

    if gap_pct > 70:
        print(f"[WARNING] High gap percentage ({gap_pct:.1f}%)")

    return df_reg, observed_row_mask, observed_col_mask, base_dt


# =============================================================================
# Feature engineering (conditioning columns)
# =============================================================================

# Registry: every feature function appends the column names it creates.
_COND_COL_REGISTRY: list[str] = []


def _register_cond(*names):
    """Decorator-style helper to register conditioning column names."""
    for n in names:
        if n not in _COND_COL_REGISTRY:
            _COND_COL_REGISTRY.append(n)


def add_time_features(df: pd.DataFrame, time_col="time"):
    """Add time-based conditioning features."""
    df = df.copy()
    t = df[time_col].to_numpy()
    t_min, t_max = t.min(), t.max()
    duration = t_max - t_min

    df["t_norm"] = (t - t_min) / max(duration, 1.0)
    _register_cond("t_norm")

    if duration > 3600:
        df["sin_day"] = np.sin(2 * np.pi * t / 86400)
        df["cos_day"] = np.cos(2 * np.pi * t / 86400)
        _register_cond("sin_day", "cos_day")

    if 600 < duration < 86400:
        df["sin_hour"] = np.sin(2 * np.pi * t / 3600)
        df["cos_hour"] = np.cos(2 * np.pi * t / 3600)
        _register_cond("sin_hour", "cos_hour")

    return df


def add_gap_structure_features(df: pd.DataFrame,
                               observed_row_mask: np.ndarray):
    """
    Add gap-structure conditioning features.

    Uses the row-level mask (conservative: True only if the grid point
    matched an original observation within tolerance).
    """
    df = df.copy()
    n = len(df)
    obs = observed_row_mask.astype(bool)

    df["is_gap"] = (~obs).astype(np.float32)
    _register_cond("is_gap")

    # Forward: time since last observed point
    time_since = np.zeros(n, dtype=np.float32)
    last = 0
    for i in range(n):
        if obs[i]:
            last = 0
        else:
            last += 1
        time_since[i] = last
    df["time_since_last_obs"] = time_since
    _register_cond("time_since_last_obs")

    # Backward: time to next observed point
    time_to = np.zeros(n, dtype=np.float32)
    nxt = 0
    for i in range(n - 1, -1, -1):
        if obs[i]:
            nxt = 0
        else:
            nxt += 1
        time_to[i] = nxt
    df["time_to_next_obs"] = time_to
    _register_cond("time_to_next_obs")

    return df


def add_per_column_gap_features(df: pd.DataFrame,
                                observed_col_mask: dict,
                                target_cols: list):
    """
    Add per-column gap depth features.

    For each target column, creates a 'gap_depth_{col}' feature
    encoding how deep into a column-specific gap each point is
    (normalised to [0, 1]).  This helps the model distinguish
    "column A is observed but column B is missing" situations.

    Only created when >=2 target columns exist and their missingness
    patterns actually differ.
    """
    if len(target_cols) < 2:
        return df

    df = df.copy()
    n = len(df)
    added = []

    # Check if column masks actually differ
    masks = np.column_stack([observed_col_mask.get(c, np.ones(n, dtype=bool))
                             for c in target_cols])
    if masks.min(axis=1).sum() == masks.max(axis=1).sum():
        # All columns have identical missingness — no need
        return df

    for c in target_cols:
        col_obs = observed_col_mask.get(c, np.ones(n, dtype=bool))
        if col_obs.all():
            continue
        # Compute fractional gap depth for this column
        depth = np.zeros(n, dtype=np.float32)
        run_len = 0
        for i in range(n):
            if col_obs[i]:
                run_len = 0
            else:
                run_len += 1
            depth[i] = run_len
        # Normalise: divide by max gap length for this column
        max_gap = depth.max()
        if max_gap > 0:
            depth = depth / max_gap
        fname = f"gap_depth_{c}"
        df[fname] = depth
        added.append(fname)
        _register_cond(fname)

    if added:
        print(f"[INFO] Added {len(added)} per-column gap depth features")

    return df


# =============================================================================
# Outlier analysis (informational, does not modify data)
# =============================================================================

def analyze_outliers(df: pd.DataFrame, target_cols: list):
    """
    Analyze outlier characteristics of each target column.

    Returns a dict with per-column stats that downstream (train.py)
    can use to decide between hard clip, soft clip, or robust scaling.
    """
    print(f"\n{'='*70}")
    print(f"OUTLIER ANALYSIS")
    print(f"{'='*70}")

    stats = {}
    for c in target_cols:
        vals = df[c].dropna().to_numpy()
        if len(vals) < 10:
            stats[c] = {"n": len(vals), "skip": True}
            continue

        mean = float(np.mean(vals))
        std = float(np.std(vals))
        median = float(np.median(vals))
        q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        iqr = q3 - q1

        if std < 1e-12:
            stats[c] = {"n": len(vals), "constant": True}
            print(f"  {c}: CONSTANT (std ≈ 0)")
            continue

        z = (vals - mean) / std
        n_3sigma = int((np.abs(z) > 3).sum())
        n_5sigma = int((np.abs(z) > 5).sum())
        pct_3sigma = 100 * n_3sigma / len(vals)
        max_z = float(np.abs(z).max())

        # Skewness (simple)
        skew = float(np.mean(((vals - mean) / std) ** 3))
        # Kurtosis excess
        kurt = float(np.mean(((vals - mean) / std) ** 4) - 3)

        stats[c] = {
            "n": len(vals),
            "mean": mean, "std": std,
            "median": median, "iqr": iqr,
            "q1": q1, "q3": q3,
            "n_3sigma": n_3sigma, "pct_3sigma": pct_3sigma,
            "n_5sigma": n_5sigma,
            "max_abs_z": max_z,
            "skewness": skew,
            "kurtosis_excess": kurt,
        }

        # Determine tail behaviour
        if pct_3sigma > 1.0 or max_z > 6:
            flag = "❌ HEAVY TAILS"
            recommendation = "soft_clip"
        elif pct_3sigma > 0.3:
            flag = "⚠  moderate outliers"
            recommendation = "soft_clip"
        else:
            flag = "✓  well-behaved"
            recommendation = "hard_clip"

        stats[c]["recommendation"] = recommendation
        print(f"  {c}: {flag} | >3σ: {pct_3sigma:.2f}% ({n_3sigma}), "
              f"max|z|={max_z:.1f}, skew={skew:.2f}, kurt={kurt:.2f}")

    # Global recommendation
    recs = [s.get("recommendation", "hard_clip")
            for s in stats.values() if not s.get("skip") and not s.get("constant")]
    if any(r == "soft_clip" for r in recs):
        global_rec = "soft_clip"
    else:
        global_rec = "hard_clip"

    print(f"\n  → Global recommendation: {global_rec}")
    return stats, global_rec


# =============================================================================
# Robust scaler stats
# =============================================================================

def compute_scaler_stats(df: pd.DataFrame, target_cols: list,
                         observed_col_mask: dict):
    """
    Compute both standard (mean/std) and robust (median/IQR) stats
    using ONLY observed values.  Saves both so train.py can choose.
    """
    stats = {}
    for c in target_cols:
        obs_mask = observed_col_mask.get(c, np.ones(len(df), dtype=bool))
        vals = df.loc[obs_mask, c].dropna().to_numpy()

        if len(vals) < 2:
            stats[c] = {
                "mean": 0.0, "std": 1.0,
                "median": 0.0, "iqr": 1.0,
                "q1": 0.0, "q3": 1.0,
                "observed_min": 0.0, "observed_max": 0.0,
                "p005": 0.0, "p995": 0.0,
            }
            continue

        q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        iqr = q3 - q1
        if iqr < 1e-8:
            iqr = 1.0

        std = float(np.std(vals))
        if std < 1e-8:
            std = 1.0

        stats[c] = {
            "mean": float(np.mean(vals)),
            "std": std,
            "median": float(np.median(vals)),
            "iqr": iqr,
            "q1": q1, "q3": q3,
            # Value bounds for post-processing clamp
            "observed_min": float(np.min(vals)),
            "observed_max": float(np.max(vals)),
            "p005": float(np.percentile(vals, 0.5)),
            "p995": float(np.percentile(vals, 99.5)),
        }

    return stats


# =============================================================================
# Train / test split
# =============================================================================

def train_test_split_by_time(df: pd.DataFrame, split_ratio=0.8):
    n = len(df)
    cut = int(n * split_ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def make_eval_holdout_mask(T: int, holdout_frac=0.15, avg_block=5, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros(T, dtype=bool)
    target = int(T * holdout_frac)
    attempts = 0
    while mask.sum() < target and attempts < 1000:
        start = int(rng.integers(0, max(1, T - 1)))
        block = max(1, int(rng.normal(avg_block, max(1, avg_block / 3))))
        end = min(start + block, T)
        mask[start:end] = True
        attempts += 1
    return mask


# =============================================================================
# Main entry point
# =============================================================================

def preprocess_csv(
    input_csv: str,
    output_dir: str,
    time_col=None,
    base_dt=None,
    split_ratio=0.8,
    holdout_frac=0.15,
    holdout_block_size=5,
    seed=0,
    add_cond_features=True,
    convert_units=True,
    extract_main_segment=False,
    extract_all_segments_flag=False,
    min_segment_length=50,
    skip_regularize_if_sparse=True,
):
    """
    Preprocess time series CSV for diffusion-based imputation.

    New options vs v1:
        extract_all_segments_flag: keep all segments (not just longest)
        min_segment_length: minimum segment length to keep
    """
    global _COND_COL_REGISTRY
    _COND_COL_REGISTRY = []  # reset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"TIME SERIES PREPROCESSING  v2")
    print(f"{'='*70}")
    print(f"Input:  {input_csv}")
    print(f"Output: {output_dir}")

    # ── Load ─────────────────────────────────────────────────────────
    raw = pd.read_csv(input_csv)

    if time_col is None or time_col not in raw.columns:
        time_col = raw.columns[0]
        print(f"[INFO] Using first column as time: '{time_col}'")

    print(f"[INFO] Raw shape: {raw.shape}")
    print(f"[INFO] Columns: {list(raw.columns)}")

    raw[time_col] = pd.to_numeric(raw[time_col], errors="coerce")
    raw = raw.dropna(subset=[time_col])
    times = raw[time_col].values

    diag = diagnose_time_range(times, verbose=True)

    # ── Segment handling ─────────────────────────────────────────────
    if extract_all_segments_flag and diag["n_large_gaps"] > 0:
        raw = extract_all_segments(raw, time_col,
                                   min_segment_length=min_segment_length)
    elif extract_main_segment and diag["n_large_gaps"] > 0:
        raw = extract_longest_segment(raw, time_col)

    # ── Unit conversions ─────────────────────────────────────────────
    if convert_units:
        print(f"\n{'='*70}")
        print(f"UNIT CONVERSIONS")
        print(f"{'='*70}")

        if "ram_limit" in raw.columns:
            print(f"[INFO] ram_limit → MB")
            raw["ram_limit_mb"] = raw["ram_limit"].apply(ram_to_mb)
            raw = raw.drop(columns=["ram_limit"])

        if "ram_usage" in raw.columns:
            print(f"[INFO] ram_usage (bytes) → MB")
            raw["ram_usage_mb"] = raw["ram_usage"].apply(bytes_to_mb)
            raw = raw.drop(columns=["ram_usage"])

        # Safer latency detection: require 'lat' prefix followed by
        # underscore or end-of-name, OR exact known names.
        latency_exact = {"mean", "Min", "min"}
        latency_cols = [
            c for c in raw.columns
            if (c.startswith("lat") and (len(c) == 3 or c[3] in "_0123456789"))
            or c in latency_exact
        ]
        if latency_cols:
            print(f"[INFO] μs → ms: {latency_cols}")
            for c in latency_cols:
                new_name = f"{c}_ms" if not c.endswith("_ms") else c
                raw[new_name] = raw[c].apply(microseconds_to_ms)
                if new_name != c:
                    raw = raw.drop(columns=[c])

        if "cpu_usage" in raw.columns:
            sample = raw["cpu_usage"].dropna().head(100)
            if len(sample) > 0 and sample.max() > 1.5:
                print(f"[INFO] cpu_usage % → fraction")
                raw["cpu_usage"] = raw["cpu_usage"] / 100.0

    # Numeric coercion
    for c in raw.columns:
        if c != time_col:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # ── Regularize ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"REGULARIZATION")
    print(f"{'='*70}")

    df_reg, observed_row_mask, observed_col_mask, base_dt = regularize(
        raw, time_col=time_col, base_dt=base_dt,
        skip_if_sparse=skip_regularize_if_sparse,
    )

    # ── Identify target / cond columns BEFORE adding features ────────
    feature_cols_before = set(df_reg.columns) - {time_col}

    # ── Feature engineering ──────────────────────────────────────────
    if add_cond_features:
        print(f"\n{'='*70}")
        print(f"FEATURE ENGINEERING")
        print(f"{'='*70}")
        df_reg = add_time_features(df_reg, time_col=time_col)
        df_reg = add_gap_structure_features(df_reg, observed_row_mask)

    # target_cols = original data columns (before we added features)
    target_cols = sorted(feature_cols_before)
    # cond_cols = everything we added (auto-registered)
    cond_cols = [c for c in _COND_COL_REGISTRY if c in df_reg.columns]

    # Optional per-column gap features
    if add_cond_features and len(target_cols) >= 2:
        df_reg = add_per_column_gap_features(
            df_reg, observed_col_mask, target_cols
        )
        # Re-read registry (may have grown)
        cond_cols = [c for c in _COND_COL_REGISTRY if c in df_reg.columns]

    print(f"[INFO] Target columns ({len(target_cols)}): {target_cols}")
    print(f"[INFO] Cond columns   ({len(cond_cols)}):   {cond_cols}")

    # ── Outlier analysis ─────────────────────────────────────────────
    outlier_stats, clip_recommendation = analyze_outliers(df_reg, target_cols)

    # ── Scaler stats (observed-only) ─────────────────────────────────
    scaler_stats = compute_scaler_stats(df_reg, target_cols, observed_col_mask)

    # ── Train / test split ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"TRAIN / TEST SPLIT")
    print(f"{'='*70}")

    train_df, test_df = train_test_split_by_time(df_reg, split_ratio)
    print(f"[INFO] Train: {len(train_df)} ({100*split_ratio:.0f}%)")
    print(f"[INFO] Test:  {len(test_df)} ({100*(1-split_ratio):.0f}%)")

    # Evaluation holdout
    T_test = len(test_df)
    holdout_1d = make_eval_holdout_mask(
        T_test, holdout_frac, holdout_block_size, seed
    )
    test_input = test_df.copy()
    hidden_count = 0
    for c in target_cols:
        observed = test_input[c].notna().to_numpy()
        hide = holdout_1d & observed
        test_input.loc[test_input.index[hide], c] = np.nan
        hidden_count += hide.sum()
    print(f"[INFO] Holdout: {holdout_1d.sum()} rows, {hidden_count} values hidden")

    # ── Save ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SAVING")
    print(f"{'='*70}")

    train_df.to_csv(out / "train.csv", index=False)
    test_df.to_csv(out / "test_gt.csv", index=False)
    test_input.to_csv(out / "test_input.csv", index=False)
    np.save(out / "eval_holdout_mask.npy", holdout_1d.astype(np.bool_))

    # Save per-column observed masks (for training)
    mask_dir = out / "col_masks"
    mask_dir.mkdir(exist_ok=True)
    for c, m in observed_col_mask.items():
        np.save(mask_dir / f"{c}.npy", m.astype(np.bool_))

    # Save scaler stats
    scaler_dir = out / "scaler"
    scaler_dir.mkdir(exist_ok=True)
    with open(scaler_dir / "stats.json", "w") as f:
        json.dump(scaler_stats, f, indent=2)

    # Arrays for quick loading in train.py
    mean_arr = np.array([scaler_stats[c]["mean"] for c in target_cols],
                        dtype=np.float32)
    std_arr = np.array([scaler_stats[c]["std"] for c in target_cols],
                       dtype=np.float32)
    median_arr = np.array([scaler_stats[c]["median"] for c in target_cols],
                          dtype=np.float32)
    iqr_arr = np.array([scaler_stats[c]["iqr"] for c in target_cols],
                       dtype=np.float32)
    np.save(scaler_dir / "mean.npy", mean_arr)
    np.save(scaler_dir / "std.npy", std_arr)
    np.save(scaler_dir / "median.npy", median_arr)
    np.save(scaler_dir / "iqr.npy", iqr_arr)

    # Per-column value bounds for inference-time clamping
    lower_arr = np.array([scaler_stats[c]["observed_min"] for c in target_cols],
                         dtype=np.float32)
    upper_arr = np.array([scaler_stats[c]["p995"] for c in target_cols],
                         dtype=np.float32)
    p005_arr = np.array([scaler_stats[c]["p005"] for c in target_cols],
                        dtype=np.float32)
    obs_max_arr = np.array([scaler_stats[c]["observed_max"] for c in target_cols],
                           dtype=np.float32)
    np.save(scaler_dir / "lower_bound.npy", lower_arr)
    np.save(scaler_dir / "upper_bound_p995.npy", upper_arr)
    np.save(scaler_dir / "lower_bound_p005.npy", p005_arr)
    np.save(scaler_dir / "observed_max.npy", obs_max_arr)

    # Metadata
    meta = {
        "time_col": time_col,
        "base_dt": float(base_dt),
        "target_cols": target_cols,
        "cond_cols": cond_cols,
        "all_model_cols": target_cols + cond_cols,
        "units_converted": convert_units,
        "split_ratio": split_ratio,
        "holdout_frac": holdout_frac,
        "original_rows": int(diag["n_points"]),
        "regularized_rows": len(df_reg),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "clip_recommendation": clip_recommendation,
        "preprocessing_version": 2,
        "notes": "v2: col-level masks, robust stats, outlier analysis",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # Save outlier report
    (out / "outlier_report.json").write_text(
        json.dumps(outlier_stats, indent=2, default=str)
    )

    print(f"[INFO] Saved to: {out}")

    # ── Gap statistics ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"GAP STATISTICS (train set)")
    print(f"{'='*70}")
    for col in target_cols[:15]:
        gaps = train_df[col].isna().sum()
        gap_pct = 100 * gaps / len(train_df)
        icon = "✓" if gap_pct < 30 else "⚠" if gap_pct < 70 else "❌"
        rec = outlier_stats.get(col, {}).get("recommendation", "?")
        print(f"  {icon} {col}: {gap_pct:.1f}% gaps | scale: {rec}")
    if len(target_cols) > 15:
        print(f"  … ({len(target_cols) - 15} more)")

    print(f"\n{'='*70}")
    print(f"PREPROCESSING COMPLETE  (v2)")
    print(f"{'='*70}\n")

    return meta
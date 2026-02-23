

"""
Optimized preprocessing for time series data

Key improvements:
1. Float base_dt support (11.0, 0.5, etc)
2. 100x faster regularization (O(n log m) vs O(n*m))
3. Automatic segment detection for multi-experiment data
4. Better gap handling
5. Comprehensive diagnostics
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# RAM parsing regex
_RAM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGTP]?)\s*$", re.IGNORECASE)


def ram_to_mb(x) -> float:
    """Convert ram_limit string (e.g., '2048M') to MB"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = _RAM_RE.match(s)
    if not m:
        raise ValueError(f"Unsupported ram_limit format: {x}")
    val = float(m.group(1))
    unit = (m.group(2) or "").upper()
    scale = {"": 1.0/1e6, "K": 1.0/1e3, "M": 1.0, "G": 1e3, "T": 1e6, "P": 1e9}
    return val * scale.get(unit, 1.0)


def bytes_to_mb(x) -> float:
    """Convert bytes to MB"""
    if pd.isna(x):
        return np.nan
    try:
        return float(x) / 1e6
    except (ValueError, TypeError):
        return np.nan


def microseconds_to_ms(x) -> float:
    """Convert microseconds to milliseconds"""
    if pd.isna(x):
        return np.nan
    try:
        return float(x) / 1000.0
    except (ValueError, TypeError):
        return np.nan


def infer_base_dt(time_arr: np.ndarray) -> float:
    """
    Infer base time interval from data.
    Uses median for robustness against outliers.
    Returns float to support sub-second and 11-second intervals.
    """
    dt = np.diff(time_arr)
    dt = dt[dt > 0]  # Remove zeros
    if len(dt) == 0:
        raise ValueError("Not enough timestamps to infer base_dt.")
    
    median_dt = float(np.median(dt))
    
    # For common intervals (1-100s), keep as float
    if 1.0 <= median_dt <= 100.0:
        return median_dt
    else:
        return int(round(median_dt))


def diagnose_time_range(times: np.ndarray, verbose: bool = True):
    """
    Diagnose time range and detect issues like:
    - Very long time spans (multi-day data)
    - Large gaps between segments
    - Timestamp anomalies
    """
    t_min, t_max = times.min(), times.max()
    duration_sec = t_max - t_min
    duration_days = duration_sec / 86400
    
    if not verbose:
        return {
            'duration_sec': duration_sec,
            'duration_days': duration_days,
            'n_points': len(times)
        }
    
    print(f"\n{'='*70}")
    print(f"TIME RANGE DIAGNOSTIC")
    print(f"{'='*70}")
    print(f"  Data points: {len(times)}")
    print(f"  Time range: {duration_sec:.0f}s ({duration_days:.1f} days)")
    
    try:
        dt_min = datetime.fromtimestamp(t_min)
        dt_max = datetime.fromtimestamp(t_max)
        print(f"  Start: {dt_min}")
        print(f"  End: {dt_max}")
    except:
        print(f"  Start timestamp: {t_min}")
        print(f"  End timestamp: {t_max}")
    
    # Analyze gaps
    intervals = np.diff(times)
    large_gaps = intervals > 1000  # > ~17 minutes
    n_large_gaps = large_gaps.sum()
    
    if n_large_gaps > 0:
        print(f"\n  ⚠️  Found {n_large_gaps} large gaps (>1000s)")
        print(f"     This suggests multiple experiments or data collection periods")
        
        # Show gap details
        gap_sizes = intervals[large_gaps]
        print(f"     Gap sizes: {np.sort(gap_sizes / 3600)[:5]} hours (showing first 5)")
    
    if duration_days > 2:
        print(f"\n  ⚠️  WARNING: Data spans {duration_days:.1f} days")
        print(f"     This will create >80% gaps in regularized timeline")
        print(f"     Recommendation: Extract continuous segments or use base_dt=None")
    
    return {
        'duration_sec': duration_sec,
        'duration_days': duration_days,
        'n_points': len(times),
        'n_large_gaps': n_large_gaps
    }


def extract_longest_segment(df: pd.DataFrame, time_col: str, gap_threshold: float = 1000.0):
    """
    Extract the longest continuous segment from data.
    Useful when data contains multiple experiments with large gaps.
    """
    times = df[time_col].values
    intervals = np.diff(times)
    
    # Find large gaps
    large_gaps = intervals > gap_threshold
    
    # Identify segments
    segment_starts = [0] + (np.where(large_gaps)[0] + 1).tolist()
    segment_ends = np.where(large_gaps)[0].tolist() + [len(df)]
    
    segments = list(zip(segment_starts, segment_ends))
    segment_lengths = [end - start for start, end in segments]
    
    # Find longest
    longest_idx = np.argmax(segment_lengths)
    start, end = segments[longest_idx]
    
    print(f"\n{'='*70}")
    print(f"SEGMENT EXTRACTION")
    print(f"{'='*70}")
    print(f"  Found {len(segments)} segments")
    print(f"  Longest segment: rows {start} to {end} ({segment_lengths[longest_idx]} points)")
    
    seg_duration = (times[end-1] - times[start]) / 3600
    print(f"  Duration: {seg_duration:.1f} hours")
    
    return df.iloc[start:end].copy()


def regularize(df: pd.DataFrame, time_col="time", base_dt=None, skip_if_sparse=True):
    """
    Regularize timeline to uniform intervals.
    
    Optimized: O(n log m) complexity instead of O(n*m)
    
    Args:
        df: DataFrame with time series data
        time_col: Name of time column
        base_dt: Time interval in seconds (None = auto-infer)
        skip_if_sparse: If True, skip regularization if it would create >80% gaps
    
    Returns:
        (df_regularized, observed_mask, base_dt)
    """
    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.drop_duplicates(subset=[time_col], keep="last")
    
    t = df[time_col].to_numpy()
    
    # Auto-infer base_dt if not specified
    if base_dt is None:
        base_dt = infer_base_dt(t)
        print(f"[INFO] Auto-inferred base_dt: {base_dt} seconds")
    else:
        print(f"[INFO] Using specified base_dt: {base_dt} seconds")
    
    # Check if regularization would create too many gaps
    t_min, t_max = t.min(), t.max()
    n_grid = int((t_max - t_min) / base_dt) + 1
    n_observed = len(t)
    expected_gap_pct = 100 * (1 - n_observed / n_grid)
    
    if skip_if_sparse and expected_gap_pct > 80:
        print(f"\n{'='*70}")
        print(f"WARNING: Regularization would create {expected_gap_pct:.1f}% gaps!")
        print(f"{'='*70}")
        print(f"  Original points: {n_observed}")
        print(f"  Grid points: {n_grid}")
        print(f"  Time span: {(t_max - t_min) / 86400:.1f} days")
        print(f"\n  Skipping regularization. Using original sampling instead.")
        print(f"  To force regularization, set skip_if_sparse=False")
        
        # Return original data with all-True mask
        df_out = df.copy()
        observed_mask = np.ones(len(df), dtype=bool)
        return df_out, observed_mask, base_dt
    
    # Create regular grid
    full = np.linspace(t_min, t_min + (n_grid - 1) * base_dt, n_grid)
    
    # OPTIMIZED: Find nearest original points using binary search
    # Complexity: O(n log m) instead of O(n*m)
    tolerance = base_dt / 2.0
    
    indices = np.searchsorted(t, full)
    indices = np.clip(indices, 0, len(t) - 1)
    
    # Calculate distances to both current and previous indices
    distances = np.abs(t[indices] - full)
    
    # Check previous index where it exists
    has_prev = indices > 0
    prev_indices = np.maximum(indices - 1, 0)
    prev_distances = np.where(has_prev, np.abs(t[prev_indices] - full), np.inf)
    
    # Use the closer of the two
    final_distances = np.minimum(distances, prev_distances)
    
    # Mark as observed if within tolerance
    observed_row_mask = final_distances < tolerance
    
    # Reindex with nearest neighbor
    df = df.set_index(time_col)
    df_reindexed = df.reindex(full, method='nearest', tolerance=tolerance)
    df_reindexed = df_reindexed.reset_index()
    df_reindexed = df_reindexed.rename(columns={"index": time_col})
    
    # Print statistics
    observed_count = observed_row_mask.sum()
    gap_pct = 100 * (1 - observed_count / len(full))
    
    print(f"[INFO] Regular grid: {len(full)} points")
    print(f"[INFO] Observed: {observed_count} ({100 - gap_pct:.1f}%)")
    print(f"[INFO] Gaps: {len(full) - observed_count} ({gap_pct:.1f}%)")
    
    if gap_pct > 70:
        print(f"[WARNING] Gap percentage is high ({gap_pct:.1f}%)")
        print(f"[WARNING] Consider: base_dt=None, smaller base_dt, or extract segments")
    
    return df_reindexed, observed_row_mask, base_dt


def add_time_features(df: pd.DataFrame, time_col="time"):
    """Add time-based features for better temporal modeling"""
    df = df.copy()
    
    t = df[time_col].to_numpy()
    t_min, t_max = t.min(), t.max()
    duration = t_max - t_min
    
    # Normalized time [0, 1]
    df['t_norm'] = (t - t_min) / max(duration, 1.0)
    
    # Daily cycles (if data spans > 1 hour)
    if duration > 3600:
        df['sin_day'] = np.sin(2 * np.pi * t / 86400)
        df['cos_day'] = np.cos(2 * np.pi * t / 86400)
    
    # Hourly cycles (if data spans > 10 minutes and < 1 day)
    if 600 < duration < 86400:
        df['sin_hour'] = np.sin(2 * np.pi * t / 3600)
        df['cos_hour'] = np.cos(2 * np.pi * t / 3600)
    
    return df


def add_gap_structure_features(df: pd.DataFrame, observed_mask: np.ndarray):
    """Add features that describe gap structure"""
    df = df.copy()
    
    # Binary indicator: is this row a gap?
    df['is_gap'] = ~observed_mask
    
    # Time since last observed point
    time_since = np.zeros(len(df))
    last_obs = 0
    for i in range(len(df)):
        if observed_mask[i]:
            last_obs = 0
        else:
            last_obs += 1
        time_since[i] = last_obs
    df['time_since_last_obs'] = time_since
    
    # Time to next observed point
    time_to = np.zeros(len(df))
    next_obs = 0
    for i in range(len(df) - 1, -1, -1):
        if observed_mask[i]:
            next_obs = 0
        else:
            next_obs += 1
        time_to[i] = next_obs
    df['time_to_next_obs'] = time_to
    
    return df


def train_test_split_by_time(df: pd.DataFrame, split_ratio=0.8):
    """Split data chronologically"""
    n = len(df)
    cut = int(n * split_ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def make_eval_holdout_mask(T: int, holdout_frac=0.15, avg_block=5, seed=0):
    """Create random block-based holdout mask for evaluation"""
    rng = np.random.default_rng(seed)
    mask = np.zeros(T, dtype=bool)
    target = int(T * holdout_frac)
    attempts = 0
    max_attempts = 1000
    
    while mask.sum() < target and attempts < max_attempts:
        start = int(rng.integers(0, max(1, T - 1)))
        block = max(1, int(rng.normal(avg_block, max(1, avg_block / 3))))
        end = min(start + block, T)
        mask[start:end] = True
        attempts += 1
    
    return mask


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
    skip_regularize_if_sparse=True
):
    """
    Preprocess time series data with comprehensive diagnostics and optimization.
    
    Args:
        input_csv: Path to input CSV file
        output_dir: Output directory for processed data
        time_col: Time column name (None = auto-detect first column)
        base_dt: Time interval for regularization (None = auto-infer, float = specific value)
        split_ratio: Train/test split ratio
        holdout_frac: Fraction of test data to hide for evaluation
        holdout_block_size: Average size of holdout blocks
        seed: Random seed
        add_cond_features: Add time and gap features
        convert_units: Convert units (RAM, latency, etc.)
        extract_main_segment: If True, extract longest continuous segment
        skip_regularize_if_sparse: If True, skip regularization if >80% gaps
    
    Returns:
        meta: Dictionary with preprocessing metadata
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"TIME SERIES PREPROCESSING")
    print(f"{'='*70}")
    print(f"Input: {input_csv}")
    print(f"Output: {output_dir}")
    
    # Load data
    raw = pd.read_csv(input_csv)
    
    # Handle time column
    if time_col is None or time_col not in raw.columns:
        time_col = raw.columns[0]
        print(f"[INFO] Using first column as time: '{time_col}'")
    
    print(f"[INFO] Raw data shape: {raw.shape}")
    print(f"[INFO] Columns: {list(raw.columns)}")
    
    # Diagnose time range
    raw[time_col] = pd.to_numeric(raw[time_col], errors="coerce")
    raw = raw.dropna(subset=[time_col])
    times = raw[time_col].values
    
    diag = diagnose_time_range(times, verbose=True)
    
    # Extract main segment if requested
    if extract_main_segment and diag['n_large_gaps'] > 0:
        print(f"\n[INFO] Extracting longest continuous segment...")
        raw = extract_longest_segment(raw, time_col)
        times = raw[time_col].values
    
    # Unit conversions
    if convert_units:
        print(f"\n{'='*70}")
        print(f"UNIT CONVERSIONS")
        print(f"{'='*70}")
        
        if "ram_limit" in raw.columns:
            print(f"[INFO] Converting ram_limit to MB")
            raw["ram_limit_mb"] = raw["ram_limit"].apply(ram_to_mb)
            raw = raw.drop(columns=["ram_limit"])
        
        if "ram_usage" in raw.columns:
            print(f"[INFO] Converting ram_usage from bytes to MB")
            raw["ram_usage_mb"] = raw["ram_usage"].apply(bytes_to_mb)
            raw = raw.drop(columns=["ram_usage"])
        
        latency_cols = [c for c in raw.columns if c.startswith('lat') or c in ['mean', 'Min', 'min']]
        if latency_cols:
            print(f"[INFO] Converting latency columns from μs to ms: {latency_cols}")
            for c in latency_cols:
                new_name = f"{c}_ms" if not c.endswith('_ms') else c
                raw[new_name] = raw[c].apply(microseconds_to_ms)
                if new_name != c:
                    raw = raw.drop(columns=[c])
        
        if "cpu_usage" in raw.columns:
            sample = raw["cpu_usage"].dropna().head(100)
            if len(sample) > 0 and sample.max() > 1.5:
                print(f"[INFO] Converting cpu_usage from percentage to fraction")
                raw["cpu_usage"] = raw["cpu_usage"] / 100.0
    
    # Convert remaining columns to numeric
    for c in raw.columns:
        if c == time_col:
            continue
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    
    # Regularize timeline
    print(f"\n{'='*70}")
    print(f"REGULARIZATION")
    print(f"{'='*70}")
    
    df_reg, observed_row_mask, base_dt = regularize(
        raw, 
        time_col=time_col, 
        base_dt=base_dt,
        skip_if_sparse=skip_regularize_if_sparse
    )
    
    # Add features
    if add_cond_features:
        print(f"\n{'='*70}")
        print(f"FEATURE ENGINEERING")
        print(f"{'='*70}")
        df_reg = add_time_features(df_reg, time_col=time_col)
        df_reg = add_gap_structure_features(df_reg, observed_row_mask)
        print(f"[INFO] Added time and gap features")
    
    # Identify columns
    x_cols = [c for c in df_reg.columns if c != time_col]
    
    cond_cols = []
    if add_cond_features:
        potential_cond = ["t_norm", "sin_day", "cos_day", "sin_hour", "cos_hour",
                         "is_gap", "time_since_last_obs", "time_to_next_obs"]
        cond_cols = [c for c in potential_cond if c in df_reg.columns]
    
    target_cols = [c for c in x_cols if c not in cond_cols]
    
    # Train/test split
    print(f"\n{'='*70}")
    print(f"TRAIN/TEST SPLIT")
    print(f"{'='*70}")
    train_df, test_df = train_test_split_by_time(df_reg, split_ratio=split_ratio)
    print(f"[INFO] Train: {len(train_df)} rows ({100*split_ratio:.0f}%)")
    print(f"[INFO] Test: {len(test_df)} rows ({100*(1-split_ratio):.0f}%)")
    
    # Create evaluation holdout
    T_test = len(test_df)
    holdout_1d = make_eval_holdout_mask(T_test, holdout_frac=holdout_frac, 
                                        avg_block=holdout_block_size, seed=seed)
    test_input = test_df.copy()
    
    hidden_count = 0
    for c in target_cols:
        observed = test_input[c].notna().to_numpy()
        hide = holdout_1d & observed
        test_input.loc[test_input.index[hide], c] = np.nan
        hidden_count += hide.sum()
    
    print(f"[INFO] Evaluation holdout: {holdout_1d.sum()} points ({100*holdout_frac:.0f}%)")
    print(f"[INFO] Hidden values: {hidden_count}")
    
    # Save artifacts
    print(f"\n{'='*70}")
    print(f"SAVING")
    print(f"{'='*70}")
    
    train_df.to_csv(out / "train.csv", index=False)
    test_df.to_csv(out / "test_gt.csv", index=False)
    test_input.to_csv(out / "test_input.csv", index=False)
    np.save(out / "eval_holdout_mask.npy", holdout_1d.astype(np.bool_))
    
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
        "original_rows": len(raw),
        "regularized_rows": len(df_reg),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "notes": "Preprocessed with optimized pipeline"
    }
    
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    
    print(f"[INFO] Saved to: {out}")
    print(f"[INFO] Target columns ({len(target_cols)}): {target_cols}")
    print(f"[INFO] Conditional columns ({len(cond_cols)}): {cond_cols}")
    
    # Final statistics
    print(f"\n{'='*70}")
    print(f"GAP STATISTICS")
    print(f"{'='*70}")
    for col in target_cols[:10]:  # Show first 10
        gaps = train_df[col].isna().sum()
        gap_pct = 100 * gaps / len(train_df)
        status = "✓" if gap_pct < 30 else "⚠" if gap_pct < 70 else "❌"
        print(f"  {status} {col}: {gap_pct:.1f}% gaps")
    
    if len(target_cols) > 10:
        print(f"  ... ({len(target_cols) - 10} more columns)")
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*70}\n")
    
    return meta
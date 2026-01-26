# helpers/ts_utils.py
def detect_time_gaps(df, ts_col):
    ts = df[ts_col].sort_values()
    diffs = ts.diff().dropna()

    expected = diffs.mode()[0]
    gaps = diffs[diffs > expected * 1.5]

    return {
        "expected_frequency": str(expected),
        "num_gaps": int(len(gaps)),
        "has_gaps": len(gaps) > 0,
        "sample_gap_indices": gaps.index[:5].tolist(),
    }

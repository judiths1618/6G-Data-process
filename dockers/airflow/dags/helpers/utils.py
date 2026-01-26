import pandas as pd
import numpy as np
def analyze_csv_time_series_df(df, configured_name=None, sample_ratio=0.9):
    """
    Analyze whether a dataframe represents time-series data.

    Parameters
    ----------
    df : pd.DataFrame
    configured_name : optional str
        Preferred timestamp column name (if provided)
    sample_ratio : float
        Ratio of parsable datetime values to qualify as timestamp

    Returns
    -------
    dict with keys:
      - is_time_series: bool
      - timestamp_column: str | None
      - detected_type: str
    """
    import pandas as pd

    result = {
        "is_time_series": False,
        "timestamp_column": None,
        "detected_type": "Not Time Series",
    }

    # 0️. 优先使用配置的列
    if configured_name and configured_name in df.columns:
        result.update(
            {
                "is_time_series": True,
                "timestamp_column": configured_name,
                "detected_type": "Configured Timestamp Column",
            }
        )
        return result

    potential_step_cols = []

    for col in df.columns:
        series = df[col]

        # A️⃣ Datetime string
        if series.dtype == "object":
            try:
                parsed = pd.to_datetime(series, errors="coerce")
                if parsed.notna().mean() >= sample_ratio:
                    result.update(
                        {
                            "is_time_series": True,
                            "timestamp_column": col,
                            "detected_type": "Datetime String",
                        }
                    )
                    return result
            except Exception:
                pass

        # B️. Unix timestamp
        if pd.api.types.is_numeric_dtype(series):
            mean_val = series.dropna().mean()
            if 1e9 < mean_val < 3e9:
                result.update(
                    {
                        "is_time_series": True,
                        "timestamp_column": col,
                        "detected_type": "Unix Timestamp",
                    }
                )
                return result

        # C️. Step / simulation index (weak signal)
        if (
            pd.api.types.is_numeric_dtype(series)
            and series.is_monotonic_increasing
            and series.nunique() > len(series) * 0.9
        ):
            potential_step_cols.append(col)

    if potential_step_cols:
        result.update(
            {
                "is_time_series": True,
                "timestamp_column": potential_step_cols[0],
                "detected_type": "Step Index",
            }
        )
        return result

    return result


def detect_primary_key(
    df: pd.DataFrame,
    max_combo: int = 2,
    uniqueness_threshold: float = 0.999
):
    """
    Heuristic primary key detection.
    """
    n = len(df)
    results = []

    # 1. Single-column candidates
    for col in df.columns:
        null_ratio = df[col].isna().mean()
        uniq_ratio = df[col].nunique(dropna=True) / max(n, 1)

        if uniq_ratio >= uniqueness_threshold and null_ratio < 0.01:
            results.append({
                "columns": [col],
                "uniqueness_ratio": round(uniq_ratio, 4),
                "null_ratio": round(null_ratio, 4)
            })

    # 2. Two-column composite keys (optional)
    if not results:
        for i, c1 in enumerate(df.columns):
            for c2 in df.columns[i+1:]:
                combo = df[[c1, c2]].dropna()
                uniq_ratio = len(combo.drop_duplicates()) / max(len(combo), 1)

                if uniq_ratio >= uniqueness_threshold:
                    results.append({
                        "columns": [c1, c2],
                        "uniqueness_ratio": round(uniq_ratio, 4),
                        "null_ratio": round(
                            max(df[c1].isna().mean(), df[c2].isna().mean()), 4
                        )
                    })

    if results:
        best = results[0]
        return {
            "type": "composite" if len(best["columns"]) > 1 else "single",
            "columns": best["columns"],
            "uniqueness_ratio": best["uniqueness_ratio"],
            "null_ratio": best["null_ratio"],
            "is_hard_pk": best["uniqueness_ratio"] > 0.999
        }

    return {
        "type": "none",
        "columns": [],
        "uniqueness_ratio": 0.0,
        "null_ratio": None,
        "is_hard_pk": False
    }

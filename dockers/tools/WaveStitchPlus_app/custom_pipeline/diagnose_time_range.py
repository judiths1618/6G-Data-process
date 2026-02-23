#!/usr/bin/env python3
"""
Diagnose extreme gap percentage in AMF data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def diagnose_time_range(csv_path):
    """Analyze time range and identify issues"""
    
    print("="*70)
    print("TIME RANGE DIAGNOSTIC")
    print("="*70)
    
    # Load data
    df = pd.read_csv(csv_path)
    time_col = df.columns[0]
    times = df[time_col].values
    
    print(f"\n[1] Basic Statistics")
    print(f"  Total rows: {len(times)}")
    print(f"  Time column: '{time_col}'")
    
    # Time range
    t_min, t_max = times.min(), times.max()
    duration_sec = t_max - t_min
    duration_days = duration_sec / 86400
    
    print(f"\n[2] Time Range")
    print(f"  Min timestamp: {t_min}")
    print(f"  Max timestamp: {t_max}")
    print(f"  Duration: {duration_sec:.0f} seconds = {duration_days:.1f} days")
    
    # Convert to human-readable dates
    try:
        dt_min = datetime.fromtimestamp(t_min)
        dt_max = datetime.fromtimestamp(t_max)
        print(f"  Min date: {dt_min}")
        print(f"  Max date: {dt_max}")
    except:
        print(f"  (Could not convert to dates)")
    
    # Analyze gaps
    print(f"\n[3] Gap Analysis")
    intervals = np.diff(times)
    
    print(f"  Intervals:")
    print(f"    Min: {intervals.min():.1f}s")
    print(f"    Median: {np.median(intervals):.1f}s")
    print(f"    Max: {intervals.max():.1f}s")
    print(f"    Mean: {intervals.mean():.1f}s")
    
    # Find large gaps
    large_gaps = intervals[intervals > 1000]  # > 15 minutes
    print(f"\n  Large gaps (>1000s):")
    print(f"    Count: {len(large_gaps)}")
    if len(large_gaps) > 0:
        print(f"    Sizes: {np.sort(large_gaps)}")
        
        # Find where they are
        gap_indices = np.where(intervals > 1000)[0]
        print(f"\n  Locations of large gaps:")
        for idx in gap_indices[:10]:  # Show first 10
            t1 = times[idx]
            t2 = times[idx + 1]
            gap_sec = t2 - t1
            gap_hours = gap_sec / 3600
            try:
                dt1 = datetime.fromtimestamp(t1)
                dt2 = datetime.fromtimestamp(t2)
                print(f"    Gap {idx+1}: {dt1} → {dt2} ({gap_hours:.1f} hours)")
            except:
                print(f"    Gap {idx+1}: {t1} → {t2} ({gap_hours:.1f} hours)")
    
    # Check for outliers
    print(f"\n[4] Outlier Detection")
    q1, q3 = np.percentile(times, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    outliers_low = times[times < lower_bound]
    outliers_high = times[times > upper_bound]
    
    if len(outliers_low) > 0:
        print(f"  ⚠️  {len(outliers_low)} outliers below normal range")
        print(f"     Values: {outliers_low[:5]}")
    
    if len(outliers_high) > 0:
        print(f"  ⚠️  {len(outliers_high)} outliers above normal range")
        print(f"     Values: {outliers_high[:5]}")
    
    if len(outliers_low) == 0 and len(outliers_high) == 0:
        print(f"  ✓ No extreme outliers detected")
    
    # Recommendations
    print(f"\n[5] Recommendations")
    
    expected_points = duration_sec / 11  # With 11s interval
    actual_points = len(times)
    coverage = actual_points / expected_points
    
    print(f"  Expected points (11s interval): {expected_points:.0f}")
    print(f"  Actual points: {actual_points}")
    print(f"  Coverage: {coverage*100:.1f}%")
    
    if duration_days > 2:
        print(f"\n  ⚠️  WARNING: Data spans {duration_days:.1f} days!")
        print(f"     This seems unusual for performance data.")
        print(f"\n  Possible issues:")
        print(f"     1. Multiple experiments concatenated together")
        print(f"     2. Timestamp errors or corruption")
        print(f"     3. Data collected over a long period with gaps")
        
        print(f"\n  Solutions:")
        print(f"     A. Split data by experiment (segment on config changes)")
        print(f"     B. Remove outlier timestamps")
        print(f"     C. Filter to continuous segments only")
    
    if coverage < 0.3:
        print(f"\n  ⚠️  WARNING: Only {coverage*100:.1f}% coverage!")
        print(f"     Most of the time range has no data.")
        
        print(f"\n  Solutions:")
        print(f"     A. Don't regularize - use original sampling")
        print(f"        base_dt=None in config")
        print(f"     B. Split into continuous segments")
        print(f"     C. Remove data outside main collection period")
    
    # Suggest filtering
    if len(large_gaps) > 0:
        print(f"\n[6] Suggested Data Filtering")
        print(f"\n  Option 1: Keep only the main continuous segment")
        
        # Find longest continuous segment
        gaps_large = intervals > 1000
        segment_starts = [0] + (np.where(gaps_large)[0] + 1).tolist()
        segment_ends = np.where(gaps_large)[0].tolist() + [len(times)]
        
        segments = list(zip(segment_starts, segment_ends))
        segment_lengths = [end - start for start, end in segments]
        
        longest_idx = np.argmax(segment_lengths)
        longest_start, longest_end = segments[longest_idx]
        
        print(f"     Longest segment: rows {longest_start} to {longest_end}")
        print(f"     Length: {segment_lengths[longest_idx]} points")
        
        seg_duration = times[longest_end] - times[longest_start]
        print(f"     Duration: {seg_duration/3600:.1f} hours")
        
        print(f"\n     To extract:")
        print(f"     df_filtered = df.iloc[{longest_start}:{longest_end}]")
        print(f"     df_filtered.to_csv('amf-filtered.csv', index=False)")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnose_time_range.py <csv_file>")
        print("\nOr import and use:")
        print("  from diagnose_time_range import diagnose_time_range")
        print("  diagnose_time_range('amf-performance.csv')")
    else:
        diagnose_time_range(sys.argv[1])
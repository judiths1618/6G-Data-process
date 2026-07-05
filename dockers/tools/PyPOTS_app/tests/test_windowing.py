"""Windowed-inference helpers: tiling a long series into fixed-length windows
and stitching them back keeps PyPOTS attention at O(window**2) instead of
O(series_len**2) (which OOMs). Pure NumPy — no pypots import needed."""
from __future__ import annotations

import numpy as np

from pypots_run_imputation import _stitch, _to_windows, _window_starts


def test_window_starts_align_last_to_end():
    assert _window_starts(1000, 100, 100)[-1] == 900        # tail covered
    assert _window_starts(105, 100, 100) == [0, 5]          # non-divisible → shifted last
    assert _window_starts(50, 100, 100) == [0]              # short series → single window
    ov = _window_starts(250, 100, 80)
    assert ov[0] == 0 and ov[-1] == 150                     # overlapping, ends at series end


def test_to_windows_shapes_and_dtype():
    arr = np.arange(1000 * 3, dtype=float).reshape(1000, 3)
    starts = _window_starts(1000, 100, 100)
    wins = _to_windows(arr, starts, 100)
    assert wins.shape == (10, 100, 3) and wins.dtype == np.float32


def test_short_series_padded_with_nan():
    arr = np.arange(50 * 4, dtype=float).reshape(50, 4)
    starts = _window_starts(50, 100, 100)
    wins = _to_windows(arr, starts, 100)
    assert wins.shape == (1, 100, 4)
    assert np.isnan(wins[0, 50:]).all()                     # pad = NaN (treated as missing)


def test_roundtrip_nonoverlapping():
    arr = np.arange(1000 * 3, dtype=float).reshape(1000, 3)
    starts = _window_starts(1000, 100, 100)
    rec = _stitch(_to_windows(arr, starts, 100), starts, 100, 1000, 3)
    assert np.allclose(rec, arr)


def test_roundtrip_overlapping_averages_back():
    arr = np.arange(250 * 2, dtype=float).reshape(250, 2)
    starts = _window_starts(250, 100, 80)                   # overlapping windows
    rec = _stitch(_to_windows(arr, starts, 100), starts, 100, 250, 2)
    assert np.allclose(rec, arr)                            # identical overlaps average to original


def test_stitch_crops_short_series():
    arr = np.arange(50 * 4, dtype=float).reshape(50, 4)
    starts = _window_starts(50, 100, 100)
    rec = _stitch(_to_windows(arr, starts, 100), starts, 100, 50, 4)
    assert rec.shape == (50, 4) and np.allclose(rec, arr)

"""Unit tests for the optional local I/O adapter (S3 path not exercised here)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline_modules import io_utils


def test_is_s3_and_split():
    assert io_utils.is_s3("s3://bucket/a/b.csv")
    assert not io_utils.is_s3("/local/path.csv")
    assert io_utils.split_s3("s3://bucket/a/b.csv") == ("bucket", "a/b.csv")


def test_csv_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    p = tmp_path / "sub" / "out.csv"
    io_utils.write_csv(df, str(p))           # also creates parent dir
    assert p.exists()
    pd.testing.assert_frame_equal(io_utils.read_csv(str(p)), df)


def test_json_roundtrip(tmp_path):
    obj = {"k": 1, "nested": {"x": [1, 2]}}
    p = tmp_path / "r.json"
    io_utils.write_json(obj, str(p))
    assert io_utils.read_json(str(p)) == obj


def test_npy_roundtrip(tmp_path):
    arr = np.array([True, False, True])
    p = tmp_path / "m.npy"
    io_utils.write_npy(arr, str(p))
    np.testing.assert_array_equal(np.load(p), arr)

"""
Integration tests for the GX-backed check modules (ts_checks / tabular_checks).

These exercise the full ``run()`` including Great Expectations. They are gated on
the **GX 0.18 fluent API** the modules target (matching the project's
``great-expectations==0.18.19`` pin). When GX is absent, or a different major
line is installed (e.g. 1.x in the local conda envs), they skip with a reason.
"""
from __future__ import annotations

import pytest

gx = pytest.importorskip("great_expectations")

_GX_018 = gx.__version__.startswith("0.")
pytestmark = pytest.mark.skipif(
    not _GX_018,
    reason=f"ts_checks/tabular_checks target the GX 0.18 fluent API; "
           f"installed GX is {gx.__version__}",
)


def test_ts_checks_clean_series(regular_ts_df):
    from pipeline_modules import ts_checks
    r = ts_checks.run(regular_ts_df, ts_col="time")
    # ``gx`` carries the per-expectation pass/fail breakdown (added for the
    # dashboard's failed-expectations panel) alongside the boolean ``gx_passed``.
    assert set(r) == {"mode", "gx_passed", "gx", "issues", "recommendations", "summary"}
    assert r["mode"] == "time_series"
    assert r["issues"]["ts_gaps"]["has_gaps"] is False
    assert r["recommendations"]["ts_imputation"] is False


def test_ts_checks_flags_gaps(gappy_ts_df):
    from pipeline_modules import ts_checks
    r = ts_checks.run(gappy_ts_df, ts_col="time")
    assert r["issues"]["ts_gaps"]["has_gaps"] is True
    assert r["recommendations"]["ts_imputation"] is True


def test_tabular_checks_shape_and_pk(tabular_df):
    from pipeline_modules import tabular_checks
    r = tabular_checks.run(tabular_df)
    assert r["mode"] == "tabular"
    assert r["primary_key"]["columns"] == ["id"]
    # 'sometimes_missing' is ~17% null -> reported and recommended for imputation
    assert "sometimes_missing" in r["missing_columns"]
    assert any("imputation" in s for s in r["recommendations"])

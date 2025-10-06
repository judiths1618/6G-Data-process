# EUR Dataset Quality Report Summary

This note summarizes the Beam/sequential data-quality run for the EUR workload CSVs
(`6GDALI_Datasets/EUR/6907619/*.csv`). The report was generated with `python dq_local_beam.py`
using the sequential engine so it can run without Apache Beam locally.

## Run metadata

- **Generated at:** 2025-10-05T16:27:29Z
- **Files processed:** 4
- **Total good rows:** 0
- **Total issue rows:** 124,040

## Issue overview

All issue rows are attributed to the **Staleness** dimension. Every record across the four CSVs
exceeded the configured freshness SLO of 24 hours (each event timestamp is ~34,202 hours old).
No Accuracy, Completeness, Consistency, or Duplication issues were detected.

## File-level observations

| File | Rows scanned | Good rows | Notes |
| --- | --- | --- | --- |
| `amf-performance.csv` | 27,413 | 0 | Every row marked stale (`stale_event:time`). |
| `golang-web-server-performance.csv` | 58,763 | 0 | All rows stale; numeric profiles only. |
| `python-web-server-performance.csv` | 15,308 | 0 | All rows stale; numeric profiles only. |
| `rabbitmq-performance.csv` | 22,556 | 0 | All rows stale; numeric profiles only. |

The total 124,040 stale rows across the four files matches the scenario count reported for the
Staleness dimension.

## Next steps

- Confirm whether the freshness rule (24h SLO and 1h future guardrail) is still appropriate for
  the archival EUR workload. If these CSVs are historical baselines, consider disabling or
  relaxing the freshness SLO for this dataset when re-running checks.
- Otherwise, refresh the source data so the event timestamps fall within the freshness SLO and
  re-run the quality pipeline.

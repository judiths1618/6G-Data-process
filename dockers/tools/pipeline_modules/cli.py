"""
cli — optional convenience wrapper for standalone runs.

    python -m pipeline_modules manifest
    python -m pipeline_modules profile     --input data.csv            --output profile.json
    python -m pipeline_modules ts-qc       --input data.csv            --output report.json  --ts-col time
    python -m pipeline_modules tabular-qc  --input data.csv            --output report.json
    python -m pipeline_modules preprocess  --input raw.csv             --prepared-dir ./prepared_amf
    python -m pipeline_modules split       --prepared-dir ./prepared_amf

``--input/--output/--prepared-dir`` accept a local path or ``s3://bucket/key``.
This file is *not* the integration contract — that is ``import pipeline_modules`` — it
just chains the I/O adapter around the pure modules.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import io_utils, profiling, ts_checks, tabular_checks, transform, split
from .registry import MANIFEST


def _cmd_manifest(args):
    json.dump(MANIFEST, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def _cmd_profile(args):
    df = io_utils.read_csv(args.input)
    report = profiling.profile(df, timestamp_col=args.ts_col)
    io_utils.write_json(report, args.output)
    print(f"[profile] wrote {args.output}")


def _cmd_ts_qc(args):
    df = io_utils.read_csv(args.input)
    ts_col = args.ts_col or profiling.profile(df)["timestamp_column"]
    if not ts_col:
        sys.exit("ts-qc: could not determine a timestamp column; pass --ts-col")
    report = ts_checks.run(df, ts_col=ts_col)
    io_utils.write_json(report, args.output)
    print(f"[ts-qc] wrote {args.output}  (gx_passed={report['gx_passed']})")


def _cmd_tabular_qc(args):
    df = io_utils.read_csv(args.input)
    report = tabular_checks.run(df)
    io_utils.write_json(report, args.output)
    print(f"[tabular-qc] wrote {args.output}  (gx_passed={report['gx_passed']})")


def _cmd_preprocess(args):
    if io_utils.is_s3(args.input) or io_utils.is_s3(args.prepared_dir):
        sys.exit("preprocess: the end-to-end transform is local-path only for now; "
                 "stage S3 objects to a local dir first.")
    meta = transform.preprocess_csv(
        args.input, args.prepared_dir,
        time_col=args.ts_col, convert_units=args.convert_units,
    )
    print(f"[preprocess] wrote bundle to {args.prepared_dir} "
          f"(train={meta.get('train_rows')}, test={meta.get('test_rows')})")


def _cmd_split(args):
    pdir = args.prepared_dir.rstrip("/")
    meta = io_utils.read_json(f"{pdir}/meta.json")
    df = io_utils.read_csv(f"{pdir}/regularized.csv") if args.regularized \
        else io_utils.read_csv(f"{pdir}/train.csv")
    parts = split.train_test(
        df, meta,
        split_ratio=args.split_ratio, holdout_frac=args.holdout_frac,
        holdout_block_size=args.block_size, seed=args.seed,
    )
    io_utils.write_csv(parts.train, f"{pdir}/train.csv")
    io_utils.write_csv(parts.test_input, f"{pdir}/test_input.csv")
    io_utils.write_csv(parts.test_gt, f"{pdir}/test_gt.csv")
    io_utils.write_npy(parts.eval_mask, f"{pdir}/eval_holdout_mask.npy")
    print(f"[split] train={parts.meta['train_rows']} test={parts.meta['test_rows']} "
          f"holdout_rows={parts.meta['holdout_rows']} hidden={parts.meta['hidden_values']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline_modules", description="6G-DALI DataOps tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("manifest", help="print the module metadata manifest").set_defaults(
        func=_cmd_manifest)

    sp = sub.add_parser("profile", help="dataset profiling")
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--ts-col", default=None)
    sp.set_defaults(func=_cmd_profile)

    sp = sub.add_parser("ts-qc", help="time-series quality checks")
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--ts-col", default=None)
    sp.set_defaults(func=_cmd_ts_qc)

    sp = sub.add_parser("tabular-qc", help="tabular quality checks")
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=_cmd_tabular_qc)

    sp = sub.add_parser("preprocess", help="end-to-end preprocessing → prepared bundle")
    sp.add_argument("--input", required=True)
    sp.add_argument("--prepared-dir", required=True)
    sp.add_argument("--ts-col", default=None)
    sp.add_argument("--convert-units", action="store_true")
    sp.set_defaults(func=_cmd_preprocess)

    sp = sub.add_parser("split", help="train/test split + eval holdout (1:1)")
    sp.add_argument("--prepared-dir", required=True)
    sp.add_argument("--regularized", action="store_true",
                    help="split regularized.csv instead of train.csv")
    sp.add_argument("--split-ratio", type=float, default=0.8)
    sp.add_argument("--holdout-frac", type=float, default=0.15)
    sp.add_argument("--block-size", type=int, default=5)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=_cmd_split)

    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

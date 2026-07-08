#!/usr/bin/env python3
"""
Automated imputation off the DataOps handoff, then a clean-vs-imputed comparison.

Flow:
  1. Read the pipeline report (``--report``) — or run the pipeline first
     (``--run``) — to get the handoff (prepared-dir bundle + selected app/method).
  2. Run imputation on the bundle. The default is Darts/``nearest``; use
     ``--method all`` or ``--methods ...`` to populate the generated directory
     with every dependency-free Darts/ImputeGAP built-in. The ``pandas`` engine is
     Darts-faithful for interpolation methods; ``--engine darts`` subprocesses
     the real Darts runner where it is installed.
  3. Compare the regularized (gappy) input against the imputed output — fill
     coverage + accuracy on the test eval cells — and write
     ``<report_stem>_imputation_compare.json``.

Examples:
    python scripts/auto_impute.py --run --timestamp-col time
    python scripts/auto_impute.py --report reports/amf-performance_report.json --method nearest
    python scripts/auto_impute.py --report reports/amf-performance_report.json --method all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataops.imputation_runner import (
    builtin_methods,
    build_final_dataset,
    compare_clean_vs_imputed,
    impute_bundle,
)

ALL_BUILTIN_SPECS = [
    *[("darts", method) for method in builtin_methods("darts")],
    *[("imputegap", method) for method in builtin_methods("imputegap")],
]


def _resolve_method(handoff: dict, override: str | None) -> str:
    if override:
        return override
    sel = (handoff.get("selection") or {})
    if sel.get("app") == "Darts" and sel.get("method"):
        return sel["method"]
    return "nearest"  # the Darts baseline this automation targets


def _parse_method_spec(spec: str) -> tuple[str, str]:
    """Parse method specs accepted by the CLI.

    Supported forms:
      nearest             -> darts/nearest
      darts/nearest       -> darts/nearest
      darts_nearest       -> darts/nearest
      imputegap_mean      -> imputegap/mean
    """
    raw = spec.strip()
    if not raw:
        raise ValueError("empty method spec")
    if "/" in raw:
        lib, method = raw.split("/", 1)
        return lib.strip(), method.strip()
    for prefix in ("darts_", "imputegap_"):
        if raw.startswith(prefix):
            return prefix[:-1], raw[len(prefix):]
    return "darts", raw


def _resolve_method_specs(handoff: dict, method: str | None, methods: str | None) -> list[tuple[str, str]]:
    if methods:
        if methods.strip().lower() == "all":
            return list(ALL_BUILTIN_SPECS)
        return [_parse_method_spec(part) for part in methods.split(",")]
    resolved = _resolve_method(handoff, method)
    if resolved.lower() == "all":
        return list(ALL_BUILTIN_SPECS)
    return [_parse_method_spec(resolved)]


def _default_final_path(report: dict, report_path: Path) -> Path:
    out_csv = Path(report.get("output", "data/processed/clean.csv"))
    base = out_csv.stem.removesuffix("_remediated")
    return out_csv.with_name(f"{base}_final.csv")


def _method_final_path(output_dir: str | Path, lib: str, method: str) -> Path:
    return Path(output_dir) / f"{lib}_{method}_final.csv"


def _print_summary(result: dict, comparison: dict, final: dict) -> None:
    method_label = f"{result['lib']}/{result['method']}"
    print(f"\nImputation: {method_label} ({result['engine']} engine) → {result['output_dir']}")
    for kind, info in result["files"].items():
        print(f"  {kind:5s}: filled {info['filled']:,}/{info['nan_before']:,} NaN cells "
              f"(residual {info['nan_after']:,})  → {Path(info['path']).name}")
    for kind, rep in comparison["splits"].items():
        acc = rep.get("accuracy")
        print(f"  {kind:5s}: fill_rate {rep['fill_rate']*100:.1f}%  "
              f"residual_nan {rep['residual_nan']:,}")
        if acc and acc.get("eval_cells"):
            print(f"         eval cells {acc['eval_cells']:,} — per-column MAE "
                  "(interpretable; pooled is scale-mixed):")
            for col, cm in sorted(acc.get("per_column", {}).items(),
                                  key=lambda kv: kv[1]["MAE"], reverse=True):
                print(f"           {col:12s} MAE {cm['MAE']:>12.4g}  "
                      f"MAPE {cm['MAPE_%']:>8.2f}%  (n={cm['eval_cells']})")
    print(f"\n★ FINAL cleaned data → {final['path']}")
    print(f"  {final['rows']:,} rows · {len(final['columns'])} cols · "
          f"gaps filled {final['gaps_before'] - final['gaps_after']:,}/{final['gaps_before']:,} "
          f"(residual {final['gaps_after']:,})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Automated handoff imputation + comparison")
    p.add_argument("--config", default="config/dataops.yaml")
    p.add_argument("--report", default=None,
                   help="Existing pipeline report JSON. If omitted, derived from --config.")
    p.add_argument("--run", action="store_true",
                   help="Run the pipeline first (refreshes the handoff/bundle).")
    p.add_argument("--timestamp-col", default=None)
    p.add_argument("--method", default=None,
                   help="Override imputation method (default: handoff/nearest). "
                        "Use 'all' for all dependency-free built-ins.")
    p.add_argument("--methods", default=None,
                   help="Comma-separated method specs, e.g. "
                        "darts_linear,darts_nearest,imputegap_mean. "
                        "Use 'all' for all dependency-free built-ins.")
    p.add_argument("--engine", default="pandas", choices=["pandas", "darts"])
    p.add_argument("--inputs", nargs="+", default=["train", "test"], choices=["train", "test"])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--final-path", default=None,
                   help="Where to write the final cleaned dataset "
                        "(default: <output_stem>_final.csv next to the pipeline output).")
    args = p.parse_args(argv)

    if args.run or not args.report:
        from pipelines.minimal_dataops import run_from_config
        report = run_from_config(args.config, timestamp_col=args.timestamp_col)
        report_path = Path(report["report_path"])
    else:
        report_path = Path(args.report)
        report = json.loads(report_path.read_text(encoding="utf-8"))

    handoff = report.get("handoff", {})
    if not handoff.get("needs_ts_imputation"):
        print(f"No time-series imputation needed (reason: {handoff.get('reason')}). Nothing to do.")
        return 0
    prepared_dir = handoff.get("prepared_dir")
    if not prepared_dir or not Path(prepared_dir).exists():
        print(f"Handoff has no usable prepared_dir ({prepared_dir!r}); "
              f"re-run with --run to build the bundle.", file=sys.stderr)
        return 2

    specs = _resolve_method_specs(handoff, args.method, args.methods)
    if args.final_path and len(specs) > 1:
        print("--final-path can only be used with a single method; multi-method runs "
              "write <lib>_<method>_final.csv under the generated directory.",
              file=sys.stderr)
        return 2

    runs = []
    for lib, method in specs:
        print(f"→ Imputing bundle {prepared_dir}  method={lib}/{method}  engine={args.engine}")
        result = impute_bundle(
            prepared_dir, method=method, lib=lib, output_dir=args.output_dir,
            inputs=args.inputs, engine=args.engine,
        )
        comparison = compare_clean_vs_imputed(prepared_dir, result)

        # Single-method mode preserves the historical endpoint:
        # data/processed/<name>_final.csv. Multi-method mode writes method-scoped
        # finals beside the imputed splits so no method overwrites another.
        if args.final_path:
            final_path = Path(args.final_path)
        elif len(specs) == 1:
            final_path = _default_final_path(report, report_path)
        else:
            final_path = _method_final_path(result["output_dir"], lib, method)
        final = build_final_dataset(
            prepared_dir, method=method, output_path=final_path, engine=args.engine,
            lib=result["lib"], bundle_result=result, imputed_dir=result["output_dir"],
        )
        run = {"imputation": result, "comparison": comparison, "final_dataset": final}
        if len(specs) > 1 and lib == "darts" and method == "nearest" and not args.final_path:
            run["canonical_final_dataset"] = build_final_dataset(
                prepared_dir,
                method=method,
                output_path=_default_final_path(report, report_path),
                engine=args.engine,
                lib=result["lib"],
                bundle_result=result,
                imputed_dir=result["output_dir"],
            )
        runs.append(run)

    if len(runs) == 1:
        out = runs[0]
    else:
        out = {
            "runs": runs,
            "imputations": [run["imputation"] for run in runs],
            "comparisons": [run["comparison"] for run in runs],
            "final_datasets": [run["final_dataset"] for run in runs],
            "canonical_final_dataset": next(
                (run["canonical_final_dataset"] for run in runs if "canonical_final_dataset" in run),
                None,
            ),
        }
    cbase = report_path.stem.removesuffix("_report")
    compare_path = report_path.with_name(f"{cbase}_imputation_compare.json")
    compare_path.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")

    # ---- summary ----------------------------------------------------------
    for run in runs:
        _print_summary(run["imputation"], run["comparison"], run["final_dataset"])
        if "canonical_final_dataset" in run:
            print(f"  canonical final alias → {run['canonical_final_dataset']['path']}")
    print(f"\nWrote comparison → {compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

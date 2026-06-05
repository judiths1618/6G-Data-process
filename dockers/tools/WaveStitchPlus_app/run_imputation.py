#!/usr/bin/env python3
"""
WaveStitch+ imputation runner — baseline-compatible CLI wrapper.

Exposes the same interface as the baseline runners (Darts/ImputeGAP/PyPOTS)
``run_imputation.py`` so the dashboard's "Run experiment" tab can drive
WaveStitch+ the same way:

    run_imputation.py --prepared-dir <dir> --output-dir <dir> --method <m> [--device ...]

Internally it orchestrates the two native scripts on the existing
``prepared_<subset>/`` bundle:

    train_improved.py      (trains, reusing the prepared dir; no re-preprocess)
    synthesis_improved.py  (synthesizes the imputed test CSV)

and writes baseline-compatible split outputs:

    wavestitchplus_train_imputed.csv
    wavestitchplus_test_imputed.csv

The default ``full`` method uses the method-free filenames above. Explicit
``em`` and ``standard`` runs keep their method token in the filename. The train
output is the EM train-imputation artifact emitted by training. Standard non-EM
training does not emit that artifact, so a requested standard train output is
skipped with a warning.

Both outputs preserve the originally-observed cells from the prepared split
CSVs: only cells that were missing (NaN) in the input carry WaveStitch+ values,
matching the PyPOTS baseline so the comparison reflects imputation quality, not
whole-series reconstruction noise.

Device selection (PyTorch picks cuda automatically when available):
    --device auto   use the GPU if present, else CPU         (default)
    --device gpu    same as auto (kept explicit for the UI)
    --device cpu    force CPU by hiding CUDA (CUDA_VISIBLE_DEVICES="")
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd

APP_DIR = Path(__file__).resolve().parent

# method -> (model_type passed to synthesis, whether training uses the EM loop)
# ``v1`` is the canonical token (parallel to v2); ``full`` is kept as a
# back-compat alias so older scripts/Airflow configs keep working.
METHODS = {
    "v1": ("em", True),          # default high-quality EM pipeline (was "full")
    "full": ("em", True),        # alias for v1
    "em": ("em", True),
    "standard": ("standard", False),
}


def _env_for_device(device: str) -> dict:
    env = os.environ.copy()
    if device == "cpu":
        # Hide all GPUs so torch.cuda.is_available() -> False everywhere.
        env["CUDA_VISIBLE_DEVICES"] = ""
    # "gpu"/"auto": leave CUDA_VISIBLE_DEVICES untouched; torch falls back to
    # CPU on its own when no device is present.
    return env


def _run(cmd: List[str], env: dict) -> None:
    print(f"[WaveStitch+] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"step failed (exit {proc.returncode}): {cmd[0]} {cmd[1]}")


def _output_name(method: str, split: str) -> str:
    # Default ``full`` outputs are tagged ``v1`` so they sit alongside
    # ``wavestitchplus_v2_<split>_imputed.csv`` with a parallel naming
    # convention. Other methods (em/standard) keep their explicit token.
    method_tag = "v1" if method == "full" else method
    return f"wavestitchplus_{method_tag}_{split}_imputed.csv"


def _sync_model_artifacts(prepared: Path, output_dir: Path, method: str) -> Path | None:
    """Copy trained checkpoints into the generated output tree.

    Training keeps ``prepared/saved_model`` for backwards compatibility with the
    synthesis script. We also mirror the checkpoints under
    ``generated_<subset>/saved_models/wavestitchplus/<method>/`` so experiment
    outputs and saved models are grouped together in the standard results tree.
    """
    src = prepared / "saved_model"
    if not src.exists():
        return None
    dest = output_dir / "saved_models" / "wavestitchplus" / method
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.glob("*.pth"):
        shutil.copy2(path, dest / path.name)
    return dest


def _prepared_labels(prepared: Path) -> tuple[str | None, str | None]:
    """Return (experiment, subset) for ``experiments/<experiment>/prepared_<subset>``."""
    subset = None
    if prepared.name.startswith("prepared_"):
        subset = prepared.name.removeprefix("prepared_")
    experiment = prepared.parent.name if prepared.parent.name else None
    return experiment, subset


def _cleanup_native_artifacts(prepared: Path, output_dir: Path) -> None:
    """Remove duplicate side-effect outputs after publishing standard artifacts."""
    for path in [
        prepared / "saved_model",
        prepared / "train_imputed.npy",
        prepared / "train_imputed_denorm.npy",
        prepared / "train_imputed_denorm.csv",
        prepared / "training_completed.json",
    ]:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    experiment, subset = _prepared_labels(prepared)
    if experiment and subset:
        for path in [
            Path("saved_models") / experiment / subset,
            Path("generated") / experiment / subset,
            APP_DIR / "saved_models" / experiment / subset,
            APP_DIR / "generated" / experiment / subset,
        ]:
            if path.exists():
                shutil.rmtree(path)

    print(f"[WaveStitch+] cleaned duplicate native artifacts; grouped outputs remain in {output_dir}")


def _fill_missing_only(input_df: pd.DataFrame, imputed_df: pd.DataFrame) -> pd.DataFrame:
    """Return ``input_df`` with originally-missing (NaN) cells filled from
    ``imputed_df``; observed input cells are preserved exactly.

    Aligned by row position, so the output keeps ``input_df``'s schema and row
    count (this also restores any column, e.g. the timestamp, that the imputed
    artifact may omit). Keeps the input's non-null values and fills only the
    nulls from the imputed frame.
    """
    out = input_df.reset_index(drop=True).copy()
    imp = imputed_df.reset_index(drop=True).reindex(out.index)
    for col in out.columns:
        if col in imp.columns:
            out[col] = out[col].where(out[col].notna(), imp[col])
    return out


def _publish_train_output(prepared: Path, output_dir: Path, method: str) -> Path | None:
    train_imputed = prepared / "train_imputed_denorm.csv"
    train_input = prepared / "train.csv"
    if not train_imputed.exists():
        print(f"[WaveStitch+] skip train output: {train_imputed} not found")
        return None

    imputed_df = pd.read_csv(train_imputed)
    if train_input.exists():
        # Preserve observed train cells; fill only originally-missing ones from
        # the EM train-imputation (also restores the train.csv schema, which
        # train_imputed_denorm.csv may omit the timestamp from).
        imputed_df = _fill_missing_only(pd.read_csv(train_input), imputed_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / _output_name(method, "train")
    imputed_df.to_csv(out_csv, index=False)
    print(f"[WaveStitch+] wrote {out_csv}")
    return out_csv


def run(args: argparse.Namespace) -> List[Path]:
    prepared = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.fast:
        # Tiny hyperparams so a dashboard click returns in well under a minute
        # on CPU — for verifying the wiring end-to-end, not for quality.
        args.em_iterations, args.epochs_per_em = 1, 5
        args.ddim_steps, args.repaint_rounds = 5, 1
        print("[WaveStitch+] FAST smoke mode: em=1 epochs/em=5 ddim=5 repaint=1")

    model_type, use_em = METHODS[args.method]
    env = _env_for_device(args.device)

    dev_note = "CPU (forced)" if args.device == "cpu" else f"{args.device} (cuda if available)"
    print(f"[WaveStitch+] method={args.method} model_type={model_type} "
          f"use_em={use_em} device={dev_note}")
    print(f"[WaveStitch+] prepared_dir={prepared}  output_dir={output_dir}")

    # ---- 1. Train (reuse the existing prepared dir; no re-preprocess) -------
    train_cmd = [
        sys.executable, str(APP_DIR / "train_improved.py"),
        "-d", "custom_csv",
        "-prepared_dir", str(prepared),
        "-repaint_rounds", str(args.repaint_rounds),
        "-ddim_steps", str(args.ddim_steps),
        "-save_train_imputed_denorm",
        "-train_imputed_clamp", "bounds",
        "-model_tag", args.method,
    ]
    if use_em:
        train_cmd += [
            "-use_em",
            "-em_iterations", str(args.em_iterations),
            "-epochs_per_em", str(args.epochs_per_em),
        ]
    _run(train_cmd, env)
    model_dir = _sync_model_artifacts(prepared, output_dir, args.method)
    if model_dir is not None:
        print(f"[WaveStitch+] saved model artifacts grouped under {model_dir}")

    written: List[Path] = []
    if "train" in args.inputs and use_em:
        train_csv = _publish_train_output(prepared, output_dir, args.method)
        if train_csv is not None:
            written.append(train_csv)
    elif "train" in args.inputs:
        print("[WaveStitch+] skip train output: standard training does not emit "
              "train_imputed_denorm.csv")

    # ---- 2. Synthesis (writes the imputed test CSV) -------------------------
    if "test" in args.inputs:
        out_csv = output_dir / _output_name(args.method, "test")
        synth_cmd = [
            sys.executable, str(APP_DIR / "synthesis_improved.py"),
            "-d", "custom_csv",
            "-prepared_dir", str(prepared),
            "-out_csv", str(out_csv),
            "-model_type", model_type,
            "-model_tag", args.method,
            "-clamp_mode", "bounds",
            "-repaint_rounds", str(args.repaint_rounds),
            "-guidance_scale", str(args.guidance_scale),
            "-n_trials", str(args.n_trials),
            "-ddim_steps", str(args.ddim_steps),
            "-bound_headroom", "1.2",
        ]
        _run(synth_cmd, env)

        # Preserve observed test cells; only originally-missing cells keep the
        # synthesized values (matches the PyPOTS baseline's behaviour).
        test_input = prepared / "test_input.csv"
        if test_input.exists():
            merged = _fill_missing_only(pd.read_csv(test_input), pd.read_csv(out_csv))
            merged.to_csv(out_csv, index=False)
            print(f"[WaveStitch+] preserved observed cells in {out_csv}")

        print(f"[WaveStitch+] wrote {out_csv}")
        written.append(out_csv)

    _cleanup_native_artifacts(prepared, output_dir)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="WaveStitch+ imputation runner (baseline-compatible CLI)")
    p.add_argument("--prepared-dir", required=True,
                   help="prepared_<subset>/ folder (meta.json, train.csv, test_input.csv, ...)")
    p.add_argument("--output-dir", required=True,
                   help="where to write WaveStitch+ split outputs; the default v1 "
                        "method writes wavestitchplus_v1_<train|test>_imputed.csv")
    p.add_argument("--method", default="v1", choices=sorted(METHODS),
                   help="v1/full/em = EM diffusion pipeline; standard = non-EM "
                        "(``full`` is a back-compat alias for ``v1``)")
    p.add_argument("--inputs", nargs="+", default=["train", "test"],
                   choices=["train", "test"],
                   help="Which split outputs to publish (default: both)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"],
                   help="cpu forces CUDA off; auto/gpu use the GPU when available")
    p.add_argument("--fast", action="store_true",
                   help="tiny hyperparams (em=1, epochs/em=5, ddim=5, repaint=1) for a "
                        "quick end-to-end smoke test; overrides the values below")
    # Hyperparameters (modest defaults so interactive dashboard runs stay tractable).
    p.add_argument("--em-iterations", type=int, default=3)
    p.add_argument("--epochs-per-em", type=int, default=50)
    p.add_argument("--ddim-steps", type=int, default=30)
    p.add_argument("--repaint-rounds", type=int, default=3)
    p.add_argument("--guidance-scale", type=float, default=0.1)
    p.add_argument("--n-trials", type=int, default=1)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

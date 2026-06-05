"""HARPOON synthesis on top of a pre-trained WaveStitch+ model.

Applies HARPOON's inference-time manifold-bound guidance during the RePaint-DDIM
reverse loop — no retraining required. The pre-trained WaveStitch+ checkpoint
acts as the unconditional denoiser; HARPOON guides each step toward the
observed-data feasible box ``[lb, ub]`` (in z-space), with an optional hard
positive projection at the end.

The script mirrors ``synthesis_improved.py``'s I/O so it integrates with the
existing pipeline (``prepared_<subset>/`` layout, FlexibleScaler, ValueBounds,
``-test_csv`` / ``-ignore_col_masks`` override, models under
``generated_<subset>/saved_models``). Only the *guidance term* differs.

Algorithm (per imputed target cell):
    pen(x) = ReLU(x - ub)^p + ReLU(lb - x)^p
    loss_HARPOON = bound_lambda * Σ over imputed-cells [ pen(x0_pred) ]
    total guidance loss = l1 (stitch) + l2 (obs consistency) + loss_HARPOON

Bounds are auto-derived from observed data (per target column) as
``[pos_eps, quantile(observed, auto_ub_q) * (1 + auto_ub_pad)]``, then mapped
into z-space via the scaler so they live in the same space as the diffusion
sample. Setting ``-bound_lambda 0`` falls back to vanilla v1 behaviour.

Reference: HARPOON — Generalised Manifold Guidance for Conditional Tabular
Diffusion (Shankar, Wang, Hai, Chen; ICLR 2026).  https://arxiv.org/abs/2602.07875
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse everything we already have in the canonical synthesis script — the only
# thing that changes is the extra HARPOON guidance term inside the sampling loop.
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from helper.training_utils import MyDataset, fetchModel, fetchDiffusionConfig  # noqa: E402
import synthesis_improved as si  # noqa: E402
from custom_pipeline.directory_manager import get_save_dir, get_generated_dir  # noqa: E402


# =============================================================================
# HARPOON-specific bits
# =============================================================================

def bound_penalty(x: torch.Tensor, mask: torch.Tensor,
                  lb: torch.Tensor, ub: torch.Tensor, power: float) -> torch.Tensor:
    """Per-cell hinge penalty raised to ``power``, masked to imputed positions.

    ``x``, ``mask``: ``[B, W, n_target]`` (mask True/1.0 at imputed cells).
    ``lb``, ``ub``: ``[n_target]`` (broadcast over B, W).
    Returns a per-window tensor ``[B]`` so it composes with the existing
    l1/l2 per-window sums in :class:`HarpoonDDIMSynthesizer`.
    """
    above = F.relu(x - ub)
    below = F.relu(lb - x)
    pen = above.pow(power) + below.pow(power) if power != 1.0 else above + below
    return (pen * mask).sum(dim=(1, 2))


def auto_pos_bounds_from_observed(
    df_input: pd.DataFrame,
    target_cols: list,
    missing_row_mask: np.ndarray,
    eps_pos: float = 1e-6,
    q_high: float = 0.99,
    pad_ratio: float = 0.05,
    fallback_ub: float = 1e6,
) -> dict:
    """Per-target ``[lb, ub]`` derived from *observed* rows (raw scale).

    ``missing_row_mask`` is True where the row is to-be-synthesized (matching
    HARPOON's ``rows_to_synth``); observed rows = the complement. Falls back to
    a single observation's max or ``fallback_ub`` when too few observations.
    """
    obs = df_input.loc[~missing_row_mask, target_cols]
    bounds: dict = {}
    for c in target_cols:
        x = obs[c].dropna().to_numpy(dtype=np.float64)
        lb = float(eps_pos)
        if len(x) >= 10:
            ub = float(np.quantile(x, q_high))
            if not np.isfinite(ub) or ub <= lb:
                ub = float(np.max(x))
        elif len(x) > 0:
            ub = float(np.max(x))
        else:
            ub = float(fallback_ub)
        if pad_ratio > 0:
            ub *= (1.0 + pad_ratio)
        if ub <= lb + 1e-12:
            ub = lb + 1.0
        bounds[c] = [lb, ub]
    return bounds


class HarpoonDDIMSynthesizer(si.RePaintDDIMSynthesizer):
    """RePaint-DDIM + HARPOON manifold-bound guidance.

    ``lb``/``ub`` are *normalized* (z-space) per-target tensors of shape
    ``[n_target]`` matching the model's input space; ``bound_lambda`` scales
    the penalty within the same autograd target as l1/l2.
    """

    def __init__(self, *args, lb: torch.Tensor, ub: torch.Tensor,
                 bound_lambda: float = 0.3, bound_power: float = 2.0, **kw):
        super().__init__(*args, **kw)
        self.lb = lb
        self.ub = ub
        self.bound_lambda = float(bound_lambda)
        self.bound_power = float(bound_power)
        print(f"[HARPOON] bound_lambda={self.bound_lambda} bound_power={self.bound_power}")

    def _extra_guidance_loss(self, x0c, obs_mask_batch_f, synth_mask_batch_f, test_batch):
        if self.bound_lambda <= 0 or self.lb is None:
            return torch.zeros((), device=x0c.device)
        tgt = x0c[:, :, self.target_indices]                 # [B, W, n_target]
        pen = bound_penalty(tgt, synth_mask_batch_f,
                            self.lb, self.ub, self.bound_power)
        return self.bound_lambda * pen


# =============================================================================
# Main — mirrors synthesis_improved's custom_csv flow + HARPOON specifics
# =============================================================================

def _normalize_bounds(lb_raw: np.ndarray, ub_raw: np.ndarray,
                      scaler, model_clip_bound: float, device) -> tuple:
    """Map raw-scale per-target bounds to the model's z-space tensors."""
    if scaler is not None:
        z_lo = (lb_raw - scaler.center_) / scaler.scale_
        z_hi = (ub_raw - scaler.center_) / scaler.scale_
        z_lo = np.clip(z_lo, -model_clip_bound, model_clip_bound)
        z_hi = np.clip(z_hi, -model_clip_bound, model_clip_bound)
    else:
        z_lo, z_hi = lb_raw, ub_raw
    return (torch.as_tensor(z_lo, dtype=torch.float32, device=device),
            torch.as_tensor(z_hi, dtype=torch.float32, device=device))


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-dataset", "-d", type=str, default="custom_csv")
    p.add_argument("-prepared_dir", type=str, required=True)
    p.add_argument("-out_csv", type=str, default=None)
    p.add_argument("-test_csv", type=str, default=None)
    p.add_argument("-ignore_col_masks", action="store_true")
    # Sampling
    p.add_argument("-backbone", type=str, default="S4")
    p.add_argument("-beta_0", type=float, default=0.0001)
    p.add_argument("-beta_T", type=float, default=0.02)
    p.add_argument("-timesteps", "-T", type=int, default=200)
    p.add_argument("-hdim", type=int, default=64)
    p.add_argument("-lr", type=float, default=1e-4)
    p.add_argument("-batch_size", type=int, default=1024)
    p.add_argument("-layers", type=int, default=4)
    p.add_argument("-window_size", type=int, default=32)
    p.add_argument("-stride", type=int, default=1)
    p.add_argument("-num_res_layers", type=int, default=4)
    p.add_argument("-res_channels", type=int, default=64)
    p.add_argument("-skip_channels", type=int, default=64)
    p.add_argument("-diff_step_embed_in", type=int, default=32)
    p.add_argument("-diff_step_embed_mid", type=int, default=64)
    p.add_argument("-diff_step_embed_out", type=int, default=64)
    p.add_argument("-s4_lmax", type=int, default=100)
    p.add_argument("-s4_dstate", type=int, default=64)
    p.add_argument("-s4_dropout", type=float, default=0.0)
    p.add_argument("-s4_bidirectional", type=si.str2bool, default=True)
    p.add_argument("-s4_layernorm", type=si.str2bool, default=True)
    p.add_argument("-propCycEnc", type=si.str2bool, default=False)
    p.add_argument("-synth_mask", type=str, default="gap_imputation")
    p.add_argument("-n_trials", type=int, default=1)
    p.add_argument("-guidance_scale", type=float, default=0.1)
    p.add_argument("-repaint_rounds", type=int, default=3)
    p.add_argument("-ddim_steps", type=int, default=50)
    p.add_argument("-use_ddpm", action="store_true")
    p.add_argument("-model_type", type=str, default="em",
                   choices=["auto", "em", "standard"])
    p.add_argument("-model_tag", type=str, default="")
    # ValueBounds clamp (post-processing)
    p.add_argument("-clamp_mode", type=str, default="bounds",
                   choices=["none", "nonneg", "bounds"])
    p.add_argument("-bound_headroom", type=float, default=1.2)
    p.add_argument("-nonneg_cols", type=str, nargs="*", default=None)
    # HARPOON knobs
    p.add_argument("-bounds_json", type=str, default=None,
                   help="optional JSON ``{col: [lb, ub]}`` (raw scale); else auto")
    p.add_argument("-bound_lambda", type=float, default=0.3,
                   help="weight on HARPOON manifold-bound penalty in the guidance loss")
    p.add_argument("-bound_power", type=float, default=2.0)
    p.add_argument("-pos_eps", type=float, default=1e-6,
                   help="lower bound floor (positivity epsilon, raw scale)")
    p.add_argument("-auto_ub_q", type=float, default=0.99,
                   help="upper-bound observed-data quantile for auto-bounds")
    p.add_argument("-auto_ub_pad", type=float, default=0.05,
                   help="multiplicative padding above the quantile")
    p.add_argument("-hard_project_positive", action="store_true",
                   help="after denorm/clamp, floor target values at pos_eps")
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HARPOON] device={dev}  prepared_dir={args.prepared_dir}")

    # -------------------------------------------------------------------------
    # 1) Load prepared data (mirrors synthesis_improved)
    # -------------------------------------------------------------------------
    df_input, meta = si.load_custom_prepared(args.prepared_dir, test_csv=args.test_csv)
    time_col = meta.get("time_col", "time")
    cond_cols = meta.get("cond_cols", [])
    target_cols = meta.get("target_cols", [])
    if not target_cols:
        raise SystemExit("meta.json target_cols empty")

    split_ratio = meta.get("split_ratio", 0.8)
    regularized_rows = meta.get("regularized_rows", None)
    train_rows = meta.get("train_rows", None)
    if train_rows is None:
        if regularized_rows is None:
            raise SystemExit("meta.json missing train_rows and regularized_rows")
        train_rows = int(split_ratio * regularized_rows)

    orig_obs_mask_test = None if args.ignore_col_masks else si.load_test_target_mask(
        args.prepared_dir, target_cols=target_cols,
        n_rows=len(df_input), train_rows=train_rows,
    )
    if orig_obs_mask_test is None:
        orig_obs_mask_test = (~df_input[target_cols].isna()).to_numpy().astype(np.float32)

    input_obs_mask = (~df_input[target_cols].isna()).to_numpy().astype(np.float32)
    synth_mask_test = (input_obs_mask == 0)
    observed_row_mask = orig_obs_mask_test.any(axis=1)
    df_input = si.recompute_cond_features(
        df_input, time_col, cond_cols, target_cols, observed_row_mask,
    )

    df_synth = si.fill_targets_like_training(df_input.copy(), target_cols)
    for c in cond_cols:
        if c in df_synth.columns:
            df_synth[c] = df_synth[c].fillna(0.0)
    expected_model_cols = meta.get("all_model_cols", [c for c in df_synth.columns if c != time_col])
    df_synth = df_synth[[time_col] + expected_model_cols]
    model_cols = expected_model_cols
    target_indices = [model_cols.index(c) for c in target_cols if c in model_cols]

    # -------------------------------------------------------------------------
    # 2) Scaler + per-target HARPOON bounds (in z-space)
    # -------------------------------------------------------------------------
    scaler_dir = os.path.join(args.prepared_dir, "scaler")
    scaler = None
    model_clip_bound = 3.0
    if os.path.exists(scaler_dir):
        try:
            scaler = si.FlexibleScaler.load(scaler_dir)
            model_clip_bound = scaler.get_model_clip_bound()
            print(f"[HARPOON] scaler: {scaler.mode}/{scaler.clip_mode} bound={model_clip_bound}")
        except Exception as e:
            print(f"[WARNING] scaler load failed: {e}")

    if args.bounds_json and os.path.exists(args.bounds_json):
        with open(args.bounds_json) as f:
            bounds = json.load(f)
        print(f"[HARPOON] bounds: loaded {args.bounds_json}")
    else:
        # "Rows to synth" = rows with ANY missing target — these are the rows we
        # impute; bounds are derived from the *complement* (rows with all observed).
        any_missing = np.isnan(df_input[target_cols].to_numpy()).any(axis=1)
        bounds = auto_pos_bounds_from_observed(
            df_input, target_cols, any_missing,
            eps_pos=args.pos_eps, q_high=args.auto_ub_q, pad_ratio=args.auto_ub_pad,
        )
        print(f"[HARPOON] bounds: auto (q={args.auto_ub_q}, pad={args.auto_ub_pad}, eps={args.pos_eps})")
    lb_raw = np.array([max(bounds[c][0], args.pos_eps) for c in target_cols], dtype=np.float32)
    ub_raw = np.array([max(bounds[c][1], lb_raw[i] + 1.0) for i, c in enumerate(target_cols)], dtype=np.float32)
    lb_t, ub_t = _normalize_bounds(lb_raw, ub_raw, scaler, model_clip_bound, dev)

    # -------------------------------------------------------------------------
    # 3) Normalize inputs + build windows / masks (reuse helpers)
    # -------------------------------------------------------------------------
    d_vals = df_synth.drop(columns=[time_col]).values.astype(np.float32)
    if scaler is not None:
        d_vals[:, target_indices] = scaler.transform(d_vals[:, target_indices])

    hier_cols = np.array([model_cols.index(c) for c in cond_cols if c in model_cols], dtype=int)
    windows, window_starts = si.build_windows_from_array(d_vals, args.window_size, args.stride)
    synth_masks = si.build_windows_from_mask(
        synth_mask_test.astype(bool), args.window_size, args.stride, starts=window_starts)
    obs_masks = si.build_windows_from_mask(
        input_obs_mask.astype(bool), args.window_size, args.stride, starts=window_starts)

    # ValueBounds (post-process clamp; honours preprocessing/p995 if present)
    vbounds = si.ValueBounds(target_cols)
    if args.clamp_mode == "bounds":
        vbounds.load_from_preprocessing(scaler_dir, args.bound_headroom)
        vbounds.apply_nonneg(args.nonneg_cols)
    elif args.clamp_mode == "nonneg":
        vbounds.apply_nonneg(args.nonneg_cols)
    vbounds.summary()

    # -------------------------------------------------------------------------
    # 4) Model + HARPOON synthesizer
    # -------------------------------------------------------------------------
    in_dim = windows.shape[2]
    out_dim = in_dim - len(hier_cols)
    test_dl = DataLoader(MyDataset(windows.float(), window_size=args.window_size), batch_size=args.batch_size)
    smb_dl = DataLoader(MyDataset(synth_masks), batch_size=args.batch_size)
    omb_dl = DataLoader(MyDataset(obs_masks), batch_size=args.batch_size)

    model = fetchModel(in_dim, out_dim, args).to(dev)
    diffusion_config = fetchDiffusionConfig(args)
    non_hier_cols = np.setdiff1d(np.arange(in_dim), hier_cols)

    saved_dir = get_save_dir(args.prepared_dir)
    tag = args.model_tag or ""
    if tag and os.path.exists(os.path.join(saved_dir, f"model_{tag}_best.pth")):
        mp = os.path.join(saved_dir, f"model_{tag}_best.pth")
    elif tag and os.path.exists(os.path.join(saved_dir, f"model_{tag}.pth")):
        mp = os.path.join(saved_dir, f"model_{tag}.pth")
    elif args.model_type == "em":
        mp = os.path.join(saved_dir, "model_em_best.pth")
        if not os.path.exists(mp):
            mp = os.path.join(saved_dir, "model_em.pth")
    elif args.model_type == "standard":
        mp = os.path.join(saved_dir, "model_prop.pth" if args.propCycEnc else "model.pth")
    else:
        saved_params, mp = si.load_model_with_fallback(saved_dir, args, dev)

    if args.model_type != "auto" or tag:
        if not os.path.exists(mp):
            raise FileNotFoundError(f"model not found: {mp}")
        saved_params = torch.load(mp, map_location=dev)
    print(f"[HARPOON] loading model: {mp}")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in saved_params:
                param.copy_(saved_params[name])
    model.eval()

    synth = HarpoonDDIMSynthesizer(
        model=model, diffusion_config=diffusion_config, device=dev,
        non_hier_cols=non_hier_cols, hier_cols=hier_cols, target_indices=target_indices,
        window_size=args.window_size, stride=args.stride,
        repaint_rounds=args.repaint_rounds, ddim_steps=args.ddim_steps,
        use_ddim=not args.use_ddpm, guidance_scale=args.guidance_scale,
        clip_bound=model_clip_bound,
        lb=lb_t, ub=ub_t,
        bound_lambda=args.bound_lambda, bound_power=args.bound_power,
    )

    # -------------------------------------------------------------------------
    # 5) Sample → overlap-add → denorm → clamp → optional hard-positive
    # -------------------------------------------------------------------------
    exec_times = []
    for trial in range(args.n_trials):
        t0 = timer()
        win_out = []
        for tb, smb, omb in zip(test_dl, smb_dl, omb_dl):
            tb = si.ensure_bwc(tb, args.window_size).to(dev)
            x = synth.synthesize_batch(tb, smb.to(dev).bool(), omb.to(dev).bool())
            win_out.append(x.detach().cpu().numpy())
        elapsed = timer() - t0
        exec_times.append(elapsed)
        win_arr = np.concatenate(win_out, axis=0)
        arr = si.hann_overlap_add(win_arr, len(df_input), args.window_size,
                                  args.stride, starts=window_starts)
        if scaler is not None:
            arr[:, target_indices] = scaler.inverse_transform(arr[:, target_indices])
        if args.clamp_mode != "none":
            diag = vbounds.clamp(arr, target_indices)
            print(f"[HARPOON] clamped: lower={diag['n_clamped_lower']}  upper={diag['n_clamped_upper']}")

        out_df = pd.DataFrame(arr, columns=model_cols)
        final = df_input[[time_col]].copy()
        for c in target_cols:
            if c in out_df.columns:
                final[c] = out_df[c].to_numpy()[:len(final)]
        for c in cond_cols:
            if c in df_input.columns:
                final[c] = df_input[c].to_numpy()[:len(final)]

        if args.hard_project_positive:
            for c in target_cols:
                if c in final.columns:
                    final[c] = np.maximum(final[c].to_numpy(dtype=np.float64), float(args.pos_eps))

        out_dir = (f"{get_generated_dir(args.prepared_dir)}"
                   f"/{args.synth_mask}_wavestitchplus_harpoon/")
        os.makedirs(out_dir, exist_ok=True)
        mode = "ddpm" if args.use_ddpm else f"ddim{args.ddim_steps}"
        native = os.path.join(
            out_dir,
            f"imputed_harpoon_{mode}_rp{args.repaint_rounds}"
            f"_bl{args.bound_lambda}_{args.clamp_mode}"
            f"_stride_{args.stride}_trial_{trial}.csv"
        )
        final.to_csv(native, index=False)
        if args.out_csv and trial == 0:
            os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
            final.to_csv(args.out_csv, index=False)
        print(f"  [T{trial}] {elapsed:.2f}s → {native}")

    print(f"\n[HARPOON DONE] {np.mean(exec_times):.2f}s ± {np.std(exec_times):.2f}s")


if __name__ == "__main__":
    main()

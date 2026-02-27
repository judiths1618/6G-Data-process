import argparse
import os
import json
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from helper.training_utils import MyDataset, fetchModel, fetchDiffusionConfig
from helper.data_utils import Preprocessor
from helper.metasynth import metadataMask
from custom_pipeline.directory_manager import get_save_dir, get_generated_dir


# =============================================================================
# Helpers
# =============================================================================

SOFT_CLIP_BOUND = 4.0


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def soft_clip_inv(x, bound=SOFT_CLIP_BOUND):
    safe = np.clip(x / bound, -0.98, 0.98)
    return bound * np.arctanh(safe)


def soft_clip(x, bound=SOFT_CLIP_BOUND):
    return bound * np.tanh(x / bound)


# =============================================================================
# FlexibleScaler (mirrors train.py)
# =============================================================================

class FlexibleScaler:
    def __init__(self, mode="standard", clip_mode="hard", clip_bound=3.0,
                 soft_bound=SOFT_CLIP_BOUND):
        self.mode = mode
        self.clip_mode = clip_mode
        self.clip_bound = clip_bound
        self.soft_bound = soft_bound
        self.center_ = None
        self.scale_ = None

    def transform(self, x):
        z = (x - self.center_) / self.scale_
        if self.clip_mode == "soft":
            return soft_clip(z, self.soft_bound)
        return np.clip(z, -self.clip_bound, self.clip_bound)

    def inverse_transform(self, z):
        if self.clip_mode == "soft":
            z = soft_clip_inv(z, self.soft_bound)
        return z * self.scale_ + self.center_

    def get_model_clip_bound(self):
        return self.soft_bound if self.clip_mode == "soft" else self.clip_bound

    @classmethod
    def load(cls, directory):
        info_path = os.path.join(directory, "scaler_info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            scaler = cls(
                mode=info.get("mode", "standard"),
                clip_mode=info.get("clip_mode", "hard"),
                clip_bound=info.get("clip_bound", 3.0),
                soft_bound=info.get("soft_bound", SOFT_CLIP_BOUND),
            )
        else:
            scaler = cls()

        cp = os.path.join(directory, "center.npy")
        if os.path.exists(cp):
            scaler.center_ = np.load(cp).astype(np.float32)
            scaler.scale_ = np.load(os.path.join(directory, "scale.npy")).astype(np.float32)
        else:
            scaler.center_ = np.load(os.path.join(directory, "mean.npy")).astype(np.float32)
            scaler.scale_ = np.load(os.path.join(directory, "std.npy")).astype(np.float32)

        scaler.scale_[scaler.scale_ < 1e-8] = 1.0
        return scaler


# =============================================================================
# Value bounds
# =============================================================================

class ValueBounds:
    """
    Per-column lower/upper bounds for post-processing clamp.
    """

    def __init__(self, target_cols: list):
        self.target_cols = target_cols
        n = len(target_cols)
        self.lower = np.full(n, -np.inf, dtype=np.float32)
        self.upper = np.full(n, np.inf, dtype=np.float32)
        self._loaded_from_data = False

    def load_from_preprocessing(self, scaler_dir: str, headroom: float = 1.2):
        upper_path = os.path.join(scaler_dir, "upper_bound_p995.npy")
        lower_path = os.path.join(scaler_dir, "lower_bound_p005.npy")
        obs_max_path = os.path.join(scaler_dir, "observed_max.npy")

        if os.path.exists(upper_path):
            p995 = np.load(upper_path).astype(np.float32)
            obs_max = (
                np.load(obs_max_path).astype(np.float32)
                if os.path.exists(obs_max_path) else p995
            )
            self.upper = p995 + headroom * np.maximum(obs_max - p995, 0.0)
            self._loaded_from_data = True
            print(f"[Bounds] Upper loaded: p99.5 * headroom={headroom}")

        if os.path.exists(lower_path):
            p005 = np.load(lower_path).astype(np.float32)
            for i in range(len(self.target_cols)):
                if self.lower[i] == -np.inf:
                    self.lower[i] = p005[i]

    def apply_nonneg(self, cols=None):
        if cols is None:
            self.lower = np.maximum(self.lower, 0.0)
        else:
            for c in cols:
                if c in self.target_cols:
                    idx = self.target_cols.index(c)
                    self.lower[idx] = max(self.lower[idx], 0.0)

    def apply_manual_overrides(self, lower_json=None, upper_json=None):
        if lower_json:
            overrides = json.loads(lower_json) if isinstance(lower_json, str) else lower_json
            for col, val in overrides.items():
                if col in self.target_cols:
                    self.lower[self.target_cols.index(col)] = float(val)

        if upper_json:
            overrides = json.loads(upper_json) if isinstance(upper_json, str) else upper_json
            for col, val in overrides.items():
                if col in self.target_cols:
                    self.upper[self.target_cols.index(col)] = float(val)

    def clamp(self, arr: np.ndarray, col_indices: list) -> dict:
        n_lo = 0
        n_hi = 0
        for i, ci in enumerate(col_indices):
            col_data = arr[:, ci]
            lo = self.lower[i]
            hi = self.upper[i]

            below = col_data < lo
            above = col_data > hi
            n_lo += int(below.sum())
            n_hi += int(above.sum())

            arr[:, ci] = np.clip(col_data, lo, hi)

        return {"n_clamped_lower": n_lo, "n_clamped_upper": n_hi}

    def summary(self):
        print("[Bounds] Per-column bounds:")
        for i, c in enumerate(self.target_cols[:10]):
            lo = f"{self.lower[i]:.4g}" if np.isfinite(self.lower[i]) else "-∞"
            hi = f"{self.upper[i]:.4g}" if np.isfinite(self.upper[i]) else "+∞"
            print(f"    {c}: [{lo}, {hi}]")
        if len(self.target_cols) > 10:
            print(f"    ... ({len(self.target_cols) - 10} more)")


# =============================================================================
# Utilities
# =============================================================================

def ensure_bwc(x, window_size):
    if x.ndim != 3:
        raise ValueError(f"Expected 3D, got {x.shape}")
    if x.shape[1] == window_size:
        return x
    if x.shape[2] == window_size:
        return x.transpose(1, 2)
    raise ValueError(f"Can't infer window axis: {x.shape}")


def load_custom_prepared(prepared_dir):
    meta_path = os.path.join(prepared_dir, "meta.json")
    test_path = os.path.join(prepared_dir, "test_input.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("meta.json not found")
    if not os.path.exists(test_path):
        raise FileNotFoundError("test_input.csv not found")

    with open(meta_path) as f:
        meta = json.load(f)

    df = pd.read_csv(test_path)
    tc = meta.get("time_col", "time")
    mc = meta.get("all_model_cols")
    if mc:
        expected = [tc] + mc
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"test_input.csv missing columns: {missing}")
        df = df[expected]
    else:
        cols = [c for c in df.columns if c != tc]
        df = df[[tc] + cols]

    return df, meta


def load_model_with_fallback(saved_dir, args, dev):
    for fn, desc in [
        ("model_em_best.pth", "EM best"),
        ("model_em.pth", "EM"),
        ("model_best.pth", "Best"),
        ("model_prop.pth" if args.propCycEnc else "model.pth", "Std"),
    ]:
        p = os.path.join(saved_dir, fn)
        if os.path.exists(p):
            print(f"[INFO] Loading: {p} ({desc})")
            return torch.load(p, map_location=dev), p
    raise FileNotFoundError(f"No model in {saved_dir}")


def load_test_target_mask(prepared_dir, target_cols, n_rows, train_rows):
    """
    Load ORIGINAL column-level observed masks for the test split
    from preprocessing artifacts.

    Returns:
        observed_mask_test: [T_test, n_target] float32
            1 = originally observed in the regularized data
            0 = originally missing
    """
    col_mask_dir = os.path.join(prepared_dir, "col_masks")
    if not os.path.isdir(col_mask_dir):
        return None

    mask_arrays = []
    for c in target_cols:
        mask_path = os.path.join(col_mask_dir, f"{c}.npy")
        if not os.path.exists(mask_path):
            return None
        full_mask = np.load(mask_path).astype(np.float32)
        mask_arrays.append(full_mask[train_rows:train_rows + n_rows])

    return np.column_stack(mask_arrays).astype(np.float32)


def recompute_cond_features(df, time_col, cond_cols, target_cols, observed_row_mask):
    from custom_pipeline.features import add_time_features, add_gap_structure_features

    df = df.copy()
    df = add_time_features(df, time_col=time_col)
    df = add_gap_structure_features(df, observed_row_mask)

    for c in cond_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    return df


def fill_targets_like_training(df, target_cols):
    """
    Match training-side target initialization:
    interpolate + ffill + bfill, then fallback to mean/0.
    """
    df = df.copy()
    for col in target_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .interpolate(method="linear", limit_direction="both")
                .ffill()
                .bfill()
            )
            if df[col].isna().any():
                m = df[col].mean()
                fill_val = m if not np.isnan(m) else 0.0
                df[col] = df[col].fillna(fill_val)
    return df


def compute_window_starts(T_seq, window_size, stride):
    """
    Return window start indices with guaranteed tail coverage.
    """
    if T_seq < window_size:
        raise ValueError(f"T_seq ({T_seq}) < window_size ({window_size})")

    starts = list(range(0, T_seq - window_size + 1, stride))
    last_start = T_seq - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def build_windows_from_array(arr, window_size, stride):
    T_seq, C = arr.shape
    starts = compute_window_starts(T_seq, window_size, stride)

    windows = torch.zeros(len(starts), window_size, C, dtype=torch.float32)
    for i, s in enumerate(starts):
        windows[i] = torch.from_numpy(arr[s:s + window_size])
    return windows, starts


def build_windows_from_mask(mask_arr, window_size, stride, starts=None):
    """
    mask_arr: [T, n_target]
    returns: [nw, window_size, n_target]
    """
    T_seq, n_tgt = mask_arr.shape
    if starts is None:
        starts = compute_window_starts(T_seq, window_size, stride)

    masks = torch.zeros(len(starts), window_size, n_tgt, dtype=torch.bool)
    for i, s in enumerate(starts):
        masks[i] = torch.from_numpy(mask_arr[s:s + window_size].astype(bool))
    return masks


def hann_overlap_add(window_outputs, seq_len, window_size, stride, starts=None):
    """
    window_outputs: np.ndarray [num_windows, window_size, C]
    returns: np.ndarray [seq_len, C]
    """
    num_windows, _, C = window_outputs.shape
    if starts is None:
        starts = compute_window_starts(seq_len, window_size, stride)

    hann = np.hanning(window_size).reshape(-1, 1).astype(np.float32)
    out_sum = np.zeros((seq_len, C), dtype=np.float32)
    out_wt = np.zeros((seq_len, 1), dtype=np.float32)

    for i, s in enumerate(starts):
        e = s + window_size
        out_sum[s:e] += window_outputs[i] * hann
        out_wt[s:e] += hann

    out_wt = np.maximum(out_wt, 1e-8)
    return out_sum / out_wt


# =============================================================================
# RePaint-DDIM Synthesis Engine
# =============================================================================

class RePaintDDIMSynthesizer:
    """
    Conditional synthesis using RePaint-DDIM/DDPM.
    synth_mask_batch: True = missing target entry to synthesize
    obs_mask_batch  : True = currently visible target entry in test_input
    """

    def __init__(self, model, diffusion_config, device,
                 non_hier_cols, hier_cols, target_indices,
                 window_size, stride,
                 repaint_rounds=3, ddim_steps=50, use_ddim=True,
                 guidance_scale=0.1, clip_bound=3.0):
        self.model = model
        self.device = device
        self.non_hier_cols = np.array(non_hier_cols)
        self.hier_cols = np.array(hier_cols)
        self.target_indices = np.array(target_indices)
        self.window_size = window_size
        self.stride = stride
        self.repaint_rounds = max(1, repaint_rounds)
        self.use_ddim = use_ddim
        self.guidance_scale = guidance_scale
        self.clip_bound = clip_bound

        self.alpha_bars = diffusion_config["alpha_bars"].to(device)
        self.betas = diffusion_config["betas"].to(device)
        self.T = diffusion_config["T"]

        self._ts = (
            np.linspace(0, self.T - 1, ddim_steps, dtype=int)[::-1].copy()
            if use_ddim else np.arange(self.T - 1, -1, -1)
        )

        print(f"[Synth] {'DDIM' if use_ddim else 'DDPM'} "
              f"steps={len(self._ts)} repaint={self.repaint_rounds} "
              f"guide={self.guidance_scale} clip={self.clip_bound}")

    def synthesize_batch(self, test_batch, synth_mask_batch, obs_mask_batch):
        """
        test_batch      : [B, W, C]
        synth_mask_batch: [B, W, n_target] bool, True=missing target entry
        obs_mask_batch  : [B, W, n_target] bool, True=visible target entry
        """
        B, W, C = test_batch.shape
        dev = self.device

        cond_mask = torch.zeros(B, W, C, device=dev)
        cond_mask[:, :, self.hier_cols] = 1.0

        synth_mask_full = torch.zeros(B, W, C, device=dev)
        obs_mask_full = torch.zeros(B, W, C, device=dev)

        synth_mask_batch_f = synth_mask_batch.float()
        obs_mask_batch_f = obs_mask_batch.float()

        for j, tc in enumerate(self.target_indices):
            synth_mask_full[:, :, tc] = synth_mask_batch_f[:, :, j]
            obs_mask_full[:, :, tc] = obs_mask_batch_f[:, :, j]

        known_mask = torch.clamp(cond_mask + obs_mask_full, 0.0, 1.0)

        # i.i.d. Gaussian init
        x_t = torch.randn_like(test_batch)
        x_t = cond_mask * test_batch + (1.0 - cond_mask) * x_t

        for si, t in enumerate(self._ts):
            ab_t = self.alpha_bars[t]
            if si < len(self._ts) - 1:
                t_n = self._ts[si + 1]
                ab_n = self.alpha_bars[t_n]
            else:
                t_n = None
                ab_n = None

            for r in range(self.repaint_rounds):
                tv = torch.full((B,), t, device=dev, dtype=torch.long)
                x_in = cond_mask * test_batch + (1.0 - cond_mask) * x_t
                x_in_g = x_in.detach().requires_grad_(True)

                with torch.enable_grad():
                    np_ = self.model(x_in_g, tv.reshape(-1, 1)).permute(0, 2, 1)
                    nf = torch.zeros(B, W, C, device=dev)
                    nf[:, :, self.non_hier_cols] = np_

                    x0 = (x_t - torch.sqrt(1 - ab_t) * nf) / torch.sqrt(ab_t)
                    x0c = torch.clamp(x0, -self.clip_bound, self.clip_bound)

                    if self.guidance_scale > 0 and r == self.repaint_rounds - 1:
                        tgt = x0c[:, :, self.target_indices]

                        # Stitch loss
                        rolled = tgt.roll(1, 0)
                        rolled[0, self.stride:, :] = tgt[0, :(W - self.stride), :]
                        l1 = torch.sum(
                            (tgt[:, :(W - self.stride), :]
                             - rolled[:, self.stride:W, :]) ** 2,
                            dim=(1, 2)
                        )

                        # Observation consistency ONLY on currently visible target entries
                        ok = obs_mask_batch_f
                        l2 = torch.sum(
                            ok * (x0c[:, :, self.target_indices]
                                  - test_batch[:, :, self.target_indices]) ** 2,
                            dim=(1, 2)
                        )

                        grad = torch.autograd.grad(
                            l1 + l2,
                            x_in_g,
                            grad_outputs=torch.ones_like(l1)
                        )[0]
                    else:
                        grad = None

                nf = nf.detach()
                x0c = x0c.detach()

                if t_n is not None:
                    unk = torch.sqrt(ab_n) * x0c + torch.sqrt(1 - ab_n) * nf
                    if not self.use_ddim and t > 0:
                        pv = self.betas[t] * (1 - self.alpha_bars[t - 1]) / (1 - ab_t)
                        unk += torch.sqrt(pv) * torch.randn_like(unk)
                else:
                    unk = x0c

                if t_n is not None:
                    kn = (
                        torch.sqrt(ab_n) * test_batch
                        + torch.sqrt(1 - ab_n) * torch.randn_like(test_batch)
                    )
                else:
                    kn = test_batch

                merged = known_mask * kn + (1.0 - known_mask) * unk

                if grad is not None:
                    merged[:, :, self.target_indices] -= (
                        self.guidance_scale * grad[:, :, self.target_indices]
                    )

                if r < self.repaint_rounds - 1:
                    rn = torch.randn_like(merged)
                    if t_n is not None:
                        a_s = ab_t / ab_n
                        x_t = torch.sqrt(a_s) * merged + torch.sqrt(1 - a_s) * rn
                    else:
                        x_t = torch.sqrt(ab_t) * merged + torch.sqrt(1 - ab_t) * rn
                else:
                    x_t = merged

                del nf, x0c, unk, merged

            if si % max(1, len(self._ts) // 8) == 0:
                print(f"    step {si+1}/{len(self._ts)} (t={t})")

        # restore exact known values
        x_t = known_mask * test_batch + (1.0 - known_mask) * x_t
        return x_t


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "-d", type=str, required=True)
    parser.add_argument("-prepared_dir", type=str, default="./work/prepared")
    parser.add_argument("-out_csv", type=str, default=None)

    # Model
    parser.add_argument("-backbone", type=str, default="S4")
    parser.add_argument("-beta_0", type=float, default=0.0001)
    parser.add_argument("-beta_T", type=float, default=0.02)
    parser.add_argument("-timesteps", "-T", type=int, default=200)
    parser.add_argument("-hdim", type=int, default=64)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-batch_size", type=int, default=1024)
    parser.add_argument("-layers", type=int, default=4)
    parser.add_argument("-window_size", type=int, default=32)
    parser.add_argument("-stride", type=int, default=1)
    parser.add_argument("-num_res_layers", type=int, default=4)
    parser.add_argument("-res_channels", type=int, default=64)
    parser.add_argument("-skip_channels", type=int, default=64)
    parser.add_argument("-diff_step_embed_in", type=int, default=32)
    parser.add_argument("-diff_step_embed_mid", type=int, default=64)
    parser.add_argument("-diff_step_embed_out", type=int, default=64)
    parser.add_argument("-s4_lmax", type=int, default=100)
    parser.add_argument("-s4_dstate", type=int, default=64)
    parser.add_argument("-s4_dropout", type=float, default=0.0)
    parser.add_argument("-s4_bidirectional", type=str2bool, default=True)
    parser.add_argument("-s4_layernorm", type=str2bool, default=True)
    parser.add_argument("-propCycEnc", type=str2bool, default=False)
    parser.add_argument("-synth_mask", type=str, default="gap_imputation")
    parser.add_argument("-n_trials", type=int, default=5)

    # Synthesis
    parser.add_argument("-guidance_scale", type=float, default=0.1)
    parser.add_argument("-repaint_rounds", type=int, default=3)
    parser.add_argument("-ddim_steps", type=int, default=50)
    parser.add_argument("-use_ddpm", action="store_true")
    parser.add_argument("-model_type", type=str, default="auto",
                        choices=["auto", "em", "standard"])

    # Value bounds
    parser.add_argument("-clamp_mode", type=str, default="bounds",
                        choices=["none", "nonneg", "bounds"])
    parser.add_argument("-bound_headroom", type=float, default=1.2)
    parser.add_argument("-nonneg_cols", type=str, nargs="*", default=None)
    parser.add_argument("-upper_bounds", type=str, default=None)
    parser.add_argument("-lower_bounds", type=str, default=None)

    args = parser.parse_args()
    dataset = args.dataset
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print("RePaint-DDIM Synthesis")
    print(f"{'='*60}")
    print(f"[INFO] Device: {dev}")
    print(f"[INFO] Mode: {'DDPM' if args.use_ddpm else f'DDIM-{args.ddim_steps}'}")
    print(f"[INFO] RePaint: {args.repaint_rounds}, Guidance: {args.guidance_scale}")
    print(f"[INFO] Clamp: {args.clamp_mode}, Headroom: {args.bound_headroom}")

    scaler = None
    bounds = None

    # -----------------------------------------------------------------
    # 1) Load data
    # -----------------------------------------------------------------
    if dataset == "custom_csv":
        df_input, meta = load_custom_prepared(args.prepared_dir)
        time_col = meta.get("time_col", "time")
        cond_cols = meta.get("cond_cols", [])
        target_cols = meta.get("target_cols", [])
        if not target_cols:
            raise ValueError("target_cols empty")

        split_ratio = meta.get("split_ratio", 0.8)
        regularized_rows = meta.get("regularized_rows", None)
        train_rows = meta.get("train_rows", None)
        if train_rows is None:
            if regularized_rows is None:
                raise ValueError("meta.json missing train_rows and regularized_rows")
            train_rows = int(split_ratio * regularized_rows)

        # ORIGINAL observed mask from preprocessing
        orig_obs_mask_test = load_test_target_mask(
            args.prepared_dir,
            target_cols=target_cols,
            n_rows=len(df_input),
            train_rows=train_rows,
        )

        if orig_obs_mask_test is None:
            print("[WARNING] col_masks not available for test split; "
                  "falling back to input-derived mask as original mask.")
            orig_obs_mask_test = (~df_input[target_cols].isna()).to_numpy().astype(np.float32)

        # CURRENT visible mask from test_input.csv
        input_obs_mask_test = (~df_input[target_cols].isna()).to_numpy().astype(np.float32)

        # Missing entries to synthesize
        synth_mask_test = (input_obs_mask_test == 0)

        # Row-level mask for conditioning features from ORIGINAL observation support
        observed_row_mask = orig_obs_mask_test.any(axis=1)

        df_input = recompute_cond_features(
            df_input, time_col, cond_cols, target_cols, observed_row_mask
        )

        model_cols = [c for c in df_input.columns if c != time_col]

        print(f"[INFO] Rows: {len(df_input)}")
        print(f"[INFO] Original target observed rate: {orig_obs_mask_test.mean():.2%}")
        print(f"[INFO] Input-visible target rate:    {input_obs_mask_test.mean():.2%}")
        print(f"[INFO] Target missing-to-synthesize: {(1.0 - input_obs_mask_test.mean()):.2%}")

        df_synth = df_input.copy()
        df_synth = fill_targets_like_training(df_synth, target_cols)

        for c in cond_cols:
            if c in df_synth.columns:
                df_synth[c] = df_synth[c].fillna(0.0)

        expected_model_cols = meta.get("all_model_cols", model_cols)
        df_synth = df_synth[[time_col] + expected_model_cols]
        model_cols = expected_model_cols

        # Scaler
        scaler_dir = os.path.join(args.prepared_dir, "scaler")
        model_clip_bound = 3.0
        if os.path.exists(scaler_dir):
            try:
                scaler = FlexibleScaler.load(scaler_dir)
                model_clip_bound = scaler.get_model_clip_bound()
                print(f"[INFO] Scaler: {scaler.mode}/{scaler.clip_mode}, "
                      f"bound={model_clip_bound}")
            except Exception as e:
                print(f"[WARNING] Scaler load failed: {e}")

        target_indices = [model_cols.index(c) for c in target_cols if c in model_cols]
        d_vals = df_synth.drop(columns=[time_col]).values.astype(np.float32)

        if scaler is not None:
            d_vals[:, target_indices] = scaler.transform(d_vals[:, target_indices])

        hierarchical_column_indices = np.array(
            [model_cols.index(c) for c in cond_cols if c in model_cols],
            dtype=int
        )

        windows, window_starts = build_windows_from_array(
            d_vals, args.window_size, args.stride
        )
        synth_masks = build_windows_from_mask(
            synth_mask_test.astype(bool),
            args.window_size,
            args.stride,
            starts=window_starts,
        )
        obs_masks = build_windows_from_mask(
            input_obs_mask_test.astype(bool),
            args.window_size,
            args.stride,
            starts=window_starts,
        )

        # Bounds
        bounds = ValueBounds(target_cols)
        if args.clamp_mode == "bounds":
            bounds.load_from_preprocessing(scaler_dir, args.bound_headroom)
            bounds.apply_nonneg(args.nonneg_cols)
        elif args.clamp_mode == "nonneg":
            bounds.apply_nonneg(args.nonneg_cols)

        bounds.apply_manual_overrides(args.lower_bounds, args.upper_bounds)
        bounds.summary()

    else:
        # Legacy dataset branch
        preprocessor = Preprocessor(dataset, args.propCycEnc)
        df = preprocessor.df_cleaned
        end = preprocessor.test_indices[-1]
        start = preprocessor.test_indices[0]
        wc = ((end + 1 - args.window_size - start) // args.stride) + 1
        ts = end + 1 - args.window_size - (wc * args.stride)
        ai = start - ts

        test_df = df.loc[
            preprocessor.train_indices[-ai:] + preprocessor.test_indices
        ]
        tdh = preprocessor.cyclicDecode(test_df)
        md = tdh[preprocessor.hierarchical_features_uncyclic]
        rows_to_synth = metadataMask(md, args.synth_mask, args.dataset)

        df_synth = test_df.copy()
        hierarchical_column_indices = df_synth.columns.get_indexer(
            preprocessor.hierarchical_features_cyclic
        )
        dv = df_synth.values.astype(np.float32)

        model_cols = df_synth.columns.tolist()
        target_indices = list(range(dv.shape[1] - len(hierarchical_column_indices)))
        target_cols = list(df_synth.columns[:len(target_indices)])
        model_clip_bound = 3.0

        windows, window_starts = build_windows_from_array(
            dv, args.window_size, args.stride
        )

        row_mask_2d = np.repeat(rows_to_synth.values.astype(bool)[:, None],
                                len(target_indices), axis=1)
        obs_mask_2d = ~row_mask_2d

        synth_masks = build_windows_from_mask(
            row_mask_2d, args.window_size, args.stride, starts=window_starts
        )
        obs_masks = build_windows_from_mask(
            obs_mask_2d, args.window_size, args.stride, starts=window_starts
        )

        df_input = test_df.copy()
        cond_cols = []

    # -----------------------------------------------------------------
    # 2) Model
    # -----------------------------------------------------------------
    in_dim = windows.shape[2]
    out_dim = in_dim - len(hierarchical_column_indices)

    test_dl = DataLoader(
        MyDataset(windows.float(), window_size=args.window_size),
        batch_size=args.batch_size
    )
    synth_mask_dl = DataLoader(MyDataset(synth_masks), batch_size=args.batch_size)
    obs_mask_dl = DataLoader(MyDataset(obs_masks), batch_size=args.batch_size)

    model = fetchModel(in_dim, out_dim, args).to(dev)
    diffusion_config = fetchDiffusionConfig(args)

    non_hier_cols = np.setdiff1d(np.arange(in_dim), hierarchical_column_indices)

    saved_dir = get_save_dir(args.prepared_dir)
    if args.model_type == "em":
        mp = os.path.join(saved_dir, "model_em_best.pth")
        if not os.path.exists(mp):
            mp = os.path.join(saved_dir, "model_em.pth")
    elif args.model_type == "standard":
        mp = os.path.join(saved_dir,
                          "model_prop.pth" if args.propCycEnc else "model.pth")
    else:
        saved_params, mp = load_model_with_fallback(saved_dir, args, dev)

    if args.model_type != "auto":
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Not found: {mp}")
        saved_params = torch.load(mp, map_location=dev)
        print(f"[INFO] Loading: {mp}")

    model_path = mp

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in saved_params:
                param.copy_(saved_params[name])

    model.eval()

    # -----------------------------------------------------------------
    # 3) Synthesizer
    # -----------------------------------------------------------------
    synth = RePaintDDIMSynthesizer(
        model=model,
        diffusion_config=diffusion_config,
        device=dev,
        non_hier_cols=non_hier_cols,
        hier_cols=hierarchical_column_indices,
        target_indices=target_indices,
        window_size=args.window_size,
        stride=args.stride,
        repaint_rounds=args.repaint_rounds,
        ddim_steps=args.ddim_steps,
        use_ddim=not args.use_ddpm,
        guidance_scale=args.guidance_scale,
        clip_bound=model_clip_bound,
    )

    exec_times = []

    # -----------------------------------------------------------------
    # 4) Synthesis
    # -----------------------------------------------------------------
    for trial in range(args.n_trials):
        t0 = timer()
        window_outputs = []

        for idx, (tb, smb, omb) in enumerate(zip(test_dl, synth_mask_dl, obs_mask_dl)):
            tb = ensure_bwc(tb, args.window_size).to(dev)
            smb = smb.to(dev).bool()
            omb = omb.to(dev).bool()

            print(f"  [T{trial}] Batch {idx+1} {tb.shape}")

            x = synth.synthesize_batch(tb, smb, omb)
            window_outputs.append(x.detach().cpu().numpy())

        elapsed = timer() - t0
        exec_times.append(elapsed)

        win_arr = np.concatenate(window_outputs, axis=0)
        seq_len = len(df_input)
        arr = hann_overlap_add(
            win_arr,
            seq_len,
            args.window_size,
            args.stride,
            starts=window_starts,
        )

        # ── Output ──────────────────────────────────────────────────
        if dataset == "custom_csv":
            if scaler is not None:
                arr[:, target_indices] = scaler.inverse_transform(arr[:, target_indices])

            if bounds is not None and args.clamp_mode != "none":
                diag = bounds.clamp(arr, target_indices)
                print(f"[INFO] Clamped: {diag['n_clamped_lower']} below, "
                      f"{diag['n_clamped_upper']} above")

            synth_df = pd.DataFrame(arr, columns=model_cols)

            final_df = df_input[[time_col]].copy()
            for c in target_cols:
                if c in synth_df.columns:
                    final_df[c] = synth_df[c].to_numpy()[:len(final_df)]
            for c in cond_cols:
                if c in df_input.columns:
                    final_df[c] = df_input[c].to_numpy()[:len(final_df)]

            out_dir = (
                f"{get_generated_dir(args.prepared_dir)}"
                f"/{args.synth_mask}_wavestitchPlus/"
            )
            os.makedirs(out_dir, exist_ok=True)

            mtag = "em" if "em" in model_path.lower() else "std"
            mode = "ddpm" if args.use_ddpm else f"ddim{args.ddim_steps}"
            out_name = os.path.join(
                out_dir,
                f"imputed_{mtag}_{mode}_rp{args.repaint_rounds}"
                f"_{args.clamp_mode}"
                f"_stride_{args.stride}_trial_{trial}.csv"
            )
            final_df.to_csv(out_name, index=False)

            if args.out_csv and trial == 0:
                os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
                final_df.to_csv(args.out_csv, index=False)

            print(f"  [T{trial}] {elapsed:.2f}s → {out_name}")

    # -----------------------------------------------------------------
    # 5) Timing
    # -----------------------------------------------------------------
    out_dir = (
        f"{get_generated_dir(args.prepared_dir)}"
        f"/{args.synth_mask}_wavestitchPlus/"
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"timing_stride_{args.stride}.txt"), "a") as f:
        a = np.array(exec_times)
        f.write(f"\n{args.clamp_mode} headroom={args.bound_headroom} "
                f"| {np.mean(a):.2f}s ± {np.std(a):.2f}s\n")

    print(f"\n[DONE] {np.mean(exec_times):.2f}s ± {np.std(exec_times):.2f}s")
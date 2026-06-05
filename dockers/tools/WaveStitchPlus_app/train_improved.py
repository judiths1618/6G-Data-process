from __future__ import annotations
import argparse
import os
import json

import numpy as np
import pandas as pd
import torch
from torch import optim, nn, randint, normal, sqrt
from torch.utils.data import DataLoader

from helper.data_utils import Preprocessor
from helper.training_utils import MyDataset, fetchModel, fetchDiffusionConfig
from custom_pipeline.directory_manager import get_save_dir


# =============================================================================
# Helpers
# =============================================================================

SOFT_CLIP_BOUND = 4.0  # default; overridden by meta / CLI


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def soft_clip(x: np.ndarray, bound: float = SOFT_CLIP_BOUND) -> np.ndarray:
    """Smooth tanh saturation. Invertible approximately via soft_clip_inv."""
    return bound * np.tanh(x / bound)


def soft_clip_inv(x: np.ndarray, bound: float = SOFT_CLIP_BOUND) -> np.ndarray:
    """Inverse of soft_clip (arctanh) with safety clipping."""
    safe = np.clip(x / bound, -0.9999, 0.9999)
    return bound * np.arctanh(safe)


def soft_clip_torch(x: torch.Tensor, bound: float = SOFT_CLIP_BOUND) -> torch.Tensor:
    """Torch version of soft_clip."""
    return bound * torch.tanh(x / bound)


# =============================================================================
# Normalization
# =============================================================================

# IQR / std for a Gaussian (0.6745 * 2). Dividing IQR by this constant yields
# a robust estimator of std, so robust-mode post-scaled data has unit variance
# instead of ~0.74 — fixes the per-feature variance miscalibration that was
# making the latency columns collapse and the near-constant columns over-inflate
# at inference time.
IQR_TO_STD_GAUSSIAN = 1.349


class FlexibleScaler:
    """
    Scaler that supports both standard (mean/std) and robust (median/IQR)
    modes, and both hard clip and soft clip.

    Robust mode uses ``scale = IQR / 1.349`` (a robust std estimator) so the
    post-scaled values have unit variance under near-Gaussian assumptions —
    matching what diffusion architectures expect from their noise schedule.
    """

    def __init__(self, mode="standard", clip_mode="hard", clip_bound=3.0,
                 soft_bound=SOFT_CLIP_BOUND):
        self.mode = mode
        self.clip_mode = clip_mode
        self.clip_bound = clip_bound
        self.soft_bound = soft_bound

        self.center_ = None   # mean or median
        self.scale_ = None    # std or IQR

    def fit_from_arrays(self, center: np.ndarray, scale: np.ndarray):
        self.center_ = center.copy().astype(np.float32)
        self.scale_ = scale.copy().astype(np.float32)
        self.scale_[self.scale_ < 1e-8] = 1.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.center_) / self.scale_
        if self.clip_mode == "soft":
            z = soft_clip(z, self.soft_bound)
        else:
            z = np.clip(z, -self.clip_bound, self.clip_bound)
        return z

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.clip_mode == "soft":
            z = soft_clip_inv(z, self.soft_bound)
        return z * self.scale_ + self.center_

    def get_model_clip_bound(self) -> float:
        if self.clip_mode == "soft":
            return self.soft_bound
        return self.clip_bound

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "center.npy"), self.center_)
        np.save(os.path.join(directory, "scale.npy"), self.scale_)

        info = {
            "mode": self.mode,
            "clip_mode": self.clip_mode,
            "clip_bound": self.clip_bound,
            "soft_bound": self.soft_bound,
        }
        with open(os.path.join(directory, "scaler_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        # Backward-compat aliases
        np.save(os.path.join(directory, "mean.npy"), self.center_)
        np.save(os.path.join(directory, "std.npy"), self.scale_)

    @classmethod
    def load(cls, directory: str) -> "FlexibleScaler":
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

        center_path = os.path.join(directory, "center.npy")
        scale_path = os.path.join(directory, "scale.npy")

        if os.path.exists(center_path):
            scaler.center_ = np.load(center_path).astype(np.float32)
            scaler.scale_ = np.load(scale_path).astype(np.float32)
        else:
            scaler.center_ = np.load(os.path.join(directory, "mean.npy")).astype(np.float32)
            scaler.scale_ = np.load(os.path.join(directory, "std.npy")).astype(np.float32)

        scaler.scale_[scaler.scale_ < 1e-8] = 1.0
        return scaler


# =============================================================================
# Value bounds (for denormalized train_imputed post-processing)
# =============================================================================

class ValueBounds:
    """
    Per-column lower/upper bounds for post-processing clamp.

    Upper bound source:
        p99.5 with optional headroom toward observed_max.

    Lower bound source:
        p0.5, optionally overridden by nonneg.
    """

    def __init__(self, target_cols: list[str]):
        self.target_cols = target_cols
        n = len(target_cols)
        self.lower = np.full(n, -np.inf, dtype=np.float32)
        self.upper = np.full(n, np.inf, dtype=np.float32)

    def load_from_preprocessing(self, scaler_dir: str, headroom: float = 1.2):
        upper_path = os.path.join(scaler_dir, "upper_bound_p995.npy")
        lower_path = os.path.join(scaler_dir, "lower_bound_p005.npy")
        obs_max_path = os.path.join(scaler_dir, "observed_max.npy")

        if os.path.exists(upper_path):
            p995 = np.load(upper_path).astype(np.float32)
            if os.path.exists(obs_max_path):
                obs_max = np.load(obs_max_path).astype(np.float32)
            else:
                obs_max = p995.copy()

            # Allow slight extrapolation above p99.5 toward observed max
            self.upper = p995 + headroom * np.maximum(obs_max - p995, 0.0)

        if os.path.exists(lower_path):
            p005 = np.load(lower_path).astype(np.float32)
            self.lower = p005

    def apply_nonneg(self, cols=None):
        if cols is None:
            self.lower = np.maximum(self.lower, 0.0)
        else:
            for c in cols:
                if c in self.target_cols:
                    idx = self.target_cols.index(c)
                    self.lower[idx] = max(self.lower[idx], 0.0)

    def clamp_array(self, arr: np.ndarray, target_indices: list[int]) -> dict:
        """
        Clamp arr[:, target_indices] in-place.
        Returns diagnostics.
        """
        n_lo = 0
        n_hi = 0

        for i, col_idx in enumerate(target_indices):
            lo = self.lower[i]
            hi = self.upper[i]

            col = arr[:, col_idx]
            below = col < lo
            above = col > hi

            n_lo += int(below.sum())
            n_hi += int(above.sum())

            arr[:, col_idx] = np.clip(col, lo, hi)

        return {
            "n_clamped_lower": n_lo,
            "n_clamped_upper": n_hi,
        }

    def summary(self, max_cols: int = 10):
        print("[Bounds] Per-column bounds:")
        for i, c in enumerate(self.target_cols[:max_cols]):
            lo = f"{self.lower[i]:.4g}" if np.isfinite(self.lower[i]) else "-∞"
            hi = f"{self.upper[i]:.4g}" if np.isfinite(self.upper[i]) else "+∞"
            print(f"    {c}: [{lo}, {hi}]")
        if len(self.target_cols) > max_cols:
            print(f"    ... ({len(self.target_cols) - max_cols} more)")


# =============================================================================
# Data loading
# =============================================================================

def load_custom_train_df(prepared_dir: str):
    """
    Load training data and observation masks.

    Uses per-column masks from preprocessing v2 (col_masks/) when
    available, falling back to row-level NaN detection.
    """
    meta_path = os.path.join(prepared_dir, "meta.json")
    train_path = os.path.join(prepared_dir, "train.csv")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    time_col = meta.get("time_col", "time")
    cond_cols = meta.get("cond_cols", [])
    target_cols = meta.get("target_cols", [])
    model_cols = meta.get("all_model_cols", None)

    df = pd.read_csv(train_path)

    if model_cols is not None:
        expected = [time_col] + model_cols
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"train.csv missing expected columns: {missing}")
        df = df[expected]

    training_df = df.drop(columns=[time_col])

    # ── Observation mask ────────────────────────────────────────────
    col_mask_dir = os.path.join(prepared_dir, "col_masks")
    n_rows = len(training_df)

    if os.path.isdir(col_mask_dir):
        print("[INFO] Loading per-column observed masks from col_masks/")
        mask_arrays = []
        for c in target_cols:
            mask_path = os.path.join(col_mask_dir, f"{c}.npy")
            if os.path.exists(mask_path):
                full_mask = np.load(mask_path).astype(np.float32)
                mask_arrays.append(full_mask[:n_rows])
            else:
                mask_arrays.append(
                    (~training_df[c].isna()).to_numpy().astype(np.float32)
                )

        obs_mask = pd.DataFrame(
            np.column_stack(mask_arrays), columns=target_cols
        )
    else:
        print("[INFO] col_masks/ not found, deriving mask from NaN")
        obs_mask = (~training_df[target_cols].isna()).astype(np.float32)

    # ── Initial fill ────────────────────────────────────────────────
    training_df_filled = training_df.copy()

    for col in target_cols:
        if col in training_df_filled.columns:
            training_df_filled[col] = (
                training_df_filled[col]
                .interpolate(method="linear", limit_direction="both")
                .ffill().bfill()
            )
            if training_df_filled[col].isna().any():
                col_mean = training_df[col].mean()
                fill_val = col_mean if not np.isnan(col_mean) else 0.0
                training_df_filled[col] = training_df_filled[col].fillna(fill_val)

    for col in cond_cols:
        if col in training_df_filled.columns:
            training_df_filled[col] = training_df_filled[col].fillna(0.0)

    remaining = training_df_filled.isna().sum().sum()
    if remaining > 0:
        print(f"[WARNING] {remaining} NaN remaining → filling with 0")
        training_df_filled = training_df_filled.fillna(0.0)

    assert list(obs_mask.columns) == target_cols
    print(f"[INFO] training_df: {training_df_filled.shape}, "
          f"obs_mask: {obs_mask.shape}")

    return training_df_filled, cond_cols, target_cols, obs_mask, meta


# =============================================================================
# EM Dataset
# =============================================================================

class EMDataset(torch.utils.data.Dataset):
    def __init__(self, samples, masks):
        self.samples = samples
        self.masks = masks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.masks[idx]


# =============================================================================
# EM Trainer (RePaint + annealed weights + Hann E-step)
# =============================================================================

class DiffPuterEMTrainer:

    def __init__(self, model, diffusion_config, device,
                 non_hier_cols, hier_cols, target_indices,
                 lr=1e-3, em_iterations=5, repaint_rounds=3,
                 clip_bound=3.0):
        self.model = model
        self.diffusion_config = diffusion_config
        self.device = device
        self.non_hier_cols = np.array(non_hier_cols)
        self.hier_cols = np.array(hier_cols)
        self.target_indices = np.array(target_indices)
        self.lr = lr
        self.em_iterations = em_iterations
        self.repaint_rounds = repaint_rounds
        self.clip_bound = clip_bound

        self.alpha_bars = diffusion_config["alpha_bars"].to(device)
        self.betas = diffusion_config["betas"].to(device)
        self.alphas = 1 - self.betas
        self.T = diffusion_config["T"]

        print(f"[Trainer] target={list(self.non_hier_cols)}, "
              f"cond={list(self.hier_cols)}, T={self.T}, "
              f"repaint={self.repaint_rounds}, clip={self.clip_bound}")

    # ── M-step ──────────────────────────────────────────────────────

    def m_step(self, dataloader, epochs_per_iter, missing_weight):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20
        )
        best_loss = float("inf")

        for epoch in range(epochs_per_iter):
            total_loss = 0.0
            n_batches = 0

            for batch, mask in dataloader:
                batch = batch.to(self.device)
                mask = mask.to(self.device)

                timesteps = torch.randint(
                    self.T, size=(batch.shape[0],), device=self.device
                )
                sigmas = torch.randn_like(batch)

                c1 = torch.sqrt(self.alpha_bars[timesteps]).reshape(-1, 1, 1)
                c2 = torch.sqrt(1 - self.alpha_bars[timesteps]).reshape(-1, 1, 1)

                cond_mask = torch.ones_like(batch, device=self.device)
                cond_mask[:, :, self.non_hier_cols] = 0.0

                batch_noised = (
                    (1 - cond_mask) * (c1 * batch + c2 * sigmas)
                    + cond_mask * batch
                )

                times = timesteps.reshape(-1, 1)
                pred = self.model(batch_noised, times)

                optimizer.zero_grad()

                gt = sigmas[:, :, self.non_hier_cols].permute(0, 2, 1)
                mask_p = mask.permute(0, 2, 1)

                elem_loss = (pred - gt) ** 2
                weights = missing_weight + (1.0 - missing_weight) * mask_p
                loss = (elem_loss * weights).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                del batch_noised, pred, gt, elem_loss

            avg = total_loss / max(1, n_batches)
            scheduler.step(avg)
            if avg < best_loss:
                best_loss = avg

            if epoch % 50 == 0:
                print(f"    M-step {epoch:3d}: loss={avg:.6f}, "
                      f"best={best_loss:.6f}, w_miss={missing_weight:.3f}")

            if epoch % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return best_loss

    # ── E-step ──────────────────────────────────────────────────────

    @torch.no_grad()
    def e_step_fast(self, data_np, obs_mask_np, window_size, stride,
                    batch_size=32, num_samples=1, ddim_steps=50):
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        T_seq, C = data_np.shape
        n_tgt = len(self.target_indices)
        assert obs_mask_np.shape == (T_seq, n_tgt)

        if T_seq < window_size:
            raise ValueError(
                f"T_seq ({T_seq}) < window_size ({window_size}) in E-step."
            )

        num_win = (T_seq - window_size) // stride + 1
        print(f"    [E-step] {num_win} windows, batch={batch_size}, "
              f"ddim={ddim_steps}, repaint={self.repaint_rounds}, "
              f"samples={num_samples}")

        hann = np.hanning(window_size).reshape(-1, 1).astype(np.float32)
        imp_sum = np.zeros((T_seq, C), dtype=np.float32)
        imp_wt = np.zeros((T_seq, 1), dtype=np.float32)

        n_bat = (num_win + batch_size - 1) // batch_size

        for si in range(num_samples):
            if num_samples > 1:
                print(f"      Sample {si + 1}/{num_samples}")

            for bi in range(n_bat):
                bs = bi * batch_size
                be = min(bs + batch_size, num_win)
                actual = be - bs

                if bi % 50 == 0:
                    mem = (
                        torch.cuda.memory_allocated() / 1024**3
                        if torch.cuda.is_available() else 0
                    )
                    print(f"      Batch {bi + 1}/{n_bat}, GPU: {mem:.2f} GB")

                bd = np.zeros((actual, window_size, C), dtype=np.float32)
                bm = np.zeros((actual, window_size, n_tgt), dtype=np.float32)

                for i in range(actual):
                    s = (bs + i) * stride
                    e = s + window_size
                    bd[i] = data_np[s:e]
                    bm[i] = obs_mask_np[s:e]

                bd_t = torch.from_numpy(bd).to(self.device)
                bm_t = torch.from_numpy(bm).to(self.device)

                samp = self._conditional_sample_ddim(
                    bd_t, bm_t, ddim_steps, self.repaint_rounds
                ).cpu().numpy()

                del bd_t, bm_t
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                for i in range(actual):
                    s = (bs + i) * stride
                    e = s + window_size
                    imp_sum[s:e] += samp[i] * hann
                    imp_wt[s:e] += hann

        imp_wt = np.maximum(imp_wt, 1e-8)
        imputed = imp_sum / imp_wt

        new_data = data_np.copy()
        for ci, tc in enumerate(self.target_indices):
            miss = obs_mask_np[:, ci] == 0
            new_data[miss, tc] = imputed[miss, tc]

        self.model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return new_data

    # ── RePaint DDIM sampler ────────────────────────────────────────

    def _conditional_sample_ddim(self, batch_data, batch_mask,
                                 ddim_steps=50, repaint_rounds=3):
        B, T, C = batch_data.shape
        x_t = torch.randn_like(batch_data)

        cond_mask = torch.zeros_like(batch_data, device=self.device)
        cond_mask[:, :, self.hier_cols] = 1.0

        obs_full = torch.zeros_like(batch_data, device=self.device)
        obs_full[:, :, self.target_indices] = batch_mask

        known = torch.clamp(cond_mask + obs_full, 0.0, 1.0)

        ts = np.linspace(0, self.T - 1, ddim_steps, dtype=int)[::-1]

        for i, t in enumerate(ts):
            ab_t = self.alpha_bars[t]
            if i < len(ts) - 1:
                t_nxt = ts[i + 1]
                ab_nxt = self.alpha_bars[t_nxt]
            else:
                t_nxt = None
                ab_nxt = None

            for r in range(repaint_rounds):
                tv = torch.full((B,), t, device=self.device, dtype=torch.long)
                x_in = cond_mask * batch_data + (1 - cond_mask) * x_t

                np_ = self.model(x_in, tv.reshape(-1, 1)).permute(0, 2, 1)
                nf = torch.zeros_like(x_t)
                nf[:, :, self.non_hier_cols] = np_

                x0 = (x_t - torch.sqrt(1 - ab_t) * nf) / torch.sqrt(ab_t)
                x0 = torch.clamp(x0, -self.clip_bound, self.clip_bound)

                if t_nxt is not None:
                    unk = torch.sqrt(ab_nxt) * x0 + torch.sqrt(1 - ab_nxt) * nf
                else:
                    unk = x0

                if t_nxt is not None:
                    kn_noise = torch.randn_like(batch_data)
                    kn = (torch.sqrt(ab_nxt) * batch_data
                          + torch.sqrt(1 - ab_nxt) * kn_noise)
                else:
                    kn = batch_data

                merged = known * kn + (1 - known) * unk

                if r < repaint_rounds - 1:
                    rn = torch.randn_like(merged)
                    if t_nxt is not None:
                        a_s = ab_t / ab_nxt
                        x_t = torch.sqrt(a_s) * merged + torch.sqrt(1 - a_s) * rn
                    else:
                        x_t = (torch.sqrt(ab_t) * merged
                               + torch.sqrt(1 - ab_t) * rn)
                    del rn
                else:
                    x_t = merged

                del np_, nf, x0, unk, merged

        x_t = obs_full * batch_data + (1 - obs_full) * x_t
        x_t = cond_mask * batch_data + (1 - cond_mask) * x_t
        return x_t

    # ── DataLoader builder ──────────────────────────────────────────

    def _build_dataloader(self, data_np, obs_mask_np, window_size,
                          stride, batch_size):
        T_seq, C = data_np.shape
        n_tgt = obs_mask_np.shape[1]

        if T_seq < window_size:
            raise ValueError(
                f"T_seq ({T_seq}) < window_size ({window_size}) for training."
            )

        nw = (T_seq - window_size) // stride + 1

        samples = torch.zeros(nw, window_size, C, dtype=torch.float32)
        masks = torch.zeros(nw, window_size, n_tgt, dtype=torch.float32)

        for i in range(nw):
            s = i * stride
            e = s + window_size
            samples[i] = torch.from_numpy(data_np[s:e])
            masks[i] = torch.from_numpy(obs_mask_np[s:e])

        return DataLoader(
            EMDataset(samples, masks),
            batch_size=batch_size, shuffle=True, drop_last=True,
        )

    # ── EM loop ─────────────────────────────────────────────────────

    def train_em(self, data_np, obs_mask_np, window_size, stride,
                 batch_size, epochs_per_iter=200,
                 e_step_batch_size=32, e_step_samples=1,
                 ddim_steps=50, save_dir=None, model_base="model_em"):
        current = data_np.copy()
        best_loss = float("inf")

        for it in range(self.em_iterations):
            print(f"\n{'='*60}")
            print(f"EM {it + 1}/{self.em_iterations}")
            print(f"{'='*60}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Anneal: start near 0 (trust observed), ramp up
            mw = 0.05 + 0.25 * (it / max(1, self.em_iterations - 1))

            dl = self._build_dataloader(
                current, obs_mask_np, window_size, stride, batch_size
            )
            print(f"  [M-step] {epochs_per_iter} epochs, w_miss={mw:.3f}")
            ml = self.m_step(dl, epochs_per_iter, mw)
            print(f"  [M-step] Done, loss={ml:.6f}")

            if ml < best_loss and save_dir:
                best_loss = ml
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(save_dir, f"{model_base}_best.pth"),
                )
                print(f"  [SAVE] Best (loss={best_loss:.6f})")

            if it < self.em_iterations - 1:
                print(f"  [E-step] repaint={self.repaint_rounds}")
                old = current.copy()
                current = self.e_step_fast(
                    current, obs_mask_np, window_size, stride,
                    batch_size=e_step_batch_size,
                    num_samples=e_step_samples,
                    ddim_steps=ddim_steps,
                )
                mm = obs_mask_np == 0
                if mm.sum() > 0:
                    d = np.abs(
                        current[:, self.target_indices][mm]
                        - old[:, self.target_indices][mm]
                    )
                    print(f"  [E-step] Δ mean={d.mean():.4f}, "
                          f"max={d.max():.4f}")

            if save_dir and (it + 1) % 2 == 0:
                p = os.path.join(save_dir, f"{model_base}_iter{it + 1}.pth")
                torch.save(self.model.state_dict(), p)

        return current


# =============================================================================
# Validation
# =============================================================================

def validate_data(d_vals, obs_mask_np, target_indices, hier_cols):
    print(f"\n[VALIDATION]")
    print(f"  d_vals:   {d_vals.shape}")
    print(f"  obs_mask: {obs_mask_np.shape}")
    assert obs_mask_np.shape[1] == len(target_indices)
    assert np.all(np.isin(np.unique(obs_mask_np), [0, 1]))
    print(f"  Obs rate: {obs_mask_np.mean():.2%}")
    assert np.isnan(d_vals).sum() == 0
    print(f"[VALIDATION] ✓")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", "-d", type=str, required=True)
    parser.add_argument("-input_csv", type=str, default=None)
    parser.add_argument("-prepared_dir", type=str, default="./work/prepared")

    # Model
    parser.add_argument("-backbone", type=str, default="S4")
    parser.add_argument("-beta_0", type=float, default=0.0001)
    parser.add_argument("-beta_T", type=float, default=0.02)
    parser.add_argument("-timesteps", "-T", type=int, default=200)
    parser.add_argument("-hdim", type=int, default=64)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-batch_size", type=int, default=1024)
    parser.add_argument("-epochs", type=int, default=1000)
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
    parser.add_argument("-normalize", type=str2bool, default=True)

    # EM
    parser.add_argument("-em_iterations", type=int, default=5)
    parser.add_argument("-epochs_per_em", type=int, default=200)
    parser.add_argument("-use_em", action="store_true")

    # E-step
    parser.add_argument("-e_step_batch_size", type=int, default=126)
    parser.add_argument("-ddim_steps", type=int, default=50)
    parser.add_argument("-e_step_samples", type=int, default=1)

    # RePaint
    parser.add_argument("-repaint_rounds", type=int, default=3)

    # Normalization
    parser.add_argument("-scaler_mode", type=str, default="auto",
                        choices=["auto", "standard", "robust"])
    parser.add_argument("-clip_mode", type=str, default="auto",
                        choices=["auto", "hard", "soft"])
    parser.add_argument("-clip_bound", type=float, default=3.0)
    parser.add_argument("-soft_bound", type=float, default=4.0)

    # Save denormalized / bounded train_imputed
    parser.add_argument("-save_train_imputed_denorm", action="store_true",
                        help="Save EM-imputed training data in original scale.")
    parser.add_argument("-train_imputed_clamp", type=str, default="bounds",
                        choices=["none", "nonneg", "bounds"],
                        help="Post-process denormalized train_imputed.")
    parser.add_argument("-bound_headroom", type=float, default=1.2,
                        help="Headroom above p99.5 toward observed_max.")
    parser.add_argument("-nonneg_cols", type=str, nargs="*", default=None,
                        help="Apply non-negative lower bound only to these "
                             "target columns. Default=None means all targets.")

    parser.add_argument("-run_id", type=str, default=None)
    parser.add_argument("-model_name", type=str, default="wavestitchplus")
    parser.add_argument("-model_filename", type=str, default="model.pth")
    parser.add_argument("-model_tag", type=str, default="",
                        help="if set, checkpoints are saved as model_<tag>*.pth "
                             "so different run methods don't overwrite each other "
                             "(empty = historical model_em*/model* names)")

    args = parser.parse_args()
    dataset = args.dataset
    prepared_dir = args.prepared_dir or "./work/prepared"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repaint_rounds = max(1, args.repaint_rounds)

    scaler_mode = "standard"
    clip_mode = "hard"
    model_clip_bound = args.clip_bound
    scaler = None
    bounds = None

    print(f"\n{'='*60}")
    print(f"DiffPuter Training  (RePaint + FlexibleScaler)")
    print(f"{'='*60}")
    print(f"[INFO] Device: {dev}, EM: {args.use_em}, RePaint: {repaint_rounds}")

    # ── Load data ───────────────────────────────────────────────────
    if dataset == "custom_csv":
        from custom_pipeline.preprocess import preprocess_csv

        if args.input_csv:
            preprocess_csv(
                input_csv=args.input_csv,
                output_dir=prepared_dir,
                time_col=None,
                base_dt=None,
                extract_main_segment=True,
                skip_regularize_if_sparse=True,
                convert_units=True,
            )

        training_df, cond_cols, target_cols, obs_mask, meta = \
            load_custom_train_df(prepared_dir)

        print(f"\n[INFO] Data: {training_df.shape}")
        print(f"  Targets ({len(target_cols)}): {target_cols}")
        print(f"  Conds   ({len(cond_cols)}):   {cond_cols}")

        d_vals = training_df.values.astype(np.float32)
        obs_mask_np = obs_mask.values.astype(np.float32)

        hierarchical_column_indices = training_df.columns.get_indexer(cond_cols)
        target_indices = [
            training_df.columns.get_loc(c) for c in target_cols
            if c in training_df.columns
        ]

        # ── Resolve scaler & clip modes from preprocessing ──────────
        clip_rec = meta.get("clip_recommendation", "hard_clip")

        if args.clip_mode == "auto":
            clip_mode = "soft" if clip_rec == "soft_clip" else "hard"
        else:
            clip_mode = args.clip_mode

        if args.scaler_mode == "auto":
            scaler_mode = "robust" if clip_mode == "soft" else "standard"
        else:
            scaler_mode = args.scaler_mode

        print(f"[INFO] Scaler: {scaler_mode}, Clip: {clip_mode} "
              f"(rec='{clip_rec}')")

        # ── Normalize ───────────────────────────────────────────────
        if args.normalize:
            scaler_dir = os.path.join(prepared_dir, "scaler")
            stats_path = os.path.join(scaler_dir, "stats.json")

            scaler = FlexibleScaler(
                mode=scaler_mode,
                clip_mode=clip_mode,
                clip_bound=args.clip_bound,
                soft_bound=args.soft_bound,
            )

            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    stats = json.load(f)

                if scaler_mode == "robust":
                    center = np.array(
                        [stats[c]["median"] for c in target_cols],
                        dtype=np.float32
                    )
                    # Robust std estimator: IQR / 1.349 (Gaussian-equivalent).
                    # Prefer a pre-computed robust_std if preprocessing supplied
                    # one; otherwise derive it from the raw IQR on the fly.
                    scale = np.array(
                        [stats[c].get("robust_std", stats[c]["iqr"] / IQR_TO_STD_GAUSSIAN)
                         for c in target_cols],
                        dtype=np.float32
                    )
                else:
                    center = np.array(
                        [stats[c]["mean"] for c in target_cols],
                        dtype=np.float32
                    )
                    scale = np.array(
                        [stats[c]["std"] for c in target_cols],
                        dtype=np.float32
                    )
            else:
                observed = d_vals[:, target_indices].copy()
                observed[obs_mask_np == 0] = np.nan

                if scaler_mode == "robust":
                    center = np.nanmedian(observed, axis=0).astype(np.float32)
                    q1 = np.nanpercentile(observed, 25, axis=0)
                    q3 = np.nanpercentile(observed, 75, axis=0)
                    # IQR/1.349 → unit-variance robust scaling.
                    scale = ((q3 - q1) / IQR_TO_STD_GAUSSIAN).astype(np.float32)
                else:
                    center = np.nanmean(observed, axis=0).astype(np.float32)
                    scale = np.nanstd(observed, axis=0).astype(np.float32)

            scaler.fit_from_arrays(center, scale)
            d_vals[:, target_indices] = scaler.transform(d_vals[:, target_indices])
            scaler.save(os.path.join(prepared_dir, "scaler"))

            model_clip_bound = scaler.get_model_clip_bound()
            print(f"  Center[:3]: {scaler.center_[:3]}")
            print(f"  Scale[:3]:  {scaler.scale_[:3]}")
            print(f"  Model clip: {model_clip_bound}")

        # ── Bounds for denormalized train_imputed ───────────────────
        scaler_dir = os.path.join(prepared_dir, "scaler")
        bounds = ValueBounds(target_cols)

        if args.train_imputed_clamp == "bounds":
            bounds.load_from_preprocessing(
                scaler_dir, headroom=args.bound_headroom
            )
            bounds.apply_nonneg(args.nonneg_cols)
        elif args.train_imputed_clamp == "nonneg":
            bounds.apply_nonneg(args.nonneg_cols)

        bounds.summary()

    else:
        preprocessor = Preprocessor(dataset, args.propCycEnc)
        df = preprocessor.df_cleaned
        training_df = df.loc[preprocessor.train_indices]

        hierarchical_column_indices = training_df.columns.get_indexer(
            preprocessor.hierarchical_features_cyclic
        )
        d_vals = training_df.values.astype(np.float32)
        target_indices = list(
            range(d_vals.shape[1] - len(hierarchical_column_indices))
        )
        obs_mask_np = np.ones(
            (d_vals.shape[0], len(target_indices)), dtype=np.float32
        )

        cond_cols = []
        target_cols = list(training_df.columns[:len(target_indices)])

    # ── Model ───────────────────────────────────────────────────────
    in_dim = d_vals.shape[1]
    out_dim = in_dim - len(hierarchical_column_indices)
    print(f"\n[INFO] Model: in={in_dim}, out={out_dim}")

    model = fetchModel(in_dim, out_dim, args).to(dev)
    diffusion_config = fetchDiffusionConfig(args)
    all_indices = np.arange(in_dim)
    non_hier_cols = np.setdiff1d(all_indices, hierarchical_column_indices)

    validate_data(d_vals, obs_mask_np, target_indices, hierarchical_column_indices)

    save_dir = get_save_dir(prepared_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Checkpoint base name. With -model_tag (the run method, e.g. full/em/
    # standard) the saved files become model_<tag>*.pth so different methods
    # don't overwrite each other; without it the historical names are kept.
    em_base = f"model_{args.model_tag}" if args.model_tag else "model_em"
    std_base = f"model_{args.model_tag}" if args.model_tag else "model"

    # ── Training ────────────────────────────────────────────────────
    if args.use_em:
        print(f"\n{'='*60}")
        print(f"EM Training (repaint={repaint_rounds}, clip={model_clip_bound})")
        print(f"{'='*60}")

        trainer = DiffPuterEMTrainer(
            model=model,
            diffusion_config=diffusion_config,
            device=dev,
            non_hier_cols=non_hier_cols,
            hier_cols=hierarchical_column_indices,
            target_indices=target_indices,
            lr=args.lr,
            em_iterations=args.em_iterations,
            repaint_rounds=repaint_rounds,
            clip_bound=model_clip_bound,
        )

        final_data = trainer.train_em(
            data_np=d_vals,
            obs_mask_np=obs_mask_np,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            epochs_per_iter=args.epochs_per_em,
            e_step_batch_size=args.e_step_batch_size,
            e_step_samples=args.e_step_samples,
            ddim_steps=args.ddim_steps,
            save_dir=save_dir,
            model_base=em_base,
        )

        # Save normalized-space EM result
        np.save(os.path.join(prepared_dir, "train_imputed.npy"), final_data)
        print("[INFO] Saved normalized EM output: train_imputed.npy")

        # Optional: save original-scale bounded version
        if dataset == "custom_csv" and args.save_train_imputed_denorm:
            final_denorm = final_data.copy()

            if args.normalize and scaler is not None:
                final_denorm[:, target_indices] = scaler.inverse_transform(
                    final_denorm[:, target_indices]
                )
                print("[INFO] Inverse transform applied to train_imputed.")

            if bounds is not None and args.train_imputed_clamp != "none":
                diag = bounds.clamp_array(final_denorm, target_indices)
                print(f"[INFO] train_imputed clamp: "
                      f"{diag['n_clamped_lower']} below, "
                      f"{diag['n_clamped_upper']} above")

            np.save(
                os.path.join(prepared_dir, "train_imputed_denorm.npy"),
                final_denorm
            )

            train_imputed_df = pd.read_csv(
                os.path.join(prepared_dir, "train.csv")
            )
            denorm_df = pd.DataFrame(
                final_denorm, columns=training_df.columns
            )
            for col in denorm_df.columns:
                if col in train_imputed_df.columns:
                    train_imputed_df[col] = denorm_df[col].to_numpy()
            train_imputed_df.to_csv(
                os.path.join(prepared_dir, "train_imputed_denorm.csv"),
                index=False
            )

            print("[INFO] Saved denormalized EM output: train_imputed_denorm.npy")
            print("[INFO] Saved denormalized EM CSV: train_imputed_denorm.csv")

        torch.save(model.state_dict(), os.path.join(save_dir, f"{em_base}.pth"))

    else:
        print(f"\n{'='*60}")
        print("Standard Training")
        print(f"{'='*60}")

        T_seq, C = d_vals.shape
        if T_seq < args.window_size:
            raise ValueError(
                f"T_seq ({T_seq}) < window_size ({args.window_size}) "
                "for standard training."
            )

        nw = (T_seq - args.window_size) // args.stride + 1
        samples = torch.zeros(nw, args.window_size, C, dtype=torch.float32)

        for i in range(nw):
            s = i * args.stride
            samples[i] = torch.from_numpy(d_vals[s:s + args.window_size])

        dl = DataLoader(
            MyDataset(samples.float(), window_size=args.window_size),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20,
        )
        alpha_bars = diffusion_config["alpha_bars"].to(dev)
        best_loss = float("inf")

        for epoch in range(args.epochs):
            total = 0.0
            nb = 0

            for batch in dl:
                batch = batch.to(dev)
                ts = randint(diffusion_config["T"], size=(batch.shape[0],), device=dev)
                sigmas = normal(0, 1, size=batch.shape).to(dev)
                c1 = sqrt(alpha_bars[ts]).reshape(-1, 1, 1)
                c2 = sqrt(1 - alpha_bars[ts]).reshape(-1, 1, 1)

                cm = torch.ones_like(batch, device=dev)
                cm[:, :, non_hier_cols] = 0.0
                noised = (1 - cm) * (c1 * batch + c2 * sigmas) + cm * batch

                pred = model(noised, ts.reshape(-1, 1))
                optimizer.zero_grad()
                gt = sigmas[:, :, non_hier_cols].permute(0, 2, 1).to(dev)
                loss = criterion(pred, gt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total += loss.item()
                nb += 1

            avg = total / max(1, nb)
            scheduler.step(avg)

            if avg < best_loss:
                best_loss = avg
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, f"{std_base}_best.pth")
                )

            if epoch % 10 == 0:
                print(f"epoch {epoch:4d}: loss={avg:.6f}, best={best_loss:.6f}")

        torch.save(model.state_dict(), os.path.join(save_dir, f"{std_base}.pth"))

    # ── Post-training ───────────────────────────────────────────────
    import shutil
    from datetime import datetime

    expected = os.path.join(prepared_dir, "saved_model")
    if os.path.abspath(save_dir) != os.path.abspath(expected):
        if os.path.exists(save_dir):
            if os.path.exists(expected):
                shutil.rmtree(expected)
            shutil.copytree(save_dir, expected)

    mfiles = [f for f in os.listdir(expected) if f.endswith(".pth")]
    print(f"[POST] {len(mfiles)} model(s): {mfiles}")

    comp = {
        "completed_at": datetime.now().isoformat(),
        "model_count": len(mfiles),
        "ready": True,
        "repaint_rounds": repaint_rounds,
        "scaler_mode": scaler_mode,
        "clip_mode": clip_mode,
        "clip_bound": model_clip_bound,
        "save_train_imputed_denorm": bool(args.save_train_imputed_denorm),
        "train_imputed_clamp": args.train_imputed_clamp,
        "bound_headroom": args.bound_headroom,
        "nonneg_cols": args.nonneg_cols,
    }
    with open(os.path.join(prepared_dir, "training_completed.json"), "w") as f:
        json.dump(comp, f, indent=2)

    print(f"[DONE] → {prepared_dir}")

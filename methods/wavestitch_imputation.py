"""WaveStitch-based imputation helpers tailored for the 6G datasets.

The original `WaveStitch` project (https://github.com/adis98/WaveStitch)
exposes a diffusion-based model for time-series imputation. This module wraps
the core architecture that already lives under ``methods/TSImputer`` and adapts
it to the aligned CSV bundles that ship with the 6GDALI datasets.  The focus is
on preparing the data, training a lightweight S4 backbone, and running
conditional sampling to fill in missing values.

Example
-------

```bash
# Train a model on the EUR slice of the dataset.
python -m methods.wavestitch_imputation train \
    6GDALI_Datasets/EUR/6907619 \
    --output saved_models/eur_wavestitch.pt \
    --window-size 48 \
    --epochs 100

# Apply the trained model to impute the same directory and export a CSV.
python -m methods.wavestitch_imputation impute \
    6GDALI_Datasets/EUR/6907619 \
    --model saved_models/eur_wavestitch.pt \
    --output out/eur_imputed.csv
```

The ``train`` sub-command writes a checkpoint containing both the neural
network weights and the preprocessing metadata (feature list, scaler
statistics, window size, …). The ``impute`` sub-command restores that
checkpoint and produces an imputed CSV aligned with the input timestamps.
``saved_models`` and ``out`` directories are created automatically when they do
not exist.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .data_augmentation_beam import load_and_align_time_series
from .TSImputer.SSSDS4Imputer import SSSDS4Imputer


def _mps_is_available() -> bool:
    """Return ``True`` when PyTorch can target the Apple Metal backend."""

    mps = getattr(torch.backends, "mps", None)
    if mps is None:
        return False
    # ``is_available`` may raise when Metal support is compiled out, hence the guard.
    try:
        return bool(mps.is_available()) and bool(getattr(mps, "is_built", lambda: True)())
    except (RuntimeError, TypeError):  # pragma: no cover - defensive guard
        return False


def _select_device(preferred: str | None = None) -> torch.device:
    """Choose a torch device honoring user preference and hardware support."""

    if preferred:
        preference = preferred.lower()
        if preference in {"cuda", "gpu"}:
            if torch.cuda.is_available():
                return torch.device("cuda")
            raise ValueError("CUDA device requested but not available")
        if preference in {"mps", "metal"}:
            if _mps_is_available():
                return torch.device("mps")
            raise ValueError("MPS device requested but not available")
        if preference == "cpu":
            return torch.device("cpu")
        if preference != "auto":
            raise ValueError(
                "Unsupported device '{preferred}'. Choose from auto, cpu, cuda, or mps.".format(
                    preferred=preferred
                )
            )

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class WaveStitchConfig:
    """Hyper-parameters for training and sampling."""

    beta_0: float = 1e-4
    beta_T: float = 2e-2
    timesteps: int = 200
    res_channels: int = 64
    skip_channels: int = 64
    num_res_layers: int = 4
    diff_step_embed_in: int = 32
    diff_step_embed_mid: int = 64
    diff_step_embed_out: int = 64
    s4_lmax: int = 100
    s4_dstate: int = 64
    s4_dropout: float = 0.0
    s4_bidirectional: bool = True
    s4_layernorm: bool = True
    window_size: int = 32
    stride: int = 1
    batch_size: int = 128
    epochs: int = 200
    learning_rate: float = 1e-4


@dataclass
class PreparedDataset:
    """Container with tensors and metadata ready for WaveStitch training."""

    original: pd.DataFrame
    feature_columns: List[str]
    hierarchical_columns: List[str]
    non_hier_columns: List[str]
    train_windows: Tensor
    full_windows: Tensor
    observed_mask_windows: Tensor
    scaler_state: dict
    window_size: int
    stride: int
    index: pd.Index
    time_column: str
    feature_matrix: np.ndarray
    min_valid_ratio: float

    @property
    def non_hier_indices(self) -> List[int]:
        return [self.feature_columns.index(col) for col in self.non_hier_columns]

    @property
    def hierarchical_indices(self) -> List[int]:
        return [self.feature_columns.index(col) for col in self.hierarchical_columns]


def _coerce_numeric(value: object) -> float | np.nan:
    """Best-effort conversion of heterogeneous string values to floats."""

    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return math.nan
    # Accept strings such as "2048M" or "3.2GHz" by extracting the numeric part.
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return math.nan
    try:
        return float(match.group())
    except ValueError:
        return math.nan


def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _cyclic_encode(series: pd.Series, period: int) -> tuple[np.ndarray, np.ndarray]:
    values = series.to_numpy(dtype=float)
    sin = np.sin(2 * np.pi * values / period)
    cos = np.cos(2 * np.pi * values / period)
    return sin.astype(np.float32), cos.astype(np.float32)


def _prepare_feature_frame(
    records: Sequence[dict],
    *,
    time_column: str,
    min_valid_ratio: float,
    feature_columns: Sequence[str] | None,
) -> tuple[pd.DataFrame, List[str], List[str]]:
    """Convert aligned dictionaries into a numeric feature dataframe."""

    dataframe = pd.DataFrame(records)
    if time_column not in dataframe.columns:
        raise ValueError(f"Time column '{time_column}' not present in dataset")
    dataframe[time_column] = pd.to_datetime(dataframe[time_column], utc=True)
    dataframe = dataframe.sort_values(time_column).reset_index(drop=True)

    numeric = dataframe.copy()
    for column in numeric.columns:
        if column == time_column:
            continue
        if pd.api.types.is_numeric_dtype(numeric[column]):
            numeric[column] = numeric[column].astype(float)
        else:
            numeric[column] = numeric[column].map(_coerce_numeric)

    selected: List[str]
    if feature_columns is None:
        selected = []
        for column in numeric.columns:
            if column == time_column:
                continue
            series = numeric[column]
            valid_ratio = series.notna().sum() / max(len(series), 1)
            if valid_ratio < min_valid_ratio:
                continue
            if series.nunique(dropna=True) <= 1:
                continue
            selected.append(column)
    else:
        missing = [col for col in feature_columns if col not in numeric.columns]
        if missing:
            raise ValueError(
                "Dataset is missing expected features: " + ", ".join(sorted(missing))
            )
        selected = list(feature_columns)

    if not selected:
        raise ValueError("No usable numeric features detected in dataset")

    feature_df = numeric[[time_column] + selected].copy()

    timestamp = feature_df[time_column]
    components = {
        "year": timestamp.dt.year.astype(float),
        "month": timestamp.dt.month.astype(float),
        "day": timestamp.dt.day.astype(float),
        "hour": timestamp.dt.hour.astype(float),
        "minute": timestamp.dt.minute.astype(float),
    }
    hierarchical_cols: List[str] = []
    for name, values in components.items():
        period = int(values.max() - values.min() + 1) if values.nunique() > 1 else 1
        period = max(period, 1)
        sin, cos = _cyclic_encode(values, period)
        feature_df[f"{time_column}_{name}_sin"] = sin
        feature_df[f"{time_column}_{name}_cos"] = cos
        hierarchical_cols.extend(
            [f"{time_column}_{name}_sin", f"{time_column}_{name}_cos"]
        )

    feature_df["time_index"] = np.linspace(0.0, 1.0, len(feature_df), dtype=np.float32)
    hierarchical_cols.append("time_index")

    non_hierarchical = [col for col in selected if col not in hierarchical_cols]

    return feature_df, hierarchical_cols, non_hierarchical


def _build_windows(array: np.ndarray, window: int, stride: int) -> Tensor:
    tensor = torch.from_numpy(array.astype(np.float32))
    windows = tensor.unfold(0, window, stride).transpose(1, 2).contiguous()
    return windows


def prepare_dataset(
    sources: Sequence[str | Path],
    *,
    time_column: str = "time",
    min_valid_ratio: float = 0.1,
    window_size: int = 32,
    stride: int = 1,
    feature_columns: Sequence[str] | None = None,
) -> PreparedDataset:
    """Load CSV sources and return tensors required for WaveStitch."""

    records = load_and_align_time_series(sources, time_column=time_column, join="outer", on_duplicate="last")
    feature_df, hierarchical_cols, non_hier_cols = _prepare_feature_frame(
        records,
        time_column=time_column,
        min_valid_ratio=min_valid_ratio,
        feature_columns=feature_columns,
    )

    if len(feature_df) < window_size:
        raise ValueError(
            f"Dataset length ({len(feature_df)}) is smaller than the window size ({window_size})."
        )

    feature_order = hierarchical_cols + non_hier_cols
    feature_matrix = feature_df[feature_order].to_numpy(dtype=np.float32)

    non_hier_matrix = feature_df[non_hier_cols]
    observed_mask = (~non_hier_matrix.isna()).to_numpy(dtype=np.float32)

    complete_rows = non_hier_matrix.dropna().index
    if len(complete_rows) < window_size:
        raise ValueError(
            "Not enough fully-observed rows to assemble a training window."
        )

    scaler = StandardScaler()
    scaler.fit(non_hier_matrix.loc[complete_rows])
    scaler_state = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_features_in": scaler.n_features_in_,
        "n_samples_seen": float(scaler.n_samples_seen_),
        "feature_names": non_hier_cols,
    }

    filled_non_hier = non_hier_matrix.fillna(non_hier_matrix.loc[complete_rows].mean())
    scaled_non_hier = scaler.transform(filled_non_hier)

    scaled_matrix = np.concatenate(
        [feature_df[hierarchical_cols].to_numpy(dtype=np.float32), scaled_non_hier.astype(np.float32)],
        axis=1,
    )

    train_matrix = scaled_matrix[complete_rows]

    train_windows = _build_windows(train_matrix, window_size, stride)
    full_windows = _build_windows(scaled_matrix, window_size, stride)

    mask_windows = _build_windows(observed_mask, window_size, stride)

    return PreparedDataset(
        original=feature_df,
        feature_columns=feature_order,
        hierarchical_columns=hierarchical_cols,
        non_hier_columns=non_hier_cols,
        train_windows=train_windows,
        full_windows=full_windows,
        observed_mask_windows=mask_windows,
        scaler_state=scaler_state,
        window_size=window_size,
        stride=stride,
        index=feature_df.index,
        time_column=time_column,
        feature_matrix=scaled_matrix,
        min_valid_ratio=min_valid_ratio,
    )


def _restore_scaler(state: dict) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.array(state["mean"], dtype=np.float64)
    scaler.scale_ = np.array(state["scale"], dtype=np.float64)
    scaler.var_ = np.array(state["var"], dtype=np.float64)
    scaler.n_features_in_ = int(state["n_features_in"])
    scaler.n_samples_seen_ = state.get("n_samples_seen", len(state["mean"]))
    scaler.feature_names_in_ = np.array(state.get("feature_names", []))
    return scaler


def _diffusion_config(config: WaveStitchConfig, device: torch.device) -> dict:
    betas = torch.linspace(config.beta_0, config.beta_T, config.timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {"betas": betas, "alphas": alphas, "alpha_bars": alpha_bars, "T": config.timesteps}


def _train_model(prepared: PreparedDataset, config: WaveStitchConfig, device: torch.device) -> SSSDS4Imputer:
    dataset = TensorDataset(prepared.train_windows)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    model = SSSDS4Imputer(
        in_channels=len(prepared.feature_columns),
        res_channels=config.res_channels,
        skip_channels=config.skip_channels,
        out_channels=len(prepared.non_hier_columns),
        num_res_layers=config.num_res_layers,
        diffusion_step_embed_dim_in=config.diff_step_embed_in,
        diffusion_step_embed_dim_mid=config.diff_step_embed_mid,
        diffusion_step_embed_dim_out=config.diff_step_embed_out,
        s4_lmax=config.s4_lmax,
        s4_d_state=config.s4_dstate,
        s4_dropout=config.s4_dropout,
        s4_bidirectional=config.s4_bidirectional,
        s4_layernorm=config.s4_layernorm,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()

    diffusion = _diffusion_config(config, device)
    mask_template = torch.ones(
        (1, len(prepared.feature_columns), prepared.window_size), device=device
    )
    mask_template[:, prepared.non_hier_indices, :] = 0.0

    for epoch in range(config.epochs):
        total_loss = 0.0
        for (batch,) in dataloader:
            batch = batch.to(device)
            timesteps = torch.randint(
                diffusion["T"], size=(batch.size(0),), device=device, dtype=torch.long
            )
            noise = torch.randn_like(batch, device=device)
            alpha_bars = diffusion["alpha_bars"]
            coeff_1 = torch.sqrt(alpha_bars[timesteps]).view(-1, 1, 1)
            coeff_2 = torch.sqrt(1.0 - alpha_bars[timesteps]).view(-1, 1, 1)
            conditional_mask = mask_template.expand(batch.size(0), -1, -1)
            noised = (1 - conditional_mask) * (coeff_1 * batch + coeff_2 * noise) + conditional_mask * batch
            output = model(noised, timesteps.unsqueeze(1))
            target = noise[:, prepared.non_hier_indices, :]
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        print(f"epoch={epoch:04d} loss={total_loss:.4f}")

    return model


def _sample_windows(
    model: SSSDS4Imputer,
    prepared: PreparedDataset,
    config: WaveStitchConfig,
    device: torch.device,
) -> np.ndarray:
    diffusion = _diffusion_config(config, device)
    windows = prepared.full_windows.to(device)
    observed_mask = prepared.observed_mask_windows.to(device)

    samples = torch.randn_like(windows, device=device)
    samples[:, prepared.hierarchical_indices, :] = windows[:, prepared.hierarchical_indices, :]

    betas = diffusion["betas"]
    alphas = diffusion["alphas"]
    alpha_bars = diffusion["alpha_bars"]

    for step in range(config.timesteps - 1, -1, -1):
        t = torch.full((samples.size(0),), step, device=device, dtype=torch.long)
        eps = model(samples, t.unsqueeze(1))
        non_hier = samples[:, prepared.non_hier_indices, :]
        coeff = betas[step] / torch.sqrt(1.0 - alpha_bars[step])
        non_hier = (non_hier - coeff * eps) / torch.sqrt(alphas[step])
        if step > 0:
            sigma = betas[step] * (1.0 - alpha_bars[step - 1]) / (1.0 - alpha_bars[step])
            non_hier = non_hier + torch.sqrt(sigma) * torch.randn_like(non_hier)
        cond = windows[:, prepared.non_hier_indices, :]
        non_hier = non_hier * (1.0 - observed_mask) + cond * observed_mask
        samples[:, prepared.non_hier_indices, :] = non_hier
        samples[:, prepared.hierarchical_indices, :] = windows[:, prepared.hierarchical_indices, :]

    return samples.cpu().numpy()


def _reconstruct_sequence(windows: np.ndarray, window: int, stride: int) -> np.ndarray:
    num_windows, channels, length = windows.shape
    total_length = stride * (num_windows - 1) + window
    result = np.zeros((total_length, channels), dtype=np.float32)
    counts = np.zeros((total_length, channels), dtype=np.float32)
    for index in range(num_windows):
        start = index * stride
        segment = np.transpose(windows[index], (1, 0))
        result[start : start + window] += segment
        counts[start : start + window] += 1
    np.divide(result, counts, out=result, where=counts > 0)
    return result


def _impute_from_prepared(
    model: SSSDS4Imputer,
    prepared: PreparedDataset,
    config: WaveStitchConfig,
    device: torch.device,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Run the diffusion sampler and return an imputed dataframe."""

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        imputed_windows = _sample_windows(model, prepared, config, device)

    reconstructed = _reconstruct_sequence(
        imputed_windows, prepared.window_size, prepared.stride
    )

    scaled_full = prepared.feature_matrix
    reconstructed[:, prepared.hierarchical_indices] = scaled_full[
        :, prepared.hierarchical_indices
    ]

    mask_matrix = prepared.observed_mask_windows.detach().cpu().numpy()
    mask_sequence = _reconstruct_sequence(
        mask_matrix, prepared.window_size, prepared.stride
    )
    for idx, column in enumerate(prepared.non_hier_indices):
        observed = mask_sequence[:, idx] >= 0.5
        reconstructed[observed, column] = scaled_full[observed, column]

    non_hier_scaled = reconstructed[:, prepared.non_hier_indices]
    non_hier_values = scaler.inverse_transform(non_hier_scaled)

    output_df = pd.DataFrame(
        non_hier_values,
        index=prepared.original.index,
        columns=prepared.non_hier_columns,
    )
    output_df.insert(0, prepared.time_column, prepared.original[prepared.time_column])

    return output_df


class WaveStitchImputer:
    """High level wrapper around the WaveStitch diffusion imputer."""

    def __init__(
        self,
        config: WaveStitchConfig | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = replace(config) if config is not None else WaveStitchConfig()
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = _select_device(device)

        self.model: SSSDS4Imputer | None = None
        self.scaler_state: dict | None = None
        self.feature_columns: List[str] | None = None
        self.hierarchical_columns: List[str] | None = None
        self.non_hier_columns: List[str] | None = None
        self.time_column: str | None = None
        self._min_valid_ratio: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def _ensure_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "WaveStitchImputer has not been fitted yet. Call 'fit' or 'fit_transform' first."
            )

    def _finalize_fit(
        self, prepared: PreparedDataset, model: SSSDS4Imputer, min_valid_ratio: float
    ) -> None:
        self.model = model
        self.scaler_state = dict(prepared.scaler_state)
        self.feature_columns = list(prepared.feature_columns)
        self.hierarchical_columns = list(prepared.hierarchical_columns)
        self.non_hier_columns = list(prepared.non_hier_columns)
        self.time_column = prepared.time_column
        self._min_valid_ratio = float(min_valid_ratio)
        self.config.window_size = prepared.window_size
        self.config.stride = prepared.stride

    def fit(
        self,
        sources: Sequence[str | Path],
        *,
        time_column: str = "time",
        min_valid_ratio: float = 0.1,
        feature_columns: Sequence[str] | None = None,
    ) -> "WaveStitchImputer":
        """Train the model and store the fitted parameters on the instance."""

        prepared = prepare_dataset(
            sources,
            time_column=time_column,
            min_valid_ratio=min_valid_ratio,
            window_size=self.config.window_size,
            stride=self.config.stride,
            feature_columns=feature_columns,
        )
        model = _train_model(prepared, self.config, self.device)
        self._finalize_fit(prepared, model, min_valid_ratio)
        return self

    def fit_transform(
        self,
        sources: Sequence[str | Path],
        *,
        time_column: str = "time",
        min_valid_ratio: float = 0.1,
        feature_columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Train the model and immediately return the imputed dataframe."""

        prepared = prepare_dataset(
            sources,
            time_column=time_column,
            min_valid_ratio=min_valid_ratio,
            window_size=self.config.window_size,
            stride=self.config.stride,
            feature_columns=feature_columns,
        )
        model = _train_model(prepared, self.config, self.device)
        self._finalize_fit(prepared, model, min_valid_ratio)
        scaler = _restore_scaler(self.scaler_state)
        return _impute_from_prepared(model, prepared, self.config, self.device, scaler)

    def transform(
        self,
        sources: Sequence[str | Path],
        *,
        time_column: str | None = None,
        min_valid_ratio: float | None = None,
        feature_columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Apply the trained model to a dataset and return the imputed dataframe."""

        self._ensure_fitted()
        assert self.model is not None  # for type-checkers
        assert self.scaler_state is not None
        assert self.non_hier_columns is not None

        prepared = prepare_dataset(
            sources,
            time_column=time_column or (self.time_column or "time"),
            min_valid_ratio=(
                min_valid_ratio
                if min_valid_ratio is not None
                else (self._min_valid_ratio if self._min_valid_ratio is not None else 0.1)
            ),
            window_size=self.config.window_size,
            stride=self.config.stride,
            feature_columns=feature_columns or self.non_hier_columns,
        )

        if prepared.non_hier_columns != self.non_hier_columns:
            raise ValueError(
                "Prepared dataset does not match the feature layout used during training."
            )

        scaler = _restore_scaler(self.scaler_state)
        return _impute_from_prepared(self.model, prepared, self.config, self.device, scaler)

    def save(self, path: str | Path) -> Path:
        """Serialize the fitted model to ``path``."""

        self._ensure_fitted()
        assert self.model is not None
        assert self.scaler_state is not None
        assert self.feature_columns is not None
        assert self.hierarchical_columns is not None
        assert self.non_hier_columns is not None
        assert self.time_column is not None

        output_path = Path(path)
        _ensure_directory(output_path)

        state_dict = {
            key: value.detach().cpu()
            for key, value in self.model.state_dict().items()
        }
        checkpoint = {
            "model_state": state_dict,
            "config": asdict(self.config),
            "scaler_state": self.scaler_state,
            "feature_columns": self.feature_columns,
            "hierarchical_columns": self.hierarchical_columns,
            "non_hier_columns": self.non_hier_columns,
            "time_column": self.time_column,
            "min_valid_ratio": self._min_valid_ratio,
        }
        torch.save(checkpoint, output_path)
        return output_path

    @classmethod
    def load(
        cls, path: str | Path, *, device: str | torch.device | None = None
    ) -> "WaveStitchImputer":
        """Restore a fitted model from disk."""

        checkpoint = torch.load(path, map_location="cpu")
        config = WaveStitchConfig(**checkpoint["config"])
        imputer = cls(config=config, device=device)
        imputer.scaler_state = checkpoint["scaler_state"]
        imputer.feature_columns = list(checkpoint["feature_columns"])
        imputer.hierarchical_columns = list(checkpoint["hierarchical_columns"])
        imputer.non_hier_columns = list(checkpoint["non_hier_columns"])
        imputer.time_column = checkpoint["time_column"]
        imputer._min_valid_ratio = float(checkpoint.get("min_valid_ratio", 0.1))

        model = SSSDS4Imputer(
            in_channels=len(imputer.feature_columns),
            res_channels=config.res_channels,
            skip_channels=config.skip_channels,
            out_channels=len(imputer.non_hier_columns),
            num_res_layers=config.num_res_layers,
            diffusion_step_embed_dim_in=config.diff_step_embed_in,
            diffusion_step_embed_dim_mid=config.diff_step_embed_mid,
            diffusion_step_embed_dim_out=config.diff_step_embed_out,
            s4_lmax=config.s4_lmax,
            s4_d_state=config.s4_dstate,
            s4_dropout=config.s4_dropout,
            s4_bidirectional=config.s4_bidirectional,
            s4_layernorm=config.s4_layernorm,
        )
        model.load_state_dict(checkpoint["model_state"])
        imputer.model = model.to(imputer.device)
        imputer.model.eval()
        return imputer

def train_command(args: argparse.Namespace) -> None:
    config = WaveStitchConfig(
        beta_0=args.beta_0,
        beta_T=args.beta_T,
        timesteps=args.timesteps,
        res_channels=args.res_channels,
        skip_channels=args.skip_channels,
        num_res_layers=args.num_res_layers,
        diff_step_embed_in=args.diff_embed_in,
        diff_step_embed_mid=args.diff_embed_mid,
        diff_step_embed_out=args.diff_embed_out,
        s4_lmax=args.s4_lmax,
        s4_dstate=args.s4_dstate,
        s4_dropout=args.s4_dropout,
        s4_bidirectional=not args.s4_unidirectional,
        s4_layernorm=not args.s4_no_layernorm,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    imputer = WaveStitchImputer(config=config, device=getattr(args, "device", None))
    print(f"Training on device: {imputer.device}")
    imputer.fit(
        args.dataset,
        time_column=args.time_column,
        min_valid_ratio=args.min_valid_ratio,
    )

    output_path = imputer.save(args.output)
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump({"feature_columns": imputer.feature_columns}, fp, indent=2)
    print(f"Saved checkpoint to {output_path}")


def impute_command(args: argparse.Namespace) -> None:
    imputer = WaveStitchImputer.load(
        args.model, device=getattr(args, "device", None)
    )
    output_df = imputer.transform(
        args.dataset,
        min_valid_ratio=args.min_valid_ratio,
    )

    output_path = Path(args.output)
    _ensure_directory(output_path)
    output_df.to_csv(output_path, index=False)
    print(f"Wrote imputed dataset to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WaveStitch imputation for 6G datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a WaveStitch model")
    train_parser.add_argument("dataset", nargs="+", help="CSV files or directories")
    train_parser.add_argument("--output", type=str, required=True, help="Checkpoint path")
    train_parser.add_argument("--time-column", default="time")
    train_parser.add_argument("--min-valid-ratio", type=float, default=0.1)
    train_parser.add_argument("--window-size", type=int, default=32)
    train_parser.add_argument("--stride", type=int, default=1)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--beta-0", type=float, default=1e-4)
    train_parser.add_argument("--beta-T", type=float, default=2e-2)
    train_parser.add_argument("--timesteps", type=int, default=200)
    train_parser.add_argument("--res-channels", type=int, default=64)
    train_parser.add_argument("--skip-channels", type=int, default=64)
    train_parser.add_argument("--num-res-layers", type=int, default=4)
    train_parser.add_argument("--diff-embed-in", type=int, default=32)
    train_parser.add_argument("--diff-embed-mid", type=int, default=64)
    train_parser.add_argument("--diff-embed-out", type=int, default=64)
    train_parser.add_argument("--s4-lmax", type=int, default=100)
    train_parser.add_argument("--s4-dstate", type=int, default=64)
    train_parser.add_argument("--s4-dropout", type=float, default=0.0)
    train_parser.add_argument("--s4-unidirectional", action="store_true")
    train_parser.add_argument("--s4-no-layernorm", action="store_true")
    train_parser.add_argument(
        "--device",
        default="auto",
        help="Device to run training on (auto, cpu, cuda, mps)",
    )
    train_parser.set_defaults(func=train_command)

    impute_parser = subparsers.add_parser("impute", help="Impute a dataset with a trained model")
    impute_parser.add_argument("dataset", nargs="+", help="CSV files or directories")
    impute_parser.add_argument("--model", required=True, help="Checkpoint produced by the train command")
    impute_parser.add_argument("--output", required=True, help="Path to the imputed CSV")
    impute_parser.add_argument("--min-valid-ratio", type=float, default=0.05)
    impute_parser.add_argument(
        "--device",
        default="auto",
        help="Device to run inference on (auto, cpu, cuda, mps)",
    )
    impute_parser.set_defaults(func=impute_command)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover - command line interface
    main()


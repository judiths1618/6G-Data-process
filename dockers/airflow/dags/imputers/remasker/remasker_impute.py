# # stdlib
# from typing import Any, List, Tuple, Union

# # third party
# import numpy as np
# import math, sys, argparse
# import pandas as pd
# import torch
# from torch import nn
# from functools import partial
# import time, os, json
# from .remasker_utils import NativeScaler, MAEDataset, adjust_learning_rate, get_dataset
# from . import model_mae
# from tqdm import tqdm
# from torch.utils.data import DataLoader, RandomSampler
# import sys
# import timm.optim.optim_factory as optim_factory
# from .remasker_utils import get_args_parser

# # hyperimpute absolute
# from hyperimpute.plugins.imputers import ImputerPlugin
# from sklearn.datasets import load_iris
# from hyperimpute.utils.benchmarks import compare_models
# from hyperimpute.plugins.imputers import Imputers

# eps = 1e-8
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class ReMasker:

#     def __init__(self):
#         args = get_args_parser().parse_args()

#         self.batch_size = args.batch_size
#         self.accum_iter = args.accum_iter
#         self.min_lr = args.min_lr
#         self.norm_field_loss = args.norm_field_loss
#         self.weight_decay = args.weight_decay
#         self.lr = args.lr
#         self.blr = args.blr
#         self.warmup_epochs = args.warmup_epochs
#         self.model = None
#         self.norm_parameters = None

#         self.embed_dim = args.embed_dim
#         self.depth = args.depth
#         self.decoder_depth = args.decoder_depth
#         self.num_heads = args.num_heads
#         self.mlp_ratio = args.mlp_ratio
#         self.max_epochs = args.max_epochs
#         self.mask_ratio = args.mask_ratio
#         self.encode_func = args.encode_func

#     def fit(self, X_raw: pd.DataFrame):
#         X = X_raw.clone()

#         # Parameters
#         no = len(X)
#         dim = len(X[0, :])

#         X = X.cpu()

#         min_val = np.zeros(dim)
#         max_val = np.zeros(dim)

#         for i in range(dim):
#             min_val[i] = np.nanmin(X[:, i])
#             max_val[i] = np.nanmax(X[:, i])
#             X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)

#         self.norm_parameters = {"min": min_val, "max": max_val}

#         # Set missing
#         M = 1 - (1 * (np.isnan(X)))
#         M = M.float().to(device)

#         X = torch.nan_to_num(X)
#         X = X.to(device)

#         self.model = model_mae.MaskedAutoencoder(
#             rec_len=dim,
#             embed_dim=self.embed_dim,
#             depth=self.depth,
#             num_heads=self.num_heads,
#             decoder_embed_dim=self.embed_dim,
#             decoder_depth=self.decoder_depth,
#             decoder_num_heads=self.num_heads,
#             mlp_ratio=self.mlp_ratio,
#             norm_layer=partial(nn.LayerNorm, eps=eps),
#             norm_field_loss=self.norm_field_loss,
#             encode_func=self.encode_func
#         )

#         # if self.improve and os.path.exists(self.path):
#         #     self.model.load_state_dict(torch.load(self.path))
#         #     self.model.to(device)
#         #     return self

#         self.model.to(device)

#         # set optimizers
#         # param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
#         eff_batch_size = self.batch_size * self.accum_iter
#         if self.lr is None:  # only base_lr is specified
#             self.lr = self.blr * eff_batch_size / 64
#         # param_groups = optim_factory.add_weight_decay(self.model, self.weight_decay)
#         # optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.95))
#         optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95))
#         loss_scaler = NativeScaler()

#         dataset = MAEDataset(X, M)
#         dataloader = DataLoader(
#             dataset, sampler=RandomSampler(dataset),
#             batch_size=self.batch_size,
#         )

#         # if self.resume and os.path.exists(self.path):
#         #     self.model.load_state_dict(torch.load(self.path))
#         #     self.lr *= 0.5

#         self.model.train()
#         pbar = tqdm(range(self.max_epochs), desc='Training')
#         for epoch in pbar:

#             optimizer.zero_grad()
#             total_loss = 0

#             iter = 0
#             for iter, (samples, masks) in enumerate(dataloader):

#                 # we use a per iteration (instead of per epoch) lr scheduler
#                 if iter % self.accum_iter == 0:
#                     adjust_learning_rate(optimizer, iter / len(dataloader) + epoch, self.lr, self.min_lr,
#                                          self.max_epochs, self.warmup_epochs)

#                 samples = samples.unsqueeze(dim=1)
#                 samples = samples.to(device, non_blocking=True)
#                 masks = masks.to(device, non_blocking=True)

#                 # print(samples, masks)

#                 with torch.cuda.amp.autocast():
#                     loss, _, _, _ = self.model(samples, masks, mask_ratio=self.mask_ratio)
#                     loss_value = loss.item()
#                     total_loss += loss_value

#                 if not math.isfinite(loss_value):
#                     print("Loss is {}, stopping training".format(loss_value))
#                     sys.exit(1)

#                 loss /= self.accum_iter
#                 loss_scaler(loss, optimizer, parameters=self.model.parameters(),
#                             update_grad=(iter + 1) % self.accum_iter == 0)

#                 if (iter + 1) % self.accum_iter == 0:
#                     optimizer.zero_grad()

#             total_loss = (total_loss / (iter + 1)) ** 0.5
#             pbar.set_postfix(loss=total_loss)
#             # if total_loss < best_loss:
#             #     best_loss = total_loss
#             #     torch.save(self.model.state_dict(), self.path)
#             # if (epoch + 1) % 10 == 0 or epoch == 0:
#             # print((epoch+1),',', total_loss)

#         # torch.save(self.model.state_dict(), self.path)
#         return self

#     def transform(self, X_raw: torch.Tensor):

#         # X = X_raw.clone()
#         #
#         # min_val = self.norm_parameters["min"]
#         # max_val = self.norm_parameters["max"]
#         #
#         # no, dim = X.shape
#         # X = X.cpu()
#         #
#         # # MinMaxScaler normalization
#         # for i in range(dim):
#         #     X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)
#         #
#         # # Set missing
#         # M = 1 - (1 * (np.isnan(X)))
#         # X = np.nan_to_num(X)
#         #
#         # X = torch.from_numpy(X).to(device)
#         # M = M.to(device)
#         #
#         # self.model.eval()
#         #
#         # # Imputed data
#         # with torch.no_grad():
#         #     for i in range(no):
#         #         sample = torch.reshape(X[i], (1, 1, -1))
#         #         mask = torch.reshape(M[i], (1, -1))
#         #         _, pred, _, _ = self.model(sample, mask)
#         #         pred = pred.squeeze(dim=2)
#         #         if i == 0:
#         #             imputed_data = pred
#         #         else:
#         #             imputed_data = torch.cat((imputed_data, pred), 0)
#         #
#         #             # Renormalize
#         # for i in range(dim):
#         #     imputed_data[:, i] = imputed_data[:, i] * (max_val[i] - min_val[i] + eps) + min_val[i]
#         #
#         # if np.all(np.isnan(imputed_data.detach().cpu().numpy())):
#         #     err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
#         #     raise RuntimeError(err)
#         #
#         # M = M.cpu()
#         # imputed_data = imputed_data.detach().cpu()
#         # # print('imputed', imputed_data, M)
#         # # print('imputed', M * np.nan_to_num(X_raw.cpu()) + (1 - M) * imputed_data)
#         # return M * np.nan_to_num(X_raw.cpu()) + (1 - M) * imputed_data
#         X = X_raw.clone()
#         min_val = self.norm_parameters["min"]
#         max_val = self.norm_parameters["max"]
#         eps = 1e-8

#         # Move to CPU for NaN handling
#         X = X.cpu()

#         # Vectorized MinMax normalization
#         X = ((X - min_val) / (max_val - min_val + eps)).to(torch.float32)

#         # Handle missing values
#         M = ~torch.isnan(X)
#         X = torch.nan_to_num(X)

#         # Back to device
#         X = X.to(device)
#         M = M.to(device)

#         self.model.eval()
#         with torch.no_grad():
#             # Batch all samples at once
#             samples = X.view(X.shape[0], 1, -1)
#             masks = M.view(M.shape[0], -1)

#             _, preds, _, _ = self.model(samples, masks)
#             preds = preds.squeeze(dim=2)

#         # Vectorized renormalization
#         preds = preds.cpu()
#         imputed_data = (preds * (max_val - min_val + eps)) + min_val

#         # Safety check
#         if torch.isnan(imputed_data).any():
#             raise RuntimeError("The imputed result contains NaNs. Please report this bug.")

#         # Return imputed data
#         M = M.cpu()
#         # imputed_data = imputed_data.cpu()
#         return M * torch.nan_to_num(X_raw) + (~M) * imputed_data

#     def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
#         """Imputes the provided dataset using the GAIN strategy.
#         Args:
#             X: np.ndarray
#                 A dataset with missing values.
#         Returns:
#             Xhat: The imputed dataset.
#         """
#         X = torch.tensor(X.values, dtype=torch.float32)
#         return self.fit(X).transform(X).detach().cpu().numpy()

# stdlib / typing
# remasker_impute.py

# stdlib / typing
from typing import Any, Optional, Union
import math
from functools import partial

# third party
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

# local
from .remasker_utils import MAEDataset, adjust_learning_rate, get_args_parser
from . import model_mae

eps = 1e-8
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- helpers ----------

def _to_tensor_f32(x: Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32)
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_numpy()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=torch.float32)
    raise TypeError(f"Unsupported X type: {type(x)}")


def _nanmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    """NaN-robust min along dim (works on torch 2.2.x)."""
    pos_inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    x_min = torch.where(torch.isnan(x), pos_inf, x)
    out = torch.amin(x_min, dim=dim)
    all_nan = torch.isnan(x).all(dim=dim)
    return torch.where(all_nan, torch.zeros_like(out), out)


def _nanmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """NaN-robust max along dim (works on torch 2.2.x)."""
    neg_inf = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
    x_max = torch.where(torch.isnan(x), neg_inf, x)
    out = torch.amax(x_max, dim=dim)
    all_nan = torch.isnan(x).all(dim=dim)
    return torch.where(all_nan, torch.ones_like(out), out)


# ---------- model ----------

class ReMasker:
    """
    MAE-based imputer.

    - Safe in Airflow: no CLI parsing from ambient sys.argv (uses parser defaults; override via kwargs).
    - Works on CPU or CUDA; AMP only if CUDA is available (no GradScaler warning on CPU).
    - Robust NaN handling compatible with PyTorch 2.2.x (no torch.nanmin/nanmax).
    - Accepts torch / numpy / pandas inputs.
    """

    def __init__(self, **overrides: Any):
        # get defaults WITHOUT touching process argv (e.g., 'scheduler' in Airflow)
        parser = get_args_parser()
        args = parser.parse_args([])

        # hyperparams (allow programmatic overrides)
        self.batch_size     = overrides.get("batch_size", args.batch_size)
        self.accum_iter     = overrides.get("accum_iter", args.accum_iter)
        self.min_lr         = overrides.get("min_lr", args.min_lr)
        self.norm_field_loss= overrides.get("norm_field_loss", args.norm_field_loss)
        self.weight_decay   = overrides.get("weight_decay", args.weight_decay)
        self.lr             = overrides.get("lr", args.lr)
        self.blr            = overrides.get("blr", args.blr)
        self.warmup_epochs  = overrides.get("warmup_epochs", args.warmup_epochs)

        self.embed_dim      = overrides.get("embed_dim", args.embed_dim)
        self.depth          = overrides.get("depth", args.depth)
        self.decoder_depth  = overrides.get("decoder_depth", args.decoder_depth)
        self.num_heads      = overrides.get("num_heads", args.num_heads)
        self.mlp_ratio      = overrides.get("mlp_ratio", args.mlp_ratio)
        self.max_epochs     = overrides.get("max_epochs", args.max_epochs)
        self.mask_ratio     = overrides.get("mask_ratio", args.mask_ratio)
        self.encode_func    = overrides.get("encode_func", args.encode_func)

        self.model: Optional[torch.nn.Module] = None
        # store min/max on CPU for serialization portability
        self.norm_parameters: Optional[dict[str, torch.Tensor]] = None

    # ---------- API ----------

    def fit(self, X_raw: Union[torch.Tensor, np.ndarray, pd.DataFrame]) -> "ReMasker":
        """
        Train the MAE on rows with observed/masked features.
        X_raw: (N, D) tensor/ndarray/df with NaNs marking missing values.
        """
        X = _to_tensor_f32(X_raw).clone()  # (N, D) float32

        # Per-feature min/max (NaN-safe)
        min_val = _nanmin(X, dim=0)
        max_val = _nanmax(X, dim=0)

        denom = (max_val - min_val).clamp_min(eps)
        Xn = (X - min_val) / denom

        # Observed mask BEFORE filling; then fill NaNs with 0
        M = ~torch.isnan(Xn)
        Xn = torch.nan_to_num(Xn)

        # Keep params on CPU for saving
        self.norm_parameters = {"min": min_val.cpu(), "max": max_val.cpu()}

        # Build model
        D = X.shape[1]
        self.model = model_mae.MaskedAutoencoder(
            rec_len=D,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=eps),
            norm_field_loss=self.norm_field_loss,
            encode_func=self.encode_func,
        ).to(_device)

        # Optimizer / LR
        eff_bs = self.batch_size * self.accum_iter
        base_lr = self.lr if self.lr is not None else self.blr * eff_bs / 64.0
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=base_lr, betas=(0.9, 0.95))

        # Dataset/loader
        ds = MAEDataset(Xn.to(_device), M.to(_device))
        dl = DataLoader(ds, sampler=RandomSampler(ds), batch_size=self.batch_size)

        # AMP only if CUDA is available
        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Train
        self.model.train()
        pbar = tqdm(range(self.max_epochs), desc="Training", leave=False)
        for epoch in pbar:
            optimizer.zero_grad(set_to_none=True)
            total_loss, steps = 0.0, 0

            for steps, (samples, masks) in enumerate(dl, start=1):
                # Per-iteration scheduler
                adjust_learning_rate(
                    optimizer,
                    (steps - 1) / max(len(dl), 1) + epoch,
                    base_lr, self.min_lr, self.max_epochs, self.warmup_epochs,
                )

                samples = samples.unsqueeze(1).to(_device, non_blocking=True)  # (B,1,D)
                masks   = masks.to(_device, non_blocking=True)                 # (B,D)

                if use_amp:
                    with torch.cuda.amp.autocast(True):
                        loss, _, _, _ = self.model(samples, masks, mask_ratio=self.mask_ratio)
                else:
                    loss, _, _, _ = self.model(samples, masks, mask_ratio=self.mask_ratio)

                loss_val = float(loss.item())
                if not math.isfinite(loss_val):
                    raise RuntimeError(f"Non-finite loss {loss_val}")

                # Gradient accumulation
                loss = loss / self.accum_iter
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if steps % self.accum_iter == 0:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                total_loss += loss_val

            mean_loss = (total_loss / max(steps, 1)) ** 0.5
            pbar.set_postfix(loss=f"{mean_loss:.6f}")

        return self

    def transform(self, X_raw: Union[torch.Tensor, np.ndarray, pd.DataFrame]) -> torch.Tensor:
        if self.model is None or self.norm_parameters is None:
            raise RuntimeError("Call fit() before transform().")

        X = _to_tensor_f32(X_raw)
        min_val = self.norm_parameters["min"].to(X.device)
        max_val = self.norm_parameters["max"].to(X.device)
        denom   = (max_val - min_val).clamp_min(eps)

        # Normalize/Mask
        Xn = (X - min_val) / denom
        M  = ~torch.isnan(Xn)
        Xn = torch.nan_to_num(Xn)

        # Forward
        self.model.eval()
        with torch.no_grad():
            samples = Xn.view(Xn.shape[0], 1, -1).to(_device)
            masks   = M.view(M.shape[0], -1).to(_device)
            _, preds, _, _ = self.model(samples, masks)
            preds = preds.squeeze(2).to(X.device)  # (N,D)

        # Denormalize predictions
        imputed = preds * denom + min_val
        if torch.isnan(imputed).any():
            raise RuntimeError("Imputed result contains NaNs.")

        # Keep observed, fill missing
        return M.to(X.device) * torch.nan_to_num(X) + (~M.to(X.device)) * imputed

    def fit_transform(self, X: Union[torch.Tensor, np.ndarray, pd.DataFrame]) -> np.ndarray:
        Xt = _to_tensor_f32(X)
        return self.fit(Xt).transform(Xt).detach().cpu().numpy()

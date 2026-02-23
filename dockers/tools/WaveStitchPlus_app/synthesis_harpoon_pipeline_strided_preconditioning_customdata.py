import argparse
import os
import json
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
from torch import from_numpy, sqrt
from torch.utils.data import DataLoader
import torch.nn.functional as F

from helper.training_utils import MyDataset, fetchModel, fetchDiffusionConfig

# built-in datasets only
from helper.data_utils import Preprocessor
from helper.metasynth import metadataMask
from pathlib import Path
from custom_pipeline.directory_manager import get_save_dir, get_generated_dir


def ensure_bwc(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Ensure x is [B, W, C]. Accepts [B, W, C] or [B, C, W].
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {x.shape}")
    if x.shape[1] == window_size:
        return x
    if x.shape[2] == window_size:
        return x.transpose(1, 2)
    raise ValueError(f"Cannot infer window axis: {x.shape} vs window_size={window_size}")


def create_pipelined_noise(test_batch, args):
    """
    test_batch: [B, W, C]
    returns:    [B, W, C]
    """
    B, W, C = test_batch.shape
    sampled = torch.normal(
        0, 1,
        (args.stride * (B - 1) + W, C),
        device=test_batch.device
    )
    sampled_noise = sampled.unfold(0, W, args.stride)

    if sampled_noise.shape[1] == C and sampled_noise.shape[2] == W:
        sampled_noise = sampled_noise.transpose(1, 2)

    return sampled_noise


def recompute_cond_features(df, time_col, cond_cols, target_cols):
    """重新计算条件特征"""
    from custom_pipeline.features import add_time_features, add_gap_structure_features
    df = df.copy()

    # 更安全的 is_gap 检测
    if "is_gap" in df.columns:
        is_gap_vals = df["is_gap"].fillna(0).to_numpy(dtype=float)
        observed_row_mask = ~(is_gap_vals > 0.5)
    else:
        target_cols_in_df = [c for c in target_cols if c in df.columns]
        if len(target_cols_in_df) > 0:
            observed_row_mask = ~df[target_cols_in_df].isna().all(axis=1).to_numpy()
        else:
            observed_row_mask = np.ones(len(df), dtype=bool)

    df = add_time_features(df, time_col=time_col)
    df = add_gap_structure_features(df, observed_row_mask)

    for c in cond_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    
    return df


def load_custom_prepared(prepared_dir: str):
    """加载准备好的数据"""
    meta_path = os.path.join(prepared_dir, "meta.json")
    test_input_path = os.path.join(prepared_dir, "test_input.csv")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    if not os.path.exists(test_input_path):
        raise FileNotFoundError(f"test_input.csv not found: {test_input_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    df_input = pd.read_csv(test_input_path)

    time_col = meta.get("time_col", "time")
    all_model_cols = meta.get("all_model_cols", None)

    if all_model_cols is not None:
        expected = [time_col] + all_model_cols
        missing = [c for c in expected if c not in df_input.columns]
        if missing:
            raise ValueError(f"Prepared test_input.csv missing columns: {missing}")
        df_input = df_input[expected]
    else:
        cols = [c for c in df_input.columns if c != time_col]
        df_input = df_input[[time_col] + cols]

    return df_input, meta


def find_best_model(saved_dir: str, prop_cyc_enc: bool = False) -> str:
    """按优先级查找最佳模型"""
    candidates = [
        "model_em.pth",       # EM 训练的最终模型
        "model_em_best.pth",  # EM 训练的最佳模型
        "model_best.pth",     # 标准训练的最佳模型
        "model.pth",          # 标准训练的最终模型
    ]
    
    if prop_cyc_enc:
        candidates.insert(0, "model_prop.pth")
    
    for name in candidates:
        path = os.path.join(saved_dir, name)
        if os.path.exists(path):
            return path
    
    return None


def stitch_windows_weighted(windows: torch.Tensor, start_indices: list, 
                            T_total: int, device: torch.device) -> torch.Tensor:
    """
    使用加权平均拼接重叠窗口
    
    Args:
        windows: [N, W, C] tensor
        start_indices: list of start positions
        T_total: total sequence length
        device: torch device
    
    Returns:
        stitched: [T_total, C] tensor
    """
    N, W, C = windows.shape
    
    result = torch.zeros(T_total, C, device=device)
    weights = torch.zeros(T_total, 1, device=device)
    
    for i, start in enumerate(start_indices):
        end = min(start + W, T_total)
        actual_len = end - start
        result[start:end] += windows[i, :actual_len]
        weights[start:end] += 1
    
    # 避免除以零
    weights = torch.clamp(weights, min=1)
    result = result / weights
    
    return result


def bound_penalty_masked(
    x_target: torch.Tensor,
    imputed_mask: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    power: float = 2.0,
):
    """
    计算边界惩罚（只对 imputed 位置）
    
    x_target:     [B, W, n_target]
    imputed_mask: [B, W, n_target] True means "imputed positions to penalize"
    lb, ub:       [n_target] broadcastable to x_target
    """
    above = F.relu(x_target - ub)
    below = F.relu(lb - x_target)

    if power == 1.0:
        pen = above + below
    else:
        pen = above.pow(power) + below.pow(power)

    return (pen * imputed_mask).sum()


def auto_pos_bounds_from_observed(
    df_input: pd.DataFrame,
    target_cols: list,
    rows_to_synth: np.ndarray,
    eps_pos: float = 1e-6,
    q_high: float = 0.99,
    pad_ratio: float = 0.05,
    fallback_ub: float = 1e6,
):
    """
    自动决定 bounds（RAW scale）
    """
    obs = df_input.loc[~rows_to_synth, target_cols]
    bounds = {}
    
    for c in target_cols:
        x = obs[c].dropna().to_numpy(dtype=np.float64)
        lb = float(eps_pos)

        if len(x) >= 10:
            ub = float(np.quantile(x, q_high))
            if not np.isfinite(ub) or ub <= lb:
                ub = float(np.max(x)) if len(x) else float(fallback_ub)
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


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "-d", type=str, required=True)
    parser.add_argument("-prepared_dir", type=str, default="./work/prepared")
    parser.add_argument("-out_csv", type=str, default=None)

    # 模型参数
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
    parser.add_argument("-s4_bidirectional", type=bool, default=True)
    parser.add_argument("-s4_layernorm", type=bool, default=True)

    parser.add_argument("-propCycEnc", type=bool, default=False)
    parser.add_argument("-synth_mask", type=str, default="gap_imputation")
    parser.add_argument("-n_trials", type=int, default=5)

    # Bounds 参数
    parser.add_argument("-bounds_json", type=str, default=None)
    parser.add_argument("-bound_lambda", type=float, default=0.3)
    parser.add_argument("-bound_power", type=float, default=2.0)
    parser.add_argument("-pos_eps", type=float, default=1e-6)
    parser.add_argument("-auto_ub_q", type=float, default=0.99)
    parser.add_argument("-auto_ub_pad", type=float, default=0.05)
    parser.add_argument("-hard_project_positive", action="store_true")

    # 🔥 新增：guidance scale
    parser.add_argument("-guidance_scale", type=float, default=0.1,
                        help="Scale for gradient correction (0.05-0.2 typical)")
    
    # 🔥 新增：使用加权 stitching
    parser.add_argument("-use_weighted_stitch", action="store_true",
                        help="Use weighted averaging for window stitching")

    args = parser.parse_args()
    dataset = args.dataset
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"WaveStitch Synthesis Script")
    print(f"{'='*60}")
    print(f"[INFO] Device: {dev}")
    print(f"[INFO] Dataset: {dataset}")
    print(f"[INFO] Guidance scale: {args.guidance_scale}")
    print(f"[INFO] Bound lambda: {args.bound_lambda}")

    # -----------------------------
    # 1) Load data & masks
    # -----------------------------
    scaler_mean = None
    scaler_std = None
    target_indices = None
    lb = ub = None

    if dataset == "custom_csv":
        df_input, meta = load_custom_prepared(args.prepared_dir)
        time_col = meta.get("time_col", "time")
        cond_cols = meta.get("cond_cols", [])
        target_cols = meta.get("target_cols", [])

        if len(target_cols) == 0:
            raise ValueError("custom_csv: target_cols empty. Check meta.json.")

        print(f"[INFO] Recomputing conditional features for test set...")
        df_input = recompute_cond_features(df_input, time_col, cond_cols, target_cols)

        model_cols = [c for c in df_input.columns if c != time_col]
        
        if "is_gap" in df_input.columns:
            rows_to_synth = df_input["is_gap"].fillna(0).to_numpy(dtype=float) > 0.5
        else:
            rows_to_synth = df_input[target_cols].isna().all(axis=1).to_numpy(dtype=bool)

        print(f"[INFO] Total rows: {len(df_input)}, Gap rows to synthesize: {rows_to_synth.sum()}")

        df_synth = df_input.copy()
        df_synth[target_cols] = df_synth[target_cols].fillna(0.0)
        for c in cond_cols:
            if c in df_synth.columns:
                df_synth[c] = df_synth[c].fillna(0.0)

        expected_model_cols = meta.get("all_model_cols", model_cols)
        df_synth = df_synth[[time_col] + expected_model_cols]
        model_cols = expected_model_cols

        # 加载 scaler
        scaler_dir = os.path.join(args.prepared_dir, "scaler")
        if os.path.exists(scaler_dir):
            mean_path = os.path.join(scaler_dir, "mean.npy")
            std_path = os.path.join(scaler_dir, "std.npy")

            if os.path.exists(mean_path) and os.path.exists(std_path):
                print(f"[INFO] Loading scaler for input normalization")
                scaler_mean = np.load(mean_path).astype(np.float32)
                scaler_std = np.load(std_path).astype(np.float32)

        target_indices = [model_cols.index(c) for c in target_cols if c in model_cols]
        if len(target_indices) != len(target_cols):
            missing = [c for c in target_cols if c not in model_cols]
            raise ValueError(f"Some target_cols not found in model cols: {missing}")

        # 准备 bounds
        bounds = None
        if args.bounds_json is not None and os.path.exists(args.bounds_json):
            with open(args.bounds_json, "r") as f:
                bounds = json.load(f)
            print(f"[INFO] Loaded bounds from: {args.bounds_json}")
        else:
            print("[INFO] Auto-estimating bounds (targets strictly > 0).")
            bounds = auto_pos_bounds_from_observed(
                df_input=df_input,
                target_cols=target_cols,
                rows_to_synth=rows_to_synth,
                eps_pos=args.pos_eps,
                q_high=args.auto_ub_q,
                pad_ratio=args.auto_ub_pad,
                fallback_ub=1e6,
            )

        lb_raw = []
        ub_raw = []
        for c in target_cols:
            if c not in bounds:
                raise ValueError(f"Bounds missing target col: {c}")
            lo, hi = float(bounds[c][0]), float(bounds[c][1])
            lo = max(lo, float(args.pos_eps))
            if hi <= lo:
                hi = lo + 1.0
            lb_raw.append(lo)
            ub_raw.append(hi)

        lb_raw = np.array(lb_raw, dtype=np.float32)
        ub_raw = np.array(ub_raw, dtype=np.float32)

        # 构建归一化输入
        d_vals_raw = df_synth.drop(columns=[time_col]).values.astype(np.float32)

        if scaler_mean is not None and scaler_std is not None:
            if scaler_mean.ndim != 1 or scaler_std.ndim != 1 or len(scaler_mean) != len(target_cols):
                raise ValueError(
                    f"Scaler mean/std must be 1D and align with target_cols length. "
                    f"Got mean:{scaler_mean.shape}, std:{scaler_std.shape}, target_cols:{len(target_cols)}"
                )
            print(f"[INFO] Normalizing target columns using scaler")

            d_vals_raw[:, target_indices] = (d_vals_raw[:, target_indices] - scaler_mean) / (scaler_std + 1e-12)
            d_vals_raw[:, target_indices] = np.clip(d_vals_raw[:, target_indices], -3.0, 3.0)

            lb = (lb_raw - scaler_mean) / (scaler_std + 1e-12)
            ub = (ub_raw - scaler_mean) / (scaler_std + 1e-12)
            lb = np.clip(lb, -3.0, 3.0)
            ub = np.clip(ub, -3.0, 3.0)
        else:
            print(f"[WARNING] Scaler not found; using raw values")
            lb, ub = lb_raw, ub_raw

        hierarchical_column_indices = np.array(
            [model_cols.index(c) for c in cond_cols if c in model_cols],
            dtype=int
        )

        print(f"[INFO] Model columns: {len(model_cols)}")
        print(f"[INFO] Cond columns (indices): {hierarchical_column_indices}")
        print(f"[INFO] Target columns: {target_cols}")

        # 构建 windows
        m_vals = rows_to_synth.astype(np.bool_)
        Tt, Ct = d_vals_raw.shape
        W = args.window_size

        start_indices = list(range(0, Tt - W + 1, args.stride))
        last_start = Tt - W
        if len(start_indices) == 0 or start_indices[-1] != last_start:
            start_indices.append(last_start)

        num_windows = len(start_indices)
        print(f"[INFO] num_windows={num_windows}, T={Tt}, W={W}, stride={args.stride}")

        windows = torch.zeros(num_windows, W, Ct, dtype=torch.float32)
        masks = torch.zeros(num_windows, W, dtype=torch.bool)

        for i, s in enumerate(start_indices):
            e = s + W
            windows[i] = torch.from_numpy(d_vals_raw[s:e, :])
            masks[i] = torch.from_numpy(m_vals[s:e])

        print(f"[INFO] Windows shape: {windows.shape}")
        print(f"[INFO] Masks shape: {masks.shape}")

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented in this version")

    # -----------------------------
    # 2) Dataset & Model
    # -----------------------------
    in_dim = windows.shape[2]
    out_dim = in_dim - len(hierarchical_column_indices)

    print(f"[INFO] in_dim={in_dim}, out_dim={out_dim}")

    test_dataset = MyDataset(windows.float(), window_size=args.window_size)
    mask_dataset = MyDataset(masks)

    model = fetchModel(in_dim, out_dim, args).to(dev)
    diffusion_config = fetchDiffusionConfig(args)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    mask_dataloader = DataLoader(mask_dataset, batch_size=args.batch_size)

    all_indices = np.arange(in_dim)
    non_hier_cols = np.setdiff1d(all_indices, hierarchical_column_indices)

    # Bounds tensors
    lb_t = ub_t = target_idx_t = None
    if dataset == "custom_csv":
        lb_t = torch.tensor(lb, device=dev, dtype=torch.float32)
        ub_t = torch.tensor(ub, device=dev, dtype=torch.float32)
        target_idx_t = torch.tensor(target_indices, device=dev, dtype=torch.long)

    # 🔥 修复：使用新的模型查找函数
    saved_dir = get_save_dir(args.prepared_dir)
    model_path = find_best_model(saved_dir, args.propCycEnc)
    
    if model_path is None:
        raise FileNotFoundError(f"No model found in {saved_dir}. Please train first!")
    
    print(f"[INFO] Loading model from: {model_path}")
    
    saved_params = torch.load(model_path, map_location=dev)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in saved_params:
                param.copy_(saved_params[name])
            else:
                print(f"[WARNING] Parameter '{name}' not found in checkpoint!")
            param.requires_grad = True
    
    model.eval()
    
    # 统计模型参数
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {n_params:,}")

    num_ops = 0
    exec_times = []

    # -----------------------------
    # 3) Synthesis loop
    # -----------------------------
    for trial in range(args.n_trials):
        print(f"\n[Trial {trial + 1}/{args.n_trials}]")
        start_t = timer()
        
        # 🔥 修复：收集所有窗口结果用于加权 stitching
        all_windows_results = []
        global_win_ptr = 0

        with torch.no_grad():
            synth_tensor = torch.empty(0, in_dim).to(dev)

            for idx, (test_batch, mask_batch) in enumerate(zip(test_dataloader, mask_dataloader)):
                test_batch = ensure_bwc(test_batch, args.window_size).to(dev)
                mask_batch = mask_batch.to(dev).bool()

                x = create_pipelined_noise(test_batch, args).to(dev)
                x.requires_grad_()

                x[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]

                mask_expanded = torch.zeros_like(test_batch, dtype=torch.bool)
                cols = torch.tensor(non_hier_cols, device=test_batch.device, dtype=torch.long)
                mask_expanded[:, :, cols] = mask_batch.unsqueeze(-1)

                for step in range(diffusion_config["T"] - 1, -1, -1):
                    times = torch.full(size=(test_batch.shape[0], 1), fill_value=step, device=dev)

                    alpha_bar_t = diffusion_config["alpha_bars"][step].to(dev)
                    alpha_bar_t_1 = diffusion_config["alpha_bars"][step - 1].to(dev) if step > 0 else diffusion_config["alpha_bars"][0].to(dev)
                    alpha_t = diffusion_config["alphas"][step].to(dev)
                    beta_t = diffusion_config["betas"][step].to(dev)

                    sampled_noise = create_pipelined_noise(test_batch, args).to(dev)
                    conditional_fwd = sqrt(alpha_bar_t) * test_batch + sqrt(1 - alpha_bar_t) * sampled_noise

                    if step == diffusion_config["T"] - 1:
                        x[~mask_expanded] = conditional_fwd[~mask_expanded]

                    x[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]

                    with torch.enable_grad():
                        epsilon_pred = model(x, times).permute((0, 2, 1))

                        if step > 0:
                            vari = beta_t * ((1 - alpha_bar_t_1) / (1 - alpha_bar_t)) * torch.normal(
                                0, 1, size=epsilon_pred.shape, device=dev
                            )
                        else:
                            vari = torch.zeros_like(epsilon_pred)

                        normal_denoising = create_pipelined_noise(test_batch, args).to(dev)
                        normal_denoising[:, :, non_hier_cols] = (
                            x[:, :, non_hier_cols] - (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred
                        ) / torch.sqrt(alpha_t)
                        normal_denoising[:, :, non_hier_cols] += vari[:, :, non_hier_cols]

                        # Loss 1: continuity
                        rolled_x = normal_denoising.roll(1, 0)
                        rolled_x[0, args.stride:, :] = normal_denoising[0, :(args.window_size - args.stride), :]

                        loss1 = torch.sum(
                            (normal_denoising[:, :(args.window_size - args.stride), non_hier_cols] -
                             rolled_x[:, args.stride:args.window_size, non_hier_cols]) ** 2,
                            dim=(1, 2)
                        )

                        # Loss 2: reconstruction
                        recon = (x[:, :, non_hier_cols] - torch.sqrt(1 - alpha_bar_t) * epsilon_pred[:, :, non_hier_cols]) / torch.sqrt(alpha_bar_t)
                        recon_full = x.clone()
                        recon_full[:, :, non_hier_cols] = recon

                        loss2 = torch.sum(
                            (~mask_expanded[:, :, non_hier_cols]) * (recon_full[:, :, non_hier_cols] - test_batch[:, :, non_hier_cols]) ** 2,
                            dim=(1, 2)
                        )

                        # Loss 3: bounds
                        if (dataset == "custom_csv") and (args.bound_lambda > 0) and (target_idx_t is not None):
                            recon_target = recon_full.index_select(dim=2, index=target_idx_t)
                            imputed_mask_target = mask_batch.unsqueeze(-1).expand_as(recon_target)
                            loss3 = bound_penalty_masked(
                                x_target=recon_target,
                                imputed_mask=imputed_mask_target,
                                lb=lb_t,
                                ub=ub_t,
                                power=args.bound_power
                            )
                        else:
                            loss3 = torch.zeros((), device=dev)

                        loss = loss1 + loss2 + (args.bound_lambda * loss3)

                        grad = torch.autograd.grad(loss, x, grad_outputs=torch.ones_like(loss))[0]

                    # 🔥 修复：使用可配置的 guidance scale
                    x[:, :, non_hier_cols] = normal_denoising[:, :, non_hier_cols]
                    x[:, :, non_hier_cols] = x[:, :, non_hier_cols] + (-args.guidance_scale * grad[:, :, non_hier_cols])

                    if trial == 0:
                        num_ops += 1

                # 强制观测值
                x[~mask_expanded] = test_batch[~mask_expanded]

                # 收集窗口结果
                if args.use_weighted_stitch:
                    all_windows_results.append(x.clone())
                else:
                    # 原始 stitching 逻辑
                    if dataset == "custom_csv":
                        B, W, C = x.shape
                        pieces = []
                        for j in range(B):
                            g = global_win_ptr + j
                            if g == 0:
                                pieces.append(x[j])
                            else:
                                delta = start_indices[g] - start_indices[g - 1]
                                pieces.append(x[j, W - delta:, :])
                        generated = torch.cat(pieces, dim=0)
                        synth_tensor = torch.cat((synth_tensor, generated), dim=0)
                        global_win_ptr += B

        # 🔥 加权 stitching
        if args.use_weighted_stitch and len(all_windows_results) > 0:
            all_windows = torch.cat(all_windows_results, dim=0)
            synth_tensor = stitch_windows_weighted(
                all_windows, start_indices, Tt, dev
            )
        
        exec_times.append(timer() - start_t)
        print(f"  Time: {exec_times[-1]:.2f}s")

        # -----------------------------
        # 4) Output
        # -----------------------------
        if dataset == "custom_csv":
            if synth_tensor.shape[0] != len(df_input):
                print(f"[WARNING] synth_len={synth_tensor.shape[0]} != T={len(df_input)}, truncating/padding")
                if synth_tensor.shape[0] > len(df_input):
                    synth_tensor = synth_tensor[:len(df_input)]
                else:
                    padding = torch.zeros(len(df_input) - synth_tensor.shape[0], in_dim, device=dev)
                    synth_tensor = torch.cat([synth_tensor, padding], dim=0)
            
            synth_array = synth_tensor.detach().cpu().numpy()

            if scaler_mean is not None and scaler_std is not None:
                print(f"  Applying inverse transform")
                synth_array[:, target_indices] = synth_array[:, target_indices] * (scaler_std + 1e-12) + scaler_mean

            synth_df = pd.DataFrame(synth_array, columns=model_cols)

            for c in cond_cols:
                if c in synth_df.columns:
                    synth_df.drop(columns=[c], inplace=True)

            final_df = df_input[[time_col]].copy()

            for c in target_cols:
                if c in synth_df.columns:
                    final_df[c] = synth_df[c].to_numpy()[:len(final_df)]

            for c in cond_cols:
                if c in df_input.columns:
                    final_df[c] = df_input[c].to_numpy()[:len(final_df)]

            if args.hard_project_positive:
                for c in target_cols:
                    if c in final_df.columns:
                        final_df[c] = np.maximum(final_df[c].to_numpy(dtype=np.float64), float(args.pos_eps))

            out_dir = f"{get_generated_dir(args.prepared_dir)}/{args.synth_mask}_harpoon/"
            os.makedirs(out_dir, exist_ok=True)
            out_name = os.path.join(out_dir, f"full_imputed_stride_{args.stride}_trial_{trial}.csv")
            final_df.to_csv(out_name, index=False)

            if args.out_csv and trial == 0:
                os.makedirs(os.path.dirname(args.out_csv) if os.path.dirname(args.out_csv) else ".", exist_ok=True)
                final_df.to_csv(args.out_csv, index=False)

            print(f"  Saved: {out_name}")

    # 保存 timing 信息
    out_dir = f"{get_generated_dir(args.prepared_dir)}/{args.synth_mask}_harpoon/"
    os.makedirs(out_dir, exist_ok=True)
    timing_file = os.path.join(out_dir, f"timing_stride_{args.stride}.txt")
    with open(timing_file, "w") as f:
        arr = np.array(exec_times)
        f.write(f"Mean: {np.mean(arr):.2f}s\n")
        f.write(f"Std: {np.std(arr):.2f}s\n")
        f.write(f"All: {exec_times}\n")

    print(f"\n{'='*60}")
    print(f"[DONE] Average exec time: {np.mean(exec_times):.2f}s ± {np.std(exec_times):.2f}s")
    print(f"{'='*60}")
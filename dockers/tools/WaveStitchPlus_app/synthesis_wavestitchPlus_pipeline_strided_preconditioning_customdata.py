import argparse
import os
import json
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
from torch import from_numpy, sqrt
from torch.utils.data import DataLoader

from helper.training_utils import MyDataset, fetchModel, fetchDiffusionConfig
from helper.data_utils import Preprocessor
from helper.metasynth import metadataMask
from pathlib import Path
from custom_pipeline.directory_manager import get_save_dir, get_generated_dir


def ensure_bwc(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Ensure x is [B, W, C]. Accepts [B, W, C] or [B, C, W]."""
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


def recompute_cond_features(df: pd.DataFrame, time_col: str, cond_cols: list, target_cols: list):
    """重新计算条件特征"""
    from custom_pipeline.features import add_time_features, add_gap_structure_features
    
    df = df.copy()
    observed_row_mask = ~df[target_cols].isna().all(axis=1).to_numpy()
    df = add_time_features(df, time_col=time_col)
    df = add_gap_structure_features(df, observed_row_mask)
    
    for c in cond_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    
    return df


def load_custom_prepared(prepared_dir: str):
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


# =============================================================================
# 🔥 新增：支持 EM 模型的推理增强
# =============================================================================

def load_model_with_fallback(saved_dir: str, args, dev):
    """
    按优先级加载模型：
    1. model_em.pth (DiffPuter EM 训练)
    2. model_best.pth (最佳 checkpoint)
    3. model.pth (最终模型)
    """
    candidates = [
        ("model_em.pth", "DiffPuter EM"),
        ("model_best.pth", "Best checkpoint"),
        ("model_prop.pth" if args.propCycEnc else "model.pth", "Standard"),
    ]
    
    for filename, desc in candidates:
        path = os.path.join(saved_dir, filename)
        if os.path.exists(path):
            print(f"[INFO] Loading model: {path} ({desc})")
            return torch.load(path, map_location=dev), path
    
    raise FileNotFoundError(f"No model found in {saved_dir}")


def apply_repaint_guidance(x, test_batch, mask_expanded, non_hier_cols, 
                           alpha_bar_t, num_resample=3):
    """
    🔥 RePaint 风格的重采样（可选增强）
    在每个去噪步骤中多次重采样，更好地融合观测值约束
    """
    # 简化版：直接替换观测值位置
    # 完整版需要在循环中多次 forward-backward
    x_guided = x.clone()
    
    # 观测值位置用原始数据的加噪版本替换
    noise = torch.randn_like(test_batch)
    noised_obs = sqrt(alpha_bar_t) * test_batch + sqrt(1 - alpha_bar_t) * noise
    
    # 只替换非缺失位置
    x_guided[~mask_expanded] = noised_obs[~mask_expanded]
    
    return x_guided


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
    
    # 🔥 新增：推理增强参数
    parser.add_argument("-guidance_scale", type=float, default=0.1,
                        help="Gradient guidance scale (default: 0.1)")
    parser.add_argument("-use_repaint", action="store_true",
                        help="Use RePaint-style resampling for better conditioning")
    parser.add_argument("-repaint_steps", type=int, default=3,
                        help="Number of RePaint resample steps")
    parser.add_argument("-model_type", type=str, default="auto",
                        choices=["auto", "em", "standard"],
                        help="Model type to load")

    args = parser.parse_args()
    dataset = args.dataset
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[INFO] Device: {dev}")
    print(f"[INFO] Guidance scale: {args.guidance_scale}")

    # -----------------------------
    # 1) Load data & masks
    # -----------------------------
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
        rows_to_synth = df_input[target_cols].isna().all(axis=1).to_numpy(dtype=bool)
        
        print(f"[INFO] Total rows: {len(df_input)}, Gap rows to synthesize: {rows_to_synth.sum()}")
        print(f"[INFO] Observation rate: {1 - rows_to_synth.mean():.2%}")

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
        scaler_mean = None
        scaler_std = None
        
        if os.path.exists(scaler_dir):
            mean_path = os.path.join(scaler_dir, "mean.npy")
            std_path = os.path.join(scaler_dir, "std.npy")
            
            if os.path.exists(mean_path) and os.path.exists(std_path):
                print(f"[INFO] Loading scaler for input normalization")
                scaler_mean = np.load(mean_path)
                scaler_std = np.load(std_path)
                
                target_indices = [model_cols.index(c) for c in target_cols if c in model_cols]
                d_vals_raw = df_synth.drop(columns=[time_col]).values.astype(np.float32)
                d_vals_raw[:, target_indices] = (d_vals_raw[:, target_indices] - scaler_mean) / scaler_std
                d_vals_raw[:, target_indices] = np.clip(d_vals_raw[:, target_indices], -3.0, 3.0)
                
                print(f"[INFO] Normalized input data")
            else:
                print(f"[WARNING] Scaler not found, using raw values")
                d_vals_raw = df_synth.drop(columns=[time_col]).values.astype(np.float32)
        else:
            d_vals_raw = df_synth.drop(columns=[time_col]).values.astype(np.float32)

        hierarchical_column_indices = np.array(
            [model_cols.index(c) for c in cond_cols if c in model_cols], dtype=int
        )
        
        print(f"[INFO] Model columns: {model_cols}")
        print(f"[INFO] Cond columns (indices): {hierarchical_column_indices}")
        print(f"[INFO] Target columns: {target_cols}")

        m_vals = rows_to_synth.astype(np.bool_)

        T, C = d_vals_raw.shape
        num_windows = (T - args.window_size) // args.stride + 1
        
        print(f"[DEBUG] d_vals shape: {d_vals_raw.shape}, num_windows: {num_windows}")
        
        windows = torch.zeros(num_windows, args.window_size, C, dtype=torch.float32)
        masks = torch.zeros(num_windows, args.window_size, dtype=torch.bool)
        
        for i in range(num_windows):
            start = i * args.stride
            end = start + args.window_size
            windows[i] = torch.from_numpy(d_vals_raw[start:end, :])
            masks[i] = torch.from_numpy(m_vals[start:end])
        
        print(f"[INFO] Windows shape: {windows.shape}")
        print(f"[INFO] Masks shape: {masks.shape}")

    else:
        # 原有 dataset 逻辑保持不变
        preprocessor = Preprocessor(dataset, args.propCycEnc)
        df = preprocessor.df_cleaned

        end = preprocessor.test_indices[-1]
        start = preprocessor.test_indices[0]
        window_cnt = ((end + 1 - args.window_size - start) // args.stride) + 1
        tilde_start = end + 1 - args.window_size - (window_cnt * args.stride)
        additional_indices = start - tilde_start

        test_df = df.loc[preprocessor.train_indices[-additional_indices:] + preprocessor.test_indices]
        test_df_with_hierarchy = preprocessor.cyclicDecode(test_df)

        metadata = test_df_with_hierarchy[preprocessor.hierarchical_features_uncyclic]
        rows_to_synth = metadataMask(metadata, args.synth_mask, args.dataset)

        df_synth = test_df.copy()
        hierarchical_column_indices = df_synth.columns.get_indexer(
            preprocessor.hierarchical_features_cyclic
        )

        d_vals = df_synth.values.astype(np.float32)
        m_vals = rows_to_synth.values

        d_vals_tensor = from_numpy(d_vals)
        m_vals_tensor = from_numpy(m_vals)

        windows = d_vals_tensor.unfold(0, args.window_size, args.stride)
        masks = m_vals_tensor.unfold(0, args.window_size, args.stride)
        
        model_cols = df_synth.columns.tolist()
        scaler_mean = None
        scaler_std = None
        target_cols = []

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

    # 🔥 改进：使用 fallback 加载模型
    # saved_dir = get_save_dir(args.prepared_dir)
    saved_dir = os.path.join(args.prepared_dir, 'saved_model')

    print("saved_model dir: ", saved_dir)
    
    if args.model_type == "em":
        model_path = os.path.join(saved_dir, "model_em_best.pth")
    elif args.model_type == "standard":
        model_path = os.path.join(saved_dir, "model_prop.pth" if args.propCycEnc else "model.pth")
    else:  # auto
        saved_params, model_path = load_model_with_fallback(saved_dir, args, dev)
    
    if args.model_type != "auto":
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        saved_params = torch.load(model_path, map_location=dev)
        print(f"[INFO] Loading model: {model_path}")

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in saved_params:
                param.copy_(saved_params[name])
            else:
                print(f"[WARNING] Parameter '{name}' not found in checkpoint!")
            param.requires_grad = True
    model.eval()

    num_ops = 0
    exec_times = []

    # -----------------------------
    # 3) Synthesis loop
    # -----------------------------
    for trial in range(args.n_trials):
        start_t = timer()
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
                    
                    # 减少打印频率
                    if step % 50 == 0 or step < 5:
                        print(f"  step={step}")
                    
                    alpha_bar_t = diffusion_config["alpha_bars"][step].to(dev)
                    alpha_bar_t_1 = diffusion_config["alpha_bars"][step - 1].to(dev) if step > 0 else diffusion_config["alpha_bars"][0].to(dev)
                    alpha_t = diffusion_config["alphas"][step].to(dev)
                    beta_t = diffusion_config["betas"][step].to(dev)

                    sampled_noise = create_pipelined_noise(test_batch, args).to(dev)
                    conditional_fwd = sqrt(alpha_bar_t) * test_batch + sqrt(1 - alpha_bar_t) * sampled_noise

                    # 🔥 可选：RePaint 风格重采样
                    if args.use_repaint and step > 10:
                        x = apply_repaint_guidance(
                            x, test_batch, mask_expanded, non_hier_cols,
                            alpha_bar_t, args.repaint_steps
                        )

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
                            vari = 0.0

                        normal_denoising = create_pipelined_noise(test_batch, args).to(dev)
                        normal_denoising[:, :, non_hier_cols] = (
                            x[:, :, non_hier_cols] - (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred
                        ) / torch.sqrt(alpha_t)
                        normal_denoising[:, :, non_hier_cols] += vari

                        rolled_x = normal_denoising.roll(1, 0)
                        rolled_x[0, args.stride:, :] = normal_denoising[0, :(args.window_size - args.stride), :]

                        # Stitching loss
                        loss1 = torch.sum(
                            (normal_denoising[:, :(args.window_size - args.stride), non_hier_cols] -
                             rolled_x[:, args.stride:args.window_size, non_hier_cols]) ** 2,
                            dim=(1, 2)
                        )

                        # Observation guidance loss
                        recon = (x[:, :, non_hier_cols] - torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_bar_t)
                        loss2 = torch.sum(
                            (~mask_expanded[:, :, non_hier_cols]) * (recon - test_batch[:, :, non_hier_cols]) ** 2,
                            dim=(1, 2)
                        )

                        loss = loss1 + loss2
                        grad = torch.autograd.grad(loss, x, grad_outputs=torch.ones_like(loss))[0]

                    x[:, :, non_hier_cols] = normal_denoising[:, :, non_hier_cols]
                    
                    # 🔥 使用可配置的 guidance scale
                    x[:, :, non_hier_cols] = x[:, :, non_hier_cols] + (-args.guidance_scale * grad[:, :, non_hier_cols])

                    if trial == 0:
                        num_ops += 1

                # 最终替换观测值
                x[~mask_expanded] = test_batch[~mask_expanded]

                first_sample = x[0]
                last_timesteps = x[1:, (args.window_size - args.stride):, :]

                if idx == 0:
                    generated = torch.cat((first_sample, last_timesteps.reshape(-1, last_timesteps.shape[2])), dim=0)
                else:
                    generated = x[:, (args.window_size - args.stride):, :].reshape(-1, x.shape[2])

                synth_tensor = torch.cat((synth_tensor, generated), dim=0)

        exec_times.append(timer() - start_t)

        # -----------------------------
        # 4) Output
        # -----------------------------
        if dataset == "custom_csv":
            synth_array = synth_tensor.detach().cpu().numpy()
            
            if scaler_mean is not None and scaler_std is not None:
                print(f"[INFO] Applying inverse transform to target columns")
                target_indices = [model_cols.index(c) for c in target_cols if c in model_cols]
                synth_array[:, target_indices] = synth_array[:, target_indices] * scaler_std + scaler_mean
            
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

            # out_dir = f"{get_generated_dir(args.prepared_dir)}/{args.synth_mask}_wavestitchplus/"
            out_dir = os.path.join(args.prepared_dir, f'generated/{args.synth_mask}')
            print("out_dir: ", out_dir)
            os.makedirs(out_dir, exist_ok=True)
            
            # 🔥 改进：输出文件名包含模型类型
            model_tag = "em" if "em" in model_path.lower() else "std"
            out_name = os.path.join(out_dir, f"imputed_{model_tag}_stride_{args.stride}_trial_{trial}.csv")
            final_df.to_csv(out_name, index=False)

            if args.out_csv and trial == 0:
                os.makedirs(os.path.dirname(args.out_csv) if os.path.dirname(args.out_csv) else ".", exist_ok=True)
                final_df.to_csv(args.out_csv, index=False)

            print(f"[Trial {trial}] Saved: {out_name}")

    out_dir = f"{get_generated_dir(args.prepared_dir)}/{args.synth_mask}_wavestitchplus/"
    os.makedirs(out_dir, exist_ok=True)
    timing_file = os.path.join(out_dir, f"timing_stride_{args.stride}.txt")
    with open(timing_file, "a") as f:
        arr = np.array(exec_times)
        f.write(f"\nMean: {np.mean(arr):.2f}s, Std: {np.std(arr):.2f}s\n")
    
    print(f"\n[DONE] Average exec time: {np.mean(exec_times):.2f}s ± {np.std(exec_times):.2f}s")
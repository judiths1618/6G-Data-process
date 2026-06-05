import argparse
import os
import json

import numpy as np
import pandas as pd
import torch
from torch import from_numpy, optim, nn, randint, normal, sqrt
from torch.utils.data import DataLoader

from helper.data_utils import Preprocessor
from helper.training_utils import MyDataset, fetchModel, fetchDiffusionConfig

import argparse
from pathlib import Path

from custom_pipeline.directory_manager import get_save_dir



class StandardScaler:
    """简单的标准化器"""
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, x):
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
    
    def transform(self, x):
        return (x - self.mean_) / self.std_
    
    def inverse_transform(self, x):
        return x * self.std_ + self.mean_


def load_custom_train_df(prepared_dir: str):

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
        df = df[expected]

    training_df = df.drop(columns=[time_col])
    training_df = training_df.interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
    
    return training_df, cond_cols, target_cols


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()

    
    parser.add_argument("-dataset", "-d", type=str, required=True)
    parser.add_argument("-input_csv", type=str, default=None)
    parser.add_argument("-prepared_dir", type=str, default="./work/prepared")


    parser.add_argument("-backbone", type=str, default="S4")
    parser.add_argument("-beta_0", type=float, default=0.0001)
    parser.add_argument("-beta_T", type=float, default=0.02)
    parser.add_argument("-timesteps", "-T", type=int, default=200)
    parser.add_argument("-hdim", type=int, default=64)
    parser.add_argument("-lr", type=float, default=1e-3)  # 增加学习率
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
    parser.add_argument("-s4_bidirectional", type=bool, default=True)
    parser.add_argument("-s4_layernorm", type=bool, default=True)
    parser.add_argument("-propCycEnc", type=bool, default=False)
    parser.add_argument("-normalize", type=bool, default=True, help="Normalize target columns")

    args = parser.parse_args()
    dataset = args.dataset
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    if args.dataset == "custom_csv":    # under the folder datasets/EUR/...
        from custom_pipeline.preprocess import preprocess_csv

        prepared_dir = args.prepared_dir or "./work/prepared"
        
        if args.input_csv:
            preprocess_csv(input_csv=args.input_csv,
                           output_dir=prepared_dir,
                           time_col=None,
                           base_dt=None,
                           extract_main_segment=True,
                           skip_regularize_if_sparse=True,
                           convert_units=True,
            )

        training_df, cond_cols, target_cols = load_custom_train_df(prepared_dir)
        
        print(f"[INFO] Training data shape: {training_df.shape}")
        print(f"[INFO] Target columns: {target_cols}")
        print(f"[INFO] Cond columns: {cond_cols}")
        
        # 🔥 关键改进：标准化 target columns
        d_vals = training_df.values.astype(np.float32)
        
        if args.normalize:
            target_indices = [training_df.columns.get_loc(c) for c in target_cols if c in training_df.columns]
            
            print(f"[INFO] Original data statistics:")
            for i, idx in enumerate(target_indices):
                col_name = target_cols[i] if i < len(target_cols) else f"col_{idx}"
                col_data = d_vals[:, idx]
                print(f"   {col_name:20s}: min={np.min(col_data):12.2f}, max={np.max(col_data):12.2f}, "
                      f"mean={np.mean(col_data):12.2f}, std={np.std(col_data):12.2f}")
            
            # 先裁剪极端异常值（使用更激进的分位数：1%-99%）
            for idx in target_indices:
                col_data = d_vals[:, idx]
                p_low = np.percentile(col_data, 1.0)   # 1% 而非 0.1%
                p_high = np.percentile(col_data, 99.0) # 99% 而非 99.9%
                d_vals[:, idx] = np.clip(col_data, p_low, p_high)
            
            scaler = StandardScaler()
            scaler.fit(d_vals[:, target_indices])
            d_vals[:, target_indices] = scaler.transform(d_vals[:, target_indices])
            
            # 再次裁剪到合理范围（-3σ 到 +3σ，而非 5σ）
            d_vals[:, target_indices] = np.clip(d_vals[:, target_indices], -3.0, 3.0)
            
            # 保存 scaler
            scaler_dir = os.path.join(prepared_dir, "scaler")
            os.makedirs(scaler_dir, exist_ok=True)
            np.save(os.path.join(scaler_dir, "mean.npy"), scaler.mean_)
            np.save(os.path.join(scaler_dir, "std.npy"), scaler.std_)
            
            print(f"\n[INFO] Normalized target columns:")
            print(f"   Mean: {scaler.mean_}")
            print(f"   Std: {scaler.std_}")
        
        # 检查数据范围
        print(f"[INFO] Data range after normalization:")
        print(f"   Min: {d_vals.min():.3f}, Max: {d_vals.max():.3f}")
        print(f"   Mean: {d_vals.mean():.3f}, Std: {d_vals.std():.3f}")
        
        hierarchical_column_indices = training_df.columns.get_indexer(cond_cols)

        # 构建窗口
        T, C = d_vals.shape
        num_windows = (T - args.window_size) // args.stride + 1
        
        training_samples = torch.zeros(num_windows, args.window_size, C, dtype=torch.float32)
        
        for i in range(num_windows):
            start = i * args.stride
            end = start + args.window_size
            training_samples[i] = torch.from_numpy(d_vals[start:end, :])
        
        print(f"[INFO] Training samples shape: {training_samples.shape}")

    else:
        preprocessor = Preprocessor(dataset, args.propCycEnc)
        df = preprocessor.df_cleaned
        training_df = df.loc[preprocessor.train_indices]
        hierarchical_column_indices = training_df.columns.get_indexer(preprocessor.hierarchical_features_cyclic)
        
        d_vals_tensor = from_numpy(training_df.values)
        training_samples = d_vals_tensor.unfold(0, args.window_size, args.stride)

    # Setup model
    in_dim = training_samples.shape[2]
    out_dim = in_dim - len(hierarchical_column_indices)

    print(f"[INFO] in_dim={in_dim}, out_dim={out_dim}")

    training_dataset = MyDataset(training_samples.float(), window_size=args.window_size)
    # print(training_dataset[100])
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = fetchModel(in_dim, out_dim, args).to(dev)
    diffusion_config = fetchDiffusionConfig(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    all_indices = np.arange(in_dim)
    non_hier_cols = np.setdiff1d(all_indices, hierarchical_column_indices)

    print(f"[INFO] Non-hierarchical columns: {non_hier_cols}")
    print(f"[INFO] Starting training with lr={args.lr}")

    # exit()
    # Training loop
    alpha_bars = diffusion_config["alpha_bars"].to(dev)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = batch.to(dev)

            timesteps = randint(diffusion_config["T"], size=(batch.shape[0],), device=dev)
            sigmas = normal(0, 1, size=batch.shape).to(dev)

            coeff_1 = sqrt(alpha_bars[timesteps]).reshape((len(timesteps), 1, 1))
            coeff_2 = sqrt(1 - alpha_bars[timesteps]).reshape((len(timesteps), 1, 1))

            conditional_mask = torch.ones_like(batch, device=dev)
            conditional_mask[:, :, non_hier_cols] = 0.0

            batch_noised = (1 - conditional_mask) * (coeff_1 * batch + coeff_2 * sigmas) + conditional_mask * batch

            times = timesteps.reshape((-1, 1))
            sigmas_predicted = model(batch_noised, times)

            optimizer.zero_grad()

            sigmas_gt = sigmas[:, :, non_hier_cols].permute((0, 2, 1)).to(dev)

            loss = criterion(sigmas_predicted, sigmas_gt)
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        
        # 更新学习率
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"[LR] Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            # save_dir = f"saved_models/EUR/{args.dataset}/"
            save_dir = get_save_dir(args.prepared_dir)
            os.makedirs(save_dir, exist_ok=True)
            filename = "model_prop_best.pth" if args.propCycEnc else "model_best.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, filename))
        
        if epoch % 10 == 0 or epoch < 5:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"epoch: {epoch:4d}, avg_loss: {avg_loss:.6f}, best_loss: {best_loss:.6f}, lr: {current_lr:.6f}")

    # Save final model
    # save_dir = f"saved_models/{args.dataset}/"
    save_dir = get_save_dir(args.prepared_dir)
    print(f"[INFO] Auto-generated save_dir: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    filename = "model_prop.pth" if args.propCycEnc else "model.pth"
    filepath = os.path.join(save_dir, filename)

    torch.save(model.state_dict(), filepath)
    print(f"\n[DONE] Saved final model to: {filepath}")
    # ============ 文件整理 ============
    import shutil
    from datetime import datetime
    
    print(f"\n[POST-TRAINING] Organizing files...")
    
    # 确保模型在 prepared_dir/saved_model 中
    expected_model_dir = os.path.join(prepared_dir, "saved_model")
    
    if os.path.abspath(save_dir) != os.path.abspath(expected_model_dir):
        if os.path.exists(save_dir):
            if os.path.exists(expected_model_dir):
                shutil.rmtree(expected_model_dir)
            shutil.copytree(save_dir, expected_model_dir)
            print(f"[POST] ✓ Moved models to: {expected_model_dir}")
    
    # 验证文件
    model_files = [f for f in os.listdir(expected_model_dir) if f.endswith('.pth')]
    print(f"[POST] ✓ Found {len(model_files)} model(s): {model_files}")
    
    # 保存完成标记
    completion_info = {
        "completed_at": datetime.now().isoformat(),
        "model_count": len(model_files),
        "ready": True
    }
    with open(os.path.join(prepared_dir, "training_completed.json"), 'w') as f:
        json.dump(completion_info, f, indent=2)
    
    print(f"[DONE] Training completed! Files in: {prepared_dir}")
    # print(f"       Saved best model to: {os.path.join(save_dir, 'model_best.pth')}")
    # print(f"       Final loss: {avg_loss:.6f}, Best loss: {best_loss:.6f}")
    # print(f"Model trained with in_dim={in_dim}, out_dim={out_dim}")
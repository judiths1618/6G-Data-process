# import argparse
# import os
# import json
# from copy import deepcopy

# import numpy as np
# import torch
# from torch import from_numpy, optim, nn, randint, normal, sqrt
# from torch.utils.data import DataLoader

# from helper.data_utils import Preprocessor
# from helper.training_utils import MyDataset, fetchModel, fetchDiffusionConfig
# from custom_pipeline.directory_manager import get_save_dir, get_generated_dir


# class StandardScaler:
#     """简单的标准化器"""
#     def __init__(self):
#         self.mean_ = None
#         self.std_ = None
    
#     def fit(self, x):
#         self.mean_ = np.mean(x, axis=0)
#         self.std_ = np.std(x, axis=0)
#         self.std_[self.std_ < 1e-8] = 1.0
    
#     def transform(self, x):
#         return (x - self.mean_) / self.std_
    
#     def inverse_transform(self, x):
#         return x * self.std_ + self.mean_


# def load_custom_train_df(prepared_dir: str):
#     """加载数据，保留缺失值标记"""
#     import pandas as pd

#     meta_path = os.path.join(prepared_dir, "meta.json")
#     train_path = os.path.join(prepared_dir, "train.csv")
    
#     with open(meta_path, "r") as f:
#         meta = json.load(f)

#     time_col = meta.get("time_col", "time")
#     cond_cols = meta.get("cond_cols", [])
#     target_cols = meta.get("target_cols", [])
#     model_cols = meta.get("all_model_cols", None)

#     df = pd.read_csv(train_path)

#     if model_cols is not None:
#         expected = [time_col] + model_cols
#         df = df[expected]

#     training_df = df.drop(columns=[time_col])
    
#     # 🔥 关键改动：不再直接填充，而是保留缺失位置信息
#     # 创建观测掩码 (1 = 观测值存在, 0 = 缺失)
#     obs_mask = (~training_df[target_cols].isna()).astype(np.float32)
    
#     # 初始填充mean（用于 EM 迭代的起点）
#     training_df_filled = training_df.copy()
#     training_df_filled = training_df_filled.interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
    
#     return training_df_filled, cond_cols, target_cols, obs_mask


# class DiffPuterEMTrainer:
#     """DiffPuter EM 训练器（内存优化版）"""
    
#     def __init__(self, model, diffusion_config, device, 
#                  non_hier_cols, hier_cols, target_indices,
#                  lr=1e-3, em_iterations=5):
#         self.model = model
#         self.diffusion_config = diffusion_config
#         self.device = device
#         self.non_hier_cols = np.array(non_hier_cols)
#         self.hier_cols = np.array(hier_cols)
#         self.target_indices = np.array(target_indices)
#         self.lr = lr
#         self.em_iterations = em_iterations
        
#         self.alpha_bars = diffusion_config["alpha_bars"].to(device)
#         self.betas = diffusion_config["betas"].to(device)
#         self.alphas = 1 - self.betas
#         self.T = diffusion_config["T"]
    
#     def m_step(self, dataloader, epochs_per_iter=200):
#         """M-step: 训练扩散模型"""
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='min', factor=0.5, patience=20
#         )
        
#         best_loss = float('inf')
        
#         for epoch in range(epochs_per_iter):
#             total_loss = 0.0
#             n_batches = 0
            
#             for batch, mask in dataloader:
#                 batch = batch.to(self.device)
#                 mask = mask.to(self.device)
                
#                 timesteps = torch.randint(self.T, size=(batch.shape[0],), device=self.device)
#                 sigmas = torch.randn_like(batch)
                
#                 coeff_1 = torch.sqrt(self.alpha_bars[timesteps]).reshape((-1, 1, 1))
#                 coeff_2 = torch.sqrt(1 - self.alpha_bars[timesteps]).reshape((-1, 1, 1))
                
#                 conditional_mask = torch.ones_like(batch, device=self.device)
#                 conditional_mask[:, :, self.non_hier_cols] = 0.0
                
#                 batch_noised = (1 - conditional_mask) * (coeff_1 * batch + coeff_2 * sigmas) \
#                               + conditional_mask * batch
                
#                 times = timesteps.reshape((-1, 1))
#                 sigmas_predicted = self.model(batch_noised, times)
                
#                 optimizer.zero_grad()
                
#                 sigmas_gt = sigmas[:, :, self.non_hier_cols].permute((0, 2, 1))
#                 mask_permuted = mask.permute((0, 2, 1))
                
#                 loss_per_element = (sigmas_predicted - sigmas_gt) ** 2
#                 weighted_loss = loss_per_element * (0.2 + 0.8 * mask_permuted)
#                 loss = weighted_loss.mean()
                
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#                 optimizer.step()
                
#                 total_loss += loss.item()
#                 n_batches += 1
                
#                 # 清理
#                 del batch_noised, sigmas_predicted, sigmas_gt
            
#             avg_loss = total_loss / max(1, n_batches)
#             scheduler.step(avg_loss)
            
#             if avg_loss < best_loss:
#                 best_loss = avg_loss
            
#             if epoch % 50 == 0:
#                 print(f"    M-step epoch {epoch:3d}, loss: {avg_loss:.6f}")
            
#             # 定期清理 GPU 缓存
#             if epoch % 10 == 0:
#                 torch.cuda.empty_cache()
        
#         return best_loss
    
#     @torch.no_grad()
#     def e_step_fast(self, data_tensor, obs_mask, window_size, stride, 
#                     batch_size=32, num_samples=1, ddim_steps=50):
#         """内存优化的 E-step"""
#         self.model.eval()
#         torch.cuda.empty_cache()
        
#         T_seq, C = data_tensor.shape
#         num_windows = (T_seq - window_size) // stride + 1
        
#         print(f"    [E-step] {num_windows} windows, batch={batch_size}, ddim={ddim_steps}")
        
#         # CPU 上的累加器
#         imputed_sum = np.zeros((T_seq, C), dtype=np.float32)
#         imputed_count = np.zeros((T_seq, 1), dtype=np.float32)
        
#         num_batches = (num_windows + batch_size - 1) // batch_size
        
#         for sample_idx in range(num_samples):
#             for batch_idx in range(num_batches):
#                 batch_start = batch_idx * batch_size
#                 batch_end = min(batch_start + batch_size, num_windows)
#                 actual_batch_size = batch_end - batch_start
                
#                 if batch_idx % 50 == 0:
#                     mem_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
#                     print(f"      Batch {batch_idx+1}/{num_batches}, GPU mem: {mem_used:.2f} GB")
                
#                 # 构建 batch
#                 batch_data = np.zeros((actual_batch_size, window_size, C), dtype=np.float32)
#                 batch_mask = np.zeros((actual_batch_size, window_size, len(self.target_indices)), 
#                                       dtype=np.float32)
                
#                 for i in range(actual_batch_size):
#                     win_idx = batch_start + i
#                     start = win_idx * stride
#                     end = start + window_size
#                     batch_data[i] = data_tensor[start:end]
#                     batch_mask[i] = obs_mask[start:end]
                
#                 # GPU 采样
#                 batch_data_t = torch.from_numpy(batch_data).to(self.device)
#                 batch_mask_t = torch.from_numpy(batch_mask).to(self.device)
                
#                 sampled = self._conditional_sample_ddim(batch_data_t, batch_mask_t, ddim_steps)
                
#                 # 立即转回 CPU
#                 sampled_np = sampled.cpu().numpy()
                
#                 # 释放 GPU 内存
#                 del sampled, batch_data_t, batch_mask_t
#                 torch.cuda.empty_cache()
                
#                 # 累加
#                 for i in range(actual_batch_size):
#                     win_idx = batch_start + i
#                     start = win_idx * stride
#                     end = start + window_size
#                     imputed_sum[start:end] += sampled_np[i]
#                     imputed_count[start:end] += 1
        
#         # 平均
#         imputed_count = np.maximum(imputed_count, 1)
#         imputed = imputed_sum / imputed_count
        
#         # 只更新缺失位置
#         obs_mask_full = np.zeros((T_seq, C))
#         obs_mask_full[:, self.target_indices] = obs_mask
        
#         new_data = data_tensor.copy()
#         missing_mask = (obs_mask_full == 0)
#         new_data[missing_mask] = imputed[missing_mask]
        
#         self.model.train()
#         torch.cuda.empty_cache()
        
#         return new_data
    
#     def _conditional_sample_ddim(self, batch_data, batch_mask, ddim_steps=50):
#         """DDIM 采样"""
#         B, T, C = batch_data.shape
        
#         x_t = torch.randn_like(batch_data)
        
#         # 掩码
#         cond_mask = torch.zeros_like(batch_data, device=self.device)
#         cond_mask[:, :, self.hier_cols] = 1.0
        
#         obs_mask_full = torch.zeros_like(batch_data, device=self.device)
#         obs_mask_full[:, :, self.target_indices] = batch_mask
        
#         ddim_timesteps = np.linspace(0, self.T - 1, ddim_steps, dtype=int)[::-1]
        
#         for i, t in enumerate(ddim_timesteps):
#             t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
            
#             x_input = cond_mask * batch_data + (1 - cond_mask) * x_t
            
#             noise_pred = self.model(x_input, t_tensor.reshape(-1, 1))
#             noise_pred = noise_pred.permute(0, 2, 1)
            
#             noise_full = torch.zeros_like(x_t)
#             noise_full[:, :, self.non_hier_cols] = noise_pred
            
#             alpha_bar_t = self.alpha_bars[t]
            
#             x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_full) / torch.sqrt(alpha_bar_t)
#             x_0_pred = torch.clamp(x_0_pred, -3.0, 3.0)
            
#             if i < len(ddim_timesteps) - 1:
#                 t_next = ddim_timesteps[i + 1]
#                 alpha_bar_next = self.alpha_bars[t_next]
#                 x_t = torch.sqrt(alpha_bar_next) * x_0_pred + \
#                       torch.sqrt(1 - alpha_bar_next) * noise_full
#             else:
#                 x_t = x_0_pred
            
#             x_t = obs_mask_full * batch_data + (1 - obs_mask_full) * x_t
#             x_t = cond_mask * batch_data + (1 - cond_mask) * x_t
            
#             del noise_pred, noise_full, x_0_pred
        
#         return x_t
    
#     def _build_dataloader(self, data_np, obs_mask_np, window_size, stride, batch_size):
#         """构建 DataLoader"""
#         T, C = data_np.shape
#         num_windows = (T - window_size) // stride + 1
        
#         samples = torch.zeros(num_windows, window_size, C, dtype=torch.float32)
#         masks = torch.zeros(num_windows, window_size, len(self.target_indices), dtype=torch.float32)
        
#         for i in range(num_windows):
#             start = i * stride
#             end = start + window_size
#             samples[i] = torch.from_numpy(data_np[start:end])
#             masks[i] = torch.from_numpy(obs_mask_np[start:end])
        
#         dataset = EMDataset(samples, masks)
#         return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
#     def train_em(self, data_np, obs_mask_np, window_size, stride, 
#                  batch_size, epochs_per_iter=200,
#                  e_step_batch_size=32, ddim_steps=50):
#         """EM 训练主循环"""
        
#         current_data = data_np.copy()
        
#         for em_iter in range(self.em_iterations):
#             print(f"\n{'='*60}")
#             print(f"EM Iteration {em_iter + 1}/{self.em_iterations}")
#             print(f"{'='*60}")
            
#             # 清理 GPU
#             torch.cuda.empty_cache()
            
#             dataloader = self._build_dataloader(
#                 current_data, obs_mask_np, window_size, stride, batch_size
#             )
            
#             print(f"  [M-step] Training...")
#             m_loss = self.m_step(dataloader, epochs_per_iter)
#             print(f"  [M-step] Done, loss: {m_loss:.6f}")
            
#             if em_iter < self.em_iterations - 1:
#                 print(f"  [E-step] Updating estimates...")
#                 current_data = self.e_step_fast(
#                     current_data, obs_mask_np, window_size, stride,
#                     batch_size=e_step_batch_size,
#                     num_samples=1,
#                     ddim_steps=ddim_steps
#                 )
                
#                 # 统计更新幅度
#                 diff = np.abs(current_data - data_np)
#                 target_diff = diff[:, self.target_indices]
#                 missing_mask = (obs_mask_np == 0)
#                 if missing_mask.sum() > 0:
#                     missing_diff = target_diff[missing_mask]
#                     print(f"  [E-step] Update: mean={missing_diff.mean():.4f}, "
#                           f"max={missing_diff.max():.4f}")
        
#         return current_data


# class EMDataset(torch.utils.data.Dataset):
#     """支持观测掩码的数据集"""
#     def __init__(self, samples, masks):
#         self.samples = samples
#         self.masks = masks
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         return self.samples[idx], self.masks[idx]


# if __name__ == "__main__":
#     np.random.seed(42)
#     torch.manual_seed(42)

#     parser = argparse.ArgumentParser()
    
#     parser.add_argument("-dataset", "-d", type=str, required=True)
#     parser.add_argument("-input_csv", type=str, default=None)
#     parser.add_argument("-prepared_dir", type=str, default="./work/prepared")
    
#     # 模型参数

#     parser.add_argument("-backbone", type=str, default="S4")
#     parser.add_argument("-beta_0", type=float, default=0.0001)
#     parser.add_argument("-beta_T", type=float, default=0.02)
#     parser.add_argument("-timesteps", "-T", type=int, default=200)
#     parser.add_argument("-hdim", type=int, default=64)
#     parser.add_argument("-lr", type=float, default=1e-3)
#     parser.add_argument("-batch_size", type=int, default=1024)
#     parser.add_argument("-epochs", type=int, default=1000)
#     parser.add_argument("-layers", type=int, default=4)
#     parser.add_argument("-window_size", type=int, default=32)
#     parser.add_argument("-stride", type=int, default=1)
#     parser.add_argument("-num_res_layers", type=int, default=4)
#     parser.add_argument("-res_channels", type=int, default=64)
#     parser.add_argument("-skip_channels", type=int, default=64)
#     parser.add_argument("-diff_step_embed_in", type=int, default=32)
#     parser.add_argument("-diff_step_embed_mid", type=int, default=64)
#     parser.add_argument("-diff_step_embed_out", type=int, default=64)
#     parser.add_argument("-s4_lmax", type=int, default=100)
#     parser.add_argument("-s4_dstate", type=int, default=64)
#     parser.add_argument("-s4_dropout", type=float, default=0.0)
#     parser.add_argument("-s4_bidirectional", type=bool, default=True)
#     parser.add_argument("-s4_layernorm", type=bool, default=True)
#     parser.add_argument("-propCycEnc", type=bool, default=False)
#     parser.add_argument("-normalize", type=bool, default=True)
    
#     # 🔥 新增：EM 相关参数
#     parser.add_argument("-em_iterations", type=int, default=5, 
#                         help="Number of EM iterations (DiffPuter style)")
#     parser.add_argument("-epochs_per_em", type=int, default=200,
#                         help="Training epochs per EM iteration")
#     parser.add_argument("-use_em", action="store_true",
#                         help="Use DiffPuter-style EM training")
#         # E-step 优化参数
#     parser.add_argument("-e_step_batch_size", type=int, default=256,
#                         help="Batch size for E-step sampling")
#     parser.add_argument("-ddim_steps", type=int, default=10,
#                         help="DDIM steps for E-step (faster than full T)")
#     parser.add_argument("-e_step_samples", type=int, default=1,
#                         help="Number of samples per E-step (1 is usually enough)")

#     args = parser.parse_args()
#     dataset = args.dataset
#     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     print(f"[INFO] Device: {dev}")
#     print(f"[INFO] EM training: {args.use_em}")

#     # Load data
#     if args.dataset == "custom_csv":
#         from custom_pipeline.preprocess import preprocess_csv

#         prepared_dir = args.prepared_dir or "./work/prepared"
        
#         if args.input_csv:
#             preprocess_csv(
#                 input_csv=args.input_csv, 
#                 output_dir=prepared_dir, 
#                 time_col=None, 
#                 base_dt=None,
#                 extract_main_segment=True,
#                 skip_regularize_if_sparse=True
#             )

#         # 🔥 改用新的加载函数
#         training_df, cond_cols, target_cols, obs_mask = load_custom_train_df(prepared_dir)
        
#         print(f"[INFO] Training data shape: {training_df.shape}")
#         print(f"[INFO] Target columns ({len(target_cols)}): {target_cols}")
#         print(f"[INFO] Cond columns ({len(cond_cols)}): {cond_cols}")
        
#         # 计算缺失率
#         obs_rate = obs_mask.values.mean()
#         print(f"[INFO] Observation rate: {obs_rate:.2%} (missing: {1-obs_rate:.2%})")
        
#         d_vals = training_df.values.astype(np.float32)
#         obs_mask_np = obs_mask.values.astype(np.float32)
        
#         # 标准化
#         if args.normalize:
#             target_indices = [training_df.columns.get_loc(c) for c in target_cols 
#                             if c in training_df.columns]
            
#             # 只用观测值计算统计量
#             observed_data = d_vals[:, target_indices].copy()
#             observed_data[obs_mask_np == 0] = np.nan
            
#             scaler = StandardScaler()
#             # 用观测值的均值和标准差
#             scaler.mean_ = np.nanmean(observed_data, axis=0)
#             scaler.std_ = np.nanstd(observed_data, axis=0)
#             scaler.std_[scaler.std_ < 1e-8] = 1.0
            
#             d_vals[:, target_indices] = (d_vals[:, target_indices] - scaler.mean_) / scaler.std_
#             d_vals[:, target_indices] = np.clip(d_vals[:, target_indices], -3.0, 3.0)
            
#             # 保存 scaler
#             scaler_dir = os.path.join(prepared_dir, "scaler")
#             os.makedirs(scaler_dir, exist_ok=True)
#             np.save(os.path.join(scaler_dir, "mean.npy"), scaler.mean_)
#             np.save(os.path.join(scaler_dir, "std.npy"), scaler.std_)
            
#             print(f"[INFO] Normalized using observed values only")
        
#         hierarchical_column_indices = training_df.columns.get_indexer(cond_cols)
#         target_indices = [training_df.columns.get_loc(c) for c in target_cols 
#                          if c in training_df.columns]

#     else:
#         # 原有逻辑...
#         preprocessor = Preprocessor(dataset, args.propCycEnc)
#         df = preprocessor.df_cleaned
#         training_df = df.loc[preprocessor.train_indices]
#         hierarchical_column_indices = training_df.columns.get_indexer(
#             preprocessor.hierarchical_features_cyclic
#         )
#         d_vals = training_df.values.astype(np.float32)
#         # obs_mask_np = np.ones((d_vals.shape[0], len(target_cols)), dtype=np.float32)
#         # target_indices = list(range(d_vals.shape[1] - len(hierarchical_column_indices)))

#     # Setup model
#     in_dim = d_vals.shape[1]
#     out_dim = in_dim - len(hierarchical_column_indices)
    
#     print(f"[INFO] in_dim={in_dim}, out_dim={out_dim}")

#     model = fetchModel(in_dim, out_dim, args).to(dev)
#     diffusion_config = fetchDiffusionConfig(args)
    
#     all_indices = np.arange(in_dim)
#     non_hier_cols = np.setdiff1d(all_indices, hierarchical_column_indices)
    
#     print(f"[INFO] Hierarchical (cond) columns: {list(hierarchical_column_indices)}")
#     print(f"[INFO] Non-hierarchical (target) columns: {list(non_hier_cols)}")

#     if args.use_em:
#         # 🔥 DiffPuter 风格的 EM 训练
#         print(f"\n{'='*60}")
#         print(f"Starting DiffPuter-style EM Training")
#         print(f"  EM iterations: {args.em_iterations}")
#         print(f"  Epochs per iteration: {args.epochs_per_em}")
#         print(f"{'='*60}")
        
#         trainer = DiffPuterEMTrainer(
#             model=model,
#             diffusion_config=diffusion_config,
#             device=dev,
#             non_hier_cols=non_hier_cols,
#             hier_cols=hierarchical_column_indices,
#             target_indices=target_indices,
#             lr=args.lr,
#             em_iterations=args.em_iterations
#         )
        
#         final_data = trainer.train_em(
#             data_np=d_vals,
#             obs_mask_np=obs_mask_np,
#             window_size=args.window_size,
#             stride=args.stride,
#             batch_size=args.batch_size,
#             epochs_per_iter=args.epochs_per_em,
#             e_step_batch_size=args.e_step_batch_size,  # 新增
#             ddim_steps=args.ddim_steps  # 新增
#         )
        
#         # 保存最终的 imputed 数据
#         imputed_path = os.path.join(prepared_dir, "train_imputed.npy")
#         np.save(imputed_path, final_data)
#         print(f"[INFO] Saved imputed data to: {imputed_path}")
        
#     else:
#         # 原有的单次训练逻辑
#         T_seq, C = d_vals.shape
#         num_windows = (T_seq - args.window_size) // args.stride + 1
        
#         training_samples = torch.zeros(num_windows, args.window_size, C, dtype=torch.float32)
#         for i in range(num_windows):
#             start = i * args.stride
#             end = start + args.window_size
#             training_samples[i] = torch.from_numpy(d_vals[start:end])
        
#         training_dataset = MyDataset(training_samples.float(), window_size=args.window_size)
#         dataloader = DataLoader(training_dataset, batch_size=args.batch_size, 
#                                shuffle=True, drop_last=True)
        
#         optimizer = optim.Adam(model.parameters(), lr=args.lr)
#         criterion = nn.MSELoss()
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='min', factor=0.5, patience=20
#         )
        
#         alpha_bars = diffusion_config["alpha_bars"].to(dev)
#         best_loss = float('inf')
        
#         for epoch in range(args.epochs):
#             total_loss = 0.0
#             n_batches = 0
            
#             for batch in dataloader:
#                 batch = batch.to(dev)
                
#                 timesteps = randint(diffusion_config["T"], size=(batch.shape[0],), device=dev)
#                 sigmas = normal(0, 1, size=batch.shape).to(dev)
                
#                 coeff_1 = sqrt(alpha_bars[timesteps]).reshape((-1, 1, 1))
#                 coeff_2 = sqrt(1 - alpha_bars[timesteps]).reshape((-1, 1, 1))
                
#                 conditional_mask = torch.ones_like(batch, device=dev)
#                 conditional_mask[:, :, non_hier_cols] = 0.0
                
#                 batch_noised = (1 - conditional_mask) * (coeff_1 * batch + coeff_2 * sigmas) \
#                               + conditional_mask * batch
                
#                 times = timesteps.reshape((-1, 1))
#                 sigmas_predicted = model(batch_noised, times)
                
#                 optimizer.zero_grad()
#                 sigmas_gt = sigmas[:, :, non_hier_cols].permute((0, 2, 1)).to(dev)
#                 loss = criterion(sigmas_predicted, sigmas_gt)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
                
#                 total_loss += loss.item()
#                 n_batches += 1
            
#             avg_loss = total_loss / max(1, n_batches)
#             scheduler.step(avg_loss)
            
#             if avg_loss < best_loss:
#                 best_loss = avg_loss
#                 save_dir = get_save_dir(args.prepared_dir)
#                 os.makedirs(save_dir, exist_ok=True)
#                 torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pth"))
            
#             if epoch % 10 == 0:
#                 print(f"epoch: {epoch:4d}, avg_loss: {avg_loss:.6f}, best: {best_loss:.6f}")

#     # Save final model
#     save_dir = get_save_dir(args.prepared_dir)
#     os.makedirs(save_dir, exist_ok=True)
#     filename = "model_em.pth" if args.use_em else "model.pth"
#     torch.save(model.state_dict(), os.path.join(save_dir, filename))
#     print(f"\n[DONE] Saved model to: {os.path.join(save_dir, filename)}")

import argparse
import os
import json
from copy import deepcopy

import numpy as np
import torch
from torch import from_numpy, optim, nn, randint, normal, sqrt
from torch.utils.data import DataLoader

from helper.data_utils import Preprocessor
from helper.training_utils import MyDataset, fetchModel, fetchDiffusionConfig
from custom_pipeline.directory_manager import get_save_dir, get_generated_dir


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
    """
    加载数据，保留缺失值标记
    
    返回:
        training_df_filled: 填充后的 DataFrame
        cond_cols: 条件列名列表
        target_cols: 目标列名列表
        obs_mask: 观测掩码 DataFrame (1=观测, 0=缺失)，只包含 target_cols
    """
    import pandas as pd

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
    
    # 创建观测掩码 (1 = 观测值存在, 0 = 缺失)
    # 只针对 target_cols
    obs_mask = (~training_df[target_cols].isna()).astype(np.float32)
    
    # 初始填充
    training_df_filled = training_df.copy()
    
    # Target 列：用插值填充（适合时序数据）
    for col in target_cols:
        if col in training_df_filled.columns:
            # 先尝试线性插值
            training_df_filled[col] = (
                training_df_filled[col]
                .interpolate(method='linear', limit_direction='both')
                .ffill()
                .bfill()
            )
            # 如果还有 NaN（例如全列缺失），用列均值或 0 填充
            if training_df_filled[col].isna().any():
                col_mean = training_df[col].mean()
                fill_val = col_mean if not np.isnan(col_mean) else 0.0
                training_df_filled[col] = training_df_filled[col].fillna(fill_val)
    
    # Cond 列：应该已经完整，但以防万一
    for col in cond_cols:
        if col in training_df_filled.columns:
            training_df_filled[col] = training_df_filled[col].fillna(0.0)
    
    # 最终确保没有 NaN
    remaining_nan = training_df_filled.isna().sum().sum()
    if remaining_nan > 0:
        print(f"[WARNING] {remaining_nan} NaN values remaining, filling with 0")
        training_df_filled = training_df_filled.fillna(0.0)
    
    # 验证
    print(f"[DEBUG] training_df_filled shape: {training_df_filled.shape}")
    print(f"[DEBUG] obs_mask shape: {obs_mask.shape}")
    print(f"[DEBUG] obs_mask columns: {list(obs_mask.columns)}")
    print(f"[DEBUG] target_cols: {target_cols}")
    
    # 确保 obs_mask 列与 target_cols 一致
    assert list(obs_mask.columns) == target_cols, \
        f"obs_mask columns mismatch! Expected {target_cols}, got {list(obs_mask.columns)}"
    
    return training_df_filled, cond_cols, target_cols, obs_mask


class DiffPuterEMTrainer:
    """DiffPuter EM 训练器（内存优化版）"""
    
    def __init__(self, model, diffusion_config, device, 
                 non_hier_cols, hier_cols, target_indices,
                 lr=1e-3, em_iterations=5):
        """
        参数:
            model: 扩散模型
            diffusion_config: 扩散配置
            device: 设备
            non_hier_cols: 非层级列索引（target 列在 d_vals 中的索引）
            hier_cols: 层级列索引（cond 列在 d_vals 中的索引）
            target_indices: target 列在 d_vals 中的索引（应该与 non_hier_cols 相同）
            lr: 学习率
            em_iterations: EM 迭代次数
        """
        self.model = model
        self.diffusion_config = diffusion_config
        self.device = device
        self.non_hier_cols = np.array(non_hier_cols)
        self.hier_cols = np.array(hier_cols)
        self.target_indices = np.array(target_indices)
        self.lr = lr
        self.em_iterations = em_iterations
        
        self.alpha_bars = diffusion_config["alpha_bars"].to(device)
        self.betas = diffusion_config["betas"].to(device)
        self.alphas = 1 - self.betas
        self.T = diffusion_config["T"]
        
        # 验证
        print(f"[DEBUG] DiffPuterEMTrainer initialized:")
        print(f"  non_hier_cols (target): {self.non_hier_cols}")
        print(f"  hier_cols (cond): {self.hier_cols}")
        print(f"  target_indices: {self.target_indices}")
        print(f"  T (diffusion steps): {self.T}")
    
    def m_step(self, dataloader, obs_mask_windows, epochs_per_iter=200):
        """
        M-step: 训练扩散模型
        
        参数:
            dataloader: 数据加载器，返回 (batch_data, batch_mask)
            obs_mask_windows: 未使用（mask 已经在 dataloader 中）
            epochs_per_iter: 每次 EM 迭代的训练 epoch 数
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )
        
        best_loss = float('inf')
        
        for epoch in range(epochs_per_iter):
            total_loss = 0.0
            n_batches = 0
            
            for batch, mask in dataloader:
                batch = batch.to(self.device)
                mask = mask.to(self.device)  # shape: (B, window_size, len(target_cols))
                
                # 随机时间步
                timesteps = torch.randint(self.T, size=(batch.shape[0],), device=self.device)
                sigmas = torch.randn_like(batch)
                
                # 扩散系数
                coeff_1 = torch.sqrt(self.alpha_bars[timesteps]).reshape((-1, 1, 1))
                coeff_2 = torch.sqrt(1 - self.alpha_bars[timesteps]).reshape((-1, 1, 1))
                
                # 条件掩码：cond 列保持不变，target 列加噪
                conditional_mask = torch.ones_like(batch, device=self.device)
                conditional_mask[:, :, self.non_hier_cols] = 0.0
                
                # 加噪
                batch_noised = (1 - conditional_mask) * (coeff_1 * batch + coeff_2 * sigmas) \
                              + conditional_mask * batch
                
                # 预测噪声
                times = timesteps.reshape((-1, 1))
                sigmas_predicted = self.model(batch_noised, times)
                
                optimizer.zero_grad()
                
                # Ground truth 噪声（只针对 target 列）
                sigmas_gt = sigmas[:, :, self.non_hier_cols].permute((0, 2, 1))
                
                # 加权损失：观测位置权重=1.0，缺失位置权重=0.2
                # mask 形状: (B, window_size, len(target_cols)) -> permute to (B, len(target_cols), window_size)
                mask_permuted = mask.permute((0, 2, 1))
                
                loss_per_element = (sigmas_predicted - sigmas_gt) ** 2
                weighted_loss = loss_per_element * (0.2 + 0.8 * mask_permuted)
                loss = weighted_loss.mean()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
                
                # 清理
                del batch_noised, sigmas_predicted, sigmas_gt, loss_per_element, weighted_loss
            
            avg_loss = total_loss / max(1, n_batches)
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            if epoch % 50 == 0:
                print(f"    M-step epoch {epoch:3d}, loss: {avg_loss:.6f}, best: {best_loss:.6f}")
            
            # 定期清理 GPU 缓存
            if epoch % 10 == 0:
                torch.cuda.empty_cache()
        
        return best_loss
    
    @torch.no_grad()
    def e_step_fast(self, data_np, obs_mask_np, window_size, stride, 
                    batch_size=32, num_samples=1, ddim_steps=50):
        """
        内存优化的 E-step：使用 DDIM 采样更新缺失值估计
        
        参数:
            data_np: 当前数据估计，形状 (T_seq, C)
            obs_mask_np: 观测掩码，形状 (T_seq, len(target_cols))，1=观测，0=缺失
            window_size: 窗口大小
            stride: 步长
            batch_size: E-step 采样的批大小
            num_samples: 每个位置采样次数（取平均）
            ddim_steps: DDIM 采样步数
        
        返回:
            new_data: 更新后的数据估计
        """
        self.model.eval()
        torch.cuda.empty_cache()
        
        T_seq, C = data_np.shape
        n_target_cols = len(self.target_indices)
        
        # 验证 obs_mask 形状
        assert obs_mask_np.shape == (T_seq, n_target_cols), \
            f"obs_mask shape mismatch! Expected ({T_seq}, {n_target_cols}), got {obs_mask_np.shape}"
        
        num_windows = (T_seq - window_size) // stride + 1
        
        print(f"    [E-step] {num_windows} windows, batch={batch_size}, ddim={ddim_steps}, samples={num_samples}")
        
        # CPU 上的累加器
        imputed_sum = np.zeros((T_seq, C), dtype=np.float32)
        imputed_count = np.zeros((T_seq, 1), dtype=np.float32)
        
        num_batches = (num_windows + batch_size - 1) // batch_size
        
        for sample_idx in range(num_samples):
            if num_samples > 1:
                print(f"      Sample {sample_idx + 1}/{num_samples}")
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, num_windows)
                actual_batch_size = batch_end - batch_start
                
                if batch_idx % 50 == 0:
                    mem_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    print(f"      Batch {batch_idx+1}/{num_batches}, GPU mem: {mem_used:.2f} GB")
                
                # 构建 batch（在 CPU 上）
                batch_data = np.zeros((actual_batch_size, window_size, C), dtype=np.float32)
                batch_mask = np.zeros((actual_batch_size, window_size, n_target_cols), dtype=np.float32)
                
                for i in range(actual_batch_size):
                    win_idx = batch_start + i
                    start = win_idx * stride
                    end = start + window_size
                    batch_data[i] = data_np[start:end]
                    batch_mask[i] = obs_mask_np[start:end]
                
                # 转到 GPU
                batch_data_t = torch.from_numpy(batch_data).to(self.device)
                batch_mask_t = torch.from_numpy(batch_mask).to(self.device)
                
                # DDIM 采样
                sampled = self._conditional_sample_ddim(batch_data_t, batch_mask_t, ddim_steps)
                
                # 立即转回 CPU
                sampled_np = sampled.cpu().numpy()
                
                # 释放 GPU 内存
                del sampled, batch_data_t, batch_mask_t
                torch.cuda.empty_cache()
                
                # 累加到结果
                for i in range(actual_batch_size):
                    win_idx = batch_start + i
                    start = win_idx * stride
                    end = start + window_size
                    imputed_sum[start:end] += sampled_np[i]
                    imputed_count[start:end] += 1
        
        # 计算平均值
        imputed_count = np.maximum(imputed_count, 1)
        imputed = imputed_sum / imputed_count
        
        # 构建完整的 obs_mask（扩展到所有列）
        obs_mask_full = np.ones((T_seq, C), dtype=np.float32)  # 默认全部观测
        obs_mask_full[:, self.target_indices] = obs_mask_np  # 只有 target 列有缺失
        
        # 更新缺失位置
        new_data = data_np.copy()
        
        # 只更新 target 列的缺失值，保持 cond 列和观测值不变
        for col_idx, target_col in enumerate(self.target_indices):
            col_missing_mask = (obs_mask_np[:, col_idx] == 0)
            new_data[col_missing_mask, target_col] = imputed[col_missing_mask, target_col]
        
        self.model.train()
        torch.cuda.empty_cache()
        
        return new_data
    
    def _conditional_sample_ddim(self, batch_data, batch_mask, ddim_steps=50):
        """
        DDIM 条件采样
        
        参数:
            batch_data: 当前数据，形状 (B, T, C)
            batch_mask: 观测掩码，形状 (B, T, len(target_cols))
            ddim_steps: DDIM 步数
        
        返回:
            采样结果，形状 (B, T, C)
        """
        B, T, C = batch_data.shape
        
        # 从纯噪声开始
        x_t = torch.randn_like(batch_data)
        
        # 条件掩码：cond 列始终保持原值
        cond_mask = torch.zeros_like(batch_data, device=self.device)
        cond_mask[:, :, self.hier_cols] = 1.0
        
        # 观测掩码：扩展到完整维度
        obs_mask_full = torch.zeros_like(batch_data, device=self.device)
        obs_mask_full[:, :, self.target_indices] = batch_mask
        
        # DDIM 时间步（从 T-1 到 0，均匀采样）
        ddim_timesteps = np.linspace(0, self.T - 1, ddim_steps, dtype=int)[::-1]
        
        for i, t in enumerate(ddim_timesteps):
            t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
            
            # 条件列保持原值
            x_input = cond_mask * batch_data + (1 - cond_mask) * x_t
            
            # 预测噪声
            noise_pred = self.model(x_input, t_tensor.reshape(-1, 1))
            noise_pred = noise_pred.permute(0, 2, 1)  # (B, len(target), T) -> (B, T, len(target))
            
            # 扩展到完整维度
            noise_full = torch.zeros_like(x_t)
            noise_full[:, :, self.non_hier_cols] = noise_pred
            
            # DDIM 更新
            alpha_bar_t = self.alpha_bars[t]
            
            # 预测 x_0
            x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_full) / torch.sqrt(alpha_bar_t)
            x_0_pred = torch.clamp(x_0_pred, -3.0, 3.0)
            
            # 计算下一步
            if i < len(ddim_timesteps) - 1:
                t_next = ddim_timesteps[i + 1]
                alpha_bar_next = self.alpha_bars[t_next]
                x_t = torch.sqrt(alpha_bar_next) * x_0_pred + \
                      torch.sqrt(1 - alpha_bar_next) * noise_full
            else:
                x_t = x_0_pred
            
            # 强制观测值和条件列保持原值
            x_t = obs_mask_full * batch_data + (1 - obs_mask_full) * x_t
            x_t = cond_mask * batch_data + (1 - cond_mask) * x_t
            
            # 清理中间变量
            del noise_pred, noise_full, x_0_pred
        
        return x_t
    
    def _build_dataloader(self, data_np, obs_mask_np, window_size, stride, batch_size):
        """
        构建 DataLoader
        
        参数:
            data_np: 数据，形状 (T_seq, C)
            obs_mask_np: 观测掩码，形状 (T_seq, len(target_cols))
            window_size: 窗口大小
            stride: 步长
            batch_size: 批大小
        """
        T_seq, C = data_np.shape
        n_target_cols = obs_mask_np.shape[1]
        
        num_windows = (T_seq - window_size) // stride + 1
        
        samples = torch.zeros(num_windows, window_size, C, dtype=torch.float32)
        masks = torch.zeros(num_windows, window_size, n_target_cols, dtype=torch.float32)
        
        for i in range(num_windows):
            start = i * stride
            end = start + window_size
            samples[i] = torch.from_numpy(data_np[start:end])
            masks[i] = torch.from_numpy(obs_mask_np[start:end])
        
        dataset = EMDataset(samples, masks)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    def train_em(self, data_np, obs_mask_np, window_size, stride, 
                 batch_size, epochs_per_iter=200,
                 e_step_batch_size=32, ddim_steps=50, save_dir=None):
        """
        EM 训练主循环
        
        参数:
            data_np: 初始数据（已填充），形状 (T_seq, C)
            obs_mask_np: 观测掩码，形状 (T_seq, len(target_cols))
            window_size: 窗口大小
            stride: 步长
            batch_size: M-step 批大小
            epochs_per_iter: 每次 EM 迭代的 M-step epoch 数
            e_step_batch_size: E-step 采样批大小
            ddim_steps: DDIM 步数
            save_dir: 模型保存目录
        
        返回:
            current_data: 最终的数据估计
        """
        current_data = data_np.copy()
        best_loss = float('inf')
        
        for em_iter in range(self.em_iterations):
            print(f"\n{'='*60}")
            print(f"EM Iteration {em_iter + 1}/{self.em_iterations}")
            print(f"{'='*60}")
            
            # 清理 GPU
            torch.cuda.empty_cache()
            
            # 构建 DataLoader
            dataloader = self._build_dataloader(
                current_data, obs_mask_np, window_size, stride, batch_size
            )
            
            print(f"  [M-step] Training for {epochs_per_iter} epochs...")
            m_loss = self.m_step(dataloader, None, epochs_per_iter)
            print(f"  [M-step] Done, final loss: {m_loss:.6f}")
            
            # 保存当前最佳模型
            if m_loss < best_loss and save_dir is not None:
                best_loss = m_loss
                os.makedirs(save_dir, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(save_dir, "model_em_best.pth"))
                print(f"  [SAVE] New best model saved (loss: {best_loss:.6f})")
            
            # E-step（除了最后一次迭代）
            if em_iter < self.em_iterations - 1:
                print(f"  [E-step] Updating missing value estimates...")
                
                old_data = current_data.copy()
                
                current_data = self.e_step_fast(
                    current_data, obs_mask_np, window_size, stride,
                    batch_size=e_step_batch_size,
                    num_samples=1,
                    ddim_steps=ddim_steps
                )
                
                # 统计更新幅度
                target_indices = self.target_indices
                missing_mask = (obs_mask_np == 0)
                
                if missing_mask.sum() > 0:
                    old_missing = old_data[:, target_indices][missing_mask]
                    new_missing = current_data[:, target_indices][missing_mask]
                    diff = np.abs(new_missing - old_missing)
                    print(f"  [E-step] Update stats: mean={diff.mean():.4f}, "
                          f"std={diff.std():.4f}, max={diff.max():.4f}")
            
            # 定期保存 checkpoint
            if save_dir is not None and (em_iter + 1) % 2 == 0:
                ckpt_path = os.path.join(save_dir, f"model_em_iter{em_iter+1}.pth")
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  [SAVE] Checkpoint saved: {ckpt_path}")
        
        return current_data


class EMDataset(torch.utils.data.Dataset):
    """支持观测掩码的数据集"""
    def __init__(self, samples, masks):
        """
        参数:
            samples: 数据窗口，形状 (N, window_size, C)
            masks: 观测掩码，形状 (N, window_size, len(target_cols))
        """
        self.samples = samples
        self.masks = masks
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.masks[idx]


def validate_data(d_vals, obs_mask_np, target_indices, hier_cols):
    """验证数据的一致性"""
    print(f"\n[VALIDATION]")
    print(f"  d_vals shape: {d_vals.shape}")
    print(f"  obs_mask_np shape: {obs_mask_np.shape}")
    print(f"  target_indices: {target_indices}")
    print(f"  hier_cols (cond): {list(hier_cols)}")
    print(f"  len(target_indices): {len(target_indices)}")
    
    # 检查 obs_mask 列数是否匹配 target_indices
    assert obs_mask_np.shape[1] == len(target_indices), \
        f"obs_mask columns ({obs_mask_np.shape[1]}) != len(target_indices) ({len(target_indices)})"
    
    # 检查 obs_mask 的值
    unique_vals = np.unique(obs_mask_np)
    print(f"  obs_mask unique values: {unique_vals}")
    assert np.all(np.isin(unique_vals, [0, 1])), "obs_mask should only contain 0 and 1"
    
    # 观测率
    obs_rate = obs_mask_np.mean()
    print(f"  Observation rate: {obs_rate:.2%} (missing: {1-obs_rate:.2%})")
    
    # 检查数据中是否有 NaN
    nan_count = np.isnan(d_vals).sum()
    print(f"  NaN count in d_vals: {nan_count}")
    assert nan_count == 0, "d_vals should not contain NaN after filling"
    
    print(f"[VALIDATION] All checks passed! ✓")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-dataset", "-d", type=str, required=True)
    parser.add_argument("-input_csv", type=str, default=None)
    parser.add_argument("-prepared_dir", type=str, default="./work/prepared")
    
    # 模型参数
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
    parser.add_argument("-s4_bidirectional", type=bool, default=True)
    parser.add_argument("-s4_layernorm", type=bool, default=True)
    parser.add_argument("-propCycEnc", type=bool, default=False)
    parser.add_argument("-normalize", type=bool, default=True)
    
    # EM 相关参数
    parser.add_argument("-em_iterations", type=int, default=5, 
                        help="Number of EM iterations (DiffPuter style)")
    parser.add_argument("-epochs_per_em", type=int, default=200,
                        help="Training epochs per EM iteration")
    parser.add_argument("-use_em", action="store_true",
                        help="Use DiffPuter-style EM training")
    
    # E-step 优化参数
    parser.add_argument("-e_step_batch_size", type=int, default=126,
                        help="Batch size for E-step sampling (smaller = less GPU memory)")
    parser.add_argument("-ddim_steps", type=int, default=50,
                        help="DDIM steps for E-step (50 is a good balance)")
    parser.add_argument("-e_step_samples", type=int, default=1,
                        help="Number of samples per E-step (1 is usually enough)")

    # from datetime import datetime

    parser.add_argument("-run_id", type=str, default=None)
    parser.add_argument("-model_name", type=str, default="wavestitchplus")
    parser.add_argument("-model_filename", type=str, default="model.pth")

    args = parser.parse_args()
    dataset = args.dataset
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"DiffPuter Training Script")
    print(f"{'='*60}")
    print(f"[INFO] Device: {dev}")
    print(f"[INFO] EM training: {args.use_em}")
    if args.use_em:
        print(f"[INFO] EM iterations: {args.em_iterations}")
        print(f"[INFO] Epochs per EM: {args.epochs_per_em}")
        print(f"[INFO] E-step batch size: {args.e_step_batch_size}")
        print(f"[INFO] DDIM steps: {args.ddim_steps}")

    # Load data
    if args.dataset == "custom_csv":
        from custom_pipeline.preprocess import preprocess_csv

        prepared_dir = args.prepared_dir or "./work/prepared"
        
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

        # 加载数据
        training_df, cond_cols, target_cols, obs_mask = load_custom_train_df(prepared_dir)
        
        print(f"\n[INFO] Data loaded:")
        print(f"  Training data shape: {training_df.shape}")
        print(f"  Target columns ({len(target_cols)}): {target_cols}")
        print(f"  Cond columns ({len(cond_cols)}): {cond_cols}")
        
        d_vals = training_df.values.astype(np.float32)
        obs_mask_np = obs_mask.values.astype(np.float32)
        
        # 计算列索引
        hierarchical_column_indices = training_df.columns.get_indexer(cond_cols)
        target_indices = [training_df.columns.get_loc(c) for c in target_cols 
                         if c in training_df.columns]
        
        # 标准化
        if args.normalize:
            print(f"\n[INFO] Normalizing data...")
            
            # 只用观测值计算统计量
            observed_data = d_vals[:, target_indices].copy()
            observed_data[obs_mask_np == 0] = np.nan
            
            scaler = StandardScaler()
            scaler.mean_ = np.nanmean(observed_data, axis=0)
            scaler.std_ = np.nanstd(observed_data, axis=0)
            scaler.std_[scaler.std_ < 1e-8] = 1.0
            
            # 标准化 target 列
            d_vals[:, target_indices] = (d_vals[:, target_indices] - scaler.mean_) / scaler.std_
            d_vals[:, target_indices] = np.clip(d_vals[:, target_indices], -3.0, 3.0)
            
            # 保存 scaler
            scaler_dir = os.path.join(prepared_dir, "scaler")
            os.makedirs(scaler_dir, exist_ok=True)
            np.save(os.path.join(scaler_dir, "mean.npy"), scaler.mean_)
            np.save(os.path.join(scaler_dir, "std.npy"), scaler.std_)
            
            print(f"  Scaler saved to: {scaler_dir}")
            print(f"  Mean: {scaler.mean_[:5]}...")
            print(f"  Std: {scaler.std_[:5]}...")

    else:
        # 原有逻辑（其他数据集）
        preprocessor = Preprocessor(dataset, args.propCycEnc)
        df = preprocessor.df_cleaned
        training_df = df.loc[preprocessor.train_indices]
        hierarchical_column_indices = training_df.columns.get_indexer(
            preprocessor.hierarchical_features_cyclic
        )
        d_vals = training_df.values.astype(np.float32)
        target_indices = list(range(d_vals.shape[1] - len(hierarchical_column_indices)))
        obs_mask_np = np.ones((d_vals.shape[0], len(target_indices)), dtype=np.float32)

    # Setup model
    in_dim = d_vals.shape[1]
    out_dim = in_dim - len(hierarchical_column_indices)
    
    print(f"\n[INFO] Model setup:")
    print(f"  in_dim: {in_dim}")
    print(f"  out_dim: {out_dim}")

    model = fetchModel(in_dim, out_dim, args).to(dev)
    diffusion_config = fetchDiffusionConfig(args)
    
    all_indices = np.arange(in_dim)
    non_hier_cols = np.setdiff1d(all_indices, hierarchical_column_indices)
    
    print(f"  Hierarchical (cond) columns: {list(hierarchical_column_indices)}")
    print(f"  Non-hierarchical (target) columns: {list(non_hier_cols)}")
    
    # 验证数据
    validate_data(d_vals, obs_mask_np, target_indices, hierarchical_column_indices)

    # 保存目录
    save_dir = get_save_dir(args.prepared_dir)
    os.makedirs(save_dir, exist_ok=True)

    if args.use_em:
        # DiffPuter 风格的 EM 训练
        print(f"\n{'='*60}")
        print(f"Starting DiffPuter-style EM Training")
        print(f"{'='*60}")
        
        trainer = DiffPuterEMTrainer(
            model=model,
            diffusion_config=diffusion_config,
            device=dev,
            non_hier_cols=non_hier_cols,
            hier_cols=hierarchical_column_indices,
            target_indices=target_indices,
            lr=args.lr,
            em_iterations=args.em_iterations
        )
        
        final_data = trainer.train_em(
            data_np=d_vals,
            obs_mask_np=obs_mask_np,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            epochs_per_iter=args.epochs_per_em,
            e_step_batch_size=args.e_step_batch_size,
            ddim_steps=args.ddim_steps,
            save_dir=save_dir
        )
        
        # 保存最终的 imputed 数据
        imputed_path = os.path.join(prepared_dir, "train_imputed.npy")
        np.save(imputed_path, final_data)
        print(f"\n[INFO] Saved imputed training data to: {imputed_path}")
        
        # 保存最终模型
        final_model_path = os.path.join(save_dir, "model_em.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"[INFO] Saved final EM model to: {final_model_path}")
        
    else:
        # 原有的单次训练逻辑
        print(f"\n{'='*60}")
        print(f"Starting Standard Training (no EM)")
        print(f"{'='*60}")
        
        T_seq, C = d_vals.shape
        num_windows = (T_seq - args.window_size) // args.stride + 1
        
        training_samples = torch.zeros(num_windows, args.window_size, C, dtype=torch.float32)
        for i in range(num_windows):
            start = i * args.stride
            end = start + args.window_size
            training_samples[i] = torch.from_numpy(d_vals[start:end])
        
        training_dataset = MyDataset(training_samples.float(), window_size=args.window_size)
        dataloader = DataLoader(training_dataset, batch_size=args.batch_size, 
                               shuffle=True, drop_last=True)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )
        
        alpha_bars = diffusion_config["alpha_bars"].to(dev)
        best_loss = float('inf')
        
        for epoch in range(args.epochs):
            total_loss = 0.0
            n_batches = 0
            
            for batch in dataloader:
                batch = batch.to(dev)
                
                timesteps = randint(diffusion_config["T"], size=(batch.shape[0],), device=dev)
                sigmas = normal(0, 1, size=batch.shape).to(dev)
                
                coeff_1 = sqrt(alpha_bars[timesteps]).reshape((-1, 1, 1))
                coeff_2 = sqrt(1 - alpha_bars[timesteps]).reshape((-1, 1, 1))
                
                conditional_mask = torch.ones_like(batch, device=dev)
                conditional_mask[:, :, non_hier_cols] = 0.0
                
                batch_noised = (1 - conditional_mask) * (coeff_1 * batch + coeff_2 * sigmas) \
                              + conditional_mask * batch
                
                times = timesteps.reshape((-1, 1))
                sigmas_predicted = model(batch_noised, times)
                
                optimizer.zero_grad()
                sigmas_gt = sigmas[:, :, non_hier_cols].permute((0, 2, 1)).to(dev)
                loss = criterion(sigmas_predicted, sigmas_gt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / max(1, n_batches)
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pth"))
            
            if epoch % 10 == 0:
                print(f"epoch: {epoch:4d}, avg_loss: {avg_loss:.6f}, best: {best_loss:.6f}")
        
        # 保存最终模型
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))

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

    # print(f"\n{'='*60}")
    # print(f"[DONE] Training completed!")
    # print(f"  Models saved to: {save_dir}")
    # print(f"{'='*60}")
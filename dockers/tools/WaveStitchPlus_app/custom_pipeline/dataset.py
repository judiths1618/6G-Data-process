import json
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray):
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0

    def transform(self, x: np.ndarray):
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray):
        return x * self.std_ + self.mean_


class CustomCSVDataset(Dataset):
    def __init__(self, prepared_dir: str, split: str, window_len: int, stride: int, fit_scaler_on_train=True):
        p = Path(prepared_dir)
        meta = json.loads((p / "meta.json").read_text())
        self.meta = meta
        self.time_col = meta["time_col"]
        self.cols = meta["all_model_cols"]
        self.target_cols = meta["target_cols"]
        self.cond_cols = meta["cond_cols"]

        if split == "train":
            df = pd.read_csv(p / "train.csv")
        elif split == "test_input":
            df = pd.read_csv(p / "test_input.csv")
        elif split == "test_gt":
            df = pd.read_csv(p / "test_gt.csv")
        else:
            raise ValueError("split must be train/test_input/test_gt")

        # 保证列顺序
        self.time = df[self.time_col].to_numpy(dtype=np.int64)
        x = df[self.cols].to_numpy(dtype=np.float32)

        # 构造 missing mask：target_cols 上 NaN 为 missing；cond_cols 永远 observed
        mask = np.zeros_like(x, dtype=np.float32)  # 0=observed
        # target部分：NaN -> missing=1
        target_idx = [self.cols.index(c) for c in self.target_cols]
        nan_target = np.isnan(x[:, target_idx])
        mask[:, target_idx] = nan_target.astype(np.float32)

        # cond部分：如果有 NaN（理论上不该有），也强行当 observed=0 并填0
        cond_idx = [self.cols.index(c) for c in self.cond_cols] if len(self.cond_cols) else []
        if len(cond_idx):
            x[:, cond_idx] = np.nan_to_num(x[:, cond_idx], nan=0.0)
            mask[:, cond_idx] = 0.0

        # 对缺失值先填0（模型输入不能是NaN）
        x = np.nan_to_num(x, nan=0.0)

        self.window_len = int(window_len)
        self.stride = int(stride)

        # scaler：只对 target_cols 标准化更合理；cond_cols可不标准化/或单独标准化
        self.scaler = StandardScaler()
        self.target_idx = np.array(target_idx, dtype=int)

        if split == "train" and fit_scaler_on_train:
            self.scaler.fit(x[:, self.target_idx])
            (p / "scaler_mean.npy").write_bytes(self.scaler.mean_.astype(np.float32).tobytes())
            (p / "scaler_std.npy").write_bytes(self.scaler.std_.astype(np.float32).tobytes())
        else:
            # 读训练保存的 scaler
            mean_path = p / "scaler_mean.npy"
            std_path = p / "scaler_std.npy"
            if mean_path.exists() and std_path.exists():
                # 用 frombuffer 读 bytes（简洁一点）
                self.scaler.mean_ = np.frombuffer(mean_path.read_bytes(), dtype=np.float32)
                self.scaler.std_ = np.frombuffer(std_path.read_bytes(), dtype=np.float32)
            else:
                # 如果没有，就退化不标准化（但你最好保证先训再推理）
                self.scaler.mean_ = np.zeros(len(self.target_idx), dtype=np.float32)
                self.scaler.std_ = np.ones(len(self.target_idx), dtype=np.float32)

        x[:, self.target_idx] = self.scaler.transform(x[:, self.target_idx])
        self.x = x
        self.mask = mask

        # 生成所有窗口起点
        T = len(self.x)
        self.starts = list(range(0, max(1, T - self.window_len + 1), self.stride))
        if len(self.starts) == 0 and T >= self.window_len:
            self.starts = [0]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.window_len
        z = self.x[s:e]
        m = self.mask[s:e]
        # 返回 time 用于推理拼接可选
        t = self.time[s:e]
        return {"z": z, "mask": m, "time": t}

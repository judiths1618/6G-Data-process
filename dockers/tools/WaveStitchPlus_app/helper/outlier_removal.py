#!/usr/bin/env python3
"""
时序数据 Outlier Removal 工具
支持多种检测方法：IQR, Z-score, Rolling Statistics, 物理约束
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


class TimeSeriesOutlierRemover:
    """时序数据异常值处理器"""
    
    def __init__(self, df: pd.DataFrame, target_cols: List[str], 
                 time_col: str = "time"):
        self.df = df.copy()
        self.target_cols = [c for c in target_cols if c in df.columns]
        self.time_col = time_col
        self.outlier_mask = pd.DataFrame(False, index=df.index, columns=self.target_cols)
        self.stats = {}
    
    def detect_iqr(self, cols: Optional[List[str]] = None, 
                   k: float = 1.5) -> pd.DataFrame:
        """
        IQR 方法检测异常值
        outlier if: x < Q1 - k*IQR or x > Q3 + k*IQR
        """
        cols = cols or self.target_cols
        
        for col in cols:
            if col not in self.df.columns:
                continue
            
            data = self.df[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower = Q1 - k * IQR
            upper = Q3 + k * IQR
            
            mask = (self.df[col] < lower) | (self.df[col] > upper)
            self.outlier_mask[col] |= mask
            
            self.stats[f"{col}_iqr"] = {
                "Q1": Q1, "Q3": Q3, "IQR": IQR,
                "lower": lower, "upper": upper,
                "n_outliers": mask.sum()
            }
        
        return self.outlier_mask
    
    def detect_zscore(self, cols: Optional[List[str]] = None, 
                      threshold: float = 3.0) -> pd.DataFrame:
        """
        Z-score 方法检测异常值
        outlier if: |z| > threshold
        """
        cols = cols or self.target_cols
        
        for col in cols:
            if col not in self.df.columns:
                continue
            
            data = self.df[col]
            mean = data.mean()
            std = data.std()
            
            if std < 1e-8:
                continue
            
            z_scores = np.abs((data - mean) / std)
            mask = z_scores > threshold
            self.outlier_mask[col] |= mask
            
            self.stats[f"{col}_zscore"] = {
                "mean": mean, "std": std, "threshold": threshold,
                "n_outliers": mask.sum()
            }
        
        return self.outlier_mask
    
    def detect_rolling(self, cols: Optional[List[str]] = None,
                       window: int = 10, n_std: float = 3.0) -> pd.DataFrame:
        """
        滚动窗口方法检测异常值（适合时序数据）
        outlier if: |x - rolling_mean| > n_std * rolling_std
        """
        cols = cols or self.target_cols
        
        for col in cols:
            if col not in self.df.columns:
                continue
            
            data = self.df[col]
            rolling_mean = data.rolling(window=window, center=True, min_periods=1).mean()
            rolling_std = data.rolling(window=window, center=True, min_periods=1).std()
            rolling_std = rolling_std.fillna(rolling_std.mean()).replace(0, 1e-8)
            
            deviation = np.abs(data - rolling_mean)
            mask = deviation > n_std * rolling_std
            self.outlier_mask[col] |= mask
            
            self.stats[f"{col}_rolling"] = {
                "window": window, "n_std": n_std,
                "n_outliers": mask.sum()
            }
        
        return self.outlier_mask
    
    def detect_physical_constraints(self) -> pd.DataFrame:
        """
        基于物理/业务约束检测异常值
        针对你的具体数据类型设置合理范围
        """
        constraints = {
            # CPU 相关
            "cpu_limit": (0, 128),           # CPU cores: 0-128
            "cpu_usage": (0, 128),           # 使用量不超过限制（假设最大128核）
            
            # 请求数相关
            "n": (0, 1e7),                   # 请求数: 0 到 1000万
            "c": (0, 1e5),                   # 并发数: 0 到 10万
            
            # 内存相关 (MB)
            "ram_limit_mb": (0, 1e6),        # 0 到 1TB
            "ram_usage_mb": (0, 1e6),        # 0 到 1TB
            
            # 延迟相关 (ms) - 不同百分位应该递增
            "lat50_ms": (0, 1e6),            # 0 到 1000秒
            "lat66_ms": (0, 1e6),
            "lat75_ms": (0, 1e6),
            "lat80_ms": (0, 1e6),
            "lat90_ms": (0, 1e6),
            "lat95_ms": (0, 1e6),
            "lat98_ms": (0, 1e6),
            "lat99_ms": (0, 1e6),
            "lat100_ms": (0, 1e6),
        }
        
        for col, (lower, upper) in constraints.items():
            if col not in self.df.columns:
                continue
            
            mask = (self.df[col] < lower) | (self.df[col] > upper)
            self.outlier_mask[col] |= mask
            
            self.stats[f"{col}_physical"] = {
                "lower": lower, "upper": upper,
                "n_outliers": mask.sum()
            }
        
        # 特殊检查：延迟百分位应该递增
        lat_cols = ["lat50_ms", "lat66_ms", "lat75_ms", "lat80_ms", 
                    "lat90_ms", "lat95_ms", "lat98_ms", "lat99_ms", "lat100_ms"]
        lat_cols = [c for c in lat_cols if c in self.df.columns]
        
        if len(lat_cols) > 1:
            for i in range(len(lat_cols) - 1):
                col_lower = lat_cols[i]
                col_higher = lat_cols[i + 1]
                
                # 如果较低百分位 > 较高百分位，标记为异常
                mask = self.df[col_lower] > self.df[col_higher] * 1.1  # 允许10%容差
                self.outlier_mask[col_lower] |= mask
                self.outlier_mask[col_higher] |= mask
                
                self.stats[f"lat_order_{col_lower}_{col_higher}"] = {
                    "n_violations": mask.sum()
                }
        
        # 特殊检查：usage 不应该远超 limit
        if "cpu_usage" in self.df.columns and "cpu_limit" in self.df.columns:
            mask = self.df["cpu_usage"] > self.df["cpu_limit"] * 1.5  # 允许50%超额
            self.outlier_mask["cpu_usage"] |= mask
            self.stats["cpu_usage_vs_limit"] = {"n_violations": mask.sum()}
        
        if "ram_usage_mb" in self.df.columns and "ram_limit_mb" in self.df.columns:
            mask = self.df["ram_usage_mb"] > self.df["ram_limit_mb"] * 1.5
            self.outlier_mask["ram_usage_mb"] |= mask
            self.stats["ram_usage_vs_limit"] = {"n_violations": mask.sum()}
        
        return self.outlier_mask
    
    def detect_negative_values(self, cols: Optional[List[str]] = None) -> pd.DataFrame:
        """检测负值（这些指标不应该为负）"""
        cols = cols or self.target_cols
        
        for col in cols:
            if col not in self.df.columns:
                continue
            
            mask = self.df[col] < 0
            self.outlier_mask[col] |= mask
            
            self.stats[f"{col}_negative"] = {"n_outliers": mask.sum()}
        
        return self.outlier_mask
    
    def remove_outliers(self, method: str = "interpolate") -> pd.DataFrame:
        """
        处理检测到的异常值
        
        method:
            - "interpolate": 线性插值
            - "ffill": 前向填充
            - "bfill": 后向填充
            - "mean": 用列均值填充
            - "median": 用列中位数填充
            - "rolling_mean": 用滚动均值填充
            - "clip": 裁剪到合理范围
            - "nan": 设为 NaN（不填充）
        """
        df_clean = self.df.copy()
        
        for col in self.target_cols:
            if col not in df_clean.columns:
                continue
            
            outlier_idx = self.outlier_mask[col]
            n_outliers = outlier_idx.sum()
            
            if n_outliers == 0:
                continue
            
            print(f"  {col}: {n_outliers} outliers ({100*n_outliers/len(df_clean):.2f}%)")
            
            if method == "nan":
                df_clean.loc[outlier_idx, col] = np.nan
                
            elif method == "interpolate":
                df_clean.loc[outlier_idx, col] = np.nan
                df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
                df_clean[col] = df_clean[col].ffill().bfill()
                
            elif method == "ffill":
                df_clean.loc[outlier_idx, col] = np.nan
                df_clean[col] = df_clean[col].ffill().bfill()
                
            elif method == "bfill":
                df_clean.loc[outlier_idx, col] = np.nan
                df_clean[col] = df_clean[col].bfill().ffill()
                
            elif method == "mean":
                col_mean = df_clean.loc[~outlier_idx, col].mean()
                df_clean.loc[outlier_idx, col] = col_mean
                
            elif method == "median":
                col_median = df_clean.loc[~outlier_idx, col].median()
                df_clean.loc[outlier_idx, col] = col_median
                
            elif method == "rolling_mean":
                df_clean.loc[outlier_idx, col] = np.nan
                rolling = df_clean[col].rolling(window=5, center=True, min_periods=1).mean()
                df_clean[col] = df_clean[col].fillna(rolling)
                df_clean[col] = df_clean[col].ffill().bfill()
                
            elif method == "clip":
                # 裁剪到 IQR 范围
                data = df_clean.loc[~outlier_idx, col]
                Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                df_clean[col] = df_clean[col].clip(lower=max(0, lower), upper=upper)
        
        return df_clean
    
    def get_summary(self) -> pd.DataFrame:
        """获取异常值检测摘要"""
        summary = []
        
        for col in self.target_cols:
            if col not in self.outlier_mask.columns:
                continue
            
            n_total = len(self.df)
            n_outliers = self.outlier_mask[col].sum()
            
            summary.append({
                "column": col,
                "total_rows": n_total,
                "outliers": n_outliers,
                "outlier_pct": 100 * n_outliers / n_total,
                "min": self.df[col].min(),
                "max": self.df[col].max(),
                "mean": self.df[col].mean(),
                "std": self.df[col].std()
            })
        
        return pd.DataFrame(summary)


def remove_outliers_from_generated(
    input_csv: str,
    output_csv: str,
    target_cols: List[str],
    time_col: str = "time",
    methods: List[str] = ["physical", "iqr", "rolling"],
    fill_method: str = "interpolate",
    iqr_k: float = 2.0,
    zscore_threshold: float = 3.0,
    rolling_window: int = 10,
    rolling_n_std: float = 3.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    完整的异常值处理流程
    """
    print(f"\n{'='*60}")
    print(f"Outlier Removal Pipeline")
    print(f"{'='*60}")
    
    # 加载数据
    df = pd.read_csv(input_csv)
    print(f"[INFO] Loaded {len(df)} rows from {input_csv}")
    print(f"[INFO] Target columns: {target_cols}")
    
    # 创建处理器
    remover = TimeSeriesOutlierRemover(df, target_cols, time_col)
    
    # 检测异常值
    print(f"\n[STEP 1] Detecting outliers...")
    
    if "negative" in methods:
        print("  - Checking negative values...")
        remover.detect_negative_values()
    
    if "physical" in methods:
        print("  - Checking physical constraints...")
        remover.detect_physical_constraints()
    
    if "iqr" in methods:
        print(f"  - IQR method (k={iqr_k})...")
        remover.detect_iqr(k=iqr_k)
    
    if "zscore" in methods:
        print(f"  - Z-score method (threshold={zscore_threshold})...")
        remover.detect_zscore(threshold=zscore_threshold)
    
    if "rolling" in methods:
        print(f"  - Rolling window method (window={rolling_window}, n_std={rolling_n_std})...")
        remover.detect_rolling(window=rolling_window, n_std=rolling_n_std)
    
    # 打印摘要
    print(f"\n[STEP 2] Outlier Summary:")
    summary = remover.get_summary()
    print(summary.to_string(index=False))
    
    total_outliers = remover.outlier_mask.any(axis=1).sum()
    print(f"\n  Total rows with outliers: {total_outliers} ({100*total_outliers/len(df):.2f}%)")
    
    # 处理异常值
    print(f"\n[STEP 3] Removing outliers (method: {fill_method})...")
    df_clean = remover.remove_outliers(method=fill_method)
    
    # 保存结果
    df_clean.to_csv(output_csv, index=False)
    print(f"\n[DONE] Saved cleaned data to: {output_csv}")
    
    # 返回统计信息
    stats = {
        "input_rows": len(df),
        "outlier_rows": total_outliers,
        "outlier_pct": 100 * total_outliers / len(df),
        "methods_used": methods,
        "fill_method": fill_method,
        "per_column": remover.stats
    }
    
    return df_clean, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Outlier Removal")
    
    parser.add_argument("-input", "-i", type=str, required=True,
                        help="Input CSV file (generated data)")
    parser.add_argument("-output", "-o", type=str, default=None,
                        help="Output CSV file (cleaned data)")
    parser.add_argument("-prepared_dir", type=str, default=None,
                        help="Prepared dir to load meta.json for target_cols")
    parser.add_argument("-time_col", type=str, default="time",
                        help="Time column name")
    
    # 检测方法
    parser.add_argument("-methods", type=str, nargs="+", 
                        default=["negative", "physical", "iqr", "rolling"],
                        choices=["negative", "physical", "iqr", "zscore", "rolling"],
                        help="Outlier detection methods")
    
    # 填充方法
    parser.add_argument("-fill", type=str, default="interpolate",
                        choices=["interpolate", "ffill", "bfill", "mean", 
                                 "median", "rolling_mean", "clip", "nan"],
                        help="How to fill outlier values")
    
    # 方法参数
    parser.add_argument("-iqr_k", type=float, default=2.0,
                        help="IQR multiplier (default: 2.0)")
    parser.add_argument("-zscore_threshold", type=float, default=3.0,
                        help="Z-score threshold (default: 3.0)")
    parser.add_argument("-rolling_window", type=int, default=10,
                        help="Rolling window size (default: 10)")
    parser.add_argument("-rolling_n_std", type=float, default=3.0,
                        help="Rolling n_std threshold (default: 3.0)")
    
    args = parser.parse_args()
    
    # 确定 target_cols
    target_cols = [
        "cpu_limit", "cpu_usage", "n", "c",
        "ram_limit_mb", "ram_usage_mb",
        "lat50_ms", "lat66_ms", "lat75_ms", "lat80_ms",
        "lat90_ms", "lat95_ms", "lat98_ms", "lat99_ms", "lat100_ms"
    ]
    
    # 如果提供了 prepared_dir，从 meta.json 读取
    if args.prepared_dir:
        meta_path = os.path.join(args.prepared_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            target_cols = meta.get("target_cols", target_cols)
            print(f"[INFO] Loaded target_cols from {meta_path}")
    
    # 确定输出路径
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_cleaned{ext}"
    
    # 运行
    df_clean, stats = remove_outliers_from_generated(
        input_csv=args.input,
        output_csv=args.output,
        target_cols=target_cols,
        time_col=args.time_col,
        methods=args.methods,
        fill_method=args.fill,
        iqr_k=args.iqr_k,
        zscore_threshold=args.zscore_threshold,
        rolling_window=args.rolling_window,
        rolling_n_std=args.rolling_n_std
    )
    
    # 打印前后对比
    print(f"\n{'='*60}")
    print("Before vs After Statistics")
    print(f"{'='*60}")
    
    df_orig = pd.read_csv(args.input)
    
    for col in target_cols:
        if col not in df_orig.columns:
            continue
        print(f"\n{col}:")
        print(f"  Before: min={df_orig[col].min():.2f}, max={df_orig[col].max():.2f}, "
              f"mean={df_orig[col].mean():.2f}, std={df_orig[col].std():.2f}")
        print(f"  After:  min={df_clean[col].min():.2f}, max={df_clean[col].max():.2f}, "
              f"mean={df_clean[col].mean():.2f}, std={df_clean[col].std():.2f}")
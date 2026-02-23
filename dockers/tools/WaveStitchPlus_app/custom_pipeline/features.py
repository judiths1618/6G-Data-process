import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame, time_col="time"):
    """
    输入 df 必须有 time_col（int seconds），且已规则化（等间隔 index）。
    输出新增一组永远存在的条件特征 C(t)，用于 gap 内 conditioning。
    """
    t = df[time_col].to_numpy(dtype=np.float64)

    # 归一化时间 [0,1]
    t0, t1 = float(np.nanmin(t)), float(np.nanmax(t))
    if t1 <= t0:
        t_norm = np.zeros_like(t)
    else:
        t_norm = (t - t0) / (t1 - t0)

    # 可选：日周期(86400s)编码（如果你觉得日志有昼夜特征就留着；否则也无妨）
    day = 86400.0
    sin_day = np.sin(2 * np.pi * (t % day) / day)
    cos_day = np.cos(2 * np.pi * (t % day) / day)

    df["t_norm"] = t_norm
    df["sin_day"] = sin_day
    df["cos_day"] = cos_day
    return df


def add_gap_structure_features(df: pd.DataFrame, observed_row_mask: np.ndarray):
    """
    observed_row_mask: shape [T], True 表示该 timestamp 在原始数据中存在一行观测（非 gap 插入行）
    生成 time_since_last_obs / time_to_next_obs / is_gap
    """
    T = len(df)
    time = df["time"].to_numpy(dtype=np.int64)

    is_gap = ~observed_row_mask
    df["is_gap"] = is_gap.astype(np.float32)

    # time_since_last_obs
    last = np.full(T, np.nan, dtype=np.float64)
    last_t = np.nan
    for i in range(T):
        if observed_row_mask[i]:
            last_t = float(time[i])
        last[i] = last_t
    df["time_since_last_obs"] = np.where(np.isnan(last), 0.0, (time.astype(np.float64) - last))

    # time_to_next_obs
    nxt = np.full(T, np.nan, dtype=np.float64)
    nxt_t = np.nan
    for i in range(T - 1, -1, -1):
        if observed_row_mask[i]:
            nxt_t = float(time[i])
        nxt[i] = nxt_t
    df["time_to_next_obs"] = np.where(np.isnan(nxt), 0.0, (nxt - time.astype(np.float64)))

    # 归一化一下（避免尺度太大，可按 base_dt 做缩放更好）
    # 这里先简单压缩到秒->分钟级别
    df["time_since_last_obs"] = df["time_since_last_obs"] / 60.0
    df["time_to_next_obs"] = df["time_to_next_obs"] / 60.0
    return df

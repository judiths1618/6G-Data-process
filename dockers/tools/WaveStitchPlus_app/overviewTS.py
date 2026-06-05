# #!/usr/bin/env python3
# """
# 诊断 test_gt.csv 在原始数据中的位置
# """

# import pandas as pd
# import numpy as np
# import os
# import json
# from datetime import datetime

# # ============ 配置 ============
# datafilename = 'python'
# base_dir = f'./work/EUR'
# prepared_dir = f'{base_dir}/prepared_{datafilename}'
# generated_dir = f'{base_dir}/generated_{datafilename}'

# # ============ 加载 meta ============
# meta_path = os.path.join(prepared_dir, 'meta.json')
# with open(meta_path, 'r') as f:
#     meta = json.load(f)

# time_col = meta.get('time_col', 'time')
# split_ratio = meta.get('split_ratio', 0.8)

# print(f"{'='*70}")
# print(f"DATA LOCATION DIAGNOSTIC")
# print(f"{'='*70}")
# print(f"Dataset: {datafilename}")
# print(f"Time column: {time_col}")
# print(f"Split ratio: {split_ratio}")

# # ============ 加载所有数据 ============
# print(f"\n[1] Loading Data")

# # 加载 prepared 数据
# train_path = os.path.join(prepared_dir, 'train.csv')
# test_gt_path = os.path.join(prepared_dir, 'test_gt.csv')
# test_input_path = os.path.join(prepared_dir, 'test_input.csv')

# train_df = pd.read_csv(train_path) if os.path.exists(train_path) else None
# test_gt = pd.read_csv(test_gt_path) if os.path.exists(test_gt_path) else None
# test_input = pd.read_csv(test_input_path) if os.path.exists(test_input_path) else None

# # 加载 imputed 数据
# ws_path = os.path.join(generated_dir, 'wavestitch_full_imputed_cleaned.csv')
# wsp_path = os.path.join(generated_dir, 'wavestitchplus_v1_test_imputed_cleaned.csv')

# pred_wavestitch = pd.read_csv(ws_path) if os.path.exists(ws_path) else None
# pred_wavestitchPlus = pd.read_csv(wsp_path) if os.path.exists(wsp_path) else None

# # 打印行数
# print(f"\n  Prepared data:")
# print(f"    train.csv:      {len(train_df) if train_df is not None else 'N/A'} rows")
# print(f"    test_gt.csv:    {len(test_gt) if test_gt is not None else 'N/A'} rows")
# print(f"    test_input.csv: {len(test_input) if test_input is not None else 'N/A'} rows")

# print(f"\n  Imputed data:")
# print(f"    WaveStitch:     {len(pred_wavestitch) if pred_wavestitch is not None else 'N/A'} rows")
# print(f"    WaveStitch+:    {len(pred_wavestitchPlus) if pred_wavestitchPlus is not None else 'N/A'} rows")

# # ============ 时间范围分析 ============
# print(f"\n{'='*70}")
# print(f"[2] TIME RANGE ANALYSIS")
# print(f"{'='*70}")

# def format_timestamp(ts):
#     """格式化时间戳"""
#     try:
#         if ts > 1e12:
#             return datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
#         else:
#             return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
#     except:
#         return str(ts)

# def analyze_time_range(df, name, time_col):
#     """分析时间范围"""
#     if df is None or time_col not in df.columns:
#         print(f"\n  {name}: N/A")
#         return None
    
#     times = df[time_col].values
#     t_min, t_max = times.min(), times.max()
#     duration = t_max - t_min
    
#     print(f"\n  {name}:")
#     print(f"    Rows: {len(df)}")
#     print(f"    Time range: {t_min:.0f} - {t_max:.0f}")
#     print(f"    Duration: {duration:.0f}s ({duration/3600:.2f} hours)")
#     print(f"    Start: {format_timestamp(t_min)}")
#     print(f"    End:   {format_timestamp(t_max)}")
    
#     return {'min': t_min, 'max': t_max, 'duration': duration, 'rows': len(df)}

# train_info = analyze_time_range(train_df, "train.csv", time_col)
# test_gt_info = analyze_time_range(test_gt, "test_gt.csv", time_col)
# test_input_info = analyze_time_range(test_input, "test_input.csv", time_col)
# ws_info = analyze_time_range(pred_wavestitch, "WaveStitch imputed", time_col)
# wsp_info = analyze_time_range(pred_wavestitchPlus, "WaveStitch+ imputed", time_col)

# # ============ 数据流可视化 ============
# print(f"\n{'='*70}")
# print(f"[3] DATA FLOW VISUALIZATION")
# print(f"{'='*70}")

# if train_info and test_gt_info:
#     total_rows = train_info['rows'] + test_gt_info['rows']
#     total_duration = train_info['duration'] + test_gt_info['duration']
    
#     print(f"\n  Original regularized data (train + test):")
#     print(f"    Total rows: {total_rows}")
#     print(f"    Total duration: {total_duration:.0f}s ({total_duration/3600:.2f} hours)")
    
#     train_pct = 100 * train_info['rows'] / total_rows
#     test_pct = 100 * test_gt_info['rows'] / total_rows
    
#     print(f"\n  Split:")
#     print(f"    Train: {train_info['rows']} rows ({train_pct:.1f}%)")
#     print(f"    Test:  {test_gt_info['rows']} rows ({test_pct:.1f}%)")
    
#     # 可视化时间线
#     print(f"\n  Timeline:")
#     print(f"    |{'='*30}|{'='*10}|")
#     print(f"    |         TRAIN ({train_pct:.0f}%)         |  TEST ({test_pct:.0f}%) |")
#     print(f"    |{'='*30}|{'='*10}|")
#     print(f"    ^                              ^          ^")
#     print(f"    {format_timestamp(train_info['min'])}      {format_timestamp(test_gt_info['min'])}   {format_timestamp(test_gt_info['max'])}")

# # ============ Imputed 数据对齐检查 ============
# print(f"\n{'='*70}")
# print(f"[4] IMPUTED DATA ALIGNMENT CHECK")
# print(f"{'='*70}")

# def check_alignment(imputed_df, test_df, name, time_col):
#     """检查 imputed 数据与 test 数据的对齐情况"""
#     if imputed_df is None or test_df is None:
#         print(f"\n  {name}: Cannot check (data not available)")
#         return None
    
#     print(f"\n  {name}:")
#     print(f"    Imputed rows: {len(imputed_df)}")
#     print(f"    Test rows:    {len(test_df)}")
    
#     if len(imputed_df) == len(test_df):
#         print(f"    ✓ Row count matches!")
        
#         # 检查时间是否对齐
#         if time_col in imputed_df.columns and time_col in test_df.columns:
#             t_imp = imputed_df[time_col].values
#             t_test = test_df[time_col].values
            
#             if np.allclose(t_imp, t_test):
#                 print(f"    ✓ Time column aligned!")
#                 return 'aligned'
#             else:
#                 diff_idx = np.where(~np.isclose(t_imp, t_test))[0]
#                 print(f"    ✗ Time mismatch at {len(diff_idx)} positions")
#                 print(f"      First mismatch at index {diff_idx[0]}: imputed={t_imp[diff_idx[0]]}, test={t_test[diff_idx[0]]}")
#                 return 'time_mismatch'
#     else:
#         print(f"    ✗ Row count mismatch!")
        
#         # 分析 imputed 数据的来源
#         if train_info and test_gt_info:
#             total_rows = train_info['rows'] + test_gt_info['rows']
            
#             if len(imputed_df) == total_rows:
#                 print(f"    → Imputed seems to be FULL data (train + test)")
#                 print(f"    → Need to extract rows {train_info['rows']} to {total_rows} for test portion")
#                 return 'full_data'
#             elif len(imputed_df) == train_info['rows']:
#                 print(f"    → Imputed seems to be TRAIN data only")
#                 return 'train_only'
#             else:
#                 print(f"    → Unknown data source")
                
#                 # 尝试通过时间范围判断
#                 if time_col in imputed_df.columns:
#                     t_imp = imputed_df[time_col].values
#                     t_test = test_df[time_col].values
                    
#                     imp_min, imp_max = t_imp.min(), t_imp.max()
#                     test_min, test_max = t_test.min(), t_test.max()
                    
#                     print(f"\n    Time range comparison:")
#                     print(f"      Imputed: {format_timestamp(imp_min)} - {format_timestamp(imp_max)}")
#                     print(f"      Test:    {format_timestamp(test_min)} - {format_timestamp(test_max)}")
                    
#                     # 检查是否有重叠
#                     overlap_start = max(imp_min, test_min)
#                     overlap_end = min(imp_max, test_max)
                    
#                     if overlap_start < overlap_end:
#                         overlap_mask = (t_imp >= test_min) & (t_imp <= test_max)
#                         overlap_count = overlap_mask.sum()
#                         print(f"      Overlap: {overlap_count} points in test time range")
#                     else:
#                         print(f"      ✗ No time overlap!")
                
#                 return 'unknown'
    
#     return 'aligned'

# ws_status = check_alignment(pred_wavestitch, test_input, "WaveStitch", time_col)
# wsp_status = check_alignment(pred_wavestitchPlus, test_input, "WaveStitch+", time_col)

# # ============ 建议的对齐方案 ============
# print(f"\n{'='*70}")
# print(f"[5] ALIGNMENT SOLUTION")
# print(f"{'='*70}")

# def suggest_alignment(imputed_df, test_df, train_df, name, time_col):
#     """建议对齐方案并返回对齐后的数据"""
#     if imputed_df is None or test_df is None:
#         return None
    
#     if len(imputed_df) == len(test_df):
#         print(f"\n  {name}: Already aligned, no action needed")
#         return imputed_df
    
#     print(f"\n  {name}:")
    
#     # 方案1: 如果是 train+test 拼接
#     if train_df is not None:
#         total_rows = len(train_df) + len(test_df)
#         if len(imputed_df) == total_rows:
#             train_len = len(train_df)
#             print(f"    Solution: Extract test portion (rows {train_len} to {total_rows})")
#             print(f"    Code:")
#             print(f"      aligned_df = imputed_df.iloc[{train_len}:].reset_index(drop=True)")
            
#             aligned_df = imputed_df.iloc[train_len:].reset_index(drop=True)
#             print(f"    Result: {len(aligned_df)} rows")
#             return aligned_df
    
#     # 方案2: 按时间筛选
#     if time_col in imputed_df.columns and time_col in test_df.columns:
#         t_test = test_df[time_col].values
#         test_min, test_max = t_test.min(), t_test.max()
        
#         mask = (imputed_df[time_col] >= test_min) & (imputed_df[time_col] <= test_max)
#         aligned_df = imputed_df[mask].reset_index(drop=True)
        
#         if len(aligned_df) == len(test_df):
#             print(f"    Solution: Filter by time range [{test_min}, {test_max}]")
#             print(f"    Code:")
#             print(f"      mask = (imputed_df['{time_col}'] >= {test_min}) & (imputed_df['{time_col}'] <= {test_max})")
#             print(f"      aligned_df = imputed_df[mask].reset_index(drop=True)")
#             print(f"    Result: {len(aligned_df)} rows")
#             return aligned_df
#         else:
#             print(f"    ✗ Time filtering gave {len(aligned_df)} rows, expected {len(test_df)}")
    
#     print(f"    ✗ Could not find automatic alignment solution")
#     print(f"    Please check how imputed data was generated")
#     return None

# pred_wavestitch_aligned = suggest_alignment(pred_wavestitch, test_input, train_df, "WaveStitch", time_col)
# pred_wavestitchPlus_aligned = suggest_alignment(pred_wavestitchPlus, test_input, train_df, "WaveStitch+", time_col)

# # ============ 最终验证 ============
# print(f"\n{'='*70}")
# print(f"[6] FINAL VERIFICATION")
# print(f"{'='*70}")

# def verify_alignment(aligned_df, test_df, name, time_col):
#     """最终验证对齐结果"""
#     if aligned_df is None or test_df is None:
#         print(f"\n  {name}: ✗ Not available")
#         return False
    
#     print(f"\n  {name}:")
    
#     # 检查行数
#     if len(aligned_df) != len(test_df):
#         print(f"    ✗ Row count: {len(aligned_df)} vs {len(test_df)}")
#         return False
#     print(f"    ✓ Row count: {len(aligned_df)}")
    
#     # 检查时间
#     if time_col in aligned_df.columns and time_col in test_df.columns:
#         t_aligned = aligned_df[time_col].values
#         t_test = test_df[time_col].values
        
#         if np.allclose(t_aligned, t_test):
#             print(f"    ✓ Time column aligned")
#         else:
#             print(f"    ✗ Time column mismatch")
#             return False
    
#     return True

# ws_ok = verify_alignment(pred_wavestitch_aligned, test_input, "WaveStitch", time_col)
# wsp_ok = verify_alignment(pred_wavestitchPlus_aligned, test_input, "WaveStitch+", time_col)

# if ws_ok and wsp_ok:
#     print(f"\n{'='*70}")
#     print(f"✓ ALL DATA ALIGNED - Ready for comparison!")
#     print(f"{'='*70}")
# else:
#     print(f"\n{'='*70}")
#     print(f"✗ ALIGNMENT ISSUES - Please resolve before comparison")
#     print(f"{'='*70}")

# # ============ 导出对齐后的数据（可选） ============
# # if pred_wavestitch_aligned is not None:
# #     pred_wavestitch_aligned.to_csv(os.path.join(generated_dir, 'wavestitch_test_aligned.csv'), index=False)
# #     print(f"\n[SAVED] wavestitch_test_aligned.csv")

# # if pred_wavestitchPlus_aligned is not None:
# #     pred_wavestitchPlus_aligned.to_csv(os.path.join(generated_dir, 'wavestitchPlus_test_aligned.csv'), index=False)
# #     print(f"\n[SAVED] wavestitchPlus_test_aligned.csv")


#!/usr/bin/env python3
"""
诊断 test_gt.csv 在原始数据中的位置 + 简单可视化
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

# ============ 配置 ============
datafilename = 'amf'
base_dir = f'./work/EUR'
prepared_dir = f'{base_dir}/prepared_{datafilename}'
generated_dir = f'{base_dir}/generated_{datafilename}'

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ============ 加载 meta ============
meta_path = os.path.join(prepared_dir, 'meta.json')
with open(meta_path, 'r') as f:
    meta = json.load(f)

time_col = meta.get('time_col', 'time')
target_cols = meta.get('target_cols', [])

print(f"{'='*70}")
print(f"DATA LOCATION DIAGNOSTIC")
print(f"{'='*70}")

# ============ 加载所有数据 ============
train_df = pd.read_csv(os.path.join(prepared_dir, 'train.csv'))
test_gt = pd.read_csv(os.path.join(prepared_dir, 'test_gt.csv'))
test_input = pd.read_csv(os.path.join(prepared_dir, 'test_input.csv'))

ws_path = os.path.join(generated_dir, 'wavestitch_full_imputed_cleaned.csv')
wsp_path = os.path.join(generated_dir, 'wavestitchplus_v1_test_imputed_cleaned.csv')

pred_wavestitch = pd.read_csv(ws_path) if os.path.exists(ws_path) else None
pred_wavestitchPlus = pd.read_csv(wsp_path) if os.path.exists(wsp_path) else None

print(f"\n[Row Counts]")
print(f"  train:        {len(train_df)}")
print(f"  test_gt:      {len(test_gt)}")
print(f"  test_input:   {len(test_input)}")
print(f"  WaveStitch:   {len(pred_wavestitch) if pred_wavestitch is not None else 'N/A'}")
print(f"  WaveStitch+:  {len(pred_wavestitchPlus) if pred_wavestitchPlus is not None else 'N/A'}")

# ============ 拼接原始 train + test ============
original_full = pd.concat([train_df, test_gt], ignore_index=True)
print(f"  train+test:   {len(original_full)}")

# ============ 检查 imputed 数据对齐 ============
def check_and_align(imputed_df, original_full, train_df, test_df, name):
    """检查并对齐 imputed 数据"""
    if imputed_df is None:
        return None, None
    
    print(f"\n[{name} Alignment]")
    
    # 情况1: imputed 是完整数据 (train + test)
    if len(imputed_df) == len(original_full):
        print(f"  ✓ Imputed is FULL data (train + test)")
        imputed_test = imputed_df.iloc[len(train_df):].reset_index(drop=True)
        return imputed_df, imputed_test
    
    # 情况2: imputed 只有 test 部分
    elif len(imputed_df) == len(test_df):
        print(f"  ✓ Imputed is TEST data only")
        # 构造完整数据: train 原始 + test imputed
        imputed_full = pd.concat([train_df, imputed_df], ignore_index=True)
        return imputed_full, imputed_df
    
    else:
        print(f"  ✗ Unknown format: {len(imputed_df)} rows")
        return None, None

ws_full, ws_test = check_and_align(pred_wavestitch, original_full, train_df, test_input, "WaveStitch")
wsp_full, wsp_test = check_and_align(pred_wavestitchPlus, original_full, train_df, test_input, "WaveStitch+")

# ============ 可视化函数 ============
def plot_data_overview(feature_name, save=True):
    """
    可视化原始数据 vs imputed 数据
    显示 train/test 分割位置
    """
    if feature_name not in original_full.columns:
        print(f"[SKIP] {feature_name} not found")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    # 时间轴
    t_full = original_full[time_col].values
    t_train = train_df[time_col].values
    t_test = test_gt[time_col].values
    
    train_len = len(train_df)
    split_time = t_train[-1]  # train 最后一个时间点
    
    # 颜色
    c_train = '#2196F3'  # 蓝色
    c_test = '#FF9800'   # 橙色
    c_imputed = '#4CAF50'  # 绿色
    c_gap = '#F44336'    # 红色
    
    # === 子图1: 原始数据 (train + test_gt) ===
    ax1 = axes[0]
    
    # Train 部分
    train_vals = train_df[feature_name].values
    ax1.plot(range(train_len), train_vals, color=c_train, alpha=0.8, linewidth=0.8, label='Train')
    
    # Test 部分 (test_gt)
    test_vals = test_gt[feature_name].values
    ax1.plot(range(train_len, len(original_full)), test_vals, color=c_test, alpha=0.8, linewidth=0.8, label='Test (GT)')
    
    # 分割线
    ax1.axvline(x=train_len, color='black', linestyle='--', linewidth=2, label='Train/Test Split')
    
    # 标记缺失
    train_missing = np.isnan(train_vals).sum()
    test_missing = np.isnan(test_vals).sum()
    
    ax1.set_ylabel(feature_name)
    ax1.set_title(f'Original Data (train + test_gt) | Train missing: {train_missing}, Test missing: {test_missing}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # === 子图2: WaveStitch Imputed ===
    ax2 = axes[1]
    
    if ws_full is not None and feature_name in ws_full.columns:
        ws_vals = ws_full[feature_name].values
        
        # Train 部分
        ax2.plot(range(train_len), ws_vals[:train_len], color=c_train, alpha=0.8, linewidth=0.8, label='Train')
        
        # Test 部分
        ax2.plot(range(train_len, len(ws_full)), ws_vals[train_len:], color=c_imputed, alpha=0.8, linewidth=0.8, label='Test (Imputed)')
        
        ax2.axvline(x=train_len, color='black', linestyle='--', linewidth=2)
        
        # 统计
        ws_missing = np.isnan(ws_vals).sum()
        ax2.set_title(f'WaveStitch Imputed | Remaining missing: {ws_missing}')
    else:
        ax2.set_title('WaveStitch Imputed | N/A')
    
    ax2.set_ylabel(feature_name)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # === 子图3: WaveStitch+ Imputed ===
    ax3 = axes[2]
    
    if wsp_full is not None and feature_name in wsp_full.columns:
        wsp_vals = wsp_full[feature_name].values
        
        # Train 部分
        ax3.plot(range(train_len), wsp_vals[:train_len], color=c_train, alpha=0.8, linewidth=0.8, label='Train')
        
        # Test 部分
        ax3.plot(range(train_len, len(wsp_full)), wsp_vals[train_len:], color=c_imputed, alpha=0.8, linewidth=0.8, label='Test (Imputed)')
        
        ax3.axvline(x=train_len, color='black', linestyle='--', linewidth=2)
        
        wsp_missing = np.isnan(wsp_vals).sum()
        ax3.set_title(f'WaveStitch+ Imputed | Remaining missing: {wsp_missing}')
    else:
        ax3.set_title('WaveStitch+ Imputed | N/A')
    
    ax3.set_ylabel(feature_name)
    ax3.set_xlabel('Index')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'{datafilename} - {feature_name} | Train: {train_len} rows, Test: {len(test_gt)} rows', fontsize=12)
    plt.tight_layout()
    
    if save:
        output_dir = os.path.join(generated_dir, 'diagnostic_plots')
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{feature_name}_data_overview.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    
    plt.show()
    plt.close()


def plot_test_comparison(feature_name, save=True):
    """
    只看 test 部分：对比 test_input (有mask), test_gt, imputed
    """
    if feature_name not in test_input.columns:
        print(f"[SKIP] {feature_name} not found")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    n_test = len(test_input)
    x = range(n_test)
    
    # test_gt (Ground Truth)
    gt_vals = test_gt[feature_name].values
    ax.plot(x, gt_vals, color='#FF9800', alpha=0.6, linewidth=1.5, linestyle=':', label='Ground Truth')
    
    # test_input (有 mask 的输入)
    input_vals = test_input[feature_name].values
    observed_mask = ~np.isnan(input_vals)
    ax.scatter(np.array(x)[observed_mask], input_vals[observed_mask], 
               s=10, color='#2196F3', alpha=0.8, label='Observed', zorder=3)
    
    # WaveStitch imputed
    if ws_test is not None and feature_name in ws_test.columns:
        ws_vals = ws_test[feature_name].values
        ax.plot(x, ws_vals, color='#4CAF50', alpha=0.8, linewidth=1, label='WaveStitch')
    
    # WaveStitch+ imputed
    if wsp_test is not None and feature_name in wsp_test.columns:
        wsp_vals = wsp_test[feature_name].values
        ax.plot(x, wsp_vals, color='#9C27B0', alpha=0.8, linewidth=1, linestyle='--', label='WaveStitch+')
    
    # 统计
    n_observed = observed_mask.sum()
    n_masked = (~observed_mask).sum()
    
    ax.set_xlabel('Index (Test portion)')
    ax.set_ylabel(feature_name)
    ax.set_title(f'Test Data Comparison | Observed: {n_observed}, Masked/Gap: {n_masked}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        output_dir = os.path.join(generated_dir, 'diagnostic_plots')
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{feature_name}_test_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    
    plt.show()
    plt.close()


def plot_timeline_bar():
    """
    绘制简单的时间线条形图，显示各数据集的位置关系
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    datasets = []
    
    # 收集数据集信息
    if train_df is not None:
        datasets.append(('train.csv', 0, len(train_df), '#2196F3'))
    
    if test_gt is not None:
        datasets.append(('test_gt.csv', len(train_df), len(train_df) + len(test_gt), '#FF9800'))
    
    if test_input is not None:
        datasets.append(('test_input.csv', len(train_df), len(train_df) + len(test_input), '#FFC107'))
    
    if pred_wavestitch is not None:
        if len(pred_wavestitch) == len(train_df) + len(test_gt):
            datasets.append(('WaveStitch (full)', 0, len(pred_wavestitch), '#4CAF50'))
        else:
            datasets.append(('WaveStitch', len(train_df), len(train_df) + len(pred_wavestitch), '#4CAF50'))
    
    if pred_wavestitchPlus is not None:
        if len(pred_wavestitchPlus) == len(train_df) + len(test_gt):
            datasets.append(('WaveStitch+ (full)', 0, len(pred_wavestitchPlus), '#9C27B0'))
        else:
            datasets.append(('WaveStitch+', len(train_df), len(train_df) + len(pred_wavestitchPlus), '#9C27B0'))
    
    # 绘制条形
    y_pos = 0
    y_labels = []
    
    for name, start, end, color in datasets:
        ax.barh(y_pos, end - start, left=start, height=0.6, color=color, alpha=0.8, edgecolor='black')
        ax.text(start + (end - start) / 2, y_pos, f'{end - start} rows', 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        y_labels.append(name)
        y_pos += 1
    
    # 分割线
    if train_df is not None:
        ax.axvline(x=len(train_df), color='red', linestyle='--', linewidth=2, label='Train/Test Split')
    
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Row Index')
    ax.set_title(f'{datafilename} - Data Layout Overview')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加注释
    if train_df is not None and test_gt is not None:
        total = len(train_df) + len(test_gt)
        train_pct = 100 * len(train_df) / total
        test_pct = 100 * len(test_gt) / total
        ax.text(0.02, 0.98, f'Train: {train_pct:.1f}% | Test: {test_pct:.1f}%', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_dir = os.path.join(generated_dir, 'diagnostic_plots')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'data_layout_overview.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    
    plt.show()
    plt.close()


# ============ 运行可视化 ============
print(f"\n{'='*70}")
print(f"GENERATING VISUALIZATIONS")
print(f"{'='*70}")

# 1. 数据布局总览
print("\n[1] Data Layout Overview")
plot_timeline_bar()

# 2. 为前几个特征绘图
print("\n[2] Feature Plots")
for feature_name in target_cols[:3]:
    print(f"\n--- {feature_name} ---")
    plot_data_overview(feature_name)
    plot_test_comparison(feature_name)

print(f"\n{'='*70}")
print(f"[DONE] All diagnostic plots saved!")
print(f"{'='*70}")
# ```

# 这个脚本会生成：

# ## 1. 数据布局总览 (`data_layout_overview.png`)
# ```
# train.csv          |████████████████████████████████| 8000 rows
# test_gt.csv                                          |██████████| 2000 rows  
# test_input.csv                                       |██████████| 2000 rows
# WaveStitch (full)  |██████████████████████████████████████████| 10000 rows
# WaveStitch+ (full) |██████████████████████████████████████████| 10000 rows
#                    ^                                 ^
#                    0                            Train/Test Split
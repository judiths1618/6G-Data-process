
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import matplotlib

# ============ 设置中文字体 ============
# 方法1：使用系统中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 方法2：如果上面不生效，尝试查找可用字体
try:
    # Linux 系统
    if os.path.exists('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'):
        matplotlib.font_manager.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] + plt.rcParams['font.sans-serif']
except:
    pass

# 或者直接用英文避免问题
USE_ENGLISH = True  # 设为 False 使用中文



# ============ 配置 ============
datafilename = 'amf'
base_dir = f'./work/EUR'
prepared_dir = f'{base_dir}/prepared_{datafilename}'
generated_dir = f'{base_dir}/generated_{datafilename}'

# ============ 从 meta.json 读取配置 ============
meta_path = os.path.join(prepared_dir, 'meta.json')

with open(meta_path, 'r') as f:
    meta = json.load(f)

time_col = meta.get('time_col', 'time')
target_cols = meta.get('target_cols', [])
cond_cols = meta.get('cond_cols', [])

print(f"{'='*60}")
print(f"Dataset: {datafilename}")
print(f"Target columns ({len(target_cols)}): {target_cols}")
print(f"{'='*60}\n")

# ============ 加载数据 ============
raw_data = pd.read_csv(os.path.join(prepared_dir, 'test_input.csv'))
gt_path = os.path.join(prepared_dir, 'test_gt.csv')
gt_data = pd.read_csv(gt_path) if os.path.exists(gt_path) else None

pred_wavestitch = pd.read_csv(os.path.join(generated_dir, 'wavestitch_full_imputed_cleaned.csv'))
pred_wavestitchPlus = pd.read_csv(os.path.join(generated_dir, 'wavestitchplus_v1_test_imputed_cleaned.csv'))

print(f"Raw data shape: {raw_data.shape}")
print(f"WaveStitch shape: {pred_wavestitch.shape}")
print(f"WaveStitch+ shape: {pred_wavestitchPlus.shape}")
if gt_data is not None:
    print(f"Ground Truth shape: {gt_data.shape}")

# ============ 颜色定义（修改后） ============
COLORS = {
    'observed': '#2196F3',        # 蓝色 - 观测值
    'true_missing': '#F44336',    # 红色 - 真实缺失（gap + original_missing）
    'masked': '#9E9E9E',          # 灰色 - 人为 masked（用于测试）
    'wavestitch': '#4CAF50',      # 绿色 - WaveStitch
    'wavestitchplus': '#9C27B0',  # 紫色 - WaveStitch+
    'gt': '#FF9800',              # 橙色 - Ground Truth
}

# ============ 解析时间列 ============
def parse_time_column(df, time_col):
    if time_col not in df.columns:
        return pd.to_datetime(range(len(df)), unit='s')
    
    time_data = df[time_col]
    
    if pd.api.types.is_numeric_dtype(time_data):
        if time_data.max() > 1e12:
            return pd.to_datetime(time_data, unit='ms')
        else:
            return pd.to_datetime(time_data, unit='s')
    
    return pd.to_datetime(time_data)

time_index = parse_time_column(raw_data, time_col)
print(f"\nTime range: {time_index.min()} to {time_index.max()}")

# ============ 创建 Mask 分类函数（简化版） ============
def create_mask_classification(raw_data, gt_data, feature_name):
    """
    创建 mask 分类（简化为两类）:
    - observed: 有观测值
    - true_missing: 真实缺失（is_gap=True 或 GT也没有值）→ 红色
    - masked: GT有值但input没有（人为掩盖用于测试）→ 灰色
    
    返回: mask_type array
    """
    n = len(raw_data)
    mask_type = np.array(['observed'] * n, dtype=object)
    
    # 检查 is_gap 列
    has_is_gap = 'is_gap' in raw_data.columns
    if has_is_gap:
        is_gap = raw_data['is_gap'].values.astype(bool)
    else:
        is_gap = np.zeros(n, dtype=bool)
    
    # input 中的缺失
    if feature_name in raw_data.columns:
        input_missing = raw_data[feature_name].isna().values
    else:
        input_missing = np.ones(n, dtype=bool)
    
    # GT 中是否有值
    if gt_data is not None and feature_name in gt_data.columns:
        gt_has_value = ~gt_data[feature_name].isna().values
    else:
        gt_has_value = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if not input_missing[i]:
            mask_type[i] = 'observed'
        elif is_gap[i]:
            # is_gap=True → 真实缺失（红色）
            mask_type[i] = 'true_missing'
        elif gt_has_value[i]:
            # GT有值但input没有 → 人为masked（灰色）
            mask_type[i] = 'masked'
        else:
            # GT也没有 → 真实缺失（红色）
            mask_type[i] = 'true_missing'
    
    return mask_type

# ============ 打印 Mask 统计 ============
def print_mask_summary():
    print(f"\n{'='*70}")
    print(f"{'Feature':<20} {'Observed':>12} {'True Missing':>15} {'Masked':>12}")
    print(f"{'='*70}")
    
    for feature_name in target_cols:
        if feature_name not in raw_data.columns:
            continue
        
        mask_type = create_mask_classification(raw_data, gt_data, feature_name)
        
        n_observed = (mask_type == 'observed').sum()
        n_true_missing = (mask_type == 'true_missing').sum()
        n_masked = (mask_type == 'masked').sum()
        
        print(f"{feature_name:<20} {n_observed:>12} {n_true_missing:>15} {n_masked:>12}")
    
    print(f"{'='*70}")

print_mask_summary()

# ============ 创建输出目录 ============
output_dir = os.path.join(generated_dir, 'comparison_plots')
os.makedirs(output_dir, exist_ok=True)

# ============ 设置时间轴格式 ============
def setup_time_axis(ax, time_index):
    duration = time_index.max() - time_index.min()
    
    if duration.days > 30:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif duration.days > 1:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    elif duration.total_seconds() > 3600:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ============ 计算 MSE（只在 masked 区域） ============
def compute_mse_on_masked(feature_name):
    """
    只在 masked 区域计算 MSE（因为只有这部分有 ground truth）
    """
    if gt_data is None or feature_name not in gt_data.columns:
        return None, None, 0
    
    mask_type = create_mask_classification(raw_data, gt_data, feature_name)
    masked_indices = (mask_type == 'masked')
    
    if masked_indices.sum() == 0:
        return None, None, 0
    
    gt_values = gt_data[feature_name].values[masked_indices]
    
    mse_ws = None
    mse_wsp = None
    
    if feature_name in pred_wavestitch.columns:
        ws_values = pred_wavestitch[feature_name].values[masked_indices]
        valid = ~(np.isnan(gt_values) | np.isnan(ws_values))
        if valid.sum() > 0:
            mse_ws = np.mean((gt_values[valid] - ws_values[valid]) ** 2)
    
    if feature_name in pred_wavestitchPlus.columns:
        wsp_values = pred_wavestitchPlus[feature_name].values[masked_indices]
        valid = ~(np.isnan(gt_values) | np.isnan(wsp_values))
        if valid.sum() > 0:
            mse_wsp = np.mean((gt_values[valid] - wsp_values[valid]) ** 2)
    
    return mse_ws, mse_wsp, masked_indices.sum()

# ============ 打印 MSE 结果 ============
print("\n" + "="*80)
print("MSE on MASKED regions (where we have ground truth for evaluation)")
print("="*80)
print(f"{'Feature':<20} {'WaveStitch MSE':>18} {'WaveStitch+ MSE':>18} {'N Masked':>12}")
print("-"*80)

for feature_name in target_cols:
    mse_ws, mse_wsp, n_masked = compute_mse_on_masked(feature_name)
    
    ws_str = f"{mse_ws:.6f}" if mse_ws is not None else "N/A"
    wsp_str = f"{mse_wsp:.6f}" if mse_wsp is not None else "N/A"
    
    print(f"{feature_name:<20} {ws_str:>18} {wsp_str:>18} {n_masked:>12}")

print("="*80)

# ============ 绘图：完整对比图 ============
print("\n[PLOTTING] Comparison plots...")

for feature_name in target_cols:
    if feature_name not in raw_data.columns:
        continue
    
    mask_type = create_mask_classification(raw_data, gt_data, feature_name)
    
    raw_values = raw_data[feature_name].values
    observed_mask = (mask_type == 'observed')
    true_missing_mask = (mask_type == 'true_missing')
    masked_mask = (mask_type == 'masked')
    
    # 计算 MSE
    mse_ws, mse_wsp, n_masked = compute_mse_on_masked(feature_name)
    
    # 创建图：3个子图
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                              gridspec_kw={'height_ratios': [0.4, 2, 2]})
    
    # --- 子图 0: Mask 类型条 ---
    ax_mask = axes[0]
    
    for i in range(len(mask_type)):
        if mask_type[i] == 'true_missing':
            color = COLORS['true_missing']  # 红色
        elif mask_type[i] == 'masked':
            color = COLORS['masked']  # 灰色
        else:
            color = 'white'
        
        if i < len(time_index) - 1:
            ax_mask.axvspan(time_index[i], time_index[min(i+1, len(time_index)-1)], 
                           alpha=0.8, color=color, linewidth=0)
    
    ax_mask.set_ylim(0, 1)
    ax_mask.set_yticks([])
    
    # 图例
    legend_elements = [
        Patch(facecolor=COLORS['true_missing'], alpha=0.8, label='True Missing (红色)'),
        Patch(facecolor=COLORS['masked'], alpha=0.8, label='Masked for Test (灰色)'),
    ]
    ax_mask.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=9)
    
    n_true_missing = true_missing_mask.sum()
    n_masked_count = masked_mask.sum()
    ax_mask.set_title(f'Mask Types | True Missing: {n_true_missing} | Masked: {n_masked_count}', fontsize=10)
    
    # --- 子图 1: WaveStitch ---
    ax1 = axes[1]
    
    # 背景高亮
    for i in range(len(mask_type)):
        if mask_type[i] == 'true_missing':
            ax1.axvspan(time_index[i], time_index[min(i+1, len(time_index)-1)], 
                       alpha=0.2, color=COLORS['true_missing'], linewidth=0)
        elif mask_type[i] == 'masked':
            ax1.axvspan(time_index[i], time_index[min(i+1, len(time_index)-1)], 
                       alpha=0.2, color=COLORS['masked'], linewidth=0)
    
    # 绘制数据
    if feature_name in pred_wavestitch.columns:
        ws_values = pred_wavestitch[feature_name].values
        ax1.plot(time_index, ws_values, color=COLORS['wavestitch'], 
                alpha=0.8, linewidth=1, label='WaveStitch')
    
    # 观测值
    ax1.scatter(time_index[observed_mask], raw_values[observed_mask], 
                s=5, color=COLORS['observed'], alpha=0.7, label='Observed')
    
    # Ground Truth（只在 masked 区域显示）
    if gt_data is not None and feature_name in gt_data.columns:
        gt_values = gt_data[feature_name].values
        ax1.scatter(time_index[masked_mask], gt_values[masked_mask], 
                   s=15, color=COLORS['gt'], alpha=0.9, 
                   marker='x', label='Ground Truth', zorder=5)
    
    mse_str = f"MSE: {mse_ws:.6f}" if mse_ws is not None else "MSE: N/A"
    ax1.set_ylabel(feature_name, fontsize=10)
    ax1.set_title(f'WaveStitch | {mse_str}', fontsize=11)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # --- 子图 2: WaveStitch+ ---
    ax2 = axes[2]
    
    # 背景高亮
    for i in range(len(mask_type)):
        if mask_type[i] == 'true_missing':
            ax2.axvspan(time_index[i], time_index[min(i+1, len(time_index)-1)], 
                       alpha=0.2, color=COLORS['true_missing'], linewidth=0)
        elif mask_type[i] == 'masked':
            ax2.axvspan(time_index[i], time_index[min(i+1, len(time_index)-1)], 
                       alpha=0.2, color=COLORS['masked'], linewidth=0)
    
    # 绘制数据
    if feature_name in pred_wavestitchPlus.columns:
        wsp_values = pred_wavestitchPlus[feature_name].values
        ax2.plot(time_index, wsp_values, color=COLORS['wavestitchplus'], 
                alpha=0.8, linewidth=1, label='WaveStitch+')
    
    # 观测值
    ax2.scatter(time_index[observed_mask], raw_values[observed_mask], 
                s=5, color=COLORS['observed'], alpha=0.7, label='Observed')
    
    # Ground Truth
    if gt_data is not None and feature_name in gt_data.columns:
        ax2.scatter(time_index[masked_mask], gt_values[masked_mask], 
                   s=15, color=COLORS['gt'], alpha=0.9, 
                   marker='x', label='Ground Truth', zorder=5)
    
    mse_str = f"MSE: {mse_wsp:.6f}" if mse_wsp is not None else "MSE: N/A"
    ax2.set_ylabel(feature_name, fontsize=10)
    ax2.set_title(f'WaveStitch+ | {mse_str}', fontsize=11)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    setup_time_axis(ax2, time_index)
    ax2.set_xlabel('Time')
    
    plt.suptitle(f'{datafilename} - {feature_name}\n'
                 f'红色=真实缺失(无GT) | 灰色=Masked(有GT，用于评估)', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'{feature_name}_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()
    
    print(f"[SAVED] {save_path}")

# ============ 汇总 MSE 表格 ============
print("\n" + "="*80)
print("SUMMARY: MSE Comparison on Masked Regions")
print("="*80)

results = []
for feature_name in target_cols:
    mse_ws, mse_wsp, n_masked = compute_mse_on_masked(feature_name)
    if mse_ws is not None or mse_wsp is not None:
        results.append({
            'Feature': feature_name,
            'WaveStitch_MSE': mse_ws,
            'WaveStitch+_MSE': mse_wsp,
            'N_Masked': n_masked,
            'Better': 'WaveStitch' if (mse_ws and mse_wsp and mse_ws < mse_wsp) else 
                      ('WaveStitch+' if (mse_ws and mse_wsp and mse_wsp < mse_ws) else 'Tie/NA')
        })

if results:
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # 保存结果
    results_df.to_csv(os.path.join(output_dir, 'mse_comparison.csv'), index=False)
    print(f"\n[SAVED] {os.path.join(output_dir, 'mse_comparison.csv')}")
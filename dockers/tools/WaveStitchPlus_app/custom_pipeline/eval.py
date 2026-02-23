import json
import numpy as np
import pandas as pd
from pathlib import Path

def evaluate_predictions(prepared_dir: str, pred_csv: str):
    p = Path(prepared_dir)
    meta = json.loads((p / "meta.json").read_text())
    cols = meta["target_cols"]  # 只评估 target_cols（cond_cols不是生成目标）
    gt = pd.read_csv(p / "test_gt.csv")
    pred = pd.read_csv(pred_csv)

    holdout = np.load(p / "eval_holdout_mask.npy")  # shape [T_test]
    gt_arr = gt[cols].to_numpy(dtype=np.float32)
    pr_arr = pred[cols].to_numpy(dtype=np.float32)

    # mask: 在holdout行且GT不为NaN的元素
    mask = np.repeat(holdout[:, None], len(cols), axis=1) & ~np.isnan(gt_arr)

    mae = np.abs(pr_arr - gt_arr)[mask].mean()
    rmse = np.sqrt(((pr_arr - gt_arr) ** 2)[mask].mean())
    print("mae:", {float(mae)}, "rmse:", float(rmse))
    return {"mae": float(mae), "rmse": float(rmse)}

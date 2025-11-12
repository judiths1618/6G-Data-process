# import numpy as np
# import pandas as pd
# import torch

# from scipy import optimize

# import os
# import json

# import argparse

# from pandas.api.types import is_string_dtype
# from pandas.api.types import is_numeric_dtype

# import pdb

# DATA_DIR = 'datasets'
# torch.set_default_dtype(torch.float32)

# '''
# Load dataset, the category data is replaced by index of the category

# Note: 
# the returned data's column order is the same as the original data column order 

# '''


# def load_dataset(dataname, idx=0):
#     data_dir = f'{DATA_DIR}/{dataname}'

#     info_path = f'{DATA_DIR}/Info/{dataname}.json'

#     with open(info_path, 'r') as f:
#         info = json.load(f)

#     num_col_idx = info['num_col_idx']
#     cat_col_idx = info['cat_col_idx']
#     target_col_idx = info['target_col_idx']

#     data_path = f'{data_dir}/data.csv'
#     train_path = f'{data_dir}/train.csv'
#     test_path = f'{data_dir}/test.csv'

#     data_df = pd.read_csv(data_path)
#     train_df = pd.read_csv(train_path)
#     test_df = pd.read_csv(test_path)

#     cols = train_df.columns

#     # data_num = data_df[cols[num_col_idx]].values.astype(np.float32)
#     data_cat = data_df[cols[cat_col_idx]].astype(str)
#     data_y = data_df[cols[target_col_idx]]

#     train_num = train_df[cols[num_col_idx]].values.astype(np.float32)
#     train_cat = train_df[cols[cat_col_idx]].astype(str)
#     train_y = train_df[cols[target_col_idx]]

#     test_num = test_df[cols[num_col_idx]].values.astype(np.float32)
#     test_cat = test_df[cols[cat_col_idx]].astype(str)
#     test_y = test_df[cols[target_col_idx]]

#     cat_columns = data_cat.columns
#     target_columns = data_y.columns

#     train_cat_idx, test_cat_idx = None, None

#     # Save target idx for target columns
#     if len(target_col_idx) != 0 and not is_numeric_dtype(data_y[target_columns[0]]):
#         if not os.path.exists(f'{data_dir}/{target_columns[0]}_map_idx.json'):
#             print('Creating maps')
#             for column in target_columns:
#                 map_path_bin = f'{data_dir}/{column}_map_bin.json'
#                 map_path_idx = f'{data_dir}/{column}_map_idx.json'
#                 categories = data_y[column].unique()
#                 num_categories = len(categories)

#                 num_bits = (num_categories - 1).bit_length()

#                 category_to_binary = {category: format(index, '0' + str(num_bits) + 'b') for index, category in
#                                       enumerate(categories)}
#                 category_to_idx = {category: index for index, category in enumerate(categories)}

#                 with open(map_path_bin, 'w') as f:
#                     json.dump(category_to_binary, f)
#                 with open(map_path_idx, 'w') as f:
#                     json.dump(category_to_idx, f)

#         train_target_idx = []
#         test_target_idx = []

#         for column in target_columns:
#             map_path_idx = f'{data_dir}/{column}_map_idx.json'

#             with open(map_path_idx, 'r') as f:
#                 category_to_idx = json.load(f)

#             train_target_idx_i = train_y[column].map(category_to_idx).to_numpy().astype(np.float32)
#             test_target_idx_i = test_y[column].map(category_to_idx).to_numpy().astype(np.float32)

#             train_target_idx.append(train_target_idx_i)
#             test_target_idx.append(test_target_idx_i)

#         train_target_idx = np.stack(train_target_idx, axis=1)
#         test_target_idx = np.stack(test_target_idx, axis=1)

#     else:
#         # abuse notation, if the target column is numeric, we still use call it target_idx
#         train_target_idx = train_y.to_numpy().astype(np.float32)
#         test_target_idx = test_y.to_numpy().astype(np.float32)

#     # ========================================================

#     # Save cat idx for cat columns
#     if len(cat_col_idx) != 0 and not os.path.exists(f'{data_dir}/{cat_columns[0]}_map_idx.json'):
#         print('Creating maps')
#         for column in cat_columns:
#             map_path_bin = f'{data_dir}/{column}_map_bin.json'
#             map_path_idx = f'{data_dir}/{column}_map_idx.json'
#             categories = data_cat[column].unique()
#             num_categories = len(categories)

#             num_bits = (num_categories - 1).bit_length()

#             category_to_binary = {category: format(index, '0' + str(num_bits) + 'b') for index, category in
#                                   enumerate(categories)}
#             category_to_idx = {category: index for index, category in enumerate(categories)}

#             with open(map_path_bin, 'w') as f:
#                 json.dump(category_to_binary, f)
#             with open(map_path_idx, 'w') as f:
#                 json.dump(category_to_idx, f)

#     train_cat_idx = []
#     test_cat_idx = []

#     for column in cat_columns:
#         map_path_idx = f'{data_dir}/{column}_map_idx.json'

#         with open(map_path_idx, 'r') as f:
#             category_to_idx = json.load(f)

#         train_cat_idx_i = train_cat[column].map(category_to_idx).to_numpy().astype(np.float32)
#         test_cat_idx_i = test_cat[column].map(category_to_idx).to_numpy().astype(np.float32)

#         train_cat_idx.append(train_cat_idx_i)
#         test_cat_idx.append(test_cat_idx_i)

#     # Four situations:
#     # 1. No target columns, no cat columns
#     # 2. No target columns, has cat columns
#     # 3. Has target columns, no cat columns
#     # 4. Has target columns, has cat columns
#     if len(target_col_idx) == 0:

#         if len(cat_col_idx) == 0:
#             train_X = train_num
#             test_X = test_num

#             # rearange the column order
#             train_X = train_X[:, num_col_idx]
#             test_X = test_X[:, num_col_idx]
#         else:
#             train_cat_idx = np.stack(train_cat_idx, axis=1)
#             test_cat_idx = np.stack(test_cat_idx, axis=1)

#             train_X = np.concatenate([train_num, train_cat_idx], axis=1)
#             test_X = np.concatenate([test_num, test_cat_idx], axis=1)

#             # rearange the column order
#             train_X = train_X[:, np.concatenate([num_col_idx, cat_col_idx])]
#             test_X = test_X[:, np.concatenate([num_col_idx, cat_col_idx])]

#     else:
#         if len(cat_col_idx) == 0:
#             train_X = np.concatenate([train_num, train_target_idx], axis=1)
#             test_X = np.concatenate([test_num, test_target_idx], axis=1)

#             # rearange the column order
#             train_X = train_X[:, np.concatenate([num_col_idx, target_col_idx])]
#             test_X = test_X[:, np.concatenate([num_col_idx, target_col_idx])]

#         else:
#             train_cat_idx = np.stack(train_cat_idx, axis=1)
#             test_cat_idx = np.stack(test_cat_idx, axis=1)

#             train_X = np.concatenate([train_num, train_cat_idx, train_target_idx], axis=1)
#             test_X = np.concatenate([test_num, test_cat_idx, test_target_idx], axis=1)

#             # rearange the column order
#             train_X = train_X[:, np.concatenate([num_col_idx, cat_col_idx, target_col_idx])]
#             test_X = test_X[:, np.concatenate([num_col_idx, cat_col_idx, target_col_idx])]

#     return train_X, test_X


# #### Quantile ######
# def quantile(X, q, dim=None):
#     """
#     Returns the q-th quantile.

#     Parameters
#     ----------
#     X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
#         Input data.

#     q : float
#         Quantile level (starting from lower values).

#     dim : int or None, default = None
#         Dimension allong which to compute quantiles. If None, the tensor is flattened and one value is returned.


#     Returns
#     -------
#         quantiles : torch.DoubleTensor

#     """
#     return X.kthvalue(int(q * len(X)), dim=dim)[0]


# ##################### MISSING DATA MECHANISMS #############################

# ##### Missing At Random ######

# def MAR_mask(X, p, p_obs):
#     """
#     Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
#     randomly selected. The remaining variables have missing values according to a logistic model with random weights,
#     re-scaled so as to attain the desired proportion of missing values on those variables.

#     Parameters
#     ----------
#     X : torch.DoubleTensor or np.ndarray, shape (n, d)
#         Data for which missing values will be simulated. If a numpy array is provided,
#         it will be converted to a pytorch tensor.

#     p : float
#         Proportion of missing values to generate for variables which will have missing values.

#     p_obs : float
#         Proportion of variables with *no* missing values that will be used for the logistic masking model.

#     Returns
#     -------
#     mask : torch.BoolTensor or np.ndarray (depending on type of X)
#         Mask of generated missing values (True if the value is missing).

#     """

#     n, d = X.shape

#     to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
#     if not to_torch:
#         X = torch.from_numpy(X)

#     mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

#     d_obs = max(int(p_obs * d), 1)  ## number of variables that will have no missing values (at least one variable)
#     d_na = d - d_obs  ## number of variables that will have missing values

#     ### Sample variables that will all be observed, and those with missing values:
#     idxs_obs = np.random.choice(d, d_obs, replace=False)
#     idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

#     ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
#     ### The parameters of this logistic model are random.

#     ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
#     coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
#     ### Pick the intercepts to have a desired amount of missing values
#     intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

#     ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

#     ber = torch.rand(n, d_na)
#     mask[:, idxs_nas] = ber < ps

#     return mask


# ##### Missing not at random ######

# def MNAR_mask_logistic(X, p, p_params=.3, exclude_inputs=True):
#     """
#     Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
#     (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
#     inputs can also be missing.
#     (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
#     determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
#     depend on masked values.
#     In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.

#     Parameters
#     ----------
#     X : torch.DoubleTensor or np.ndarray, shape (n, d)
#         Data for which missing values will be simulated.
#         If a numpy array is provided, it will be converted to a pytorch tensor.

#     p : float
#         Proportion of missing values to generate for variables which will have missing values.

#     p_params : float
#         Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).

#     exclude_inputs : boolean, default=True
#         True: mechanism (ii) is used, False: (i)

#     Returns
#     -------
#     mask : torch.BoolTensor or np.ndarray (depending on type of X)
#         Mask of generated missing values (True if the value is missing).

#     """

#     n, d = X.shape

#     to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
#     if not to_torch:
#         X = torch.from_numpy(X)

#     mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

#     d_params = max(int(p_params * d), 1) if exclude_inputs else d  ## number of variables used as inputs (at least 1)
#     d_na = d - d_params if exclude_inputs else d  ## number of variables masked with the logistic model

#     ### Sample variables that will be parameters for the logistic regression:
#     idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
#     idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

#     ### Other variables will have NA proportions selected by a logistic model
#     ### The parameters of this logistic model are random.

#     ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
#     coeffs = pick_coeffs(X, idxs_params, idxs_nas)
#     ### Pick the intercepts to have a desired amount of missing values
#     intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

#     ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)

#     ber = torch.rand(n, d_na)
#     mask[:, idxs_nas] = ber < ps

#     ## If the inputs of the logistic model are excluded from MNAR missingness,
#     ## mask some values used in the logistic model at random.
#     ## This makes the missingness of other variables potentially dependent on masked values

#     if exclude_inputs:
#         mask[:, idxs_params] = torch.rand(n, d_params) < p

#     return mask


# def MNAR_self_mask_logistic(X, p):
#     """
#     Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
#     given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
#     to another). The intercepts are selected to attain the desired missing rate.

#     Parameters
#     ----------
#     X : torch.DoubleTensor or np.ndarray, shape (n, d)
#         Data for which missing values will be simulated.
#         If a numpy array is provided, it will be converted to a pytorch tensor.

#     p : float
#         Proportion of missing values to generate for variables which will have missing values.

#     Returns
#     -------
#     mask : torch.BoolTensor or np.ndarray (depending on type of X)
#         Mask of generated missing values (True if the value is missing).

#     """

#     n, d = X.shape

#     to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
#     if not to_torch:
#         X = torch.from_numpy(X)

#     ### Variables will have NA proportions that depend on those observed variables, through a logistic model
#     ### The parameters of this logistic model are random.

#     ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
#     coeffs = pick_coeffs(X, self_mask=True)
#     ### Pick the intercepts to have a desired amount of missing values
#     intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

#     ps = torch.sigmoid(X * coeffs + intercepts)

#     ber = torch.rand(n, d) if to_torch else np.random.rand(n, d)
#     mask = ber < ps if to_torch else ber < ps.numpy()

#     return mask


# def MNAR_mask_quantiles(X, p, q, p_params, cut='both', MCAR=False):
#     """
#     Missing not at random mechanism with quantile censorship. First, a subset of variables which will have missing
#     variables is randomly selected. Then, missing values are generated on the q-quantiles at random. Since
#     missingness depends on quantile information, it depends on masked values, hence this is a MNAR mechanism.

#     Parameters
#     ----------
#     X : torch.DoubleTensor or np.ndarray, shape (n, d)
#         Data for which missing values will be simulated.
#         If a numpy array is provided, it will be converted to a pytorch tensor.

#     p : float
#         Proportion of missing values to generate for variables which will have missing values.

#     q : float
#         Quantile level at which the cuts should occur

#     p_params : float
#         Proportion of variables that will have missing values

#     cut : 'both', 'upper' or 'lower', default = 'both'
#         Where the cut should be applied. For instance, if q=0.25 and cut='upper', then missing values will be generated
#         in the upper quartiles of selected variables.
        
#     MCAR : bool, default = True
#         If true, masks variables that were not selected for quantile censorship with a MCAR mechanism.
        
#     Returns
#     -------
#     mask : torch.BoolTensor or np.ndarray (depending on type of X)
#         Mask of generated missing values (True if the value is missing).

#     """
#     n, d = X.shape

#     to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
#     if not to_torch:
#         X = torch.from_numpy(X)

#     mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

#     d_na = max(int(p_params * d), 1)  ## number of variables that will have NMAR values

#     ### Sample variables that will have imps at the extremes
#     idxs_na = np.random.choice(d, d_na, replace=False)  ### select at least one variable with missing values

#     ### check if values are greater/smaller that corresponding quantiles
#     if cut == 'upper':
#         quants = quantile(X[:, idxs_na], 1 - q, dim=0)
#         m = X[:, idxs_na] >= quants
#     elif cut == 'lower':
#         quants = quantile(X[:, idxs_na], q, dim=0)
#         m = X[:, idxs_na] <= quants
#     elif cut == 'both':
#         u_quants = quantile(X[:, idxs_na], 1 - q, dim=0)
#         l_quants = quantile(X[:, idxs_na], q, dim=0)
#         m = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)

#     ### Hide some values exceeding quantiles
#     ber = torch.rand(n, d_na)
#     mask[:, idxs_na] = (ber < p) & m

#     if MCAR:
#         ## Add a mcar mecanism on top
#         mask = mask | (torch.rand(n, d) < p)

#     return mask


# def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
#     n, d = X.shape
#     if self_mask:
#         coeffs = torch.randn(d)
#         Wx = X * coeffs
#         coeffs /= torch.std(Wx, 0)
#     else:
#         d_obs = len(idxs_obs)
#         d_na = len(idxs_nas)
#         coeffs = torch.randn(d_obs, d_na, dtype=X.dtype)
#         Wx = X[:, idxs_obs].mm(coeffs)
#         coeffs /= torch.std(Wx, 0, keepdim=True)
#     return coeffs


# def fit_intercepts(X, coeffs, p, self_mask=False):
#     if self_mask:
#         d = len(coeffs)
#         intercepts = torch.zeros(d)
#         for j in range(d):
#             def f(x):
#                 return torch.sigmoid(X * coeffs[j] + x).mean().item() - p

#             intercepts[j] = optimize.bisect(f, -50, 50)
#     else:
#         d_obs, d_na = coeffs.shape
#         intercepts = torch.zeros(d_na)
#         for j in range(d_na):
#             def f(x):
#                 return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p

#             left, right = -500, 500
#             intercepts[j] = optimize.bisect(f, left, right)
#     return intercepts


# # def generate_mask(X, mask_type, p, mask_num, reproduce=True):
# #     print('missing probability:', p)

# #     q = 0.3  # by default 30% will be held out and not missing for MAR and MNAR
# #     if p > 0.7:
# #         q = 0.1
# #     masks = []
# #     for i in range(mask_num):
# #         mask = None
# #         if mask_type == 'MCAR':
# #             mask = np.random.rand(*X.shape) < p
# #         elif mask_type == 'MAR':
# #             mask = MAR_mask(X, p=p / (1 - q), p_obs=q)
# #         elif mask_type == 'MNAR_logistic_T2':
# #             mask = MNAR_mask_logistic(X, p=p, p_params=q, exclude_inputs=True)
# #         else:
# #             print("error, unspecified masking pattern")
# #             exit()
# #         # row, col = X.shape
# #         # print(f'{mask_type}, {p}, missing prob:', np.sum(mask) / (row * col))
# #         masks.append(mask)
# #     return np.array(masks)

# def generate_mask(X, mask_type, p, mask_num, reproduce=True, exclude_cols=None, return_observed: bool = False):
#     """
#     Generate masks for dataset X.
#     - X: pandas.DataFrame or numpy.ndarray
#     - exclude_cols: list of original column NAMES to exclude from MAR/MNAR generation (these will be set as NOT missing)
#     - return_observed: if True return masks with True==OBSERVED (i.e. invert final masks)
#     Returns: np.ndarray shape (mask_num, N, D_full), dtype=bool, default True==MISSING
#     """
#     print('missing probability:', p)

#     # heuristic guard: if user accidentally passes encoded high-dimensional matrix, refuse and give guidance
#     if isinstance(X, np.ndarray):
#         if X.dtype == object:
#             # leave conversion to downstream code
#             pass
#         else:
#             # simple heuristic for one-hot: many columns with values in {0,1}
#             unique_vals = np.unique(X)
#             if X.shape[1] > 50 and set(np.unique(unique_vals)).issubset({0.0, 1.0, 0, 1}):
#                 raise ValueError(
#                     "generate_mask was given a high-dimensional encoded array (likely OHE). "
#                     "Generate masks on the original DataFrame (preprocessor.df_test) and then "
#                     "use Preprocessor.extend_mask to expand column-level masks to encoded width, "
#                     "or call generate_mask(..., exclude_cols=[...])."
#                 )

#     # prepare numeric matrix for mask generation, possibly dropping excluded columns
#     excluded = list(exclude_cols) if exclude_cols is not None else []
#     if isinstance(X, pd.DataFrame):
#         col_names = list(X.columns)
#         proc_cols = [c for c in col_names if c not in excluded]
#         if len(proc_cols) == 0:
#             raise ValueError("All columns excluded from mask generation.")
#         X_proc_df = X[proc_cols].copy()
#         for col in X_proc_df.columns:
#             if is_numeric_dtype(X_proc_df[col]):
#                 X_proc_df[col] = pd.to_numeric(X_proc_df[col], errors="coerce")
#             else:
#                 X_proc_df[col] = pd.Categorical(X_proc_df[col].astype(str)).codes.astype(np.float32)
#         X_num = X_proc_df.to_numpy(dtype=np.float32)
#     else:
#         # numpy path
#         arr = np.asarray(X)
#         if arr.dtype == object:
#             # convert columns robustly via pandas
#             X_num = pd.DataFrame(arr).apply(
#                 lambda s: pd.to_numeric(s, errors="coerce") if is_numeric_dtype(s) else pd.Categorical(s.astype(str)).codes
#             ).to_numpy(dtype=np.float32)
#             col_names = [f"col{i}" for i in range(X_num.shape[1])]
#             proc_cols = col_names
#             excluded = []
#         else:
#             X_num = arr.astype(np.float32, copy=False)
#             col_names = [f"col{i}" for i in range(X_num.shape[1])]
#             proc_cols = col_names

#     q = 0.3
#     if p > 0.7:
#         q = 0.1

#     masks_proc = []
#     for _ in range(mask_num):
#         if mask_type == 'MCAR':
#             mask_p = np.random.rand(*X_num.shape) < p
#         elif mask_type == 'MAR':
#             mask_p = MAR_mask(X_num, p=p / (1 - q), p_obs=q)
#         elif mask_type == 'MNAR_logistic_T2':
#             mask_p = MNAR_mask_logistic(X_num, p=p, p_params=q, exclude_inputs=True)
#         else:
#             raise ValueError(f"error, unspecified masking pattern: {mask_type}")
#         masks_proc.append(mask_p)
#     masks_proc = np.array(masks_proc, dtype=bool)  # (mask_num, N, D_proc) True==MISSING

#     # reinsert excluded columns as NOT missing (False) into full-order mask
#     if isinstance(X, pd.DataFrame):
#         D_full = len(col_names)
#         N = masks_proc.shape[1]
#         masks_full = np.zeros((mask_num, N, D_full), dtype=bool)

#         # map processed column names back to positions in the full DataFrame
#         proc_idx_map = np.array([col_names.index(c) for c in proc_cols], dtype=int)

#         for i in range(mask_num):
#             # Avoid advanced-indexing axis swap by indexing in two steps
#             masks_full[i][:, proc_idx_map] = masks_proc[i]

#         # excluded columns remain False (NOT missing)
#         masks = masks_full
#     else:
#         masks = masks_proc


#     if return_observed:
#         return ~masks
#     return masks

# imputers/generate_mask.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Sequence

from pandas.api.types import is_numeric_dtype


def _safe_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric (fill non-numeric with 0 or median)."""
    out = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().all():
            out[c] = pd.Series(np.zeros(len(df)), index=df.index)
        else:
            out[c] = s.fillna(s.median())
    return pd.DataFrame(out, index=df.index)


def _pick_drivers(df: pd.DataFrame, k: int = 3, exclude: set[str] = None) -> List[str]:
    """Select informative columns with variance to drive MAR masking."""
    exclude = exclude or set()
    candidates = []
    for c in df.columns:
        if c in exclude:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 5 and s.var() > 0:
            candidates.append((c, s.var()))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in candidates[:k]] or list(df.columns[:k])


def generate_mask(
    df: pd.DataFrame,
    *,
    mask_type: str = "MCAR",           # "MCAR", "MAR", "MNAR_logistic_T2"
    mask_num: int = 5,
    p: float = 0.2,                    # target missing prob for maskable cells
    exclude_cols: Optional[Sequence[str]] = None,
    include_cols: Optional[Sequence[str]] = None,
    disallow_full_row_missing: bool = True,
    random_state: Optional[int] = 42,
    return_observed: bool = False,
) -> List[np.ndarray]:
    """
    Generate missingness masks for a given DataFrame.

    Returns a list of (N, D) boolean arrays aligned with df.columns.
      - If return_observed=False: True == MISSING
      - If return_observed=True:  True == OBSERVED
    """
    assert 0 <= p <= 1, "Missing probability p must be in [0, 1]"
    N, D = df.shape
    if mask_num <= 0:
        return []

    rng = np.random.default_rng(random_state)

    exclude = set(exclude_cols or [])
    if include_cols is None:
        maskable_cols = [c for c in df.columns if c not in exclude]
    else:
        maskable_cols = [c for c in include_cols if c in df.columns and c not in exclude]

    if len(maskable_cols) == 0:
        base = np.ones((N, D), dtype=bool) if return_observed else np.zeros((N, D), dtype=bool)
        return [base.copy() for _ in range(mask_num)]

    col2idx = {c: i for i, c in enumerate(df.columns)}
    maskable_idx = np.array([col2idx[c] for c in maskable_cols], dtype=int)

    out: List[np.ndarray] = []
    for _ in range(mask_num):
        m = np.zeros((N, D), dtype=bool)  # True == MISSING (internal convention)

        if mask_type.upper() == "MCAR":
            m[:, maskable_idx] = rng.random((N, len(maskable_idx))) < p

        elif mask_type.upper() == "MAR":
            drivers = _pick_drivers(df[maskable_cols], k=min(3, len(maskable_cols)))
            Xd = _safe_numeric(df, drivers).to_numpy(dtype=float)  # (N, k)
            w = rng.normal(size=Xd.shape[1])
            logits = (Xd @ w) / (np.std(Xd @ w) + 1e-8)
            probs = 1 / (1 + np.exp(-logits))
            probs = (probs - probs.mean()) / (probs.std() + 1e-8) * 0.5 + 0.5
            probs = np.clip(probs, 0, 1)
            row_draws = rng.random((N, len(maskable_idx)))
            m[:, maskable_idx] = row_draws < probs[:, None] * (p / max(probs.mean(), 1e-8))

        elif mask_type.upper() in {"MNAR", "MNAR_LOGISTIC_T2"}:
            S = _safe_numeric(df, maskable_cols).to_numpy(dtype=float)  # (N, M)
            S = (S - np.nanmean(S, axis=0)) / (np.nanstd(S, axis=0) + 1e-8)
            logits = np.abs(S)
            probs = 1 / (1 + np.exp(-logits))
            scale = p / max(probs.mean(), 1e-8)
            probs = np.clip(probs * scale, 0, 1)
            draws = rng.random(probs.shape)
            miss = draws < probs
            m[:, maskable_idx] = miss

        else:
            raise ValueError(f"Unknown mask_type='{mask_type}'")

        if disallow_full_row_missing:
            all_missing = m[:, maskable_idx].all(axis=1)
            if np.any(all_missing):
                flip_idx = rng.integers(0, len(maskable_idx), size=all_missing.sum())
                m[all_missing, maskable_idx[flip_idx]] = False

        out.append(~m if return_observed else m)

    return out

def eval_numeric_and_categorical(X_pred, X_true, mask_missing, num_numeric, train_std_numeric):
    """
    X_pred, X_true: 形状 (N, D_encoded)，已 decodeNp，前 num_numeric 列是数值(float)，其后为字符串类别
    mask_missing:   形状 (N, D_encoded)，True==缺失（即当时被遮掉的位置）
    train_std_numeric: 形状 (num_numeric, )，来自训练集，用于 NRMSE
    返回: (mse_in_original_units, acc_percent, nrmse_mean_over_numeric_columns)
    """
    # --- 数值 ---
    num_mask = mask_missing[:, :num_numeric]
    num_true = X_true[:, :num_numeric].astype(float)
    num_pred = X_pred[:, :num_numeric].astype(float)

    # 仅缺失处
    diff2 = (num_true[num_mask] - num_pred[num_mask]) ** 2
    mse = float(diff2.mean()) if diff2.size else 0.0

    # NRMSE：对每个数值列, 只在该列缺失处计算 RMSE / std_train_col，然后对列取平均
    nrmse_list = []
    for j in range(num_numeric):
        col_mask = num_mask[:, j]
        if not np.any(col_mask):
            continue
        rmse_j = np.sqrt(np.mean((num_true[col_mask, j] - num_pred[col_mask, j]) ** 2))
        denom = float(train_std_numeric[j]) if float(train_std_numeric[j]) > 1e-12 else 1.0
        nrmse_list.append(rmse_j / denom)
    nrmse = float(np.mean(nrmse_list)) if nrmse_list else None

    # --- 分类 ---
    cat_true = X_true[:, num_numeric:]
    cat_pred = X_pred[:, num_numeric:]
    cat_mask = mask_missing[:, num_numeric:]

    if cat_true.size == 0 or not np.any(cat_mask):
        acc = 100.0
    else:
        # 防空格：strip 一下
        true_flat = np.char.strip(cat_true[cat_mask].astype(str))
        pred_flat = np.char.strip(cat_pred[cat_mask].astype(str))
        acc = float((true_flat == pred_flat).mean() * 100.0)

    return mse, acc, nrmse

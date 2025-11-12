# helpers/mask_mapping.py
import numpy as np
from typing import List

def mask_original_to_encoded(prepper, mask_orig: np.ndarray) -> np.ndarray:
    """
    Map a (N, D_original) mask (aligned to df.columns) to encoded layout
    [num_idx..., cat_idx...]. Returns shape (N, D_encoded).
    Assumes mask uses the SAME semantic as given (True==MISSING or True==OBSERVED).
    """
    assert mask_orig.ndim == 2 and mask_orig.shape[1] == len(prepper.df.columns)
    enc_idx = prepper.num_idx + prepper.cat_idx
    return mask_orig[:, enc_idx]

def mask_encoded_to_original(prepper, mask_enc: np.ndarray) -> np.ndarray:
    """
    Inverse mapping: (N, D_encoded) -> (N, D_original).
    Fills excluded columns with False.
    """
    assert mask_enc.ndim == 2 and mask_enc.shape[1] == (len(prepper.num_idx) + len(prepper.cat_idx))
    N = mask_enc.shape[0]
    D = len(prepper.df.columns)
    out = np.zeros((N, D), dtype=bool)
    enc_idx = np.array(prepper.num_idx + prepper.cat_idx, dtype=int)
    out[:, enc_idx] = mask_enc
    return out

# tests/test_mask_correctness.py
import numpy as np
import pandas as pd
from helpers.preprocessor import Preprocessor
from imputers.generate_mask import generate_mask
from helpers.mask_mapping import mask_original_to_encoded

def check_mask(df: pd.DataFrame, prepper: Preprocessor):
    # 1) Shapes
    masks = generate_mask(df, mask_type="MCAR", mask_num=3, p=0.3, exclude_cols=[], random_state=123)
    assert len(masks) == 3
    for m in masks:
        assert m.shape == (len(df), len(df.columns))

    # 2) Mapping shapes
    m0_enc = mask_original_to_encoded(prepper, masks[0])
    Denc = len(prepper.num_idx) + len(prepper.cat_idx)
    assert m0_enc.shape == (len(df), Denc)

    # 3) Encode/Decode idempotence on “no-missing” path
    X_ord = prepper.encodeDf("Ordinal", df)
    # decode then re-encode — just checks shape/ordering consistency, not exact roundtrip of categories
    dec = prepper.decodeNp("Ordinal", X_ord)
    assert dec.shape == X_ord.shape  # encoded width preserved during decode/encode flow

    # 4) Semantics True==MISSING
    # mask 30% cells; ensure empirical rate ~p on maskable subset
    m_all = masks[0]
    maskable = np.ones_like(m_all, dtype=bool)
    # if you exclude some columns: set those positions False in `maskable`
    emp = m_all[maskable].mean()
    assert 0.15 <= emp <= 0.45, f"Empirical missing {emp:.3f} far from target"

    print("Mask generation + mapping checks passed.")

if __name__ == "__main__":
    # tiny demo df
    df = pd.DataFrame({
        "num1": [1,2,3,4,5],
        "num2": [10,20,30,40,50],
        "cat":  ["a","b","a","b","a"],
        "path": ["/a","/b","/c","/d","/e"],
    })
    # Suppose your Preprocessor excludes 'path' automatically
    prepper = Preprocessor(dataname="Scenario33", data_dir="curated/DeepSense/scenario33/")
    check_mask(df, prepper)

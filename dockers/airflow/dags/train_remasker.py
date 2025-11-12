import numpy as np
import pandas as pd
# from utils import get_args_parser
from imputers.remasker.remasker_impute import ReMasker
import os, pandas as pd, pickle
import numpy as np
import torch
from tqdm import tqdm
from hyperimpute.plugins.imputers import Imputers
from helpers.preprocessor import Preprocessor
from imputers.generate_mask import generate_mask
from hyperimpute.plugins.utils.metrics import RMSE
from hyperimpute.plugins.utils.simulate import simulate_nan
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from hyperimpute.utils.serialization import load, save
import argparse

parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='Scenario1', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')


args = parser.parse_args()
# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = f'cuda:{args.gpu}'
else:
    args.device = 'cpu'


# --- stability on macOS / BLAS ---
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS",
          "CATBOOST_THREAD_COUNT","XGB_NUM_THREADS"]:
    os.environ[k] = "1"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"

# 2) Generate mask based on the amputation mechanism
# def ampute(x, mechanism, p_miss):
#     x_simulated = simulate_nan(np.asarray(x), p_miss, mechanism)
#
#     mask = x_simulated["mask"]
#     x_miss = x_simulated["X_incomp"]
#
#     return pd.DataFrame(x), pd.DataFrame(x_miss), pd.DataFrame(mask)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    # Initialize other args
    dataname = args.dataname
    device = args.device
    models_dir = f'./saved_models/{dataname}/'

    # train = pd.read_csv(f"./helpers/datasets/{dataname}/train.csv")  # source train data file
    # test = pd.read_csv(f"./helpers/datasets/{dataname}/test.csv")  # source test data file

    # prepare the data: train and test data (true), test data (with missing values), and the mask
    prepper = Preprocessor(dataname)
    train_X = prepper.encodeDf('Ordinal', prepper.df_train)  # train_X is a numpy array
    test_X = prepper.encodeDf('Ordinal', prepper.df_test)  # test_X is a numpy array
    num_numeric = prepper.numerical_indices_np_end  # index of the last numeric column

    mean_X, std_X = (
        np.mean(train_X, axis=0), np.std(train_X, axis=0)
    )

    in_dim = train_X.shape[1]
    X = (train_X - mean_X) / std_X
    imputer = ReMasker(
        d_in=in_dim,
        device=device, 
        num_numeric=num_numeric,
        
    )
    remasker = imputer.fit(torch.as_tensor(X.copy(), dtype=torch.float32))

    print(remasker.model)

    # Save to a file

    buff = save(imputer)  # get the model as bytes

    # exit()
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"remasker.pkl")

    # after imputer.fit(...)
    with open(path, "wb") as f:
        f.write(buff)  # .model() returns bytes
    # with open(f"{models_dir}/hyperimpute_{plugin}_final.pkl", "wb") as f:
    #     f.write(imputer.save())  # imputer.save() returns a bytes object
    print(f"Saved imputer model: {f.name}")

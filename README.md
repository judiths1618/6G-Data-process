<!-- How to run -->
python -m venv .venv && source .venv/bin/activate
pip install "apache-beam[gcp]" python-dateutil

> **Note**
> Some execution environments (including the one used for automated
> verification) block outbound package downloads behind a proxy. If you
> encounter repeated `ProxyError` messages when attempting to install
> `apache-beam`, you will need to perform the installation from a network
> location with PyPI access or provide the wheel files via an internal
> mirror before running the pipeline.

# Example 1: the EUR/6907619 files
python dq_local_beam.py \
  --inputs "6GDALI_Datasets/EUR/6907619/*.csv" \
  --good_out "out/good/eur6907619" \
  --bad_out "out/bad/eur6907619" \
  --dq_out "out/dq/eur6907619"

# Example 2: all KUL “user_*.csv” files under antennas_as_features
python dq_local_beam.py \
  --inputs "KUL/nomadic_dataset_ULA_static/antennas_as_features/user_*.csv" \
  --good_out "out/good/kul_ant" \
  --bad_out "out/bad/kul_ant" \
  --dq_out "out/dq/kul_ant"

# Example 3: recursively hit other feature folders
python dq_local_beam.py \
  --inputs "KUL/nomadic_dataset_ULA_static/**/*.csv" \
  --good_out "out/good/kul_all" \
  --bad_out "out/bad/kul_all" \
  --dq_out "out/dq/kul_all"

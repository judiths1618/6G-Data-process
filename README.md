<!-- How to run -->
python -m venv .venv && source .venv/bin/activate
pip install "apache-beam[gcp]" python-dateutil

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

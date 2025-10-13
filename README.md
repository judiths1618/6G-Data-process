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

If installing `apache-beam` is not feasible you can switch the script to a
pure-Python fallback engine:

```
python dq_local_beam.py \
  --input_pattern "6GDALI_Datasets/EUR/6907619/*.csv" \
  --config eur_dq_rules.yaml \
  --good_out out/good/eur6907619 \
  --bad_out out/bad/eur6907619 \
  --dq_out out/dq/eur6907619 \
  --engine sequential
```

The sequential engine produces the same outputs (good rows, rejected rows,
and data-quality profiles) without requiring any Beam dependencies, although
it runs in a single process so very large datasets may take longer to finish.

## Benchmarking the engines

To compare the runtime of the sequential fallback against the Apache Beam
implementation, use the `benchmarks/compare_engines.py` helper. The script
executes the pipeline for each requested engine, measures the wall-clock
duration, and surfaces key metrics from the generated quality report.

```
python benchmarks/compare_engines.py \
  --input_pattern "6GDALI_Datasets/EUR/6907619/*.csv" \
  --config dq_rules.yaml \
  --output_root out/benchmarks/eur6907619
```

When `apache-beam` is unavailable, the Beam run is skipped automatically and
only the sequential results are reported.

```
python dq_local_beam.py \  --input_pattern "6GDALI_Datasets/KUL/nomadic_dataset_ULA_static/antennas_as_features/user_*.csv" --config dq_rules.yaml \  --good_out "out/good/kul_antennas_as_features" \  --bad_out "out/bad/kul_antennas_as_features" \  --dq_out "out/dq/kul_antennas_as_features"

python dq_local_beam.py \  --input_pattern "6GDALI_Datasets/KUL/nomadic_dataset_ULA_static/csi_as_features/user_*.csv" --config dq_rules.yaml \  --good_out "out/good/kul_csi_as_features" \  --bad_out "out/bad/kul_csi_as_features" \  --dq_out "out/dq/kul_csi_as_features"

python dq_local_beam.py \  --input_pattern "6GDALI_Datasets/KUL/nomadic_dataset_ULA_static/subcarriers_as_features_complex/user_*.csv" --config dq_rules.yaml \  --good_out "out/good/kul_subcarriers_as_features_complex" \  --bad_out "out/bad/kul_subcarriers_as_features_complex" \  --dq_out "out/dq/kul_subcarriers_as_features_complex"

python dq_local_beam.py \  --input_pattern "6GDALI_Datasets/KUL/nomadic_dataset_ULA_static/subcarriers_as_features_real/user_*.csv" --config dq_rules.yaml \  --good_out "out/good/kul_subcarriers_as_features_real" \  --bad_out "out/bad/kul_subcarriers_as_features_real" \  --dq_out "out/dq/kul_subcarriers_as_features_real"


```
/Users/yuandouwang/Documents/projects/6G-Data-process/6GDALI_Datasets/KUL/nomadic_dataset_ULA_static/subcarriers_as_features_real

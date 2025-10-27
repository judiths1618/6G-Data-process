# Methods

This directory hosts lightweight data processing utilities that can run either as
standalone Python helpers or inside Apache Beam pipelines. The modules are kept
free of heavy dependencies so they integrate cleanly with existing workflows.

## Available modules

- **`data_augmentation_beam.py`** – helpers for aligning multiple time-series
  CSV tables, prefixing their feature columns, and optionally enriching rows
  with engineered timestamp attributes.
- **`evaluation_pipeline.py`** – utilities that fit simple linear models to
  compare baseline and augmented datasets, returning RMSE-based improvement
  metrics.
- **`wavestitch_imputation.py`** – wraps the WaveStitch diffusion imputer for
  6G-ready datasets, providing ``train`` and ``impute`` CLI commands that
  handle preprocessing, checkpointing, and conditional sampling.

## Using the helpers

All modules are plain Python files. Import them into your scripts, notebooks,
or Beam transforms to drive augmentation and evaluation workflows. For
reproducible environments, install project dependencies first:

```bash
pip install -r requirements.txt
```

### Example

```python
from methods.data_augmentation_beam import augment_with_time
from methods.evaluation_pipeline import evaluate_time_series_augmentation

augmented = augment_with_time(["data/a.csv", "data/b.csv"])
metrics = evaluate_time_series_augmentation(
    ["data/a.csv", "data/b.csv"],
    target_feature="metrics_target",
)
```

### Working with the EUR dataset

The helpers also accept directories, so you can point them at a dataset folder
like the built-in EUR bundle. All ``*.csv`` files inside the directory are
loaded and aligned automatically:

```python
from pathlib import Path

from methods.evaluation_pipeline import evaluate_time_series_augmentation

eur_dataset = Path("6GDALI_Datasets/EUR/6907619")
results = evaluate_time_series_augmentation(
    [eur_dataset],
    target_feature="amf-performance_mean",  # choose the metric you want to predict
    join="outer",  # include rows even if some tables miss the timestamp
    on_duplicate="last",  # keep the most recent measurement when timestamps repeat
    regularization=1e-6,  # small ridge term for numerical stability
)
print(results)
```

### Imputing missing values with WaveStitch

To train and apply the diffusion-based imputer on one of the bundled 6G
datasets:

```bash
python -m methods.wavestitch_imputation train 6GDALI_Datasets/EUR/6907619 \
    --output saved_models/eur_wavestitch.pt --window-size 48 --epochs 100 \
    --device mps  # use "cuda" on NVIDIA GPUs or keep the default "auto"

python -m methods.wavestitch_imputation impute 6GDALI_Datasets/EUR/6907619 \
    --model saved_models/eur_wavestitch.pt --output out/eur_imputed.csv
```

For programmatic pipelines you can use the high level ``WaveStitchImputer``
class:

```python
from methods.wavestitch_imputation import WaveStitchConfig, WaveStitchImputer

imputer = WaveStitchImputer(WaveStitchConfig(window_size=48))
imputer.fit(["6GDALI_Datasets/EUR/6907619"], min_valid_ratio=0.1)
imputed = imputer.transform(["6GDALI_Datasets/EUR/6907619"])
```

Refer to the unit tests under `tests/` for additional usage examples or adapt
the helper functions inside these modules to your pipeline needs.

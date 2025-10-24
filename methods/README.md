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

Refer to the unit tests under `tests/` for additional usage examples or adapt
the helper functions inside these modules to your pipeline needs.

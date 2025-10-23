"""Evaluation helpers for measuring augmentation impact on model training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Mapping, Sequence

from .data_augmentation_beam import augment_with_time, augment_without_time


NumericRow = Mapping[str, object]


def _to_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError("NaN or infinite values are not supported")
        return float(value)
    text = str(value).strip()
    if text == "":
        raise ValueError("Empty value cannot be converted to float")
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Value {value!r} is not numeric") from exc
    if math.isnan(number) or math.isinf(number):
        raise ValueError("NaN or infinite values are not supported")
    return number


def _detect_numeric_features(rows: Sequence[NumericRow], target_feature: str) -> List[str]:
    numeric_candidates: dict[str, bool] = {}
    numeric_values: dict[str, List[float]] = {}
    for row in rows:
        for column, value in row.items():
            if column == target_feature:
                continue
            if column in numeric_candidates and not numeric_candidates[column]:
                continue
            try:
                number = _to_float(value)
            except ValueError:
                numeric_candidates[column] = False
            else:
                numeric_candidates.setdefault(column, True)
                numeric_values.setdefault(column, []).append(number)

    numeric_columns: List[str] = []
    for column, is_numeric in numeric_candidates.items():
        if not is_numeric:
            continue
        values = numeric_values.get(column, [])
        if len(values) < 2:
            continue
        if max(values) - min(values) < 1e-12:
            continue
        numeric_columns.append(column)

    return sorted(numeric_columns)


@dataclass(frozen=True)
class _PreparedDataset:
    features: List[str]
    design_matrix: List[List[float]]
    targets: List[float]


def _prepare_dataset(
    rows: Sequence[NumericRow],
    *,
    target_feature: str,
    feature_columns: Sequence[str] | None,
) -> _PreparedDataset:
    if not rows:
        raise ValueError("Dataset must contain at least one row")
    if target_feature not in rows[0]:
        raise ValueError(f"Target feature '{target_feature}' not found in dataset")

    features = list(feature_columns) if feature_columns else _detect_numeric_features(rows, target_feature)
    if not features:
        raise ValueError("No numeric feature columns detected for training")

    matrix: list[list[float]] = []
    targets: list[float] = []

    for row in rows:
        if target_feature not in row:
            raise ValueError(f"Row is missing target feature '{target_feature}'")
        target = _to_float(row[target_feature])
        feature_values: list[float] = []
        for column in features:
            value = row.get(column, 0.0)
            feature_values.append(_to_float(value))
        matrix.append(feature_values)
        targets.append(target)

    if not matrix or not matrix[0]:
        raise ValueError("Design matrix must be two-dimensional")
    if len(matrix) < 2:
        raise ValueError("Dataset must contain at least two rows for training")

    return _PreparedDataset(features=features, design_matrix=matrix, targets=targets)


def _split_train_test(matrix: List[List[float]], target: List[float], test_ratio: float) -> tuple[List[List[float]], List[float], List[List[float]], List[float]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")
    n_samples = len(matrix)
    if n_samples < 3:
        raise ValueError("At least three samples are required for a train/test split")
    n_test = max(1, int(round(n_samples * test_ratio)))
    if n_test >= n_samples:
        n_test = n_samples - 1
    split_index = n_samples - n_test
    return (
        matrix[:split_index],
        target[:split_index],
        matrix[split_index:],
        target[split_index:],
    )


def _fit_linear_regression(design_matrix: List[List[float]], targets: List[float]) -> List[float]:
    augmented = [[1.0] + row for row in design_matrix]
    size = len(augmented[0])
    xtx = [[0.0 for _ in range(size)] for _ in range(size)]
    xty = [0.0 for _ in range(size)]

    for row, target in zip(augmented, targets):
        for i in range(size):
            xty[i] += row[i] * target
            for j in range(size):
                xtx[i][j] += row[i] * row[j]

    return _solve_linear_system(xtx, xty)


def _predict(design_matrix: List[List[float]], coefficients: List[float]) -> List[float]:
    augmented = [[1.0] + row for row in design_matrix]
    predictions: List[float] = []
    for row in augmented:
        pred = sum(val * coef for val, coef in zip(row, coefficients))
        predictions.append(pred)
    return predictions


def _rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("Predictions and targets must have the same length")
    errors = [(a - b) ** 2 for a, b in zip(y_true, y_pred)]
    return math.sqrt(sum(errors) / len(errors))


def _solve_linear_system(matrix: List[List[float]], vector: List[float]) -> List[float]:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]

    for pivot in range(size):
        pivot_row = max(range(pivot, size), key=lambda r: abs(augmented[r][pivot]))
        if abs(augmented[pivot_row][pivot]) < 1e-12:
            raise ValueError("Singular matrix encountered during regression fitting")
        if pivot_row != pivot:
            augmented[pivot], augmented[pivot_row] = augmented[pivot_row], augmented[pivot]

        pivot_val = augmented[pivot][pivot]
        augmented[pivot] = [value / pivot_val for value in augmented[pivot]]

        for row_index in range(size):
            if row_index == pivot:
                continue
            factor = augmented[row_index][pivot]
            augmented[row_index] = [
                current - factor * pivot_value
                for current, pivot_value in zip(augmented[row_index], augmented[pivot])
            ]

    return [row[-1] for row in augmented]


def evaluate_model_improvement(
    baseline_rows: Sequence[NumericRow],
    augmented_rows: Sequence[NumericRow],
    *,
    target_feature: str,
    baseline_features: Sequence[str] | None = None,
    augmented_features: Sequence[str] | None = None,
    test_ratio: float = 0.2,
    metric: str = "rmse",
) -> Mapping[str, float]:
    """Train simple linear models and compare their performance."""

    if metric != "rmse":
        raise ValueError("Only 'rmse' metric is currently supported")

    baseline = _prepare_dataset(
        baseline_rows,
        target_feature=target_feature,
        feature_columns=baseline_features,
    )
    augmented = _prepare_dataset(
        augmented_rows,
        target_feature=target_feature,
        feature_columns=augmented_features,
    )

    (
        X_train_base,
        y_train_base,
        X_test_base,
        y_test_base,
    ) = _split_train_test(baseline.design_matrix, baseline.targets, test_ratio)
    (
        X_train_aug,
        y_train_aug,
        X_test_aug,
        y_test_aug,
    ) = _split_train_test(augmented.design_matrix, augmented.targets, test_ratio)

    coef_base = _fit_linear_regression(X_train_base, y_train_base)
    coef_aug = _fit_linear_regression(X_train_aug, y_train_aug)

    predictions_base = _predict(X_test_base, coef_base)
    predictions_aug = _predict(X_test_aug, coef_aug)

    baseline_rmse = _rmse(y_test_base, predictions_base)
    augmented_rmse = _rmse(y_test_aug, predictions_aug)

    return {
        "metric": metric,
        "baseline": baseline_rmse,
        "augmented": augmented_rmse,
        "improvement": baseline_rmse - augmented_rmse,
    }


def evaluate_time_series_augmentation(
    tables: Sequence[str],
    *,
    target_feature: str,
    time_column: str = "time",
    parse_dates: bool = True,
    time_format: str | None = None,
    join: str = "inner",
    test_ratio: float = 0.2,
    metric: str = "rmse",
) -> Mapping[str, float]:
    """Evaluate the benefit of temporal feature engineering for time-series tables."""

    baseline_rows = augment_without_time(
        tables,
        time_column=time_column,
        parse_dates=parse_dates,
        time_format=time_format,
        join=join,
    )
    augmented_rows = augment_with_time(
        tables,
        time_column=time_column,
        parse_dates=parse_dates,
        time_format=time_format,
        join=join,
    )

    return evaluate_model_improvement(
        baseline_rows,
        augmented_rows,
        target_feature=target_feature,
        test_ratio=test_ratio,
        metric=metric,
    )


from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from methods.wavestitch_imputation import PreparedDataset, _select_device, prepare_dataset


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


@pytest.fixture
def toy_dataset(tmp_path: Path) -> list[Path]:
    root = tmp_path / "eur"
    root.mkdir()
    _write_csv(
        root / "amf-performance.csv",
        [
            {"time": 1, "ram_limit": "1024M", "latency": 12.0},
            {"time": 2, "ram_limit": "2048M", "latency": 18.0},
            {"time": 3, "ram_limit": "2048M", "latency": 21.0},
            {"time": 4, "ram_limit": "2048M", "latency": 25.0},
        ],
    )
    _write_csv(
        root / "web-performance.csv",
        [
            {"time": 1, "cpu": 0.5, "throughput": 2000},
            {"time": 2, "cpu": 0.7, "throughput": np.nan},
            {"time": 3, "cpu": 0.9, "throughput": 2300},
            {"time": 4, "cpu": 0.4, "throughput": 2100},
        ],
    )
    return [root]


def test_prepare_dataset_generates_expected_tensors(toy_dataset: list[Path]) -> None:
    prepared = prepare_dataset(toy_dataset, window_size=3, stride=1)

    assert isinstance(prepared, PreparedDataset)
    assert prepared.train_windows.shape[1] == len(prepared.feature_columns)
    assert prepared.full_windows.shape[1] == len(prepared.feature_columns)
    assert prepared.observed_mask_windows.shape[1] == len(prepared.non_hier_columns)
    # Two windows can be created from four rows with a window size of three.
    assert prepared.train_windows.shape[0] == 2

    # The RAM limit column should be coerced to numeric and treated as non-hierarchical.
    assert any(col.endswith("ram_limit") for col in prepared.non_hier_columns)

    # Re-create the dataset using the saved feature list to ensure deterministic behaviour.
    prepared_repeat = prepare_dataset(
        toy_dataset,
        window_size=3,
        stride=1,
        feature_columns=prepared.non_hier_columns,
    )
    assert prepared_repeat.non_hier_columns == prepared.non_hier_columns


def test_prepare_dataset_rejects_short_sequences(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "eur"
    dataset_dir.mkdir()
    _write_csv(dataset_dir / "single.csv", [{"time": 1, "metric": 10.0}])

    with pytest.raises(ValueError):
        prepare_dataset([dataset_dir], window_size=3)


def test_select_device_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        monkeypatch.setattr(mps, "is_available", lambda: False, raising=False)
        monkeypatch.setattr(mps, "is_built", lambda: False, raising=False)

    device = _select_device()
    assert device.type == "cpu"


def test_select_device_rejects_missing_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError):
        _select_device("cuda")

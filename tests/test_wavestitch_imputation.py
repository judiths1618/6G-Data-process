from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

import methods.wavestitch_imputation as wavestitch_imputation
from methods.wavestitch_imputation import (
    PreparedDataset,
    WaveStitchConfig,
    WaveStitchImputer,
    _select_device,
    prepare_dataset,
)
import pandas.testing as pdt


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


def test_wavestitch_imputer_roundtrip(monkeypatch: pytest.MonkeyPatch, toy_dataset: list[Path], tmp_path: Path) -> None:
    class DummyImputer(torch.nn.Module):
        def __init__(self, in_channels: int, res_channels: int, skip_channels: int, out_channels: int, **_: object) -> None:
            super().__init__()
            self.out_channels = out_channels
            self.register_buffer("_bias", torch.zeros(out_channels))

        def forward(self, inputs: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            batch, _, window = inputs.shape
            return torch.zeros(batch, self.out_channels, window, device=inputs.device)

    def fake_train_model(prepared: PreparedDataset, config: WaveStitchConfig, device: torch.device) -> DummyImputer:
        model = wavestitch_imputation.SSSDS4Imputer(
            in_channels=len(prepared.feature_columns),
            res_channels=config.res_channels,
            skip_channels=config.skip_channels,
            out_channels=len(prepared.non_hier_columns),
            num_res_layers=config.num_res_layers,
            diffusion_step_embed_dim_in=config.diff_step_embed_in,
            diffusion_step_embed_dim_mid=config.diff_step_embed_mid,
            diffusion_step_embed_dim_out=config.diff_step_embed_out,
            s4_lmax=config.s4_lmax,
            s4_d_state=config.s4_dstate,
            s4_dropout=config.s4_dropout,
            s4_bidirectional=config.s4_bidirectional,
            s4_layernorm=config.s4_layernorm,
        )
        return model.to(device)

    def fake_sample_windows(
        model: torch.nn.Module, prepared: PreparedDataset, config: WaveStitchConfig, device: torch.device
    ) -> np.ndarray:
        return prepared.full_windows.detach().cpu().numpy()

    monkeypatch.setattr(wavestitch_imputation, "SSSDS4Imputer", DummyImputer)
    monkeypatch.setattr(wavestitch_imputation, "_train_model", fake_train_model)
    monkeypatch.setattr(wavestitch_imputation, "_sample_windows", fake_sample_windows)

    config = WaveStitchConfig(window_size=3, stride=1, epochs=1, batch_size=2, timesteps=4)
    imputer = WaveStitchImputer(config=config, device="cpu")
    imputed = imputer.fit_transform(toy_dataset, min_valid_ratio=0.25)

    assert list(imputed.columns)[0] == "time"
    assert imputed.shape[0] == 4
    assert imputer.non_hier_columns is not None
    assert any(col.endswith("throughput") for col in imputer.non_hier_columns)

    checkpoint = tmp_path / "model.pt"
    imputer.save(checkpoint)
    reloaded = WaveStitchImputer.load(checkpoint, device="cpu")
    re_imputed = reloaded.transform(toy_dataset, min_valid_ratio=0.25)

    pdt.assert_frame_equal(imputed, re_imputed)


def test_transform_reuses_trained_min_valid_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyModel(torch.nn.Module):
        def to(self, *_: object, **__: object) -> "DummyModel":
            return self

        def eval(self) -> "DummyModel":
            return self

    imputer = WaveStitchImputer(config=WaveStitchConfig(window_size=1, stride=1), device="cpu")
    imputer.model = DummyModel()
    imputer.scaler_state = {
        "mean": [0.0],
        "scale": [1.0],
        "var": [1.0],
        "n_features_in": 1,
        "n_samples_seen": 1.0,
        "feature_names": ["metric"],
    }
    imputer.feature_columns = ["metric"]
    imputer.hierarchical_columns = []
    imputer.non_hier_columns = ["metric"]
    imputer.time_column = "time"
    imputer._min_valid_ratio = 0.42

    original = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=1, freq="H"), "metric": [1.0]})
    prepared = PreparedDataset(
        original=original,
        feature_columns=["metric"],
        hierarchical_columns=[],
        non_hier_columns=["metric"],
        train_windows=torch.zeros((1, 1, 1)),
        full_windows=torch.zeros((1, 1, 1)),
        observed_mask_windows=torch.ones((1, 1, 1)),
        scaler_state=imputer.scaler_state,
        window_size=1,
        stride=1,
        index=original.index,
        time_column="time",
        feature_matrix=np.zeros((1, 1), dtype=np.float32),
        min_valid_ratio=0.42,
    )

    captured: dict[str, float | list[str]] = {}

    def fake_prepare_dataset(
        sources: list[Path],
        *,
        time_column: str,
        min_valid_ratio: float,
        window_size: int,
        stride: int,
        feature_columns: list[str] | None,
    ) -> PreparedDataset:
        captured["min_valid_ratio"] = min_valid_ratio
        captured["feature_columns"] = feature_columns or []
        return prepared

    def fake_impute_from_prepared(
        model: torch.nn.Module,
        prepared_dataset: PreparedDataset,
        config: WaveStitchConfig,
        device: torch.device,
        scaler: object,
    ) -> pd.DataFrame:
        return prepared_dataset.original[["time"] + prepared_dataset.non_hier_columns]

    monkeypatch.setattr(wavestitch_imputation, "prepare_dataset", fake_prepare_dataset)
    monkeypatch.setattr(wavestitch_imputation, "_impute_from_prepared", fake_impute_from_prepared)

    result = imputer.transform([Path("dummy")])

    assert list(result.columns) == ["time", "metric"]
    assert captured["min_valid_ratio"] == pytest.approx(0.42)
    assert captured["feature_columns"] == ["metric"]

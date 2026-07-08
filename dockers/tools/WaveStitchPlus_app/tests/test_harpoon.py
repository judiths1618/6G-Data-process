from __future__ import annotations

import argparse
import subprocess
import sys
import types

import pytest

torch = pytest.importorskip("torch")

import run_imputation_harpoon as runner

training_utils = types.ModuleType("helper.training_utils")
training_utils.MyDataset = object
training_utils.fetchModel = lambda *args, **kwargs: None
training_utils.fetchDiffusionConfig = lambda *args, **kwargs: None
sys.modules.setdefault("helper.training_utils", training_utils)

import synthesis_harpoon as harpoon


def test_direct_x0_correction_projects_only_missing_targets():
    synth = harpoon.HarpoonDDIMSynthesizer.__new__(harpoon.HarpoonDDIMSynthesizer)
    synth.target_indices = torch.tensor([0, 1])
    synth.lb = torch.tensor([0.0, -1.0])
    synth.ub = torch.tensor([1.0, 2.0])
    synth.bound_lambda = 0.3
    synth.bound_power = 2.0
    synth.guidance_scale = 0.1
    synth.clip_bound = 4.0
    synth.project_bounds = True
    synth.prior_lambda = 0.0
    synth.smooth_lambda = 0.0
    synth.monotone_lambda = 0.0
    synth.monotone_groups = []
    synth._prior_batch = None

    x0 = torch.tensor([[[2.5, -3.0, 99.0], [0.5, 0.5, 88.0]]])
    synth_mask = torch.tensor([[[True, True], [False, True]]])
    obs_mask = ~synth_mask
    out = synth._correct_x0(x0, obs_mask.float(), synth_mask.float(), x0)

    # Missing target cells are projected into the HARPOON feasible interval.
    assert out[0, 0, 0].item() == 1.0
    assert out[0, 0, 1].item() == -1.0
    assert out[0, 1, 1].item() == 0.5
    # Observed target cells and non-target columns are untouched.
    assert out[0, 1, 0].item() == 0.5
    assert out[0, 0, 2].item() == 99.0


def test_prior_guidance_pulls_missing_cells_toward_prior():
    synth = harpoon.HarpoonDDIMSynthesizer.__new__(harpoon.HarpoonDDIMSynthesizer)
    synth.target_indices = torch.tensor([0])
    synth.lb = torch.tensor([-10.0])
    synth.ub = torch.tensor([10.0])
    synth.bound_lambda = 0.0
    synth.bound_power = 2.0
    synth.guidance_scale = 0.5
    synth.clip_bound = 10.0
    synth.project_bounds = False
    synth.prior_lambda = 1.0
    synth.smooth_lambda = 0.0
    synth.monotone_lambda = 0.0
    synth.monotone_groups = []
    synth._prior_batch = torch.tensor([[[1.0], [1.0]]])

    x0 = torch.tensor([[[5.0], [5.0]]])
    synth_mask = torch.tensor([[[True], [False]]])
    obs_mask = ~synth_mask
    out = synth._correct_x0(x0, obs_mask.float(), synth_mask.float(), x0)

    assert out[0, 0, 0].item() < x0[0, 0, 0].item()
    assert out[0, 1, 0].item() == x0[0, 1, 0].item()


def test_runner_passes_v1_model_tag_and_projection_flag(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    out_csv = tmp_path / "generated" / "harpoon.csv"

    calls = []

    def fake_run(cmd, env=None, cwd=None):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    args = argparse.Namespace(
        repaint_rounds=1,
        guidance_scale=0.1,
        ddim_steps=5,
        bound_lambda=0.3,
        bound_power=2.0,
        project_bounds="True",
        prior_lambda=0.25,
        prior_method="nearest",
        smooth_lambda=0.02,
        monotone_lambda=0.05,
        pos_eps=1e-6,
        auto_ub_q=0.99,
        auto_ub_pad=0.05,
        hard_project_positive=False,
        model_tag="v1",
    )

    runner._run_synthesis(prepared, out_csv, args, env={})
    cmd = calls[0]

    assert "-model_tag" in cmd
    assert cmd[cmd.index("-model_tag") + 1] == "v1"
    assert "-project_bounds" in cmd
    assert cmd[cmd.index("-project_bounds") + 1] == "True"
    assert "-prior_lambda" in cmd
    assert cmd[cmd.index("-prior_lambda") + 1] == "0.25"
    assert "-prior_method" in cmd
    assert cmd[cmd.index("-prior_method") + 1] == "nearest"

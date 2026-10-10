'''Tests for ROOT input, dataset splits, checkpointing, and command-line entry points.'''

import argparse
import os
from pathlib import Path
import subprocess
import sys

import awkward as ak
import numpy as np
import pytest
import torch
import uproot

from djt import baseline, pair_bias, pair_bias_base


@pytest.mark.parametrize("module", ["baseline", "pair_bias", "evaluate_baseline", "evaluate_uds", "evaluate_soft_fusion"])
def test_command_line_help(module):
    repo = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", f"djt.{module}", "--help"],
        cwd=repo, env=environment, text=True, capture_output=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--root-files" in completed.stdout
    assert "--seed" in completed.stdout
    if module == "pair_bias":
        assert "--pair-feature-set" in completed.stdout
        assert "no_kaon" in completed.stdout


def test_root_loading_preserves_order_and_padding(tmp_path):
    rows = 12
    branches = {name: np.arange(rows, dtype=np.float32) for name in baseline.GLOBAL_BRANCHES}
    for names in (baseline.CPF_BRANCHES, baseline.NPF_BRANCHES, baseline.SV_BRANCHES, baseline.V0_BRANCHES):
        for name in names:
            branches[name] = ak.Array([[float(i + 1)] * (i % 3 + 1) for i in range(rows)])
    for column, name in enumerate(baseline.TRUTH_BRANCHES):
        branches[name] = (np.arange(rows) % 6 == column).astype(np.float32)
    path = tmp_path / "events.root"
    with uproot.recreate(path) as output:
        output.mktree("deepntuplizer/tree", branches)
    arrays = baseline.load_data([str(path)], "deepntuplizer/tree", limit=7)
    expected_shapes = [(7, 12), (7, 25, 19), (7, 25, 11), (7, 4, 11), (7, 4, 11), (7, 6)]
    assert [values.shape for values in arrays] == expected_shapes
    assert all(values.dtype == np.float32 for values in arrays)
    np.testing.assert_array_equal(arrays[0][:, 0], np.arange(7))
    np.testing.assert_array_equal(arrays[-1].argmax(1), np.arange(7) % 6)
    np.testing.assert_array_equal(arrays[1][0, 1:], 0)
    np.testing.assert_array_equal(arrays[1][2, :3, 0], 3)


def test_split_and_uds_selection_are_consistent():
    truth = np.eye(6, dtype=np.float32)[np.arange(600) % 6]
    splits = baseline.make_split_indices(len(truth), .15, .15, 1123)
    uds_splits = pair_bias.make_split_indices(len(truth), .15, .15, 1123)
    assert [len(indices) for indices in splits] == [420, 90, 90]
    np.testing.assert_array_equal(np.sort(np.concatenate(splits)), np.arange(600))
    for indices, uds_indices in zip(splits, uds_splits):
        np.testing.assert_array_equal(indices, uds_indices)
        selected = pair_bias.filter_uds_indices(indices, truth)
        expected = indices[np.isin(truth[indices].argmax(1), [2, 3, 4])]
        np.testing.assert_array_equal(selected, expected)
        np.testing.assert_array_equal(pair_bias.make_uds_truth(truth)[selected].sum(1), 1)


@pytest.mark.parametrize("module", [baseline, pair_bias_base])
def test_optimizer_checkpoint_roundtrip(tmp_path, module):
    torch.manual_seed(1123)
    model = torch.nn.Linear(3, 2)
    optimizer = module.Ranger(model.parameters(), lr=.005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer.optimizer, milestones=[2], gamma=.1)
    inputs = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 0, 1])
    for _ in range(6):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(inputs), labels).backward()
        optimizer.step()
    scheduler.step()
    args = argparse.Namespace(pair_hidden_dim=32)
    path = tmp_path / "checkpoint.pth"
    module.save_checkpoint(path, model, optimizer, scheduler, 1, .5, args)
    restored = torch.nn.Linear(3, 2)
    restored_optimizer = module.Ranger(restored.parameters(), lr=.005)
    restored_scheduler = torch.optim.lr_scheduler.MultiStepLR(restored_optimizer.optimizer, milestones=[2], gamma=.1)
    epoch, best_loss = module.load_checkpoint(path, restored, restored_optimizer, restored_scheduler, "cpu")
    assert (epoch, best_loss) == (2, .5)
    torch.testing.assert_close(restored(inputs), model(inputs))
    assert restored_optimizer.step_counter == optimizer.step_counter
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    for current_model, current_optimizer in [(model, optimizer), (restored, restored_optimizer)]:
        current_optimizer.zero_grad()
        torch.nn.functional.cross_entropy(current_model(inputs), labels).backward()
        current_optimizer.step()
    torch.testing.assert_close(restored(inputs), model(inputs))

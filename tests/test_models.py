'''Tests for the baseline, pair-bias specialist, and probability fusion.'''

import numpy as np
import pytest
import torch

from djt import baseline, pair_bias, pair_bias_base
from djt.soft_fusion import soft_fuse_light_flavours


@pytest.fixture
def jet_tensors():
    torch.manual_seed(1123)
    inputs = [
        torch.randn(4, len(baseline.GLOBAL_BRANCHES)),
        torch.randn(4, 5, len(baseline.CPF_BRANCHES)),
        torch.randn(4, 4, len(baseline.NPF_BRANCHES)),
        torch.randn(4, 2, len(baseline.SV_BRANCHES)),
        torch.randn(4, 2, len(baseline.V0_BRANCHES)),
    ]
    for values in inputs[1:]:
        values[:, -1] = 0
    return inputs


@pytest.mark.parametrize("model_type,n_classes", [("baseline", 6), ("no_kaon", 3)])
def test_model_forward_backward_with_padding(jet_tensors, model_type, n_classes):
    if model_type == "baseline":
        model = baseline.DeepJetTransformerV0()
    else:
        model = pair_bias.UDSSpecialistPairBiasTransformer(pair_feature_set="no_kaon")
    logits = model(*jet_tensors)
    assert logits.shape == (4, n_classes)
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(logits, torch.arange(4) % n_classes)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


@pytest.mark.parametrize("model_type", ["baseline", "no_kaon"])
def test_inference_ignores_additional_zero_padding(jet_tensors, model_type):
    if model_type == "baseline":
        model = baseline.DeepJetTransformerV0().eval()
    else:
        model = pair_bias.UDSSpecialistPairBiasTransformer(pair_feature_set="no_kaon").eval()
    padded = [jet_tensors[0]] + [
        torch.cat([values, torch.zeros(values.shape[0], 2, values.shape[2])], dim=1)
        for values in jet_tensors[1:]
    ]
    with torch.no_grad():
        expected = model(*jet_tensors)
        actual = model(*padded)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_no_kaon_features_exclude_pair_pid(jet_tensors):
    meta = pair_bias_base.make_token_metadata(*jet_tensors[1:])
    expected = pair_bias.build_pair_features(meta, "no_kaon")
    changed = meta.clone()
    changed[..., 4] += 10
    actual = pair_bias.build_pair_features(changed, "no_kaon")
    assert expected.shape == (4, 13, 13, 17)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    names = pair_bias.get_included_pair_feature_names("no_kaon")
    assert not any("kaon" in name for name in names)
    assert "RPj_charged_is_Kaon_smearedUniform090" in baseline.CPF_BRANCHES


def test_pair_bias_masks_padded_pairs(jet_tensors):
    meta = pair_bias_base.make_token_metadata(*jet_tensors[1:])
    mask = torch.cat([values.abs().sum(-1) == 0 for values in jet_tensors[1:]], dim=1)
    builder = pair_bias.PairBiasBuilder(pair_feature_set="no_kaon", nhead=2)
    with torch.no_grad():
        builder.net[-1].bias.fill_(1)
    bias = builder(meta, mask).reshape(4, 2, 13, 13)
    invalid = (mask[:, None, :, None] | mask[:, None, None, :]).expand_as(bias)
    assert torch.count_nonzero(bias[invalid]) == 0
    torch.testing.assert_close(bias[~invalid], torch.ones_like(bias[~invalid]))


def test_soft_fusion_preserves_probability_mass_and_class_order():
    baseline_scores = np.array([[.1, .2, .15, .25, .2, .1], [.3, .2, .1, .1, .1, .2]])
    uds_scores = np.array([[.2, .3, .5], [.5, .3, .2]])
    before = baseline_scores.copy()
    fused = soft_fuse_light_flavours(baseline_scores, uds_scores)
    np.testing.assert_array_equal(baseline_scores, before)
    np.testing.assert_array_equal(fused[:, [0, 1, 5]], before[:, [0, 1, 5]])
    np.testing.assert_allclose(fused[:, 2:5], before[:, 2:5].sum(1, keepdims=True) * uds_scores)
    np.testing.assert_allclose(fused.sum(1), 1)
    order = [5, 4, 3, 2, 1, 0]
    uds_order = [2, 0, 1]
    reordered = soft_fuse_light_flavours(
        baseline_scores[:, order], uds_scores[:, uds_order],
        [baseline.CLASS_NAMES[i] for i in order],
        [pair_bias.CLASS_NAMES[i] for i in uds_order],
    )
    np.testing.assert_allclose(reordered, fused[:, order])

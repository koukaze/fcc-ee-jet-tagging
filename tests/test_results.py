'''Regression checks for the saved FCC-ee prediction analysis.'''

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('reproduce_results', ROOT / 'scripts' / 'reproduce_results.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


@pytest.fixture(scope='module')
def saved_results():
    return analysis.load_results(ROOT / 'results' / 'final')


@pytest.fixture(scope='module')
def saved_tables(saved_results):
    return analysis.comparison_tables(saved_results)


def test_saved_sample_and_fusion(saved_results):
    np.testing.assert_array_equal(
        np.bincount(saved_results['labels']), [29945, 30003, 29775, 30083, 30222, 29972]
    )
    np.testing.assert_array_equal(
        saved_results['fused_scores'][:, [0, 1, 5]],
        saved_results['baseline_scores'][:, [0, 1, 5]],
    )


def test_accuracy_matches_recorded_results(saved_tables):
    accuracy, _ = saved_tables
    expected = np.array([
        [61.28055555555556, 61.35295834168149, 91.06695608615796, 80.18531480185315,
         64.28211586901763, 38.06468769737061, 22.74502018397194, 71.77365541171761],
        [62.47611111111111, 62.52742855572868, 91.07363499749542, 80.12198780121987,
         60.37951301427372, 35.917295482498425, 36.60247501819867, 71.06966502068597],
    ])
    np.testing.assert_allclose(accuracy.iloc[:2, 1:].to_numpy(dtype=float), expected,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(accuracy.iloc[2, 1:].to_numpy(dtype=float), expected[1] - expected[0],
                               rtol=0, atol=1e-10)


@pytest.mark.parametrize('task,expected', [
    ('s vs ud', [38.9487825356843, 41.97816960537363]),
    ('u vs d', [14.722600804441047, 20.426819133730014]),
])
def test_light_flavour_working_points(saved_tables, task, expected):
    _, working_points = saved_tables
    row = working_points[(working_points['task'] == task) & (working_points['background_eff_%'] == 10)]
    np.testing.assert_allclose(row[['baseline_sig_eff_%', 'fusion_sig_eff_%']].to_numpy()[0],
                               expected, rtol=0, atol=1e-10)


def test_selected_ablation_row_matches_predictions(saved_tables):
    accuracy, _ = saved_tables
    ablation = pd.read_csv(ROOT / 'results' / 'studies' / 'pairfeature_ablation_summary.csv')
    selected = ablation.set_index('pair_feature_set').loc['no_kaon']
    for column, metric in [('soft_fused_overall', 'overall_%'), ('soft_fused_macro', 'macro_%'),
                           ('soft_fused_acc_d', 'd_%')]:
        assert selected[column] * 100 == pytest.approx(accuracy.loc[1, metric], abs=1e-10)


def test_mismatched_baseline_rows_are_rejected(tmp_path):
    scores = np.array([[0.5, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.5, 0.1, 0.1, 0.1, 0.1]])
    labels = np.array([0, 0])
    np.savez(tmp_path / analysis.BASELINE_FILE, scores=scores, labels=labels,
             class_names=np.array(analysis.CLASS_NAMES))
    np.savez(tmp_path / analysis.FUSION_FILE, baseline_scores=scores[::-1], labels=labels,
             class_names=np.array(analysis.CLASS_NAMES), uds_class_names=np.array(analysis.UDS_CLASS_NAMES))
    with pytest.raises(ValueError, match='different baseline scores'):
        analysis.load_results(tmp_path)


@pytest.mark.parametrize('scores', [np.ones(8), np.array([.8, .5, .8, .2, .5, .2, .8, .5])])
def test_roc_groups_equal_scores_independently_of_row_order(scores):
    from sklearn.metrics import roc_curve
    from djt.metrics import roc_curve_xy

    labels = np.array([True, False, False, True, True, False, True, False])
    expected_fpr, expected_tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    for order in (np.arange(8), np.arange(8)[::-1], np.random.default_rng(1123).permutation(8)):
        tpr, fpr = roc_curve_xy(labels[order], scores[order])
        np.testing.assert_array_equal(tpr, expected_tpr)
        np.testing.assert_array_equal(fpr, expected_fpr)

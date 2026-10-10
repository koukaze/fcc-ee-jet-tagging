'''Reproduce comparison tables and figures from saved model predictions.'''

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from djt.metrics import (
    accuracy_table_with_delta,
    pairwise_discriminant,
    roc_curve_xy,
    signal_efficiency_at_background,
)
from djt.soft_fusion import soft_fuse_light_flavours


CLASS_NAMES = ('b', 'c', 's', 'u', 'd', 'g')
UDS_CLASS_NAMES = ('s', 'u', 'd')
BASELINE_FILE = 'baseline090_6cls_predictions.npz'
FUSION_FILE = 'A_no_kaon_baseline_softfusion_predictions.npz'
TASKS = ('b vs c', 'c vs s', 's vs ud', 'u vs d', 'd vs u', 'g vs q')
WORKING_POINTS = (0.10, 0.01, 0.001)


def load_results(results_dir: Path) -> dict[str, np.ndarray]:
    '''Load and validate the saved six-class comparison.
    Args
        results_dir (Path): Directory containing the baseline and fusion archives.
    Returns
        dict: Baseline and fused scores with shape (n_jets, 6), UDS scores
            with shape (n_jets, 3), and integer labels with shape (n_jets,).
    '''

    with np.load(results_dir / BASELINE_FILE, allow_pickle=False) as baseline:
        baseline_scores = baseline['scores']
        labels = baseline['labels']
        names = tuple(baseline['class_names'].tolist())
    with np.load(results_dir / FUSION_FILE, allow_pickle=False) as fusion:
        if names != CLASS_NAMES or tuple(fusion['class_names'].tolist()) != names:
            raise ValueError('Prediction class order must be b, c, s, u, d, g.')
        if tuple(fusion['uds_class_names'].tolist()) != UDS_CLASS_NAMES:
            raise ValueError('Specialist class order must be s, u, d.')
        if not np.array_equal(labels, fusion['labels']):
            raise ValueError('Prediction labels do not match row by row.')
        if not np.array_equal(baseline_scores, fusion['baseline_scores']):
            raise ValueError('The fusion archive contains different baseline scores.')
        fused_scores = fusion['fused_scores']
        uds_scores = fusion['uds_scores']
        p_light = fusion['p_light']

    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Labels must be a one-dimensional integer array.')
    if len(labels) == 0 or np.any((labels < 0) | (labels >= len(CLASS_NAMES))):
        raise ValueError('Labels must contain class IDs from 0 to 5.')
    for name, scores, n_classes in (
        ('baseline', baseline_scores, 6),
        ('fusion', fused_scores, 6),
        ('specialist', uds_scores, 3),
    ):
        if scores.shape != (len(labels), n_classes):
            raise ValueError(f'Unexpected {name} score shape: {scores.shape}.')
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError(f'{name} scores must be finite values between 0 and 1.')
        np.testing.assert_allclose(scores.sum(axis=1), 1.0, rtol=1e-6, atol=1e-7)

    np.testing.assert_array_equal(
        p_light, baseline_scores[:, 2] + baseline_scores[:, 3] + baseline_scores[:, 4]
    )
    np.testing.assert_allclose(
        fused_scores,
        soft_fuse_light_flavours(baseline_scores, uds_scores),
        rtol=1e-6,
        atol=1e-7,
    )
    return {
        'baseline_scores': baseline_scores,
        'fused_scores': fused_scores,
        'uds_scores': uds_scores,
        'labels': labels,
    }


def task_scores(
    scores: np.ndarray, labels: np.ndarray, task: str
) -> tuple[np.ndarray, np.ndarray]:
    '''Select truth labels and discriminant scores for a tagging task.
    Args
        scores (ndarray): Six-class probabilities with shape (n_jets, 6).
        labels (ndarray): Integer truth labels in the class order b, c, s, u, d, g.
        task (str): Pairwise, s-vs-ud, or g-vs-q task from TASKS.
    Returns
        tuple: Boolean signal mask and discriminant values for the selected jets.
    '''

    name_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    if task == 's vs ud':
        mask = np.isin(labels, [2, 3, 4])
        denominator = scores[:, 2] + scores[:, 3] + scores[:, 4]
        discriminant = np.divide(
            scores[:, 2], denominator, out=np.zeros_like(denominator), where=denominator > 0
        )
        return labels[mask] == 2, discriminant[mask]
    if task == 'g vs q':
        return labels == 5, scores[:, 5]
    if task not in TASKS:
        raise ValueError(f'Unknown tagging task: {task}.')
    signal, background = (name_to_idx[name] for name in task.split(' vs '))
    mask = (labels == signal) | (labels == background)
    discriminant = pairwise_discriminant(scores, signal, background)
    return labels[mask] == signal, discriminant[mask]


def comparison_tables(results: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Compute accuracy and fixed-background working-point tables.
    Args
        results (dict): Validated prediction arrays returned by load_results.
    Returns
        tuple: Accuracy and working-point DataFrames, with efficiencies in percent
            and differences in percentage points.
    '''

    baseline = results['baseline_scores']
    fused = results['fused_scores']
    labels = results['labels']
    accuracy = accuracy_table_with_delta(
        baseline, fused, labels, CLASS_NAMES,
        baseline_name='baseline090', improved_name='no_kaon_soft_fusion',
    )
    accuracy = accuracy[['model', 'overall_%', 'macro_%', *[f'{name}_%' for name in CLASS_NAMES]]]
    rows = []
    for task in TASKS:
        signal, base_score = task_scores(baseline, labels, task)
        _, fused_score = task_scores(fused, labels, task)
        for background_efficiency in WORKING_POINTS:
            base_eff, base_threshold = signal_efficiency_at_background(
                signal, base_score, background_efficiency
            )
            fused_eff, fused_threshold = signal_efficiency_at_background(
                signal, fused_score, background_efficiency
            )
            rows.append({
                'task': task,
                'background_eff_%': background_efficiency * 100,
                'baseline_sig_eff_%': base_eff * 100,
                'fusion_sig_eff_%': fused_eff * 100,
                'delta_pp': (fused_eff - base_eff) * 100,
                'baseline_threshold': base_threshold,
                'fusion_threshold': fused_threshold,
            })
    return accuracy, pd.DataFrame(rows)


def plot_comparison(results: dict[str, np.ndarray], accuracy: pd.DataFrame) -> plt.Figure:
    '''Plot per-class recalls and light-flavour ROC comparisons.
    Args
        results (dict): Validated baseline and fused predictions with truth labels.
        accuracy (DataFrame): Accuracy table returned by comparison_tables.
    Returns
        Figure: Two-panel comparison on the saved test sample.
    '''

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    positions = np.arange(len(CLASS_NAMES))
    columns = [f'{name}_%' for name in CLASS_NAMES]
    axes[0].bar(positions - 0.19, accuracy.loc[0, columns].to_numpy(dtype=float),
                width=0.38, color='#31688e', label='Baseline090')
    axes[0].bar(positions + 0.19, accuracy.loc[1, columns].to_numpy(dtype=float),
                width=0.38, color='#d68632', label='No-kaon soft fusion')
    axes[0].set(xticks=positions, xticklabels=CLASS_NAMES, ylim=(0, 100),
                xlabel='True jet flavour', ylabel='Per-class recall [%]',
                title='Six-class classification')
    axes[0].legend(frameon=False, fontsize=9)
    for task, color in [('s vs ud', '#31688e'), ('u vs d', '#a23b72')]:
        for key, linestyle, model in [
            ('baseline_scores', '--', 'baseline'),
            ('fused_scores', '-', 'soft fusion'),
        ]:
            signal, score = task_scores(results[key], results['labels'], task)
            signal_eff, background_eff = roc_curve_xy(signal, score)
            axes[1].plot(signal_eff, np.maximum(background_eff, 1e-4),
                         color=color, linestyle=linestyle, linewidth=1.7,
                         label=f'{task}: {model}')
    axes[1].set(xlim=(0, 1), ylim=(1e-3, 1), yscale='log',
                xlabel='Signal efficiency', ylabel='Background efficiency',
                title='Light-flavour discrimination')
    axes[1].legend(frameon=False, fontsize=8.5, loc='lower right')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=0.18)
    fig.suptitle(f'FCC-ee simulation · {len(results["labels"]):,} test jets', fontsize=13)
    return fig


def plot_ablation(summary: pd.DataFrame) -> plt.Figure:
    '''Plot macro-recall gains from the saved pair-feature ablation table.
    Args
        summary (DataFrame): Historical pairfeature_ablation_summary.csv table.
    Returns
        Figure: Macro-recall change for each tested pair-feature configuration.
    '''

    labels = {
        'full': 'Full pair features',
        'geometry_only': 'Geometry',
        'geometry_charge': 'Geometry + charge',
        'geometry_kaon': 'Geometry + kaon',
        'no_kaon': 'No kaon',
        'no_charge': 'No charge',
        'geometry_momentum': 'Geometry + momentum',
    }
    names = [labels[name] for name in summary['pair_feature_set']]
    gains = summary['soft_delta_macro'].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    positions = np.arange(len(summary))
    colors = ['#d68632' if name == 'no_kaon' else '#31688e'
              for name in summary['pair_feature_set']]
    bars = ax.barh(positions, gains, color=colors, height=0.65)
    ax.bar_label(bars, labels=[f'{gain:+.3f}' for gain in gains], padding=5, fontsize=9)
    ax.set(yticks=positions, yticklabels=names, xlim=(-0.22, 1.4),
           xlabel='Macro-recall change from baseline [percentage points]',
           title='Pair-feature ablation on the test sample')
    ax.invert_yaxis()
    ax.axvline(0, color='0.4', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=0.18)
    return fig


def main() -> None:
    '''Write comparison figures and tables from the saved prediction archives.'''

    plt.switch_backend('Agg')
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--results-dir', type=Path, default=root / 'results' / 'final')
    parser.add_argument('--ablation-table', type=Path,
                        default=root / 'results' / 'studies' / 'pairfeature_ablation_summary.csv')
    parser.add_argument('--output-dir', type=Path, default=root / 'outputs' / 'results')
    args = parser.parse_args()
    results = load_results(args.results_dir)
    accuracy, working_points = comparison_tables(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accuracy.to_csv(args.output_dir / 'accuracy_summary.csv', index=False)
    working_points.to_csv(args.output_dir / 'working_points.csv', index=False)
    figures = {
        'final_comparison': plot_comparison(results, accuracy),
        'pair_feature_ablation': plot_ablation(pd.read_csv(args.ablation_table)),
    }
    for name, fig in figures.items():
        for extension in ('png', 'svg'):
            fig.savefig(args.output_dir / f'{name}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    print(accuracy.round(3).to_string(index=False))
    print(f'Wrote comparison tables and figures to {args.output_dir}')


if __name__ == '__main__':
    main()

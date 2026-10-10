'''Metrics used by the flavour-tagging notebooks and scripts.'''

from __future__ import annotations

import numpy as np


def confusion_counts(labels: np.ndarray, predictions: np.ndarray, n_classes: int) -> np.ndarray:
    '''Return integer confusion counts with rows=true and columns=predicted.
    Args
        labels (ndarray): Integer truth labels.
        predictions (ndarray): Integer predicted labels.
        n_classes (int): Number of classes.
    Returns
        ndarray: Confusion-count matrix.
    '''

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(labels, predictions):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def per_class_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
    n_classes: int | None = None,
) -> list[float]:
    '''Return one accuracy value per true class.
    Args
        labels (ndarray): Integer truth labels.
        predictions (ndarray): Integer predicted labels.
        n_classes (int): Number of classes. If None, infer from labels.
    Returns
        list: Per-class accuracies as fractions.
    '''

    if n_classes is None:
        n_classes = int(np.nanmax(labels)) + 1 if len(labels) else 0
    out = []
    for i in range(n_classes):
        mask = labels == i
        out.append(float(np.mean(predictions[mask] == i)) if np.any(mask) else np.nan)
    return out


def row_normalised_confusion(labels: np.ndarray, predictions: np.ndarray, n_classes: int) -> np.ndarray:
    '''Return row-normalised confusion matrix in percent.
    Args
        labels (ndarray): Integer truth labels.
        predictions (ndarray): Integer predicted labels.
        n_classes (int): Number of classes.
    Returns
        ndarray: Row-normalised confusion matrix in percent.
    '''

    cm = confusion_counts(labels, predictions, n_classes)
    denom = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0) * 100.0


def row_normalised_confusion_from_scores(scores: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    '''Return row-normalised confusion in percent from score arrays.
    Args
        scores (ndarray): Model scores with shape (n_jets, n_classes).
        labels (ndarray): Integer truth labels.
        n_classes (int): Number of classes.
    Returns
        ndarray: Row-normalised confusion matrix in percent.
    '''

    return row_normalised_confusion(labels, scores.argmax(axis=1), n_classes)


def accuracy_summary(
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | tuple[str, ...],
    model_name: str,
) -> dict[str, float | str]:
    '''Return overall, macro, and per-class accuracy in percent.
    Args
        scores (ndarray): Model scores with shape (n_jets, n_classes).
        labels (ndarray): Integer truth labels.
        class_names (list or tuple): Class names in score-column order.
        model_name (str): Name to store in the output row.
    Returns
        dict: Accuracy summary row.
    '''

    pred = scores.argmax(axis=1)
    row: dict[str, float | str] = {
        "model": model_name,
        "overall_%": float(np.mean(pred == labels) * 100.0),
    }
    per_class = []
    for i, name in enumerate(class_names):
        mask = labels == i
        acc = float(np.mean(pred[mask] == i) * 100.0) if np.any(mask) else np.nan
        row[f"{name}_%"] = acc
        per_class.append(acc)
    row["macro_%"] = float(np.nanmean(per_class))
    return row


def pairwise_discriminant(scores: np.ndarray, signal_idx: int, background_idx: int) -> np.ndarray:
    '''Return S_i / (S_i + S_j) for classes i and j.
    Args
        scores (ndarray): Model scores with shape (n_jets, n_classes).
        signal_idx (int): Column index for the signal class.
        background_idx (int): Column index for the background class.
    Returns
        ndarray: Pairwise discriminant values.
    '''

    denom = scores[:, signal_idx] + scores[:, background_idx]
    return np.divide(scores[:, signal_idx], denom, out=np.zeros_like(denom), where=denom > 0)


def roc_curve_xy(y_signal: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Return signal efficiency and background efficiency arrays.
    Args
        y_signal (ndarray): Boolean mask selecting signal jets.
        score (ndarray): Classifier score used to rank jets.
    Returns
        tuple: Signal efficiency and background efficiency arrays.
    '''

    y_signal = np.asarray(y_signal, dtype=bool)
    score = np.asarray(score)
    order = np.argsort(score)[::-1]
    y_sorted = y_signal[order]
    n_signal = y_sorted.sum()
    n_background = len(y_sorted) - n_signal
    if n_signal == 0 or n_background == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    signal_eff = np.r_[0.0, tp / n_signal, 1.0]
    background_eff = np.r_[0.0, fp / n_background, 1.0]
    return signal_eff, background_eff


def signal_efficiency_at_background(
    y_signal: np.ndarray,
    score: np.ndarray,
    background_efficiency: float,
) -> tuple[float, float]:
    '''Return signal efficiency and threshold at fixed background efficiency.
    Args
        y_signal (ndarray): Boolean mask selecting signal jets.
        score (ndarray): Classifier score used to rank jets.
        background_efficiency (float): Target background efficiency.
    Returns
        tuple: Signal efficiency and selected threshold.
    '''

    y_signal = np.asarray(y_signal, dtype=bool)
    signal_scores = score[y_signal]
    background_scores = score[~y_signal]
    if len(signal_scores) == 0 or len(background_scores) == 0:
        return np.nan, np.nan
    threshold = float(np.quantile(background_scores, 1.0 - background_efficiency))
    return float(np.mean(signal_scores >= threshold)), threshold


def accuracy_table_with_delta(
    baseline_scores: np.ndarray,
    improved_scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | tuple[str, ...],
    baseline_name: str = "baseline",
    improved_name: str = "improved",
) -> object:
    '''Return baseline/improved accuracy table plus improved-minus-baseline row.
    Args
        baseline_scores (ndarray): Baseline score array.
        improved_scores (ndarray): Improved-model score array.
        labels (ndarray): Integer truth labels.
        class_names (list or tuple): Class names in score-column order.
        baseline_name (str): Name for the baseline row.
        improved_name (str): Name for the improved-model row.
    Returns
        DataFrame: Accuracy table with a percentage-point delta row.
    '''

    import pandas as pd

    table = pd.DataFrame(
        [
            accuracy_summary(baseline_scores, labels, class_names, baseline_name),
            accuracy_summary(improved_scores, labels, class_names, improved_name),
        ]
    )
    delta = table.iloc[1].copy()
    delta["model"] = "delta_pp_improved_minus_baseline"
    for col in table.columns:
        if col != "model":
            delta[col] = table.loc[1, col] - table.loc[0, col]
    return pd.concat([table, pd.DataFrame([delta])], ignore_index=True)

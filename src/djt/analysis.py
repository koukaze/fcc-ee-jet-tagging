'''Analysis helpers shared by the result notebooks.'''

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_npz(path: str | Path, *, allow_pickle: bool = True) -> Any:
    '''Load an NPZ file with a clear error if it is missing.
    Args
        path (str or Path): Path to the NPZ file.
        allow_pickle (bool): Whether numpy may load pickled objects.
    Returns
        NpzFile: Loaded numpy archive.
    '''

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=allow_pickle)


def as_class_names(values) -> list[str]:
    '''Convert class-name arrays from NPZ files to plain Python strings.
    Args
        values (array-like): Class names loaded from a result archive.
    Returns
        list: Class names as strings.
    '''

    return [str(value) for value in values]


def load_prediction_file(
    path: str | Path,
    *,
    score_key: str = "scores",
    expected_class_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    '''Load a standard prediction NPZ file.
    Args
        path (str or Path): Path to the prediction archive.
        score_key (str): Key containing the model score array.
        expected_class_names (list or tuple): Optional class order to validate.
    Returns
        dict: Prediction arrays, class names, file path, and the open archive.
    '''

    data = load_npz(path)
    class_names = as_class_names(data["class_names"])
    if expected_class_names is not None and class_names != list(expected_class_names):
        raise ValueError(f"Unexpected class order in {path}: {class_names}")
    return {
        "path": Path(path),
        "scores": data[score_key],
        "labels": data["labels"].astype(int),
        "class_names": class_names,
        "npz": data,
    }


def load_softfusion_file(
    path: str | Path,
    *,
    expected_class_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    '''Load a soft-fusion prediction NPZ file.
    Args
        path (str or Path): Path to the soft-fusion prediction archive.
        expected_class_names (list or tuple): Optional class order to validate.
    Returns
        dict: Fused scores, component scores, labels, class names, and archive.
    '''

    data = load_npz(path)
    class_names = as_class_names(data["class_names"])
    if expected_class_names is not None and class_names != list(expected_class_names):
        raise ValueError(f"Unexpected class order in {path}: {class_names}")
    return {
        "path": Path(path),
        "fused_scores": data["fused_scores"],
        "baseline_scores": data["baseline_scores"],
        "uds_scores": data["uds_scores"],
        "labels": data["labels"].astype(int),
        "class_names": class_names,
        "npz": data,
    }


def load_metrics_csv(path: str | Path):
    '''Load a metrics CSV with a clear error if it is missing.
    Args
        path (str or Path): Path to the metrics CSV file.
    Returns
        DataFrame: Loaded metrics table.
    '''

    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def best_and_last_metrics(metrics, loss_col: str = "eval_loss"):
    '''Return the best row by evaluation loss and the last metrics row.
    Args
        metrics (DataFrame): Training metrics table.
        loss_col (str): Column used to select the best epoch.
    Returns
        tuple: Best-loss row and final row.
    '''

    return metrics.loc[metrics[loss_col].idxmin()], metrics.iloc[-1]


def multiclass_ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | tuple[str, ...],
) -> dict[str, float]:
    '''Return macro one-vs-rest AUC and average precision.
    Args
        scores (ndarray): Model scores with shape (n_jets, n_classes).
        labels (ndarray): Integer truth labels.
        class_names (list or tuple): Class names in score-column order.
    Returns
        dict: Macro one-vs-rest AUC and macro average precision.
    '''

    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.eye(len(class_names))[labels]
    return {
        "macro_auc": float(roc_auc_score(y_true, scores, average="macro", multi_class="ovr")),
        "macro_ap": float(average_precision_score(y_true, scores, average="macro")),
    }

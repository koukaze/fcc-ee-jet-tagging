'''Soft-fusion utilities for combining the 6-class tagger with a UDS specialist.'''

from __future__ import annotations

import numpy as np


def soft_fuse_light_flavours(
    baseline_scores: np.ndarray,
    uds_scores: np.ndarray,
    class_names: list[str] | tuple[str, ...] = ("b", "c", "s", "u", "d", "g"),
    uds_class_names: list[str] | tuple[str, ...] = ("s", "u", "d"),
) -> np.ndarray:
    '''Return 6-class scores after redistributing baseline light probability.
    Args
        baseline_scores (ndarray): Six-class baseline probabilities.
        uds_scores (ndarray): Three-class UDS specialist probabilities.
        class_names (list or tuple): Six-class score-column order.
        uds_class_names (list or tuple): UDS score-column order.
    Returns
        ndarray: Fused six-class probabilities.
    '''

    fused = np.array(baseline_scores, copy=True)
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    uds_to_idx = {name: i for i, name in enumerate(uds_class_names)}

    light = ["s", "u", "d"]
    p_light = sum(baseline_scores[:, name_to_idx[name]] for name in light)
    for name in light:
        fused[:, name_to_idx[name]] = p_light * uds_scores[:, uds_to_idx[name]]
    return fused


def light_probability(
    scores: np.ndarray,
    class_names: list[str] | tuple[str, ...] = ("b", "c", "s", "u", "d", "g"),
) -> np.ndarray:
    '''Return P(s)+P(u)+P(d) from a 6-class score array.
    Args
        scores (ndarray): Six-class score array.
        class_names (list or tuple): Score-column order.
    Returns
        ndarray: Total light-flavour probability for each jet.
    '''

    name_to_idx = {name: i for i, name in enumerate(class_names)}
    return scores[:, name_to_idx["s"]] + scores[:, name_to_idx["u"]] + scores[:, name_to_idx["d"]]

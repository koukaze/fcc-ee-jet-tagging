'''Plotting helpers for flavour-tagging result notebooks.'''

from __future__ import annotations

from pathlib import Path

import numpy as np


def annotate_confusion_matrix(
    ax,
    matrix: np.ndarray,
    fmt: str = ".1f",
    *,
    threshold: float | None = None,
    dark_color: str = "black",
    light_color: str = "white",
) -> None:
    '''Write values into an already-drawn confusion matrix axis.
    Args
        ax (Axes): Matplotlib axis containing the matrix image.
        matrix (ndarray): Matrix values to write.
        fmt (str): Format string for cell values.
        threshold (float): Value above which light text is used.
        dark_color (str): Text colour for lighter cells.
        light_color (str): Text colour for darker cells.
    Returns
        None.
    '''

    if threshold is None:
        threshold = float(np.nanmax(matrix) / 2.0) if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = light_color if value > threshold else dark_color
            ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8, color=color)


def plot_confusion_matrix(
    ax,
    matrix: np.ndarray,
    class_names: list[str] | tuple[str, ...],
    title: str,
    *,
    fmt: str = ".1f",
    text_threshold: float | None = None,
    **imshow_kwargs,
):
    '''Draw a confusion matrix on a supplied matplotlib axis.
    Args
        ax (Axes): Matplotlib axis to draw on.
        matrix (ndarray): Confusion matrix values.
        class_names (list or tuple): Class labels for both axes.
        title (str): Axis title.
        fmt (str): Format string for cell annotations.
        text_threshold (float): Value above which light text is used.
        **imshow_kwargs: Extra keyword arguments passed to imshow.
    Returns
        AxesImage: Image returned by matplotlib imshow.
    '''

    if "vmin" not in imshow_kwargs:
        imshow_kwargs["vmin"] = 0
    if "vmax" not in imshow_kwargs:
        imshow_kwargs["vmax"] = 100
    if "cmap" not in imshow_kwargs:
        imshow_kwargs["cmap"] = "Blues"
    image = ax.imshow(matrix, **imshow_kwargs)
    ax.set_title(title)
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    annotate_confusion_matrix(ax, matrix, fmt=fmt, threshold=text_threshold)
    return image


def save_figure_pdf(
    fig,
    name: str,
    *,
    figures_dir: str | Path = "figures",
    subdir: str | Path | None = None,
) -> Path:
    '''Save a matplotlib figure as a tightly cropped PDF.
    Args
        fig (Figure): Matplotlib figure to save.
        name (str): Output filename without the .pdf extension.
        figures_dir (str or Path): Root figures directory.
        subdir (str or Path): Optional subdirectory inside figures_dir.
    Returns
        Path: Path to the saved PDF file.
    '''

    output_dir = Path(figures_dir)
    if subdir is not None:
        output_dir = output_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{name}.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    return output_path

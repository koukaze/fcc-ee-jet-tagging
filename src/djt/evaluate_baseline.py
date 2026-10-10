#!/usr/bin/env python3
'''Evaluate a trained baseline090 6-class model and make score/ROC plots.'''

from __future__ import annotations

import argparse
import glob as globlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from .baseline import (
    CLASS_NAMES,
    DeepJetTransformerV0,
    JetDataset,
    load_data,
    make_split_indices,
    move_batch,
)


PLOT_LABELS = {
    "b": "b-jets",
    "c": "c-jets",
    "s": "s-jets",
    "u": "u-jets",
    "d": "d-jets",
    "g": "g-jets",
}

COLORS = {
    "b": "#3b82f6",
    "c": "#f59e0b",
    "s": "#e11d48",
    "u": "#9333ea",
    "d": "#9ca3af",
    "g": "#16a34a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-files", nargs="+", default=["data/chunk_*.root"])
    parser.add_argument("--tree-name", default="deepntuplizer/tree")
    parser.add_argument("--checkpoint", default="runs/baseline090_6cls/checkpoint_best_loss.pth")
    parser.add_argument("--plot-dir", default="plots")
    parser.add_argument(
        "--name",
        default="baseline090_6cls",
        help="Filename prefix. Outputs are <name>_ROC.pdf and <name>_scores.pdf.",
    )
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument(
        "--eval-fraction",
        "--val-fraction",
        dest="eval_fraction",
        type=float,
        default=0.15,
        help="Fraction reserved for checkpoint selection.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction reserved for final test evaluation.",
    )
    parser.add_argument("--seed", type=int, default=1123)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional event limit for smoke tests.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--split",
        choices=["validation", "evaluation", "test", "all"],
        default="validation",
        help="Evaluate a seeded split from training, or all loaded events.",
    )
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def expand_root_files(root_files: list[str]) -> list[str]:
    expanded: list[str] = []
    for item in root_files:
        matches = sorted(globlib.glob(item)) if any(ch in item for ch in "*?[") else []
        expanded.extend(matches or [item])
    return expanded


def make_eval_indices(
    n_total: int,
    eval_fraction: float,
    test_fraction: float,
    seed: int,
    split: str,
) -> np.ndarray:
    if split == "all":
        return np.arange(n_total)
    train_idx, eval_idx, test_idx = make_split_indices(n_total, eval_fraction, test_fraction, seed)
    if split in ("validation", "evaluation"):
        return eval_idx
    return test_idx


@torch.no_grad()
def predict(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for batch in tqdm(loader, desc="Evaluating"):
        glob, cpf, npf, sv, v0, y = move_batch(batch, device)
        logits = model(glob, cpf, npf, sv, v0)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.argmax(dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)


def plot_score_distributions(scores: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    bins = np.linspace(0.0, 1.0, 51)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()

    for score_idx, score_name in enumerate(CLASS_NAMES):
        ax = axes[score_idx]
        order = [score_name] + [name for name in CLASS_NAMES if name != score_name]
        hist_values = [
            scores[labels == CLASS_NAMES.index(class_name), score_idx]
            for class_name in order
        ]
        ax.hist(
            hist_values,
            bins=bins,
            stacked=True,
            histtype="stepfilled",
            label=[PLOT_LABELS[name] for name in order],
            color=[COLORS[name] for name in order],
            alpha=0.95,
        )
        ax.set_yscale("log")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(f"{score_name} score")
        ax.set_ylabel("Jets / 0.02")
        ax.text(0.02, 0.95, "DeepJetTransformerV0", transform=ax.transAxes, va="top", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_one_vs_one_rocs(scores: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()

    for sig_idx, sig_name in enumerate(CLASS_NAMES):
        ax = axes[sig_idx]
        for bkg_idx, bkg_name in enumerate(CLASS_NAMES):
            if bkg_idx == sig_idx:
                continue
            mask = (labels == sig_idx) | (labels == bkg_idx)
            binary_truth = (labels[mask] == sig_idx).astype(np.int32)
            sig_scores = scores[mask, sig_idx]
            fpr, tpr, _ = roc_curve(binary_truth, sig_scores)
            ax.plot(
                tpr,
                np.maximum(fpr, 1e-3),
                label=f"{sig_name} vs {PLOT_LABELS[bkg_name]}",
                color=COLORS[bkg_name],
                linewidth=1.8,
            )

        ax.set_title(f"{sig_name}-tagging")
        ax.set_xlabel(r"Signal Efficiency ($\epsilon_{sig}$)")
        ax.set_ylabel(r"Background Efficiency ($\epsilon_{bkg}$)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1e-3, 1.0)
        ax.set_yscale("log")
        ax.grid(False)
        ax.legend(fontsize=8, loc="lower right")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root_files = expand_root_files(args.root_files)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inputs: {root_files}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Output prefix: {args.name}")

    glob, cpf, npf, sv, v0, truth = load_data(root_files, args.tree_name, args.limit)
    eval_idx = make_eval_indices(
        len(glob),
        args.eval_fraction,
        args.test_fraction,
        args.seed,
        args.split,
    )
    eval_ds = JetDataset(glob, cpf, npf, sv, v0, truth, eval_idx)
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    print(f"Eval jets: {len(eval_ds)} ({len(loader)} batches)")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    model = DeepJetTransformerV0(
        d_model=int(ckpt_args.get("d_model", 128)),
        nhead=int(ckpt_args.get("nhead", 8)),
        n_encoder_layers=int(ckpt_args.get("n_encoder_layers", 3)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    print(
        f"Loaded checkpoint epoch={checkpoint.get('epoch')} best_loss={checkpoint.get('best_loss')} "
        f"d_model={ckpt_args.get('d_model', 128)} layers={ckpt_args.get('n_encoder_layers', 3)} "
        f"dropout={ckpt_args.get('dropout', 0.1)}"
    )

    scores, labels = predict(model, loader, device)
    print(f"Predictions: scores={scores.shape}, labels={labels.shape}")

    scores_path = plot_dir / f"{args.name}_scores.pdf"
    roc_path = plot_dir / f"{args.name}_ROC.pdf"
    plot_score_distributions(scores, labels, scores_path)
    plot_one_vs_one_rocs(scores, labels, roc_path)
    print(f"Saved {scores_path}")
    print(f"Saved {roc_path}")

    if args.save_predictions:
        pred_path = plot_dir / f"{args.name}_predictions.npz"
        np.savez(pred_path, scores=scores, labels=labels, class_names=np.array(CLASS_NAMES))
        print(f"Saved {pred_path}")


if __name__ == "__main__":
    main()

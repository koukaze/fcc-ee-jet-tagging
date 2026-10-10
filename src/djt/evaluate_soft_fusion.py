#!/usr/bin/env python3
'''Evaluate probability-level soft fusion of the six-class tagger and UDS specialist.'''

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

from . import baseline as full6
from . import pair_bias as uds3

CLASS_NAMES = ["b", "c", "s", "u", "d", "g"]
UDS_CLASS_NAMES = ["s", "u", "d"]

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
    parser.add_argument("--baseline-checkpoint", default="runs/full_tuning/A_baseline/checkpoint_best_loss.pth")
    parser.add_argument("--uds-checkpoint", default="runs/uds_pairfeature_ablation/A_pair_full/checkpoint_best_loss.pth")
    parser.add_argument("--plot-dir", default="plots/soft_fusion_pairfeature_ablation")
    parser.add_argument("--name", default="A_pair_full_softfusion")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-fraction", "--val-fraction", dest="eval_fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1123)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def expand_root_files(root_files: list[str]) -> list[str]:
    expanded: list[str] = []
    for item in root_files:
        matches = sorted(globlib.glob(item)) if any(ch in item for ch in "*?[") else []
        expanded.extend(matches or [item])
    return expanded


def make_test_indices(n_total: int, eval_fraction: float, test_fraction: float, seed: int) -> np.ndarray:
    _, _, test_idx = full6.make_split_indices(n_total, eval_fraction, test_fraction, seed)
    return test_idx


def build_full6_model(checkpoint: dict, device: torch.device) -> full6.DeepJetTransformerV0:
    ckpt_args = checkpoint.get("args", {})
    model = full6.DeepJetTransformerV0(
        d_model=int(ckpt_args.get("d_model", 128)),
        nhead=int(ckpt_args.get("nhead", 8)),
        n_encoder_layers=int(ckpt_args.get("n_encoder_layers", 3)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def build_uds_model(checkpoint: dict, device: torch.device) -> uds3.UDSSpecialistPairBiasTransformer:
    ckpt_args = checkpoint.get("args", {})
    model = uds3.UDSSpecialistPairBiasTransformer(
        d_model=int(ckpt_args.get("d_model", 128)),
        nhead=int(ckpt_args.get("nhead", 8)),
        n_encoder_layers=int(ckpt_args.get("n_encoder_layers", 3)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
        pair_hidden_dim=int(ckpt_args.get("pair_hidden_dim", 32)),
        pair_feature_set=ckpt_args.get("pair_feature_set", checkpoint.get("pair_feature_set", "full")),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


@torch.no_grad()
def predict_both(baseline_model, uds_model, loader, device: torch.device):
    baseline_model.eval()
    uds_model.eval()
    baseline_probs, uds_probs, labels = [], [], []
    for batch in tqdm(loader, desc="Evaluating soft fusion"):
        glob, cpf, npf, sv, v0, y = full6.move_batch(batch, device)
        baseline_logits = baseline_model(glob, cpf, npf, sv, v0)
        uds_logits = uds_model(glob, cpf, npf, sv, v0)
        baseline_probs.append(torch.softmax(baseline_logits, dim=1).cpu().numpy())
        uds_probs.append(torch.softmax(uds_logits, dim=1).cpu().numpy())
        labels.append(y.argmax(dim=1).cpu().numpy())
    return (
        np.concatenate(baseline_probs, axis=0),
        np.concatenate(uds_probs, axis=0),
        np.concatenate(labels, axis=0),
    )


def fuse_scores(baseline_scores: np.ndarray, uds_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p_light = baseline_scores[:, 2] + baseline_scores[:, 3] + baseline_scores[:, 4]
    fused = np.zeros_like(baseline_scores)
    fused[:, 0] = baseline_scores[:, 0]
    fused[:, 1] = baseline_scores[:, 1]
    fused[:, 5] = baseline_scores[:, 5]
    fused[:, 2] = p_light * uds_scores[:, 0]
    fused[:, 3] = p_light * uds_scores[:, 1]
    fused[:, 4] = p_light * uds_scores[:, 2]
    return fused, p_light


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def accuracy_summary(scores: np.ndarray, labels: np.ndarray) -> dict:
    pred = scores.argmax(axis=1)
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = labels == i
        per_class[name] = float((pred[mask] == i).mean()) if mask.any() else float("nan")
    return {
        "overall": float((pred == labels).mean()),
        "macro": float(np.nanmean(list(per_class.values()))),
        "per_class": per_class,
        "confusion": confusion_matrix(labels, pred, len(CLASS_NAMES)),
    }


def format_confusion_counts(cm: np.ndarray) -> list[str]:
    lines = ["      " + " ".join(f"{name:>10s}" for name in CLASS_NAMES)]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"{name:>4s} " + " ".join(f"{value:10d}" for value in cm[i]))
    return lines


def format_confusion_pct(cm: np.ndarray) -> list[str]:
    cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0
    lines = ["      " + " ".join(f"{name:>10s}" for name in CLASS_NAMES)]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"{name:>4s} " + " ".join(f"{value:10.2f}" for value in cm_pct[i]))
    return lines


def write_summary(path: Path, baseline_scores, fused_scores, uds_scores, labels, p_light, baseline_ckpt, uds_ckpt):
    baseline = accuracy_summary(baseline_scores, labels)
    fused = accuracy_summary(fused_scores, labels)
    sums = fused_scores.sum(axis=1)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))

    lines = []
    lines.append("Soft-fusion baseline090 + UDS pair-feature ablation held-out test summary")
    lines.append("")
    lines.append(f"Test jets: {len(labels)}")
    lines.append("Class counts:")
    for name, count in zip(CLASS_NAMES, counts):
        lines.append(f"  {name}: {int(count)}")
    lines.append("")
    lines.append(f"Baseline checkpoint epoch: {baseline_ckpt.get('epoch', 'unknown')} best_loss: {baseline_ckpt.get('best_loss', 'unknown')}")
    lines.append(f"UDS checkpoint epoch: {uds_ckpt.get('epoch', 'unknown')} best_loss: {uds_ckpt.get('best_loss', 'unknown')}")
    uds_args = uds_ckpt.get("args", {})
    pair_feature_set = uds_args.get("pair_feature_set", uds_ckpt.get("pair_feature_set", "full"))
    pair_features = uds_ckpt.get("pair_features", uds3.get_included_pair_feature_names(pair_feature_set))
    lines.append(f"UDS pair feature set: {pair_feature_set}")
    lines.append("UDS pair features: " + ", ".join(pair_features))
    lines.append("")
    lines.append("Baseline performance:")
    lines.append(f"  overall: {100.0 * baseline['overall']:.4f}%")
    lines.append(f"  macro:   {100.0 * baseline['macro']:.4f}%")
    for name in CLASS_NAMES:
        lines.append(f"  {name}: {100.0 * baseline['per_class'][name]:.4f}%")
    lines.append("")
    lines.append("Fused performance:")
    lines.append(f"  overall: {100.0 * fused['overall']:.4f}%")
    lines.append(f"  macro:   {100.0 * fused['macro']:.4f}%")
    for name in CLASS_NAMES:
        lines.append(f"  {name}: {100.0 * fused['per_class'][name]:.4f}%")
    lines.append("")
    lines.append("Delta fused - baseline [percentage points]:")
    lines.append(f"  overall: {100.0 * (fused['overall'] - baseline['overall']):+.4f}")
    lines.append(f"  macro:   {100.0 * (fused['macro'] - baseline['macro']):+.4f}")
    for name in CLASS_NAMES:
        lines.append(f"  {name}: {100.0 * (fused['per_class'][name] - baseline['per_class'][name]):+.4f}")
    lines.append("")
    lines.append("Fused probability sum check:")
    lines.append(f"  min:  {sums.min():.8f}")
    lines.append(f"  max:  {sums.max():.8f}")
    lines.append(f"  mean: {sums.mean():.8f}")
    lines.append("")
    lines.append("p_light summary:")
    lines.append(f"  min:  {p_light.min():.8f}")
    lines.append(f"  max:  {p_light.max():.8f}")
    lines.append(f"  mean: {p_light.mean():.8f}")
    lines.append("")
    lines.append("Baseline confusion matrix counts (rows=true, cols=pred):")
    lines.extend(format_confusion_counts(baseline["confusion"]))
    lines.append("")
    lines.append("Fused confusion matrix counts (rows=true, cols=pred):")
    lines.extend(format_confusion_counts(fused["confusion"]))
    lines.append("")
    lines.append("Baseline confusion matrix row-normalised [%] (rows=true, cols=pred):")
    lines.extend(format_confusion_pct(baseline["confusion"]))
    lines.append("")
    lines.append("Fused confusion matrix row-normalised [%] (rows=true, cols=pred):")
    lines.extend(format_confusion_pct(fused["confusion"]))
    path.write_text("\n".join(lines) + "\n")


def plot_confusion(baseline_scores, fused_scores, labels, out_path: Path):
    baseline_cm = accuracy_summary(baseline_scores, labels)["confusion"]
    fused_cm = accuracy_summary(fused_scores, labels)["confusion"]
    base_pct = baseline_cm / np.maximum(baseline_cm.sum(axis=1, keepdims=True), 1) * 100.0
    fused_pct = fused_cm / np.maximum(fused_cm.sum(axis=1, keepdims=True), 1) * 100.0
    delta = fused_pct - base_pct

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, title, mat, cmap, vmin, vmax in [
        (axes[0], "Baseline row-normalised [%]", base_pct, "Blues", 0, 100),
        (axes[1], "Soft fusion row-normalised [%]", fused_pct, "Blues", 0, 100),
        (axes[2], "Fusion - baseline [pp]", delta, "RdBu", -30, 30),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                val = mat[i, j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_rocs(baseline_scores, fused_scores, labels, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()
    for idx, name in enumerate(CLASS_NAMES):
        ax = axes[idx]
        truth = (labels == idx).astype(np.int32)
        for scores, label, style in [
            (baseline_scores, "baseline", "--"),
            (fused_scores, "soft fusion", "-"),
        ]:
            fpr, tpr, _ = roc_curve(truth, scores[:, idx])
            ax.plot(tpr, np.maximum(fpr, 1e-4), linestyle=style, label=label, color=COLORS[name], linewidth=1.8)
        ax.set_title(f"{name}-tagging one-vs-rest")
        ax.set_xlabel(r"Signal Efficiency ($\epsilon_{sig}$)")
        ax.set_ylabel(r"Background Efficiency ($\epsilon_{bkg}$)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1e-4, 1.0)
        ax.set_yscale("log")
        ax.grid(False)
        ax.legend(fontsize=8)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(scores, labels, out_path: Path):
    bins = np.linspace(0.0, 1.0, 51)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()
    for score_idx, score_name in enumerate(CLASS_NAMES):
        ax = axes[score_idx]
        order = [score_name] + [name for name in CLASS_NAMES if name != score_name]
        hist_values = [scores[labels == CLASS_NAMES.index(class_name), score_idx] for class_name in order]
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
        ax.set_xlabel(f"fused {score_name} score")
        ax.set_ylabel("Jets / 0.02")
        ax.text(0.02, 0.95, "Soft fusion", transform=ax.transAxes, va="top", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root_files = expand_root_files(args.root_files)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inputs: {root_files}")
    print(f"Baseline checkpoint: {args.baseline_checkpoint}")
    print(f"UDS checkpoint: {args.uds_checkpoint}")
    print(f"Output: {plot_dir / args.name}")

    glob, cpf, npf, sv, v0, truth = full6.load_data(root_files, args.tree_name, args.limit)
    test_idx = make_test_indices(len(glob), args.eval_fraction, args.test_fraction, args.seed)
    eval_ds = full6.JetDataset(glob, cpf, npf, sv, v0, truth, test_idx)
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    print(f"Held-out test jets: {len(eval_ds)} ({len(loader)} batches)")

    device = torch.device(args.device)
    baseline_ckpt = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    uds_ckpt = torch.load(args.uds_checkpoint, map_location=device, weights_only=False)
    baseline_model = build_full6_model(baseline_ckpt, device)
    uds_model = build_uds_model(uds_ckpt, device)

    baseline_scores, uds_scores, labels = predict_both(baseline_model, uds_model, loader, device)
    fused_scores, p_light = fuse_scores(baseline_scores, uds_scores)
    sums = fused_scores.sum(axis=1)
    print(f"Predictions: baseline={baseline_scores.shape}, uds={uds_scores.shape}, fused={fused_scores.shape}, labels={labels.shape}")
    print(f"Fused probability sums: min={sums.min():.8f}, max={sums.max():.8f}, mean={sums.mean():.8f}")

    pred_path = plot_dir / f"{args.name}_predictions.npz"
    np.savez(
        pred_path,
        fused_scores=fused_scores,
        baseline_scores=baseline_scores,
        uds_scores=uds_scores,
        labels=labels,
        class_names=np.array(CLASS_NAMES),
        uds_class_names=np.array(UDS_CLASS_NAMES),
        p_light=p_light,
        pair_feature_set=np.array(uds_ckpt.get("args", {}).get("pair_feature_set", uds_ckpt.get("pair_feature_set", "full"))),
        pair_features=np.array(uds_ckpt.get("pair_features", uds3.get_included_pair_feature_names(uds_ckpt.get("args", {}).get("pair_feature_set", uds_ckpt.get("pair_feature_set", "full"))))),
    )
    print(f"Saved {pred_path}")

    summary_path = plot_dir / f"{args.name}_summary.txt"
    write_summary(summary_path, baseline_scores, fused_scores, uds_scores, labels, p_light, baseline_ckpt, uds_ckpt)
    print(f"Saved {summary_path}")

    confusion_path = plot_dir / f"{args.name}_confusion.pdf"
    roc_path = plot_dir / f"{args.name}_ROC.pdf"
    scores_path = plot_dir / f"{args.name}_scores.pdf"
    plot_confusion(baseline_scores, fused_scores, labels, confusion_path)
    plot_rocs(baseline_scores, fused_scores, labels, roc_path)
    plot_score_distributions(fused_scores, labels, scores_path)
    print(f"Saved {confusion_path}")
    print(f"Saved {roc_path}")
    print(f"Saved {scores_path}")


if __name__ == "__main__":
    main()

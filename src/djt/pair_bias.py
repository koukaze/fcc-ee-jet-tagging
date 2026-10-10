#!/usr/bin/env python3
'''Train a baseline090 UDS pair-bias specialist with selectable pair-feature sets.'''

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import pair_bias_base as base

CLASS_NAMES = base.CLASS_NAMES
NUM_CLASSES = base.NUM_CLASSES
EPS = base.EPS
TOKEN_TYPES = base.TOKEN_TYPES
N_TOKEN_TYPES = base.N_TOKEN_TYPES

JetDataset = base.JetDataset
filter_uds_indices = base.filter_uds_indices
load_data = base.load_data
make_split_indices = base.make_split_indices
make_uds_truth = base.make_uds_truth
move_batch = base.move_batch

SCALAR_FEATURE_GROUPS = {
    "geometry": ["dtheta", "sin_dphi", "cos_dphi", "log_deltaR"],
    "momentum": ["log_p_i", "log_p_j"],
    "charge": ["charge_i_times_charge_j", "same_nonzero_charge", "opposite_charge"],
    "kaon": ["is_kaon_i", "is_kaon_j", "both_kaon", "one_kaon"],
}

PAIR_FEATURE_SETS = {
    "full": ["geometry", "momentum", "charge", "kaon"],
    "geometry_only": ["geometry"],
    "geometry_charge": ["geometry", "charge"],
    "geometry_kaon": ["geometry", "kaon"],
    "no_kaon": ["geometry", "momentum", "charge"],
    "no_charge": ["geometry", "momentum", "kaon"],
    "geometry_momentum": ["geometry", "momentum"],
}

RUN_FEATURE_SETS = {
    "A_pair_full": "full",
    "B_pair_geometry_only": "geometry_only",
    "C_pair_geometry_charge": "geometry_charge",
    "D_pair_geometry_kaon": "geometry_kaon",
    "E_pair_no_kaon": "no_kaon",
    "F_pair_no_charge": "no_charge",
    "G_pair_geometry_momentum": "geometry_momentum",
}


def normalise_pair_feature_set(pair_feature_set: str) -> str:
    if pair_feature_set not in PAIR_FEATURE_SETS:
        allowed = ", ".join(PAIR_FEATURE_SETS)
        raise ValueError(f"Unknown pair feature set {pair_feature_set!r}; allowed: {allowed}")
    return pair_feature_set


def get_included_scalar_features(pair_feature_set: str) -> list[str]:
    pair_feature_set = normalise_pair_feature_set(pair_feature_set)
    names: list[str] = []
    for group in PAIR_FEATURE_SETS[pair_feature_set]:
        names.extend(SCALAR_FEATURE_GROUPS[group])
    return names


def get_included_pair_feature_names(pair_feature_set: str) -> list[str]:
    return get_included_scalar_features(pair_feature_set) + [
        "one_hot_type_i_cpf",
        "one_hot_type_i_npf",
        "one_hot_type_i_sv",
        "one_hot_type_i_v0",
        "one_hot_type_j_cpf",
        "one_hot_type_j_npf",
        "one_hot_type_j_sv",
        "one_hot_type_j_v0",
    ]


def n_pair_features_for_feature_set(pair_feature_set: str) -> int:
    return len(get_included_scalar_features(pair_feature_set)) + 2 * N_TOKEN_TYPES


def build_pair_features(meta: torch.Tensor, pair_feature_set: str = "full") -> torch.Tensor:
    '''Build ordered-pair features, keeping only the requested feature groups.
    Args
        meta (Tensor): Per-token metadata with shape (batch, tokens, features).
        pair_feature_set (str): Name of the scalar feature-group selection.
    Returns
        Tensor: Pairwise feature tensor with shape (batch, tokens, tokens, features).
    '''
    selected = get_included_scalar_features(pair_feature_set)

    theta = meta[..., 0]
    phi = meta[..., 1]
    log_p = meta[..., 2]
    charge = meta[..., 3]
    is_kaon = meta[..., 4]
    token_type = meta[..., 5].long().clamp(0, N_TOKEN_TYPES - 1)

    theta_i = theta.unsqueeze(2)
    theta_j = theta.unsqueeze(1)
    phi_i = phi.unsqueeze(2)
    phi_j = phi.unsqueeze(1)
    log_p_i = log_p.unsqueeze(2)
    log_p_j = log_p.unsqueeze(1)
    charge_i = charge.unsqueeze(2)
    charge_j = charge.unsqueeze(1)
    kaon_i = is_kaon.unsqueeze(2)
    kaon_j = is_kaon.unsqueeze(1)

    dtheta = theta_i - theta_j
    dphi = phi_i - phi_j
    delta_r = torch.sqrt(dtheta.square() + dphi.square() + EPS)
    charge_product = charge_i * charge_j
    same_nonzero_charge = ((charge_i != 0) & (charge_i == charge_j)).to(meta.dtype)
    opposite_charge = ((charge_i * charge_j) < 0).to(meta.dtype)
    both_kaon = kaon_i * kaon_j
    one_kaon = ((kaon_i + kaon_j) > 0).to(meta.dtype)

    scalar_map = {
        "dtheta": dtheta,
        "sin_dphi": torch.sin(dphi),
        "cos_dphi": torch.cos(dphi),
        "log_deltaR": base.safe_log_tensor(delta_r),
        "log_p_i": log_p_i.expand_as(dtheta),
        "log_p_j": log_p_j.expand_as(dtheta),
        "charge_i_times_charge_j": charge_product,
        "same_nonzero_charge": same_nonzero_charge,
        "opposite_charge": opposite_charge,
        "is_kaon_i": kaon_i.expand_as(dtheta),
        "is_kaon_j": kaon_j.expand_as(dtheta),
        "both_kaon": both_kaon,
        "one_kaon": one_kaon,
    }

    type_i = F.one_hot(token_type, num_classes=N_TOKEN_TYPES).to(meta.dtype).unsqueeze(2)
    type_j = F.one_hot(token_type, num_classes=N_TOKEN_TYPES).to(meta.dtype).unsqueeze(1)
    n_tokens = meta.shape[1]
    type_i = type_i.expand(-1, -1, n_tokens, -1)
    type_j = type_j.expand(-1, n_tokens, -1, -1)

    scalar_features = torch.stack([scalar_map[name] for name in selected], dim=-1)
    return torch.cat([scalar_features, type_i, type_j], dim=-1)


class PairBiasBuilder(nn.Module):
    def __init__(self, pair_feature_set: str = "full", nhead: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.pair_feature_set = normalise_pair_feature_set(pair_feature_set)
        self.included_pair_features = get_included_pair_feature_names(self.pair_feature_set)
        n_pair_features = n_pair_features_for_feature_set(self.pair_feature_set)
        self.nhead = nhead
        self.net = nn.Sequential(
            nn.Linear(n_pair_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nhead),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, meta: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        pair_features = build_pair_features(meta, self.pair_feature_set)
        pair_bias = self.net(pair_features).permute(0, 3, 1, 2).contiguous()
        if padding_mask is not None:
            valid_pair = (~padding_mask).unsqueeze(1).unsqueeze(3) & (~padding_mask).unsqueeze(1).unsqueeze(2)
            pair_bias = pair_bias.masked_fill(~valid_pair, 0.0)
        batch, nhead, n_tokens, _ = pair_bias.shape
        return pair_bias.reshape(batch * nhead, n_tokens, n_tokens)


class UDSSpecialistPairBiasTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = base.NUM_CLASSES,
        n_global: int = len(base.GLOBAL_BRANCHES),
        cpf_feat: int = len(base.CPF_BRANCHES),
        npf_feat: int = len(base.NPF_BRANCHES),
        sv_feat: int = len(base.SV_BRANCHES),
        v0_feat: int = len(base.V0_BRANCHES),
        d_model: int = 128,
        nhead: int = 8,
        n_encoder_layers: int = 3,
        dropout: float = 0.1,
        pair_hidden_dim: int = 32,
        pair_feature_set: str = "full",
    ):
        super().__init__()
        self.pair_feature_set = normalise_pair_feature_set(pair_feature_set)
        self.cpf_embed = base.EmbeddingTower(cpf_feat, d_model=d_model, dropout=dropout)
        self.npf_embed = base.EmbeddingTower(npf_feat, d_model=d_model, dropout=dropout)
        self.sv_embed = base.EmbeddingTower(sv_feat, d_model=d_model, dropout=dropout)
        self.v0_embed = base.EmbeddingTower(v0_feat, d_model=d_model, dropout=dropout)
        self.pair_bias_builder = PairBiasBuilder(
            pair_feature_set=self.pair_feature_set,
            nhead=nhead,
            hidden_dim=pair_hidden_dim,
        )
        self.encoder_layers = nn.ModuleList(
            [base.PairBiasTransformerEncoderLayer(d_model, nhead, dropout=dropout) for _ in range(n_encoder_layers)]
        )
        self.pooling = base.AttentionPooling(d_model, dropout=dropout)
        self.global_bn = nn.BatchNorm1d(n_global, eps=1e-3, momentum=0.1)
        cls_dim = n_global + d_model
        self.dense1 = base.LinLayer(cls_dim, cls_dim, dropout_rate=dropout)
        self.dense2 = base.LinLayer(cls_dim, cls_dim, dropout_rate=dropout)
        self.dense3 = base.LinLayer(cls_dim, cls_dim, dropout_rate=dropout)
        self.output = nn.Linear(cls_dim, num_classes)

    def forward(self, global_vars, cpf, npf, vtx, v0):
        cpf_mask = cpf.abs().sum(dim=-1) == 0
        npf_mask = npf.abs().sum(dim=-1) == 0
        vtx_mask = vtx.abs().sum(dim=-1) == 0
        v0_mask = v0.abs().sum(dim=-1) == 0
        padding_mask = torch.cat((cpf_mask, npf_mask, vtx_mask, v0_mask), dim=1)
        meta = base.make_token_metadata(cpf, npf, vtx, v0)
        pair_attn_mask = self.pair_bias_builder(meta, padding_mask=padding_mask)

        global_vars = self.global_bn(global_vars)
        enc = torch.cat(
            (self.cpf_embed(cpf), self.npf_embed(npf), self.sv_embed(vtx), self.v0_embed(v0)),
            dim=1,
        )
        for layer in self.encoder_layers:
            enc = layer(enc, pair_attn_mask=pair_attn_mask, key_padding_mask=padding_mask)
        enc = self.pooling(enc, key_padding_mask=padding_mask)

        x = torch.cat((global_vars, enc), dim=1)
        x = self.dense1(x, x, skip=True)
        x = self.dense2(x, x, skip=True)
        x = self.dense3(x, x, skip=True)
        return self.output(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-files", nargs="+", default=["data/fcc_data.root"])
    parser.add_argument("--tree-name", default="deepntuplizer/tree")
    parser.add_argument("--output-dir", default="runs/uds_pairfeature_ablation/B_pair_geometry_only")
    parser.add_argument("--pair-feature-set", choices=list(PAIR_FEATURE_SETS), default="full")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-fraction", "--val-fraction", dest="eval_fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1123)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--d-model", dest="d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--n-encoder-layers", dest="n_encoder_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pair-hidden-dim", dest="pair_hidden_dim", type=int, default=32)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_loss: float, args, best_eval_acc=None) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_loss": best_loss,
            "best_eval_acc": best_eval_acc,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "class_names": CLASS_NAMES,
            "architecture": "UDSSpecialistPairBiasTransformer",
            "pair_feature_set": args.pair_feature_set,
            "pair_features": get_included_pair_feature_names(args.pair_feature_set),
            "branches": {
                "global": base.GLOBAL_BRANCHES,
                "cpf": base.CPF_BRANCHES,
                "npf": base.NPF_BRANCHES,
                "sv": base.SV_BRANCHES,
                "v0": base.V0_BRANCHES,
                "truth": base.TRUTH_BRANCHES,
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    base.set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    resume_path = Path(args.resume_path) if args.resume_path else output_dir / "checkpoint_latest.pth"

    existing_run = metrics_path.exists() or resume_path.exists()
    if existing_run and not args.resume:
        if not args.overwrite:
            raise SystemExit(
                f"{output_dir} already contains a run. Use --resume to continue it, "
                "choose a new --output-dir, or pass --overwrite to start again."
            )
        for pattern in ("metrics.csv", "training_curve.png", "checkpoint*.pth"):
            for path in output_dir.glob(pattern):
                path.unlink()

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"Inputs: {args.root_files}")
    print("CPF baseline PID feature: RPj_charged_is_Kaon_smearedUniform090")
    print(f"Pair feature set: {args.pair_feature_set}")
    print("Included pair features:", get_included_pair_feature_names(args.pair_feature_set))

    glob, cpf, npf, sv, v0, truth = base.load_data(args.root_files, args.tree_name, args.limit)
    train_idx, eval_idx, test_idx = base.make_split_indices(len(glob), args.eval_fraction, args.test_fraction, args.seed)
    train_uds_idx = base.filter_uds_indices(train_idx, truth)
    eval_uds_idx = base.filter_uds_indices(eval_idx, truth)
    test_uds_idx = base.filter_uds_indices(test_idx, truth)
    uds_truth = base.make_uds_truth(truth)

    print(f"Full split sizes: train={len(train_idx)} eval={len(eval_idx)} test={len(test_idx)}")
    print(f"UDS split sizes: train={len(train_uds_idx)} eval={len(eval_uds_idx)} test={len(test_uds_idx)}")
    print(f"UDS train class counts {CLASS_NAMES}: {uds_truth[train_uds_idx].sum(axis=0).astype(int)}")
    print(f"UDS eval class counts {CLASS_NAMES}: {uds_truth[eval_uds_idx].sum(axis=0).astype(int)}")
    print(f"UDS test class counts {CLASS_NAMES}: {uds_truth[test_uds_idx].sum(axis=0).astype(int)}")

    train_ds = base.JetDataset(glob, cpf, npf, sv, v0, uds_truth, train_uds_idx)
    eval_ds = base.JetDataset(glob, cpf, npf, sv, v0, uds_truth, eval_uds_idx)
    pin_memory = args.device.startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    print(f"Train jets: {len(train_ds)} ({len(train_loader)} batches)")
    print(f"Eval jets:  {len(eval_ds)} ({len(eval_loader)} batches)")
    print(f"Test UDS jets: {len(test_uds_idx)}")

    device = torch.device(args.device)
    model = UDSSpecialistPairBiasTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        n_encoder_layers=args.n_encoder_layers,
        dropout=args.dropout,
        pair_hidden_dim=args.pair_hidden_dim,
        pair_feature_set=args.pair_feature_set,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    print(
        "Hyperparameters: "
        f"lr={args.lr} d_model={args.d_model} nhead={args.nhead} layers={args.n_encoder_layers} "
        f"dropout={args.dropout} pair_hidden_dim={args.pair_hidden_dim} weight_decay={args.weight_decay} "
        f"batch_size={args.batch_size} pair_feature_set={args.pair_feature_set}"
    )

    lr_epochs = max(1, int(args.epochs * 0.3))
    lr_rate = 0.01 ** (1.0 / lr_epochs)
    milestones = list(range(args.epochs - lr_epochs, args.epochs))
    optimizer = base.Ranger(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer.optimizer, milestones=milestones, gamma=lr_rate)

    start_epoch = 1
    best_loss = float("inf")
    if args.resume and resume_path.exists():
        start_epoch, best_loss = base.load_checkpoint(resume_path, model, optimizer, scheduler, device)
        print(f"Resumed from {resume_path} at epoch {start_epoch}; best_loss={best_loss:.6f}")

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs} | lr={current_lr:.8f}")
        train_loss = base.train_one_epoch(model, train_loader, optimizer, device, epoch)
        eval_loss, eval_acc, per_class_acc = base.validate(model, eval_loader, device)
        scheduler.step()

        row = {"epoch": epoch, "lr": current_lr, "train_loss": train_loss, "eval_loss": eval_loss, "eval_acc": eval_acc}
        for name, acc in zip(CLASS_NAMES, per_class_acc):
            row[f"eval_acc_{name}"] = acc
        base.append_metrics(metrics_path, row)

        print(f"  train_loss={train_loss:.6f} eval_loss={eval_loss:.6f} eval_acc={100.0 * eval_acc:.2f}%")
        print("  per-class eval acc:", {n: f"{100.0 * a:.2f}%" for n, a in zip(CLASS_NAMES, per_class_acc)})

        if eval_loss < best_loss:
            best_loss = eval_loss
            save_checkpoint(output_dir / "checkpoint_best_loss.pth", model, optimizer, scheduler, epoch, best_loss, args, best_eval_acc=eval_acc)
            print(f"  new best evaluation loss: {best_loss:.6f}")

        save_checkpoint(output_dir / "checkpoint_latest.pth", model, optimizer, scheduler, epoch, best_loss, args)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch}.pth", model, optimizer, scheduler, epoch, best_loss, args)

        base.plot_training_curve(metrics_path, output_dir / "training_curve.png")


if __name__ == "__main__":
    main()

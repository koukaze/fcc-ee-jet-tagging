#!/usr/bin/env python3
'''Train a UDS specialist with particle-pair attention bias.

The full dataset is split before selecting true s/u/d jets within each subset.
An additive per-head attention bias is learned from pairwise token metadata.
The evaluation subset selects checkpoints; the test subset is used for evaluation.
'''

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uproot
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

STYLE_PATH = Path(__file__).resolve().parents[2] / "styles" / "mphil.mplstyle"
if STYLE_PATH.exists():
    plt.style.use(str(STYLE_PATH))


GLOBAL_BRANCHES = [
    "jets_p",
    "jets_theta",
    "jets_phi",
    "jets_m",
    "jets_e",
    "jets_nRP_charged",
    "jets_nRP_neutral",
    "jets_angularity_00",
    "jets_angularity_105",
    "jets_angularity_11",
    "jets_angularity_12",
    "jets_angularity_20",
]

CPF_BRANCHES = [
    "RPj_charged_mass",
    "RPj_charged_charge",
    "RPj_charged_Z0",
    "RPj_charged_D0",
    "RPj_charged_Z0_sig",
    "RPj_charged_D0_sig",
    "RPj_charged_Curv",
    "RPj_charged_pRel",
    "RPj_charged_eRel",
    "RPj_charged_dTheta",
    "RPj_charged_dPhi",
    "RPj_charged_p_log",
    "RPj_charged_pRel_log",
    "RPj_charged_e_log",
    "RPj_charged_eRel_log",
    "RPj_charged_dAngle",
    "RPj_charged_isMuon",
    "RPj_charged_isElectron",
    "RPj_charged_is_Kaon_smearedUniform090",
]

NPF_BRANCHES = [
    "RPj_neutral_mass",
    "RPj_neutral_pRel",
    "RPj_neutral_eRel",
    "RPj_neutral_dTheta",
    "RPj_neutral_dPhi",
    "RPj_neutral_p_log",
    "RPj_neutral_pRel_log",
    "RPj_neutral_e_log",
    "RPj_neutral_eRel_log",
    "RPj_neutral_dAngle",
    "RPj_neutral_isPhoton",
]

SV_BRANCHES = [
    "sv_mass",
    "sv_p",
    "sv_ntracks",
    "sv_chi2",
    "sv_normchi2",
    "sv_ndf",
    "sv_thetarel",
    "sv_phirel",
    "sv_costhetasvpv",
    "sv_dxy",
    "sv_d3d",
]

V0_BRANCHES = [
    "v0_mass",
    "v0_p",
    "v0_ntracks",
    "v0_chi2",
    "v0_normchi2",
    "v0_ndf",
    "v0_thetarel",
    "v0_phirel",
    "v0_costhetasvpv",
    "v0_dxy",
    "v0_d3d",
]

TRUTH_BRANCHES = ["isB_H", "isC_H", "isS_H", "isU_H", "isD_H", "isG_H"]
FULL_CLASS_NAMES = ["b", "c", "s", "u", "d", "g"]
CLASS_NAMES = ["s", "u", "d"]
UDS_FULL_CLASS_INDICES = [2, 3, 4]

N_CPF = 25
N_NPF = 25
N_SV = 4
N_V0 = 4
NUM_CLASSES = 3
EPS = 1e-6

CPF = {name: i for i, name in enumerate(CPF_BRANCHES)}
NPF = {name: i for i, name in enumerate(NPF_BRANCHES)}
SV = {name: i for i, name in enumerate(SV_BRANCHES)}
V0 = {name: i for i, name in enumerate(V0_BRANCHES)}
TOKEN_TYPES = {"cpf": 0, "npf": 1, "sv": 2, "v0": 3}
N_TOKEN_TYPES = len(TOKEN_TYPES)
N_PAIR_FEATURES = 13 + 2 * N_TOKEN_TYPES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-files",
        nargs="+",
        default=["data/fcc_data.root"],
        help="ROOT files to process. Use data/chunk_*.root on a shell if desired.",
    )
    parser.add_argument("--tree-name", default="deepntuplizer/tree")
    parser.add_argument("--output-dir", default="runs/uds_pairbias/UDS_d128_pairbias")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
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
        help="Fraction reserved for final evaluation plots.",
    )
    parser.add_argument("--seed", type=int, default=1123)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--d-model", dest="d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--n-encoder-layers", dest="n_encoder_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pair-hidden-dim", dest="pair_hidden_dim", type=int, default=32)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional event limit for smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_latest.pth.")
    parser.add_argument("--resume-path", default=None, help="Specific checkpoint to resume from.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing run in output-dir.")
    parser.add_argument("--save-every", type=int, default=0, help="Save epoch checkpoint every N epochs.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jagged_and_pad(tree, branch_names: list[str], max_n: int, entry_stop: int | None = None) -> np.ndarray:
    arrays = tree.arrays(branch_names, entry_stop=entry_stop, library="ak")
    n_jets = len(arrays[branch_names[0]])
    result = np.zeros((n_jets, max_n, len(branch_names)), dtype=np.float32)
    for i_f, bname in enumerate(branch_names):
        padded = ak.fill_none(ak.pad_none(arrays[bname], max_n, clip=True), 0.0)
        result[:, :, i_f] = ak.to_numpy(padded)
    return result


def read_flat(tree, branch_names: list[str], entry_stop: int | None = None) -> np.ndarray:
    arrays = tree.arrays(branch_names, entry_stop=entry_stop, library="np")
    return np.column_stack([arrays[b].astype(np.float32) for b in branch_names])


def make_truth(tree, entry_stop: int | None = None) -> np.ndarray:
    arrays = tree.arrays(TRUTH_BRANCHES, entry_stop=entry_stop, library="np")
    return np.column_stack([arrays[b].astype(np.float32) for b in TRUTH_BRANCHES])


def load_one_file(path: Path, tree_name: str, entry_stop: int | None = None) -> tuple[np.ndarray, ...]:
    print(f"Loading {path}")
    tree = uproot.open(path)[tree_name]
    print(f"  entries: {tree.num_entries}")
    return (
        read_flat(tree, GLOBAL_BRANCHES, entry_stop),
        read_jagged_and_pad(tree, CPF_BRANCHES, N_CPF, entry_stop),
        read_jagged_and_pad(tree, NPF_BRANCHES, N_NPF, entry_stop),
        read_jagged_and_pad(tree, SV_BRANCHES, N_SV, entry_stop),
        read_jagged_and_pad(tree, V0_BRANCHES, N_V0, entry_stop),
        make_truth(tree, entry_stop),
    )


def load_data(root_files: list[str], tree_name: str, limit: int | None = None) -> tuple[np.ndarray, ...]:
    remaining = limit
    chunks: list[tuple[np.ndarray, ...]] = []

    for root_file in root_files:
        path = Path(root_file)
        if remaining is not None and remaining <= 0:
            break
        entry_stop = remaining
        data = load_one_file(path, tree_name, entry_stop=entry_stop)
        chunks.append(data)
        if remaining is not None:
            remaining -= len(data[0])

    merged = tuple(np.concatenate([chunk[i] for chunk in chunks], axis=0) for i in range(6))
    names = ["global", "cpf", "npf", "sv", "v0", "truth"]
    for name, arr in zip(names, merged):
        bad = ~np.isfinite(arr)
        if np.any(bad):
            print(f"  replacing {bad.sum()} non-finite values in {name}")
            arr[bad] = 0.0
        print(f"  {name}: {arr.shape}, {arr.dtype}")
    print(f"  full class counts {FULL_CLASS_NAMES}: {merged[-1].sum(axis=0).astype(int)}")
    return merged


def make_split_indices(
    n_total: int,
    eval_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if eval_fraction < 0.0 or test_fraction < 0.0 or eval_fraction + test_fraction >= 1.0:
        raise ValueError(
            "eval_fraction and test_fraction must be non-negative and sum to less than 1."
        )

    n_eval = int(round(n_total * eval_fraction))
    n_test = int(round(n_total * test_fraction))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)
    eval_idx = indices[:n_eval]
    test_idx = indices[n_eval : n_eval + n_test]
    train_idx = indices[n_eval + n_test :]
    return train_idx, eval_idx, test_idx


def make_uds_truth(full_truth: np.ndarray) -> np.ndarray:
    '''Return one-hot s/u/d labels from the full b/c/s/u/d/g truth matrix.
    Args
        full_truth (ndarray): One-hot six-class truth matrix.
    Returns
        ndarray: One-hot UDS truth matrix.
    '''
    return full_truth[:, UDS_FULL_CLASS_INDICES].astype(np.float32)


def filter_uds_indices(indices: np.ndarray, full_truth: np.ndarray) -> np.ndarray:
    '''Filter a pre-existing full-dataset split to true s/u/d jets.
    Args
        indices (ndarray): Candidate row indices from the full-dataset split.
        full_truth (ndarray): One-hot six-class truth matrix.
    Returns
        ndarray: Indices whose truth label is s, u, or d.
    '''
    uds_mask = full_truth[indices][:, UDS_FULL_CLASS_INDICES].sum(axis=1) > 0.5
    return indices[uds_mask]


class JetDataset(Dataset):
    def __init__(self, glob, cpf, npf, sv, v0, truth, indices):
        self.glob = torch.from_numpy(glob[indices])
        self.cpf = torch.from_numpy(cpf[indices])
        self.npf = torch.from_numpy(npf[indices])
        self.sv = torch.from_numpy(sv[indices])
        self.v0 = torch.from_numpy(v0[indices])
        self.truth = torch.from_numpy(truth[indices])

    def __len__(self) -> int:
        return len(self.glob)

    def __getitem__(self, idx):
        return self.glob[idx], self.cpf[idx], self.npf[idx], self.sv[idx], self.v0[idx], self.truth[idx]


# Architecture and feature inputs adapted from DeepJetFCC.
# See THIRD_PARTY_NOTICES.md for the upstream revision and attribution.
class InputConv(nn.Module):
    def __init__(self, in_chn, out_chn, dropout_rate=0.1):
        super().__init__()
        self.lin = nn.Conv1d(in_chn, out_chn, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_chn, eps=1e-3, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(out_chn, eps=1e-3, momentum=0.1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sc, skip=True):
        x2 = self.dropout(self.bn1(self.act(self.lin(x))))
        return self.bn2(sc + x2) if skip else self.bn2(x2)


class LinLayer(nn.Module):
    def __init__(self, in_chn, out_chn, dropout_rate=0.1):
        super().__init__()
        self.lin = nn.Linear(in_chn, out_chn)
        self.bn1 = nn.BatchNorm1d(out_chn, eps=1e-3, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(out_chn, eps=1e-3, momentum=0.1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sc, skip=True):
        x2 = self.dropout(self.bn1(self.act(self.lin(x))))
        return self.bn2(sc + x2) if skip else self.bn2(x2)


class EmbeddingTower(nn.Module):
    def __init__(self, n_in, d_model=128, dropout=0.1):
        super().__init__()
        hidden = min(64, d_model)
        self.bn0 = nn.BatchNorm1d(n_in, eps=1e-3, momentum=0.1)
        self.conv1 = InputConv(n_in, hidden, dropout_rate=dropout)
        self.conv2 = InputConv(hidden, d_model, dropout_rate=dropout)
        self.conv3 = InputConv(d_model, d_model, dropout_rate=dropout)
        self.skip_proj = nn.Conv1d(hidden, d_model, kernel_size=1)
        self.skip_bn = nn.BatchNorm1d(d_model, eps=1e-3, momentum=0.1)
        self.skip_act = nn.ReLU()
        self.skip_drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.bn0(x.transpose(1, 2))
        x = self.conv1(x, x, skip=False)
        sc = self.skip_drop(self.skip_bn(self.skip_act(self.skip_proj(x))))
        x = self.conv2(x, sc, skip=True)
        x = self.conv3(x, x, skip=True)
        return x.transpose(1, 2)


class AttentionPooling(nn.Module):
    def __init__(self, d_model=128, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(d_model, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm1d(d_model, eps=1e-3, momentum=0.1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        scores = self.conv(x.transpose(1, 2)).squeeze(1)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, -1e9)
        weights = self.softmax(scores).unsqueeze(1)
        y = torch.matmul(weights, x).squeeze(1)
        return self.dropout(self.bn(self.act(y)))


def safe_log_tensor(x, eps=EPS):
    return torch.log(torch.clamp(x, min=eps))


def make_token_metadata(cpf, npf, sv, v0):
    '''Build compact per-token metadata used by the pair-bias MLP.
    Args
        cpf (Tensor): Charged-particle features.
        npf (Tensor): Neutral-particle features.
        sv (Tensor): Secondary-vertex features.
        v0 (Tensor): V0 features.
    Returns
        Tensor: Concatenated per-token metadata.
    '''
    device = cpf.device
    dtype = cpf.dtype

    cpf_meta = torch.stack(
        [
            cpf[..., CPF["RPj_charged_dTheta"]],
            cpf[..., CPF["RPj_charged_dPhi"]],
            cpf[..., CPF["RPj_charged_p_log"]],
            cpf[..., CPF["RPj_charged_charge"]],
            cpf[..., CPF["RPj_charged_is_Kaon_smearedUniform090"]],
            torch.full(cpf.shape[:2], TOKEN_TYPES["cpf"], device=device, dtype=dtype),
        ],
        dim=-1,
    )
    npf_meta = torch.stack(
        [
            npf[..., NPF["RPj_neutral_dTheta"]],
            npf[..., NPF["RPj_neutral_dPhi"]],
            npf[..., NPF["RPj_neutral_p_log"]],
            torch.zeros(npf.shape[:2], device=device, dtype=dtype),
            torch.zeros(npf.shape[:2], device=device, dtype=dtype),
            torch.full(npf.shape[:2], TOKEN_TYPES["npf"], device=device, dtype=dtype),
        ],
        dim=-1,
    )
    sv_meta = torch.stack(
        [
            sv[..., SV["sv_thetarel"]],
            sv[..., SV["sv_phirel"]],
            safe_log_tensor(sv[..., SV["sv_p"]]),
            torch.zeros(sv.shape[:2], device=device, dtype=dtype),
            torch.zeros(sv.shape[:2], device=device, dtype=dtype),
            torch.full(sv.shape[:2], TOKEN_TYPES["sv"], device=device, dtype=dtype),
        ],
        dim=-1,
    )
    v0_meta = torch.stack(
        [
            v0[..., V0["v0_thetarel"]],
            v0[..., V0["v0_phirel"]],
            safe_log_tensor(v0[..., V0["v0_p"]]),
            torch.zeros(v0.shape[:2], device=device, dtype=dtype),
            torch.zeros(v0.shape[:2], device=device, dtype=dtype),
            torch.full(v0.shape[:2], TOKEN_TYPES["v0"], device=device, dtype=dtype),
        ],
        dim=-1,
    )
    return torch.cat([cpf_meta, npf_meta, sv_meta, v0_meta], dim=1)


def build_pair_features(meta):
    '''Build ordered-pair geometry, PID, charge, and token-type features.
    Args
        meta (Tensor): Per-token metadata with shape (batch, tokens, features).
    Returns
        Tensor: Pairwise feature tensor with shape (batch, tokens, tokens, features).
    '''
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

    type_i = F.one_hot(token_type, num_classes=N_TOKEN_TYPES).to(meta.dtype).unsqueeze(2)
    type_j = F.one_hot(token_type, num_classes=N_TOKEN_TYPES).to(meta.dtype).unsqueeze(1)
    n_tokens = meta.shape[1]
    type_i = type_i.expand(-1, -1, n_tokens, -1)
    type_j = type_j.expand(-1, n_tokens, -1, -1)

    scalar_features = torch.stack(
        [
            dtheta,
            torch.sin(dphi),
            torch.cos(dphi),
            safe_log_tensor(delta_r),
            log_p_i.expand_as(dtheta),
            log_p_j.expand_as(dtheta),
            charge_product,
            same_nonzero_charge,
            opposite_charge,
            kaon_i.expand_as(dtheta),
            kaon_j.expand_as(dtheta),
            both_kaon,
            one_kaon,
        ],
        dim=-1,
    )
    return torch.cat([scalar_features, type_i, type_j], dim=-1)


class PairBiasBuilder(nn.Module):
    def __init__(self, n_pair_features=N_PAIR_FEATURES, nhead=8, hidden_dim=32):
        super().__init__()
        self.nhead = nhead
        self.net = nn.Sequential(
            nn.Linear(n_pair_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nhead),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, meta, padding_mask=None):
        pair_features = build_pair_features(meta)
        pair_bias = self.net(pair_features).permute(0, 3, 1, 2).contiguous()
        if padding_mask is not None:
            valid_pair = (~padding_mask).unsqueeze(1).unsqueeze(3) & (~padding_mask).unsqueeze(1).unsqueeze(2)
            pair_bias = pair_bias.masked_fill(~valid_pair, 0.0)
        batch, nhead, n_tokens, _ = pair_bias.shape
        return pair_bias.reshape(batch * nhead, n_tokens, n_tokens)


class PairBiasTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=128, nhead=8, dropout=0.1):
        super().__init__()
        self.input_conv = InputConv(d_model, d_model, dropout_rate=dropout)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout0 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, src, pair_attn_mask=None, key_padding_mask=None):
        src = self.input_conv(src.transpose(1, 2), src.transpose(1, 2), skip=True).transpose(1, 2)

        attn_key_padding_mask = key_padding_mask
        if key_padding_mask is not None and pair_attn_mask is not None and pair_attn_mask.dtype.is_floating_point:
            attn_key_padding_mask = torch.zeros(
                key_padding_mask.shape,
                dtype=src.dtype,
                device=src.device,
            ).masked_fill(key_padding_mask, float("-inf"))

        src2 = self.self_attn(
            src,
            src,
            src,
            attn_mask=pair_attn_mask,
            key_padding_mask=attn_key_padding_mask,
            need_weights=False,
        )[0]
        src = self.norm0(src + src2)
        src2 = self.linear2(self.dropout0(self.activation(self.linear1(src))))
        return self.norm1(src + src2)


class UDSSpecialistPairBiasTransformer(nn.Module):
    def __init__(
        self,
        num_classes=NUM_CLASSES,
        n_global=len(GLOBAL_BRANCHES),
        cpf_feat=len(CPF_BRANCHES),
        npf_feat=len(NPF_BRANCHES),
        sv_feat=len(SV_BRANCHES),
        v0_feat=len(V0_BRANCHES),
        d_model=128,
        nhead=8,
        n_encoder_layers=3,
        dropout=0.1,
        pair_hidden_dim=32,
    ):
        super().__init__()
        self.cpf_embed = EmbeddingTower(cpf_feat, d_model=d_model, dropout=dropout)
        self.npf_embed = EmbeddingTower(npf_feat, d_model=d_model, dropout=dropout)
        self.sv_embed = EmbeddingTower(sv_feat, d_model=d_model, dropout=dropout)
        self.v0_embed = EmbeddingTower(v0_feat, d_model=d_model, dropout=dropout)
        self.pair_bias_builder = PairBiasBuilder(nhead=nhead, hidden_dim=pair_hidden_dim)
        self.encoder_layers = nn.ModuleList(
            [PairBiasTransformerEncoderLayer(d_model, nhead, dropout=dropout) for _ in range(n_encoder_layers)]
        )
        self.pooling = AttentionPooling(d_model, dropout=dropout)
        self.global_bn = nn.BatchNorm1d(n_global, eps=1e-3, momentum=0.1)
        cls_dim = n_global + d_model
        self.dense1 = LinLayer(cls_dim, cls_dim, dropout_rate=dropout)
        self.dense2 = LinLayer(cls_dim, cls_dim, dropout_rate=dropout)
        self.dense3 = LinLayer(cls_dim, cls_dim, dropout_rate=dropout)
        self.output = nn.Linear(cls_dim, num_classes)

    def forward(self, global_vars, cpf, npf, vtx, v0):
        cpf_mask = cpf.abs().sum(dim=-1) == 0
        npf_mask = npf.abs().sum(dim=-1) == 0
        vtx_mask = vtx.abs().sum(dim=-1) == 0
        v0_mask = v0.abs().sum(dim=-1) == 0
        padding_mask = torch.cat((cpf_mask, npf_mask, vtx_mask, v0_mask), dim=1)
        meta = make_token_metadata(cpf, npf, vtx, v0)
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


# RAdam implementation adapted from Liyuan Liu (2019), Apache-2.0.
# Modified optimizer integration; see THIRD_PARTY_NOTICES.md.
class RAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                p_fp32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                state["step"] += 1

                buf = group["buffer"][int(state["step"] % 10)]
                if state["step"] == buf[0]:
                    n_sma, step_size = buf[1], buf[2]
                else:
                    buf[0] = state["step"]
                    beta2_t = beta2**state["step"]
                    n_sma_max = 2 / (1 - beta2) - 1
                    n_sma = n_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buf[1] = n_sma
                    if n_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (n_sma - 4)
                            / (n_sma_max - 4)
                            * (n_sma - 2)
                            / n_sma
                            * n_sma_max
                            / (n_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    buf[2] = step_size

                if group["weight_decay"] != 0:
                    p_fp32.add_(p_fp32, alpha=-group["weight_decay"] * group["lr"])
                if n_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_fp32.addcdiv_(exp_avg, denom, value=-step_size * group["lr"])
                elif step_size > 0:
                    p_fp32.add_(exp_avg, alpha=-step_size * group["lr"])
                p.data.copy_(p_fp32)
        return loss


# Lookahead implementation adapted from lonePatient/lookahead_pytorch (MIT).
# Modified checkpoint state handling; see THIRD_PARTY_NOTICES.md.
class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer, alpha=0.5, k=6):
        self.optimizer = optimizer
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        self.param_groups = optimizer.param_groups
        self.state = defaultdict(dict)
        for group in optimizer.param_groups:
            for p in group["params"]:
                self.state[p]["cached_params"] = p.data.clone()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        slow_state = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                slow_state.append(self.state[p]["cached_params"])
        return {
            "optimizer": self.optimizer.state_dict(),
            "slow_state": slow_state,
            "step_counter": self.step_counter,
        }

    def load_state_dict(self, state_dict):
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.step_counter = state_dict.get("step_counter", 0)
            slow_state = iter(state_dict.get("slow_state", []))
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    cached = next(slow_state, None)
                    self.state[p]["cached_params"] = p.data.clone() if cached is None else cached.to(p.device)
        else:
            self.optimizer.load_state_dict(state_dict)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.step_counter += 1
        if self.step_counter >= self.k:
            self.step_counter = 0
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    cached = self.state[p]["cached_params"]
                    p.data.mul_(self.alpha).add_(cached, alpha=1.0 - self.alpha)
                    cached.copy_(p.data)
        return loss


def Ranger(params, lr=1e-3, betas=(0.95, 0.999), eps=1e-5, weight_decay=0):
    return Lookahead(RAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))


def cross_entropy_one_hot(pred, target):
    labels = target.argmax(dim=1)
    return nn.CrossEntropyLoss()(pred, labels)


def move_batch(batch, device):
    return [x.to(device, non_blocking=True) for x in batch]


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]")
    for batch in pbar:
        glob, cpf, npf, sv, v0, y = move_batch(batch, device)
        pred = model(glob, cpf, npf, sv, v0)
        loss = cross_entropy_one_hot(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size
        pbar.set_postfix(loss=f"{total_loss / total:.6f}")
    return total_loss / total


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_class_total = np.zeros(NUM_CLASSES, dtype=np.int64)

    for batch in loader:
        glob, cpf, npf, sv, v0, y = move_batch(batch, device)
        pred = model(glob, cpf, npf, sv, v0)
        loss = cross_entropy_one_hot(pred, y)
        labels = y.argmax(dim=1)
        preds = pred.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (preds == labels).sum().item()
        total += batch_size
        for cls in range(NUM_CLASSES):
            mask = labels == cls
            per_class_total[cls] += mask.sum().item()
            per_class_correct[cls] += ((preds == cls) & mask).sum().item()

    per_class_acc = per_class_correct.astype(np.float64) / np.maximum(per_class_total, 1)
    return total_loss / total, correct / total, per_class_acc


def append_metrics(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_existing_metrics(path: Path) -> tuple[list[int], list[float], list[float]]:
    if not path.exists():
        return [], [], []
    epochs, train_losses, val_losses = [], [], []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
    return epochs, train_losses, val_losses


def plot_training_curve(metrics_path: Path, out_path: Path) -> None:
    if not metrics_path.exists():
        return

    epochs, train_losses, eval_losses, eval_accs = [], [], [], []
    with metrics_path.open() as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            eval_losses.append(float(row.get("eval_loss", row.get("val_loss"))))
            eval_accs.append(float(row.get("eval_acc", row.get("val_acc"))))

    if not epochs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_losses, "b-o", label="Train", markersize=3)
    ax1.plot(epochs, eval_losses, "r-o", label="Evaluation", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(False)
    ax1.legend()

    ax2.plot(epochs, np.array(eval_accs) * 100.0, "g-o", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Evaluation accuracy [%]")
    ax2.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_loss: float,
    args,
    best_eval_acc: float | None = None,
) -> None:
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
            "pair_features": [
                "dtheta",
                "sin_dphi",
                "cos_dphi",
                "log_deltaR",
                "log_p_i",
                "log_p_j",
                "charge_i_times_charge_j",
                "same_nonzero_charge",
                "opposite_charge",
                "is_kaon_i",
                "is_kaon_j",
                "both_kaon",
                "one_kaon",
                "one_hot_type_i",
                "one_hot_type_j",
            ],
            "branches": {
                "global": GLOBAL_BRANCHES,
                "cpf": CPF_BRANCHES,
                "npf": NPF_BRANCHES,
                "sv": SV_BRANCHES,
                "v0": V0_BRANCHES,
                "truth": TRUTH_BRANCHES,
            },
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer, scheduler, device) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_loss", float("inf")))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
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
    print(f"CPF baseline PID feature: RPj_charged_is_Kaon_smearedUniform090")

    glob, cpf, npf, sv, v0, truth = load_data(args.root_files, args.tree_name, args.limit)

    train_idx, eval_idx, test_idx = make_split_indices(
        len(glob),
        args.eval_fraction,
        args.test_fraction,
        args.seed,
    )

    train_uds_idx = filter_uds_indices(train_idx, truth)
    eval_uds_idx = filter_uds_indices(eval_idx, truth)
    test_uds_idx = filter_uds_indices(test_idx, truth)
    uds_truth = make_uds_truth(truth)

    print(
        "Full split sizes: "
        f"train={len(train_idx)} eval={len(eval_idx)} test={len(test_idx)}"
    )
    print(
        "UDS split sizes: "
        f"train={len(train_uds_idx)} eval={len(eval_uds_idx)} test={len(test_uds_idx)}"
    )
    print(f"UDS train class counts {CLASS_NAMES}: {uds_truth[train_uds_idx].sum(axis=0).astype(int)}")
    print(f"UDS eval class counts {CLASS_NAMES}: {uds_truth[eval_uds_idx].sum(axis=0).astype(int)}")
    print(f"UDS test class counts {CLASS_NAMES}: {uds_truth[test_uds_idx].sum(axis=0).astype(int)}")

    train_ds = JetDataset(glob, cpf, npf, sv, v0, uds_truth, train_uds_idx)
    eval_ds = JetDataset(glob, cpf, npf, sv, v0, uds_truth, eval_uds_idx)
    pin_memory = args.device.startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
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
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    print(
        "Hyperparameters: "
        f"lr={args.lr} d_model={args.d_model} nhead={args.nhead} "
        f"layers={args.n_encoder_layers} dropout={args.dropout} "
        f"pair_hidden_dim={args.pair_hidden_dim} weight_decay={args.weight_decay} "
        f"batch_size={args.batch_size}"
    )

    lr_epochs = max(1, int(args.epochs * 0.3))
    lr_rate = 0.01 ** (1.0 / lr_epochs)
    milestones = list(range(args.epochs - lr_epochs, args.epochs))
    optimizer = Ranger(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer.optimizer,
        milestones=milestones,
        gamma=lr_rate,
    )

    start_epoch = 1
    best_loss = float("inf")
    if args.resume and resume_path.exists():
        start_epoch, best_loss = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        print(f"Resumed from {resume_path} at epoch {start_epoch}; best_loss={best_loss:.6f}")

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs} | lr={current_lr:.8f}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        eval_loss, eval_acc, per_class_acc = validate(model, eval_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
        }
        for name, acc in zip(CLASS_NAMES, per_class_acc):
            row[f"eval_acc_{name}"] = acc
        append_metrics(metrics_path, row)

        print(
            f"  train_loss={train_loss:.6f} eval_loss={eval_loss:.6f} "
            f"eval_acc={100.0 * eval_acc:.2f}%"
        )
        print("  per-class eval acc:", {n: f"{100.0 * a:.2f}%" for n, a in zip(CLASS_NAMES, per_class_acc)})

        if eval_loss < best_loss:
            best_loss = eval_loss
            save_checkpoint(
                output_dir / "checkpoint_best_loss.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_loss,
                args,
                best_eval_acc=eval_acc,
            )
            print(f"  new best evaluation loss: {best_loss:.6f}")

        save_checkpoint(
            output_dir / "checkpoint_latest.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_loss,
            args,
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch}.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_loss,
                args,
            )

        plot_training_curve(metrics_path, output_dir / "training_curve.png")


if __name__ == "__main__":
    main()

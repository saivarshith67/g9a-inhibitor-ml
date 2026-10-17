"""Chemprop-style directed message-passing neural network (D-MPNN) in PyTorch.

Trains from SMILES graphs built with RDKit. No SMOTE — uses pos_weight for
class imbalance. Intentionally self-contained to avoid chemprop v1/v2 API churn.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
from torch.utils.data import DataLoader, Dataset

from g9a_ml.metrics import binary_metrics, confusion_and_report, save_metrics_json
from g9a_ml.models.mol_splits import SplitName, split_molecule_frame

RDLogger.DisableLog("rdApp.*")

ATOM_LIST = list(range(1, 37))  # H..Kr one-hot coverage for drug-like atoms
HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def _one_hot(x, choices: list) -> list[float]:
    return [1.0 if x == c else 0.0 for c in choices]


def atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetAtomicNum(), ATOM_LIST)
        + _one_hot(atom.GetTotalDegree(), list(range(6)))
        + _one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
        + _one_hot(atom.GetHybridization(), HYBRIDIZATION)
        + [float(atom.GetIsAromatic()), float(atom.GetTotalNumHs(includeNeighbors=True) > 0)]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    bt = bond.GetBondType()
    return [
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


@dataclass
class MolGraph:
    atom_f: torch.Tensor  # [n_atoms, a_dim]
    bond_f: torch.Tensor  # [n_bonds, b_dim] directed
    a2b: torch.Tensor  # [n_bonds] atom index of bond start
    b2a: torch.Tensor  # [n_bonds] atom index of bond end
    b2revb: torch.Tensor  # [n_bonds] reverse directed bond index
    n_atoms: int
    n_bonds: int


def smiles_to_graph(smiles: str) -> MolGraph | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_f = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    if atom_f.shape[0] == 0:
        return None

    bond_f_list: list[list[float]] = []
    a2b: list[int] = []
    b2a: list[int] = []
    b2revb: list[int] = []
    undirected = list(mol.GetBonds())
    for i, bond in enumerate(undirected):
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        # forward
        bond_f_list.append(feat)
        a2b.append(a)
        b2a.append(b)
        b2revb.append(2 * i + 1)
        # reverse
        bond_f_list.append(feat)
        a2b.append(b)
        b2a.append(a)
        b2revb.append(2 * i)

    if not bond_f_list:
        # single-atom molecule: fake self-loop with zeros
        bdim = 6
        bond_f = torch.zeros((1, bdim), dtype=torch.float32)
        return MolGraph(
            atom_f=atom_f,
            bond_f=bond_f,
            a2b=torch.tensor([0], dtype=torch.long),
            b2a=torch.tensor([0], dtype=torch.long),
            b2revb=torch.tensor([0], dtype=torch.long),
            n_atoms=atom_f.shape[0],
            n_bonds=1,
        )

    bond_f = torch.tensor(bond_f_list, dtype=torch.float32)
    return MolGraph(
        atom_f=atom_f,
        bond_f=bond_f,
        a2b=torch.tensor(a2b, dtype=torch.long),
        b2a=torch.tensor(b2a, dtype=torch.long),
        b2revb=torch.tensor(b2revb, dtype=torch.long),
        n_atoms=atom_f.shape[0],
        n_bonds=bond_f.shape[0],
    )


class MolDataset(Dataset):
    def __init__(self, smiles: list[str], labels: list[int], extra: np.ndarray | None = None):
        self.graphs: list[MolGraph] = []
        self.labels: list[float] = []
        self.extra: list[torch.Tensor | None] = []
        for i, (s, y) in enumerate(zip(smiles, labels)):
            g = smiles_to_graph(s)
            if g is None:
                continue
            self.graphs.append(g)
            self.labels.append(float(y))
            if extra is None:
                self.extra.append(None)
            else:
                self.extra.append(torch.tensor(extra[i], dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return self.graphs[idx], self.labels[idx], self.extra[idx]


def collate_graphs(batch):
    graphs, labels, extras = zip(*batch)
    return list(graphs), torch.tensor(labels, dtype=torch.float32), list(extras)


class DMPNNEncoder(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, hidden: int = 300, depth: int = 3, dropout: float = 0.05):
        super().__init__()
        self.depth = depth
        self.dropout = dropout
        self.W_i = nn.Linear(atom_dim + bond_dim, hidden, bias=False)
        self.W_h = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(atom_dim + hidden, hidden)

    def forward(self, graph: MolGraph) -> torch.Tensor:
        # input messages on directed bonds
        x_atom = graph.atom_f  # [n_a, a]
        f_bond = graph.bond_f  # [n_b, b]
        a_start = graph.a2b
        a_end = graph.b2a
        rev = graph.b2revb

        input_msg = self.W_i(torch.cat([x_atom[a_start], f_bond], dim=-1))  # [n_b, h]
        msg = F.relu(input_msg)

        for _ in range(self.depth - 1):
            # sum incoming messages to each atom, then subtract reverse for each bond
            n_a = graph.n_atoms
            agg = torch.zeros(n_a, msg.shape[1], device=msg.device)
            agg.index_add_(0, a_end, msg)
            msg = F.relu(input_msg + self.W_h(agg[a_start] - msg[rev]))
            msg = F.dropout(msg, p=self.dropout, training=self.training)

        # atom readout: atom features + sum of incoming bond messages
        n_a = graph.n_atoms
        incoming = torch.zeros(n_a, msg.shape[1], device=msg.device)
        incoming.index_add_(0, a_end, msg)
        atom_h = F.relu(self.W_o(torch.cat([x_atom, incoming], dim=-1)))
        atom_h = F.dropout(atom_h, p=self.dropout, training=self.training)
        return atom_h.mean(dim=0)  # molecule vector


class DMPNNClassifier(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden: int = 300,
        depth: int = 3,
        dropout: float = 0.05,
        ffn_hidden: int = 300,
        extra_dim: int = 0,
    ):
        super().__init__()
        self.encoder = DMPNNEncoder(atom_dim, bond_dim, hidden=hidden, depth=depth, dropout=dropout)
        in_dim = hidden + extra_dim
        self.ffn = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, 1),
        )

    def forward(self, graph: MolGraph, extra: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(graph)
        if extra is not None:
            h = torch.cat([h, extra], dim=-1)
        return self.ffn(h).squeeze(-1)


@dataclass
class DMPNNConfig:
    hidden: int = 300
    depth: int = 3
    dropout: float = 0.05
    ffn_hidden: int = 300
    epochs: int = 30
    batch_size: int = 50
    lr: float = 1e-4
    use_solubility: bool = False
    patience: int = 5
    num_workers: int = 0


def _probe_dims() -> tuple[int, int]:
    g = smiles_to_graph("CCO")
    assert g is not None
    return g.atom_f.shape[1], g.bond_f.shape[1]


def _extra_matrix(df: pd.DataFrame, use_solubility: bool) -> np.ndarray | None:
    if not use_solubility:
        return None
    col = "Solubility_at_pH_7_4"
    if col not in df.columns:
        raise KeyError(f"{col} required when use_solubility=True")
    vals = df[col].astype(float).fillna(df[col].astype(float).median()).values.reshape(-1, 1)
    # simple standardize
    mean, std = vals.mean(), vals.std() + 1e-8
    return (vals - mean) / std


@torch.no_grad()
def _predict_loader(model: DMPNNClassifier, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[float] = []
    labels: list[float] = []
    for graphs, y, extras in loader:
        for g, yi, ex in zip(graphs, y, extras):
            g = _graph_to_device(g, device)
            ex_t = None if ex is None else ex.to(device)
            logit = model(g, ex_t)
            probs.append(float(torch.sigmoid(logit).cpu()))
            labels.append(float(yi))
    return np.asarray(labels), np.asarray(probs)


def _graph_to_device(g: MolGraph, device: torch.device) -> MolGraph:
    return MolGraph(
        atom_f=g.atom_f.to(device),
        bond_f=g.bond_f.to(device),
        a2b=g.a2b.to(device),
        b2a=g.b2a.to(device),
        b2revb=g.b2revb.to(device),
        n_atoms=g.n_atoms,
        n_bonds=g.n_bonds,
    )


def train_dmpnn_on_frame(
    df: pd.DataFrame,
    split: SplitName,
    cfg: DMPNNConfig,
    smiles_col: str = "SMILES",
    target_col: str = "target",
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 5,
    device: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    train_df, val_df, test_df = split_molecule_frame(
        df,
        split=split,
        smiles_col=smiles_col,
        target_col=target_col,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    extra_tr = _extra_matrix(train_df, cfg.use_solubility)
    extra_va = _extra_matrix(val_df, cfg.use_solubility)
    extra_te = _extra_matrix(test_df, cfg.use_solubility)
    # recompute standardization on train only for val/test
    if cfg.use_solubility:
        col = "Solubility_at_pH_7_4"
        mean = float(train_df[col].astype(float).mean())
        std = float(train_df[col].astype(float).std()) + 1e-8

        def _std(frame: pd.DataFrame) -> np.ndarray:
            v = frame[col].astype(float).fillna(mean).values.reshape(-1, 1)
            return (v - mean) / std

        extra_tr, extra_va, extra_te = _std(train_df), _std(val_df), _std(test_df)

    train_ds = MolDataset(train_df[smiles_col].tolist(), train_df[target_col].tolist(), extra_tr)
    val_ds = MolDataset(val_df[smiles_col].tolist(), val_df[target_col].tolist(), extra_va)
    test_ds = MolDataset(test_df[smiles_col].tolist(), test_df[target_col].tolist(), extra_te)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_graphs, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_graphs, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_graphs, num_workers=cfg.num_workers
    )

    y_tr = np.asarray(train_ds.labels)
    n_pos = float((y_tr == 1).sum())
    n_neg = float((y_tr == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)

    atom_dim, bond_dim = _probe_dims()
    extra_dim = 1 if cfg.use_solubility else 0
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DMPNNClassifier(
        atom_dim,
        bond_dim,
        hidden=cfg.hidden,
        depth=cfg.depth,
        dropout=cfg.dropout,
        ffn_hidden=cfg.ffn_hidden,
        extra_dim=extra_dim,
    ).to(dev)
    pos_weight = pos_weight.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = -1.0
    best_state = None
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for graphs, y, extras in train_loader:
            opt.zero_grad()
            logits = []
            for g, ex in zip(graphs, extras):
                g = _graph_to_device(g, dev)
                ex_t = None if ex is None else ex.to(dev)
                logits.append(model(g, ex_t))
            logit = torch.stack(logits)
            y = y.to(dev)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(graphs)
            n_seen += len(graphs)

        y_val, p_val = _predict_loader(model, val_loader, dev)
        pred_val = (p_val >= 0.5).astype(int)
        val_metrics = binary_metrics(y_val, pred_val, y_score=p_val)
        row = {"epoch": epoch, "train_loss": round(total_loss / max(n_seen, 1), 4), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        score = val_metrics.get("pr_auc", val_metrics.get("roc_auc", 0.0))
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_prob = _predict_loader(model, test_loader, dev)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = binary_metrics(y_true, y_pred, y_score=y_prob)
    cm, report = confusion_and_report(y_true, y_pred)

    result: dict[str, Any] = {
        "model": "dmpnn",
        "split": split,
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "pos_weight": float(n_neg / max(n_pos, 1.0)),
        "device": str(dev),
        "config": asdict(cfg),
        "history": history,
        "best_val_score": best_val,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "atom_dim": atom_dim, "bond_dim": bond_dim}, out_dir / f"dmpnn_{split}.pt")
        save_metrics_json(result, out_dir / f"dmpnn_{split}_metrics.json")
        with open(out_dir / f"dmpnn_{split}_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return result

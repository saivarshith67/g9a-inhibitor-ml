#!/usr/bin/env python
"""Step 11 — compare RF baseline vs D-MPNN vs ChemBERTa."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g9a_ml.paths import ensure_dirs, load_config


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    metrics_dir = cfg["metrics_dir"]
    figures_dir = cfg["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    rf_path = metrics_dir / "with_solubility_smote_final_model.json"
    rf = _load_json(rf_path)
    if rf and "metrics" in rf:
        m = rf["metrics"]
        rows.append(
            {
                "model": "RF_5feat_SMOTE",
                "split": "random_paper",
                "protocol": "tabular+SMOTE",
                "accuracy": m.get("accuracy"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "roc_auc": m.get("roc_auc"),
                "pr_auc": m.get("pr_auc"),
            }
        )

    for model_key, prefix in [("DMPNN", "chemprop"), ("ChemBERTa", "chemberta")]:
        for split in ("scaffold", "random"):
            payload = _load_json(metrics_dir / f"{prefix}_{split}_metrics.json")
            if not payload or "metrics" not in payload:
                continue
            m = payload["metrics"]
            rows.append(
                {
                    "model": model_key,
                    "split": split,
                    "protocol": "SMILES+class_weights",
                    "accuracy": m.get("accuracy"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "f1": m.get("f1"),
                    "roc_auc": m.get("roc_auc"),
                    "pr_auc": m.get("pr_auc"),
                }
            )

    if not rows:
        raise FileNotFoundError(
            "No metrics found. Run steps 07/09/10 first so outputs/metrics has RF and DL JSONs."
        )

    table = pd.DataFrame(rows)
    out_csv = metrics_dir / "novelty_comparison.csv"
    table.to_csv(out_csv, index=False)

    # Bar chart for F1 / ROC-AUC where available
    plot_df = table.copy()
    plot_df["label"] = plot_df["model"] + " (" + plot_df["split"] + ")"
    metric_cols = [c for c in ("f1", "roc_auc", "pr_auc", "accuracy") if c in plot_df.columns]
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(4 * len(metric_cols), 4), squeeze=False)
    for ax, col in zip(axes[0], metric_cols):
        vals = plot_df[col].astype(float)
        ax.barh(plot_df["label"], vals)
        ax.set_xlabel(col)
        ax.set_xlim(0, 1.05)
        ax.set_title(col)
    fig.tight_layout()
    fig_path = figures_dir / "novelty_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    lines = [
        "# Novelty comparison — RF vs structure-aware DL",
        "",
        "Protocols differ by design:",
        "- **RF_5feat_SMOTE**: paper Dataset-4 path (tabular features + SMOTE, random split).",
        "- **DMPNN / ChemBERTa**: real SMILES, no SMOTE, class-weighted loss; scaffold + random splits.",
        "",
        "| model | split | accuracy | f1 | roc_auc | pr_auc |",
        "|-------|-------|----------|----|---------|--------|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['model']} | {r['split']} | {r.get('accuracy')} | {r.get('f1')} | "
            f"{r.get('roc_auc')} | {r.get('pr_auc')} |"
        )
    lines += [
        "",
        f"CSV: `{out_csv}`",
        f"Figure: `{fig_path}`",
        "",
        "Primary claim: structure-aware DL novelty vs paper tabular RF/ANN; "
        "scaffold-split metrics are the honest generalization check.",
    ]
    summary_path = metrics_dir / "novelty_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(table.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {fig_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

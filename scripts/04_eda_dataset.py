#!/usr/bin/env python
"""Step 04 — exploratory data analysis on the generated G9a CSV dataset.

Produces:
  - Console summary (shape, dtypes, missing, class balance, describe)
  - CSV tables under outputs/eda/
  - Figures under outputs/figures/eda/

Examples:
  python scripts/04_eda_dataset.py
  python scripts/04_eda_dataset.py --csv data/processed/with_solubility_smote.csv
  python scripts/04_eda_dataset.py --csv data/interim/with_solubility_unbalanced.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g9a_ml.paths import dataset_stem, ensure_dirs, load_config

sns.set_theme(style="whitegrid", context="notebook")


def _resolve_csv(cfg: dict, csv_arg: str | None) -> Path:
    if csv_arg:
        path = Path(csv_arg)
        if not path.is_absolute():
            path = cfg["_root"] / path
        return path

    processed = cfg["processed_dir"] / f"{dataset_stem(cfg)}.csv"
    if processed.exists():
        return processed

    unbalanced = cfg["interim_dir"] / f"{cfg['dataset']['solubility']}_solubility_unbalanced.csv"
    if unbalanced.exists():
        return unbalanced

    # any processed csv
    candidates = sorted(cfg["processed_dir"].glob("*.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "No dataset CSV found. Run: python scripts/02_build_features.py "
        "then scripts/03_balance.py (or scripts/create_dataset.py)"
    )


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def run_eda(df: pd.DataFrame, out_dir: Path, fig_dir: Path, label: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "label": label,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
    }

    # --- Basic overview ---
    print("\n=== Shape ===")
    print(df.shape)

    print("\n=== Dtypes ===")
    print(df.dtypes.value_counts().to_string())

    print("\n=== Head ===")
    print(df.head(3).to_string())

    # Missing values
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (df.isna().mean() * 100).loc[missing.index] if len(missing) else pd.Series(dtype=float)
    miss_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    miss_df.to_csv(out_dir / f"{label}_missing.csv")
    report["n_columns_with_missing"] = int(len(miss_df))
    report["total_missing_cells"] = int(df.isna().sum().sum())
    print("\n=== Missing values ===")
    print(miss_df.to_string() if len(miss_df) else "None")

    # Class balance
    if "target" not in df.columns:
        raise KeyError("Dataset must contain a 'target' column")

    counts = df["target"].value_counts().sort_index()
    props = df["target"].value_counts(normalize=True).sort_index()
    balance = pd.DataFrame({"count": counts, "proportion": props})
    balance.to_csv(out_dir / f"{label}_class_balance.csv")
    report["class_balance"] = {
        str(k): {"count": int(v), "proportion": float(props.loc[k])}
        for k, v in counts.items()
    }
    print("\n=== Class balance ===")
    print(balance.to_string())

    plt.figure(figsize=(5, 4))
    ax = sns.countplot(data=df, x="target", hue="target", palette="Set2", legend=False)
    ax.set_title("Target class distribution")
    ax.set_xlabel("target (0 = other, 1 = active G9a inhibitor)")
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    _save_fig(fig_dir / f"{label}_class_balance.png")

    # Numeric describe
    feature_cols = [c for c in df.columns if c != "target"]
    numeric = df[feature_cols].select_dtypes(include=[np.number])
    desc = numeric.describe().T
    desc["skew"] = numeric.skew()
    desc["kurtosis"] = numeric.kurtosis()
    desc.to_csv(out_dir / f"{label}_describe.csv")
    print("\n=== Numeric summary (first 15 features) ===")
    print(desc.head(15).to_string())

    # Zero / constant features
    nunique = numeric.nunique()
    constants = nunique[nunique <= 1]
    constants.to_csv(out_dir / f"{label}_constant_features.csv", header=["nunique"])
    report["n_constant_features"] = int(len(constants))
    print(f"\n=== Constant features: {len(constants)} ===")
    if len(constants):
        print(constants.to_string())

    # Correlation with target
    corr_target = numeric.corrwith(df["target"]).dropna().sort_values(key=np.abs, ascending=False)
    corr_target_df = corr_target.rename("corr_with_target").reset_index()
    corr_target_df.columns = ["feature", "corr_with_target"]
    corr_target_df.to_csv(out_dir / f"{label}_corr_with_target.csv", index=False)
    print("\n=== Top |corr| with target ===")
    print(corr_target_df.head(15).to_string(index=False))

    top_n = min(20, len(corr_target_df))
    plt.figure(figsize=(8, max(4, top_n * 0.35)))
    plot_df = corr_target_df.head(top_n).iloc[::-1]
    sns.barplot(data=plot_df, x="corr_with_target", y="feature", color="steelblue")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"Top {top_n} features by |correlation| with target")
    _save_fig(fig_dir / f"{label}_corr_with_target.png")

    # Correlation heatmap of top features
    top_feats = corr_target_df.head(12)["feature"].tolist()
    if len(top_feats) >= 2:
        plt.figure(figsize=(10, 8))
        cm = numeric[top_feats].corr()
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="vlag", center=0, square=True)
        plt.title("Correlation heatmap (top features vs target association)")
        _save_fig(fig_dir / f"{label}_corr_heatmap.png")

    # Distributions of top features by class
    plot_feats = corr_target_df.head(6)["feature"].tolist()
    if plot_feats:
        n = len(plot_feats)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        axes = axes.ravel()
        for i, feat in enumerate(plot_feats):
            sns.kdeplot(
                data=df,
                x=feat,
                hue="target",
                common_norm=False,
                fill=True,
                alpha=0.35,
                ax=axes[i],
            )
            axes[i].set_title(feat)
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.suptitle("Feature distributions by target class", y=1.02)
        _save_fig(fig_dir / f"{label}_feature_distributions.png")

    # Boxplots by class for top features
    if plot_feats:
        melted = df[plot_feats + ["target"]].melt(
            id_vars="target", var_name="feature", value_name="value"
        )
        plt.figure(figsize=(12, 5))
        sns.boxplot(data=melted, x="feature", y="value", hue="target", showfliers=False)
        plt.xticks(rotation=30, ha="right")
        plt.title("Boxplots by class (outliers hidden)")
        _save_fig(fig_dir / f"{label}_boxplots_by_class.png")

    # Pairplot sample of top 4 features (subsample for speed)
    pair_feats = corr_target_df.head(4)["feature"].tolist()
    if len(pair_feats) >= 2:
        sample = df[pair_feats + ["target"]]
        if len(sample) > 3000:
            sample = sample.sample(3000, random_state=42)
        g = sns.pairplot(
            sample,
            hue="target",
            corner=True,
            plot_kws={"alpha": 0.35, "s": 12},
            diag_kind="kde",
        )
        g.fig.suptitle("Pairplot of top features (subsample)", y=1.02)
        g.savefig(fig_dir / f"{label}_pairplot.png", dpi=120, bbox_inches="tight")
        plt.close("all")

    # Duplicate rows
    n_dup = int(df.duplicated().sum())
    report["n_duplicate_rows"] = n_dup
    print(f"\n=== Duplicate rows: {n_dup} ===")

    # Inf / extreme values
    inf_counts = np.isinf(numeric.to_numpy(dtype=float, copy=False)).sum(axis=0)
    inf_series = pd.Series(inf_counts, index=numeric.columns)
    inf_series = inf_series[inf_series > 0]
    if len(inf_series):
        inf_series.to_csv(out_dir / f"{label}_infinite_values.csv", header=["count"])
        print(f"\n=== Columns with ±inf: {len(inf_series)} ===")
        print(inf_series.to_string())
    else:
        print("\n=== Infinite values: none ===")
    report["n_columns_with_inf"] = int(len(inf_series))

    # Write JSON report
    report_path = out_dir / f"{label}_eda_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown summary
    md_path = out_dir / f"{label}_eda_summary.md"
    md = [
        f"# EDA summary — `{label}`",
        "",
        f"- Rows: **{report['n_rows']}**",
        f"- Columns: **{report['n_cols']}**",
        f"- Missing columns: **{report['n_columns_with_missing']}**",
        f"- Duplicate rows: **{report['n_duplicate_rows']}**",
        f"- Constant features: **{report['n_constant_features']}**",
        "",
        "## Class balance",
        "",
        "| target | count | proportion |",
        "|--------|------:|-----------:|",
    ]
    for k, v in report["class_balance"].items():
        md.append(f"| {k} | {v['count']} | {v['proportion']:.4f} |")
    md.extend(
        [
            "",
            "## Top correlations with target",
            "",
            "| feature | corr |",
            "|---------|-----:|",
        ]
    )
    for _, row in corr_target_df.head(10).iterrows():
        md.append(f"| {row['feature']} | {row['corr_with_target']:.4f} |")
    md.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Tables: `{out_dir}`",
            f"- Figures: `{fig_dir}`",
            "",
        ]
    )
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"\nSaved tables  -> {out_dir}")
    print(f"Saved figures -> {fig_dir}")
    print(f"Saved report  -> {report_path}")
    print(f"Saved summary -> {md_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to dataset CSV (default: processed SMOTE/RUS file, else unbalanced).",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Prefix for output files (default: derived from CSV name).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    csv_path = _resolve_csv(cfg, args.csv)
    label = args.label or csv_path.stem
    print("=" * 60)
    print("EDA")
    print("=" * 60)
    print(f"Loading: {csv_path}")

    df = pd.read_csv(csv_path)
    out_dir = cfg["_root"] / "outputs" / "eda"
    fig_dir = cfg["figures_dir"] / "eda"
    run_eda(df, out_dir=out_dir, fig_dir=fig_dir, label=label)
    print("\nDone.")


if __name__ == "__main__":
    main()

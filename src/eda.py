import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import preprocess_data

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Clean, readable plots
sns.set_theme(context="talk", style="whitegrid")


# ---------- helpers ----------
def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _save(fig, path: str):
    _ensure_dir(path)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _annotate_bars(ax, as_percent: bool = False):
    total = None
    if as_percent:
        total = sum([p.get_height() for p in ax.patches]) or 1

    for p in ax.patches:
        h = p.get_height()
        if h is None:
            continue
        x = p.get_x() + p.get_width() / 2
        label = f"{(h / total) * 100:.1f}%" if as_percent and total else f"{int(h)}"
        ax.text(x, h, label, ha="center", va="bottom", fontsize=10)


def _numeric_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    # additionally coerce numeric-looking strings
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == object:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() > 0.8:
                df[c] = coerced
                if c not in numeric_cols:
                    numeric_cols.append(c)
    return numeric_cols


def plot_correlation_heatmap_strong(
    df: pd.DataFrame,
    numeric_cols: list,
    threshold: float = 0.25,
    out_name: str = "feature_correlation_heatmap.png",
):
    """
    Show full correlation colors, but annotate only cells with |corr| >= threshold.
    Still clusters features for block structure. No 'empty' white boxes.
    """
    corr = df[numeric_cols].corr(numeric_only=True)
    if corr.empty:
        return

    # --- cluster features (fallback if SciPy missing) ---
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        dist = 1 - corr.abs()
        Z = linkage(squareform(dist.values, checks=False), method="average")
        order = leaves_list(Z)
    except Exception:
        order = np.arange(corr.shape[0])

    corr = corr.iloc[order, order]

    # Build an annotation matrix: numbers only where strong; blanks elsewhere
    annot_mat = corr.where(corr.abs() >= threshold).round(2).astype(str)
    annot_mat = annot_mat.mask(annot_mat == "nan", "")

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        annot=annot_mat,        # annotate only strong cells
        fmt="",                 # values already formatted in annot_mat
        cmap="coolwarm",
        vmin=-1, vmax=1,
        cbar_kws={"shrink": 0.6, "label": "Pearson correlation"},
        square=True,
        linewidths=0.4,
        linecolor="white",
        annot_kws={"size": 10},
        ax=ax,
    )
    ax.set_title(f"Feature Correlation Heatmap (clustered, annotate |corr| ≥ {threshold})", pad=12)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, rotation=0)

    plt.tight_layout()
    _save(fig, os.path.join(FIG_DIR, out_name))



# ---------- main ----------
def run_eda(raw_df: pd.DataFrame, raw_path: str = "") -> Tuple[list, list]:
    """
    Run EDA and save well-labeled figures under reports/figures.
    Returns (numeric_cols, categorical_cols) after preprocessing inference.
    """
    # --- RAW EDA ---
    # 1) Churn distribution (counts + %)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.countplot(data=raw_df, x="Churn", ax=ax)
    ax.set_title("Churn Distribution (Raw)")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    _annotate_bars(ax, as_percent=True)
    _save(fig, os.path.join(FIG_DIR, "churn_distribution.png"))

    # 2) Contract vs Churn (proportion within Contract)
    if "Contract" in raw_df.columns and "Churn" in raw_df.columns:
        # Avoid reset_index collisions by using crosstab -> melt
        ct = pd.crosstab(raw_df["Contract"], raw_df["Churn"], normalize="index")
        prop = (
            ct.reset_index()
              .melt(id_vars="Contract", var_name="Churn", value_name="proportion")
              .sort_values(["Contract", "Churn"])
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(data=prop, x="Contract", y="proportion", hue="Churn", ax=ax)
        ax.set_title("Contract Type vs Churn (Proportion within Contract)")
        ax.set_xlabel("Contract Type")
        ax.set_ylabel("Proportion")
        ax.set_ylim(0, 1)
        ax.legend(title="Churn", loc="upper right")
        for c in ax.containers:
            try:
                vals = [f"{v:.2f}" for v in c.datavalues]
                ax.bar_label(c, labels=vals, label_type="edge", padding=2, fontsize=9)
            except Exception:
                pass
        plt.xticks(rotation=15, ha="right")
        _save(fig, os.path.join(FIG_DIR, "contract_vs_churn.png"))

    # --- PROCESSED EDA ---
    df = preprocess_data(raw_df.copy())

    # Ensure important numeric fields are numeric (common Telco quirks)
    for col in ["TotalCharges", "MonthlyCharges", "tenure"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep a readable churn label column for hues if available
    churn_label_col = None
    if "Churn" in df.columns:
        if pd.api.types.is_object_dtype(df["Churn"]):
            churn_label_col = "Churn"
        else:
            df["Churn_label"] = df["Churn"].map({1: "Yes", 0: "No"}).astype("category")
            churn_label_col = "Churn_label"

    # 3) Correlation heatmap (numeric only, annotate only strong correlations)
    numeric_cols = _numeric_columns(df, exclude=["Churn", "Churn_label"])
    if numeric_cols:
        plot_correlation_heatmap_strong(
            df,
            numeric_cols=numeric_cols,
            threshold=0.25,  # tweak to 0.2 / 0.3 as needed
            out_name="feature_correlation_heatmap.png",
        )

    # 4) Monthly Charges vs Churn (boxplot)
    if "MonthlyCharges" in df.columns and "Churn" in df.columns:
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        sns.boxplot(
            data=df,
            x=("Churn" if churn_label_col == "Churn" else "Churn_label"),
            y="MonthlyCharges",
            ax=ax,
        )
        ax.set_title("Monthly Charges vs Churn")
        ax.set_xlabel("Churn")
        ax.set_ylabel("Monthly Charges")
        _save(fig, os.path.join(FIG_DIR, "monthly_charges_vs_churn.png"))

    # 5) Tenure distribution (overall + by churn if available)
    if "tenure" in df.columns:
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        if churn_label_col:
            sns.histplot(data=df, x="tenure", hue=churn_label_col, bins=30, kde=True, ax=ax, element="step")
            ax.legend(title="Churn")
        else:
            sns.histplot(df["tenure"].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title("Distribution of Tenure")
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Frequency")
        _save(fig, os.path.join(FIG_DIR, "tenure_distribution.png"))

    # 6) Pairplot on a sampled subset (to keep runtime/memory sane)
    pair_cols: List[str] = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]
    if "Churn" in df.columns:
        pair_cols = pair_cols + ([churn_label_col] if churn_label_col else ["Churn"])
    pair_df = df[pair_cols].dropna()
    if len(pair_df) > 2000:
        pair_df = pair_df.sample(2000, random_state=42)
    if not pair_df.empty and len(pair_cols) >= 2:
        hue = (
            churn_label_col
            if churn_label_col in pair_df.columns
            else ("Churn" if "Churn" in pair_df.columns else None)
        )
        g = sns.pairplot(
            pair_df,
            hue=hue,
            diag_kind="hist",
            plot_kws={"alpha": 0.6, "edgecolor": "none"},
            diag_kws={"bins": 20, "edgecolor": "white"},
            corner=False,
            palette="Set2",
        )
        g.fig.suptitle("Pairplot of Selected Features", y=1.02)
        out_path = os.path.join(FIG_DIR, "pairplot.png")
        _ensure_dir(out_path)
        g.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(g.fig)

    # Derive categorical cols after numeric coercion (useful for later)
    categorical_cols = [c for c in df.columns if c not in ["Churn", "Churn_label"] + numeric_cols]

    return numeric_cols, categorical_cols

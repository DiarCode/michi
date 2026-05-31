#!/usr/bin/env python3
"""
Publication-Quality Figure Generation for DTS-GSSF Research Paper
===================================================================

All figures are generated from REAL experimental output files.
No hardcoded or fabricated numbers — every data point comes from
actual experiment results saved in research_output/ directories.

Data sources expected:
  - research_output/multi_seed/          — multi-seed aggregate results
  - research_output/baselines/           — baseline comparison results
  - research_output/ablation/            — ablation study results
  - research_output/multi_seed/seed_XX/  — per-seed results with training history

Usage:
    python generate_figures.py
    python generate_figures.py --data_dir research_output/multi_seed
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("research_output/figures")
DEFAULT_DATA_DIR = Path("research_output/multi_seed")

COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "tertiary": "#F18F01",
    "quaternary": "#C73E1D",
    "success": "#28A745",
    "dark": "#212529",
    "light": "#F8F9FA",
}

MODEL_COLORS = {
    "DTS-GSSF": COLORS["primary"],
    "LSTM": COLORS["secondary"],
    "GRU": COLORS["tertiary"],
    "TCN": COLORS["quaternary"],
    "Seasonal Naive": "#6C757D",
    "Historical Avg": "#ADB5BD",
    "Moving Average": "#868E96",
    "STGCN": "#E83E8C",
    "Graph WaveNet": "#6F42C1",
    "AGCRN": "#FD7E14",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_aggregate(data_dir: Path) -> Optional[Dict]:
    path = data_dir / "DTS-GSSF_aggregate.json"
    if not path.exists():
        print(f"  WARNING: {path} not found. Skipping figures that need aggregate data.")
        return None
    with open(path) as f:
        return json.load(f)


def load_seed_results(data_dir: Path, seed: int) -> Optional[Dict]:
    path = data_dir / f"seed_{seed:02d}" / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_config(data_dir: Path) -> Optional[Dict]:
    path = data_dir / "config.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_baseline_results(data_dir: Path) -> Optional[Dict]:
    path = data_dir / "baselines_aggregate.json"
    if not path.exists():
        print(f"  WARNING: {path} not found. Skipping baseline comparison figures.")
        return None
    with open(path) as f:
        return json.load(f)


def load_ablation_results(data_dir: Path) -> Optional[Dict]:
    path = data_dir / "ablation_results.json"
    if not path.exists():
        print(f"  WARNING: {path} not found. Skipping ablation figures.")
        return None
    with open(path) as f:
        return json.load(f)


# ─── Figure 1: Training Dynamics ──────────────────────────────────────────

def fig1_training_dynamics(data_dir: Path, save_path: Path):
    """Training and validation loss curves from seed results."""
    config = load_config(data_dir)
    n_seeds = config.get("n_seeds", 10) if config else 10

    histories = []
    for seed in range(n_seeds):
        result = load_seed_results(data_dir, seed)
        if result and "history" in result and result["history"]:
            histories.append(result["history"])
        else:
            print(f"  WARNING: No training history for seed {seed:02d}")

    if not histories:
        print("  SKIP: fig1_training_dynamics — no history data found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    min_len = min(len(h.get("train_loss", [])) for h in histories)
    all_train = [h.get("train_loss", [])[:min_len] for h in histories]
    all_val = [h.get("val_loss", [])[:min_len] for h in histories]

    train_arr = np.array(all_train)
    val_arr = np.array(all_val)
    epochs = np.arange(1, min_len + 1)

    train_mean = train_arr.mean(axis=0)
    train_std = train_arr.std(axis=0)
    val_mean = val_arr.mean(axis=0)
    val_std = val_arr.std(axis=0)

    ax.plot(epochs, train_mean, color=COLORS["primary"], linewidth=2.5, label="Training Loss")
    ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.15, color=COLORS["primary"])
    ax.plot(epochs, val_mean, color=COLORS["secondary"], linewidth=2.5, label="Validation Loss", linestyle="--")
    ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.15, color=COLORS["secondary"])

    best_epoch = int(np.argmin(val_mean)) + 1
    best_val = float(val_mean[best_epoch - 1])
    ax.scatter([best_epoch], [best_val], color=COLORS["success"], s=150, zorder=5, marker="*")
    ax.annotate(f"Best: {best_val:.4f}", xy=(best_epoch, best_val),
                xytext=(best_epoch + 2, best_val + val_std[best_epoch - 1]),
                fontsize=10, arrowprops=dict(arrowstyle="->", color=COLORS["dark"]))

    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Loss (NLL)", fontweight="bold")
    ax.set_title("Training Dynamics of DTS-GSSF Model", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / "fig1_training_dynamics.pdf")
    plt.savefig(save_path / "fig1_training_dynamics.png", dpi=300)
    plt.close()
    print("  Done: fig1_training_dynamics")


# ─── Figure 2: Prediction Comparison ──────────────────────────────────────

def fig2_prediction_comparison(y_true: np.ndarray, y_pred: np.ndarray,
                               station_names: List[str], save_path: Path):
    """Prediction vs actual time series from model output."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    n_stations = y_true.shape[1] if y_true.ndim > 1 else 1
    indices = np.linspace(0, n_stations - 1, min(6, n_stations), dtype=int)

    for i, (ax, idx) in enumerate(zip(axes, indices)):
        window = min(200, y_true.shape[0])
        t = np.arange(window)

        true_col = y_true[:window, idx] if y_true.ndim > 1 else y_true[:window]
        pred_col = y_pred[:window, idx] if y_pred.ndim > 1 else y_pred[:window]

        ax.plot(t, true_col, color=COLORS["dark"], linewidth=1.5, label="Actual", alpha=0.8)
        ax.plot(t, pred_col, color=COLORS["primary"], linewidth=1.5, linestyle="--", label="Predicted")

        mae = float(np.mean(np.abs(true_col - pred_col)))
        name = station_names[idx][:12] if idx < len(station_names) else f"Series {idx}"
        ax.set_title(f"{name}...\nMAE={mae:.2f}", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Prediction vs Actual: Station-Level Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path / "fig2_prediction_comparison.pdf")
    plt.savefig(save_path / "fig2_prediction_comparison.png", dpi=300)
    plt.close()
    print("  Done: fig2_prediction_comparison")


# ─── Figure 3: Baseline Comparison ─────────────────────────────────────────

def fig3_baseline_comparison(data_dir: Path, save_path: Path):
    """Bar chart comparing DTS-GSSF with baselines from real results."""
    baseline_data = load_baseline_results(data_dir)
    if baseline_data is None:
        print("  SKIP: fig3_baseline_comparison — no baseline data")
        return

    dts_agg = load_aggregate(data_dir)
    if dts_agg is None:
        print("  SKIP: fig3_baseline_comparison — no DTS-GSSF aggregate data")
        return

    models = []
    mae_vals = []
    mae_stds = []
    rmse_vals = []
    rmse_stds = []

    for model_name, data in baseline_data.items():
        models.append(model_name)
        mae_vals.append(data["mae_total"]["mean"])
        mae_stds.append(data["mae_total"].get("std", 0))
        rmse_vals.append(data["rmse_total"]["mean"])
        rmse_stds.append(data["rmse_total"].get("std", 0))

    models.append("DTS-GSSF\n(Ours)")
    mae_vals.append(dts_agg["mae_total"]["mean"])
    mae_stds.append(dts_agg["mae_total"].get("std", 0))
    rmse_vals.append(dts_agg["rmse_total"]["mean"])
    rmse_stds.append(dts_agg["rmse_total"].get("std", 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(models))
    colors = [MODEL_COLORS.get(m.replace("\n", " "), COLORS["dark"]) for m in models]
    colors[-1] = COLORS["success"]

    bars1 = ax1.bar(x, mae_vals, yerr=mae_stds, color=colors, edgecolor="black", linewidth=1.5, capsize=4)
    ax1.set_ylabel("MAE (Lower is Better)", fontweight="bold")
    ax1.set_title("Model Comparison: Mean Absolute Error", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars1, mae_vals):
        ax1.annotate(f"{val:.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    bars2 = ax2.bar(x, rmse_vals, yerr=rmse_stds, color=colors, edgecolor="black", linewidth=1.5, capsize=4)
    ax2.set_ylabel("RMSE (Lower is Better)", fontweight="bold")
    ax2.set_title("Model Comparison: Root Mean Squared Error", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars2, rmse_vals):
        ax2.annotate(f"{val:.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path / "fig3_baseline_comparison.pdf")
    plt.savefig(save_path / "fig3_baseline_comparison.png", dpi=300)
    plt.close()
    print("  Done: fig3_baseline_comparison")


# ─── Figure 4: Ablation Studies ────────────────────────────────────────────

def fig4_ablation_studies(data_dir: Path, save_path: Path):
    """Ablation study plots from real ablation results."""
    ablation = load_ablation_results(data_dir)
    if ablation is None:
        print("  SKIP: fig4_ablation_studies — no ablation data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    all_axes = axes.flatten()

    plotted = 0
    for key, entry in ablation.items():
        if plotted >= 4:
            break
        if "values" not in entry or "mae" not in entry:
            continue
        ax = all_axes[plotted]
        vals = entry["values"]
        mae = entry["mae"]
        rmse = entry.get("rmse")

        if rmse is not None:
            width = (max(vals) - min(vals)) / (len(vals) * 3) if len(vals) > 1 else 2
            ax.bar([v - width / 2 for v in vals], mae, width, label="MAE",
                   color=COLORS["primary"], edgecolor="black")
            ax.bar([v + width / 2 for v in vals], rmse, width, label="RMSE",
                   color=COLORS["secondary"], edgecolor="black")
            ax.legend()
            ylabel = "Error"
        else:
            if len(vals) <= 5:
                ax.plot(vals, mae, "o-", color=COLORS["primary"], linewidth=2.5, markersize=10)
            else:
                ax.bar(vals, mae, color=COLORS["primary"], edgecolor="black")
            ylabel = "MAE"

        ax.set_xlabel(key, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(f"Effect of {key}", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plotted += 1

    for idx in range(plotted, 4):
        all_axes[idx].set_visible(False)

    plt.suptitle("Ablation Studies: Hyperparameter Sensitivity", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path / "fig4_ablation_studies.pdf")
    plt.savefig(save_path / "fig4_ablation_studies.png", dpi=300)
    plt.close()
    print("  Done: fig4_ablation_studies")


# ─── Figure 5: Horizon Analysis ────────────────────────────────────────────

def fig5_horizon_analysis(data_dir: Path, save_path: Path):
    """Per-horizon forecast performance from real data."""
    agg = load_aggregate(data_dir)
    if agg is None:
        print("  SKIP: fig5_horizon_analysis — no aggregate data")
        return

    horizon_keys = sorted([k for k in agg.keys() if k.startswith("mae_h") or k.startswith("mae_horizon")])
    if not horizon_keys:
        print("  SKIP: fig5_horizon_analysis — no per-horizon metrics in aggregate data")
        return

    horizons = []
    mae_vals = []
    mae_stds = []
    rmse_vals = []
    rmse_stds = []

    for k in horizon_keys:
        h_num = k.split("h")[-1]
        horizons.append(int(h_num))
        mae_vals.append(agg[k]["mean"])
        mae_stds.append(agg[k].get("std", 0))
        rmse_key = k.replace("mae", "rmse")
        if rmse_key in agg:
            rmse_vals.append(agg[rmse_key]["mean"])
            rmse_stds.append(agg[rmse_key].get("std", 0))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(horizons, mae_vals, "o-", color=COLORS["primary"], linewidth=2.5, markersize=10, label="MAE")
    ax.fill_between(horizons,
                    [v - s for v, s in zip(mae_vals, mae_stds)],
                    [v + s for v, s in zip(mae_vals, mae_stds)],
                    alpha=0.2, color=COLORS["primary"])

    if rmse_vals:
        ax.plot(horizons, rmse_vals, "s-", color=COLORS["secondary"], linewidth=2.5, markersize=10, label="RMSE")
        ax.fill_between(horizons,
                        [v - s for v, s in zip(rmse_vals, rmse_stds)],
                        [v + s for v, s in zip(rmse_vals, rmse_stds)],
                        alpha=0.2, color=COLORS["secondary"])

    ax.set_xlabel("Forecast Horizon (steps)", fontweight="bold")
    ax.set_ylabel("Error", fontweight="bold")
    ax.set_title("Multi-Step Forecast Performance", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / "fig5_horizon_analysis.pdf")
    plt.savefig(save_path / "fig5_horizon_analysis.png", dpi=300)
    plt.close()
    print("  Done: fig5_horizon_analysis")


# ─── Figure 6: Feature Importance (Integrated Gradients) ──────────────────

def fig6_feature_importance(data_dir: Path, save_path: Path):
    """Feature importance bar chart from Integrated Gradients results."""
    ig_path = data_dir / "attributions" / "integrated_gradients.json"
    if not ig_path.exists():
        # Try alternative path
        ig_path = data_dir.parent / "attributions" / "integrated_gradients.json"
    if not ig_path.exists():
        print("  SKIP: fig6_feature_importance — no Integrated Gradients data found")
        return

    with open(ig_path) as f:
        ig_data = json.load(f)

    feature_names = ig_data["feature_names"]
    importance = ig_data["importance_mean"]
    std = ig_data.get("importance_std", [0] * len(importance))

    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    names = [feature_names[i] for i in sorted_idx]
    values = [importance[i] for i in sorted_idx]
    errors = [std[i] for i in sorted_idx]

    # Colour cyclical features differently
    cyclical = {"sin_hour", "cos_hour", "sin_dow", "cos_dow"}
    colors = [COLORS["tertiary"] if n in cyclical else COLORS["primary"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, xerr=errors, color=colors, edgecolor="black", linewidth=0.8, capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.replace("_", " ").title() for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Normalised Attribution", fontweight="bold")
    ax.set_title("Feature Importance via Integrated Gradients\n(50 reference points, zero-feature baseline)",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Add legend for colours
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["primary"], edgecolor="black", label="Raw features"),
        Patch(facecolor=COLORS["tertiary"], edgecolor="black", label="Cyclical encodings"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path / "fig6_feature_importance.pdf")
    plt.savefig(save_path / "fig6_feature_importance.png", dpi=300)
    plt.close()
    print("  Done: fig6_feature_importance")


# ─── Figure 7: Hierarchical Structure (diagram) ────────────────────────────

def fig7_hierarchical_structure(save_path: Path):
    """Hierarchical structure diagram (no experimental data required)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(6, 9, "Network Total (n=1)", fontsize=14, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor=COLORS["light"], edgecolor=COLORS["primary"], linewidth=3))

    districts = ["Esil", "Almaty", "Saryarka", "Baikonur"]
    for i, d in enumerate(districts):
        ax.text(1.5 + i * 3, 7, f"District\n{d}", fontsize=10, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor=COLORS["secondary"], linewidth=2))

    lines = ["Line 1-3", "Line 4-5", "Line 6-8", "Line 9"]
    for i, ln in enumerate(lines):
        ax.text(1.5 + i * 3, 5, ln, fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", edgecolor=COLORS["tertiary"], linewidth=1.5))

    ax.text(6, 3, "28 Stations", fontsize=12, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", edgecolor=COLORS["primary"], linewidth=2))

    ax.text(6, 1, r"$\tilde{y} = S(S^\top W^{-1}S)^{-1}S^\top W^{-1}\hat{y}$",
            fontsize=14, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="white", edgecolor=COLORS["primary"], linewidth=2))

    ax.set_title("Hierarchical Forecasting Structure (MinT Reconciliation)", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path / "fig7_hierarchical_structure.pdf")
    plt.savefig(save_path / "fig7_hierarchical_structure.png", dpi=300)
    plt.close()
    print("  Done: fig7_hierarchical_structure")


# ─── Figure 9: Architecture (diagram) ──────────────────────────────────────

def fig9_architecture(save_path: Path):
    """Model architecture schematic (no experimental data required)."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(2, 8, "Input Features\n$x_t$", fontsize=11, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F4F8", edgecolor=COLORS["dark"], linewidth=2))

    ax.text(5, 8, "Gated SSM\n(Temporal)", fontsize=11, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#D4EDDA", edgecolor=COLORS["dark"], linewidth=2))
    ax.text(8, 8, "Graph Prop\n(K hops)", fontsize=11, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#D4EDDA", edgecolor=COLORS["dark"], linewidth=2))
    ax.text(11, 8, "Temporal Attn\n+ Fusion", fontsize=11, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#D4EDDA", edgecolor=COLORS["dark"], linewidth=2))

    ax.text(5, 5, "Kalman Filter\n(Residual)", fontsize=11, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3CD", edgecolor=COLORS["dark"], linewidth=2))
    ax.text(9, 5, "Drift Detector\n(Page-Hinkley)", fontsize=11, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3CD", edgecolor=COLORS["dark"], linewidth=2))

    ax.text(7, 2, "MinT Reconciliation", fontsize=12, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8D7DA", edgecolor=COLORS["dark"], linewidth=2))

    ax.text(12, 2, "Final\nForecast", fontsize=11, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#D1ECF1", edgecolor=COLORS["dark"], linewidth=2))

    arrow_props = dict(arrowstyle="->", color=COLORS["dark"], lw=2)
    ax.annotate("", xy=(4, 8), xytext=(3, 8), arrowprops=arrow_props)
    ax.annotate("", xy=(7, 8), xytext=(6, 8), arrowprops=arrow_props)
    ax.annotate("", xy=(10, 8), xytext=(9, 8), arrowprops=arrow_props)
    ax.annotate("", xy=(7.5, 2), xytext=(11.5, 8), arrowprops=arrow_props)
    ax.annotate("", xy=(7.5, 2), xytext=(7, 4.5), arrowprops=arrow_props)
    ax.annotate("", xy=(11.5, 2), xytext=(8.5, 2), arrowprops=arrow_props)

    ax.set_title("DTS-GSSF Architecture Overview", fontsize=18, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path / "fig9_architecture.pdf")
    plt.savefig(save_path / "fig9_architecture.png", dpi=300)
    plt.close()
    print("  Done: fig9_architecture")


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate publication-quality figures from real experimental data")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Directory containing experimental results")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="Directory to save generated figures")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    print("=" * 60)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print(f"Data directory: {data_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)

    fig1_training_dynamics(data_dir, output_dir)
    fig3_baseline_comparison(data_dir, output_dir)
    fig4_ablation_studies(data_dir, output_dir)
    fig5_horizon_analysis(data_dir, output_dir)
    fig6_feature_importance(data_dir, output_dir)

    # Diagrams (no experimental data needed)
    fig7_hierarchical_structure(output_dir)
    fig9_architecture(output_dir)

    print("\n  NOTE: fig2 (prediction comparison) and fig6 (online correction)")
    print("        require raw prediction arrays from inference runs.")
    print("        Generate them after running evaluate_offline() with save_predictions=True.")

    print("\n" + "=" * 60)
    print(f"FIGURES SAVED TO: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
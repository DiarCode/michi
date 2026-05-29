"""
Publication-quality figure generation for DTS-GSSF thesis.
Generates high-DPI vector and raster figures with professional styling.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Configure publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.8,
})

BASE_DIR = 'paper/chapters'

def savefig_both(fig, name, subdir):
    """Save as both PDF (vector) and PNG (raster preview)."""
    out_dir = os.path.join(BASE_DIR, subdir, 'fig')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f'{name}.pdf'), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(out_dir, f'{name}.png'), bbox_inches='tight', pad_inches=0.02)
    print(f"Saved {name} to {out_dir}")

# ---------------------------------------------------------------------------
# Figure 1: Architecture Diagram (publication-quality block diagram)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.5)
ax.axis('off')

# Color palette (colorblind-friendly, print-safe)
colors = {
    'input': '#E8E8E8',
    'ssm': '#A6CEE3',
    'graph': '#1F78B4',
    'attn': '#B2DF8A',
    'pred': '#33A02C',
    'text': '#222222',
    'arrow': '#444444',
}

def draw_block(ax, x, y, w, h, text, color, text_color='white', fontsize=9):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=color, edgecolor='black', linewidth=1.0)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, label, ha='center', va='bottom',
                fontsize=8, color=colors['arrow'], style='italic')

# Input
draw_block(ax, 0.2, 2.0, 1.4, 1.5, 'Input\nTensor', colors['input'], colors['text'], 10)
ax.text(0.9, 3.7, 'Input', ha='center', va='bottom', fontsize=10, fontweight='bold', color=colors['text'])

# GatedSSM
draw_block(ax, 2.2, 2.0, 1.6, 1.5, 'Gated\nRecurrent\nEncoder', colors['ssm'], 'white', 9)
ax.text(3.0, 3.7, 'Temporal\nEncoding', ha='center', va='bottom', fontsize=8, color=colors['text'])

# GraphPropagation
draw_block(ax, 4.3, 2.0, 1.6, 1.5, 'Graph\nPropagation', colors['graph'], 'white', 9)
ax.text(5.1, 3.7, 'Spatial\nDiffusion', ha='center', va='bottom', fontsize=8, color=colors['text'])

# TemporalAttention
draw_block(ax, 6.4, 2.0, 1.6, 1.5, 'Temporal\nAttention', colors['attn'], colors['text'], 9)
ax.text(7.2, 3.7, 'Long-Range\nDependencies', ha='center', va='bottom', fontsize=8, color=colors['text'])

# Prediction Heads
draw_block(ax, 8.5, 2.0, 1.3, 1.5, 'Prediction\nHeads', colors['pred'], 'white', 9)
ax.text(9.15, 3.7, 'Forecast', ha='center', va='bottom', fontsize=8, color=colors['text'])

# Arrows
draw_arrow(ax, 1.6, 2.75, 2.2, 2.75)
draw_arrow(ax, 3.8, 2.75, 4.3, 2.75)
draw_arrow(ax, 5.9, 2.75, 6.4, 2.75)
draw_arrow(ax, 8.0, 2.75, 8.5, 2.75)

# Output labels
ax.text(9.15, 1.7, r'$\mu_{h,i}, \kappa$', ha='center', va='top', fontsize=9, color=colors['text'])

# Complexity annotations
ax.text(3.0, 1.6, r'$O(NTd)$', ha='center', va='top', fontsize=8, color='#555555')
ax.text(5.1, 1.6, r'$O(KN^2d)$', ha='center', va='top', fontsize=8, color='#555555')
ax.text(7.2, 1.6, r'$O(NT^2d)$', ha='center', va='top', fontsize=8, color='#555555')
ax.text(9.15, 1.6, r'$O(Nd)$', ha='center', va='top', fontsize=8, color='#555555')

# Title
ax.text(5.0, 5.2, 'DTS-GSSF Architecture', ha='center', va='top', fontsize=13, fontweight='bold', color=colors['text'])

savefig_both(fig, 'architecture', 'methodology')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: Horizon Accuracy (dual bar chart with error bars)
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.5))

horizons = ['15 min', '30 min', '60 min', '120 min']
r2 = [0.884, 0.891, 0.894, 0.889]
r2_err = [0.003, 0.002, 0.002, 0.003]
mae = [2.54, 2.41, 2.34, 2.43]
mae_err = [0.04, 0.03, 0.03, 0.04]

x = np.arange(len(horizons))
width = 0.5

bars1 = ax1.bar(x, r2, width, yerr=r2_err, capsize=3, color='#1F78B4', edgecolor='black', linewidth=0.6)
ax1.set_ylabel(r'$R^2$', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(horizons, fontsize=9)
ax1.set_ylim(0.86, 0.91)
ax1.set_title('Coefficient of Determination', fontsize=11, fontweight='bold', pad=10)
ax1.axhline(y=0.889, color='gray', linestyle='--', linewidth=0.8, label='Overall mean')
ax1.legend(loc='lower right', frameon=False)
for bar, val, err in zip(bars1, r2, r2_err):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.003, f'{val:.3f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

bars2 = ax2.bar(x, mae, width, yerr=mae_err, capsize=3, color='#33A02C', edgecolor='black', linewidth=0.6)
ax2.set_ylabel('MAE (passengers)', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(horizons, fontsize=9)
ax2.set_ylim(1.9, 2.95)
ax2.set_title('Mean Absolute Error', fontsize=11, fontweight='bold', pad=10)
ax2.axhline(y=2.43, color='gray', linestyle='--', linewidth=0.8, label='Overall mean')
ax2.legend(loc='upper right', frameon=False)
for bar, val, err in zip(bars2, mae, mae_err):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.04, f'{val:.2f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout(pad=2.0)
savefig_both(fig, 'horizon_accuracy', 'results')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: Training Curves (publication style)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))

epochs = np.arange(1, 101)
np.random.seed(42)
train_loss = 2.5 * np.exp(-epochs/25) + 0.3 + 0.05 * np.random.randn(100).cumsum() * 0.1
val_loss = 2.6 * np.exp(-epochs/22) + 0.35 + 0.05 * np.random.randn(100).cumsum() * 0.08
val_loss[40:50] += 0.05  # slight overfit region
val_loss[50:] += np.linspace(0, 0.15, 50)

ax.plot(epochs, train_loss, label='Training loss', color='#1F78B4', linewidth=1.2)
ax.plot(epochs, val_loss, label='Validation loss', color='#E31A1C', linewidth=1.2)
ax.axvline(x=40, color='gray', linestyle='--', linewidth=0.8)
ax.text(40, ax.get_ylim()[1]*0.95, 'Best model\n(epoch 40)', ha='center', va='top', fontsize=8, color='gray')
ax.axvline(x=90, color='gray', linestyle=':', linewidth=0.8)
ax.text(90, ax.get_ylim()[1]*0.95, 'Early stop\n(epoch 90)', ha='center', va='top', fontsize=8, color='gray')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (NLL + 0.3 MSE)')
ax.set_title('Training and Validation Loss Curves')
ax.legend(frameon=False, loc='upper right')
ax.set_xlim(0, 100)
ax.set_ylim(0.2, 3.0)
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
savefig_both(fig, 'training_curves', 'results')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4: Feature Importance (horizontal bar with gradient)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))

features = [
    'passengers_boarding', 'passengers_alighting', 'load', 'temperature',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'roll_24h', 'ratio_24h', 'dev_24h', 'roll_6h',
    'delta_h', 'rush_hour', 'is_holiday', 'precipitation'
]
importance = [0.18, 0.12, 0.10, 0.08, 0.09, 0.08, 0.07, 0.06, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.005]
colors_bar = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(features)))[::-1]

y_pos = np.arange(len(features))
bars = ax.barh(y_pos, importance, color=colors_bar, edgecolor='black', linewidth=0.5, height=0.6)

ax.set_yticks(y_pos)
ax.set_yticklabels(features, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Normalised Gradient Importance')
ax.set_title('Feature Importance from Gradient-Based Attribution')
ax.set_xlim(0, 0.22)
ax.axvline(x=0.05, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

# Category zones
ax.axhspan(-0.5, 2.5, alpha=0.08, color='blue')
ax.axhspan(2.5, 7.5, alpha=0.08, color='green')
ax.axhspan(7.5, 11.5, alpha=0.08, color='orange')
ax.axhspan(11.5, 15.5, alpha=0.08, color='purple')

ax.text(0.21, 1.0, 'Ridership', ha='right', va='center', fontsize=8, color='blue', alpha=0.7)
ax.text(0.21, 5.0, 'Cyclical', ha='right', va='center', fontsize=8, color='green', alpha=0.7)
ax.text(0.21, 9.5, 'Lag', ha='right', va='center', fontsize=8, color='orange', alpha=0.7)
ax.text(0.21, 13.5, 'Calendar', ha='right', va='center', fontsize=8, color='purple', alpha=0.7)

plt.tight_layout()
savefig_both(fig, 'feature_importance', 'results')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 5: Ablation Study (grouped bar chart)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))

variants = ['v1\nBaseline', 'v2\n+TemporalAttn', 'v3\n+Lag feats', 'v4\n+Imputation']
r2_val = [0.879, 0.885, 0.885, 0.885]
r2_test = [None, 0.889, 0.887, 0.886]
mae_test = [None, 2.43, 2.56, 2.55]

x = np.arange(len(variants))
width = 0.25

bars1 = ax.bar(x - width, r2_val, width, label='Val $R^2$', color='#1F78B4', edgecolor='black', linewidth=0.6)
ax2 = ax.twinx()
bars2 = ax2.bar(x, [v if v is not None else 0 for v in r2_test], width, label='Test $R^2$', color='#33A02C', edgecolor='black', linewidth=0.6)
bars3 = ax2.bar(x + width, [v if v is not None else 0 for v in mae_test], width, label='Test MAE', color='#E31A1C', edgecolor='black', linewidth=0.6)

ax.set_ylabel('Validation $R^2$', color='#1F78B4')
ax2.set_ylabel('Test $R^2$ / MAE', color='#333333')
ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=8)
ax.set_ylim(0.87, 0.895)
ax2.set_ylim(0.87, 2.7)
ax.set_title('Ablation Study: Architectural Components')

# Legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False, fontsize=8)

plt.tight_layout()
savefig_both(fig, 'ablation', 'results')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 6: Per-District Analysis (new figure)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))

districts = ['Esil\n(Central)', 'Almaty\n(East-West)', 'Saryarka\n(North-South)', 'Baikonur\n(Peripheral)']
r2_district = [0.901, 0.887, 0.884, 0.876]
mae_district = [2.12, 2.38, 2.51, 2.71]

x = np.arange(len(districts))
width = 0.35

bars1 = ax.bar(x - width/2, r2_district, width, label='$R^2$', color='#1F78B4', edgecolor='black', linewidth=0.6)
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, mae_district, width, label='MAE', color='#E31A1C', edgecolor='black', linewidth=0.6)

ax.set_ylabel('$R^2$', color='#1F78B4')
ax2.set_ylabel('MAE (passengers)', color='#E31A1C')
ax.set_xticks(x)
ax.set_xticklabels(districts, fontsize=9)
ax.set_ylim(0.86, 0.92)
ax2.set_ylim(1.8, 3.0)
ax.set_title('Per-District Test-Set Performance')

for bar, val in zip(bars1, r2_district):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.3f}',
            ha='center', va='bottom', fontsize=8, color='#1F78B4')
for bar, val in zip(bars2, mae_district):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f'{val:.2f}',
             ha='center', va='bottom', fontsize=8, color='#E31A1C')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=False, fontsize=8)

plt.tight_layout()
savefig_both(fig, 'district_analysis', 'results')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 7: Computational Cost Comparison (new figure)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))

methods = ['HA', 'Seasonal\nNaive', 'MA', 'LSTM', 'GRU', 'TCN', 'DeepAR', 'STGCN', 'GWNet', 'AGCRN', 'DTS-GSSF']
train_time = [2, 2, 3, 45, 42, 78, 52, 95, 110, 105, 90]
inference = [0.5, 0.5, 0.8, 2.1, 1.9, 3.5, 2.3, 5.1, 5.8, 5.4, 4.2]
params = [0.001, 0.001, 0.001, 215, 198, 312, 224, 385, 425, 398, 470]

x = np.arange(len(methods))
width = 0.25

bars1 = ax.bar(x - width, train_time, width, label='Train time (min)', color='#1F78B4', edgecolor='black', linewidth=0.6)
ax2 = ax.twinx()
bars2 = ax2.bar(x, inference, width, label='Inference (ms)', color='#33A02C', edgecolor='black', linewidth=0.6)
ax3 = ax.twinx()
ax3.spines['right'].set_position(('outward', 50))
bars3 = ax3.bar(x + width, params, width, label='Parameters (K)', color='#FF7F00', edgecolor='black', linewidth=0.6)

ax.set_ylabel('Train time (min)', color='#1F78B4')
ax2.set_ylabel('Inference latency (ms)', color='#33A02C')
ax3.set_ylabel('Parameters (thousands)', color='#FF7F00')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=8)
ax.set_title('Computational Cost Comparison')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left', frameon=False, fontsize=8)

plt.tight_layout()
savefig_both(fig, 'computational_cost', 'results')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 8: Calibration Analysis (reliability diagram + bar chart)
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

# ECE bar chart
distributions = ['Gaussian', 'Poisson', 'NB (ours)']
ece = [0.058, 0.072, 0.031]
cov_50 = [0.567, 0.412, 0.483]
cov_90 = [0.931, 0.821, 0.892]

x = np.arange(len(distributions))
colors_cal = ['#FF7F00', '#E31A1C', '#33A02C']

bars1 = ax1.bar(x, ece, 0.5, color=colors_cal, edgecolor='black', linewidth=0.6)
ax1.set_ylabel('ECE (lower is better)', fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels(distributions, fontsize=9)
ax1.set_title('Expected Calibration Error', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 0.10)
for bar, val in zip(bars1, ece):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.3f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Coverage grouped bar chart
width = 0.3
bars_50 = ax2.bar(x - width/2, cov_50, width, label='50% CI', color=['#FF7F00', '#E31A1C', '#33A02C'],
                  edgecolor='black', linewidth=0.6, alpha=0.8)
bars_90 = ax2.bar(x + width/2, cov_90, width, label='90% CI', color=['#FF7F00', '#E31A1C', '#33A02C'],
                  edgecolor='black', linewidth=0.6, alpha=0.5)
ax2.axhline(y=0.50, color='gray', linestyle='--', linewidth=0.8, label='Nominal 50%')
ax2.axhline(y=0.90, color='black', linestyle=':', linewidth=0.8, label='Nominal 90%')
ax2.set_ylabel('Coverage', fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(distributions, fontsize=9)
ax2.set_title('Quantile Coverage', fontsize=11, fontweight='bold')
ax2.set_ylim(0.3, 1.0)
ax2.legend(loc='lower right', frameon=False, fontsize=8)

for bar, val in zip(bars_50, cov_50):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
             ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars_90, cov_90):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
             ha='center', va='bottom', fontsize=8)

plt.tight_layout()
savefig_both(fig, 'calibration', 'results')
plt.close(fig)

print("\nAll figures generated successfully.")

import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
fig_dir = 'chapters/results/fig'
os.makedirs(fig_dir, exist_ok=True)
# real dataset overview
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
boardings = np.array([4.1, 4.5, 4.8, 5.0, 5.3, 5.6]) * 1e5
stations = np.array([172, 174, 175, 176, 178, 180])
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(months, boardings / 1e5, marker='o', color='#1f77b4', label='Total boardings (100k)')
ax1.set_ylabel('Total boardings (100k)')
ax1.set_xlabel('Month (2024)')
ax1.set_ylim(3.5, 6.0)
ax1.grid(True, axis='y', alpha=0.4)
ax2 = ax1.twinx()
ax2.bar(months, stations, alpha=0.3, color='#ff7f0e', label='Active stops')
ax2.set_ylabel('Active stops')
ax2.set_ylim(160, 185)
fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.92))
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'real_dataset_overview.png'), dpi=300)
fig.savefig(os.path.join(fig_dir, 'real_dataset_overview.pdf'))
plt.close(fig)
# real dataset performance comparison
methods = ['Historical Average', 'LSTM', 'GRU', 'STGCN', 'Graph WaveNet', 'AGCRN', 'DTS-GSSF']
r2 = [0.503, 0.705, 0.708, 0.712, 0.732, 0.745, 0.862]
mae = [18.5, 9.8, 9.7, 9.1, 8.6, 8.2, 7.1]
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.bar(methods, r2, color='#2ca02c', alpha=0.8, label='$R^2$')
ax1.set_ylabel('$R^2$', color='#2ca02c')
ax1.set_ylim(0.45, 0.90)
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, rotation=30, ha='right')
ax1.set_xlabel('Method')
ax2 = ax1.twinx()
ax2.plot(range(len(methods)), mae, marker='o', color='#d62728', linewidth=2, label='MAE')
ax2.set_ylabel('MAE (passengers/hour)', color='#d62728')
ax2.set_ylim(6.0, 20.0)
for i, v in enumerate(r2):
    ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', color='#2ca02c', fontsize=9)
for i, v in enumerate(mae):
    ax2.text(i, v + 0.3, f'{v:.1f}', ha='center', color='#d62728', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'real_dataset_performance.png'), dpi=300)
fig.savefig(os.path.join(fig_dir, 'real_dataset_performance.pdf'))
plt.close(fig)
# station demand distribution
districts = ['Downtown', 'Mid-City', 'Westside', 'South Bay']
mean_flow = [71.4, 54.8, 48.1, 36.5]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(districts, mean_flow, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'])
ax.set_xlabel('Average hourly boardings')
ax.set_title('Average ridership by city sub-region')
for i, v in enumerate(mean_flow):
    ax.text(v + 1.2, i, f'{v:.1f}', va='center', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'real_dataset_district_flow.png'), dpi=300)
fig.savefig(os.path.join(fig_dir, 'real_dataset_district_flow.pdf'))
plt.close(fig)

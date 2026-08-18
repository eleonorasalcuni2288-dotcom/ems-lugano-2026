"""
Real-Data Validation — Poster Figure (matplotlib, static, poster-ready)
================================================================================
Three panels, one figure: MI complications (forest plot of the 7 BH
survivors), FRED-MD (4x4 method x K robustness grid), and trading (same
grid format as FRED-MD, once its bootstrap CI is available).

Column names confirmed directly against the actual CSVs (not assumed):
  bootstrap_ci_fredmd_results.csv     : method, K, mean_advantage, ci_lo, ci_hi, ..., robust
  bootstrap_ci_mi_survivors_results.csv: complication, method, K, mean_advantage, ci_lo, ci_hi, ...
    (no 'robust' column here -> derived as ci_lo > 0, i.e. CI entirely above zero)
  bootstrap_ci_trading_results.csv    : same schema as bootstrap_ci_fredmd_results.csv
    (produced by trading_downstream_validation.py; this script degrades
    gracefully with a "pending" placeholder panel if the file doesn't exist yet)

Does not modify any existing file. Output: real_data_comparison.png (300 dpi).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

COLOR_ROBUST  = '#0ca30c'
COLOR_FRAGILE = '#d03b3b'
COLOR_ROBUST_SOFT  = '#0ca30c22'
COLOR_FRAGILE_SOFT = '#d03b3b22'

K_VALUES = [3, 5, 10, 16]
METHOD_ORDER = ['MI_perfeat', 'II_perfeat', 'II_joint', 'DII_L1']
METHOD_LABELS = {'MI_perfeat': 'MI (per-feature)', 'II_perfeat': 'II (per-feature)',
                  'II_joint': 'II-joint', 'DII_L1': 'DII+L1'}


# =============================================================================
# Panel A — MI complications: forest plot of the 7 BH survivors
# =============================================================================
def plot_mi_survivors(ax):
    df = pd.read_csv('../mi_complications/bootstrap_ci_mi_survivors_results.csv')
    df['label'] = df['complication'] + ' · ' + df['method'] + ' · K=' + df['K'].astype(str)
    df['robust'] = df['ci_lo'] > 0
    df = df.sort_values('mean_advantage', ascending=True).reset_index(drop=True)

    colors = [COLOR_ROBUST if r else COLOR_FRAGILE for r in df['robust']]
    y_pos = np.arange(len(df))

    for i, row in df.iterrows():
        ax.plot([row['ci_lo'], row['ci_hi']], [i, i], color=colors[i], lw=2, zorder=2)
        ax.plot([row['ci_lo'], row['ci_lo']], [i-0.12, i+0.12], color=colors[i], lw=2, zorder=2)
        ax.plot([row['ci_hi'], row['ci_hi']], [i-0.12, i+0.12], color=colors[i], lw=2, zorder=2)

    ax.scatter(df['mean_advantage'], y_pos, color=colors, s=70, zorder=3,
               edgecolors='white', linewidths=1)
    ax.axvline(0, color='#999', ls='--', lw=1.2, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['label'], fontsize=9)
    ax.set_xlabel('Downstream advantage (Top-K accuracy − random baseline)', fontsize=10)
    ax.set_title('Clinical (MI complications)\n7 Benjamini–Hochberg survivors, bootstrap CI',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25, axis='x')
    ax.set_xlim(-0.03, 0.13)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_ROBUST, markersize=8, label='robust (CI > 0)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_FRAGILE, markersize=8, label='fragile (CI crosses 0)'),
    ]
    ax.legend(handles=legend_elems, loc='lower right', fontsize=8, frameon=True)


# =============================================================================
# Panel B/C — method x K robustness grid (FRED-MD, trading)
# =============================================================================
def plot_grid(ax, csv_path, title):
    if not os.path.exists(csv_path):
        ax.axis('off')
        ax.text(0.5, 0.55, 'Bootstrap run in progress', ha='center', va='center',
                 fontsize=12, fontweight='bold', color='#888', transform=ax.transAxes)
        ax.text(0.5, 0.42,
                 'trading_downstream_validation.py\n(same protocol as FRED-MD)\nnot yet complete',
                 ha='center', va='center', fontsize=9, color='#aaa', transform=ax.transAxes)
        ax.set_title(title, fontsize=11, fontweight='bold')
        return

    df = pd.read_csv(csv_path)
    methods = [m for m in METHOD_ORDER if m in df['method'].unique()]
    n_rows, n_cols = len(methods), len(K_VALUES)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels([f'K={k}' for k in K_VALUES], fontsize=9)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, method in enumerate(methods):
        for j, k in enumerate(K_VALUES):
            row = df[(df.method == method) & (df.K == k)]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            robust = bool(row['robust'])
            face = COLOR_ROBUST_SOFT if robust else COLOR_FRAGILE_SOFT
            edge = COLOR_ROBUST if robust else COLOR_FRAGILE
            rect = plt.Rectangle((j + 0.04, i + 0.04), 0.92, 0.92,
                                  facecolor=face, edgecolor=edge, linewidth=1.5)
            ax.add_patch(rect)
            adv = row['mean_advantage']
            ax.text(j + 0.5, i + 0.38, f"{adv:+.3f}", ha='center', va='center',
                     fontsize=9.5, fontweight='bold', color=edge)
            ax.text(j + 0.5, i + 0.66, f"[{row['ci_lo']:.2f},{row['ci_hi']:.2f}]",
                     ha='center', va='center', fontsize=7, color=edge, alpha=0.85)

    ax.set_title(title, fontsize=11, fontweight='bold')


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1, 1], wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    plot_mi_survivors(ax0)

    ax1 = fig.add_subplot(gs[1])
    plot_grid(ax1, '../fredmd/bootstrap_ci_fredmd_results.csv',
              'Macro-financial (FRED-MD)\nN=794, p=120 — single target')

    ax2 = fig.add_subplot(gs[2])
    plot_grid(ax2, '../trading/bootstrap_ci_trading_results.csv',
              'Trading (train2.csv)\nN=5000, p=27 — single target')

    fig.suptitle('Real-Data Validation — Ground-Truth-Free Protocol, Three Independent Domains',
                  fontsize=14, fontweight='bold', y=1.03)
    plt.savefig('real_data_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: real_data_comparison.png")

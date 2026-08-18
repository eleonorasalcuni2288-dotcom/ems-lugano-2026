"""
Poster figures — generated directly from the project's real CSV outputs.
Run this from inside the feature_selection_analysis/ folder, where all
the CSVs already exist on disk.

Column names verified directly against the actual CSVs before writing this
(two real mismatches fixed relative to an earlier draft):
  bootstrap_ci_fredmd_results.csv:      'mean_advantage', not 'advantage'
  bootstrap_ci_mi_survivors_results.csv: 'mean_advantage' (not 'advantage'),
    no 'label' column (derived from complication+method+K), no 'robust'
    column (derived as ci_lo > 0).

Requires: matplotlib, numpy, pandas
    pip install matplotlib numpy pandas
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 15,
    'axes.titlesize': 17,
    'axes.labelsize': 16,
    'legend.fontsize': 13,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
})

NAVY = '#102A43'
ACCENT = '#0076A8'
RED = '#C0392B'
GREEN = '#1E8449'
ORANGE = '#D68910'
GRAY = '#8A94A6'
PURPLE = '#7D3C98'
TEAL = '#138D8D'

COLORS = {'MI_perfeat': ACCENT, 'II_perfeat': GREEN, 'II_joint': ORANGE, 'DII_L1': RED,
          'MINE': PURPLE, 'RF': TEAL}
LABELS = {'MI_perfeat': 'MI (per-feature)', 'II_perfeat': 'II (per-feature)',
          'II_joint': 'II (joint, LOO)', 'DII_L1': 'DII + L1',
          'MINE': 'MINE', 'RF': 'Random Forest'}



# =============================================================================
# FIGURE 1 — Synthetic scalability: tau with bootstrap CI, and synergy-pair rank
# =============================================================================
# Reads: bootstrap_ci_synthetic_results.csv (p=27),
#        bootstrap_ci_synthetic_highdim_results.csv (p=50, p=105),
#        simulation_study_v6_highdim_scalability.csv (point estimates + XOR ranks)
#        mine_synthetic_highdim_results.csv / _point_estimates.csv (MINE)
#        rf_synthetic_highdim_results.csv / _point_estimates.csv (Random Forest)
# Now covers all 6 methods that have full p=27/50/105 bootstrap + point-estimate
# coverage (InfoNCE excluded — no bootstrap CI, never extended past p=27).

boot_27 = pd.read_csv('../synthetic/bootstrap_ci_synthetic_results.csv')
boot_27['p'] = 27
boot_highdim = pd.read_csv('../synthetic/bootstrap_ci_synthetic_highdim_results.csv')
boot_mine = pd.read_csv('../synthetic/mine_synthetic_highdim_results.csv')
boot_rf = pd.read_csv('../synthetic/rf_synthetic_highdim_results.csv')
boot_all = pd.concat([boot_27, boot_highdim, boot_mine, boot_rf], ignore_index=True)

point_orig = pd.read_csv('../synthetic/simulation_study_v6_highdim_scalability.csv')[
    ['p', 'method', 'tau', 'syn_rank_1', 'syn_rank_2']]
mine_pts = pd.read_csv('../synthetic/mine_synthetic_highdim_point_estimates.csv') \
    .drop_duplicates('p')[['p', 'tau', 'syn_rank_1', 'syn_rank_2']]
mine_pts['method'] = 'MINE'
rf_pts = pd.read_csv('../synthetic/rf_synthetic_highdim_point_estimates.csv') \
    .drop_duplicates('p')[['p', 'tau', 'syn_rank_1', 'syn_rank_2']]
rf_pts['method'] = 'RF'
point_df = pd.concat([point_orig, mine_pts, rf_pts], ignore_index=True)

p_levels = [27, 50, 105]
methods = ['MI_perfeat', 'II_perfeat', 'II_joint', 'DII_L1']  # used by Figures 2 & 3
methods_fig1 = methods + ['MINE', 'RF']  # Figure 1 only (fuller method coverage)

fig, axes = plt.subplots(1, 2, figsize=(17, 4.2))

ax = axes[0]
offsets = {'MI_perfeat': -3.5, 'II_perfeat': -2.1, 'II_joint': -0.7,
           'DII_L1': 0.7, 'MINE': 2.1, 'RF': 3.5}
for method in methods_fig1:
    means, los, his, pts = [], [], [], []
    for p in p_levels:
        row = boot_all[(boot_all.p == p) & (boot_all.method == method)]
        means.append(row.tau_mean.values[0] if len(row) else np.nan)
        los.append(row.ci_lo.values[0] if len(row) else np.nan)
        his.append(row.ci_hi.values[0] if len(row) else np.nan)
        prow = point_df[(point_df.p == p) & (point_df.method == method)]
        pts.append(prow.tau.values[0] if len(prow) else np.nan)
    x = np.array(p_levels) + offsets[method]
    err_lo = [m - lo for m, lo in zip(means, los)]
    err_hi = [hi - m for hi, m in zip(his, means)]
    ax.errorbar(x, means, yerr=[err_lo, err_hi], fmt='o', capsize=4,
                color=COLORS[method], label=LABELS[method], markersize=7,
                elinewidth=1.8, capthick=1.8)
    ax.scatter(x, pts, marker='x', color=COLORS[method], s=55, zorder=5)
ax.set_xticks(p_levels)
ax.set_xlabel('Number of features $p$')
ax.set_ylabel(r"Kendall's $\tau$ vs. ground truth")
ax.set_title('(a) Ranking accuracy: bootstrap 95% CI\n(x = full-sample point estimate)')
ax.axhline(0, color=GRAY, lw=0.8, ls=':')
legend_handles, legend_labels = ax.get_legend_handles_labels()

ax = axes[1]
for method in methods_fig1:
    ranks = []
    for p in p_levels:
        row = point_df[(point_df.p == p) & (point_df.method == method)]
        if len(row) == 0:
            ranks.append(np.nan)
            continue
        r1, r2 = row.syn_rank_1.values[0], row.syn_rank_2.values[0]
        ranks.append((r1 + r2) / 2)
    ax.plot(p_levels, ranks, 'o-', color=COLORS[method], label=LABELS[method],
             markersize=7, linewidth=2)
ax.set_xticks(p_levels)
ax.set_xlabel('Number of features $p$')
ax.set_ylabel('Avg. rank of synergistic (XOR) pair\n(lower = correctly detected)')
ax.set_title('(b) Synergy-pair detection vs. dimensionality')

fig.legend(legend_handles, legend_labels, loc='upper center',
           bbox_to_anchor=(0.5, 1.14), ncol=6, frameon=False, fontsize=12)

plt.tight_layout()
plt.savefig('fig_scalability.pdf', bbox_inches='tight')
plt.savefig('fig_scalability.png', bbox_inches='tight')
plt.close()
print("Saved fig_scalability.{pdf,png}")

# =============================================================================
# FIGURE 2 — FRED-MD bootstrap advantage (4 methods x 4 K)
# =============================================================================
# Reads: bootstrap_ci_fredmd_results.csv
# real columns: method, K, mean_advantage, ci_lo, ci_hi, std, frac_positive, B, robust

fredmd = pd.read_csv('../fredmd/bootstrap_ci_fredmd_results.csv')
K_values = sorted(fredmd.K.unique())

fig, ax = plt.subplots(figsize=(11, 4.2))
width = 0.2
x = np.arange(len(K_values))
for i, method in enumerate(methods):
    means, los, his = [], [], []
    for k in K_values:
        row = fredmd[(fredmd.method == method) & (fredmd.K == k)]
        means.append(row.mean_advantage.values[0])
        los.append(row.mean_advantage.values[0] - row.ci_lo.values[0])
        his.append(row.ci_hi.values[0] - row.mean_advantage.values[0])
    ax.bar(x + (i - 1.5) * width, means, width, color=COLORS[method],
           label=LABELS[method], yerr=[los, his], capsize=3,
           error_kw=dict(elinewidth=1.3, capthick=1.3))
ax.axhline(0, color='black', lw=1)
ax.set_xticks(x)
ax.set_xticklabels([f'K={k}' for k in K_values])
ax.set_ylabel('Downstream advantage\n(method $-$ random baseline)', fontsize=14)
ax.set_title('FRED-MD: bootstrap 95% CI on predictive advantage', pad=45)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), frameon=False, ncol=4, fontsize=12)
fig.subplots_adjust(left=0.12, right=0.98, top=0.72, bottom=0.15)
plt.savefig('fig_fredmd.pdf')
plt.savefig('fig_fredmd.png')
plt.close()
print("Saved fig_fredmd.{pdf,png}")

# =============================================================================
# FIGURE 3 — MI complications: BH-survivors bootstrap CI
# =============================================================================
# Reads: bootstrap_ci_mi_survivors_results.csv
# real columns: complication, method, K, mean_advantage, ci_lo, ci_hi, std,
# frac_positive, B  (no 'label' or 'robust' column — both derived below)

mi_surv = pd.read_csv('../mi_complications/bootstrap_ci_mi_survivors_results.csv')
mi_surv['label'] = (mi_surv['complication'] + ' · ' + mi_surv['method']
                     + ' · K=' + mi_surv['K'].astype(str))
mi_surv['robust'] = mi_surv['ci_lo'] > 0

fig, ax = plt.subplots(figsize=(10, 3.8))
y_pos = np.arange(len(mi_surv))
for i, row in mi_surv.iterrows():
    color = GREEN if row.robust else GRAY
    ax.errorbar(row.mean_advantage, i, xerr=[[row.mean_advantage - row.ci_lo],
                [row.ci_hi - row.mean_advantage]], fmt='o', color=color,
                capsize=4, markersize=9, elinewidth=2, capthick=2)
ax.axvline(0, color='black', lw=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(mi_surv.label.values, fontsize=12)
ax.set_xlabel('Downstream accuracy advantage (bootstrap 95% CI)')
ax.set_title('MI complications: all 7 configurations surviving\nBenjamini-Hochberg correction')
ax.invert_yaxis()
from matplotlib.lines import Line2D
legend_elems = [Line2D([0], [0], marker='o', color='w', markerfacecolor=GREEN,
                        markersize=10, label='Robust (CI excludes 0)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=GRAY,
                        markersize=10, label='Fragile (CI crosses 0)')]
ax.legend(handles=legend_elems, loc='lower right', frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig('fig_mi_survivors.pdf', bbox_inches='tight')
plt.savefig('fig_mi_survivors.png', bbox_inches='tight')
plt.close()
print("Saved fig_mi_survivors.{pdf,png}")

# =============================================================================
# FIGURE 4 — Trading (train2.csv) bootstrap advantage (4 methods x 4 K)
# =============================================================================
# Reads: bootstrap_ci_trading_results.csv (same schema as fredmd's file).
# Same bar-chart style as Figure 2 for direct visual consistency across the
# two real-data domains, but trading is NOT high-dimensional (p=27, vs
# FRED-MD's p=120) and its CI is coarser: B=12 (B=8 for DII_L1) vs FRED-MD's
# B=100 (B=15), because N=5000 makes the O(N^2) knn_loo_accuracy step in
# every draw ~40x more expensive than on FRED-MD's N=794 — a documented,
# unbiased trade-off (wider CI, not a distorted one), not an oversight.

trading = pd.read_csv('../trading/bootstrap_ci_trading_results.csv')
K_values_tr = sorted(trading.K.unique())

fig, ax = plt.subplots(figsize=(11, 4.2))
width = 0.2
x = np.arange(len(K_values_tr))
for i, method in enumerate(methods):
    means, los, his = [], [], []
    for k in K_values_tr:
        row = trading[(trading.method == method) & (trading.K == k)]
        means.append(row.mean_advantage.values[0])
        los.append(row.mean_advantage.values[0] - row.ci_lo.values[0])
        his.append(row.ci_hi.values[0] - row.mean_advantage.values[0])
    ax.bar(x + (i - 1.5) * width, means, width, color=COLORS[method],
           label=LABELS[method], yerr=[los, his], capsize=3,
           error_kw=dict(elinewidth=1.3, capthick=1.3))
ax.axhline(0, color='black', lw=1)
ax.set_xticks(x)
ax.set_xticklabels([f'K={k}' for k in K_values_tr])
ax.set_ylabel('Downstream advantage\n(method $-$ random baseline)', fontsize=14)
ax.set_title('Trading (train2.csv, p=27): bootstrap 95% CI on predictive advantage', pad=45)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), frameon=False, ncol=4, fontsize=12)
fig.subplots_adjust(left=0.12, right=0.98, top=0.72, bottom=0.15)
plt.savefig('fig_trading.pdf')
plt.savefig('fig_trading.png')
plt.close()
print("Saved fig_trading.{pdf,png}")

print("\nAll figures generated successfully from real CSV data.")

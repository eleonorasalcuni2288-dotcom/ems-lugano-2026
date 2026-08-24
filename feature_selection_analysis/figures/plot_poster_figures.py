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

NAVY = '#4F5A6B'
ACCENT = '#1F5FAD'
RED = '#D8432B'
GREEN = '#3D8F52'
ORANGE = '#C4941F'
GRAY = '#8A94A6'
PURPLE = '#7D52B0'
TEAL = '#0F9E73'
AZURE = '#2CA8E0'

COLORS = {'MI_perfeat': ACCENT, 'II_perfeat': AZURE, 'II_joint': ORANGE, 'DII_L1': TEAL,
          'MINE': NAVY, 'RF': RED, 'LASSO': PURPLE}
LABELS = {'MI_perfeat': 'MI (per-feature)', 'II_perfeat': 'II (per-feature)',
          'II_joint': 'II (joint, LOO)', 'DII_L1': 'DII + L1',
          'MINE': 'MINE', 'RF': 'Random Forest', 'LASSO': 'LASSO'}



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
boot_lasso = pd.read_csv('../synthetic/lasso_synthetic_highdim_results.csv')
boot_all = pd.concat([boot_27, boot_highdim, boot_mine, boot_rf, boot_lasso], ignore_index=True)

point_orig = pd.read_csv('../synthetic/simulation_study_v6_highdim_scalability.csv')[
    ['p', 'method', 'tau', 'syn_rank_1', 'syn_rank_2']]
mine_pts = pd.read_csv('../synthetic/mine_synthetic_highdim_point_estimates.csv') \
    .drop_duplicates('p')[['p', 'tau', 'syn_rank_1', 'syn_rank_2']]
mine_pts['method'] = 'MINE'
rf_pts = pd.read_csv('../synthetic/rf_synthetic_highdim_point_estimates.csv') \
    .drop_duplicates('p')[['p', 'tau', 'syn_rank_1', 'syn_rank_2']]
rf_pts['method'] = 'RF'
lasso_pts = pd.read_csv('../synthetic/lasso_synthetic_highdim_point_estimates.csv') \
    .drop_duplicates('p')[['p', 'tau', 'syn_rank_1', 'syn_rank_2']]
lasso_pts['method'] = 'LASSO'
point_df = pd.concat([point_orig, mine_pts, rf_pts, lasso_pts], ignore_index=True)

p_levels = [27, 50, 105]
methods = ['MI_perfeat', 'II_perfeat', 'II_joint', 'DII_L1']  # used by Figures 2 & 3
methods_fig1 = methods + ['MINE', 'RF', 'LASSO']  # Figure 1 only (fuller method coverage)

fig, axes = plt.subplots(1, 2, figsize=(17, 5.4))

ax = axes[0]
offsets = {'MI_perfeat': -7.2, 'II_perfeat': -4.8, 'II_joint': -2.4,
           'DII_L1': 0, 'MINE': 2.4, 'RF': 4.8, 'LASSO': 7.2}
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
ax.set_xlabel('Number of features $p$', fontsize=20)
ax.set_ylabel("Kendall's $\\tau$\n(agreement with ground truth)", fontsize=20)
ax.set_title('(a) Ranking accuracy: bootstrap 95% CI\n(x = full-sample point estimate)', fontsize=19)
ax.axhline(0, color=GRAY, lw=0.8, ls=':')
# Separator between per-feature methods (MI_perfeat, II_perfeat) and joint
# methods (II_joint, DII_L1, MINE, RF, LASSO) -- see Figure 2 for rationale.
for p in p_levels:
    ax.axvline(p - 3.6, color=GRAY, lw=1, ls=':', zorder=0)
ax.tick_params(axis='both', labelsize=17)
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
ax.set_xlabel('Number of features $p$', fontsize=20)
ax.set_ylabel('Avg. synergy-pair rank\n(lower = correctly detected)', fontsize=20)
ax.set_title('(b) Synergy-pair detection vs. dimensionality', fontsize=19)
ax.tick_params(axis='both', labelsize=17)

fig.subplots_adjust(left=0.075, right=0.98, top=0.68, bottom=0.18, wspace=0.38)
fig.legend(legend_handles, legend_labels, loc='upper center',
           bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=16)

plt.savefig('fig_scalability.pdf', bbox_inches='tight')
plt.savefig('fig_scalability.png', bbox_inches='tight')
plt.close()
print("Saved fig_scalability.{pdf,png}")

# =============================================================================
# FIGURE 2 — FRED-MD bootstrap advantage (7 methods x 4 K)
# =============================================================================
# Reads: bootstrap_ci_fredmd_results.csv (4 core methods, B=100/15)
#        bootstrap_ci_fredmd_lasso_rf_results.csv (LASSO+RF, B=15, same
#        target/protocol, added for a fair comparison against DII_L1)
#        bootstrap_ci_fredmd_mine_results.csv (MINE, B=15, same protocol)
# real columns: method, K, mean_advantage, ci_lo, ci_hi, std, frac_positive, B, robust

fredmd_core = pd.read_csv('../fredmd/bootstrap_ci_fredmd_results.csv')
fredmd_extra = pd.read_csv('../fredmd/bootstrap_ci_fredmd_lasso_rf_results.csv')
fredmd_mine = pd.read_csv('../fredmd/bootstrap_ci_fredmd_mine_results.csv')
fredmd = pd.concat([fredmd_core, fredmd_extra, fredmd_mine], ignore_index=True)
K_values = sorted(fredmd.K.unique())
methods_fig2 = methods + ['LASSO', 'RF', 'MINE']  # 4 core + LASSO + RF + MINE

fig, ax = plt.subplots(figsize=(16, 5.6))
n_methods = len(methods_fig2)
width = 0.115
x = np.arange(len(K_values))
for i, method in enumerate(methods_fig2):
    means, los, his = [], [], []
    for k in K_values:
        row = fredmd[(fredmd.method == method) & (fredmd.K == k)]
        means.append(row.mean_advantage.values[0])
        los.append(row.mean_advantage.values[0] - row.ci_lo.values[0])
        his.append(row.ci_hi.values[0] - row.mean_advantage.values[0])
    ax.bar(x + (i - (n_methods - 1) / 2) * width, means, width, color=COLORS[method],
           label=LABELS[method], yerr=[los, his], capsize=3,
           error_kw=dict(elinewidth=1.3, capthick=1.3))
ax.axhline(0, color='black', lw=1)
# Separator between per-feature methods (MI_perfeat, II_perfeat) and joint
# methods (II_joint, DII_L1, LASSO, RF, MINE) -- these are evaluated under
# different procedures (independent per-feature scores vs. joint/LOO
# importance), so the split makes that distinction visible rather than
# implying a single uniform comparison.
n_perfeat = 2
boundary_offset = (n_perfeat - 0.5 - (n_methods - 1) / 2) * width
for k_idx in x:
    ax.axvline(k_idx + boundary_offset, color=GRAY, lw=1, ls=':', zorder=0)
ax.set_xticks(x)
ax.set_xticklabels([f'K={k}' for k in K_values], fontsize=17)
ax.set_ylabel('Downstream advantage\n(method $-$ random baseline)', fontsize=20)
ax.tick_params(axis='y', labelsize=17)
ax.set_title('FRED-MD: bootstrap 95% CI on predictive advantage', pad=55, fontsize=23)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), frameon=False, ncol=4, fontsize=16)
fig.subplots_adjust(left=0.09, right=0.98, top=0.64, bottom=0.13)
plt.savefig('fig_fredmd.pdf')
plt.savefig('fig_fredmd.png')
plt.close()
print("Saved fig_fredmd.{pdf,png}")

# =============================================================================
# FIGURE 3 — Post-infarction complications: BH-survivors bootstrap CI
# =============================================================================
# Reads: bootstrap_ci_mi_survivors_bh264_results.csv (12 configurations
# surviving BH correction on the combined 264-test family: the original 176
# tests for MI/II/DII/DII+L1, plus 88 new tests adding LASSO+RF on the same
# 11 per-complication targets, same fair-comparison protocol as FRED-MD and
# Trading. See post_infarction_per_target_lasso_rf.py / the combined BH step
# for how this file was built.)
# real columns: complication, method, K, mean_advantage, ci_lo, ci_hi, std,
# frac_positive, B  (no 'label' or 'robust' column — both derived below)

mi_surv = pd.read_csv('../post_infarction/bootstrap_ci_mi_survivors_bh264_results.csv')
mi_surv['label'] = (mi_surv['complication'] + ' · ' + mi_surv['method']
                     + ' · K=' + mi_surv['K'].astype(str))
mi_surv['robust'] = mi_surv['ci_lo'] > 0

fig, ax = plt.subplots(figsize=(10, 4.1))
y_pos = np.arange(len(mi_surv))
for i, row in mi_surv.iterrows():
    color = GREEN if row.robust else GRAY
    ax.errorbar(row.mean_advantage, i, xerr=[[row.mean_advantage - row.ci_lo],
                [row.ci_hi - row.mean_advantage]], fmt='o', color=color,
                capsize=4, markersize=9, elinewidth=2, capthick=2)
ax.axvline(0, color='black', lw=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(mi_surv.label.values, fontsize=12)
ax.set_xlabel('Downstream accuracy advantage (bootstrap 95% CI)', fontsize=14)
fig.suptitle('Post-infarction complications: all 12 configurations surviving\nBenjamini-Hochberg correction (264 tests, 6 methods)', fontsize=15)
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
# FIGURE 4 — Trading bootstrap advantage (7 methods x 4 K)
# =============================================================================
# Reads: bootstrap_ci_trading_results.csv (4 core methods, same schema as
#        fredmd's file), bootstrap_ci_trading_lasso_rf_results.csv (LASSO+RF,
#        B=15), bootstrap_ci_trading_mine_results.csv (MINE, B=15) — all
#        added for a fair comparison against DII_L1, same target/protocol.
# Trading is NOT high-dimensional (p=27, vs FRED-MD's p=120) and the core
# methods' CI is coarser: B=100 (B=15 for DII_L1) is still used, but N=5000
# makes the O(N^2) knn_loo_accuracy step in every draw ~40x more expensive
# than on FRED-MD's N=794 — a documented, unbiased trade-off (wider CI, not
# a distorted one), not an oversight.

trading_core = pd.read_csv('../trading/bootstrap_ci_trading_results.csv')
trading_extra = pd.read_csv('../trading/bootstrap_ci_trading_lasso_rf_results.csv')
trading_mine = pd.read_csv('../trading/bootstrap_ci_trading_mine_results.csv')
trading = pd.concat([trading_core, trading_extra, trading_mine], ignore_index=True)
K_values_tr = sorted(trading.K.unique())
methods_fig4 = methods + ['LASSO', 'RF', 'MINE']  # 4 core + LASSO + RF + MINE

fig, ax = plt.subplots(figsize=(16, 5.6))
n_methods_tr = len(methods_fig4)
width = 0.115
x = np.arange(len(K_values_tr))
for i, method in enumerate(methods_fig4):
    means, los, his = [], [], []
    for k in K_values_tr:
        row = trading[(trading.method == method) & (trading.K == k)]
        means.append(row.mean_advantage.values[0])
        los.append(row.mean_advantage.values[0] - row.ci_lo.values[0])
        his.append(row.ci_hi.values[0] - row.mean_advantage.values[0])
    ax.bar(x + (i - (n_methods_tr - 1) / 2) * width, means, width, color=COLORS[method],
           label=LABELS[method], yerr=[los, his], capsize=3,
           error_kw=dict(elinewidth=1.3, capthick=1.3))
ax.axhline(0, color='black', lw=1)
# Separator between per-feature methods (MI_perfeat, II_perfeat) and joint
# methods (II_joint, DII_L1, LASSO, RF, MINE) -- see Figure 2 for rationale.
n_perfeat = 2
boundary_offset_tr = (n_perfeat - 0.5 - (n_methods_tr - 1) / 2) * width
for k_idx in x:
    ax.axvline(k_idx + boundary_offset_tr, color=GRAY, lw=1, ls=':', zorder=0)
ax.set_xticks(x)
ax.set_xticklabels([f'K={k}' for k in K_values_tr], fontsize=17)
ax.set_ylabel('Downstream advantage\n(method $-$ random baseline)', fontsize=20)
ax.tick_params(axis='y', labelsize=17)
ax.set_title('Trading (p=27): bootstrap 95% CI on predictive advantage', pad=55, fontsize=23)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), frameon=False, ncol=4, fontsize=16)
fig.subplots_adjust(left=0.09, right=0.98, top=0.64, bottom=0.13)
plt.savefig('fig_trading.pdf')
plt.savefig('fig_trading.png')
plt.close()
print("Saved fig_trading.{pdf,png}")

# =============================================================================
# FIGURE 5 — Frequentist coverage check: current vs. Politis-Romano-Wolf CI
# =============================================================================
# Reads: ../synthetic/coverage_check_results.csv (40 independent replications,
# II-joint p=27). theta_true = 0.2763, hardcoded from coverage_check_summary.txt
# (avg tau over 200 independent N=2000 datasets, seeds 1000-1199, disjoint from
# the 40 test replications below).

cov = pd.read_csv('../synthetic/coverage_check_results.csv')
theta_true = 0.2763
n_reps = len(cov)
cur_cov_pct = cov.current_covers.mean()
corr_cov_pct = cov.corrected_covers.mean()

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
panels = [
    (axes[0], 'current_ci_lo', 'current_ci_hi', 'current_covers',
     f'Current procedure\n{cur_cov_pct:.1%} coverage ({int(cov.current_covers.sum())}/{n_reps})'),
    (axes[1], 'corrected_ci_lo', 'corrected_ci_hi', 'corrected_covers',
     f'Politis-Romano-Wolf correction\n{corr_cov_pct:.1%} coverage ({int(cov.corrected_covers.sum())}/{n_reps})'),
]
y = np.arange(n_reps)
for ax, lo_col, hi_col, cov_col, title in panels:
    colors = np.where(cov[cov_col], GREEN, RED)
    ax.hlines(y, cov[lo_col], cov[hi_col], color=colors, linewidth=1.8)
    ax.axvline(theta_true, color='black', lw=1.3, ls='--', zorder=0)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r"Kendall's $\tau$ (II-joint, p=27)", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
axes[0].set_ylabel('Independent replication', fontsize=16)
axes[0].set_yticks([])
fig.suptitle(r'Frequentist coverage: 40 independent 95% CIs vs. $\theta_{true}$='
             + f'{theta_true}', y=1.02, fontsize=18)
from matplotlib.lines import Line2D
legend_elems = [Line2D([0], [0], color=GREEN, lw=2.5, label='Contains θ_true'),
                Line2D([0], [0], color=RED, lw=2.5, label='Misses θ_true'),
                Line2D([0], [0], color='black', lw=1.3, ls='--', label='θ_true')]
fig.legend(handles=legend_elems, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=3, frameon=False, fontsize=13)
fig.tight_layout()
plt.savefig('fig_coverage.pdf', bbox_inches='tight')
plt.savefig('fig_coverage.png', bbox_inches='tight')
plt.close()
print("Saved fig_coverage.{pdf,png}")

print("\nAll figures generated successfully from real CSV data.")

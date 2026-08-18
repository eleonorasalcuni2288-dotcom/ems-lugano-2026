"""
Simulation Study: Comparative Analysis of Feature Selection Methods
====================================================================
Generates a synthetic dataset with known ground truth to evaluate
Information Imbalance (II), Differentiable Information Imbalance (DII),
and Mutual Information (MI) under controlled conditions.

Feature groups:
  Group 1 - Linear informative (varying SNR)
  Group 2 - Nonlinear symmetric (quadratic, sinusoidal, absolute value)
  Group 3 - Nonlinear directional (monotone, sigmoid, lagged)
  Group 4 - AR(1) time series (with/without dependence on Y)
  Group 5 - Pure noise (irrelevant features)

Ground truth: Groups 1, 2, 3, 4a-4b are informative; 4c and Group 5 are not.

Fair comparison design:
  - MI  measures I(Xi ; Y)  where Y = (Z > 0) is binary
  - II  measures II(Xi -> Z) where Z is the continuous latent signal
  - DII measures DII(Xi -> Z) where Z is the continuous latent signal
  All three methods thus evaluate how informative Xi is about the SAME
  underlying signal Z (binary or continuous form), ensuring comparability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, kendalltau, spearmanr
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 0. REPRODUCIBILITY
# =============================================================================
SEED = 42
np.random.seed(SEED)
N = 2000   # number of observations
NOISE = 0.5  # base noise level

print("=" * 70)
print("SIMULATION STUDY: II vs MI vs DII on Synthetic Data")
print("=" * 70)

# =============================================================================
# 1. GENERATE SYNTHETIC DATASET
# =============================================================================
print("\n[1/5] Generating synthetic dataset...")

# Latent continuous signal driving the target
Z = np.random.randn(N)

# Binary target (as in the real financial dataset)
Y = (Z > 0).astype(int)

# --- Group 1: Linear informative (varying SNR) ---
X1 = Z + 0.3 * np.random.randn(N)        # strong linear
X2 = Z + 1.0 * np.random.randn(N)        # medium linear
X3 = Z + 2.5 * np.random.randn(N)        # weak linear

# --- Group 2: Nonlinear symmetric ---
# MI handles these because it is symmetric (I(Xi;Y) = I(Y;Xi)).
# II/DII struggle: points with Z=+z and Z=-z are neighbors in Xi but NOT in Z,
# so the nearest-neighbor mapping breaks -> II/DII assign low importance.
X4 = Z**2 + NOISE * np.random.randn(N)                      # quadratic
X5 = np.sin(np.pi * Z) + NOISE * np.random.randn(N)         # sinusoidal
X6 = np.abs(Z) + NOISE * np.random.randn(N)                 # absolute value

# --- Group 3: Nonlinear directional (monotone) ---
# Monotone transformations preserve neighborhood order -> II/DII detect well.
# MI also detects but may need more bins for complex shapes.
X7 = Z + 0.3 * Z**3 + NOISE * np.random.randn(N)            # cubic (monotone)
X8 = 1 / (1 + np.exp(-2 * Z)) + NOISE * np.random.randn(N) # sigmoid
X9 = np.concatenate([[0], Z[:-1]]) + NOISE * np.random.randn(N)  # lagged (directional)

# --- Group 4: AR(1) time series ---
# X10, X11: AR(1) with partial dependence on Z (informative)
# X12: pure AR(1) with no dependence on Z (spurious structure)
phi1, phi2, phi3 = 0.8, 0.5, 0.9

X10 = np.zeros(N)
X11 = np.zeros(N)
X12 = np.zeros(N)

for t in range(1, N):
    X10[t] = phi1 * X10[t-1] + 0.4 * Z[t] + 0.2 * np.random.randn()
    X11[t] = phi2 * X11[t-1] + 0.6 * Z[t] + 0.3 * np.random.randn()
    X12[t] = phi3 * X12[t-1] + np.random.randn()  # no dependence on Z

# --- Group 5: Pure noise ---
X13 = np.random.randn(N)
X14 = np.random.randn(N)
X15 = np.random.randn(N)

# Assemble feature matrix
feature_names = [
    'X1_linear_strong', 'X2_linear_medium', 'X3_linear_weak',
    'X4_quadratic',     'X5_sinusoidal',    'X6_absolute',
    'X7_cubic',         'X8_sigmoid',       'X9_lagged',
    'X10_AR_strong',    'X11_AR_medium',    'X12_AR_spurious',
    'X13_noise',        'X14_noise',        'X15_noise'
]

X = np.column_stack([
    X1, X2, X3, X4, X5, X6,
    X7, X8, X9, X10, X11, X12,
    X13, X14, X15
])

# Ground truth: 1 = informative, 0 = not informative
ground_truth = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])

# Ground truth ranking (lower = more informative)
ground_truth_rank = np.array([1, 2, 3, 7, 6, 8, 4, 5, 9, 10, 11, 12, 13, 14, 15])

print(f"  Dataset shape: {X.shape}")
print(f"  Target distribution: {Y.mean():.2f} (fraction of 1s)")
print(f"  Informative features: {ground_truth.sum()} / {len(ground_truth)}")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
# 2. COMPUTE MUTUAL INFORMATION
# =============================================================================
print("\n[2/5] Computing Mutual Information...")

def compute_mi(X, y, n_bins=10):
    mi_scores = []
    for i in range(X.shape[1]):
        x_binned = pd.cut(X[:, i], bins=n_bins, labels=False, duplicates='drop')
        y_binned = pd.cut(y.astype(float), bins=n_bins, labels=False, duplicates='drop')
        contingency = pd.crosstab(x_binned, y_binned)
        pxy = contingency / contingency.sum().sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        mi = 0.0
        for xi in pxy.index:
            for yi in pxy.columns:
                if pxy.loc[xi, yi] > 0:
                    mi += pxy.loc[xi, yi] * np.log(pxy.loc[xi, yi] / (px[xi] * py[yi]))
        mi_scores.append(mi)
    return np.array(mi_scores)

mi_scores = compute_mi(X_scaled, Y.astype(float))
mi_ranks = rankdata(-mi_scores).astype(int)  # higher MI = better = lower rank

# =============================================================================
# 3. COMPUTE INFORMATION IMBALANCE
# =============================================================================
print("\n[3/5] Computing Information Imbalance...")

def compute_ii(X, y_true):
    n = X.shape[0]
    dist_Y = squareform(pdist(y_true, metric='euclidean'))
    np.fill_diagonal(dist_Y, np.max(dist_Y) + 1)
    ranks_Y = rankdata(dist_Y, method='average', axis=1).astype(int)

    ii_scores = []
    for i in range(X.shape[1]):
        dist_X = squareform(pdist(X[:, i:i+1], metric='euclidean'))
        np.fill_diagonal(dist_X, np.max(dist_X) + 1)
        score = sum(ranks_Y[j, np.argmin(dist_X[j])] for j in range(n))
        ii_scores.append(2.0 * score / n**2)
    return np.array(ii_scores)

# Use Z (latent signal) as reference space for II and DII.
# This is the fair choice: Z is the ground truth signal that generates Y,
# so II(Xi -> Z) measures the same quantity as MI(Xi; Y) but with a
# distance-based, rank-based approach instead of a histogram approach.
y_true = Z.reshape(-1, 1)
print(f"  Reference space: Z (latent signal), shape {y_true.shape}")

ii_scores = compute_ii(X_scaled, y_true)
ii_ranks = rankdata(ii_scores).astype(int)  # lower II = better = lower rank

# =============================================================================
# 4. COMPUTE DIFFERENTIABLE INFORMATION IMBALANCE
# =============================================================================
print("\n[4/5] Computing Differentiable Information Imbalance...")

def compute_dii(X, y_true, lambda_factor=0.1):
    n = X.shape[0]
    dist_y = squareform(pdist(y_true, metric='euclidean'))
    np.fill_diagonal(dist_y, np.max(dist_y) + 1)
    rank_y = rankdata(dist_y, method='average', axis=1).astype(int)

    dii_scores = []
    for i in range(X.shape[1]):
        dist_x = squareform(pdist(X[:, i:i+1], metric='euclidean'))
        np.fill_diagonal(dist_x, np.max(dist_x) + 1)

        nn_dists = np.min(dist_x, axis=1)
        lambd = lambda_factor * np.mean(nn_dists)

        min_dists = nn_dists[:, np.newaxis]
        exp_matrix = np.exp(-(dist_x - min_dists) / (lambd + 1e-10))
        np.fill_diagonal(exp_matrix, 0)
        c_matrix = exp_matrix / (np.sum(exp_matrix, axis=1)[:, np.newaxis] + 1e-10)

        dii_scores.append(2.0 / n**2 * np.sum(rank_y * c_matrix))
    return np.array(dii_scores)

dii_scores = compute_dii(X_scaled, y_true, lambda_factor=0.1)
dii_ranks = rankdata(dii_scores).astype(int)  # lower DII = better = lower rank

# =============================================================================
# 5. EVALUATE AGAINST GROUND TRUTH
# =============================================================================
print("\n[5/5] Evaluating against ground truth...")

results = pd.DataFrame({
    'Feature':          feature_names,
    'Group':            ['Linear']*3 + ['Nonlin_Sym']*3 + ['Nonlin_Dir']*3 +
                        ['AR']*3 + ['Noise']*3,
    'Informative':      ground_truth,
    'GT_Rank':          ground_truth_rank,
    'MI_Score':         mi_scores,
    'MI_Rank':          mi_ranks,
    'II_Score':         ii_scores,
    'II_Rank':          ii_ranks,
    'DII_Score':        dii_scores,
    'DII_Rank':         dii_ranks,
})

# Rank correlation with ground truth
tau_mi,  _ = kendalltau(ground_truth_rank, mi_ranks)
tau_ii,  _ = kendalltau(ground_truth_rank, ii_ranks)
tau_dii, _ = kendalltau(ground_truth_rank, dii_ranks)

rho_mi,  _ = spearmanr(ground_truth_rank, mi_ranks)
rho_ii,  _ = spearmanr(ground_truth_rank, ii_ranks)
rho_dii, _ = spearmanr(ground_truth_rank, dii_ranks)

# Top-K precision (fraction of truly informative features in top-K)
def top_k_precision(ranks, truth, k):
    top_k_idx = np.where(ranks <= k)[0]
    return truth[top_k_idx].sum() / k

print("\n" + "=" * 70)
print("RESULTS: Rank Correlation with Ground Truth")
print("=" * 70)
print(f"\n  {'Method':<10} {'Kendall τ':>12} {'Spearman ρ':>12}")
print(f"  {'-'*36}")
print(f"  {'MI':<10} {tau_mi:>12.3f} {rho_mi:>12.3f}")
print(f"  {'II':<10} {tau_ii:>12.3f} {rho_ii:>12.3f}")
print(f"  {'DII':<10} {tau_dii:>12.3f} {rho_dii:>12.3f}")

print("\n" + "=" * 70)
print("RESULTS: Top-K Precision (fraction of informative features in top-K)")
print("=" * 70)
print(f"\n  {'Method':<10} {'Top-3':>8} {'Top-5':>8} {'Top-8':>8} {'Top-11':>8}")
print(f"  {'-'*42}")
for name, ranks in [('MI', mi_ranks), ('II', ii_ranks), ('DII', dii_ranks)]:
    p3  = top_k_precision(ranks, ground_truth, 3)
    p5  = top_k_precision(ranks, ground_truth, 5)
    p8  = top_k_precision(ranks, ground_truth, 8)
    p11 = top_k_precision(ranks, ground_truth, 11)
    print(f"  {name:<10} {p3:>8.2f} {p5:>8.2f} {p8:>8.2f} {p11:>8.2f}")

print("\n" + "=" * 70)
print("RESULTS: Rankings by Feature Group")
print("=" * 70)
for group in ['Linear', 'Nonlin_Sym', 'Nonlin_Dir', 'AR', 'Noise']:
    sub = results[results['Group'] == group]
    print(f"\n  {group}:")
    print(f"  {'Feature':<22} {'GT':>5} {'MI':>5} {'II':>5} {'DII':>5}")
    for _, row in sub.iterrows():
        print(f"  {row['Feature']:<22} {int(row['GT_Rank']):>5} "
              f"{int(row['MI_Rank']):>5} {int(row['II_Rank']):>5} "
              f"{int(row['DII_Rank']):>5}")

# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================
print("\nCreating visualizations...")

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

colors_group = {
    'Linear':      '#2ecc71',
    'Nonlin_Sym':  '#e74c3c',
    'Nonlin_Dir':  '#3498db',
    'AR':          '#f39c12',
    'Noise':       '#95a5a6'
}

# Panel 1: MI ranks vs ground truth
ax1 = fig.add_subplot(gs[0, 0])
for group, color in colors_group.items():
    sub = results[results['Group'] == group]
    ax1.scatter(sub['GT_Rank'], sub['MI_Rank'], color=color,
                label=group, s=80, edgecolors='black', linewidth=0.5, zorder=3)
ax1.plot([1, 15], [1, 15], 'k--', alpha=0.3)
ax1.set_xlabel('Ground Truth Rank')
ax1.set_ylabel('MI Rank')
ax1.set_title(f'Mutual Information\n(τ={tau_mi:.2f}, ρ={rho_mi:.2f})', fontweight='bold')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)

# Panel 2: II ranks vs ground truth
ax2 = fig.add_subplot(gs[0, 1])
for group, color in colors_group.items():
    sub = results[results['Group'] == group]
    ax2.scatter(sub['GT_Rank'], sub['II_Rank'], color=color,
                label=group, s=80, edgecolors='black', linewidth=0.5, zorder=3)
ax2.plot([1, 15], [1, 15], 'k--', alpha=0.3)
ax2.set_xlabel('Ground Truth Rank')
ax2.set_ylabel('II Rank')
ax2.set_title(f'Information Imbalance\n(τ={tau_ii:.2f}, ρ={rho_ii:.2f})', fontweight='bold')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

# Panel 3: DII ranks vs ground truth
ax3 = fig.add_subplot(gs[0, 2])
for group, color in colors_group.items():
    sub = results[results['Group'] == group]
    ax3.scatter(sub['GT_Rank'], sub['DII_Rank'], color=color,
                label=group, s=80, edgecolors='black', linewidth=0.5, zorder=3)
ax3.plot([1, 15], [1, 15], 'k--', alpha=0.3)
ax3.set_xlabel('Ground Truth Rank')
ax3.set_ylabel('DII Rank')
ax3.set_title(f'Diff. Information Imbalance\n(τ={tau_dii:.2f}, ρ={rho_dii:.2f})', fontweight='bold')
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.3)

# Panel 4: Top-K precision comparison
ax4 = fig.add_subplot(gs[1, 0:2])
ks = [3, 5, 8, 11]
prec_mi  = [top_k_precision(mi_ranks,  ground_truth, k) for k in ks]
prec_ii  = [top_k_precision(ii_ranks,  ground_truth, k) for k in ks]
prec_dii = [top_k_precision(dii_ranks, ground_truth, k) for k in ks]

x = np.arange(len(ks))
w = 0.25
ax4.bar(x - w, prec_mi,  w, label='MI',  color='#3498db', edgecolor='black', linewidth=0.5)
ax4.bar(x,     prec_ii,  w, label='II',  color='#2ecc71', edgecolor='black', linewidth=0.5)
ax4.bar(x + w, prec_dii, w, label='DII', color='#e74c3c', edgecolor='black', linewidth=0.5)
ax4.axhline(ground_truth.mean(), color='black', linestyle='--',
            linewidth=1.5, label=f'Baseline ({ground_truth.mean():.2f})')
ax4.set_xticks(x)
ax4.set_xticklabels([f'Top-{k}' for k in ks])
ax4.set_ylabel('Precision')
ax4.set_title('Top-K Precision: fraction of truly informative features', fontweight='bold')
ax4.legend()
ax4.set_ylim(0, 1.05)
ax4.grid(True, alpha=0.3, axis='y')

# Panel 5: Heatmap of rankings by group
ax5 = fig.add_subplot(gs[1, 2])
heatmap_data = results[['MI_Rank', 'II_Rank', 'DII_Rank']].values
import seaborn as sns
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='RdYlGn_r',
            xticklabels=['MI', 'II', 'DII'],
            yticklabels=[f.replace('_', '\n') for f in feature_names],
            ax=ax5, cbar_kws={'label': 'Rank'}, linewidths=0.5)
ax5.set_title('Feature Rankings Heatmap', fontweight='bold')
ax5.tick_params(axis='y', labelsize=7)

plt.suptitle('Simulation Study: II vs MI vs DII on Synthetic Data\n'
             '(green=Linear, red=Nonlin_Sym, blue=Nonlin_Dir, orange=AR, grey=Noise)',
             fontsize=12, fontweight='bold', y=1.01)

plt.savefig('simulation_study_results.png', dpi=300, bbox_inches='tight')
print("Saved: simulation_study_results.png")

# =============================================================================
# 7. SAVE RESULTS
# =============================================================================
results.to_csv('simulation_study_rankings.csv', index=False)
print("Saved: simulation_study_rankings.csv")

# Summary
summary = f"""
SIMULATION STUDY SUMMARY
=========================
N = {N} observations | 15 features | {int(ground_truth.sum())} informative

Rank Correlation with Ground Truth (Kendall tau / Spearman rho):
  MI:  τ = {tau_mi:.3f},  ρ = {rho_mi:.3f}
  II:  τ = {tau_ii:.3f},  ρ = {rho_ii:.3f}
  DII: τ = {tau_dii:.3f},  ρ = {rho_dii:.3f}

Top-5 Precision (fraction of informative features in top-5):
  MI:  {top_k_precision(mi_ranks, ground_truth, 5):.2f}
  II:  {top_k_precision(ii_ranks, ground_truth, 5):.2f}
  DII: {top_k_precision(dii_ranks, ground_truth, 5):.2f}
  Baseline (random): {ground_truth.mean():.2f}
"""
print(summary)
with open('simulation_study_summary.txt', 'w') as f:
    f.write(summary)
print("Saved: simulation_study_summary.txt")
print("\n" + "=" * 70)
print("SIMULATION STUDY COMPLETE")
print("=" * 70)

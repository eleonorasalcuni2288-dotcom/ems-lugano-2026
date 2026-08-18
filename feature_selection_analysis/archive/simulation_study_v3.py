"""
Simulation Study v3: MI vs II vs DII (full DADApy)
===================================================
Dataset generator from Daniele (PhD reviewer), extended with:
  - MI  = sklearn mutual_info_regression  (per-feature)
  - II  = DADApy-aligned formula          (per-feature)
  - DII = full DADApy DiffImbalance       (joint, with and without L1)

All methods use target_y as reference space → fair comparison.

Reference: Glielmo et al. PNAS Nexus 2022 (II), Wild et al. Nat. Comm. 2025 (DII).
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, kendalltau, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from dadapy.diff_imbalance import DiffImbalance

# =============================================================================
# 1. DATASET GENERATOR  (Daniele's design)
# =============================================================================

def generate_synthetic_dataset(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    t   = np.arange(n)

    cycle_fast = np.sin(2 * np.pi * t / 50)
    cycle_slow = np.sin(2 * np.pi * t / 200)
    eps = rng.normal(0.0, 1.0, n)

    z1 = np.zeros(n); z2 = np.zeros(n); z3 = np.zeros(n)
    for i in range(1, n):
        z1[i] =  0.72 * z1[i-1] + 0.55 * eps[i]        + rng.normal(0, 0.25)
        z2[i] =  0.45 * z2[i-1] + 0.30 * z1[i-1]       + rng.normal(0, 0.35)
        z3[i] = -0.30 * z3[i-1] + 0.25 * z2[i-1]       + rng.normal(0, 0.30)

    vol = 0.25 + 0.18 * np.abs(cycle_slow) + 0.12 * (z1**2 / (1 + z1**2))

    # A. Linear
    x_lin_1 = z1                         + rng.normal(0, 0.10, n)
    x_lin_2 = 0.8 * z2                   + rng.normal(0, 0.10, n)
    # B. Nonlinear symmetric
    x_sym_1 = z1**2                       + rng.normal(0, 0.08, n)
    x_sym_2 = np.abs(z2)                  + rng.normal(0, 0.08, n)
    # C. Directional / asymmetric
    x_dir_1 = np.maximum(z1, 0)          + rng.normal(0, 0.06, n)
    x_dir_2 = np.maximum(-z2, 0)         + rng.normal(0, 0.06, n)
    # D. Cyclical
    x_wave_1 = cycle_fast                 + rng.normal(0, 0.05, n)
    x_wave_2 = np.cos(2*np.pi*t/50)      + rng.normal(0, 0.05, n)  # NOT in target
    # E. Interaction
    x_mix_1  = z1 * z2                    + rng.normal(0, 0.08, n)
    x_mix_2  = z1 / (1 + np.abs(z3))     + rng.normal(0, 0.08, n)  # NOT in target
    # F. Redundant
    x_red_1  = x_lin_1                    + rng.normal(0, 0.04, n)
    x_red_2  = x_sym_1                    + rng.normal(0, 0.04, n)
    x_red_3  = 0.5*x_dir_1 + 0.5*x_mix_1 + rng.normal(0, 0.05, n)
    # G. Derived time-series
    regime   = (cycle_slow > 0).astype(float)
    momentum = pd.Series(z1).rolling(10, min_periods=1).mean().to_numpy()
    rvp      = (pd.Series(np.abs(np.diff(np.r_[0.0, z1])))
                  .rolling(15, min_periods=1).mean().to_numpy())
    # H. Pure noise
    noise_1 = rng.normal(0, 1, n); noise_2 = rng.normal(0, 1, n)
    noise_3 = rng.normal(0, 1, n); noise_4 = rng.normal(0, 1, n)

    # Target (known coefficients)
    y = (
          0.90 * x_lin_1
        - 0.70 * x_lin_2
        + 0.85 * (z1**2 - np.mean(z1**2))
        + 0.65 * np.abs(z2)
        + 1.10 * np.maximum(z1, 0)
        - 0.95 * np.maximum(-z2, 0)
        + 0.55 * (z1 * z2)
        + 0.35 * cycle_fast
        + 0.20 * regime
        + rng.normal(0, vol, n)
    )

    feature_cols = [
        'x_lin_1','x_lin_2',
        'x_sym_1','x_sym_2',
        'x_dir_1','x_dir_2',
        'x_wave_1','x_wave_2',
        'x_mix_1','x_mix_2',
        'x_red_1','x_red_2','x_red_3',
        'momentum','realized_vol_proxy','regime',
        'noise_1','noise_2','noise_3','noise_4',
    ]
    X = np.column_stack([
        x_lin_1, x_lin_2,
        x_sym_1, x_sym_2,
        x_dir_1, x_dir_2,
        x_wave_1, x_wave_2,
        x_mix_1, x_mix_2,
        x_red_1, x_red_2, x_red_3,
        momentum, rvp, regime,
        noise_1, noise_2, noise_3, noise_4,
    ])

    # Ground truth: 1 = relevant to target
    gt_binary = np.array([1,1, 1,1, 1,1, 1,0, 1,0, 1,1,1, 0,0,1, 0,0,0,0])

    # Ground truth rank by |coefficient| in target formula:
    # dir_1(1.10)>dir_2(0.95)>lin_1(0.90)>sym_1(0.85)>lin_2(0.70)>sym_2(0.65)>
    # mix_1(0.55)>wave_1(0.35)>regime(0.20)>red_1~red_2~red_3(redundant)>
    # wave_2,mix_2,momentum,rvp,noise_1-4 (irrelevant)
    gt_rank = np.array([3,5, 4,6, 1,2, 8,13, 7,14, 10,11,12, 15,16,9, 17,18,19,20])

    return X, y, feature_cols, gt_binary, gt_rank

# =============================================================================
# 2. GENERATE DATA
# =============================================================================
SEED = 42
N    = 2000

print("="*70)
print("SIMULATION STUDY v3: MI vs II vs DII (full DADApy)")
print("Dataset: Daniele's design | N=2000 | 20 features")
print("="*70)

print("\n[1/6] Generating dataset...")
X_raw, Y, feature_names, gt_binary, gt_rank = generate_synthetic_dataset(n=N, seed=SEED)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
Y_scaled = (Y - Y.mean()) / Y.std()

n_features = X_scaled.shape[1]
print(f"  Shape: {X_scaled.shape} | Informative: {gt_binary.sum()} / {n_features}")

# =============================================================================
# 3. MUTUAL INFORMATION
# =============================================================================
print("\n[2/6] Computing Mutual Information...")
mi_scores = mutual_info_regression(X_scaled, Y_scaled, n_neighbors=5, random_state=SEED)
mi_ranks  = rankdata(-mi_scores).astype(int)

# =============================================================================
# 4. INFORMATION IMBALANCE (per-feature)
# =============================================================================
print("\n[3/6] Computing Information Imbalance (per-feature)...")

def compute_ii(X, y_ref):
    n = X.shape[0]
    dist_Y = squareform(pdist(y_ref.reshape(-1,1)))
    np.fill_diagonal(dist_Y, dist_Y.max() + 1.0)
    ranks_Y = rankdata(dist_Y, method='average', axis=1).astype(int)
    scores = np.empty(X.shape[1])
    for i in range(X.shape[1]):
        dist_X = squareform(pdist(X[:, i:i+1]))
        np.fill_diagonal(dist_X, dist_X.max() + 1.0)
        scores[i] = 2.0 * sum(ranks_Y[j, np.argmin(dist_X[j])] for j in range(n)) / n**2
    return scores

ii_scores = compute_ii(X_scaled, Y_scaled)
ii_ranks  = rankdata(ii_scores).astype(int)

# =============================================================================
# 5. DII — full DADApy — with and without L1
# =============================================================================
k_init  = min(100, max(5, int(0.05 * N)))
k_final = max(1,   int(0.01 * N))

def run_dii(l1, label):
    model = DiffImbalance(
        data_A             = X_scaled.astype(np.float64),
        data_B             = Y_scaled.reshape(-1,1).astype(np.float64),
        num_epochs         = 300,
        batches_per_epoch  = 1,
        seed               = SEED,
        l1_strength        = l1,
        point_adapt_lambda = True,
        k_init             = k_init,
        k_final            = k_final,
        lambda_factor      = 0.1,
        optimizer_name     = 'adam',
        learning_rate      = 1e-2,
        learning_rate_decay= 'cos',
    )
    _, imbs = model.train(bar_label=label)
    w = np.array(model.params_final)
    return w, rankdata(-w).astype(int), imbs

print("\n[4/6] DII with L1...")
dii_l1_w,   dii_l1_r,   imbs_l1   = run_dii(0.01,  "DII + L1")
print("\n[5/6] DII without L1...")
dii_nol1_w, dii_nol1_r, imbs_nol1 = run_dii(0.0,   "DII no L1")

# =============================================================================
# 6. EVALUATE
# =============================================================================
print("\n[6/6] Evaluating...")

results = pd.DataFrame({
    'Feature'      : feature_names,
    'Informative'  : gt_binary,
    'GT_Rank'      : gt_rank,
    'MI_Score'     : mi_scores,    'MI_Rank'      : mi_ranks,
    'II_Score'     : ii_scores,    'II_Rank'      : ii_ranks,
    'DII_L1_W'     : dii_l1_w,    'DII_L1_Rank'  : dii_l1_r,
    'DII_noL1_W'   : dii_nol1_w,  'DII_noL1_Rank': dii_nol1_r,
})

def tau_rho(ranks):
    t, _ = kendalltau(gt_rank, ranks)
    r, _ = spearmanr(gt_rank, ranks)
    return t, r

def topk(ranks, k):
    return gt_binary[np.where(ranks <= k)[0]].sum() / k

tau_mi,  rho_mi  = tau_rho(mi_ranks)
tau_ii,  rho_ii  = tau_rho(ii_ranks)
tau_l1,  rho_l1  = tau_rho(dii_l1_r)
tau_nl1, rho_nl1 = tau_rho(dii_nol1_r)

# =============================================================================
# PRINT RESULTS
# =============================================================================
print("\n" + "="*70)
print("RANK CORRELATION WITH GROUND TRUTH")
print("="*70)
print(f"\n  {'Method':<16} {'Kendall τ':>10} {'Spearman ρ':>12}")
print(f"  {'-'*40}")
for nm, t, r in [('MI', tau_mi, rho_mi), ('II', tau_ii, rho_ii),
                  ('DII + L1', tau_l1, rho_l1), ('DII no L1', tau_nl1, rho_nl1)]:
    print(f"  {nm:<16} {t:>10.3f} {r:>12.3f}")

print("\n" + "="*70)
print("TOP-K PRECISION")
print("="*70)
print(f"\n  {'Method':<16} {'Top-3':>7} {'Top-5':>7} {'Top-9':>7} {'Top-12':>8}")
print(f"  {'-'*44}")
for nm, rk in [('MI',mi_ranks),('II',ii_ranks),
               ('DII + L1',dii_l1_r),('DII no L1',dii_nol1_r)]:
    p = [topk(rk, k) for k in [3, 5, 9, 12]]
    print(f"  {nm:<16} {p[0]:>7.2f} {p[1]:>7.2f} {p[2]:>7.2f} {p[3]:>8.2f}")
print(f"  {'Baseline':<16} {gt_binary.mean():>7.2f} "*4)

print("\n" + "="*70)
print("REDUNDANT FEATURES — LASSO EFFECT")
print("="*70)
red_names = ['x_red_1','x_red_2','x_red_3']
red_idx   = [feature_names.index(f) for f in red_names]
print(f"\n  {'Feature':<22} {'MI r':>6} {'II r':>6} {'DII+L1 w':>10} {'DII+L1 r':>10} {'DII noL1 w':>12}")
print(f"  {'-'*68}")
for i in red_idx:
    print(f"  {feature_names[i]:<22} {mi_ranks[i]:>6} {ii_ranks[i]:>6} "
          f"{dii_l1_w[i]:>10.4f} {dii_l1_r[i]:>10} {dii_nol1_w[i]:>12.4f}")
# also show originals
for nm, i in [('x_lin_1',0),('x_sym_1',2),('x_dir_1',4),('x_mix_1',8)]:
    label = nm + " (orig)"
    print(f"  {label:<22} {mi_ranks[i]:>6} {ii_ranks[i]:>6} "
          f"{dii_l1_w[i]:>10.4f} {dii_l1_r[i]:>10} {dii_nol1_w[i]:>12.4f}")

# =============================================================================
# VISUALISATIONS
# =============================================================================
print("\nCreating visualisations...")

effect_type = [
    'linear','linear',
    'nonlin_sym','nonlin_sym',
    'directional','directional',
    'cyclical','cyclical',
    'interaction','interaction',
    'redundant','redundant','redundant',
    'derived','derived','regime',
    'noise','noise','noise','noise',
]
group_colors = {
    'linear'     : '#2ecc71', 'nonlin_sym' : '#e74c3c',
    'directional': '#3498db', 'cyclical'   : '#9b59b6',
    'interaction': '#f39c12', 'redundant'  : '#e67e22',
    'derived'    : '#1abc9c', 'regime'     : '#d35400',
    'noise'      : '#95a5a6',
}

fig = plt.figure(figsize=(20, 15))
gs  = gridspec.GridSpec(3, 4, hspace=0.5, wspace=0.35)

# Scatter plots (GT vs method rank)
for col, (mname, mranks, tau, rho) in enumerate([
    ('MI', mi_ranks, tau_mi, rho_mi),
    ('II', ii_ranks, tau_ii, rho_ii),
    ('DII + L1', dii_l1_r, tau_l1, rho_l1),
    ('DII no L1', dii_nol1_r, tau_nl1, rho_nl1),
]):
    ax = fig.add_subplot(gs[0, col])
    for idx, (feat, grp) in enumerate(zip(feature_names, effect_type)):
        ax.scatter(gt_rank[idx], mranks[idx],
                   color=group_colors[grp], s=60,
                   edgecolors='black', linewidth=0.4, zorder=3)
    ax.plot([1,n_features],[1,n_features], 'k--', alpha=0.3, lw=1)
    ax.set_xlabel('Ground Truth Rank', fontsize=8)
    ax.set_ylabel('Method Rank',       fontsize=8)
    ax.set_title(f'{mname}\nτ={tau:.3f}  ρ={rho:.3f}', fontweight='bold', fontsize=9)
    if col == 0:
        ax.legend(fontsize=5.5, loc='upper left',
                  handles=[plt.Line2D([0],[0], marker='o', color='w',
                            markerfacecolor=c, markersize=6, label=g)
                            for g, c in group_colors.items()])
    ax.grid(True, alpha=0.3)

# Top-K precision
ax4 = fig.add_subplot(gs[1, 0:2])
ks = [3, 5, 9, 12]; x = np.arange(len(ks)); w = 0.18
bar_c = {'MI':'#3498db','II':'#2ecc71','DII + L1':'#e74c3c','DII no L1':'#8e44ad'}
for off, (nm, rk) in zip([-1.5*w,-0.5*w,0.5*w,1.5*w],
    [('MI',mi_ranks),('II',ii_ranks),('DII + L1',dii_l1_r),('DII no L1',dii_nol1_r)]):
    ax4.bar(x+off, [topk(rk,k) for k in ks], w, label=nm,
            color=bar_c[nm], edgecolor='black', linewidth=0.5)
ax4.axhline(gt_binary.mean(), color='k', ls='--', lw=1.5,
            label=f'Baseline ({gt_binary.mean():.2f})')
ax4.set_xticks(x); ax4.set_xticklabels([f'Top-{k}' for k in ks])
ax4.set_ylabel('Precision'); ax4.set_ylim(0,1.1)
ax4.set_title('Top-K Precision', fontweight='bold')
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis='y')

# Training curves
ax5 = fig.add_subplot(gs[1, 2:4])
ax5.plot(imbs_l1,   color='#e74c3c', lw=1.5, label='DII + L1')
ax5.plot(imbs_nol1, color='#8e44ad', lw=1.5, label='DII no L1', ls='--')
ax5.set_xlabel('Epoch'); ax5.set_ylabel('DII value')
ax5.set_title('DII Training Curves', fontweight='bold')
ax5.legend(); ax5.grid(True, alpha=0.3)

# DII weights bar chart
ax6 = fig.add_subplot(gs[2, :])
sidx = np.argsort(-dii_l1_w)
bcol = [group_colors[effect_type[i]] for i in sidx]
bw   = 0.38
ax6.bar(np.arange(n_features)-bw/2, dii_l1_w[sidx],   bw,
        color=bcol, edgecolor='black', linewidth=0.4, label='DII + L1', alpha=0.9)
ax6.bar(np.arange(n_features)+bw/2, dii_nol1_w[sidx], bw,
        color=bcol, edgecolor='grey',  linewidth=0.4, label='DII no L1', alpha=0.45, hatch='//')
ax6.set_xticks(range(n_features))
ax6.set_xticklabels([feature_names[i] for i in sidx], rotation=45, ha='right', fontsize=7)
ax6.set_ylabel('DII Weight')
ax6.set_title('DII Feature Weights: L1 vs no-L1', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
handles = [plt.Rectangle((0,0),1,1,color=c) for c in group_colors.values()]
ax6.legend(handles+[plt.Rectangle((0,0),1,1,color='grey',alpha=0.5),
                     plt.Rectangle((0,0),1,1,color='grey',hatch='//',alpha=0.4)],
           list(group_colors.keys())+['DII+L1','DII noL1'],
           fontsize=7, loc='upper right', ncol=3)

plt.suptitle(
    'Simulation Study v3  |  MI vs II vs DII (full DADApy)  |  N=2000\n'
    'Dataset: Daniele\'s design — 3 latent AR signals, redundant features, heteroskedastic noise',
    fontsize=11, fontweight='bold', y=1.01
)
plt.savefig('simulation_study_v3_results.png', dpi=300, bbox_inches='tight')
print("Saved: simulation_study_v3_results.png")

# =============================================================================
# SAVE
# =============================================================================
results.to_csv('simulation_study_v3_rankings.csv', index=False)

summary = f"""
SIMULATION STUDY v3 – SUMMARY
================================
Dataset: Daniele's design | N={N} | {n_features} features | {int(gt_binary.sum())} informative
3 latent AR signals (z1, z2, z3) | heteroskedastic noise | redundant features

Rank Correlation with Ground Truth (Kendall τ / Spearman ρ):
  MI           τ={tau_mi:.3f}  ρ={rho_mi:.3f}
  II           τ={tau_ii:.3f}  ρ={rho_ii:.3f}
  DII + L1     τ={tau_l1:.3f}  ρ={rho_l1:.3f}
  DII no L1    τ={tau_nl1:.3f}  ρ={rho_nl1:.3f}

Top-K Precision:
  Method           Top-3  Top-5  Top-9  Top-12
  MI               {topk(mi_ranks,3):.2f}   {topk(mi_ranks,5):.2f}   {topk(mi_ranks,9):.2f}   {topk(mi_ranks,12):.2f}
  II               {topk(ii_ranks,3):.2f}   {topk(ii_ranks,5):.2f}   {topk(ii_ranks,9):.2f}   {topk(ii_ranks,12):.2f}
  DII + L1         {topk(dii_l1_r,3):.2f}   {topk(dii_l1_r,5):.2f}   {topk(dii_l1_r,9):.2f}   {topk(dii_l1_r,12):.2f}
  DII no L1        {topk(dii_nol1_r,3):.2f}   {topk(dii_nol1_r,5):.2f}   {topk(dii_nol1_r,9):.2f}   {topk(dii_nol1_r,12):.2f}
  Baseline         {gt_binary.mean():.2f}
"""
print(summary)
with open('simulation_study_v3_summary.txt', 'w') as f:
    f.write(summary)
print("Saved: simulation_study_v3_summary.txt")
print("\n" + "="*70)
print("SIMULATION STUDY v3 COMPLETE")
print("="*70)

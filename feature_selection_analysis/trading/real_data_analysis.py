"""
Real Data Analysis — train2.csv
================================
Applies the same two-level fair comparison framework as simulation_study_v5
to the real stock return prediction dataset.

Dataset: Kaggle competition "Stock Market Signal: Predict Next-Day Returns"
(https://www.kaggle.com/competitions/stock-market-signal-predict-next-day-returns,
CC BY-SA 4.0) — 440,402 rows, 100 anonymized US equities (2000-2023), 27
engineered technical-indicator features, binary next-day price-direction
target (1=up, 0=down, ~50.3% positive).
Subsample: N=5000 (stratified on target) for computational feasibility.

Methods:
  Level 1 (per-feature): MI  vs  II
  Level 2 (joint):       II joint (LOO backward)  |  DII + L1

Note: JMI greedy and MI joint LOO are excluded from the real data analysis
because JMI's CMI estimator saturates to 0 after the first few steps
(confirmed in simulation study), and MI joint LOO is unreliable in d=27
due to the curse of dimensionality.

No ground truth available → results reported as feature rankings only.
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats    import rankdata
from sklearn.preprocessing      import StandardScaler
from sklearn.feature_selection  import mutual_info_regression
from sklearn.neighbors          import NearestNeighbors
from dadapy.diff_imbalance      import DiffImbalance

SEED = 42
N_SAMPLE = 5000
np.random.seed(SEED)

# =============================================================================
# 1. LOAD AND SUBSAMPLE
# =============================================================================
print("="*65)
print("REAL DATA ANALYSIS — train2.csv")
print("="*65)
print(f"\n[1/6] Loading data...")

df = pd.read_csv('train2.csv')
print(f"  Full dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

feature_cols = [c for c in df.columns if c not in ['id','stock_id','target']]
print(f"  Features ({len(feature_cols)}): {feature_cols}")

# Stratified subsample by target
df_0 = df[df['target']==0].sample(N_SAMPLE//2, random_state=SEED)
df_1 = df[df['target']==1].sample(N_SAMPLE//2, random_state=SEED)
df_s = pd.concat([df_0, df_1]).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"  Subsample: {len(df_s)} rows (stratified, {N_SAMPLE//2} per class)")

X_raw = df_s[feature_cols].values
y_raw = df_s['target'].values.astype(float)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
Y_scaled = (y_raw - y_raw.mean()) / y_raw.std()

n_features    = len(feature_cols)
feature_names = np.array(feature_cols)

# =============================================================================
# 2. LEVEL 1 — MI per-feature
# =============================================================================
print(f"\n[2/6] Level 1 — MI per-feature...")
mi_scores = mutual_info_regression(X_scaled, Y_scaled, random_state=SEED)
mi_ranks  = rankdata(-mi_scores).astype(int)

# =============================================================================
# 3. LEVEL 1 — II per-feature
# =============================================================================
print(f"\n[3/6] Level 1 — II per-feature...")

def compute_ii_pf(xi, y):
    N  = len(xi)
    xi = xi.reshape(-1,1); y = y.reshape(-1,1)
    dx = np.abs(xi-xi.T); np.fill_diagonal(dx, np.inf)
    dy = np.abs(y -y.T);  np.fill_diagonal(dy, np.inf)
    nn = np.argmin(dx, axis=1)
    ry = np.argsort(np.argsort(dy, axis=1), axis=1)
    return 2.0/N**2 * np.sum(ry[np.arange(N), nn])

ii_scores = np.array([compute_ii_pf(X_scaled[:,i], Y_scaled)
                      for i in range(n_features)])
ii_ranks  = rankdata(ii_scores).astype(int)

# =============================================================================
# 4. LEVEL 2 — II joint backward LOO
# =============================================================================
print(f"\n[4/6] Level 2 — II joint backward LOO...")

dist_Y_mat  = np.abs(Y_scaled[:,None] - Y_scaled[None,:])
np.fill_diagonal(dist_Y_mat, np.inf)
Y_ranks_mat = np.argsort(np.argsort(dist_Y_mat, axis=1), axis=1)

def ii_joint(X_sub):
    N = len(X_sub)
    nb = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X_sub)
    _, idx = nb.kneighbors(X_sub)
    return 2.0/N**2 * float(np.sum(Y_ranks_mat[np.arange(N), idx[:,1]]))

ii_full = ii_joint(X_scaled)
print(f"  II_full = {ii_full:.5f}")
ii_loo = np.array([ii_joint(np.delete(X_scaled, i, axis=1))
                   for i in range(n_features)])
ii_joint_imp   = ii_loo - ii_full
ii_joint_ranks = rankdata(-ii_joint_imp).astype(int)

# =============================================================================
# 5. LEVEL 2 — DII + L1
# =============================================================================
N = len(X_scaled)
k_init  = max(5, int(0.025*N))
k_final = max(1, int(0.010*N))

print(f"\n[5/6] Level 2 — DII + L1 (λ=0.1)...")
m = DiffImbalance(
    data_A=X_scaled.astype(np.float64),
    data_B=Y_scaled.reshape(-1,1).astype(np.float64),
    num_epochs=300, batches_per_epoch=1, seed=SEED,
    l1_strength=0.10, point_adapt_lambda=True,
    k_init=k_init, k_final=k_final, lambda_factor=0.1,
    optimizer_name='adam', learning_rate=1e-2,
    learning_rate_decay='cos',
)
_, imbs_l1 = m.train(bar_label="DII + L1")
dii_weights = np.array(m.params_final)
dii_ranks   = rankdata(-dii_weights).astype(int)

# =============================================================================
# 6. RESULTS
# =============================================================================
print(f"\n[6/6] Results...")

results = pd.DataFrame({
    'Feature'       : feature_names,
    'MI_Score'      : mi_scores,    'MI_Rank'       : mi_ranks,
    'II_pf_Score'   : ii_scores,    'II_pf_Rank'    : ii_ranks,
    'II_jt_Imp'     : ii_joint_imp, 'II_jt_Rank'    : ii_joint_ranks,
    'DII_L1_Weight' : dii_weights,  'DII_L1_Rank'   : dii_ranks,
})
results_sorted = results.sort_values('DII_L1_Rank')
results.to_csv('real_data_rankings.csv', index=False)

print("\n" + "="*65)
print("FEATURE RANKINGS — Real Data (sorted by DII rank)")
print("="*65)
print(f"\n  {'Feature':<22} {'MI':>5} {'II_pf':>6} {'II_jt':>6} {'DII':>6}  {'DII_w':>8}")
print("  " + "-"*58)
for _, row in results_sorted.iterrows():
    print(f"  {row['Feature']:<22} {int(row['MI_Rank']):>5} "
          f"{int(row['II_pf_Rank']):>6} {int(row['II_jt_Rank']):>6} "
          f"{int(row['DII_L1_Rank']):>6}  {row['DII_L1_Weight']:>8.4f}")

# Top-10 summary
print(f"\n  Top-10 by DII + L1:")
top10_dii = results[results['DII_L1_Rank'] <= 10]['Feature'].tolist()
for i, f in enumerate(results_sorted[results_sorted['DII_L1_Rank']<=10]['Feature'], 1):
    print(f"    {i:2d}. {f}")

print(f"\n  Features with DII weight = 0 (suppressed by L1):")
zeroed = results[dii_weights < 1e-6]['Feature'].tolist()
print(f"    {zeroed}")

# =============================================================================
# 7. PLOTS
# =============================================================================
method_cfg = [
    ('MI (per-feat.)',  mi_ranks,        '#3498db'),
    ('II (per-feat.)',  ii_ranks,        '#2ecc71'),
    ('II joint (LOO)', ii_joint_ranks,   '#9b59b6'),
    ('DII + L1',        dii_ranks,       '#e74c3c'),
]
labels4 = ['MI\n(pf)', 'II\n(pf)', 'II\njoint', 'DII\n+L1']
colors4 = [c for _,_,c in method_cfg]

fig = plt.figure(figsize=(20, 12))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

# Row 0: rank-rank scatter (each pair of methods)
pairs = [
    ('MI (per-feat.)', mi_ranks,       'II (per-feat.)', ii_ranks,       '#3498db', '#2ecc71', gs[0,0]),
    ('MI (per-feat.)', mi_ranks,       'DII + L1',       dii_ranks,      '#3498db', '#e74c3c', gs[0,1]),
    ('II (per-feat.)', ii_ranks,       'DII + L1',       dii_ranks,      '#2ecc71', '#e74c3c', gs[0,2]),
    ('II joint (LOO)', ii_joint_ranks, 'DII + L1',       dii_ranks,      '#9b59b6', '#e74c3c', gs[0,3]),
]
for nm_a, rk_a, nm_b, rk_b, ca, cb, pos in pairs:
    ax = fig.add_subplot(pos)
    ax.scatter(rk_a, rk_b, alpha=0.6, s=60, color='#555555', edgecolors='white',
               linewidth=0.3, zorder=3)
    # label each point with feature name
    for i, feat in enumerate(feature_names):
        ax.annotate(feat, (rk_a[i], rk_b[i]), fontsize=4.5, alpha=0.7,
                    ha='center', va='bottom', xytext=(0,2), textcoords='offset points')
    ax.plot([1,n_features],[1,n_features],'k--',alpha=0.3,lw=1)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(rk_a, rk_b)
    ax.set_xlabel(nm_a, fontsize=8); ax.set_ylabel(nm_b, fontsize=8)
    ax.set_title(f'ρ={rho:.3f}', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)

# Row 1-left: DII weight bar chart (all features, sorted)
ax_w = fig.add_subplot(gs[1, 0:2])
idx_sorted = np.argsort(-dii_weights)
colors_bar = ['#e74c3c' if dii_weights[i] > 1e-6 else '#bdc3c7' for i in idx_sorted]
ax_w.bar(range(n_features), dii_weights[idx_sorted], color=colors_bar,
         edgecolor='black', linewidth=0.3)
ax_w.set_xticks(range(n_features))
ax_w.set_xticklabels(feature_names[idx_sorted], rotation=45, ha='right', fontsize=7)
ax_w.set_ylabel('DII weight')
ax_w.set_title('DII + L1 weights — Real Data\n(grey = suppressed by L1 regularisation)',
               fontweight='bold')
ax_w.grid(True, alpha=0.3, axis='y')

# Row 1-centre: Top-10 rank comparison heatmap
ax_h = fig.add_subplot(gs[1, 2:4])
top10_idx = np.where(dii_ranks <= 10)[0]
top10_idx = top10_idx[np.argsort(dii_ranks[top10_idx])]
data_hm = np.array([[mi_ranks[i], ii_ranks[i], ii_joint_ranks[i], dii_ranks[i]]
                    for i in top10_idx])
im = ax_h.imshow(data_hm, aspect='auto', cmap='RdYlGn_r', vmin=1, vmax=n_features)
ax_h.set_xticks([0,1,2,3])
ax_h.set_xticklabels(['MI\n(pf)','II\n(pf)','II\njoint','DII\n+L1'], fontsize=9)
ax_h.set_yticks(range(len(top10_idx)))
ax_h.set_yticklabels(feature_names[top10_idx], fontsize=8)
ax_h.set_title('Rank heatmap — Top-10 DII features\n(green=high rank=important, red=low rank)',
               fontweight='bold')
for r in range(len(top10_idx)):
    for c in range(4):
        ax_h.text(c, r, str(data_hm[r,c]), ha='center', va='center',
                  fontsize=8, fontweight='bold',
                  color='white' if data_hm[r,c] <= 5 or data_hm[r,c] >= 22 else 'black')
plt.colorbar(im, ax=ax_h, label='Rank (1=most important)')

plt.suptitle(
    f'Real Data Analysis — train2.csv  |  N={N_SAMPLE:,} (stratified subsample)\n'
    '27 financial/technical features  |  Binary target (stock return direction)',
    fontsize=11, fontweight='bold', y=1.01)
plt.savefig('real_data_results.png', dpi=300, bbox_inches='tight')
print("\nSaved: real_data_results.png")
print("Saved: real_data_rankings.csv")

# =============================================================================
# 8. SUMMARY FILE
# =============================================================================
top5_dii  = results_sorted[results_sorted['DII_L1_Rank']<=5]['Feature'].tolist()
top10_dii = results_sorted[results_sorted['DII_L1_Rank']<=10]['Feature'].tolist()
zeroed_ft = results[dii_weights < 1e-6]['Feature'].tolist()

summary = f"""
REAL DATA ANALYSIS — train2.csv
=================================
Full dataset: 440,402 observations | 27 features | binary target
Subsample used: N={N_SAMPLE} (stratified, seed={SEED})

Feature rankings (all methods):
  {'Feature':<22} {'MI':>5} {'II_pf':>6} {'II_jt':>6} {'DII':>6}  {'DII_w':>8}
  {'-'*55}
""" + "".join(
    f"  {row['Feature']:<22} {int(row['MI_Rank']):>5} "
    f"{int(row['II_pf_Rank']):>6} {int(row['II_jt_Rank']):>6} "
    f"{int(row['DII_L1_Rank']):>6}  {row['DII_L1_Weight']:>8.4f}\n"
    for _, row in results_sorted.iterrows()
) + f"""
Top-5  features (DII + L1): {top5_dii}
Top-10 features (DII + L1): {top10_dii}
Suppressed by L1 (weight≈0): {zeroed_ft}
"""

with open('real_data_summary.txt', 'w') as f:
    f.write(summary)
print("Saved: real_data_summary.txt")
print("\n" + "="*65)
print("REAL DATA ANALYSIS COMPLETE")
print("="*65)

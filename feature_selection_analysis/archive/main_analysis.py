"""
Feature Selection Analysis
Implements Information Imbalance (II) and Mutual Information (MI) methods
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("FEATURE SELECTION ANALYSIS - II AND MI")
print("="*70)

# ===== STEP 1: Load Data =====
print("\n[1/6] Loading data...")
try:
    df = pd.read_csv('train2.csv', nrows=5000)
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")
except FileNotFoundError:
    print("Error: train2.csv not found")
    print("Make sure train2.csv is in the same folder as this script")
    exit(1)

# ===== STEP 2: Prepare Data =====
print("\n[2/6] Preparing data...")
numeric_df = df.select_dtypes(include=[np.number])
target_col = 'target'
feature_cols = [c for c in numeric_df.columns if c != target_col and c != 'id']

X = numeric_df[feature_cols].values.astype(np.float64)
y = numeric_df[target_col].values.astype(np.float64)

X[np.isnan(X)] = 0.0
y[np.isnan(y)] = 0.0

scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"Data shape: {X.shape}")
print(f"Features: {len(feature_cols)}")

# ===== STEP 3: Create Ground Truth (PCA) =====
print("\n[3/6] Creating ground truth space...")
pca = PCA(n_components=3)
y_true = pca.fit_transform(X)
print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")

# ===== STEP 4: Compute Information Imbalance =====
# Formula (Glielmo et al. 2022 / DADApy):
#   II(A->B) = 2/N^2 * sum_i rank_B(i, NN_A(i))
print("\n[4/6] Computing Information Imbalance...")

# Precompute ground truth ranks once (shared across all features)
dist_Y = squareform(pdist(y_true))
np.fill_diagonal(dist_Y, np.max(dist_Y) + 1)
ranks_Y = rankdata(dist_Y, method='average', axis=1).astype(int)

ii_scores = []
for i in range(X.shape[1]):
    dist_X = squareform(pdist(X[:, i:i+1]))
    np.fill_diagonal(dist_X, np.max(dist_X) + 1)  # exclude self

    score = 0.0
    for j in range(len(X)):
        nearest = np.argmin(dist_X[j])  # safe: diagonal is max+1
        score += ranks_Y[j, nearest]

    ii_scores.append(2.0 * score / (len(X) ** 2))

print(f"Information Imbalance computed for {len(ii_scores)} features")

# ===== STEP 5: Compute Mutual Information =====
print("\n[5/6] Computing Mutual Information...")

mi_scores = []
n_bins = 10

for i in range(X.shape[1]):
    x_binned = pd.cut(X[:, i], bins=n_bins, labels=False, duplicates='drop')
    y_binned = pd.cut(y, bins=n_bins, labels=False, duplicates='drop')
    
    contingency = pd.crosstab(x_binned, y_binned)
    pxy = contingency / contingency.sum().sum()
    
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    
    mi = 0.0
    for j in range(len(px)):
        for k in range(len(py)):
            if pxy.iloc[j, k] > 0:
                mi += pxy.iloc[j, k] * np.log(pxy.iloc[j, k] / (px.iloc[j] * py.iloc[k]))
    
    mi_scores.append(mi)

print(f"Mutual Information computed for {len(mi_scores)} features")

# ===== STEP 6: Create Consensus Ranking =====
print("\n[6/6] Creating consensus ranking...")

results_df = pd.DataFrame({
    'Feature': feature_cols,
    'II_Score': ii_scores,
    'MI_Score': mi_scores,
})

results_df['II_Rank'] = results_df['II_Score'].rank()
results_df['MI_Rank'] = results_df['MI_Score'].rank(ascending=False)
results_df['Avg_Rank'] = (results_df['II_Rank'] + results_df['MI_Rank']) / 2
results_df = results_df.sort_values('Avg_Rank')

print("Consensus ranking created")

# ===== STEP 7: Save Results =====
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70 + "\n")

results_df.to_csv('consensus_ranking.csv', index=False)
print("Saved: consensus_ranking.csv")

results_df[['Feature', 'II_Score', 'II_Rank']].to_csv('results_ii.csv', index=False)
print("Saved: results_ii.csv")

results_df[['Feature', 'MI_Score', 'MI_Rank']].to_csv('results_mi.csv', index=False)
print("Saved: results_mi.csv")

# ===== STEP 8: Display Results =====
print("\n" + "="*70)
print("TOP 12 FEATURES - CONSENSUS RANKING (II + MI)")
print("="*70)

print("\n" + results_df.head(12)[['Feature', 'II_Rank', 'MI_Rank', 'Avg_Rank']].to_string(index=False))

# ===== STEP 9: Create Visualization =====
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70 + "\n")

fig, ax = plt.subplots(figsize=(12, 7))
top_12 = results_df.head(12).sort_values('Avg_Rank', ascending=True)
colors = plt.cm.RdYlGn(np.linspace(0, 1, len(top_12)))

ax.barh(range(len(top_12)), top_12['Avg_Rank'], 
       color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(top_12)))
ax.set_yticklabels(top_12['Feature'], fontsize=10)
ax.set_xlabel('Average Rank (Lower = Better)', fontsize=11)
ax.set_title('Top 12 Features - Consensus Ranking\n(Information Imbalance + Mutual Information)', 
            fontsize=12, fontweight='bold')
ax.invert_xaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('feature_selection_results.png', dpi=300, bbox_inches='tight')
print("Saved: feature_selection_results.png")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

print("\nGenerated Files:")
print("1. consensus_ranking.csv - Final ranking (II + MI)")
print("2. results_ii.csv - Information Imbalance results")
print("3. results_mi.csv - Mutual Information results")
print("4. feature_selection_results.png - Visualization")

print("\nTop 5 Features (Most Important):")
for idx, row in results_df.head(5).iterrows():
    print(f"  {row['Feature']}: Avg Rank = {row['Avg_Rank']:.2f}")
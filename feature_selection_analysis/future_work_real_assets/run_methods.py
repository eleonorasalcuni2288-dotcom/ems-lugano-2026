"""
Quick first-look: MI, II, DII+L1, RF, LASSO on the real-asset dataset
(dataset_feature_selection_2023_2025.csv). Point estimates only, no
bootstrap CI -- exploratory reconnaissance, not part of the poster.

Pairs are chosen for divergence/interaction with the target, not for
comovement with each other (comovement between two features is redundancy,
not synergy -- see the synthetic benchmark's XOR pair for the structure
this is meant to mirror: neither feature alone carries signal, only their
joint/opposing pattern does).
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'synthetic'))
from simulation_study_v6_highdim import compute_ii_pf, run_dii

SEED = 42
PAIRS = {
    'Gold-HomeDepot (safe-haven vs cyclical growth)': ('r_GC=F', 'r_HD'),
    'Oil-Delta (input cost divergence)': ('r_CL=F', 'r_DAL'),
    'EURUSD-Siemens (export exposure)': ('r_EURUSD=X', 'r_SIEGY'),
    'GoldmanSachs-Salesforce (rate-regime divergence)': ('r_GS', 'r_CRM'),
}

df = pd.read_csv('dataset_feature_selection_2023_2025.csv')
feat_cols = [c for c in df.columns if c.startswith('r_')]
X = df[feat_cols].values
y = df['VIX_t+1'].values.astype(int)
p = len(feat_cols)
print(f"N={len(df)}  p={p}")

X_scaled = StandardScaler().fit_transform(X)
Y_scaled = (y - y.mean()) / y.std()

# ---- MI per-feature ----
mi_scores = mutual_info_classif(X_scaled, y, random_state=SEED)
mi_ranks = rankdata(-mi_scores).astype(int)

# ---- II per-feature ----
_dy = np.abs(Y_scaled.reshape(-1, 1) - Y_scaled.reshape(1, -1))
np.fill_diagonal(_dy, np.inf)
ry_global = np.argsort(np.argsort(_dy, axis=1), axis=1)
ii_scores = np.array([compute_ii_pf(X_scaled[:, i], ry_global) for i in range(p)])
ii_ranks = rankdata(ii_scores).astype(int)  # lower II score = more informative

# ---- DII+L1 joint ----
w, dii_ranks, _ = run_dii(X_scaled, Y_scaled, 0.10, 'real_assets', N=len(y))

# ---- Random Forest (classifier: target is binary) ----
rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
rf.fit(X_scaled, y)
perm = permutation_importance(rf, X_scaled, y, n_repeats=10, random_state=SEED, n_jobs=-1)
rf_ranks = rankdata(-perm.importances_mean).astype(int)

# ---- LASSO ----
lasso = LassoCV(cv=5, random_state=SEED, n_jobs=-1, max_iter=5000)
lasso.fit(X_scaled, Y_scaled)
lasso_ranks = rankdata(-np.abs(lasso.coef_)).astype(int)

results = pd.DataFrame({
    'feature': feat_cols,
    'MI_rank': mi_ranks, 'II_rank': ii_ranks, 'DII_L1_rank': dii_ranks,
    'RF_rank': rf_ranks, 'LASSO_rank': lasso_ranks,
})
results.to_csv('method_rankings.csv', index=False)

print("\n=== Rank of each designed pair's two members, per method ===")
print(f"{'Pair':<50} {'Feature':<15} {'MI':>4} {'II':>4} {'DII+L1':>7} {'RF':>4} {'LASSO':>6}")
for pair_name, (f1, f2) in PAIRS.items():
    for f in (f1, f2):
        if f not in feat_cols:
            print(f"  [missing: {f}]")
            continue
        row = results[results.feature == f].iloc[0]
        print(f"{pair_name:<50} {f:<15} {row.MI_rank:>4} {row.II_rank:>4} "
              f"{row.DII_L1_rank:>7} {row.RF_rank:>4} {row.LASSO_rank:>6}")

print(f"\n(out of p={p} features; rank 1 = most important)")
print("\nSaved: method_rankings.csv")

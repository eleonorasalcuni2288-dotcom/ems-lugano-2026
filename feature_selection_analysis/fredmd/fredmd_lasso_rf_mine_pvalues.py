"""
FRED-MD — LASSO/RF/MINE point-estimate p-values (for combined BH correction)
======================================================================
Companion to fredmd_analysis.py, but for LASSO, RF, and MINE instead of
the 4 core methods. Produces one empirical p-value per (method, K) via
the SAME evaluate_method_vs_baseline procedure (n_random=100, seed=42)
used for the original 16 tests, so the two sets of p-values are directly
combinable into a single Benjamini-Hochberg correction family (28 tests
total for this dataset) -- see bh_correction_fredmd_combined.py for that
step. Closes the same fair-comparison gap already closed for
post-infarction complications.

Point estimate only (single ranking per method, matching how the core-4
p-values were computed -- not a bootstrap).

Does not modify any existing file.
"""
import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'common'))     # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # train_mine lives here
from downstream_validation import evaluate_method_vs_baseline
from mine_synthetic_highdim import train_mine

SEED = 42
DATA_PATH = "2026-07-MD.csv"
TARGET_COL = "S&P 500"
LEAKAGE_COLS = ['S&P div yield', 'S&P PE ratio']
K_VALUES = [3, 5, 10, 16]

N_ESTIMATORS = 100
N_REPEATS = 10


def apply_transform(series, code):
    if code == 1: return series
    if code == 2: return series.diff()
    if code == 3: return series.diff().diff()
    if code == 4: return np.log(series)
    if code == 5: return np.log(series).diff()
    if code == 6: return np.log(series).diff().diff()
    if code == 7: return series.pct_change().diff()
    return series


def load_fredmd():
    df = pd.read_csv(DATA_PATH)
    transform_codes = df.iloc[0].drop('sasdate').astype(float)
    data = df.iloc[1:].reset_index(drop=True)
    for c in data.columns:
        if c != 'sasdate':
            data[c] = pd.to_numeric(data[c], errors='coerce')
    transformed = {c: apply_transform(data[c], transform_codes[c])
                   for c in transform_codes.index}
    transformed = pd.DataFrame(transformed)

    miss_pct = transformed.isna().mean()
    high_missing = miss_pct[miss_pct > 0.20].index.tolist()
    transformed = transformed.drop(columns=high_missing)
    transformed = transformed.drop(columns=[c for c in LEAKAGE_COLS
                                              if c in transformed.columns])

    feature_cols = [c for c in transformed.columns if c != TARGET_COL]
    combined = transformed.dropna(subset=[TARGET_COL])
    row_nan_frac = combined[feature_cols].isna().mean(axis=1)
    combined = combined[row_nan_frac < 0.05]
    X_df = combined[feature_cols].fillna(combined[feature_cols].median())
    X = X_df.values.astype(np.float64)

    y_cont = combined[TARGET_COL].values
    y = (y_cont > np.median(y_cont)).astype(int)
    return X, y, np.array(feature_cols)


def mine_loo_ranking(X, Y_scaled, seed):
    n_features = X.shape[1]
    mi_full, _ = train_mine(X, Y_scaled, seed)
    importance = np.zeros(n_features)
    for i in range(n_features):
        X_reduced = np.delete(X, i, axis=1)
        mi_reduced, _ = train_mine(X_reduced, Y_scaled, seed)
        importance[i] = mi_full - mi_reduced
    return rankdata(-importance).astype(int)


if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70)
    print("FRED-MD — LASSO/RF/MINE POINT-ESTIMATE p-VALUES")
    print("=" * 70)

    X, y, feature_names = load_fredmd()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"X shape: {X_scaled.shape}")

    y_scaled = (y.astype(np.float64) - y.mean()) / y.std()

    print("\n[1/3] LASSO ranking...")
    lasso = LassoCV(cv=5, random_state=SEED, n_jobs=-1, max_iter=5000)
    lasso.fit(X_scaled, y_scaled)
    lasso_ranks = rankdata(-np.abs(lasso.coef_)).astype(int)

    print("[2/3] RF ranking...")
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED, n_jobs=-1)
    rf.fit(X_scaled, y)
    perm = permutation_importance(rf, X_scaled, y, n_repeats=N_REPEATS,
                                   random_state=SEED, n_jobs=-1)
    rf_ranks = rankdata(-perm.importances_mean).astype(int)

    print("[3/3] MINE ranking (LOO, slower)...")
    t0 = time.time()
    mine_ranks = mine_loo_ranking(X_scaled, y_scaled, SEED)
    print(f"  done in {time.time()-t0:.0f}s")

    all_rows = []
    for name, ranks in [('LASSO', lasso_ranks), ('RF', rf_ranks), ('MINE', mine_ranks)]:
        print(f"\n--- {name} downstream validation (n_random=100) ---")
        res = evaluate_method_vs_baseline(
            X_scaled, y, ranks, K_VALUES,
            n_random=100, k_neighbors=5, seed=SEED, method_name=name)
        for r in res:
            all_rows.append(r)

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv('fredmd_lasso_rf_mine_pvalues.csv', index=False)
    print("\nSaved: fredmd_lasso_rf_mine_pvalues.csv")
    print(results_df[['method', 'K', 'method_acc', 'baseline_mean', 'p_value']].to_string(index=False))

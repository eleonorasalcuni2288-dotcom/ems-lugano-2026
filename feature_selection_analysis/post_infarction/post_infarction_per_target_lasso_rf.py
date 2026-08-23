"""
Post-infarction — Per-Complication p-values for LASSO + RF (for BH correction)
======================================================================
Companion to post_infarction_per_target.py, but for LASSO and RF instead
of the 4 core methods. Produces one empirical p-value per (complication,
method, K) via the SAME evaluate_method_vs_baseline procedure (n_random=100,
seed=42) used for the original 176 tests, so the two sets of p-values are
directly combinable into a single Benjamini-Hochberg correction family
(264 tests total for this dataset) -- see bh_correction_mi_survivors_
combined.py for that step.

Point estimates only (no bootstrap here -- p-values need a single ranking
per complication/method, exactly like the original per-target script).

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
from downstream_validation import evaluate_method_vs_baseline

SEED = 42
DATA_PATH = "MI.data"
K_VALUES = [3, 5, 10, 16]
HIGH_MISSING_COLS = [7, 34, 35, 88]

COMPLICATION_COLS = {
    112: 'FIBR_PREDS', 113: 'PREDS_TAH', 114: 'JELUD_TAH', 115: 'FIBR_JELUD',
    116: 'A_V_BLOK', 117: 'OTEK_LANC', 118: 'RAZRIV', 119: 'DRESSLER',
    120: 'ZSN', 121: 'REC_IM', 122: 'P_IM_STEN',
}

N_ESTIMATORS = 100
N_REPEATS = 10

if __name__ == "__main__":
    print("=" * 70)
    print("POST-INFARCTION — LASSO+RF PER-COMPLICATION p-VALUES")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, header=None, na_values='?')
    input_cols = [c for c in range(1, 112) if c not in HIGH_MISSING_COLS]
    X_df = df[input_cols].copy()
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = np.nan_to_num(X_df.values.astype(np.float64), nan=0.0)
    n_features = X.shape[1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"X shape: {X.shape}")

    all_rows = []
    t_start = time.time()

    for col_idx, comp_name in COMPLICATION_COLS.items():
        y = (df[col_idx] == 1).astype(int).values
        if y.sum() < 10:
            print(f"{comp_name}: SKIPPED (< 10 positive cases)")
            continue
        y_scaled = (y.astype(np.float64) - y.mean()) / y.std()
        print(f"\n{comp_name} ({y.sum()}/{len(y)} positive)")

        lasso = LassoCV(cv=5, random_state=SEED, n_jobs=-1, max_iter=5000)
        lasso.fit(X_scaled, y_scaled)
        lasso_ranks = rankdata(-np.abs(lasso.coef_)).astype(int)

        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED, n_jobs=-1)
        rf.fit(X_scaled, y)
        perm = permutation_importance(rf, X_scaled, y, n_repeats=N_REPEATS,
                                       random_state=SEED, n_jobs=-1)
        rf_ranks = rankdata(-perm.importances_mean).astype(int)

        for name, ranks in [('LASSO', lasso_ranks), ('RF', rf_ranks)]:
            res = evaluate_method_vs_baseline(
                X_scaled, y, ranks, K_VALUES,
                n_random=100, k_neighbors=5, seed=SEED,
                method_name=f"{comp_name}_{name}")
            for r in res:
                r['complication'] = comp_name
                r['method_clean'] = name
                all_rows.append(r)

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv('post_infarction_per_target_lasso_rf_results.csv', index=False)
    print("\nSaved: post_infarction_per_target_lasso_rf_results.csv")

"""
Bootstrap CI — Post-infarction, LASSO + RF, full grid (11 complications x 4 K)
======================================================================
Extends bootstrap_mi_survivors.py's fair-comparison scope to LASSO and RF,
on the SAME 11 per-complication targets used for MI/II/DII (not the
aggregate "any complication" target -- see bootstrap_ci_mi_aggregate_
lasso_rf.py, which used the wrong target for a fair comparison against
Figure 4's per-complication survivors).

Full grid: 11 complications x 4 K x 2 methods = 88 tests, matching the
same scope as the 176 raw tests run for the 4 core methods (11 x 4 x 4).
B=15 (matching DII_L1's budget).

Ranking conventions match the synthetic-benchmark scripts:
  LASSO: LassoCV (5-fold CV alpha), ranked by |coefficient|.
  RF:    RandomForestClassifier + permutation_importance,
         n_estimators=100, n_repeats=10.

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
from downstream_validation import knn_loo_accuracy

SEED = 42
DATA_PATH = "MI.data"
HIGH_MISSING_COLS = [7, 34, 35, 88]
COMPLICATION_COLS = {
    112: 'FIBR_PREDS', 113: 'PREDS_TAH', 114: 'JELUD_TAH', 115: 'FIBR_JELUD',
    116: 'A_V_BLOK', 117: 'OTEK_LANC', 118: 'RAZRIV', 119: 'DRESSLER',
    120: 'ZSN', 121: 'REC_IM', 122: 'P_IM_STEN',
}
K_VALUES = [3, 5, 10, 16]
SUBSAMPLE_FRAC = 0.80
N_RANDOM_BASELINE = 50
B = 15  # matches DII_L1's budget

N_ESTIMATORS = 100
N_REPEATS = 10


def load_post_infarction():
    df = pd.read_csv(DATA_PATH, header=None, na_values='?')
    input_cols = [c for c in range(1, 112) if c not in HIGH_MISSING_COLS]
    X_df = df[input_cols].copy()
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = np.nan_to_num(X_df.values.astype(np.float64), nan=0.0)
    return X, df


def rank_for_method(method, Xb, yb, seed):
    yb_scaled = (yb.astype(np.float64) - yb.mean()) / (yb.std() + 1e-12)
    if method == 'LASSO':
        lasso = LassoCV(cv=5, random_state=seed, n_jobs=-1, max_iter=5000)
        lasso.fit(Xb, yb_scaled)
        return rankdata(-np.abs(lasso.coef_)).astype(int)
    if method == 'RF':
        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1)
        rf.fit(Xb, yb)
        perm = permutation_importance(rf, Xb, yb, n_repeats=N_REPEATS,
                                       random_state=seed, n_jobs=-1)
        return rankdata(-perm.importances_mean).astype(int)
    raise ValueError(method)


def bootstrap_method(X_full, y_full, method, B, seed):
    n = X_full.shape[0]
    n_sub = int(round(n * SUBSAMPLE_FRAC))
    n_features = X_full.shape[1]
    rng = np.random.default_rng(seed)
    advantages = {K: np.empty(B) for K in K_VALUES}

    for b in range(B):
        idx = rng.choice(n, size=n_sub, replace=False)
        Xb, yb = X_full[idx], y_full[idx]
        # skip a draw if the subsample happens to be single-class (rare, but
        # possible for low-prevalence complications like PREDS_TAH at 1.2%)
        if len(np.unique(yb)) < 2:
            idx2 = rng.choice(n, size=n_sub, replace=False)
            Xb, yb = X_full[idx2], y_full[idx2]

        ranks = rank_for_method(method, Xb, yb, seed + b)

        for K in K_VALUES:
            top_k_idx = np.argsort(ranks)[:K]
            method_acc = knn_loo_accuracy(Xb, yb, top_k_idx, k_neighbors=5)

            rng_baseline = np.random.default_rng(seed + 1000 + b * 10 + K)
            baseline_accs = np.array([
                knn_loo_accuracy(Xb, yb,
                                  rng_baseline.choice(n_features, size=K, replace=False),
                                  k_neighbors=5)
                for _ in range(N_RANDOM_BASELINE)
            ])
            advantages[K][b] = method_acc - baseline_accs.mean()

    return advantages


if __name__ == "__main__":
    t_start_all = time.time()
    print("=" * 70)
    print("BOOTSTRAP CI — POST-INFARCTION, LASSO+RF, FULL GRID "
          "(11 complications x 4 K, B=%d)" % B)
    print("=" * 70)

    print("\n[1/2] Loading post-infarction data...")
    X, df = load_post_infarction()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  X shape: {X_scaled.shape}")

    print("\n[2/2] Bootstrapping LASSO and RF, per complication...")
    all_results = []

    for col_idx, comp_name in COMPLICATION_COLS.items():
        y = (df[col_idx] == 1).astype(int).values
        n_pos = y.sum()
        print(f"\n--- {comp_name} ({n_pos}/{len(y)} positive, "
              f"{100*n_pos/len(y):.1f}%) ---")

        for method in ['LASSO', 'RF']:
            t0 = time.time()
            advantages_by_k = bootstrap_method(X_scaled, y, method, B, SEED)
            for K in K_VALUES:
                adv = advantages_by_k[K]
                lo, hi = np.percentile(adv, [2.5, 97.5])
                frac_pos = float(np.mean(adv > 0))
                crosses_zero = lo <= 0 <= hi
                all_results.append(dict(
                    complication=comp_name, method=method, K=K,
                    mean_advantage=adv.mean(), ci_lo=lo, ci_hi=hi,
                    std=adv.std(), frac_positive=frac_pos, B=B,
                    robust=not crosses_zero))
            flag_summary = ', '.join(
                f"K={K}:{'R' if not (np.percentile(advantages_by_k[K],2.5)<=0<=np.percentile(advantages_by_k[K],97.5)) else 'f'}"
                for K in K_VALUES)
            print(f"  {method} done in {time.time()-t0:.0f}s  [{flag_summary}]")

    print(f"\nTotal time: {(time.time()-t_start_all)/60:.1f} min")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv('bootstrap_ci_mi_survivors_lasso_rf_results.csv', index=False)
    print("\nSaved: bootstrap_ci_mi_survivors_lasso_rf_results.csv")

    n_robust = results_df['robust'].sum()
    print(f"\n{n_robust}/{len(results_df)} (complication, method, K) "
          f"configurations show a robust (non-zero-crossing) advantage.")
    print("\nRobust configurations:")
    for _, r in results_df[results_df['robust']].sort_values(
            'mean_advantage', ascending=False).iterrows():
        print(f"  {r['complication']:<12} {r['method']:<6} K={int(r['K']):<3} "
              f"adv={r['mean_advantage']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]")

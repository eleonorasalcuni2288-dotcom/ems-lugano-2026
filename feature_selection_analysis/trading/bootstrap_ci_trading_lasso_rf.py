"""
Bootstrap CI — Trading, LASSO + Random Forest (fair comparison vs DII+L1)
======================================================================
Extends trading_downstream_validation.py's bootstrap protocol to LASSO
and RF, matching the same fair-comparison treatment given to FRED-MD and
post-infarction (see bootstrap_ci_fredmd_lasso_rf.py). Same N=5000
stratified subsample, same K grid, same downstream validation, B=15
(matching DII_L1's budget, not B_CHEAP=100 which would take hours at
Trading's N=5000).

Ranking conventions match the synthetic-benchmark scripts:
  LASSO: LassoCV (5-fold CV alpha), ranked by |coefficient|.
  RF:    RandomForestClassifier + permutation_importance,
         n_estimators=100, n_repeats=10.

Does not modify any existing file. Output kept separate from
bootstrap_ci_trading_results.csv (same column schema).
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
N_SAMPLE = 5000
K_VALUES = [3, 5, 10, 16]
SUBSAMPLE_FRAC = 0.80
N_RANDOM_BASELINE = 100  # matches trading_downstream_validation.py's convention
B = 15  # matches DII_L1's budget in trading_downstream_validation.py

N_ESTIMATORS = 100
N_REPEATS = 10


def load_trading():
    """Replicates real_data_analysis.py's / trading_downstream_validation.py's
    loading/subsampling exactly."""
    df = pd.read_csv('train2.csv')
    feature_cols = [c for c in df.columns if c not in ['id', 'stock_id', 'target']]
    df_0 = df[df['target'] == 0].sample(N_SAMPLE // 2, random_state=SEED)
    df_1 = df[df['target'] == 1].sample(N_SAMPLE // 2, random_state=SEED)
    df_s = pd.concat([df_0, df_1]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    X = df_s[feature_cols].values.astype(np.float64)
    y = df_s['target'].values.astype(int)
    return X, y, np.array(feature_cols)


def rank_for_method(method, Xb, yb, seed):
    yb_scaled = (yb.astype(np.float64) - yb.mean()) / yb.std()
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

        print(f"    {method}: {b+1}/{B} draws done")

    return advantages


if __name__ == "__main__":
    t_start_all = time.time()
    print("=" * 70)
    print("BOOTSTRAP CI — TRADING, LASSO + RF (B=%d)" % B)
    print("=" * 70)

    print("\n[1/2] Loading trading data (same N=5000 stratified subsample)...")
    X, y, feature_names = load_trading()
    print(f"  X shape: {X.shape}, target: {y.sum()}/{len(y)} positive "
          f"({100*y.mean():.1f}%)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n[2/2] Bootstrapping LASSO and RF...")
    all_results = []

    for method in ['LASSO', 'RF']:
        print(f"\n --- {method} (B={B}) ---")
        t0 = time.time()
        advantages_by_k = bootstrap_method(X_scaled, y, method, B, SEED)
        print(f"  done in {time.time()-t0:.0f}s")
        for K in K_VALUES:
            adv = advantages_by_k[K]
            lo, hi = np.percentile(adv, [2.5, 97.5])
            frac_pos = float(np.mean(adv > 0))
            crosses_zero = lo <= 0 <= hi
            flag = "FRAGILE (CI crosses 0)" if crosses_zero else "ROBUST"
            print(f"    K={K:<3} advantage={adv.mean():+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}]  P(+)={frac_pos:.2f}  [{flag}]")
            all_results.append(dict(method=method, K=K, mean_advantage=adv.mean(),
                                     ci_lo=lo, ci_hi=hi, std=adv.std(),
                                     frac_positive=frac_pos, B=B,
                                     robust=not crosses_zero))

    total_min = (time.time() - t_start_all) / 60
    print(f"\nTotal time: {total_min:.1f} min")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv('bootstrap_ci_trading_lasso_rf_results.csv', index=False)
    print("\nSaved: bootstrap_ci_trading_lasso_rf_results.csv")

"""
Bootstrap CI — Trading, MINE (fair comparison vs DII+L1)
======================================================================
Extends trading_downstream_validation.py's bootstrap protocol to MINE,
matching the same fair-comparison treatment given to LASSO/RF (see
bootstrap_ci_trading_lasso_rf.py). Same N=5000 stratified subsample,
same K grid, same downstream validation, B=15 (matching DII_L1's
budget).

MINE convention matches mine_synthetic_highdim.py: Donsker-Varadhan
lower bound, seed=42, Adam, learning_rate=1e-3, epochs=300, LOO
importance (mi_full - mi_reduced_i per feature), ranking by
descending importance.

Does not modify any existing file. Output kept separate from
bootstrap_ci_trading_results.csv (same column schema).
"""
import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'common'))     # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # train_mine lives here
from downstream_validation import knn_loo_accuracy
from mine_synthetic_highdim import train_mine

SEED = 42
N_SAMPLE = 5000
K_VALUES = [3, 5, 10, 16]
SUBSAMPLE_FRAC = 0.80
N_RANDOM_BASELINE = 100  # matches trading_downstream_validation.py's convention
B = 15  # matches DII_L1's budget


def load_trading():
    df = pd.read_csv('train2.csv')
    feature_cols = [c for c in df.columns if c not in ['id', 'stock_id', 'target']]
    df_0 = df[df['target'] == 0].sample(N_SAMPLE // 2, random_state=SEED)
    df_1 = df[df['target'] == 1].sample(N_SAMPLE // 2, random_state=SEED)
    df_s = pd.concat([df_0, df_1]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    X = df_s[feature_cols].values.astype(np.float64)
    y = df_s['target'].values.astype(int)
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


def bootstrap_mine(X_full, y_full, B, seed, t_start):
    n = X_full.shape[0]
    n_sub = int(round(n * SUBSAMPLE_FRAC))
    n_features = X_full.shape[1]
    rng = np.random.default_rng(seed)
    advantages = {K: np.empty(B) for K in K_VALUES}

    for b in range(B):
        idx = rng.choice(n, size=n_sub, replace=False)
        Xb, yb = X_full[idx], y_full[idx]
        yb_scaled = (yb.astype(np.float64) - yb.mean()) / yb.std()
        ranks = mine_loo_ranking(Xb, yb_scaled, seed + b)

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

        print(f"    MINE: {b+1}/{B} draws done  [{time.time()-t_start:.0f}s elapsed]")

    return advantages


if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70)
    print("BOOTSTRAP CI — TRADING, MINE (B=%d)" % B)
    print("=" * 70)

    print("\n[1/2] Loading trading data (same N=5000 stratified subsample)...")
    X, y, feature_names = load_trading()
    print(f"  X shape: {X.shape}, target: {y.sum()}/{len(y)} positive "
          f"({100*y.mean():.1f}%)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n[2/2] Bootstrapping MINE...")
    advantages_by_k = bootstrap_mine(X_scaled, y, B, SEED, t_start)

    all_results = []
    for K in K_VALUES:
        adv = advantages_by_k[K]
        lo, hi = np.percentile(adv, [2.5, 97.5])
        frac_pos = float(np.mean(adv > 0))
        crosses_zero = lo <= 0 <= hi
        flag = "FRAGILE (CI crosses 0)" if crosses_zero else "ROBUST"
        print(f"    K={K:<3} advantage={adv.mean():+.3f} "
              f"[{lo:+.3f},{hi:+.3f}]  P(+)={frac_pos:.2f}  [{flag}]")
        all_results.append(dict(method='MINE', K=K, mean_advantage=adv.mean(),
                                 ci_lo=lo, ci_hi=hi, std=adv.std(),
                                 frac_positive=frac_pos, B=B,
                                 robust=not crosses_zero))

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv('bootstrap_ci_trading_mine_results.csv', index=False)
    print("\nSaved: bootstrap_ci_trading_mine_results.csv")

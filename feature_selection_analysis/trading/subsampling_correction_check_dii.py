"""
Subsampling Correction Check (Politis-Romano-Wolf) — DII+L1 on Trading
================================================================================
Same spot-check as fredmd/subsampling_correction_check_dii.py, applied to
trading -- the dataset where DII+L1's robust classification is closest to
the zero boundary (K=10 ci_lo=+0.004, K=16 ci_lo=+0.006, much tighter
margins than FRED-MD's +0.034), and therefore the case most at risk of
flipping under the formally corrected CI, similar to what was found for
MI_perfeat on FRED-MD.

theta_hat_n (full N=5000 point estimate) is read directly from the
existing trading_downstream_results.csv (DII_L1 rows) rather than
retrained, since that value is already computed and unaffected by this
check. Only the B=15 bootstrap draws are rerun here, with raw per-draw
values saved (the original script only saved the summary).

Does not modify any existing file.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'common'))
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))
from simulation_study_v6_highdim import run_dii
from downstream_validation import knn_loo_accuracy

SEED = 42
N_SAMPLE = 5000
K_TO_CHECK = [10, 16]   # the two robust DII_L1 results on trading
SUBSAMPLE_FRAC = 0.80
N_RANDOM_BASELINE = 100
B = 15


def load_trading():
    df = pd.read_csv('train2.csv')
    feature_cols = [c for c in df.columns if c not in ['id', 'stock_id', 'target']]
    df_0 = df[df['target'] == 0].sample(N_SAMPLE // 2, random_state=SEED)
    df_1 = df[df['target'] == 1].sample(N_SAMPLE // 2, random_state=SEED)
    df_s = pd.concat([df_0, df_1]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    X = df_s[feature_cols].values.astype(np.float64)
    y = df_s['target'].values.astype(int)
    return X, y, np.array(feature_cols)


if __name__ == "__main__":
    print("Loading trading data (same N=5000 subsample as trading_downstream_validation.py)...")
    X, y, feature_names = load_trading()
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    n = X_scaled.shape[0]
    b = int(round(n * SUBSAMPLE_FRAC))
    print(f"n={n}  b={b}  B={B}")

    # theta_hat_n: read the existing full-sample point estimate (DII_L1)
    point_df = pd.read_csv('trading_downstream_results.csv')
    theta_n = {}
    for K in K_TO_CHECK:
        row = point_df[(point_df.method == 'DII_L1') & (point_df.K == K)].iloc[0]
        theta_n[K] = row.method_acc - row.baseline_mean
        print(f"  K={K}: theta_hat_n = {theta_n[K]:+.4f} (from trading_downstream_results.csv)")

    print(f"\nRunning {B} subsample draws (same seed=42, same procedure as trading_downstream_validation.py)...")
    rng = np.random.default_rng(SEED)
    raw_draws = {K: np.empty(B) for K in K_TO_CHECK}
    n_features = X_scaled.shape[1]
    for draw in range(B):
        idx = rng.choice(n, size=b, replace=False)
        Xb, yb = X_scaled[idx], y[idx]
        yb_f = yb.astype(np.float64)
        _, ranks_b, _ = run_dii(Xb, yb_f, 0.10, f"TRADING_DII_check_draw{draw}", N=len(yb))
        for K in K_TO_CHECK:
            rng_baseline = np.random.default_rng(SEED + 1000 + draw * 10 + K)
            top_k_idx = np.argsort(ranks_b)[:K]
            method_acc = knn_loo_accuracy(Xb, yb, top_k_idx, k_neighbors=5)
            baseline_accs = np.array([
                knn_loo_accuracy(Xb, yb, rng_baseline.choice(n_features, size=K, replace=False),
                                  k_neighbors=5)
                for _ in range(N_RANDOM_BASELINE)
            ])
            raw_draws[K][draw] = method_acc - baseline_accs.mean()
        print(f"  draw {draw+1}/{B} done")

    print("\n" + "=" * 78)
    print(f"{'K':<4}{'current CI (uncorrected)':<32}{'corrected CI (Politis-Romano-Wolf)':<38}")
    print("=" * 78)
    rows = []
    for K in K_TO_CHECK:
        draws = raw_draws[K]
        cur_lo, cur_hi = np.percentile(draws, [2.5, 97.5])
        cur_robust = not (cur_lo <= 0 <= cur_hi)

        d = np.sqrt(b) * (draws - theta_n[K])
        c_lo, c_hi = np.percentile(d, [2.5, 97.5])
        corr_lo = theta_n[K] - c_hi / np.sqrt(n)
        corr_hi = theta_n[K] - c_lo / np.sqrt(n)
        corr_robust = not (corr_lo <= 0 <= corr_hi)

        print(f"K={K:<3}[{cur_lo:+.4f},{cur_hi:+.4f}] {'ROBUST' if cur_robust else 'fragile':<10}"
              f"  [{corr_lo:+.4f},{corr_hi:+.4f}] {'ROBUST' if corr_robust else 'fragile'}"
              f"  {'SAME' if cur_robust==corr_robust else '*** DIFFERENT ***'}")
        rows.append(dict(K=K, theta_hat_n=theta_n[K],
                          current_ci_lo=cur_lo, current_ci_hi=cur_hi, current_robust=cur_robust,
                          corrected_ci_lo=corr_lo, corrected_ci_hi=corr_hi, corrected_robust=corr_robust,
                          classification_changed=(cur_robust != corr_robust)))

    df = pd.DataFrame(rows)
    df.to_csv('subsampling_correction_check_dii_results.csv', index=False)
    print("\nSaved: subsampling_correction_check_dii_results.csv")
    print(f"Any robust/fragile classification changed by the correction? "
          f"{'YES' if df.classification_changed.any() else 'NO'}")

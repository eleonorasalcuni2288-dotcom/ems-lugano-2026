"""
Subsampling Correction Check (Politis-Romano-Wolf) — DII+L1 on FRED-MD
================================================================================
Same spot-check as subsampling_correction_check.py, but on DII+L1 -- the
project's headline method -- instead of MI_perfeat, to test whether the
central claim (DII+L1's downstream advantage is robust) holds up under
the formally corrected subsampling CI, not just the raw-percentile CI
used throughout the project.

B=15 (not 100), matching bootstrap_ci_fredmd.py's own B_DII budget for
this exact method/dataset combination -- DII training is the expensive
step here, this mirrors the project's existing cost/precision tradeoff
rather than introducing a new one.

Does not modify any existing file.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'common'))
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))
from simulation_study_v6_highdim import run_dii
from downstream_validation import knn_loo_accuracy

SEED = 42
DATA_PATH = "2026-07-MD.csv"
TARGET_COL = "S&P 500"
LEAKAGE_COLS = ['S&P div yield', 'S&P PE ratio']
K_VALUES = [3, 5, 10, 16]
SUBSAMPLE_FRAC = 0.80
N_RANDOM_BASELINE = 50
B = 15


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


def dii_rank(Xb, yb, label):
    yb_f = yb.astype(np.float64)
    _, ranks, _ = run_dii(Xb, yb_f, 0.10, label, N=len(yb))
    return ranks


def advantage_at_k(X, y, ranks, K, seed_offset):
    n_features = X.shape[1]
    top_k_idx = np.argsort(ranks)[:K]
    method_acc = knn_loo_accuracy(X, y, top_k_idx, k_neighbors=5)
    rng_baseline = np.random.default_rng(SEED + seed_offset)
    baseline_accs = np.array([
        knn_loo_accuracy(X, y, rng_baseline.choice(n_features, size=K, replace=False),
                          k_neighbors=5)
        for _ in range(N_RANDOM_BASELINE)
    ])
    return method_acc - baseline_accs.mean()


if __name__ == "__main__":
    print("Loading FRED-MD...")
    X, y, feature_names = load_fredmd()
    n = X.shape[0]
    b = int(round(n * SUBSAMPLE_FRAC))
    print(f"n={n}  b={b}  B={B}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nFull-sample point estimate (theta_hat_n), DII+L1...")
    ranks_full = dii_rank(X_scaled, y, "DII_full")
    theta_n = {}
    for K in K_VALUES:
        theta_n[K] = advantage_at_k(X_scaled, y, ranks_full, K, seed_offset=90000 + K)
        print(f"  K={K}: theta_hat_n = {theta_n[K]:+.4f}")

    print(f"\nRunning {B} subsample draws (same seed=42 as bootstrap_ci_fredmd.py)...")
    rng = np.random.default_rng(SEED)
    raw_draws = {K: np.empty(B) for K in K_VALUES}
    for draw in range(B):
        idx = rng.choice(n, size=b, replace=False)
        Xb, yb = X_scaled[idx], y[idx]
        ranks_b = dii_rank(Xb, yb, f"DII_draw{draw}")
        for K in K_VALUES:
            rng_baseline = np.random.default_rng(SEED + 1000 + draw * 10 + K)
            top_k_idx = np.argsort(ranks_b)[:K]
            method_acc = knn_loo_accuracy(Xb, yb, top_k_idx, k_neighbors=5)
            n_features = Xb.shape[1]
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
    for K in K_VALUES:
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
                          current_ci_lo=cur_lo, current_ci_hi=cur_hi, current_width=cur_hi-cur_lo,
                          current_robust=cur_robust,
                          corrected_ci_lo=corr_lo, corrected_ci_hi=corr_hi, corrected_width=corr_hi-corr_lo,
                          corrected_robust=corr_robust,
                          classification_changed=(cur_robust != corr_robust)))

    df = pd.DataFrame(rows)
    df.to_csv('subsampling_correction_check_dii_results.csv', index=False)
    print("\nSaved: subsampling_correction_check_dii_results.csv")
    print(f"\nAvg width -- current: {df.current_width.mean():.4f}   corrected: {df.corrected_width.mean():.4f}")
    print(f"Any robust/fragile classification changed by the correction? "
          f"{'YES' if df.classification_changed.any() else 'NO'}")

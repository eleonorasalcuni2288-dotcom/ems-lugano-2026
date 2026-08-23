"""
Subsampling Correction Check (Politis-Romano-Wolf) — 3 bootstrap-robust
BH survivors on MI complications (ZSN target)
================================================================================
Same spot-check as fredmd/subsampling_correction_check*.py, applied to the
3 (complication, method, K) configurations that are BOTH BH-significant
AND bootstrap-robust: ZSN/MI_perfeat/K=3, ZSN/MI_perfeat/K=5,
ZSN/DII_L1/K=16 -- these are explicitly highlighted in the poster's Key
Findings and Figure 4 caption as the strongest evidence in this dataset.

theta_hat_n (full-sample point estimate) is read directly from the
existing post_infarction_per_target_results.csv rather than retrained.
Only the B subsample draws are rerun, with raw per-draw values saved.

Does not modify any existing file.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'common'))
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))
from simulation_study_v6_highdim import run_dii
from downstream_validation import knn_loo_accuracy

SEED = 42
DATA_PATH = "MI.data"
HIGH_MISSING_COLS = [7, 34, 35, 88]
SUBSAMPLE_FRAC = 0.80
N_RANDOM_BASELINE = 50
COMPLICATION_COLS = {'ZSN': 120}

CHECKS = [
    ('ZSN', 'MI_perfeat', 3, 100),
    ('ZSN', 'MI_perfeat', 5, 100),
    ('ZSN', 'DII_L1', 16, 15),
]


def rank_for_method(method, Xb, yb, label):
    if method == 'MI_perfeat':
        scores = mutual_info_classif(Xb, yb, random_state=SEED)
        return rankdata(-scores).astype(int)
    if method == 'DII_L1':
        yb_f = yb.astype(np.float64)
        _, ranks, _ = run_dii(Xb, yb_f, 0.10, label, N=len(yb))
        return ranks
    raise ValueError(method)


if __name__ == "__main__":
    print("Loading MI complications data...")
    df = pd.read_csv(DATA_PATH, header=None, na_values='?')
    input_cols = [c for c in range(1, 112) if c not in HIGH_MISSING_COLS]
    X_df = df[input_cols].copy()
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = np.nan_to_num(X_df.values.astype(np.float64), nan=0.0)
    X_scaled = StandardScaler().fit_transform(X)
    n = X_scaled.shape[0]
    b = int(round(n * SUBSAMPLE_FRAC))
    print(f"n={n}  b={b}")

    point_df = pd.read_csv('post_infarction_per_target_results.csv')

    all_rows = []
    for comp_name, method, K, B in CHECKS:
        y = (df[COMPLICATION_COLS[comp_name]] == 1).astype(int).values
        prefixed_method = f"{comp_name}_{method}"
        row = point_df[(point_df.complication == comp_name) & (point_df.method == prefixed_method)
                        & (point_df.K == K)].iloc[0]
        theta_n = row.method_acc - row.baseline_mean
        print(f"\n=== {comp_name}/{method}/K={K} (B={B}) ===")
        print(f"  theta_hat_n = {theta_n:+.4f} (from post_infarction_per_target_results.csv)")

        rng = np.random.default_rng(SEED)
        draws = np.empty(B)
        n_features = X_scaled.shape[1]
        for draw in range(B):
            idx = rng.choice(n, size=b, replace=False)
            Xb, yb = X_scaled[idx], y[idx]
            ranks_b = rank_for_method(method, Xb, yb, f"{comp_name}_{method}_K{K}_draw{draw}")
            top_k_idx = np.argsort(ranks_b)[:K]
            method_acc = knn_loo_accuracy(Xb, yb, top_k_idx, k_neighbors=5)
            rng_baseline = np.random.default_rng(SEED + draw)
            baseline_accs = np.array([
                knn_loo_accuracy(Xb, yb, rng_baseline.choice(n_features, size=K, replace=False),
                                  k_neighbors=5)
                for _ in range(N_RANDOM_BASELINE)
            ])
            draws[draw] = method_acc - baseline_accs.mean()
            if (draw + 1) % max(1, B // 5) == 0:
                print(f"    {draw+1}/{B} draws done")

        cur_lo, cur_hi = np.percentile(draws, [2.5, 97.5])
        cur_robust = not (cur_lo <= 0 <= cur_hi)
        d = np.sqrt(b) * (draws - theta_n)
        c_lo, c_hi = np.percentile(d, [2.5, 97.5])
        corr_lo = theta_n - c_hi / np.sqrt(n)
        corr_hi = theta_n - c_lo / np.sqrt(n)
        corr_robust = not (corr_lo <= 0 <= corr_hi)

        print(f"  current   CI: [{cur_lo:+.4f},{cur_hi:+.4f}]  {'ROBUST' if cur_robust else 'fragile'}")
        print(f"  corrected CI: [{corr_lo:+.4f},{corr_hi:+.4f}]  {'ROBUST' if corr_robust else 'fragile'}")
        print(f"  {'SAME' if cur_robust==corr_robust else '*** DIFFERENT ***'}")

        all_rows.append(dict(complication=comp_name, method=method, K=K, theta_hat_n=theta_n,
                              current_ci_lo=cur_lo, current_ci_hi=cur_hi, current_robust=cur_robust,
                              corrected_ci_lo=corr_lo, corrected_ci_hi=corr_hi, corrected_robust=corr_robust,
                              classification_changed=(cur_robust != corr_robust)))

    out = pd.DataFrame(all_rows)
    out.to_csv('subsampling_correction_check_results.csv', index=False)
    print("\nSaved: subsampling_correction_check_results.csv")
    print(f"\nAny classification changed? {'YES' if out.classification_changed.any() else 'NO'}")

"""
Bootstrap CI — FRED-MD (all 4 methods x 4 K, full grid)
======================================================================
Quantifies uncertainty on the downstream-accuracy advantage (method's
Top-K accuracy minus random-feature baseline) for MI_perfeat, II_perfeat,
II_joint, DII_L1, at K=3,5,10,16 — the full grid, not just the earlier
single-K point estimates from fredmd_analysis.py.

EFFICIENCY NOTE: a method's ranking does not depend on K (K only selects
how many top-ranked features to evaluate). Earlier scripts in this
project recomputed the ranking separately per (method, K) pair, which is
wasteful — for DII in particular, that means retraining DII up to 4x
more than necessary. This script computes the ranking ONCE per bootstrap
draw, then evaluates all 4 K values from that same ranking — cutting
DII's cost by ~4x compared to a naive per-K loop.

Target/cleaning logic (transforms, leakage-column removal) duplicated
here from fredmd_analysis.py, since that script's logic lives inside an
`if __name__ == "__main__":` guard and isn't importable.

Run from the project folder (needs 2026-07-MD.csv, simulation_study_v6_highdim.py,
downstream_validation.py). Budget: cheap methods fast (~5-10 min total for
3 methods x 4 K); DII the bottleneck (~15-20 draws x 4 K worth of accuracy
evaluation, but only 15-20 DII trainings total thanks to the reuse above)
— estimate ~15-20 min for DII. Total ~25-35 min.
"""
import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)                              # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # simulation_study_v6_highdim.py lives here
from simulation_study_v6_highdim import compute_ii_pf, make_ii_joint, run_dii
from downstream_validation import knn_loo_accuracy

SEED = 42
DATA_PATH = "2026-07-MD.csv"
TARGET_COL = "S&P 500"
LEAKAGE_COLS = ['S&P div yield', 'S&P PE ratio']
K_VALUES = [3, 5, 10, 16]
SUBSAMPLE_FRAC = 0.80
N_RANDOM_BASELINE = 50
B_CHEAP = 100
B_DII = 15


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


def rank_for_method(method, Xb, yb, label):
    if method == 'MI_perfeat':
        scores = mutual_info_classif(Xb, yb, random_state=SEED)
        return rankdata(-scores).astype(int)
    if method == 'II_perfeat':
        yb_f = yb.astype(np.float64)
        dy = np.abs(yb_f.reshape(-1, 1) - yb_f.reshape(1, -1))
        np.fill_diagonal(dy, np.inf)
        ry = np.argsort(np.argsort(dy, axis=1), axis=1)
        scores = np.array([compute_ii_pf(Xb[:, i], ry) for i in range(Xb.shape[1])])
        return rankdata(scores).astype(int)
    if method == 'II_joint':
        yb_f = yb.astype(np.float64)
        ii_joint_fn = make_ii_joint(yb_f)
        full = ii_joint_fn(Xb)
        loo = np.array([ii_joint_fn(np.delete(Xb, i, axis=1))
                         for i in range(Xb.shape[1])])
        return rankdata(-(loo - full)).astype(int)
    if method == 'DII_L1':
        yb_f = yb.astype(np.float64)
        _, ranks, _ = run_dii(Xb, yb_f, 0.10, label, N=len(yb))
        return ranks
    raise ValueError(method)


def bootstrap_method(X_full, y_full, method, B, seed, label_prefix):
    """One method, B draws. Ranking computed ONCE per draw, then all 4
    K values evaluated from that same ranking. Returns dict K -> array
    of advantages (length B)."""
    n = X_full.shape[0]
    n_sub = int(round(n * SUBSAMPLE_FRAC))
    n_features = X_full.shape[1]
    rng = np.random.default_rng(seed)
    advantages = {K: np.empty(B) for K in K_VALUES}

    for b in range(B):
        idx = rng.choice(n, size=n_sub, replace=False)
        Xb, yb = X_full[idx], y_full[idx]
        ranks = rank_for_method(method, Xb, yb, f"{label_prefix} draw{b}")

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

        if (b + 1) % max(1, B // 5) == 0:
            print(f"    {method}: {b+1}/{B} draws done")

    return advantages


if __name__ == "__main__":
    print("=" * 70)
    print("BOOTSTRAP CI — FRED-MD (full grid: 4 methods x 4 K)")
    print(f"Subsample fraction: {SUBSAMPLE_FRAC}  |  95% percentile CI  |  "
          f"B_cheap={B_CHEAP}  B_DII={B_DII}")
    print("=" * 70)

    print("\n[1/2] Loading FRED-MD...")
    X, y, feature_names = load_fredmd()
    print(f"  X shape: {X.shape}, target: {y.sum()}/{len(y)} positive")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    methods_config = [
        ('MI_perfeat', B_CHEAP), ('II_perfeat', B_CHEAP),
        ('II_joint', B_CHEAP), ('DII_L1', B_DII),
    ]

    print("\n[2/2] Bootstrapping each method (ranking computed once per "
          "draw, all K evaluated from it)...")
    all_results = []
    t_start = time.time()

    for method, B in methods_config:
        print(f"\n --- {method} (B={B}) ---")
        t0 = time.time()
        advantages_by_k = bootstrap_method(X_scaled, y, method, B, SEED,
                                            f"FREDMD_{method}")
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

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv('bootstrap_ci_fredmd_results.csv', index=False)
    print("\nSaved: bootstrap_ci_fredmd_results.csv")

    print("\n" + "=" * 70)
    print("SUMMARY — sorted by mean advantage")
    print("=" * 70)
    for _, r in results_df.sort_values('mean_advantage', ascending=False).iterrows():
        flag = "ROBUST" if r['robust'] else "FRAGILE"
        print(f"  {r['method']:<12} K={int(r['K']):<3} "
              f"adv={r['mean_advantage']:+.3f} "
              f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  [{flag}]")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
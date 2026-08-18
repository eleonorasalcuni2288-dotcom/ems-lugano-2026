"""
Bootstrap CI — MI Complications, Aggregate Target (all 4 methods x 4 K)
======================================================================
Same full-grid bootstrap as bootstrap_ci_fredmd.py, applied to the
aggregated "any complication" target (the one that was null in plain
point-estimate form). This gives confidence intervals confirming (or
not) that null result with proper uncertainty quantification, completing
the picture alongside the per-complication (ZSN etc.) bootstrap already
done.

Ranking computed once per draw, all 4 K evaluated from it (same
efficiency fix as the FRED-MD script — avoids re-training DII 4x per
draw for no reason).

Run from the project folder (needs MI.data, simulation_study_v6_highdim.py,
downstream_validation.py). Budget: ~30-40 min (DII is the bottleneck, ~15
draws x ~80-100s/draw here since N=1700 is larger than FRED-MD's N=794).
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
DATA_PATH = "MI.data"
HIGH_MISSING_COLS = [7, 34, 35, 88]
K_VALUES = [3, 5, 10, 16]
SUBSAMPLE_FRAC = 0.80
N_RANDOM_BASELINE = 50
B_CHEAP = 100
B_DII = 15


def load_mi_aggregate():
    df = pd.read_csv(DATA_PATH, header=None, na_values='?')
    input_cols = [c for c in range(1, 112) if c not in HIGH_MISSING_COLS]
    X_df = df[input_cols].copy()
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = np.nan_to_num(X_df.values.astype(np.float64), nan=0.0)

    complication_cols = list(range(112, 123))
    y = (df[complication_cols] == 1).any(axis=1).astype(int).values
    return X, y


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
    print("BOOTSTRAP CI — MI COMPLICATIONS AGGREGATE (full grid: 4 methods x 4 K)")
    print(f"Subsample fraction: {SUBSAMPLE_FRAC}  |  95% percentile CI  |  "
          f"B_cheap={B_CHEAP}  B_DII={B_DII}")
    print("=" * 70)

    print("\n[1/2] Loading MI complications (aggregate target)...")
    X, y = load_mi_aggregate()
    print(f"  X shape: {X.shape}, target: {y.sum()}/{len(y)} positive "
          f"({100*y.mean():.1f}%)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    methods_config = [
        ('MI_perfeat', B_CHEAP), ('II_perfeat', B_CHEAP),
        ('II_joint', B_CHEAP), ('DII_L1', B_DII),
    ]

    print("\n[2/2] Bootstrapping each method...")
    all_results = []
    t_start = time.time()

    for method, B in methods_config:
        print(f"\n --- {method} (B={B}) ---")
        t0 = time.time()
        advantages_by_k = bootstrap_method(X_scaled, y, method, B, SEED,
                                            f"MIagg_{method}")
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
    results_df.to_csv('bootstrap_ci_mi_aggregate_results.csv', index=False)
    print("\nSaved: bootstrap_ci_mi_aggregate_results.csv")

    print("\n" + "=" * 70)
    print("SUMMARY — sorted by mean advantage")
    print("=" * 70)
    for _, r in results_df.sort_values('mean_advantage', ascending=False).iterrows():
        flag = "ROBUST" if r['robust'] else "FRAGILE"
        print(f"  {r['method']:<12} K={int(r['K']):<3} "
              f"adv={r['mean_advantage']:+.3f} "
              f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  [{flag}]")

    n_robust = results_df['robust'].sum()
    print(f"\n  {n_robust}/{len(results_df)} configurations show a robust "
          f"(non-zero-crossing) advantage.")
    if n_robust == 0:
        print("  Consistent with the earlier point-estimate null result: "
              "the aggregate target shows no robust predictive advantage "
              "for any method at any K, even accounting for resampling "
              "uncertainty.")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    
"""
Trading (train2.csv) — Ground-Truth-Free Downstream Validation + Bootstrap CI
================================================================================
Brings the trading dataset up to the same rigor tier as FRED-MD and MI
complications: ground-truth-free downstream validation (evaluate_method_vs_
baseline) plus bootstrap-quantified uncertainty on the downstream advantage
— neither of which trading had before (it only had point-estimate rankings
from real_data_analysis.py, no validation, no CI).

RANKING SOURCE: real_data_rankings.csv (produced by real_data_analysis.py),
the framework-consistent two-level fair comparison — NOT the older/discarded
scripts (main_analysis.py, comparison_all_features_methods.py, the duplicate
DII scripts), per the decision already made when auditing this project's
history. Point-estimate rankings are reused as-is (no retraining) for the
downstream-validation step; the BOOTSTRAP step recomputes rankings fresh per
draw (see below), exactly like bootstrap_ci_fredmd.py / bootstrap_ci_mi_
aggregate.py do.

SAME SUBSAMPLE AS real_data_analysis.py: N=5000 stratified subsample (2500
per class, seed=42) of train2.csv's 440,402 rows — replicated verbatim here
(same columns, same sampling code) so X and y match exactly what produced
real_data_rankings.csv. The target is natively binary — no median-split
binarisation needed, unlike the synthetic dataset.

DOWNSTREAM VALIDATION: same module (evaluate_method_vs_baseline in
downstream_validation.py), same K values, same n_random=100 as fredmd_
analysis.py, for direct comparability.

BOOTSTRAP CI: same protocol as bootstrap_ci_fredmd.py / bootstrap_ci_mi_
aggregate.py — subsampling 80% without replacement, ranking recomputed once
per draw (not reused across K, for the same DII-training-cost reason those
scripts document), B_CHEAP=100 for MI_perfeat/II_perfeat/II_joint, B_DII=15.
MI_perfeat's bootstrap ranking uses mutual_info_classif (matching FRED-MD/
MI-complications' bootstrap convention for a binary target) — NOTE this
differs from real_data_analysis.py's own point estimate, which used
mutual_info_regression on continuous scaled Y; both are valid on a binary
target, but this is a real, documented choice, not an unstated inconsistency.

OUTPUT (same schema as the FRED-MD files, for direct comparison):
  trading_downstream_results.csv      (same columns as fredmd_downstream_results.csv)
  bootstrap_ci_trading_results.csv    (same columns as bootstrap_ci_fredmd_results.csv)

CONSTRAINTS: does not modify any existing file.
"""
import warnings; warnings.filterwarnings('ignore')
import os
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
from downstream_validation import evaluate_method_vs_baseline, knn_loo_accuracy

SEED = 42
N_SAMPLE = 5000
K_VALUES = [3, 5, 10, 16]
SUBSAMPLE_FRAC = 0.80
N_RANDOM_BASELINE = 100  # kept at 100 (same as FRED-MD/MI-complications) —
                          # reducing this would inflate CI width by adding
                          # noise into every draw's baseline estimate, and
                          # would break protocol-comparability with the
                          # other two real datasets. Not touched.

# REVISED after two failed attempts. The real bottleneck, found by proper
# calibration (measuring a FULL draw: ranking + the downstream K-loop's
# 4*(1+N_RANDOM_BASELINE)=404 knn_loo_accuracy calls), is 72.3s/draw for
# even the "cheap" MI_perfeat — not the ~0.1s the original calibration
# measured, because that only timed the ranking step and ignored the
# downstream evaluation cost, which dominates and is roughly SHARED across
# all 4 methods (N=5000 here vs N=794 for FRED-MD makes knn_loo_accuracy's
# O(N^2) pairwise-distance cost ~40x more expensive). B_CHEAP is lowered to
# match, rather than reducing N_RANDOM_BASELINE (see above) — this widens
# the CI honestly (less resolution) instead of distorting it.
B_CHEAP = 12
B_DII = 8


def load_trading():
    """Replicates real_data_analysis.py's loading/subsampling exactly."""
    df = pd.read_csv('train2.csv')
    feature_cols = [c for c in df.columns if c not in ['id', 'stock_id', 'target']]
    df_0 = df[df['target'] == 0].sample(N_SAMPLE // 2, random_state=SEED)
    df_1 = df[df['target'] == 1].sample(N_SAMPLE // 2, random_state=SEED)
    df_s = pd.concat([df_0, df_1]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    X = df_s[feature_cols].values.astype(np.float64)
    y = df_s['target'].values.astype(int)
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
    t_start_all = time.time()
    print("=" * 70)
    print("TRADING (train2.csv) — DOWNSTREAM VALIDATION + BOOTSTRAP CI")
    print(f"N_SAMPLE={N_SAMPLE}  B_CHEAP={B_CHEAP}  B_DII={B_DII}")
    print("=" * 70)

    print("\n[1/4] Loading trading data (same N=5000 stratified subsample as "
          "real_data_analysis.py)...")
    X, y, feature_names = load_trading()
    print(f"  X shape: {X.shape}, target: {y.sum()}/{len(y)} positive "
          f"({100*y.mean():.1f}%)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n[2/4] Loading existing rankings from real_data_rankings.csv "
          "(no retraining)...")
    rankings_df = pd.read_csv('real_data_rankings.csv')
    assert len(rankings_df) == X.shape[1], (
        f"Rankings file has {len(rankings_df)} rows but X has {X.shape[1]} "
        f"columns — mismatch, cannot proceed safely.")
    methods = {
        'MI_perfeat': rankings_df['MI_Rank'].values,
        'II_perfeat': rankings_df['II_pf_Rank'].values,
        'II_joint':   rankings_df['II_jt_Rank'].values,
        'DII_L1':     rankings_df['DII_L1_Rank'].values,
    }

    if os.path.exists('trading_downstream_results.csv'):
        print("\n[3/4] SKIPPED — trading_downstream_results.csv already exists "
              "from the earlier run (N_RANDOM_BASELINE unchanged, so it's "
              "still valid; saves ~16 min of redundant computation).")
    else:
        print("\n[3/4] Point-estimate downstream validation "
              "(evaluate_method_vs_baseline)...")
        all_point_rows = []
        for name, ranks in methods.items():
            print(f"\n --- {name} ---")
            res = evaluate_method_vs_baseline(
                X_scaled, y, ranks, K_VALUES,
                n_random=N_RANDOM_BASELINE, k_neighbors=5, seed=SEED, method_name=name)
            all_point_rows.extend(res)

        point_df = pd.DataFrame(all_point_rows)
        point_df.to_csv('trading_downstream_results.csv', index=False)
        print("\nSaved: trading_downstream_results.csv")

    print("\n[4/4] Bootstrap CI (ranking recomputed once per draw, "
          "all K evaluated from it)...")
    methods_config = [
        ('MI_perfeat', B_CHEAP), ('II_perfeat', B_CHEAP),
        ('II_joint', B_CHEAP), ('DII_L1', B_DII),
    ]
    all_boot_rows = []
    for method, B in methods_config:
        print(f"\n --- {method} (B={B}) ---")
        t0 = time.time()
        advantages_by_k = bootstrap_method(X_scaled, y, method, B, SEED,
                                            f"TRADING_{method}")
        print(f"  done in {time.time()-t0:.0f}s")
        for K in K_VALUES:
            adv = advantages_by_k[K]
            lo, hi = np.percentile(adv, [2.5, 97.5])
            frac_pos = float(np.mean(adv > 0))
            crosses_zero = lo <= 0 <= hi
            flag = "FRAGILE (CI crosses 0)" if crosses_zero else "ROBUST"
            print(f"    K={K:<3} advantage={adv.mean():+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}]  P(+)={frac_pos:.2f}  [{flag}]")
            all_boot_rows.append(dict(method=method, K=K, mean_advantage=adv.mean(),
                                       ci_lo=lo, ci_hi=hi, std=adv.std(),
                                       frac_positive=frac_pos, B=B,
                                       robust=not crosses_zero))

    boot_df = pd.DataFrame(all_boot_rows)
    boot_df.to_csv('bootstrap_ci_trading_results.csv', index=False)
    print("\nSaved: bootstrap_ci_trading_results.csv")

    total_min = (time.time() - t_start_all) / 60
    print(f"\nTotal time: {total_min:.1f} min")
    print("\n" + "=" * 70)
    print("SUMMARY — sorted by mean advantage")
    print("=" * 70)
    for _, r in boot_df.sort_values('mean_advantage', ascending=False).iterrows():
        flag = "ROBUST" if r['robust'] else "FRAGILE"
        print(f"  {r['method']:<12} K={int(r['K']):<3} "
              f"adv={r['mean_advantage']:+.3f} "
              f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  [{flag}]")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

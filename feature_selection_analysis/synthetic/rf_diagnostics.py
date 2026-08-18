"""
RF Diagnostics — n_estimators stability + near-duplicate feature sensitivity
================================================================================
Closes two gaps flagged when comparing RF's testing depth to MINE/DII before
treating it as comparable in the method-comparison table:

  1. Forest-size stability check (the analogue of MINE's per-epoch convergence
     check): does the permutation-importance ranking change meaningfully as
     n_estimators grows past 100 (the value used throughout
     rf_synthetic_highdim.py)? Checked at n_estimators = 50, 100, 200, 500.

  2. Near-duplicate feature sensitivity: permutation importance is documented
     in the literature to DILUTE importance across correlated/duplicate
     features rather than concentrating it on one, unlike DII+L1 (which
     concentrates weight on 1-2 copies and zeroes the rest — see
     simulation_study_v5_summary.txt). Checked directly on the x_base + 4
     near-duplicate copies block.

Uses p=27 (same representative level as MINE's diagnostics), imports
rf_ranking-equivalent logic and constants from rf_synthetic_highdim.py rather
than reimplementing. Does not modify any existing file.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.stats import rankdata, kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'common'))     # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # simulation_study_v6_highdim.py lives here
from simulation_study_v6_highdim import generate_core_dataset
from rf_synthetic_highdim import SEED, N_REPEATS

N_ESTIMATORS_LEVELS = [50, 100, 200, 500]

# DII+L1 weights on the same near-duplicate block, from
# simulation_study_v5_summary.txt, for direct comparison.
DII_L1_DUP_WEIGHTS = {
    'x_base': 0.1001, 'x_dup_1': 0.1240, 'x_dup_2': 0.0619,
    'x_dup_3': 0.0000, 'x_dup_4': 0.0000,
}


def rf_ranking_at(X_scaled, Y_scaled, n_estimators, seed=SEED):
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    rf.fit(X_scaled, Y_scaled)
    perm = permutation_importance(rf, X_scaled, Y_scaled, n_repeats=N_REPEATS,
                                   random_state=seed, n_jobs=-1)
    importance = perm.importances_mean
    ranks = rankdata(-importance).astype(int)
    return importance, ranks


if __name__ == "__main__":
    X, Y, feature_names, feature_groups, gt_binary, gt_rank = generate_core_dataset()
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = (Y - Y.mean()) / Y.std()

    print("=" * 70)
    print("CHECK 1 — n_estimators stability (p=27)")
    print("=" * 70)
    results = {}
    for n_est in N_ESTIMATORS_LEVELS:
        imp, ranks = rf_ranking_at(X_scaled, Y_scaled, n_est)
        tau_vs_gt, _ = kendalltau(gt_rank, ranks)
        results[n_est] = (imp, ranks, tau_vs_gt)
        print(f"  n_estimators={n_est:<4} tau_vs_ground_truth={tau_vs_gt:.3f}")

    print("\n  Agreement between consecutive n_estimators levels (tau between rankings):")
    for i in range(len(N_ESTIMATORS_LEVELS) - 1):
        n1, n2 = N_ESTIMATORS_LEVELS[i], N_ESTIMATORS_LEVELS[i + 1]
        t, _ = kendalltau(results[n1][1], results[n2][1])
        print(f"    {n1} vs {n2}: tau={t:.3f}")

    tau_100 = results[100][2]
    tau_500 = results[500][2]
    rel_change = abs(tau_500 - tau_100) / abs(tau_100) if tau_100 != 0 else np.inf
    stable = rel_change < 0.05
    print(f"\n  n_estimators=100 (used throughout rf_synthetic_highdim.py) vs "
          f"n_estimators=500: tau {tau_100:.3f} -> {tau_500:.3f} "
          f"(rel. change {rel_change:.1%})  "
          f"{'STABLE' if stable else 'NOT STABLE -- consider increasing n_estimators'}")

    print("\n" + "=" * 70)
    print("CHECK 2 — near-duplicate feature sensitivity (x_base + 4 copies)")
    print("=" * 70)
    imp_100, ranks_100, _ = results[100]
    dup_features = ['x_base', 'x_dup_1', 'x_dup_2', 'x_dup_3', 'x_dup_4']
    dup_idx = [np.where(feature_names == f)[0][0] for f in dup_features]
    print(f"\n  {'Feature':<10} {'Importance':>12} {'Rank':>6}")
    for f, i in zip(dup_features, dup_idx):
        print(f"  {f:<10} {imp_100[i]:>12.5f} {ranks_100[i]:>6}")

    total_imp = imp_100.sum()
    dup_imp_values = imp_100[dup_idx]
    dup_imp_share = dup_imp_values.sum() / total_imp
    rf_concentration = dup_imp_values.max() / dup_imp_values.sum() if dup_imp_values.sum() > 0 else np.nan

    dii_total = sum(DII_L1_DUP_WEIGHTS.values())
    dii_concentration = max(DII_L1_DUP_WEIGHTS.values()) / dii_total

    print(f"\n  Combined importance share of the 5 near-duplicate features: {dup_imp_share:.1%}")
    print(f"  RF concentration (max single feature's share of the 5-feature total): "
          f"{rf_concentration:.1%}")
    print(f"  DII+L1 concentration on the same block (for comparison): "
          f"{dii_concentration:.1%} (on x_dup_1; x_dup_3/x_dup_4 zeroed completely)")
    if rf_concentration < 0.5:
        print("  -> DILUTED: RF's importance is spread fairly evenly across the "
              "duplicates rather than concentrated on one, consistent with the "
              "known literature limitation of permutation importance under "
              "correlated/redundant features.")
    else:
        print("  -> CONCENTRATED: RF favours one feature in the block, similar "
              "in spirit to DII+L1's behaviour.")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

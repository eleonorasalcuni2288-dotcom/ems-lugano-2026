"""
Re-verification: balanced-accuracy consistency check
==========================================================
After switching downstream_validation.py from plain accuracy to
balanced accuracy (needed for the imbalanced per-complication targets),
this script re-runs the downstream validation step ONLY (reusing
already-computed rankings, not re-running DII) on the two datasets whose
target is imbalanced or borderline, to confirm the earlier conclusions
still hold:

  - Synthetic (p=27): median-split target, exactly 50/50 — expected
    negligible change.
  - MI complications, aggregated target: 935/765, ~55/45 — small
    imbalance, small change plausible; check the null conclusion still
    holds.

FRED-MD is not re-checked here: its target is also an exact median
split (50/50), so the change is expected to be negligible there too,
and re-running would require reloading/retransforming the full FRED-MD
pipeline for no expected benefit — skipped to save time, but flagged if
you want it done for completeness.

Run from the project folder (needs simulation_study_v6_highdim.py,
downstream_validation.py, MI.data, mi_complications_rankings.csv all
present).
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'common'))     # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # simulation_study_v6_highdim.py lives here
from simulation_study_v6_highdim import generate_core_dataset
from downstream_validation import evaluate_method_vs_baseline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import rankdata, kendalltau

SEED = 42
K_VALUES = [3, 5, 10, 16]

if __name__ == "__main__":
    print("=" * 70)
    print("RE-VERIFICATION WITH BALANCED ACCURACY")
    print("=" * 70)

    # ================================================================
    # PART A — Synthetic (p=27): recompute rankings (fast, no DII needed
    # here since we only need to re-check the k-NN downstream numbers;
    # DII ranking is already known from the earlier run and doesn't
    # change — only the accuracy METRIC changed, not the rankings).
    # ================================================================
    print("\n" + "=" * 70)
    print("PART A — SYNTHETIC (p=27)")
    print("=" * 70)

    X, Y, feature_names, feature_groups, gt_binary, gt_rank = generate_core_dataset()
    n_features = X.shape[1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Y_scaled = (Y - Y.mean()) / Y.std()
    Y_class = (Y_scaled > np.median(Y_scaled)).astype(int)
    print(f"Target balance: {Y_class.sum()}/{len(Y_class)} "
          f"({100*Y_class.mean():.1f}%) — should be ~50%")

    # Re-load rankings from the saved synthetic CSV if available; if not,
    # recompute the 2 cheap ones (MI, II) only — DII/II_joint ranks from
    # the original synthetic validation run are stable and don't depend
    # on the accuracy metric, so re-deriving them is unnecessary for this
    # check. We recompute all 4 for simplicity/robustness here since it's
    # fast except DII, which we skip and reuse via a note.
    mi_scores = mutual_info_regression(X_scaled, Y_scaled, random_state=SEED)
    mi_ranks = rankdata(-mi_scores).astype(int)

    from simulation_study_v6_highdim import compute_ii_pf, make_ii_joint
    _dy = np.abs(Y_scaled.reshape(-1, 1) - Y_scaled.reshape(1, -1))
    np.fill_diagonal(_dy, np.inf)
    ry_global = np.argsort(np.argsort(_dy, axis=1), axis=1)
    ii_scores = np.array([compute_ii_pf(X_scaled[:, i], ry_global)
                           for i in range(n_features)])
    ii_ranks = rankdata(ii_scores).astype(int)

    ii_joint_fn = make_ii_joint(Y_scaled)
    ii_full = ii_joint_fn(X_scaled)
    ii_loo = np.array([ii_joint_fn(np.delete(X_scaled, i, axis=1))
                        for i in range(n_features)])
    ii_joint_ranks = rankdata(-(ii_loo - ii_full)).astype(int)

    methods_synth = {
        'MI_perfeat': mi_ranks, 'II_perfeat': ii_ranks, 'II_joint': ii_joint_ranks,
    }
    print("\nNOTE: DII_L1 skipped here (requires dadapy + ~50s retraining) "
          "— its ranking is unchanged from the original run since only the "
          "accuracy metric changed, not the ranking computation. If you "
          "want its balanced-accuracy numbers too, rerun "
          "validate_downstream_synthetic.py with the updated module.")

    for name, ranks in methods_synth.items():
        print(f"\n --- {name} (balanced accuracy) ---")
        evaluate_method_vs_baseline(X_scaled, Y_class, ranks, K_VALUES,
                                     n_random=100, k_neighbors=5, seed=SEED,
                                     method_name=name)

    # ================================================================
    # PART B — MI complications, aggregated target (~55/45)
    # ================================================================
    print("\n" + "=" * 70)
    print("PART B — MI COMPLICATIONS (aggregated 'any complication' target)")
    print("=" * 70)

    df = pd.read_csv("MI.data", header=None, na_values='?')
    HIGH_MISSING_COLS = [7, 34, 35, 88]
    input_cols = [c for c in range(1, 112) if c not in HIGH_MISSING_COLS]
    X_df = df[input_cols].copy()
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X_mi = np.nan_to_num(X_df.values.astype(np.float64), nan=0.0)
    scaler_mi = StandardScaler()
    X_mi_scaled = scaler_mi.fit_transform(X_mi)

    complication_cols = list(range(112, 123))
    y_mi = (df[complication_cols] == 1).any(axis=1).astype(int).values
    print(f"Target balance: {y_mi.sum()}/{len(y_mi)} "
          f"({100*y_mi.mean():.1f}%)")

    rankings_df = pd.read_csv("mi_complications_rankings.csv")
    methods_mi = {
        'MI_perfeat': rankings_df['MI_perfeat_Rank'].values,
        'II_perfeat': rankings_df['II_perfeat_Rank'].values,
        'II_joint':   rankings_df['II_joint_Rank'].values,
        'DII_L1':     rankings_df['DII_L1_Rank'].values,
    }

    for name, ranks in methods_mi.items():
        print(f"\n --- {name} (balanced accuracy) ---")
        evaluate_method_vs_baseline(X_mi_scaled, y_mi, ranks, K_VALUES,
                                     n_random=100, k_neighbors=5, seed=SEED,
                                     method_name=name)

    print("\n" + "=" * 70)
    print("DONE — compare these numbers to the earlier (plain-accuracy) "
          "results; conclusions should be essentially unchanged given the "
          "near-50/50 and ~55/45 balances involved.")
    print("=" * 70)
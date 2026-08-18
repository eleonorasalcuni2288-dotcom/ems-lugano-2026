"""
Verification: balanced-accuracy impact on MI complications RF check
======================================================================
Checks the claim that switching to balanced accuracy has negligible
impact on mi_complications_rf_results.csv (computed before the switch).
Unlike FRED-MD (exact 50/50), the aggregate MI complications target is
55/45 — NOT an exact split, so plain and balanced accuracy are not
mathematically guaranteed to coincide here. This verification actually
tests that, rather than assuming it.

Reuses the ALREADY-COMPUTED rankings (mi_complications_rankings.csv) —
no DII retraining needed.

Run from the project folder (needs MI.data, mi_complications_rankings.csv,
downstream_validation.py, mi_complications_rf_results.csv).
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'common'))     # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # simulation_study_v6_highdim.py lives here
from downstream_validation import evaluate_method_vs_baseline_rf

DATA_PATH = "MI.data"
HIGH_MISSING_COLS = [7, 34, 35, 88]
K_VALUES = [3, 5, 10, 16]
SEED = 42


def load_mi_aggregate():
    df = pd.read_csv(DATA_PATH, header=None, na_values='?')
    input_cols = [c for c in range(1, 112) if c not in HIGH_MISSING_COLS]
    X_df = df[input_cols].copy()
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = np.nan_to_num(X_df.values.astype(np.float64), nan=0.0)
    complication_cols = list(range(112, 123))
    y = (df[complication_cols] == 1).any(axis=1).astype(int).values
    return X, y


if __name__ == "__main__":
    print("=" * 70)
    print("VERIFY: plain vs balanced accuracy impact on MI complications "
          "RF check")
    print("=" * 70)

    print("\n[1/3] Reloading X, y...")
    X, y = load_mi_aggregate()
    print(f"  X shape: {X.shape}, target: {y.sum()}/{len(y)} "
          f"({100*y.mean():.1f}% positive — NOT an exact 50/50 split)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n[2/3] Loading saved rankings (no DII retraining needed)...")
    rankings_df = pd.read_csv("mi_complications_rankings.csv")
    assert len(rankings_df) == X.shape[1], (
        f"Rankings file has {len(rankings_df)} rows but X has "
        f"{X.shape[1]} columns — mismatch, cannot proceed safely.")
    methods = {
        'MI_perfeat': rankings_df['MI_perfeat_Rank'].values,
        'II_perfeat': rankings_df['II_perfeat_Rank'].values,
        'II_joint':   rankings_df['II_joint_Rank'].values,
        'DII_L1':     rankings_df['DII_L1_Rank'].values,
    }

    print("\n[3/3] Re-running RF downstream validation with CURRENT "
          "(balanced-accuracy) module...")
    new_rows = []
    for name, ranks in methods.items():
        print(f"\n --- {name} (balanced accuracy, RF) ---")
        res = evaluate_method_vs_baseline_rf(
            X_scaled, y, ranks, K_VALUES,
            n_random=50, n_estimators=50, cv=5, seed=SEED, method_name=name)
        for r in res:
            new_rows.append(r)

    new_df = pd.DataFrame(new_rows)
    old_df = pd.read_csv("mi_complications_rf_results.csv")

    print("\n" + "=" * 70)
    print("COMPARISON: old (plain accuracy) vs new (balanced accuracy)")
    print("=" * 70)
    merged = old_df.merge(new_df, on=['method', 'K'], suffixes=('_old_plain', '_new_balanced'))
    merged['acc_diff'] = (merged['method_acc_new_balanced'] - merged['method_acc_old_plain']).abs()
    print(merged[['method', 'K', 'method_acc_old_plain', 'method_acc_new_balanced',
                  'acc_diff', 'p_value_old_plain', 'p_value_new_balanced']].to_string(index=False))

    max_diff = merged['acc_diff'].max()
    mean_diff = merged['acc_diff'].mean()
    print(f"\nMax absolute difference in accuracy: {max_diff:.4f}")
    print(f"Mean absolute difference in accuracy: {mean_diff:.4f}")

    conclusion_unchanged = (
        (old_df['p_value'] < 0.05).sum() == (new_df['p_value'] < 0.05).sum()
    )
    print(f"\nConclusion (null result) preserved under both metrics: "
          f"{conclusion_unchanged}")
    if max_diff < 0.02:
        print("CONFIRMED negligible: max difference under 2 percentage "
              "points, consistent with the target being close to balanced "
              "(55/45).")
    else:
        print("NOTE: difference is non-trivial — report the exact numbers "
              "in the poster rather than asserting negligibility.")

    merged.to_csv('verify_mi_rf_balanced_comparison.csv', index=False)
    print("\nSaved: verify_mi_rf_balanced_comparison.csv")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

"""
Verification: balanced-accuracy impact on FRED-MD point estimates
======================================================================
Checks the claim that switching downstream_validation.py from plain to
balanced accuracy has negligible impact on fredmd_downstream_results.csv
(computed before the switch), since FRED-MD's target is an exact median
split (50/50) — where plain and balanced accuracy are mathematically
identical for any classifier.

Reuses the ALREADY-COMPUTED rankings (fredmd_rankings.csv) — no DII
retraining needed, this only re-runs the downstream evaluation step with
the current (balanced-accuracy) module and compares to the old values.

Run from the project folder (needs 2026-07-MD.csv, fredmd_rankings.csv,
downstream_validation.py, fredmd_downstream_results.csv).
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'common'))     # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # simulation_study_v6_highdim.py lives here
from downstream_validation import evaluate_method_vs_baseline

DATA_PATH = "2026-07-MD.csv"
TARGET_COL = "S&P 500"
LEAKAGE_COLS = ['S&P div yield', 'S&P PE ratio']
K_VALUES = [3, 5, 10, 16]
SEED = 42


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


if __name__ == "__main__":
    print("=" * 70)
    print("VERIFY: plain vs balanced accuracy impact on FRED-MD point estimates")
    print("=" * 70)

    print("\n[1/3] Reloading X, y (must match what produced fredmd_rankings.csv)...")
    X, y, feature_names = load_fredmd()
    print(f"  X shape: {X.shape}, target: {y.sum()}/{len(y)} "
          f"({100*y.mean():.1f}% positive — exact median split)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n[2/3] Loading saved rankings (no DII retraining needed)...")
    rankings_df = pd.read_csv("fredmd_rankings.csv")
    assert len(rankings_df) == X.shape[1], (
        f"Rankings file has {len(rankings_df)} rows but X has "
        f"{X.shape[1]} columns — mismatch, cannot proceed safely.")
    methods = {
        'MI_perfeat': rankings_df['MI_perfeat_Rank'].values,
        'II_perfeat': rankings_df['II_perfeat_Rank'].values,
        'II_joint':   rankings_df['II_joint_Rank'].values,
        'DII_L1':     rankings_df['DII_L1_Rank'].values,
    }

    print("\n[3/3] Re-running downstream validation with CURRENT "
          "(balanced-accuracy) module...")
    new_rows = []
    for name, ranks in methods.items():
        print(f"\n --- {name} (balanced accuracy) ---")
        res = evaluate_method_vs_baseline(
            X_scaled, y, ranks, K_VALUES,
            n_random=100, k_neighbors=5, seed=SEED, method_name=name)
        for r in res:
            new_rows.append(r)

    new_df = pd.DataFrame(new_rows)
    old_df = pd.read_csv("fredmd_downstream_results.csv")

    print("\n" + "=" * 70)
    print("COMPARISON: old (plain accuracy) vs new (balanced accuracy)")
    print("=" * 70)
    merged = old_df.merge(new_df, on=['method', 'K'], suffixes=('_old_plain', '_new_balanced'))
    merged['acc_diff'] = (merged['method_acc_new_balanced'] - merged['method_acc_old_plain']).abs()
    print(merged[['method', 'K', 'method_acc_old_plain', 'method_acc_new_balanced',
                  'acc_diff', 'p_value_old_plain', 'p_value_new_balanced']].to_string(index=False))

    max_diff = merged['acc_diff'].max()
    print(f"\nMax absolute difference in accuracy: {max_diff:.6f}")
    if max_diff < 1e-9:
        print("CONFIRMED: identical to numerical precision — plain and "
              "balanced accuracy coincide exactly on this 50/50 target, "
              "as expected mathematically.")
    elif max_diff < 0.01:
        print("CONFIRMED negligible: differences are within random-baseline "
              "noise (<0.01), consistent with the near-exact 50/50 split.")
    else:
        print("WARNING: difference is larger than expected — investigate "
              "before treating as negligible in the poster.")

    merged.to_csv('verify_fredmd_balanced_comparison.csv', index=False)
    print("\nSaved: verify_fredmd_balanced_comparison.csv")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

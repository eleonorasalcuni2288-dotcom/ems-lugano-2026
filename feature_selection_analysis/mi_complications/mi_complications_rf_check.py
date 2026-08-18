"""
MI Complications — Random Forest Secondary Check
====================================================
Re-evaluates the SAME feature rankings already computed and saved by
mi_complications_analysis.py (mi_complications_rankings.csv), but using
Random Forest + 5-fold CV instead of k-NN + LOO for the downstream
validation. Purpose: rule out that the null result seen with k-NN was
an artifact of k-NN's Euclidean-distance metric, which is a known weak
fit for datasets with many binary/categorical features mixed with
continuous ones (as here).

Does NOT recompute MI/II/DII rankings (that requires re-running DII,
the slow ~1min step) — reuses the rankings CSV already produced.

Run from the same folder as mi_complications_analysis.py and MI.data,
with the venv activated:
    python3 mi_complications_rf_check.py
"""
import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)                              # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # simulation_study_v6_highdim.py lives here
from downstream_validation import evaluate_method_vs_baseline_rf

K_VALUES = [3, 5, 10, 16]
SEED = 42

if __name__ == "__main__":
    print("=" * 70)
    print("MI COMPLICATIONS — RANDOM FOREST SECONDARY CHECK")
    print("=" * 70)

    # ---- 1. Reconstruct X, y exactly as in mi_complications_analysis.py ---
    print("\n[1/3] Reloading and rebuilding data (same steps as before)...")
    df = pd.read_csv("MI.data", header=None, na_values='?')
    HIGH_MISSING_COLS = [7, 34, 35, 88]
    input_cols = [c for c in range(1, 112) if c not in HIGH_MISSING_COLS]
    X_df = df[input_cols].copy()
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = X_df.values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)  # safety net, matches original script

    complication_cols = list(range(112, 123))
    y = (df[complication_cols] == 1).any(axis=1).astype(int).values
    print(f"  X shape: {X.shape}, target: {y.sum()}/{len(y)} positive")

    # ---- 2. Load the already-computed rankings ----------------------------
    print("\n[2/3] Loading previously computed rankings...")
    rankings_df = pd.read_csv("mi_complications_rankings.csv")
    methods = {
        'MI_perfeat': rankings_df['MI_perfeat_Rank'].values,
        'II_perfeat': rankings_df['II_perfeat_Rank'].values,
        'II_joint':   rankings_df['II_joint_Rank'].values,
        'DII_L1':     rankings_df['DII_L1_Rank'].values,
    }
    assert len(rankings_df) == X.shape[1], (
        f"Rankings file has {len(rankings_df)} rows but X has "
        f"{X.shape[1]} columns — did the cleaning step change since the "
        f"rankings were computed? Re-run mi_complications_analysis.py first."
    )

    # ---- 3. RF downstream validation --------------------------------------
    print("\n[3/3] Random Forest + 5-fold CV downstream validation...")
    print("  (n_random=50, not 200 — coarser p-value resolution, "
          "min achievable p = 1/50 = 0.02, acceptable for this "
          "confirmatory check given the time budget)")
    t0 = time.time()
    all_results = {}
    for name, ranks in methods.items():
        print(f"\n --- {name} ---")
        all_results[name] = evaluate_method_vs_baseline_rf(
            X, y, ranks, K_VALUES, n_random=50, n_estimators=50,
            cv=5, seed=SEED, method_name=name)
    print(f"\nTotal RF check time: {(time.time()-t0)/60:.1f} min")

    rows = [r for res in all_results.values() for r in res]
    pd.DataFrame(rows).to_csv('mi_complications_rf_results.csv', index=False)
    print("Saved: mi_complications_rf_results.csv")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
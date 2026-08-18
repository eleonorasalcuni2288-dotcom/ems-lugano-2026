"""
MI Complications — Per-Complication Analysis (Individual Targets)
======================================================================
Tests the hypothesis that the earlier null result (aggregated "any
complication" target) was caused by heterogeneity: 11 clinically
different complications, likely with different predictors, averaged
into one binary label may cancel out real signal.

Here each of the 11 binary complications is used as a SEPARATE target,
producing its own MI/II/DII rankings and downstream validation. Uses
balanced accuracy (not plain accuracy) throughout, since prevalence
ranges from 1.2% (PREDS_TAH) to 23.2% (ZSN) — plain accuracy would be
dominated by the majority class for the rarer ones.

MULTIPLE TESTING: this script runs 11 complications x 4 methods x 4 K
values = 176 significance tests. A naive p<0.05 threshold would expect
~9 false positives by chance alone even if there were no real effect
anywhere. The summary at the end reports the Benjamini-Hochberg
correction alongside raw p-values — any complication/method reported as
"significant" in the poster should survive that correction, not just
the raw threshold.

Run from the same folder as MI.data, simulation_study_v6_highdim.py and
downstream_validation.py, with the venv activated. This is slower than
the single-target script (~11x the ranking computation, dominated by
DII+L1 training): budget ~15-20 minutes.
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
from downstream_validation import evaluate_method_vs_baseline

SEED = 42
DATA_PATH = "MI.data"
K_VALUES = [3, 5, 10, 16]
HIGH_MISSING_COLS = [7, 34, 35, 88]

COMPLICATION_COLS = {
    112: 'FIBR_PREDS', 113: 'PREDS_TAH', 114: 'JELUD_TAH', 115: 'FIBR_JELUD',
    116: 'A_V_BLOK', 117: 'OTEK_LANC', 118: 'RAZRIV', 119: 'DRESSLER',
    120: 'ZSN', 121: 'REC_IM', 122: 'P_IM_STEN',
}

if __name__ == "__main__":
    print("=" * 70)
    print("MI COMPLICATIONS — PER-COMPLICATION ANALYSIS (11 individual targets)")
    print("=" * 70)

    # ---- 1. Load & clean X once (same as before, target changes per loop) --
    print("\n[1/2] Loading and cleaning X (shared across all 11 targets)...")
    df = pd.read_csv(DATA_PATH, header=None, na_values='?')
    input_cols = [c for c in range(1, 112) if c not in HIGH_MISSING_COLS]
    X_df = df[input_cols].copy()
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = np.nan_to_num(X_df.values.astype(np.float64), nan=0.0)
    n_features = X.shape[1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  X shape: {X.shape}")

    # ---- 2. Loop over each complication ------------------------------------
    print("\n[2/2] Running full pipeline per complication "
          "(ranking + downstream validation)...")
    all_rows = []
    t_start = time.time()

    for col_idx, comp_name in COMPLICATION_COLS.items():
        y = (df[col_idx] == 1).astype(int).values
        prevalence = y.mean()
        print(f"\n{'='*70}\n  {comp_name} (col {col_idx}): "
              f"{y.sum()}/{len(y)} positive ({100*prevalence:.1f}%)\n{'='*70}")

        if y.sum() < 10:
            print("  SKIPPED: fewer than 10 positive cases, too few for "
                  "reliable LOO/CV evaluation.")
            continue

        y_float = y.astype(np.float64)

        t0 = time.time()
        mi_scores = mutual_info_classif(X_scaled, y, random_state=SEED)
        mi_ranks = rankdata(-mi_scores).astype(int)

        _dy = np.abs(y_float.reshape(-1, 1) - y_float.reshape(1, -1))
        np.fill_diagonal(_dy, np.inf)
        ry_global = np.argsort(np.argsort(_dy, axis=1), axis=1)
        ii_scores = np.array([compute_ii_pf(X_scaled[:, i], ry_global)
                               for i in range(n_features)])
        ii_ranks = rankdata(ii_scores).astype(int)

        ii_joint_fn = make_ii_joint(y_float)
        ii_full = ii_joint_fn(X_scaled)
        ii_loo = np.array([ii_joint_fn(np.delete(X_scaled, i, axis=1))
                            for i in range(n_features)])
        ii_joint_ranks = rankdata(-(ii_loo - ii_full)).astype(int)

        dii_l1_w, dii_l1_ranks, dii_l1_imbs = run_dii(
            X_scaled, y_float, 0.10, f"DII+L1 ({comp_name})", N=len(y))

        print(f"  Rankings computed in {time.time()-t0:.1f}s "
              f"(DII final imbalance={dii_l1_imbs[-1]:.4f})")

        methods = {
            'MI_perfeat': mi_ranks, 'II_perfeat': ii_ranks,
            'II_joint': ii_joint_ranks, 'DII_L1': dii_l1_ranks,
        }

        print("  Downstream validation (balanced accuracy, n_random=100)...")
        for name, ranks in methods.items():
            res = evaluate_method_vs_baseline(
                X_scaled, y, ranks, K_VALUES,
                n_random=100, k_neighbors=5, seed=SEED,
                method_name=f"{comp_name}_{name}")
            for r in res:
                r['complication'] = comp_name
                r['prevalence'] = prevalence
                all_rows.append(r)

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

    # ---- 3. Summary with multiple-testing correction -----------------------
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv('mi_complications_per_target_results.csv', index=False)
    print("\nSaved: mi_complications_per_target_results.csv")

    valid = results_df.dropna(subset=['p_value']).copy()
    valid = valid.sort_values('p_value').reset_index(drop=True)
    n_tests = len(valid)
    valid['bh_threshold'] = (np.arange(1, n_tests + 1) / n_tests) * 0.05
    valid['bh_significant'] = valid['p_value'] <= valid['bh_threshold']

    print("\n" + "=" * 70)
    print(f"MULTIPLE TESTING CORRECTION (Benjamini-Hochberg, {n_tests} tests)")
    print("=" * 70)
    n_raw_sig = (valid['p_value'] < 0.05).sum()
    n_bh_sig = valid['bh_significant'].sum()
    print(f"  Raw p<0.05: {n_raw_sig} / {n_tests} "
          f"(expect ~{0.05*n_tests:.1f} by chance alone if no real effect)")
    print(f"  BH-corrected significant: {n_bh_sig} / {n_tests}")

    if n_bh_sig > 0:
        print("\n  Results surviving BH correction:")
        surviving = valid[valid['bh_significant']].sort_values('p_value')
        print(surviving[['complication', 'method', 'K', 'method_acc',
                          'baseline_mean', 'p_value']].to_string(index=False))
    else:
        print("\n  No result survives BH correction — consistent with the "
              "aggregated-target null result; per-complication splitting "
              "does not, on its own, recover a robust signal.")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
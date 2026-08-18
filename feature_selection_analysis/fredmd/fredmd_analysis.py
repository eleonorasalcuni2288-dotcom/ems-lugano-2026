"""
FRED-MD — Real-Data High-Dimensional Feature Ranking (Third Dataset)
========================================================================
Third dataset, added under an explicit 30-minute time-box after the null
result on MI complications. Macroeconomic panel (McCracken & Ng, FRED-MD
vintage), monthly, natively p=127 series.

Design:
  - Apply the standard FRED-MD transformation codes (row 0 of the raw
    file: 1=level, 2=diff, 3=diff^2, 4=log, 5=diff(log), 6=diff^2(log),
    7=diff(pct_change)) to every series, making them stationary.
  - Target: S&P 500 (transform code 5 = approx. monthly log-return).
    Binarized via median split ("above-median month" vs not) for
    classification, consistent with the synthetic/MI-complications setup.
  - Drop columns with >20% missing after transformation (ACOGNO,
    TWEXAFEGSMTHx, UMCSENTx — verified by direct inspection, not assumed).
  - Drop rows with >5% missing features (mostly leading rows lost to
    differencing), median-impute the rest.
  - Result: N=794, p=120 (verified) — comfortably above the >100 threshold.

CAVEAT to state in the poster: this is a macro-financial domain (monthly,
different frequency and nature from the trading/synthetic study), used
here as an independent real-data check on the same feature-selection
comparison, not as a direct extension of the trading narrative.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'common'))     # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # simulation_study_v6_highdim.py lives here
from simulation_study_v6_highdim import compute_ii_pf, make_ii_joint, run_dii
from downstream_validation import evaluate_method_vs_baseline

SEED = 42
DATA_PATH = "2026-07-MD.csv"   # place in the same folder as this script
K_VALUES = [3, 5, 10, 16]
TARGET_COL = "S&P 500"


def apply_transform(series, code):
    if code == 1: return series
    if code == 2: return series.diff()
    if code == 3: return series.diff().diff()
    if code == 4: return np.log(series)
    if code == 5: return np.log(series).diff()
    if code == 6: return np.log(series).diff().diff()
    if code == 7: return series.pct_change().diff()
    return series


if __name__ == "__main__":
    print("=" * 70)
    print("FRED-MD — REAL-DATA HIGH-DIMENSIONAL FEATURE RANKING")
    print("=" * 70)

    # ---- 1. Load, apply standard transforms --------------------------------
    print("\n[1/6] Loading and applying FRED-MD transformation codes...")
    df = pd.read_csv(DATA_PATH)
    transform_codes = df.iloc[0].drop('sasdate').astype(float)
    data = df.iloc[1:].reset_index(drop=True)
    for c in data.columns:
        if c != 'sasdate':
            data[c] = pd.to_numeric(data[c], errors='coerce')

    transformed = {c: apply_transform(data[c], transform_codes[c])
                   for c in transform_codes.index}
    transformed = pd.DataFrame(transformed)

    # ---- 2. Clean: drop high-missing columns, then rows -----------------
    print("[2/6] Cleaning...")
    miss_pct = transformed.isna().mean()
    high_missing = miss_pct[miss_pct > 0.20].index.tolist()
    print(f"  Dropping {len(high_missing)} columns (>20% missing): {high_missing}")
    transformed = transformed.drop(columns=high_missing)

    # LEAKAGE GUARD: S&P div yield and S&P PE ratio are mathematically
    # derived from the S&P 500 price itself (P/E = Price/Earnings,
    # div yield = Dividends/Price). Since the target is Δlog(S&P 500
    # price), including these as features means the target's own price
    # component leaks directly into two "predictors" — this produced
    # near-tautological accuracy (~0.9, p=0.000 almost everywhere) on
    # the first run and must be excluded, not just noted as a caveat.
    LEAKAGE_COLS = ['S&P div yield', 'S&P PE ratio']
    transformed = transformed.drop(columns=[c for c in LEAKAGE_COLS
                                              if c in transformed.columns])
    print(f"  Dropping {len(LEAKAGE_COLS)} columns for target leakage "
          f"(derived from S&P 500 price): {LEAKAGE_COLS}")

    feature_cols = [c for c in transformed.columns if c != TARGET_COL]
    combined = transformed.dropna(subset=[TARGET_COL])
    row_nan_frac = combined[feature_cols].isna().mean(axis=1)
    combined = combined[row_nan_frac < 0.05]

    X_df = combined[feature_cols].fillna(combined[feature_cols].median())
    n_remaining_nan = X_df.isna().sum().sum()
    print(f"  Remaining NaNs after cleaning: {n_remaining_nan}")
    if n_remaining_nan > 0:
        X_df = X_df.fillna(0.0)
        print("  WARNING: residual NaNs found — filled with 0, flag for review.")

    X = X_df.values.astype(np.float64)
    n_features = X.shape[1]
    feature_names = np.array(feature_cols)
    print(f"  Final shape: N={X.shape[0]}, p={n_features}")

    # ---- 3. Target: binarized S&P 500 return -------------------------------
    print("\n[3/6] Constructing target...")
    y_cont = combined[TARGET_COL].values
    y = (y_cont > np.median(y_cont)).astype(int)
    print(f"  Above-median-return target: {y.sum()}/{len(y)} positive "
          f"({100*y.mean():.1f}%)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_float = y.astype(np.float64)

    # ---- 4. Rank features ---------------------------------------------------
    print("\n[4/6] Computing MI (per-feature)...")
    mi_scores = mutual_info_classif(X_scaled, y, random_state=SEED)
    mi_ranks = rankdata(-mi_scores).astype(int)

    print("[5/6] Computing II per-feature, II joint (LOO), DII+L1 "
          "(slowest step, ~1 min)...")
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
        X_scaled, y_float, 0.10, "DII+L1 (FRED-MD)", N=len(y))
    print(f"  DII+L1 final imbalance: {dii_l1_imbs[-1]:.4f}")

    methods = {
        'MI_perfeat': mi_ranks,
        'II_perfeat': ii_ranks,
        'II_joint':   ii_joint_ranks,
        'DII_L1':     dii_l1_ranks,
    }

    print("\n" + "=" * 70)
    print("TOP-12 FEATURES PER METHOD")
    print("=" * 70)
    for name, ranks in methods.items():
        top12 = feature_names[np.argsort(ranks)[:12]]
        print(f"  {name:<12} {list(top12)}")

    # ---- 6. Downstream validation (k-NN only, time-boxed) ------------------
    print("\n" + "=" * 70)
    print("DOWNSTREAM VALIDATION: k-NN LOO accuracy vs randomized baseline")
    print("(n_random=100, not 200 — time-boxed given this is the 3rd dataset)")
    print("=" * 70)
    all_results = {}
    for name, ranks in methods.items():
        print(f"\n --- {name} ---")
        all_results[name] = evaluate_method_vs_baseline(
            X_scaled, y, ranks, K_VALUES,
            n_random=100, k_neighbors=5, seed=SEED, method_name=name)

    rows = [r for res in all_results.values() for r in res]
    pd.DataFrame(rows).to_csv('fredmd_downstream_results.csv', index=False)
    print("\nSaved: fredmd_downstream_results.csv")

    rankings_df = pd.DataFrame({'Feature': feature_names})
    for name, ranks in methods.items():
        rankings_df[f'{name}_Rank'] = ranks
    rankings_df.to_csv('fredmd_rankings.csv', index=False)
    print("Saved: fredmd_rankings.csv")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
"""
MI Complications — Real-Data High-Dimensional Feature Ranking
================================================================
Second real dataset (after FRED-MD was set aside): Myocardial Infarction
complications (Golovenkin et al.), UCI/Leicester repository,
DOI: 10.25392/leicester.data.12045261.v3.

N=1700 patients, columns 1-111 = clinical/demographic input features
(natively high-dimensional, p=111 before cleaning), columns 112-123 =
12 complication indicators (11 binary + LET_IS multi-class death cause).

Design decisions (agreed in conversation):
  - Prediction timepoint: "third day" (all 111 input columns usable,
    no admission-time exclusions) — keeps p native and >100.
  - Drop columns with >50% missing (cols 7, 34, 35, 88 — verified via
    direct inspection, not just the text description) rather than
    impute them: too sparse to impute reliably.
  - Remaining missing values: median imputation per column.
  - Target: binary "any complication" = 1 if any of the 11 binary
    complication columns (112-122) is positive, else 0. LET_IS (123,
    multi-class death cause) excluded from target construction.

IMPORTANT CAVEAT (state explicitly in the poster, not just in code):
Unlike the synthetic study, there is no continuous ground-truth signal
here — the reference target for II/DII is the binary complication
indicator itself. This gives II/DII a coarser neighbour structure than
in the synthetic case (where the underlying Z was continuous), and there
is no ground-truth ranking to check tau against. This is exactly why the
downstream validation module (accuracy vs randomized baseline) is used
here as the primary evidence, rather than tau/rho.
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
DATA_PATH = "MI.data"   # place in the same folder as this script
K_VALUES = [3, 5, 10, 16]

if __name__ == "__main__":
    print("=" * 70)
    print("MI COMPLICATIONS — REAL-DATA HIGH-DIMENSIONAL FEATURE RANKING")
    print("=" * 70)

    # ---- 1. Load & clean --------------------------------------------------
    print("\n[1/6] Loading and cleaning data...")
    df = pd.read_csv(DATA_PATH, header=None, na_values='?')
    assert df.shape == (1700, 124), f"Unexpected shape {df.shape}"

    HIGH_MISSING_COLS = [7, 34, 35, 88]   # verified >50% missing
    input_cols = [c for c in range(1, 112) if c not in HIGH_MISSING_COLS]
    print(f"  Input columns: {len(input_cols)} "
          f"(111 native - {len(HIGH_MISSING_COLS)} dropped for >50% missing)")

    X_df = df[input_cols].copy()
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = X_df.values.astype(np.float64)
    remaining_nan = np.isnan(X).sum()
    print(f"  Remaining NaNs after median imputation: {remaining_nan}")
    if remaining_nan > 0:
        # a column that was ALL NaN would survive median imputation as NaN;
        # fall back to 0 for any such residual, and flag it loudly.
        print("  WARNING: residual NaNs found — filling with 0 and flagging "
              "for manual review (likely an all-NaN column).")
        X = np.nan_to_num(X, nan=0.0)

    n_features = X.shape[1]
    feature_ids = np.array(input_cols)  # original column numbers, for reporting

    # ---- 2. Target: any complication ---------------------------------------
    print("\n[2/6] Constructing target...")
    complication_cols = list(range(112, 123))  # 11 binary complication cols
    y = (df[complication_cols] == 1).any(axis=1).astype(int).values
    print(f"  Any-complication target: {y.sum()}/{len(y)} positive "
          f"({100*y.mean():.1f}%)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_float = y.astype(np.float64)  # used as the "reference space" for II/DII

    # ---- 3. Rank features: MI, II per-feature, II joint LOO, DII+L1 -------
    print("\n[3/6] Computing MI (per-feature)...")
    mi_scores = mutual_info_classif(X_scaled, y, random_state=SEED)
    mi_ranks = rankdata(-mi_scores).astype(int)

    print("[4/6] Computing II (per-feature)...")
    _dy = np.abs(y_float.reshape(-1, 1) - y_float.reshape(1, -1))
    np.fill_diagonal(_dy, np.inf)
    ry_global = np.argsort(np.argsort(_dy, axis=1), axis=1)
    ii_scores = np.array([compute_ii_pf(X_scaled[:, i], ry_global)
                           for i in range(n_features)])
    ii_ranks = rankdata(ii_scores).astype(int)

    print("[5/6] Computing II joint (LOO)...")
    ii_joint_fn = make_ii_joint(y_float)
    ii_full = ii_joint_fn(X_scaled)
    ii_loo = np.array([ii_joint_fn(np.delete(X_scaled, i, axis=1))
                        for i in range(n_features)])
    ii_joint_ranks = rankdata(-(ii_loo - ii_full)).astype(int)

    print("[6/6] Computing DII+L1 (this is the slow step, ~1-2 min)...")
    # N must be passed explicitly: run_dii's k_init/k_final are fractions of
    # N, and the default baked into the imported function is N=2000 (the
    # synthetic study's sample size) — this dataset has N=1700.
    dii_l1_w, dii_l1_ranks, dii_l1_imbs = run_dii(
        X_scaled, y_float, 0.10, "DII+L1 (MI complications)", N=len(y))
    print(f"  DII+L1 final imbalance: {dii_l1_imbs[-1]:.4f}")

    methods = {
        'MI_perfeat': mi_ranks,
        'II_perfeat': ii_ranks,
        'II_joint':   ii_joint_ranks,
        'DII_L1':     dii_l1_ranks,
    }

    # ---- 4. Report top-12 per method (by original column id) --------------
    print("\n" + "=" * 70)
    print("TOP-12 FEATURES PER METHOD (original column numbers)")
    print("=" * 70)
    for name, ranks in methods.items():
        top12 = feature_ids[np.argsort(ranks)[:12]]
        print(f"  {name:<12} {list(top12)}")

    # ---- 5. Downstream validation (primary evidence — see caveat above) ---
    print("\n" + "=" * 70)
    print("DOWNSTREAM VALIDATION: k-NN LOO accuracy vs randomized baseline")
    print("=" * 70)
    all_results = {}
    for name, ranks in methods.items():
        print(f"\n --- {name} ---")
        all_results[name] = evaluate_method_vs_baseline(
            X_scaled, y, ranks, K_VALUES,
            n_random=200, k_neighbors=5, seed=SEED, method_name=name)

    # ---- 6. Save results -----------------------------------------------
    rows = []
    for name, res in all_results.items():
        for r in res:
            rows.append(r)
    results_df = pd.DataFrame(rows)
    results_df.to_csv('post_infarction_downstream_results.csv', index=False)
    print("\nSaved: post_infarction_downstream_results.csv")

    rankings_df = pd.DataFrame({'Feature_ID': feature_ids})
    for name, ranks in methods.items():
        rankings_df[f'{name}_Rank'] = ranks
    rankings_df.to_csv('post_infarction_rankings.csv', index=False)
    print("Saved: post_infarction_rankings.csv")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
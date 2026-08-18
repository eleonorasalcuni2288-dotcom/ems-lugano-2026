"""
Random Forest — General-Purpose ML Comparison (permutation importance)
======================================================================
Completes the method comparison table alongside MI/II/DII (classical /
lightweight differentiable) and InfoNCE/MINE (general-purpose neural ML):
Random Forest represents a THIRD, non-neural general-purpose ML paradigm
— same "no built-in problem structure, must learn everything from the
data" category as InfoNCE/MINE, but via tree ensembles instead of a
trained neural critic.

WHY THIS IS MUCH CHEAPER THAN INFONCE/MINE: those methods needed LOO
retraining (1 + p separate trainings) to get a per-feature importance,
because a trained neural critic has no natural per-feature decomposition.
A Random Forest DOES: after training ONCE on all p features, permutation
importance (shuffle one column, measure the resulting drop in prediction
score, repeated N_REPEATS times per feature) gives per-feature importance
from a SINGLE trained model — no retraining per feature, no retraining
per LOO fold. This is what makes B=20 bootstrap draws trivially cheap
here where it required real budgeting for InfoNCE (B=6) and careful
calibration for MINE (B=20 needed 56 min).

TARGET TYPE: Y is CONTINUOUS on this synthetic dataset (same convention
as MI_perfeat / II_perfeat / II_joint / DII, all of which rank against
continuous Y) — RandomForestRegressor is used for RANKING, not
RandomForestClassifier (which this project already uses elsewhere, but
only for datasets with a genuinely binary target). Y is separately
binarised via median split ONLY for the downstream_validation.py call,
matching every other method in this project's synthetic-dataset work.

HYPERPARAMETERS (explicit, not left to sklearn defaults where it
matters): n_estimators=100 (project's existing RF check elsewhere uses
50 as a "secondary confirmatory check"; bumped to 100 here since RF is
a PRIMARY comparison method in this script, and the cost difference is
negligible). n_repeats=10 for permutation_importance (sklearn's default
is 5; doubled here for more stable importance estimates, affordable
given how cheap each evaluation is). random_state=SEED=42 throughout,
consistent with the rest of this project.

SCALABILITY (p=27, 50, 105): reuses generate_core_dataset() and
build_padded_dataset() by IMPORTING them (not reimplemented), same
noise pool (SEED+1=43) as every other high-dim analysis in this project.

BOOTSTRAP CI: subsampling 80% without replacement, SAME seed=42 for
every draw's RF fit (only the resampled DATA varies between draws,
matching how DII/II_joint/MI_joint are bootstrapped elsewhere in this
project) — B=20, matching the top of the range used for MINE. NOTE:
B was NOT pushed higher despite RF's low cost easily allowing it — kept
at 20 deliberately for methodological comparability with the other
methods' bootstrap CIs in the final table, not because 20 was a cost
ceiling here (it emphatically was not).

INTRINSIC STOCHASTICITY CHECK (same spirit as the MINE diagnostics):
the full p=27 model trained 3x with different random_state (42,43,44),
same data, to report how much of RF's own randomness (bootstrap
sampling, per-split feature subsampling) affects the importance ranking
independent of data resampling.

OUTPUT:
  rf_synthetic_highdim_results.csv          — bootstrap CI, SAME SCHEMA
    as bootstrap_ci_synthetic_highdim_results.csv / mine_synthetic_
    highdim_results.csv for direct union/comparison.
  rf_synthetic_highdim_point_estimates.csv  — tau, XOR rank, downstream
    validation, per p.
  rf_synthetic_highdim_summary.txt          — human-readable summary.

CONSTRAINTS: does not modify any existing file.
"""

import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)                              # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # simulation_study_v6_highdim.py lives here
from simulation_study_v6_highdim import generate_core_dataset
from bootstrap_ci_synthetic_highdimensional import build_padded_dataset, SEED, N
from infonce_synthetic import K_VALUES
from downstream_validation import evaluate_method_vs_baseline

P_LEVELS = [27, 50, 105]
N_ESTIMATORS = 100
N_REPEATS = 10
SUBSAMPLE_FRAC = 0.80
CI_LEVEL = 95
B_BOOTSTRAP = 20
CONVERGENCE_REP_P = 27   # representative p for the stochasticity check


def rf_ranking(X_scaled, Y_scaled, seed=SEED):
    """Train once, get per-feature importance via permutation (no LOO
    retraining needed). Returns (importance, ranks)."""
    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1)
    rf.fit(X_scaled, Y_scaled)
    perm = permutation_importance(rf, X_scaled, Y_scaled, n_repeats=N_REPEATS,
                                   random_state=seed, n_jobs=-1)
    importance = perm.importances_mean
    ranks = rankdata(-importance).astype(int)
    return importance, ranks


if __name__ == "__main__":
    t_start_all = time.time()
    print("=" * 70)
    print("RANDOM FOREST — GENERAL-PURPOSE ML COMPARISON (permutation importance)")
    print(f"n_estimators={N_ESTIMATORS}  n_repeats={N_REPEATS}  B_bootstrap={B_BOOTSTRAP}")
    print("=" * 70)

    X_core, Y, core_names, core_groups, core_binary, core_rank = generate_core_dataset()
    n_core = X_core.shape[1]
    max_extra = max(P_LEVELS) - n_core
    rng_pad = np.random.default_rng(SEED + 1)
    noise_pool = rng_pad.normal(0, 1, size=(N, max_extra))

    Y_scaled = (Y - Y.mean()) / Y.std()
    Y_class = (Y_scaled > np.median(Y_scaled)).astype(int)

    # ---- Intrinsic stochasticity check (3 seeds, same p=27 data) --------------
    print(f"\n[Stochasticity check] p={CONVERGENCE_REP_P}, 3 seeds, same data...")
    X_rep, _ = build_padded_dataset(CONVERGENCE_REP_P, X_core, core_names, core_groups,
                                     core_binary, core_rank, noise_pool)
    X_rep_scaled = StandardScaler().fit_transform(X_rep)
    stoch_seeds = [42, 43, 44]
    stoch_importances = []
    for s in stoch_seeds:
        imp, _ = rf_ranking(X_rep_scaled, Y_scaled, seed=s)
        stoch_importances.append(imp)
        print(f"  seed={s}: importance vector computed "
              f"(mean={imp.mean():.5f}, max={imp.max():.5f})")
    stoch_importances = np.array(stoch_importances)
    # per-feature std across the 3 seeds, then averaged — a single summary
    # number for "how much does RF's own randomness move each feature's
    # importance, independent of data resampling"
    stoch_std_avg = float(stoch_importances.std(axis=0).mean())
    print(f"  avg per-feature std across 3 seeds: {stoch_std_avg:.5f}  "
          f"(intrinsic estimator noise, independent of data resampling)")

    point_rows = []
    bootstrap_rows = []

    for p in P_LEVELS:
        print(f"\n{'='*70}\n  p = {p}\n{'='*70}")
        X_p, gt_rank = build_padded_dataset(
            p, X_core, core_names, core_groups, core_binary, core_rank, noise_pool)
        if p > n_core:
            n_extra = p - n_core
            feature_names = np.concatenate(
                [core_names, [f'hd_noise_{i+1}' for i in range(n_extra)]])
        else:
            feature_names = core_names
        syn_idx = [np.where(feature_names == f)[0][0] for f in ['x_syn_1', 'x_syn_2']]

        X_scaled = StandardScaler().fit_transform(X_p)

        # ---- A. POINT ESTIMATE -----------------------------------------------
        t0 = time.time()
        importance, ranks = rf_ranking(X_scaled, Y_scaled)
        t_elapsed = time.time() - t0
        tau, _ = kendalltau(gt_rank, ranks)
        syn_r1, syn_r2 = int(ranks[syn_idx[0]]), int(ranks[syn_idx[1]])
        print(f"  [point p={p}] tau={tau:.3f}  XOR=({syn_r1},{syn_r2})  [{t_elapsed:.2f}s]")

        ds_results = evaluate_method_vs_baseline(
            X_scaled, Y_class, ranks, K_VALUES,
            n_random=100, k_neighbors=5, seed=SEED, method_name=f"RF_p{p}")
        for r in ds_results:
            r.update(dict(p=p, tau=tau, syn_rank_1=syn_r1, syn_rank_2=syn_r2))
            point_rows.append(r)

        # ---- B. BOOTSTRAP CI ---------------------------------------------------
        print(f"\n  Bootstrap CI (B={B_BOOTSTRAP})...")
        n = X_scaled.shape[0]
        n_sub = int(round(n * SUBSAMPLE_FRAC))
        rng = np.random.default_rng(SEED)
        taus_boot = np.empty(B_BOOTSTRAP)
        t0 = time.time()
        for b in range(B_BOOTSTRAP):
            idx = rng.choice(n, size=n_sub, replace=False)
            Xb, yb = X_scaled[idx], Y_scaled[idx]
            _, ranks_b = rf_ranking(Xb, yb, seed=SEED)
            t, _ = kendalltau(gt_rank, ranks_b)
            taus_boot[b] = t
        lo, hi = np.percentile(taus_boot, [(100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2])
        bootstrap_rows.append(dict(
            p=p, method='RF', tau_mean=float(taus_boot.mean()),
            ci_lo=float(lo), ci_hi=float(hi), tau_std=float(taus_boot.std()),
            B=B_BOOTSTRAP))
        print(f"  Bootstrap tau = {taus_boot.mean():.3f} [{lo:.3f}, {hi:.3f}]  "
              f"(B={B_BOOTSTRAP}, {time.time()-t0:.1f}s)")

    # ---- Save outputs -------------------------------------------------------
    point_df = pd.DataFrame(point_rows)
    point_df.to_csv('rf_synthetic_highdim_point_estimates.csv', index=False)
    print("\nSaved: rf_synthetic_highdim_point_estimates.csv")

    boot_df = pd.DataFrame(bootstrap_rows)
    boot_df.to_csv('rf_synthetic_highdim_results.csv', index=False)
    print("Saved: rf_synthetic_highdim_results.csv "
          "(same schema as bootstrap_ci_synthetic_highdim_results.csv)")

    total_min = (time.time() - t_start_all) / 60
    with open('rf_synthetic_highdim_summary.txt', 'w') as f:
        f.write("RANDOM FOREST — GENERAL-PURPOSE ML COMPARISON\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total runtime: {total_min:.1f} min\n")
        f.write(f"n_estimators={N_ESTIMATORS}  n_repeats={N_REPEATS}  "
                f"B_bootstrap={B_BOOTSTRAP}\n\n")
        f.write(f"--- STOCHASTICITY CHECK (p={CONVERGENCE_REP_P}, 3 seeds) ---\n")
        f.write(f"  avg per-feature importance std across seeds: {stoch_std_avg:.5f}\n\n")
        f.write("--- POINT ESTIMATES (tau, XOR pair rank) ---\n")
        for _, r in point_df.drop_duplicates('p').iterrows():
            f.write(f"  p={int(r.p):<4} tau={r.tau:.3f}  "
                    f"XOR rank=({int(r.syn_rank_1)},{int(r.syn_rank_2)})\n")
        f.write("\n--- BOOTSTRAP CI (tau) ---\n")
        for _, r in boot_df.iterrows():
            f.write(f"  p={int(r.p):<4} tau_mean={r.tau_mean:.3f}  "
                    f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]  (B={int(r.B)})\n")

    print("Saved: rf_synthetic_highdim_summary.txt")
    print(f"\nTotal time: {total_min:.1f} min")
    print("\n" + "=" * 70)
    print("RANDOM FOREST ANALYSIS COMPLETE")
    print("=" * 70)

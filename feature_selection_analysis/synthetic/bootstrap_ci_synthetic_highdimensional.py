"""
Bootstrap CI — Synthetic, High-Dimensional Levels (p=50, p=105)
======================================================================
Extends bootstrap_ci_synthetic.py (which covered p=27 only) to the two
higher-dimensionality levels used in the scalability sweep
(simulation_study_v6_highdim.py), completing the uncertainty picture
across the full p=[27,50,105] range.

The noise-padding logic (core 27 features + extra pure-noise columns to
reach p=50/105) is duplicated here from simulation_study_v6_highdim.py's
__main__ block, since it isn't exposed as an importable function there.
Uses the SAME seed (SEED+1 for the noise pool) as that script, so the
padding is IDENTICAL to what was used in the original sweep — this
dataset is not a new/different one, just the same p=50/105 datasets
already analysed, now with resampling-based uncertainty added.

Ground truth for the padded noise columns uses tied ranks (average of
the remaining rank slots), matching simulation_study_v6_highdim.py's
convention, since the noise features are exchangeable/equally
uninformative — not the earlier v6 bug (fixed) of assigning them
arbitrary sequential ranks.

Run from the project folder (needs simulation_study_v6_highdim.py).
Budget: DII is the bottleneck. At p=50, DII+L1 takes ~54s/run (measured
in the original sweep); at p=105, ~123s/run. With B_DII=15:
  p=50:  15 x 54s  ~ 13.5 min
  p=105: 15 x 123s ~ 31 min
Total (both levels, all 4 methods): ~50-60 min.
"""
import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)                              # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # simulation_study_v6_highdim.py lives here
from simulation_study_v6_highdim import (
    generate_core_dataset, compute_ii_pf, make_ii_joint, run_dii,
)

SEED = 42
SUBSAMPLE_FRAC = 0.80
B_CHEAP = 100
B_DII = 15
CI_LEVEL = 95
P_LEVELS = [50, 105]
N = 2000  # matches simulation_study_v6_highdim.py's default


def build_padded_dataset(p, X_core, core_names, core_groups, core_binary,
                          core_rank, noise_pool):
    n_core = X_core.shape[1]
    n_extra = p - n_core
    if n_extra <= 0:
        return X_core, core_rank.astype(float)
    X_p = np.column_stack([X_core, noise_pool[:, :n_extra]])
    tie_rank = np.arange(n_core + 1, n_core + 1 + n_extra).mean()
    extra_rank = np.full(n_extra, tie_rank)
    gt_rank = np.concatenate([core_rank.astype(float), extra_rank])
    return X_p, gt_rank


def rank_mi(Xb, yb):
    scores = mutual_info_regression(Xb, yb, random_state=SEED)
    return rankdata(-scores).astype(int)


def rank_ii_perfeat(Xb, yb):
    dy = np.abs(yb.reshape(-1, 1) - yb.reshape(1, -1))
    np.fill_diagonal(dy, np.inf)
    ry = np.argsort(np.argsort(dy, axis=1), axis=1)
    scores = np.array([compute_ii_pf(Xb[:, i], ry) for i in range(Xb.shape[1])])
    return rankdata(scores).astype(int)


def rank_ii_joint(Xb, yb):
    ii_joint_fn = make_ii_joint(yb)
    full = ii_joint_fn(Xb)
    loo = np.array([ii_joint_fn(np.delete(Xb, i, axis=1))
                     for i in range(Xb.shape[1])])
    return rankdata(-(loo - full)).astype(int)


def rank_dii_l1(Xb, yb, label):
    _, ranks, _ = run_dii(Xb, yb, 0.10, label, N=len(yb))
    return ranks


def subsample_tau_ci(X_full, y_full, gt_rank, rank_fn, B, seed, label=""):
    n = X_full.shape[0]
    n_sub = int(round(n * SUBSAMPLE_FRAC))
    rng = np.random.default_rng(seed)
    taus = np.empty(B)
    for b in range(B):
        idx = rng.choice(n, size=n_sub, replace=False)
        Xb, yb = X_full[idx], y_full[idx]
        ranks_b = rank_fn(Xb, yb, b) if label else rank_fn(Xb, yb)
        t, _ = kendalltau(gt_rank, ranks_b)
        taus[b] = t
        if label and (b + 1) % max(1, B // 5) == 0:
            print(f"    {label}: {b+1}/{B} draws done")
    lo, hi = np.percentile(taus, [(100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2])
    return taus, lo, hi


if __name__ == "__main__":
    print("=" * 70)
    print("BOOTSTRAP CI — SYNTHETIC, HIGH-DIMENSIONAL (p=50, p=105)")
    print(f"Subsample fraction: {SUBSAMPLE_FRAC}  |  "
          f"B_cheap={B_CHEAP}  B_DII={B_DII}  |  {CI_LEVEL}% percentile CI")
    print("=" * 70)

    X_core, Y, core_names, core_groups, core_binary, core_rank = generate_core_dataset()
    n_core = X_core.shape[1]
    max_extra = max(P_LEVELS) - n_core
    rng_pad = np.random.default_rng(SEED + 1)  # SAME seed as the original sweep
    noise_pool = rng_pad.normal(0, 1, size=(N, max_extra))

    Y_scaled = (Y - Y.mean()) / Y.std()

    all_rows = []
    t_start_all = time.time()

    for p in P_LEVELS:
        print(f"\n{'='*70}\n  p = {p}\n{'='*70}")
        X_p, gt_rank = build_padded_dataset(
            p, X_core, core_names, core_groups, core_binary, core_rank, noise_pool)
        scaler = StandardScaler()
        X_p_scaled = scaler.fit_transform(X_p)

        results = {}

        print(f"\n[MI_perfeat, p={p}]")
        taus, lo, hi = subsample_tau_ci(
            X_p_scaled, Y_scaled, gt_rank,
            lambda Xb, yb: rank_mi(Xb, yb), B_CHEAP, SEED)
        results['MI_perfeat'] = (taus, lo, hi)
        print(f"  tau = {taus.mean():.3f}  [{lo:.3f}, {hi:.3f}]")

        print(f"\n[II_perfeat, p={p}]")
        taus, lo, hi = subsample_tau_ci(
            X_p_scaled, Y_scaled, gt_rank,
            lambda Xb, yb: rank_ii_perfeat(Xb, yb), B_CHEAP, SEED)
        results['II_perfeat'] = (taus, lo, hi)
        print(f"  tau = {taus.mean():.3f}  [{lo:.3f}, {hi:.3f}]")

        print(f"\n[II_joint, p={p}]")
        taus, lo, hi = subsample_tau_ci(
            X_p_scaled, Y_scaled, gt_rank,
            lambda Xb, yb: rank_ii_joint(Xb, yb), B_CHEAP, SEED)
        results['II_joint'] = (taus, lo, hi)
        print(f"  tau = {taus.mean():.3f}  [{lo:.3f}, {hi:.3f}]")

        print(f"\n[DII_L1, p={p}] (slow: ~{B_DII} draws)")
        taus, lo, hi = subsample_tau_ci(
            X_p_scaled, Y_scaled, gt_rank,
            lambda Xb, yb, b: rank_dii_l1(Xb, yb, f"DII bootstrap p{p} {b}"),
            B_DII, SEED, label=f"DII_L1_p{p}")
        results['DII_L1'] = (taus, lo, hi)
        print(f"  tau = {taus.mean():.3f}  [{lo:.3f}, {hi:.3f}]")

        for name, (taus, lo, hi) in results.items():
            all_rows.append(dict(p=p, method=name, tau_mean=taus.mean(),
                                  ci_lo=lo, ci_hi=hi, tau_std=taus.std(),
                                  B=len(taus)))

    print(f"\nTotal time: {(time.time()-t_start_all)/60:.1f} min")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv('bootstrap_ci_synthetic_highdim_results.csv', index=False)
    print("\nSaved: bootstrap_ci_synthetic_highdim_results.csv")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for p in P_LEVELS:
        print(f"\n  p = {p}")
        sub = results_df[results_df.p == p]
        for _, r in sub.iterrows():
            print(f"    {r['method']:<12} tau = {r['tau_mean']:.3f}  "
                  f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  (B={int(r['B'])})")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
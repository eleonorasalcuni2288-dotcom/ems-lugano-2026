"""
Bootstrap Confidence Intervals — Synthetic Dataset (p=27)
================================================================
Quantifies uncertainty on tau (rank correlation with known ground truth)
for each method, via repeated resampling — the "priority high, never
done" item flagged at the very start of this project (gap vs. Trofimov's
poster, which has uncertainty quantification and this one didn't).

METHODOLOGICAL CHOICE — subsampling, not classic bootstrap:
Classic bootstrap resamples WITH replacement, which creates duplicate
rows. For distance/neighbour-based methods (II, II_joint, DII — all rely
on "who is my nearest neighbour"), a duplicated row is a trivial,
zero-distance nearest neighbour of itself, which artificially inflates
these metrics' apparent quality. This is not a hypothetical concern —
it's a direct consequence of how these methods are computed. To avoid
it, this script uses subsampling WITHOUT replacement (80% of N per
draw) instead — same resampling logic for uncertainty quantification,
without the duplicate-neighbour artifact. This should be stated
explicitly if asked, not left as an implicit assumption.

COST / ASYMMETRIC B: DII+L1 training takes ~50s per run (300 epochs);
running it at the same B as the cheap methods (MI, II) would take too
long. B is therefore asymmetric across methods:
  - B_CHEAP = 100  for MI_perfeat, II_perfeat, II_joint (fast, seconds total)
  - B_DII    = 20  for DII_L1 (~20 x 50s ~ 17 min)
This asymmetry is a real trade-off (DII's CI is estimated from fewer
draws, hence coarser) and should be reported as such, not hidden.

Run from the project folder (needs simulation_study_v6_highdim.py):
    python3 bootstrap_ci_synthetic.py
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
_sys.path.insert(0, _os.path.join(_ROOT, 'common'))     # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # simulation_study_v6_highdim.py lives here
from simulation_study_v6_highdim import (
    generate_core_dataset, compute_ii_pf, make_ii_joint, run_dii,
)

SEED = 42
SUBSAMPLE_FRAC = 0.80
B_CHEAP = 100
B_DII = 20
CI_LEVEL = 95  # percentile CI: [2.5, 97.5]


def rank_mi(Xb, yb, seed):
    scores = mutual_info_regression(Xb, yb, random_state=seed)
    return rankdata(-scores).astype(int)


def rank_ii_perfeat(Xb, yb):
    n = len(yb)
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


def rank_dii_l1(Xb, yb, seed, label):
    _, ranks, _ = run_dii(Xb, yb, 0.10, label, N=len(yb))
    return ranks


def subsample_ci(X_full, y_full, gt_rank, rank_fn, B, seed, label=""):
    """Subsample WITHOUT replacement (SUBSAMPLE_FRAC of N), recompute the
    method's ranking on each draw, compute tau against the FIXED ground
    truth, and return the array of tau values plus a percentile CI."""
    n = X_full.shape[0]
    n_sub = int(round(n * SUBSAMPLE_FRAC))
    rng = np.random.default_rng(seed)
    taus = np.empty(B)

    for b in range(B):
        idx = rng.choice(n, size=n_sub, replace=False)
        Xb, yb = X_full[idx], y_full[idx]
        ranks_b = rank_fn(Xb, yb, b)
        t, _ = kendalltau(gt_rank, ranks_b)
        taus[b] = t
        if label and (b + 1) % max(1, B // 5) == 0:
            print(f"    {label}: {b+1}/{B} draws done")

    lo, hi = np.percentile(taus, [(100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2])
    return taus, lo, hi


if __name__ == "__main__":
    print("=" * 70)
    print("BOOTSTRAP (SUBSAMPLING) CONFIDENCE INTERVALS — SYNTHETIC (p=27)")
    print(f"Subsample fraction: {SUBSAMPLE_FRAC}  |  "
          f"B_cheap={B_CHEAP}  B_DII={B_DII}  |  {CI_LEVEL}% percentile CI")
    print("=" * 70)

    X, Y, feature_names, feature_groups, gt_binary, gt_rank = generate_core_dataset()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Y_scaled = (Y - Y.mean()) / Y.std()

    results = {}
    t_start = time.time()

    print("\n[1/4] MI_perfeat...")
    taus, lo, hi = subsample_ci(
        X_scaled, Y_scaled, gt_rank,
        lambda Xb, yb, b: rank_mi(Xb, yb, SEED), B_CHEAP, SEED)
    results['MI_perfeat'] = (taus, lo, hi)
    print(f"  tau = {taus.mean():.3f}  [{lo:.3f}, {hi:.3f}]  "
          f"(std={taus.std():.3f})")

    print("\n[2/4] II_perfeat...")
    taus, lo, hi = subsample_ci(
        X_scaled, Y_scaled, gt_rank,
        lambda Xb, yb, b: rank_ii_perfeat(Xb, yb), B_CHEAP, SEED)
    results['II_perfeat'] = (taus, lo, hi)
    print(f"  tau = {taus.mean():.3f}  [{lo:.3f}, {hi:.3f}]  "
          f"(std={taus.std():.3f})")

    print("\n[3/4] II_joint...")
    taus, lo, hi = subsample_ci(
        X_scaled, Y_scaled, gt_rank,
        lambda Xb, yb, b: rank_ii_joint(Xb, yb), B_CHEAP, SEED)
    results['II_joint'] = (taus, lo, hi)
    print(f"  tau = {taus.mean():.3f}  [{lo:.3f}, {hi:.3f}]  "
          f"(std={taus.std():.3f})")

    print(f"\n[4/4] DII_L1 (slow step: ~{B_DII}x50s ~ "
          f"{B_DII*50/60:.0f} min)...")
    taus, lo, hi = subsample_ci(
        X_scaled, Y_scaled, gt_rank,
        lambda Xb, yb, b: rank_dii_l1(Xb, yb, SEED, f"DII bootstrap {b}"),
        B_DII, SEED, label="DII_L1")
    results['DII_L1'] = (taus, lo, hi)
    print(f"  tau = {taus.mean():.3f}  [{lo:.3f}, {hi:.3f}]  "
          f"(std={taus.std():.3f})")

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

    # ---- Summary table -------------------------------------------------
    print("\n" + "=" * 70)
    print(f"SUMMARY — {CI_LEVEL}% CI (subsampling, {SUBSAMPLE_FRAC:.0%} of N, "
          "without replacement)")
    print("=" * 70)
    rows = []
    for name, (taus, lo, hi) in results.items():
        B_used = len(taus)
        print(f"  {name:<12} tau = {taus.mean():.3f}  "
              f"[{lo:.3f}, {hi:.3f}]  (std={taus.std():.3f}, B={B_used})")
        rows.append(dict(method=name, tau_mean=taus.mean(), ci_lo=lo,
                          ci_hi=hi, tau_std=taus.std(), B=B_used))

    pd.DataFrame(rows).to_csv('bootstrap_ci_synthetic_results.csv', index=False)
    print("\nSaved: bootstrap_ci_synthetic_results.csv")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
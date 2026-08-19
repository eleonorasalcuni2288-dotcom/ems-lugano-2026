"""
Frequentist Coverage Check — II-joint, p=27 (synthetic benchmark)
================================================================================
Answers directly, empirically: does the project's actual 95% CI procedure
achieve close to 95% coverage of the true value, and does the formally
corrected (Politis-Romano-Wolf) procedure do better?

Inspired by a comparison with another EMS 2026 poster (Trofimov, Intrinsic
Dimension estimation) which studies frequentist coverage of its bootstrap
CIs directly -- this closes the same gap here, on a representative, cheap
case (II-joint, p=27, no DII training needed).

Design:
  1. "Truth": theta_true = E[tau] at N=2000, estimated by averaging the
     II-joint point-estimate tau over 200 INDEPENDENT fresh datasets
     (seeds 1000-1199), not reused anywhere else in this check.
  2. "Coverage test": 40 INDEPENDENT replications (seeds 2000-2039, disjoint
     from the truth-estimation seeds). For each: generate ONE dataset,
     compute the point estimate, run B=50 subsample draws (80% without
     replacement), build BOTH the current (uncorrected) CI and the
     corrected (Politis-Romano-Wolf) CI, and record whether each contains
     theta_true.
  3. Report empirical coverage (fraction of the 40 CIs containing
     theta_true) for both procedures -- the number that should be close
     to 0.95 if well-calibrated.

Does not modify any existing file.
"""
import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau
from sklearn.preprocessing import StandardScaler

from simulation_study_v6_highdim import generate_core_dataset, make_ii_joint

SUBSAMPLE_FRAC = 0.80
B = 50
N = 2000
TRUTH_SEEDS = range(1000, 1200)     # 200 independent datasets for "truth"
TEST_SEEDS = range(2000, 2040)      # 40 independent replications for coverage


def rank_ii_joint(Xb, yb):
    ii_joint_fn = make_ii_joint(yb)
    full = ii_joint_fn(Xb)
    loo = np.array([ii_joint_fn(np.delete(Xb, i, axis=1))
                     for i in range(Xb.shape[1])])
    return rankdata(-(loo - full)).astype(int)


def tau_for_seed(seed):
    X, Y, feature_names, feature_groups, gt_binary, gt_rank = generate_core_dataset(seed=seed, N=N)
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = (Y - Y.mean()) / Y.std()
    ranks = rank_ii_joint(X_scaled, Y_scaled)
    t, _ = kendalltau(gt_rank, ranks)
    return t, X_scaled, Y_scaled, gt_rank


if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70)
    print("FREQUENTIST COVERAGE CHECK -- II-joint, p=27")
    print("=" * 70)

    print(f"\n[1/2] Estimating theta_true from {len(list(TRUTH_SEEDS))} independent datasets...")
    truth_taus = []
    for i, seed in enumerate(TRUTH_SEEDS):
        t, _, _, _ = tau_for_seed(seed)
        truth_taus.append(t)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(list(TRUTH_SEEDS))} done")
    theta_true = float(np.mean(truth_taus))
    print(f"  theta_true = {theta_true:.4f}  (std across the 200 = {np.std(truth_taus):.4f})")

    print(f"\n[2/2] Coverage test: {len(list(TEST_SEEDS))} independent replications, B={B} each...")
    current_covers = []
    corrected_covers = []
    rows = []
    for rep_i, seed in enumerate(TEST_SEEDS):
        theta_n, X_scaled, Y_scaled, gt_rank = tau_for_seed(seed)
        n = X_scaled.shape[0]
        b = int(round(n * SUBSAMPLE_FRAC))

        rng = np.random.default_rng(seed * 7 + 1)  # independent from the dataset-generation seed
        draws = np.empty(B)
        for draw in range(B):
            idx = rng.choice(n, size=b, replace=False)
            Xb, yb = X_scaled[idx], Y_scaled[idx]
            ranks_b = rank_ii_joint(Xb, yb)
            t, _ = kendalltau(gt_rank, ranks_b)
            draws[draw] = t

        cur_lo, cur_hi = np.percentile(draws, [2.5, 97.5])
        cur_covers = cur_lo <= theta_true <= cur_hi

        d = np.sqrt(b) * (draws - theta_n)
        c_lo, c_hi = np.percentile(d, [2.5, 97.5])
        corr_lo = theta_n - c_hi / np.sqrt(n)
        corr_hi = theta_n - c_lo / np.sqrt(n)
        corr_covers = corr_lo <= theta_true <= corr_hi

        current_covers.append(cur_covers)
        corrected_covers.append(corr_covers)
        rows.append(dict(rep=rep_i, seed=seed, theta_hat_n=theta_n,
                          current_ci_lo=cur_lo, current_ci_hi=cur_hi, current_covers=cur_covers,
                          corrected_ci_lo=corr_lo, corrected_ci_hi=corr_hi, corrected_covers=corr_covers))
        print(f"  rep {rep_i+1}/{len(list(TEST_SEEDS))}: theta_hat={theta_n:+.3f}  "
              f"current=[{cur_lo:+.3f},{cur_hi:+.3f}]{'OK' if cur_covers else ' MISS'}  "
              f"corrected=[{corr_lo:+.3f},{corr_hi:+.3f}]{'OK' if corr_covers else ' MISS'}")

    n_reps = len(list(TEST_SEEDS))
    current_coverage = np.mean(current_covers)
    corrected_coverage = np.mean(corrected_covers)

    print("\n" + "=" * 70)
    print(f"theta_true = {theta_true:.4f}")
    print(f"CURRENT (uncorrected) empirical coverage:   {current_coverage:.1%}  "
          f"({sum(current_covers)}/{n_reps})  -- nominal target: 95%")
    print(f"CORRECTED (Politis-Romano-Wolf) coverage:   {corrected_coverage:.1%}  "
          f"({sum(corrected_covers)}/{n_reps})  -- nominal target: 95%")
    print("=" * 70)

    df = pd.DataFrame(rows)
    df.to_csv('coverage_check_results.csv', index=False)
    with open('coverage_check_summary.txt', 'w') as f:
        f.write(f"theta_true = {theta_true:.4f} (avg tau over {len(list(TRUTH_SEEDS))} independent N=2000 datasets)\n")
        f.write(f"CURRENT (uncorrected) empirical coverage: {current_coverage:.1%} ({sum(current_covers)}/{n_reps})\n")
        f.write(f"CORRECTED (Politis-Romano-Wolf) coverage: {corrected_coverage:.1%} ({sum(corrected_covers)}/{n_reps})\n")
        f.write(f"Nominal target: 95%\n")
    print("\nSaved: coverage_check_results.csv, coverage_check_summary.txt")
    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

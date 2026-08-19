"""
Subsampling Correction Check (Politis-Romano-Wolf) — II-joint on synthetic, p=27
================================================================================
Closes the last gap in the verification sweep: II-joint at p=27 has the
TIGHTEST lower CI bound of any currently-robust result on the synthetic
benchmark (ci_lo=0.097, vs 0.004-0.006 for the trading results that DID
flip) -- if even this, the tightest synthetic margin, holds up, the rest
of the synthetic benchmark's robust claims (all with wider margins) are
very unlikely to be at risk.

theta_hat_n: the full-N=2000 point estimate tau for II_joint at p=27,
read from simulation_study_v6_highdim_scalability.csv.

Does not modify any existing file.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau
from sklearn.preprocessing import StandardScaler

from simulation_study_v6_highdim import generate_core_dataset, make_ii_joint

SEED = 42
SUBSAMPLE_FRAC = 0.80
B = 100


def rank_ii_joint(Xb, yb):
    ii_joint_fn = make_ii_joint(yb)
    full = ii_joint_fn(Xb)
    loo = np.array([ii_joint_fn(np.delete(Xb, i, axis=1))
                     for i in range(Xb.shape[1])])
    return rankdata(-(loo - full)).astype(int)


if __name__ == "__main__":
    print("Generating synthetic core dataset (p=27)...")
    X, Y, feature_names, feature_groups, gt_binary, gt_rank = generate_core_dataset()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Y_scaled = (Y - Y.mean()) / Y.std()
    n = X_scaled.shape[0]
    b = int(round(n * SUBSAMPLE_FRAC))
    print(f"n={n}  b={b}  B={B}")

    point_df = pd.read_csv('simulation_study_v6_highdim_scalability.csv')
    row = point_df[(point_df.method == 'II_joint') & (point_df.p == 27)].iloc[0]
    theta_n = row.tau
    print(f"theta_hat_n (II_joint, p=27) = {theta_n:+.4f} (from simulation_study_v6_highdim_scalability.csv)")

    print(f"\nRunning {B} subsample draws (same seed=42, same procedure as bootstrap_ci_synthetic.py)...")
    rng = np.random.default_rng(SEED)
    draws = np.empty(B)
    for draw in range(B):
        idx = rng.choice(n, size=b, replace=False)
        Xb, yb = X_scaled[idx], Y_scaled[idx]
        ranks_b = rank_ii_joint(Xb, yb)
        t, _ = kendalltau(gt_rank, ranks_b)
        draws[draw] = t
        if (draw + 1) % 20 == 0:
            print(f"  {draw+1}/{B} draws done")

    cur_lo, cur_hi = np.percentile(draws, [2.5, 97.5])
    cur_robust = not (cur_lo <= 0 <= cur_hi)
    d = np.sqrt(b) * (draws - theta_n)
    c_lo, c_hi = np.percentile(d, [2.5, 97.5])
    corr_lo = theta_n - c_hi / np.sqrt(n)
    corr_hi = theta_n - c_lo / np.sqrt(n)
    corr_robust = not (corr_lo <= 0 <= corr_hi)

    print("\n" + "=" * 78)
    print(f"current   CI: [{cur_lo:+.4f},{cur_hi:+.4f}]  {'ROBUST' if cur_robust else 'fragile'}")
    print(f"corrected CI: [{corr_lo:+.4f},{corr_hi:+.4f}]  {'ROBUST' if corr_robust else 'fragile'}")
    print(f"{'SAME' if cur_robust==corr_robust else '*** DIFFERENT ***'}")

    pd.DataFrame([dict(method='II_joint', p=27, theta_hat_n=theta_n,
                        current_ci_lo=cur_lo, current_ci_hi=cur_hi, current_robust=cur_robust,
                        corrected_ci_lo=corr_lo, corrected_ci_hi=corr_hi, corrected_robust=corr_robust,
                        classification_changed=(cur_robust != corr_robust))]
                 ).to_csv('subsampling_correction_check_synthetic_results.csv', index=False)
    print("\nSaved: subsampling_correction_check_synthetic_results.csv")

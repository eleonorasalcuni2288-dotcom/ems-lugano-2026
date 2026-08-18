"""
MI Diagnostics — n_neighbors sensitivity + cross-seed stochasticity
================================================================================
Closes the same kind of gap as dii_diagnostics.py, but for MI: the Kraskov-
Stogbauer-Grassberger estimator behind mutual_info_regression has its own
hyperparameter, n_neighbors (sklearn default=3, never varied elsewhere in
this project — MI_perfeat throughout uses the plain default). MI survives
FRED-MD's Benjamini-Hochberg correction at K=10, so it is not just a
baseline here; it deserves the same robustness scrutiny as DII+L1, RF, and
MINE.

Two checks, both at p=27, using the identical core dataset/ground truth as
simulation_study_v6_highdim.py:

  1. n_neighbors sensitivity: does the ranking (and specifically, XOR
     synergy-pair detection -- which per-feature MI is already known NOT to
     detect, rank ~21-26 regardless) change with n_neighbors in [3,5,7,10]?
  2. Cross-seed stochasticity: mutual_info_regression's random_state seeds
     the noise added internally to break ties in the KSG estimator: does
     varying it (n_neighbors=3 fixed) change the ranking at all?

Does not modify any existing file.
"""
import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

from simulation_study_v6_highdim import generate_core_dataset

N_NEIGHBORS_LEVELS = [3, 5, 7, 10]
SEED_LEVELS = [42, 43, 44]
BASE_SEED = 42
BASE_N_NEIGHBORS = 3   # sklearn default, used throughout the project

X_core, Y, feature_names, feature_groups, gt_binary, gt_rank = generate_core_dataset(seed=BASE_SEED)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_core)
Y_scaled = scaler.fit_transform(Y.reshape(-1, 1)).ravel()
syn_idx = [np.where(feature_names == f)[0][0] for f in ['x_syn_1', 'x_syn_2']]

if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70)
    print("MI DIAGNOSTICS — n_neighbors sensitivity + cross-seed stochasticity")
    print("=" * 70)

    print(f"\n--- [1/2] n_neighbors sensitivity (seed fixed at {BASE_SEED}) ---")
    nn_rows, nn_scores_list = [], []
    for nn in N_NEIGHBORS_LEVELS:
        scores = mutual_info_regression(X_scaled, Y_scaled, n_neighbors=nn, random_state=BASE_SEED)
        ranks = rankdata(-scores).astype(int)
        tau, _ = kendalltau(gt_rank, ranks)
        r1, r2 = int(ranks[syn_idx[0]]), int(ranks[syn_idx[1]])
        nn_scores_list.append(scores)
        nn_rows.append(dict(n_neighbors=nn, tau=tau, xor_rank_1=r1, xor_rank_2=r2))
        print(f"  n_neighbors={nn:2d}  tau={tau:.3f}  XOR rank=({r1},{r2})")

    base_scores = nn_scores_list[N_NEIGHBORS_LEVELS.index(BASE_N_NEIGHBORS)]
    base_ranks = rankdata(-base_scores).astype(int)
    print(f"\n  Ranking agreement (tau) vs the n_neighbors={BASE_N_NEIGHBORS} "
          f"ranking used throughout the project:")
    for nn, scores in zip(N_NEIGHBORS_LEVELS, nn_scores_list):
        ranks = rankdata(-scores).astype(int)
        agree_tau, _ = kendalltau(base_ranks, ranks)
        print(f"    n_neighbors={nn:2d} vs n_neighbors={BASE_N_NEIGHBORS}: tau={agree_tau:.3f}")

    print(f"\n--- [2/2] Cross-seed stochasticity (n_neighbors={BASE_N_NEIGHBORS} fixed, "
          f"{len(SEED_LEVELS)} seeds) ---")
    seed_rows = []
    for seed in SEED_LEVELS:
        scores = mutual_info_regression(X_scaled, Y_scaled, n_neighbors=BASE_N_NEIGHBORS, random_state=seed)
        ranks = rankdata(-scores).astype(int)
        tau, _ = kendalltau(gt_rank, ranks)
        r1, r2 = int(ranks[syn_idx[0]]), int(ranks[syn_idx[1]])
        seed_rows.append(dict(seed=seed, tau=tau, xor_rank_1=r1, xor_rank_2=r2))
        print(f"  seed={seed}  tau={tau:.3f}  XOR rank=({r1},{r2})")

    tau_values = [r['tau'] for r in seed_rows]
    print(f"\n  tau across seeds: mean={np.mean(tau_values):.3f}  "
          f"std={np.std(tau_values):.3f}  range=[{min(tau_values):.3f},{max(tau_values):.3f}]")

    pd.DataFrame(nn_rows).to_csv('mi_diagnostics_n_neighbors_sensitivity.csv', index=False)
    pd.DataFrame(seed_rows).to_csv('mi_diagnostics_seed_stochasticity.csv', index=False)
    print("\nSaved: mi_diagnostics_n_neighbors_sensitivity.csv, mi_diagnostics_seed_stochasticity.csv")
    print(f"\nTotal time: {(time.time()-t_start):.1f}s")

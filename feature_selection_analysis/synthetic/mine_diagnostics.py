"""
MINE Diagnostics — epochs sensitivity
================================================================================
MINE already has per-epoch convergence tracking and a 3-seed stochasticity
check built into mine_synthetic_highdim.py itself (run_diagnostics()), but
neither of those tests whether MORE training changes the actual ranking /
XOR synergy-pair detection -- the same kind of question mi_diagnostics.py
asks of n_neighbors and dii_diagnostics.py asks of l1_strength. Given the
convergence check already flags MINE as not clearly converged at 300 epochs
(the value used throughout the project), this closes that gap directly:
does training longer change the conclusion that MINE fails to detect the
XOR synergy pair?

One check, at p=27 (same representative level as the other diagnostics),
using the identical core dataset/ground truth as simulation_study_v6_highdim.py.
Epochs varied in [100, 300, 600, 1000], seed fixed at 42 (project standard).
Full leave-one-out ranking (28 trainings per epoch level) at each level.

Does not modify any existing file.
"""
import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata
from sklearn.preprocessing import StandardScaler

from simulation_study_v6_highdim import generate_core_dataset
from mine_synthetic_highdim import train_mine

EPOCHS_LEVELS = [100, 300, 600, 1000]
BASE_SEED = 42
BASE_EPOCHS = 300   # project standard, used throughout

X_core, Y, feature_names, feature_groups, gt_binary, gt_rank = generate_core_dataset(seed=BASE_SEED)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_core)
Y_scaled = scaler.fit_transform(Y.reshape(-1, 1)).ravel()
n_features = X_scaled.shape[1]
syn_idx = [np.where(feature_names == f)[0][0] for f in ['x_syn_1', 'x_syn_2']]


def loo_ranking_at_epochs(X, Y, n_feat, epochs):
    mi_full, _ = train_mine(X, Y, BASE_SEED, epochs=epochs)
    importance = np.zeros(n_feat)
    for i in range(n_feat):
        X_reduced = np.delete(X, i, axis=1)
        mi_reduced, _ = train_mine(X_reduced, Y, BASE_SEED, epochs=epochs)
        importance[i] = mi_full - mi_reduced
    return rankdata(-importance).astype(int)


if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70)
    print("MINE DIAGNOSTICS — epochs sensitivity")
    print("=" * 70)

    print(f"\n--- Epochs sensitivity (seed fixed at {BASE_SEED}, p=27) ---")
    ep_rows, ep_ranks_list = [], []
    for ep in EPOCHS_LEVELS:
        t0 = time.time()
        ranks = loo_ranking_at_epochs(X_scaled, Y_scaled, n_features, ep)
        tau, _ = kendalltau(gt_rank, ranks)
        r1, r2 = int(ranks[syn_idx[0]]), int(ranks[syn_idx[1]])
        ep_ranks_list.append(ranks)
        ep_rows.append(dict(epochs=ep, tau=tau, xor_rank_1=r1, xor_rank_2=r2))
        print(f"  epochs={ep:4d}  tau={tau:.3f}  XOR rank=({r1},{r2})  [{time.time()-t0:.1f}s]")

    base_ranks = ep_ranks_list[EPOCHS_LEVELS.index(BASE_EPOCHS)]
    print(f"\n  Ranking agreement (tau) vs the epochs={BASE_EPOCHS} ranking "
          f"used throughout the project:")
    for ep, ranks in zip(EPOCHS_LEVELS, ep_ranks_list):
        agree_tau, _ = kendalltau(base_ranks, ranks)
        print(f"    epochs={ep:4d} vs epochs={BASE_EPOCHS}: tau={agree_tau:.3f}")

    pd.DataFrame(ep_rows).to_csv('mine_diagnostics_epochs_sensitivity.csv', index=False)
    print("\nSaved: mine_diagnostics_epochs_sensitivity.csv")
    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

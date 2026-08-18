"""
DII Diagnostics — L1-strength sensitivity + cross-seed stochasticity
================================================================================
DII+L1 is the headline method (robust downstream advantage in all three real
datasets), but unlike RF (n_estimators stability, rf_diagnostics.py) and
MINE (per-epoch convergence + cross-seed check, in
mine_synthetic_highdim.py), it had no dedicated robustness check of its own
hyperparameters before this. Closes that gap directly on the method being
championed, not just the alternatives.

Two checks, both at p=27 (same representative level as the other
diagnostics), using the identical core dataset/ground truth as
simulation_study_v6_highdim.py:

  1. L1-strength sensitivity: does the ranking (and specifically, XOR
     synergy-pair detection) hold up if l1_strength is varied around the
     value used throughout the project (0.10)? Checked at
     l1 in [0.05, 0.10, 0.15, 0.20].
  2. Cross-seed stochasticity: does DII give a materially different ranking
     across random seeds at the SAME l1_strength (0.10)? DiffImbalance's
     seed controls both weight initialization and the (single, full-batch)
     training trajectory, so this is the same kind of check as MINE's
     3-seed stochasticity test.

Does not modify any existing file.
"""
import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
from scipy.stats import kendalltau
from sklearn.preprocessing import StandardScaler

from simulation_study_v6_highdim import generate_core_dataset
from dadapy.diff_imbalance import DiffImbalance

L1_LEVELS = [0.05, 0.10, 0.15, 0.20]
SEED_LEVELS = [42, 43, 44]
BASE_SEED = 42

X_core, Y, feature_names, feature_groups, gt_binary, gt_rank = generate_core_dataset(seed=BASE_SEED)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_core)
Y_scaled = scaler.fit_transform(Y.reshape(-1, 1)).ravel()
N = X_scaled.shape[0]
syn_idx = [np.where(feature_names == f)[0][0] for f in ['x_syn_1', 'x_syn_2']]


def run_dii_with_seed(X, Y, l1, seed, label):
    k_init = max(5, int(0.025 * N))
    k_final = max(1, int(0.010 * N))
    m = DiffImbalance(
        data_A=X.astype(np.float64), data_B=Y.reshape(-1, 1).astype(np.float64),
        num_epochs=300, batches_per_epoch=1, seed=seed,
        l1_strength=l1, point_adapt_lambda=True,
        k_init=k_init, k_final=k_final, lambda_factor=0.1,
        optimizer_name='adam', learning_rate=1e-2, learning_rate_decay='cos',
    )
    m.train(bar_label=label)
    w = np.array(m.params_final)
    from scipy.stats import rankdata
    return w, rankdata(-w).astype(int)


if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70)
    print("DII DIAGNOSTICS — L1-strength sensitivity + cross-seed stochasticity")
    print("=" * 70)

    print("\n--- [1/2] L1-strength sensitivity (seed fixed at "
          f"{BASE_SEED}) ---")
    l1_rows = []
    all_weights = []
    for l1 in L1_LEVELS:
        w, ranks = run_dii_with_seed(X_scaled, Y_scaled, l1, BASE_SEED, f"DII l1={l1}")
        tau, _ = kendalltau(gt_rank, ranks)
        r1, r2 = int(ranks[syn_idx[0]]), int(ranks[syn_idx[1]])
        all_weights.append(w)
        l1_rows.append(dict(l1_strength=l1, tau=tau, xor_rank_1=r1, xor_rank_2=r2))
        print(f"  l1={l1:.2f}  tau={tau:.3f}  XOR rank=({r1},{r2})")

    # pairwise tau between the rankings produced at each l1 level, vs the
    # l1=0.10 (project-standard) ranking specifically
    base_ranks = None
    for l1, w in zip(L1_LEVELS, all_weights):
        if l1 == 0.10:
            from scipy.stats import rankdata
            base_ranks = rankdata(-w).astype(int)
    print("\n  Ranking agreement (tau) vs the l1=0.10 ranking used throughout the project:")
    from scipy.stats import rankdata
    for l1, w in zip(L1_LEVELS, all_weights):
        ranks = rankdata(-w).astype(int)
        agree_tau, _ = kendalltau(base_ranks, ranks)
        print(f"    l1={l1:.2f} vs l1=0.10: tau={agree_tau:.3f}")

    print(f"\n--- [2/2] Cross-seed stochasticity (l1=0.10 fixed, "
          f"{len(SEED_LEVELS)} seeds) ---")
    seed_rows = []
    seed_weight_list = []
    for seed in SEED_LEVELS:
        w, ranks = run_dii_with_seed(X_scaled, Y_scaled, 0.10, seed, f"DII seed={seed}")
        tau, _ = kendalltau(gt_rank, ranks)
        r1, r2 = int(ranks[syn_idx[0]]), int(ranks[syn_idx[1]])
        seed_weight_list.append(w)
        seed_rows.append(dict(seed=seed, tau=tau, xor_rank_1=r1, xor_rank_2=r2))
        print(f"  seed={seed}  tau={tau:.3f}  XOR rank=({r1},{r2})")

    tau_values = [r['tau'] for r in seed_rows]
    print(f"\n  tau across seeds: mean={np.mean(tau_values):.3f}  "
          f"std={np.std(tau_values):.3f}  range=[{min(tau_values):.3f},{max(tau_values):.3f}]")

    import pandas as pd
    pd.DataFrame(l1_rows).to_csv('dii_diagnostics_l1_sensitivity.csv', index=False)
    pd.DataFrame(seed_rows).to_csv('dii_diagnostics_seed_stochasticity.csv', index=False)
    print("\nSaved: dii_diagnostics_l1_sensitivity.csv, dii_diagnostics_seed_stochasticity.csv")
    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

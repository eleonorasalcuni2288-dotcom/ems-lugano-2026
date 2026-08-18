"""
Downstream Validation — Sanity Check on Synthetic Data (p=27)
================================================================
Applies the k-NN LOO + randomized-baseline proxy (downstream_validation.py)
to the SAME synthetic dataset used in simulation_study_v6_highdim.py,
where the ground truth IS known (gt_rank, tau against it).

Goal: verify that ranking methods "by downstream predictive accuracy"
(the proxy we can compute on real data without ground truth) agrees with
ranking them "by tau against known ground truth" (only computable here).
If they agree here, we have direct evidence the proxy is trustworthy to
apply later on MI-complications / FRED-MD, where no ground truth exists.

Run this from inside your project folder (same directory as
simulation_study_v6_highdim.py), with the venv activated:
    python3 validate_downstream_synthetic.py

Requires: simulation_study_v6_highdim.py and downstream_validation.py in
the same folder (or on PYTHONPATH).
"""
import numpy as np
from scipy.stats import kendalltau

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'common'))     # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # simulation_study_v6_highdim.py lives here
from simulation_study_v6_highdim import (
    generate_core_dataset, compute_ii_pf, make_ii_joint, run_dii,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import rankdata

from downstream_validation import evaluate_method_vs_baseline

SEED = 42
K_VALUES = [3, 5, 10, 16]

print("="*70)
print("DOWNSTREAM VALIDATION — SANITY CHECK ON SYNTHETIC DATA (p=27)")
print("="*70)

# ---- 1. Rebuild the exact p=27 synthetic dataset (same as v5/v6) --------
X, Y, feature_names, feature_groups, gt_binary, gt_rank = generate_core_dataset()
n_features = X.shape[1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = (Y - Y.mean()) / Y.std()

# Binarize target for k-NN classification (median split) — Y itself is
# continuous in v5/v6; downstream validation here needs a classification
# target, consistent with how it will be used on MI-complications
# (binary "any complication") and FRED-MD (binarized return direction).
Y_class = (Y_scaled > np.median(Y_scaled)).astype(int)
print(f"\nBinarized target via median split: "
      f"{Y_class.sum()}/{len(Y_class)} positive class")

# ---- 2. Recompute rankings for the 4 core methods ------------------------
print("\n[Recomputing method rankings...]")

mi_scores = mutual_info_regression(X_scaled, Y_scaled, random_state=SEED)
mi_ranks = rankdata(-mi_scores).astype(int)

_dy = np.abs(Y_scaled.reshape(-1, 1) - Y_scaled.reshape(1, -1))
np.fill_diagonal(_dy, np.inf)
ry_global = np.argsort(np.argsort(_dy, axis=1), axis=1)
ii_scores = np.array([compute_ii_pf(X_scaled[:, i], ry_global)
                       for i in range(n_features)])
ii_ranks = rankdata(ii_scores).astype(int)

ii_joint_fn = make_ii_joint(Y_scaled)
ii_full = ii_joint_fn(X_scaled)
ii_loo = np.array([ii_joint_fn(np.delete(X_scaled, i, axis=1))
                    for i in range(n_features)])
ii_joint_ranks = rankdata(-(ii_loo - ii_full)).astype(int)

dii_l1_w, dii_l1_ranks, _ = run_dii(X_scaled, Y_scaled, 0.10, "DII+L1 (validation)")

methods = {
    'MI_perfeat': mi_ranks,
    'II_perfeat': ii_ranks,
    'II_joint':   ii_joint_ranks,
    'DII_L1':     dii_l1_ranks,
}

# ---- 3. tau against ground truth (already known, for comparison) --------
print("\n[Tau against known ground truth]")
tau_by_method = {}
for name, ranks in methods.items():
    t, _ = kendalltau(gt_rank, ranks)
    tau_by_method[name] = t
    print(f"  {name:<12} tau={t:.3f}")

# ---- 4. Downstream validation (the proxy) --------------------------------
print("\n[Downstream validation: k-NN LOO + randomized baseline]")
all_results = {}
for name, ranks in methods.items():
    print(f"\n --- {name} ---")
    all_results[name] = evaluate_method_vs_baseline(
        X_scaled, Y_class, ranks, K_VALUES,
        n_random=200, k_neighbors=5, seed=SEED, method_name=name)

# ---- 5. Concordance check -------------------------------------------------
print("\n" + "="*70)
print("CONCORDANCE CHECK: ranking by tau vs ranking by downstream accuracy")
print("="*70)

# Use K=10 as representative summary accuracy per method
acc_at_10 = {name: next(r['method_acc'] for r in res if r['K'] == 10)
             for name, res in all_results.items()}

order_by_tau = sorted(tau_by_method, key=tau_by_method.get, reverse=True)
order_by_acc = sorted(acc_at_10, key=acc_at_10.get, reverse=True)

print(f"\n  Ranking by tau (ground truth):        {order_by_tau}")
print(f"  Ranking by downstream acc (K=10):     {order_by_acc}")
print(f"\n  {'Method':<12} {'tau':>8} {'acc@K10':>10} {'p-value@K10':>12}")
for name in methods:
    p10 = next(r['p_value'] for r in all_results[name] if r['K'] == 10)
    print(f"  {name:<12} {tau_by_method[name]:>8.3f} {acc_at_10[name]:>10.3f} {p10:>12.3f}")

if order_by_tau == order_by_acc:
    print("\n  MATCH: rankings agree completely — strong evidence the "
          "downstream proxy is trustworthy for use without ground truth.")
else:
    print("\n  PARTIAL/NO MATCH: rankings differ — report this honestly; "
          "the proxy may need caveats when applied to real data below.")

print("\n" + "="*70)
print("DONE")
print("="*70)
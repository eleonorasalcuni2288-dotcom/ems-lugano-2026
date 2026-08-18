"""
LASSO — Classical Sparse Regularized Regression Baseline
======================================================================
Closes a real gap flagged in review: RF covers "general-purpose ML" but
not "classical sparse regularized regression", which is the most direct
conceptual competitor to DII+L1 (both rank features via a sparsity-
inducing penalty; DII does so on a differentiable rank-based imbalance,
LASSO on a linear model's squared-error loss). Without this baseline, the
comparison table implicitly asks "does DII beat a black-box ensemble?"
but never "does DII beat the standard linear sparse baseline it is
conceptually closest to?".

Same treatment and scope as rf_synthetic_highdim.py (RF's own precedent):
full point-estimate + bootstrap CI on the synthetic benchmark, p=27/50/105
only — not extended to the real datasets, matching how RF itself was
scoped (full synthetic treatment, a lighter check on MI-complications
only, absent from FRED-MD/trading).

HYPERPARAMETER: alpha (L1 penalty strength) is selected via LassoCV's
built-in 5-fold cross-validation on the TRAINING data itself at every
p level and every bootstrap draw, not fixed to one hand-picked value —
this sidesteps the exact "which hyperparameter did you choose and why"
question that motivated dii_diagnostics.py / mi_diagnostics.py, since
alpha is not a free choice here but re-derived from the data each time.

RANKING: by |coefficient| magnitude (standard LASSO feature-importance
convention) on the standardized design matrix.

STOCHASTICITY: not checked separately here (unlike RF/DII/MI) because
LassoCV's default k-fold splitting (KFold, shuffle=False) is deterministic
given the data — there is no equivalent "random seed" source of variation
to probe, unlike RF's bootstrap-sampled trees or DII/MINE's initialization.

OUTPUT (same schema as rf_synthetic_highdim.py, for direct union):
  lasso_synthetic_highdim_point_estimates.csv
  lasso_synthetic_highdim_results.csv
  lasso_synthetic_highdim_summary.txt

CONSTRAINTS: does not modify any existing file.
"""

import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'common'))     # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # simulation_study_v6_highdim.py lives here
from simulation_study_v6_highdim import generate_core_dataset
from bootstrap_ci_synthetic_highdimensional import build_padded_dataset, SEED, N
from infonce_synthetic import K_VALUES
from downstream_validation import evaluate_method_vs_baseline

P_LEVELS = [27, 50, 105]
SUBSAMPLE_FRAC = 0.80
CI_LEVEL = 95
B_BOOTSTRAP = 20   # matching RF/MINE, for methodological comparability


def lasso_ranking(X_scaled, Y_scaled, seed=SEED):
    """LassoCV auto-selects alpha via 5-fold CV; ranks by |coefficient|."""
    lasso = LassoCV(cv=5, random_state=seed, n_jobs=-1, max_iter=5000)
    lasso.fit(X_scaled, Y_scaled)
    importance = np.abs(lasso.coef_)
    ranks = rankdata(-importance).astype(int)
    return importance, ranks, lasso.alpha_


if __name__ == "__main__":
    t_start_all = time.time()
    print("=" * 70)
    print("LASSO — CLASSICAL SPARSE REGULARIZED REGRESSION BASELINE")
    print(f"alpha: auto (LassoCV, 5-fold)  B_bootstrap={B_BOOTSTRAP}")
    print("=" * 70)

    X_core, Y, core_names, core_groups, core_binary, core_rank = generate_core_dataset()
    n_core = X_core.shape[1]
    max_extra = max(P_LEVELS) - n_core
    rng_pad = np.random.default_rng(SEED + 1)
    noise_pool = rng_pad.normal(0, 1, size=(N, max_extra))

    Y_scaled = (Y - Y.mean()) / Y.std()
    Y_class = (Y_scaled > np.median(Y_scaled)).astype(int)

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
        importance, ranks, alpha_used = lasso_ranking(X_scaled, Y_scaled)
        t_elapsed = time.time() - t0
        tau, _ = kendalltau(gt_rank, ranks)
        syn_r1, syn_r2 = int(ranks[syn_idx[0]]), int(ranks[syn_idx[1]])
        n_nonzero = int((importance > 1e-10).sum())
        print(f"  [point p={p}] tau={tau:.3f}  XOR=({syn_r1},{syn_r2})  "
              f"alpha={alpha_used:.4f}  nonzero={n_nonzero}/{p}  [{t_elapsed:.2f}s]")

        ds_results = evaluate_method_vs_baseline(
            X_scaled, Y_class, ranks, K_VALUES,
            n_random=100, k_neighbors=5, seed=SEED, method_name=f"LASSO_p{p}")
        for r in ds_results:
            r.update(dict(p=p, tau=tau, syn_rank_1=syn_r1, syn_rank_2=syn_r2,
                           alpha=alpha_used, n_nonzero=n_nonzero))
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
            _, ranks_b, _ = lasso_ranking(Xb, yb, seed=SEED)
            t, _ = kendalltau(gt_rank, ranks_b)
            taus_boot[b] = t
        lo, hi = np.percentile(taus_boot, [(100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2])
        bootstrap_rows.append(dict(
            p=p, method='LASSO', tau_mean=float(taus_boot.mean()),
            ci_lo=float(lo), ci_hi=float(hi), tau_std=float(taus_boot.std()),
            B=B_BOOTSTRAP))
        print(f"  Bootstrap tau = {taus_boot.mean():.3f} [{lo:.3f}, {hi:.3f}]  "
              f"(B={B_BOOTSTRAP}, {time.time()-t0:.1f}s)")

    # ---- Save outputs -------------------------------------------------------
    point_df = pd.DataFrame(point_rows)
    point_df.to_csv('lasso_synthetic_highdim_point_estimates.csv', index=False)
    print("\nSaved: lasso_synthetic_highdim_point_estimates.csv")

    boot_df = pd.DataFrame(bootstrap_rows)
    boot_df.to_csv('lasso_synthetic_highdim_results.csv', index=False)
    print("Saved: lasso_synthetic_highdim_results.csv "
          "(same schema as rf_synthetic_highdim_results.csv)")

    total_min = (time.time() - t_start_all) / 60
    with open('lasso_synthetic_highdim_summary.txt', 'w') as f:
        f.write("LASSO — CLASSICAL SPARSE REGULARIZED REGRESSION BASELINE\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total runtime: {total_min:.1f} min\n")
        f.write(f"alpha: auto (LassoCV, 5-fold)  B_bootstrap={B_BOOTSTRAP}\n\n")
        f.write("--- POINT ESTIMATES (tau, XOR pair rank, selected alpha) ---\n")
        for _, r in point_df.drop_duplicates('p').iterrows():
            f.write(f"  p={int(r.p):<4} tau={r.tau:.3f}  "
                    f"XOR rank=({int(r.syn_rank_1)},{int(r.syn_rank_2)})  "
                    f"alpha={r.alpha:.4f}  nonzero={int(r.n_nonzero)}/{int(r.p)}\n")
        f.write("\n--- BOOTSTRAP CI (tau) ---\n")
        for _, r in boot_df.iterrows():
            f.write(f"  p={int(r.p):<4} tau_mean={r.tau_mean:.3f}  "
                    f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]  (B={int(r.B)})\n")

    print("Saved: lasso_synthetic_highdim_summary.txt")
    print(f"\nTotal time: {total_min:.1f} min")
    print("\n" + "=" * 70)
    print("LASSO ANALYSIS COMPLETE")
    print("=" * 70)

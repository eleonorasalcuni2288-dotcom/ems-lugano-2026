"""
InfoNCE — Additional MI Estimator for Comparison (Synthetic Dataset, p=27)
============================================================================
Implements InfoNCE (van den Oord et al. 2018) as a neural mutual-information
lower-bound estimator, to compare against the classical estimators already
in this project (MI via sklearn/Kraskov, II via rank-neighbourhood, DII via
DADApy's DiffImbalance), on the SAME synthetic p=27 dataset used everywhere
else (imported from simulation_study_v6_highdim.py, not duplicated).

ARCHITECTURE (explicit, not left to defaults):
  g(x): MLP  n_features -> 64 -> 16   (feature encoder)
  h(y): MLP  1 -> 32 -> 16            (target encoder — Y is a SCALAR in
                                        this project, not a vector; this is
                                        NOT the typical InfoNCE tutorial
                                        setup and is handled explicitly here)
  score(x,y) = g(x) . h(y)            (dot product in the 16-dim embedding)
  loss = in-batch softmax cross-entropy (categorical, diagonal = positive
         pairs, off-diagonal = implicit in-batch negatives)
  MI lower bound estimate (the standard InfoNCE bound):
         I_hat = log(batch_size) - loss

LIBRARY: JAX + optax (already a dependency via dadapy/DiffImbalance,
kept consistent with the rest of the project rather than introducing
PyTorch).

TRAINING (explicit):
  seed = 42 (project-wide convention)
  optimizer = Adam, learning_rate = 1e-3
  epochs = 300 (matches DII's num_epochs elsewhere in this project)
  batch_size tested at BOTH 128 and 512, reported separately — InfoNCE's
  MI estimate is mathematically capped at log(batch_size) (4.85 nats at
  128, 6.24 nats at 512), a well-known bias of the estimator; comparing
  both makes this explicit instead of hiding it behind one arbitrary
  choice.
  Reported MI estimate per run = mean of the in-batch MI lower bound
  over all batches of the FINAL epoch (a convergence-window estimate,
  not a running average over all of training — documented here since it
  was not specified upstream).
  Batches are re-shuffled every epoch; the last incomplete batch (N=2000
  is not evenly divisible by 128 or 512) is DROPPED, not padded, to keep
  every batch square for the in-batch negative structure and JIT-stable
  shapes — documented since it was not specified upstream.

FEATURE RANKING (explicit, matches this project's existing LOO pattern
for MI_joint / II_joint):
  1. Train the full model (all 27 features) -> MI_hat_full
  2. For each of the 27 features, retrain from scratch on the remaining
     26 (SAME seed, so only the removed feature differs) -> MI_hat_{-i}
  3. importance_i = MI_hat_full - MI_hat_{-i}  (positive = feature helps)
  4. rank = rankdata(-importance)  (rank 1 = most important), matching
     the sign convention used for MI_joint/II_joint elsewhere.
  This means 28 total trainings per batch size (1 full + 27 LOO), 56
  total across both batch sizes.

OUTPUT (for direct comparability with existing results):
  - Kendall's tau vs known ground truth (gt_rank from generate_core_dataset),
    compared against the BOOTSTRAP-MEAN tau values already obtained for
    the other 4 methods (bootstrap_ci_synthetic_results.csv):
        MI_perfeat=0.269  II_perfeat=0.270  II_joint=0.302  DII_L1=0.370
    NOTE (documented, not hidden): this InfoNCE number is a single
    full-sample POINT ESTIMATE, not a bootstrap mean like the reference
    values — not perfectly apples-to-apples; the 4h time-box does not
    allow bootstrapping InfoNCE too (that would be 56x more trainings).
  - Rank of the XOR synergy pair (x_syn_1, x_syn_2) — the poster's central
    finding (DII and II_joint both rank it 1st/2nd at p=27).
  - Downstream validation via the EXISTING downstream_validation.py module
    (evaluate_method_vs_baseline), not a separately-computed accuracy, for
    direct comparability with all other methods. Y is binarised via median
    split (same convention as validate_downstream_synthetic.py) since that
    module expects a classification target.

CONSTRAINTS: time-boxed at 4 hours; does not modify any existing file;
writes only new outputs (infonce_synthetic_*.csv/.txt).
"""

import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax
from scipy.stats import rankdata, kendalltau
from sklearn.preprocessing import StandardScaler

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'common'))     # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # simulation_study_v6_highdim.py lives here
from simulation_study_v6_highdim import generate_core_dataset
from downstream_validation import evaluate_method_vs_baseline

SEED = 42
EPOCHS = 300
LEARNING_RATE = 1e-3
BATCH_SIZES = [128, 512]
HIDDEN_G = 64
EMBED_DIM = 16
HIDDEN_H = 32
K_VALUES = [3, 5, 10, 16]

# Reference values already obtained for the other 4 methods (bootstrap
# MEAN tau, from bootstrap_ci_synthetic_results.csv) — printed alongside
# InfoNCE's point-estimate tau for direct visual comparison.
REFERENCE_TAU = {
    'MI_perfeat': 0.269, 'II_perfeat': 0.270, 'II_joint': 0.302, 'DII_L1': 0.370,
}


# =============================================================================
# 1. MLP + InfoNCE (plain JAX, no flax — small enough not to need it)
# =============================================================================

def init_mlp(key, sizes):
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for k, n_in, n_out in zip(keys, sizes[:-1], sizes[1:]):
        wk, _ = jax.random.split(k)
        w = jax.random.normal(wk, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        b = jnp.zeros(n_out)
        params.append((w, b))
    return params


def mlp_apply(params, x):
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x


def init_infonce_params(key, n_features):
    kg, kh = jax.random.split(key)
    return {
        'g': init_mlp(kg, [n_features, HIDDEN_G, EMBED_DIM]),
        'h': init_mlp(kh, [1, HIDDEN_H, EMBED_DIM]),
    }


def infonce_loss(params, x_batch, y_batch):
    gx = mlp_apply(params['g'], x_batch)   # (B, embed)
    hy = mlp_apply(params['h'], y_batch)   # (B, embed)
    scores = gx @ hy.T                     # (B, B)
    log_probs = jax.nn.log_softmax(scores, axis=1)
    B = x_batch.shape[0]
    loss = -jnp.mean(jnp.diag(log_probs))
    mi_lb = jnp.log(B) - loss
    return loss, mi_lb


def make_train_step(optimizer):
    @jax.jit
    def train_step(params, opt_state, x_batch, y_batch):
        (loss, mi_lb), grads = jax.value_and_grad(infonce_loss, has_aux=True)(
            params, x_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, mi_lb
    return train_step


def train_infonce(X, Y, seed, batch_size, epochs=EPOCHS, lr=LEARNING_RATE):
    """Returns (final_mi_estimate, params). final_mi_estimate = mean
    in-batch MI lower bound over the batches of the LAST epoch."""
    n, n_features = X.shape
    key = jax.random.PRNGKey(seed)
    params = init_infonce_params(key, n_features)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    train_step = make_train_step(optimizer)

    X_j = jnp.asarray(X, dtype=jnp.float32)
    Y_j = jnp.asarray(Y.reshape(-1, 1), dtype=jnp.float32)
    n_batches = n // batch_size   # last incomplete batch dropped (see docstring)
    rng = np.random.default_rng(seed)

    last_epoch_mi = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        last_epoch_mi = []
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            params, opt_state, loss, mi_lb = train_step(
                params, opt_state, X_j[idx], Y_j[idx])
            last_epoch_mi.append(float(mi_lb))

    return float(np.mean(last_epoch_mi)), params


# =============================================================================
# 2. LOO ranking for one batch size
# =============================================================================

def run_loo_for_batch_size(X_scaled, Y_scaled, batch_size, n_features, log_prefix):
    t0 = time.time()
    mi_full, _ = train_infonce(X_scaled, Y_scaled, SEED, batch_size)
    print(f"  [{log_prefix}] MI_full = {mi_full:.4f}  "
          f"(cap = log({batch_size}) = {np.log(batch_size):.4f})  "
          f"[{time.time()-t0:.1f}s]")

    importance = np.zeros(n_features)
    for i in range(n_features):
        t0 = time.time()
        X_reduced = np.delete(X_scaled, i, axis=1)
        mi_reduced, _ = train_infonce(X_reduced, Y_scaled, SEED, batch_size)
        importance[i] = mi_full - mi_reduced
        print(f"  [{log_prefix}] LOO {i+1:2d}/{n_features}  "
              f"MI_-i={mi_reduced:.4f}  imp={importance[i]:+.4f}  "
              f"[{time.time()-t0:.1f}s]")

    ranks = rankdata(-importance).astype(int)
    return mi_full, importance, ranks


# =============================================================================
# 3. MAIN
# =============================================================================

if __name__ == "__main__":
    t_start_all = time.time()
    print("=" * 70)
    print("InfoNCE — ADDITIONAL MI ESTIMATOR, SYNTHETIC DATASET (p=27)")
    print(f"epochs={EPOCHS}  lr={LEARNING_RATE}  batch_sizes={BATCH_SIZES}  "
          f"g=27->{HIDDEN_G}->{EMBED_DIM}  h=1->{HIDDEN_H}->{EMBED_DIM}")
    print("=" * 70)

    X, Y, feature_names, feature_groups, gt_binary, gt_rank = generate_core_dataset()
    n_features = X.shape[1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Y_scaled = (Y - Y.mean()) / Y.std()
    Y_class = (Y_scaled > np.median(Y_scaled)).astype(int)  # for downstream validation only

    syn_idx = [np.where(feature_names == f)[0][0] for f in ['x_syn_1', 'x_syn_2']]

    all_rows = []
    rankings_by_bs = {}

    for bs in BATCH_SIZES:
        print(f"\n{'='*70}\n  batch_size = {bs}\n{'='*70}")
        mi_full, importance, ranks = run_loo_for_batch_size(
            X_scaled, Y_scaled, bs, n_features, f"bs={bs}")
        rankings_by_bs[bs] = ranks

        tau, _ = kendalltau(gt_rank, ranks)
        syn_r1, syn_r2 = int(ranks[syn_idx[0]]), int(ranks[syn_idx[1]])

        print(f"\n  [bs={bs}] tau={tau:.3f}  XOR pair rank=({syn_r1},{syn_r2})")
        print(f"  [bs={bs}] Reference (bootstrap-mean tau, other methods): "
              + "  ".join(f"{k}={v:.3f}" for k, v in REFERENCE_TAU.items()))

        print(f"\n  [bs={bs}] Downstream validation (via downstream_validation.py)...")
        ds_results = evaluate_method_vs_baseline(
            X_scaled, Y_class, ranks, K_VALUES,
            n_random=200, k_neighbors=5, seed=SEED,
            method_name=f"InfoNCE_bs{bs}")

        for r in ds_results:
            r['batch_size'] = bs
            r['tau'] = tau
            r['mi_full_estimate'] = mi_full
            r['mi_cap_log_bs'] = float(np.log(bs))
            r['syn_rank_1'] = syn_r1
            r['syn_rank_2'] = syn_r2
            all_rows.append(r)

    # ---- Save rankings CSV ------------------------------------------------
    rankings_df = pd.DataFrame({
        'Feature': feature_names, 'Group': feature_groups,
        'Informative': gt_binary, 'GT_Rank': gt_rank,
    })
    for bs in BATCH_SIZES:
        rankings_df[f'InfoNCE_bs{bs}_Rank'] = rankings_by_bs[bs]
    rankings_df.to_csv('infonce_synthetic_rankings.csv', index=False)
    print("\nSaved: infonce_synthetic_rankings.csv")

    # ---- Save results CSV --------------------------------------------------
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv('infonce_synthetic_results.csv', index=False)
    print("Saved: infonce_synthetic_results.csv")

    # ---- Summary -------------------------------------------------------
    total_min = (time.time() - t_start_all) / 60
    with open('infonce_synthetic_summary.txt', 'w') as f:
        f.write("InfoNCE — ADDITIONAL MI ESTIMATOR, SYNTHETIC DATASET (p=27)\n")
        f.write("=" * 60 + "\n")
        f.write(f"epochs={EPOCHS}  lr={LEARNING_RATE}  batch_sizes={BATCH_SIZES}\n")
        f.write(f"Total runtime: {total_min:.1f} min\n\n")
        for bs in BATCH_SIZES:
            sub = results_df[results_df.batch_size == bs]
            tau = sub.tau.iloc[0]
            r1, r2 = int(sub.syn_rank_1.iloc[0]), int(sub.syn_rank_2.iloc[0])
            mi_full = sub.mi_full_estimate.iloc[0]
            f.write(f"--- batch_size={bs} (MI cap=log({bs})={np.log(bs):.3f}) ---\n")
            f.write(f"  MI_full estimate = {mi_full:.4f}\n")
            f.write(f"  tau (point estimate) = {tau:.3f}\n")
            f.write(f"  XOR synergy pair rank = ({r1}, {r2})\n")
            f.write("  Downstream validation (Top-K accuracy vs random baseline):\n")
            for _, r in sub.iterrows():
                f.write(f"    K={int(r.K):<3} acc={r.method_acc:.3f}  "
                        f"baseline={r.baseline_mean:.3f}  p={r.p_value:.3f}\n")
            f.write("\n")
        f.write("Reference (bootstrap-mean tau, other 4 methods, p=27):\n")
        for k, v in REFERENCE_TAU.items():
            f.write(f"  {k:<12} tau={v:.3f}\n")
        f.write("\nNOTE: InfoNCE tau above is a single full-sample point estimate,\n"
                "not a bootstrap mean like the reference values — not perfectly\n"
                "apples-to-apples; the 4h time-box did not allow bootstrapping\n"
                "InfoNCE too (would require 56x more trainings).\n")
    print("Saved: infonce_synthetic_summary.txt")

    print(f"\nTotal time: {total_min:.1f} min")
    print("\n" + "=" * 70)
    print("INFONCE SYNTHETIC ANALYSIS COMPLETE")
    print("=" * 70)

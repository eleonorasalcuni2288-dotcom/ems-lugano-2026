"""
MINE — Mutual Information Neural Estimation (Belghazi et al. 2018, ICML)
======================================================================
Replaces the InfoNCE comparison (infonce_synthetic.py / infonce_synthetic_
highdim.py — left on disk as reference, not used for the poster) with MINE,
which is peer-reviewed (ICML 2018) rather than an unpublished arXiv preprint.
Covers all three scalability levels (p=27, 50, 105) in one script.

ARCHITECTURE (explicit):
  Single critic T_theta(x, y): MLP on the CONCATENATED input [x ; y]
  (dimension p+1) -> 64 -> 32 -> 1 (scalar). Reuses init_mlp/mlp_apply
  from infonce_synthetic.py (generic MLP utilities, not reimplemented).

MI ESTIMATE — Donsker-Varadhan bound:
  I(X;Y) >= E_[P(X,Y)][T(x,y)] - log( E_[P(X) x P(Y)][exp(T(x,y))] )
  The joint expectation uses the batch's real (x,y) pairs. The product-
  of-marginals expectation is estimated by SHUFFLING y within the same
  batch (breaking the x-y pairing), as specified.

BIAS-CORRECTED GRADIENT (Belghazi et al. 2018, sec. 3.2) — IMPLEMENTED,
not skipped, so this is legitimately called "MINE" rather than a plain
DV-estimator: the naive gradient of log(mean(exp(T_marginal))) w.r.t.
theta is a biased estimator under minibatching, because it is a ratio
of two batch expectations (E[exp(T)*dT/dtheta] / E[exp(T)]), and the
expectation of a ratio is not the ratio of expectations. The paper's
fix: maintain an exponential moving average (EMA) of E_q[exp(T)] across
training steps (a low-variance, de-biased running estimate), and use
THAT as a constant (stop-gradient) normaliser in the gradient path,
while still reporting the standard log-based DV value as the MI
estimate. Implemented in JAX via the standard "value/gradient split"
identity:
    output = surrogate + stop_gradient(true_value - surrogate)
which numerically EQUALS true_value (the correct log-based bound, used
for every reported MI number in this script) while its GRADIENT equals
exactly d(surrogate)/d(theta), where surrogate uses the EMA-normalised
ratio instead of the raw current-batch log. EMA decay = 0.99 and EMA
initialised at 1.0 — both values not specified upstream, chosen here as
standard defaults from the MINE literature and documented rather than
silently picked.

LIBRARY: JAX + optax (consistent with dadapy/DiffImbalance elsewhere in
this project).

TRAINING (explicit): seed=42, Adam, learning_rate=1e-3, epochs=300
(matches DII). batch_size=512 ONLY — MINE's DV bound is not hard-capped
at log(batch_size) the way InfoNCE's softmax bound is (batch size
instead affects variance/stability of the log-mean-exp estimate, not a
ceiling), and a second batch size was optional per the given
instructions; skipped here to keep the LOO-heavy sweep within budget,
stated explicitly rather than silently doing only one.

SCALABILITY (p=27, 50, 105): reuses generate_core_dataset() and
build_padded_dataset() by IMPORTING them (the latter from
bootstrap_ci_synthetic_highdimensional.py) — not reimplemented. The
noise pool uses the same seed (SEED+1=43) and same shape derivation as
every other high-dim analysis in this project; that 2-line generation
lives inside another script's __main__ guard and isn't importable as a
variable, so it is replicated here verbatim (the padding LOGIC itself
IS imported).

FEATURE RANKING: same LOO principle as MI_joint / II_joint / DII / the
InfoNCE work — train the full model, then retrain once per feature with
that feature removed (same seed every time), importance_i = MI_full -
MI_{-i}, rank = rankdata(-importance).

BOOTSTRAP CI: subsampling 80% without replacement (same protocol as
bootstrap_ci_synthetic_highdimensional.py), B draws per p level,
calibrated from measured per-draw timing (see calibration step run
before the full sweep) to fit the 1.5h time-box.

DIAGNOSTICS ADDED ON TOP OF THE ORIGINAL SPEC (per follow-up request):
  1. Convergence check: full per-epoch MI trace recorded for ONE
     representative run (p=27, full model, all features) — plateau
     assessed by comparing the mean of the last 30 epochs against the
     mean of epochs 140-170; flagged explicitly if not converged
     (threshold: <5% relative change — not specified upstream, chosen
     here and documented). Saved as both a CSV trace and a PNG plot.
  2. Intrinsic stochasticity check: the SAME full-data training (p=27,
     all features) repeated 3x with different seeds (42, 43, 44) —
     different network initialisation and batch shuffling, SAME data —
     to isolate the estimator's own noise from data-resampling noise.
     Mean and std of the 3 MI estimates reported explicitly.

OUTPUT:
  mine_synthetic_highdim_results.csv          — bootstrap CI, SAME
    SCHEMA as bootstrap_ci_synthetic_highdim_results.csv.
  mine_synthetic_highdim_point_estimates.csv  — tau, XOR rank,
    downstream validation, per p.
  mine_convergence_check_p27.csv / .png       — per-epoch MI trace.
  mine_synthetic_highdim_summary.txt          — human-readable summary,
    including the convergence and stochasticity diagnostics.

CONSTRAINTS: does not modify any existing file. Time-box 1.5h, with a
live elapsed-time safety check after each p level.
"""

import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from bootstrap_ci_synthetic_highdimensional import build_padded_dataset, SEED, N
from infonce_synthetic import init_mlp, mlp_apply, K_VALUES
from downstream_validation import evaluate_method_vs_baseline

P_LEVELS = [27, 50, 105]
EPOCHS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
EMA_DECAY = 0.99
SUBSAMPLE_FRAC = 0.80
CI_LEVEL = 95
TIME_BOX_MIN = 90
CONVERGENCE_REP_P = 27          # representative p for the convergence/stochasticity checks
CONVERGENCE_WINDOW_LATE = (270, 300)
CONVERGENCE_WINDOW_MID = (140, 170)
CONVERGENCE_REL_THRESHOLD = 0.05

# Calibrated: 1 full bootstrap draw at p=105 measured at 113s (1.88 min).
# B=20 (the top of the requested 15-20 range) -> estimated ~73 min total
# (bootstrap + point estimates + diagnostics + downstream validation),
# comfortably under the 90-min time-box with ~17 min margin.
B_BOOTSTRAP = 20


# =============================================================================
# 1. MINE critic + bias-corrected Donsker-Varadhan loss
# =============================================================================

def init_mine_critic(key, n_features):
    return init_mlp(key, [n_features + 1, 64, 32, 1])


def make_train_step(optimizer):
    @jax.jit
    def train_step(params, opt_state, ema, x_batch, y_batch, y_shuffled_batch):
        def loss_fn(p):
            xy_joint = jnp.concatenate([x_batch, y_batch], axis=1)
            xy_marginal = jnp.concatenate([x_batch, y_shuffled_batch], axis=1)
            t_joint = mlp_apply(p, xy_joint)
            t_marginal = mlp_apply(p, xy_marginal)
            mean_t_joint = jnp.mean(t_joint)

            # Numerical-stability clip: T is an UNCONSTRAINED MLP output and
            # is exponentiated below; training explicitly pushes T upward to
            # maximise the MI estimate, so occasional large T values are a
            # real risk (not just theoretical) and would overflow float32 in
            # exp(), silently producing inf/NaN that would otherwise
            # propagate uncaught into the final CSVs. +-30 is far outside
            # the range T actually takes on this standardised data in
            # practice (not specified upstream — chosen and documented here
            # as a safety clip that should not bind in the normal regime).
            t_marginal_safe = jnp.clip(t_marginal, -30.0, 30.0)
            exp_t_marginal = jnp.exp(t_marginal_safe)
            mean_exp_t_marginal = jnp.mean(exp_t_marginal)

            # true (reported) DV bound value — always the correct log form
            mi_lb_value = mean_t_joint - jnp.log(mean_exp_t_marginal)

            # gradient surrogate: EMA (stop-gradient constant) as normaliser,
            # de-biasing the gradient per Belghazi et al. 2018 sec. 3.2
            grad_surrogate = mean_t_joint - mean_exp_t_marginal / jax.lax.stop_gradient(ema)

            # forward value = mi_lb_value; gradient flows only through grad_surrogate
            mi_lb = grad_surrogate + jax.lax.stop_gradient(mi_lb_value - grad_surrogate)
            loss = -mi_lb
            return loss, (mi_lb_value, mean_exp_t_marginal)

        (loss, (mi_lb_value, mean_exp_batch)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        new_ema = EMA_DECAY * ema + (1.0 - EMA_DECAY) * mean_exp_batch
        return params, opt_state, new_ema, mi_lb_value
    return train_step


# Built ONCE at module level, not inside train_mine(): jax.jit's compiled-
# code cache is tied to the identity of the decorated function object, not
# just to input shapes. Calling make_train_step(optimizer) fresh on every
# train_mine() invocation — as an earlier version of this script did —
# would pay a full retrace+compile on EVERY one of the ~4000+ trainings in
# the full p=27/50/105 x bootstrap sweep, even for shapes already seen
# many times (e.g. every one of the 104 LOO retrainings at p=105 shares
# the same reduced shape but would still recompile individually). Sharing
# one jitted function here lets JAX cache-hit on repeated shapes instead.
_OPTIMIZER = optax.adam(LEARNING_RATE)
_TRAIN_STEP = make_train_step(_OPTIMIZER)


def train_mine(X, Y, seed, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LEARNING_RATE,
                track_history=False):
    """Returns final_mi_estimate = mean bias-corrected DV lower bound over
    the batches of the LAST epoch. If track_history=True, also returns a
    list of per-epoch mean MI estimates (for the convergence check)."""
    if lr != LEARNING_RATE:
        raise ValueError(
            f"train_mine uses a module-level optimizer/train_step cached "
            f"for LEARNING_RATE={LEARNING_RATE} (see _OPTIMIZER/_TRAIN_STEP "
            f"above); a different lr={lr} was passed but would silently "
            f"still use the LEARNING_RATE={LEARNING_RATE} optimizer. Not "
            f"supported by this caching optimisation — every call in this "
            f"script uses the default, so this should never trigger.")
    n, n_features = X.shape
    key = jax.random.PRNGKey(seed)
    params = init_mine_critic(key, n_features)
    opt_state = _OPTIMIZER.init(params)
    train_step = _TRAIN_STEP

    X_j = jnp.asarray(X, dtype=jnp.float32)
    Y_j = jnp.asarray(Y.reshape(-1, 1), dtype=jnp.float32)
    n_batches = n // batch_size
    rng = np.random.default_rng(seed)

    ema = jnp.array(1.0)   # documented default init, see module docstring
    epoch_history = []
    last_epoch_mi = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        last_epoch_mi = []
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb = X_j[idx]
            yb = Y_j[idx]
            shuf = rng.permutation(batch_size)
            yb_shuffled = yb[shuf]
            params, opt_state, ema, mi_lb_value = train_step(
                params, opt_state, ema, xb, yb, yb_shuffled)
            last_epoch_mi.append(float(mi_lb_value))
        if track_history:
            epoch_history.append(float(np.mean(last_epoch_mi)))

    final_mi = float(np.mean(last_epoch_mi))
    if not np.isfinite(final_mi):
        raise FloatingPointError(
            f"train_mine produced a non-finite MI estimate ({final_mi}) at "
            f"seed={seed}, n_features={n_features}, batch_size={batch_size} "
            f"— training diverged. Failing loudly instead of silently "
            f"propagating a broken value into a ranking/CSV.")
    if track_history:
        return final_mi, params, epoch_history
    return final_mi, params


# =============================================================================
# 2. LOO ranking
# =============================================================================

def loo_ranking(X_scaled, Y_scaled, n_features):
    mi_full, _ = train_mine(X_scaled, Y_scaled, SEED)
    importance = np.zeros(n_features)
    for i in range(n_features):
        X_reduced = np.delete(X_scaled, i, axis=1)
        mi_reduced, _ = train_mine(X_reduced, Y_scaled, SEED)
        importance[i] = mi_full - mi_reduced
    ranks = rankdata(-importance).astype(int)
    return mi_full, importance, ranks


# =============================================================================
# 3. Diagnostics: convergence check + intrinsic stochasticity check
# =============================================================================

def run_diagnostics(X_scaled_p27, Y_scaled):
    print("\n" + "=" * 70)
    print(f"DIAGNOSTICS (representative p={CONVERGENCE_REP_P}, full model)")
    print("=" * 70)

    # ---- Convergence check (1 run, full epoch history) ----------------------
    print("\n[Convergence check] training with full per-epoch MI trace...")
    t0 = time.time()
    final_mi, _, history = train_mine(X_scaled_p27, Y_scaled, SEED, track_history=True)
    print(f"  done in {time.time()-t0:.1f}s, final MI={final_mi:.4f}")

    hist = np.array(history)
    late = hist[CONVERGENCE_WINDOW_LATE[0]:CONVERGENCE_WINDOW_LATE[1]].mean()
    mid = hist[CONVERGENCE_WINDOW_MID[0]:CONVERGENCE_WINDOW_MID[1]].mean()
    rel_change = abs(late - mid) / abs(mid) if mid != 0 else np.inf
    converged = rel_change < CONVERGENCE_REL_THRESHOLD
    print(f"  mid-training mean (epochs {CONVERGENCE_WINDOW_MID})={mid:.4f}  "
          f"late mean (epochs {CONVERGENCE_WINDOW_LATE})={late:.4f}  "
          f"rel_change={rel_change:.3%}  "
          f"{'CONVERGED' if converged else 'NOT CLEARLY CONVERGED'} "
          f"(threshold {CONVERGENCE_REL_THRESHOLD:.0%})")

    pd.DataFrame({'epoch': np.arange(len(hist)), 'mi_estimate': hist}).to_csv(
        'mine_convergence_check_p27.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist, lw=1.2)
    ax.axvspan(*CONVERGENCE_WINDOW_MID, color='orange', alpha=0.15, label='mid window')
    ax.axvspan(*CONVERGENCE_WINDOW_LATE, color='green', alpha=0.15, label='late window')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MI estimate (nats)')
    ax.set_title(f'MINE convergence check — p={CONVERGENCE_REP_P}, full model\n'
                 f'{"Converged" if converged else "NOT clearly converged"} '
                 f'(rel. change={rel_change:.1%}, threshold={CONVERGENCE_REL_THRESHOLD:.0%})')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('mine_convergence_check_p27.png', dpi=200)
    plt.close(fig)
    print("  Saved: mine_convergence_check_p27.csv / .png")

    # ---- Intrinsic stochasticity check (3 seeds, same data) ------------------
    print("\n[Stochasticity check] 3 repeated trainings, different seeds, same data...")
    stoch_seeds = [42, 43, 44]
    stoch_mis = []
    for s in stoch_seeds:
        t0 = time.time()
        mi_s, _ = train_mine(X_scaled_p27, Y_scaled, s)
        stoch_mis.append(mi_s)
        print(f"  seed={s}: MI={mi_s:.4f}  [{time.time()-t0:.1f}s]")
    stoch_mis = np.array(stoch_mis)
    print(f"  mean={stoch_mis.mean():.4f}  std={stoch_mis.std():.4f}  "
          f"(intrinsic estimator noise, independent of data resampling)")

    return dict(converged=converged, rel_change=rel_change, final_mi=final_mi,
                mid_window_mean=mid, late_window_mean=late,
                stoch_mean=float(stoch_mis.mean()), stoch_std=float(stoch_mis.std()),
                stoch_values=stoch_mis.tolist())


# =============================================================================
# 4. MAIN
# =============================================================================

if __name__ == "__main__":
    if B_BOOTSTRAP is None:
        raise RuntimeError(
            "B_BOOTSTRAP not set — run the calibration step first and set "
            "B_BOOTSTRAP at the top of this file before launching the full sweep.")

    t_start_all = time.time()
    print("=" * 70)
    print("MINE — HIGH-DIMENSIONAL SCALABILITY (p=27,50,105), Belghazi et al. 2018")
    print(f"batch_size={BATCH_SIZE}  epochs={EPOCHS}  ema_decay={EMA_DECAY}  "
          f"B_bootstrap={B_BOOTSTRAP}")
    print(f"Time-box: {TIME_BOX_MIN} min (live safety check after each p level)")
    print("=" * 70)

    X_core, Y, core_names, core_groups, core_binary, core_rank = generate_core_dataset()
    n_core = X_core.shape[1]
    max_extra = max(P_LEVELS) - n_core
    rng_pad = np.random.default_rng(SEED + 1)
    noise_pool = rng_pad.normal(0, 1, size=(N, max_extra))

    Y_scaled = (Y - Y.mean()) / Y.std()
    Y_class = (Y_scaled > np.median(Y_scaled)).astype(int)

    # ---- Diagnostics on the representative p level (once, up front) ---------
    X_rep, _ = build_padded_dataset(CONVERGENCE_REP_P, X_core, core_names, core_groups,
                                     core_binary, core_rank, noise_pool)
    X_rep_scaled = StandardScaler().fit_transform(X_rep)
    diagnostics = run_diagnostics(X_rep_scaled, Y_scaled)

    point_rows = []
    bootstrap_rows = []
    stopped_early = False

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
        mi_full, importance, ranks = loo_ranking(X_scaled, Y_scaled, p)
        t_elapsed = time.time() - t0
        tau, _ = kendalltau(gt_rank, ranks)
        syn_r1, syn_r2 = int(ranks[syn_idx[0]]), int(ranks[syn_idx[1]])
        print(f"  [point p={p}] MI_full={mi_full:.4f}  tau={tau:.3f}  "
              f"XOR=({syn_r1},{syn_r2})  [{t_elapsed:.1f}s]")

        ds_results = evaluate_method_vs_baseline(
            X_scaled, Y_class, ranks, K_VALUES,
            n_random=100, k_neighbors=5, seed=SEED, method_name=f"MINE_p{p}")
        for r in ds_results:
            r.update(dict(p=p, tau=tau, mi_full_estimate=mi_full,
                           syn_rank_1=syn_r1, syn_rank_2=syn_r2))
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
            _, _, ranks_b = loo_ranking(Xb, yb, p)
            t, _ = kendalltau(gt_rank, ranks_b)
            taus_boot[b] = t
            print(f"    draw {b+1}/{B_BOOTSTRAP}: tau={t:.3f}  "
                  f"[{time.time()-t0:.0f}s elapsed]")
        lo, hi = np.percentile(taus_boot, [(100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2])
        bootstrap_rows.append(dict(
            p=p, method='MINE', tau_mean=float(taus_boot.mean()),
            ci_lo=float(lo), ci_hi=float(hi), tau_std=float(taus_boot.std()),
            B=B_BOOTSTRAP))
        print(f"  Bootstrap tau = {taus_boot.mean():.3f} [{lo:.3f}, {hi:.3f}]  "
              f"(B={B_BOOTSTRAP}, {(time.time()-t0)/60:.1f} min)")

        elapsed_min = (time.time() - t_start_all) / 60
        print(f"\n  Cumulative elapsed: {elapsed_min:.1f} min")
        if elapsed_min > TIME_BOX_MIN:
            print(f"  STOPPING: exceeded {TIME_BOX_MIN}-min time-box after p={p}.")
            stopped_early = True
            break

    # ---- Save outputs -------------------------------------------------------
    point_df = pd.DataFrame(point_rows)
    point_df.to_csv('mine_synthetic_highdim_point_estimates.csv', index=False)
    print("\nSaved: mine_synthetic_highdim_point_estimates.csv")

    boot_df = pd.DataFrame(bootstrap_rows)
    boot_df.to_csv('mine_synthetic_highdim_results.csv', index=False)
    print("Saved: mine_synthetic_highdim_results.csv "
          "(same schema as bootstrap_ci_synthetic_highdim_results.csv)")

    total_min = (time.time() - t_start_all) / 60
    with open('mine_synthetic_highdim_summary.txt', 'w') as f:
        f.write("MINE — HIGH-DIMENSIONAL SCALABILITY (p=27,50,105)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total runtime: {total_min:.1f} min"
                + (" (STOPPED EARLY — time-box exceeded)\n" if stopped_early else "\n"))
        f.write(f"batch_size={BATCH_SIZE}  ema_decay={EMA_DECAY}  "
                f"B_bootstrap={B_BOOTSTRAP}\n\n")

        f.write(f"--- CONVERGENCE CHECK (p={CONVERGENCE_REP_P}, full model) ---\n")
        f.write(f"  {'CONVERGED' if diagnostics['converged'] else 'NOT CLEARLY CONVERGED'} "
                f"within {EPOCHS} epochs (threshold {CONVERGENCE_REL_THRESHOLD:.0%})\n")
        f.write(f"  mid-window mean={diagnostics['mid_window_mean']:.4f}  "
                f"late-window mean={diagnostics['late_window_mean']:.4f}  "
                f"rel_change={diagnostics['rel_change']:.3%}\n\n")

        f.write(f"--- INTRINSIC STOCHASTICITY CHECK (p={CONVERGENCE_REP_P}, "
                f"3 seeds, same data) ---\n")
        f.write(f"  MI estimates: {diagnostics['stoch_values']}\n")
        f.write(f"  mean={diagnostics['stoch_mean']:.4f}  "
                f"std={diagnostics['stoch_std']:.4f}\n\n")

        f.write("--- POINT ESTIMATES (tau, XOR pair rank) ---\n")
        for _, r in point_df.drop_duplicates('p').iterrows():
            f.write(f"  p={int(r.p):<4} tau={r.tau:.3f}  "
                    f"XOR rank=({int(r.syn_rank_1)},{int(r.syn_rank_2)})  "
                    f"MI_full={r.mi_full_estimate:.4f}\n")
        f.write("\n--- BOOTSTRAP CI (tau) ---\n")
        for _, r in boot_df.iterrows():
            f.write(f"  p={int(r.p):<4} tau_mean={r.tau_mean:.3f}  "
                    f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]  (B={int(r.B)})\n")
        f.write("\nComparison: DII stays rank 1-3 at every p; II_joint collapses "
                "to rank ~22-70 at p=105 — see this run's XOR ranks above for "
                "where MINE falls on that spectrum.\n")

    print("Saved: mine_synthetic_highdim_summary.txt")
    print(f"\nTotal time: {total_min:.1f} min")
    print("\n" + "=" * 70)
    print("MINE HIGH-DIM ANALYSIS " + ("STOPPED EARLY" if stopped_early else "COMPLETE"))
    print("=" * 70)

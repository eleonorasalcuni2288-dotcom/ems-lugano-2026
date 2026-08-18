"""
InfoNCE — High-Dimensional Scalability Extension (p=27, 50, 105)
======================================================================
Extends infonce_synthetic.py (p=27 only) to the same p-levels used
throughout the rest of the scalability analysis (simulation_study_v6_highdim.py,
bootstrap_ci_synthetic_highdimensional.py): p=27, 50, 105.

REUSED, NOT DUPLICATED:
  - generate_core_dataset() from simulation_study_v6_highdim.py
  - build_padded_dataset() imported directly from
    bootstrap_ci_synthetic_highdimensional.py (same tie-rank convention
    for the padding noise features as every other high-dim analysis)
  - train_infonce() imported directly from infonce_synthetic.py — the
    EXACT same architecture/training already validated at p=27
  - evaluate_method_vs_baseline() from downstream_validation.py

NOISE POOL: regenerated with the SAME seed (SEED+1=43) and SAME shape
derivation (max(P_LEVELS)-n_core columns) that
bootstrap_ci_synthetic_highdimensional.py uses internally. That 2-line
generation lives inside that script's __main__ guard, so it isn't
importable as a variable — replicating just those 2 deterministic lines
here (not the padding LOGIC itself, which IS imported) reproduces a
byte-identical noise pool to every other high-dim result in this project.

TWO SEPARATE OUTPUTS PER p (mirrors the project's existing split between
simulation_study_v6_highdim.py [point estimates] and
bootstrap_ci_synthetic_highdimensional.py [tau CI only]):

  A. POINT ESTIMATES (full N=2000, one LOO sweep per batch_size): tau,
     XOR synergy-pair rank, downstream validation (Top-K accuracy vs
     random baseline). Computed for BOTH batch sizes (128, 512), all 3
     p levels — exactly as requested, no scope reduction here.

  B. BOOTSTRAP CI on tau (subsampling 80% without replacement, B draws,
     full LOO ranking recomputed per draw) — SCOPE REDUCED, WITH
     JUSTIFICATION. See "COST REALITY" below.

COST REALITY (measured, not guessed): unlike DII, MI_joint or II_joint
(1 training per bootstrap draw), InfoNCE's LOO-based ranking needs
(1 + p) trainings PER bootstrap draw (1 full + p leave-one-out models).
Calibrated directly before running anything: one full LOO sweep at
p=105, batch_size=128, on an 80%-subsample (N=1600) measured at ~170s
(2.8 min) — full training 2.06s + 104 x 1.61s LOO retrainings. At
B=15-20 as originally requested, across all 3 p levels and BOTH batch
sizes, total bootstrap cost alone would be roughly 2.5-3 HOURS — far
outside the 1-hour time-box. Scope was therefore reduced, calibrated
from this measured per-draw cost to fit comfortably under the box:
  - Bootstrap CI computed for batch_size=128 ONLY (point estimates
    still cover both batch sizes; only the CI layer is restricted)
  - B=6 draws per p level (down from the requested 15-20) — an even
    coarser CI than DII's own B=15-20 in this project, flagged
    explicitly here and in every output file, not hidden.
This mirrors the precedent already set in this project for JMI greedy
(skipped above p=60 for cost in simulation_study_v6_highdim.py) — a
principled, documented compute-cost cutoff rather than a silent
shortcut. A live elapsed-time check after each p level provides a
safety net: if the 1h box is exceeded, the run stops and reports
whatever was completed rather than silently running over.

OUTPUT:
  infonce_synthetic_highdim_results.csv        — bootstrap CI, SAME
    SCHEMA as bootstrap_ci_synthetic_highdim_results.csv (p, method,
    tau_mean, ci_lo, ci_hi, tau_std, B) for direct union/comparison.
  infonce_synthetic_highdim_point_estimates.csv — tau, XOR rank,
    downstream validation, per (p, batch_size).
  infonce_synthetic_highdim_summary.txt         — human-readable
    summary, including the explicit log(batch_size)-cap saturation
    check across p requested below.

CONSTRAINTS: does not modify any existing file.
"""

import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau
from sklearn.preprocessing import StandardScaler

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)                              # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # simulation_study_v6_highdim.py lives here
from simulation_study_v6_highdim import generate_core_dataset
from bootstrap_ci_synthetic_highdimensional import build_padded_dataset, SEED, N
from infonce_synthetic import train_infonce, BATCH_SIZES, EPOCHS, K_VALUES
from downstream_validation import evaluate_method_vs_baseline

P_LEVELS = [27, 50, 105]
SUBSAMPLE_FRAC = 0.80
CI_LEVEL = 95
TIME_BOX_MIN = 60

# Bootstrap scope reduction — see docstring "COST REALITY"
BOOTSTRAP_BATCH_SIZE = 128
B_BOOTSTRAP = 6


def loo_ranking(X_scaled, Y_scaled, batch_size, n_features):
    """One full LOO sweep: 1 full-model training + n_features leave-
    one-out retrainings (same seed for every retraining). Returns
    (mi_full, importance, ranks)."""
    mi_full, _ = train_infonce(X_scaled, Y_scaled, SEED, batch_size, epochs=EPOCHS)
    importance = np.zeros(n_features)
    for i in range(n_features):
        X_reduced = np.delete(X_scaled, i, axis=1)
        mi_reduced, _ = train_infonce(X_reduced, Y_scaled, SEED, batch_size, epochs=EPOCHS)
        importance[i] = mi_full - mi_reduced
    ranks = rankdata(-importance).astype(int)
    return mi_full, importance, ranks


if __name__ == "__main__":
    t_start_all = time.time()
    print("=" * 70)
    print("InfoNCE — HIGH-DIMENSIONAL SCALABILITY EXTENSION (p=27,50,105)")
    print(f"Point estimates: batch_sizes={BATCH_SIZES}, all p levels")
    print(f"Bootstrap CI: batch_size={BOOTSTRAP_BATCH_SIZE} ONLY, B={B_BOOTSTRAP} "
          f"(reduced from 15-20 — see docstring COST REALITY)")
    print(f"Time-box: {TIME_BOX_MIN} min (live safety check after each p level)")
    print("=" * 70)

    X_core, Y, core_names, core_groups, core_binary, core_rank = generate_core_dataset()
    n_core = X_core.shape[1]
    max_extra = max(P_LEVELS) - n_core
    rng_pad = np.random.default_rng(SEED + 1)   # replicated 2 lines — see docstring
    noise_pool = rng_pad.normal(0, 1, size=(N, max_extra))

    Y_scaled = (Y - Y.mean()) / Y.std()
    Y_class = (Y_scaled > np.median(Y_scaled)).astype(int)

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

        # ---- A. POINT ESTIMATES (both batch sizes) -------------------------
        for bs in BATCH_SIZES:
            t0 = time.time()
            mi_full, importance, ranks = loo_ranking(X_scaled, Y_scaled, bs, p)
            t_elapsed = time.time() - t0
            tau, _ = kendalltau(gt_rank, ranks)
            syn_r1, syn_r2 = int(ranks[syn_idx[0]]), int(ranks[syn_idx[1]])
            mi_cap = float(np.log(bs))
            print(f"  [point p={p} bs={bs}] MI_full={mi_full:.4f} "
                  f"(cap=log({bs})={mi_cap:.4f}, {100*mi_full/mi_cap:.1f}% of cap)  "
                  f"tau={tau:.3f}  XOR=({syn_r1},{syn_r2})  [{t_elapsed:.1f}s]")

            ds_results = evaluate_method_vs_baseline(
                X_scaled, Y_class, ranks, K_VALUES,
                n_random=100, k_neighbors=5, seed=SEED,
                method_name=f"InfoNCE_p{p}_bs{bs}")
            for r in ds_results:
                r.update(dict(p=p, batch_size=bs, tau=tau, mi_full_estimate=mi_full,
                               mi_cap_log_bs=mi_cap, syn_rank_1=syn_r1, syn_rank_2=syn_r2))
                point_rows.append(r)

        # ---- B. BOOTSTRAP CI on tau (bs=128 only, B draws) ------------------
        print(f"\n  Bootstrap CI (bs={BOOTSTRAP_BATCH_SIZE}, B={B_BOOTSTRAP})...")
        n = X_scaled.shape[0]
        n_sub = int(round(n * SUBSAMPLE_FRAC))
        rng = np.random.default_rng(SEED)
        taus_boot = np.empty(B_BOOTSTRAP)
        t0 = time.time()
        for b in range(B_BOOTSTRAP):
            idx = rng.choice(n, size=n_sub, replace=False)
            Xb, yb = X_scaled[idx], Y_scaled[idx]
            _, _, ranks_b = loo_ranking(Xb, yb, BOOTSTRAP_BATCH_SIZE, p)
            t, _ = kendalltau(gt_rank, ranks_b)
            taus_boot[b] = t
            print(f"    draw {b+1}/{B_BOOTSTRAP}: tau={t:.3f}  "
                  f"[{time.time()-t0:.0f}s elapsed]")
        lo, hi = np.percentile(taus_boot, [(100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2])
        bootstrap_rows.append(dict(
            p=p, method=f'InfoNCE_bs{BOOTSTRAP_BATCH_SIZE}',
            tau_mean=float(taus_boot.mean()), ci_lo=float(lo), ci_hi=float(hi),
            tau_std=float(taus_boot.std()), B=B_BOOTSTRAP))
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
    point_df.to_csv('infonce_synthetic_highdim_point_estimates.csv', index=False)
    print("\nSaved: infonce_synthetic_highdim_point_estimates.csv")

    boot_df = pd.DataFrame(bootstrap_rows)
    boot_df.to_csv('infonce_synthetic_highdim_results.csv', index=False)
    print("Saved: infonce_synthetic_highdim_results.csv "
          "(same schema as bootstrap_ci_synthetic_highdim_results.csv)")

    # ---- Summary, including explicit log(batch_size)-cap saturation check ---
    total_min = (time.time() - t_start_all) / 60
    with open('infonce_synthetic_highdim_summary.txt', 'w') as f:
        f.write("InfoNCE — HIGH-DIMENSIONAL SCALABILITY EXTENSION (p=27,50,105)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total runtime: {total_min:.1f} min"
                + (" (STOPPED EARLY — time-box exceeded)\n" if stopped_early else "\n"))
        f.write(f"Bootstrap CI scope: batch_size={BOOTSTRAP_BATCH_SIZE} only, "
                f"B={B_BOOTSTRAP} (reduced from requested 15-20 — see script docstring)\n\n")

        f.write("--- log(batch_size) CAP SATURATION CHECK ---\n")
        f.write("(MI_full estimate as a fraction of its own batch-size cap, per p)\n")
        for p in point_df.p.unique():
            for bs in BATCH_SIZES:
                sub = point_df[(point_df.p == p) & (point_df.batch_size == bs)]
                if len(sub) == 0:
                    continue
                mi_full = sub.mi_full_estimate.iloc[0]
                cap = sub.mi_cap_log_bs.iloc[0]
                f.write(f"  p={p:<4} bs={bs:<4} MI_full={mi_full:.4f}  "
                        f"cap={cap:.4f}  {100*mi_full/cap:5.1f}% of cap\n")
        f.write("\n")

        f.write("--- POINT ESTIMATES (tau, XOR pair rank) ---\n")
        for p in point_df.p.unique():
            for bs in BATCH_SIZES:
                sub = point_df[(point_df.p == p) & (point_df.batch_size == bs)]
                if len(sub) == 0:
                    continue
                tau = sub.tau.iloc[0]
                r1, r2 = int(sub.syn_rank_1.iloc[0]), int(sub.syn_rank_2.iloc[0])
                f.write(f"  p={p:<4} bs={bs:<4} tau={tau:.3f}  XOR rank=({r1},{r2})\n")
        f.write("\n")

        f.write("--- BOOTSTRAP CI (tau, bs=128 only) ---\n")
        for _, r in boot_df.iterrows():
            f.write(f"  p={int(r.p):<4} tau_mean={r.tau_mean:.3f}  "
                    f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]  (B={int(r.B)})\n")

    print("Saved: infonce_synthetic_highdim_summary.txt")
    print(f"\nTotal time: {total_min:.1f} min")
    print("\n" + "=" * 70)
    print("INFONCE HIGH-DIM EXTENSION " + ("STOPPED EARLY" if stopped_early else "COMPLETE"))
    print("=" * 70)

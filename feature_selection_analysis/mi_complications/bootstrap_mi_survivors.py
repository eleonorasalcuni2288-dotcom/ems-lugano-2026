"""
Bootstrap CI — All 7 BH-Correction Survivors (MI Complications)
======================================================================
The per-complication analysis (mi_complications_per_target.py) found 7
(complication, method, K) combinations surviving Benjamini-Hochberg
correction across 176 tests:

  1. ZSN         MI_perfeat  K=3
  2. ZSN         MI_perfeat  K=5
  3. ZSN         DII_L1      K=16
  4. FIBR_JELUD  MI_perfeat  K=10
  5. FIBR_JELUD  II_joint    K=10
  6. JELUD_TAH   II_perfeat  K=16
  7. OTEK_LANC   II_joint    K=16

Surviving BH correction controls the expected false-discovery RATE
across the 176 tests, but says nothing about how STABLE each individual
finding is to resampling. A finding backed by two independent methods on
the same complication (ZSN: MI+DII; FIBR_JELUD: MI+II_joint) is already
better evidence than a single isolated (complication, method, K) hit
(JELUD_TAH, OTEK_LANC each have only one surviving config) — bootstrap
CIs quantify this directly: a wide CI crossing zero, or a low fraction
of positive-advantage draws, would suggest a fragile/possibly spurious
finding even though it passed the raw correction.

Method: same subsampling-without-replacement (80% of N) as
bootstrap_ci_synthetic.py, re-deriving the ranking AND the downstream
advantage (method accuracy - random-feature baseline) on each draw.

Cost: 6 of 7 configs use fast methods (MI_perfeat/II_perfeat/II_joint,
B=100 each); only ZSN/DII_L1 needs the slow DII training (B=15). Budget
~25-30 min total.

Run from the project folder (needs MI.data, simulation_study_v6_highdim.py,
downstream_validation.py).
"""
import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'common'))     # downstream_validation.py lives here
_sys.path.insert(0, _os.path.join(_ROOT, 'synthetic'))  # simulation_study_v6_highdim.py lives here
from simulation_study_v6_highdim import compute_ii_pf, make_ii_joint, run_dii
from downstream_validation import knn_loo_accuracy

SEED = 42
DATA_PATH = "MI.data"
HIGH_MISSING_COLS = [7, 34, 35, 88]
SUBSAMPLE_FRAC = 0.80
N_RANDOM_BASELINE = 50
B_FAST = 100
B_DII = 15

COMPLICATION_COLS = {
    'ZSN': 120, 'FIBR_JELUD': 115, 'JELUD_TAH': 114, 'OTEK_LANC': 117,
}

# (complication, method, K, B)
SURVIVORS = [
    ('ZSN',        'MI_perfeat', 3,  B_FAST),
    ('ZSN',        'MI_perfeat', 5,  B_FAST),
    ('ZSN',        'DII_L1',     16, B_DII),
    ('FIBR_JELUD', 'MI_perfeat', 10, B_FAST),
    ('FIBR_JELUD', 'II_joint',   10, B_FAST),
    ('JELUD_TAH',  'II_perfeat', 16, B_FAST),
    ('OTEK_LANC',  'II_joint',   16, B_FAST),
]


def rank_for_method(method, Xb, yb, label):
    if method == 'MI_perfeat':
        scores = mutual_info_classif(Xb, yb, random_state=SEED)
        return rankdata(-scores).astype(int)
    if method == 'II_perfeat':
        yb_f = yb.astype(np.float64)
        dy = np.abs(yb_f.reshape(-1, 1) - yb_f.reshape(1, -1))
        np.fill_diagonal(dy, np.inf)
        ry = np.argsort(np.argsort(dy, axis=1), axis=1)
        scores = np.array([compute_ii_pf(Xb[:, i], ry) for i in range(Xb.shape[1])])
        return rankdata(scores).astype(int)
    if method == 'II_joint':
        yb_f = yb.astype(np.float64)
        ii_joint_fn = make_ii_joint(yb_f)
        full = ii_joint_fn(Xb)
        loo = np.array([ii_joint_fn(np.delete(Xb, i, axis=1))
                         for i in range(Xb.shape[1])])
        return rankdata(-(loo - full)).astype(int)
    if method == 'DII_L1':
        yb_f = yb.astype(np.float64)
        _, ranks, _ = run_dii(Xb, yb_f, 0.10, label, N=len(yb))
        return ranks
    raise ValueError(method)


def advantage_on_subsample(X_full, y_full, method, K, rng, draw_idx, label):
    n = X_full.shape[0]
    n_sub = int(round(n * SUBSAMPLE_FRAC))
    idx = rng.choice(n, size=n_sub, replace=False)
    Xb, yb = X_full[idx], y_full[idx]
    n_features = Xb.shape[1]

    ranks = rank_for_method(method, Xb, yb, label)
    top_k_idx = np.argsort(ranks)[:K]
    method_acc = knn_loo_accuracy(Xb, yb, top_k_idx, k_neighbors=5)

    baseline_accs = np.empty(N_RANDOM_BASELINE)
    rng_baseline = np.random.default_rng(SEED + draw_idx)
    for i in range(N_RANDOM_BASELINE):
        rand_idx = rng_baseline.choice(n_features, size=K, replace=False)
        baseline_accs[i] = knn_loo_accuracy(Xb, yb, rand_idx, k_neighbors=5)

    return method_acc - baseline_accs.mean()


if __name__ == "__main__":
    print("=" * 70)
    print("BOOTSTRAP CI — ALL 7 BH-CORRECTION SURVIVORS")
    print(f"Subsample fraction: {SUBSAMPLE_FRAC}  |  95% percentile CI")
    print("=" * 70)

    print("\n[1/2] Loading and cleaning X (shared across all targets)...")
    df = pd.read_csv(DATA_PATH, header=None, na_values='?')
    input_cols = [c for c in range(1, 112) if c not in HIGH_MISSING_COLS]
    X_df = df[input_cols].copy()
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = np.nan_to_num(X_df.values.astype(np.float64), nan=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  X shape: {X.shape}")

    print("\n[2/2] Bootstrapping each surviving finding...")
    results = {}
    t_start = time.time()

    for comp_name, method, K, B in SURVIVORS:
        y = (df[COMPLICATION_COLS[comp_name]] == 1).astype(int).values
        config_name = f"{comp_name}_{method}_K{K}"
        print(f"\n --- {config_name} (B={B}) ---")
        rng = np.random.default_rng(SEED)
        advantages = np.empty(B)
        t0 = time.time()
        for b in range(B):
            advantages[b] = advantage_on_subsample(
                X_scaled, y, method, K, rng, b,
                label=f"DII bootstrap {config_name} {b}")
            if (b + 1) % max(1, B // 5) == 0:
                print(f"    {b+1}/{B} draws done "
                      f"({time.time()-t0:.0f}s elapsed)")

        lo, hi = np.percentile(advantages, [2.5, 97.5])
        frac_positive = float(np.mean(advantages > 0))
        results[config_name] = dict(
            complication=comp_name, method=method, K=K,
            mean_advantage=advantages.mean(), ci_lo=lo, ci_hi=hi,
            std=advantages.std(), frac_positive=frac_positive, B=B)
        crosses_zero = lo <= 0 <= hi
        print(f"  advantage = {advantages.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
              f"{'(CROSSES ZERO — fragile)' if crosses_zero else '(robust)'}  "
              f"P(advantage>0)={frac_positive:.2f}")

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

    # ---- Summary, sorted by robustness ---------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY — sorted by fraction of draws with positive advantage")
    print("=" * 70)
    rows = list(results.values())
    rows.sort(key=lambda r: -r['frac_positive'])
    for r in rows:
        crosses_zero = r['ci_lo'] <= 0 <= r['ci_hi']
        flag = "ROBUST" if not crosses_zero else "FRAGILE (CI crosses 0)"
        print(f"  {r['complication']:<12} {r['method']:<12} K={r['K']:<3} "
              f"adv={r['mean_advantage']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
              f"P(+)={r['frac_positive']:.2f}  [{flag}]")

    pd.DataFrame(rows).to_csv('bootstrap_ci_mi_survivors_results.csv', index=False)
    print("\nSaved: bootstrap_ci_mi_survivors_results.csv")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
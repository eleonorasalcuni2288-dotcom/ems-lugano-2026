"""
Simulation Study v6: High-Dimensional Scalability Sweep
=========================================================
Extends v5's two-level fair comparison (MI, II, JMI greedy, DII) to a
high-dimensional regime, addressing the limitation stated in the project
README: "Scalability to >100 features not extensively tested".

Design:
  - The 27 "core" features and target Y are generated EXACTLY as in v5
    (same latent processes z1-z6, same Y formula, same XOR synergy pair
    x_syn_1/x_syn_2, same ground truth ranks 1-27). This keeps the p=27
    point directly comparable to v5's published numbers.
  - A fixed pool of extra pure-noise features is generated once and then
    features are ADDED (not regenerated) as p grows, so the p=50 run's
    extra columns are a strict subset of the p=100 run's, isolating the
    effect of growing dimensionality from randomness in the noise draw.
  - All methods are re-run at each p in P_VALUES:
        Level 1: MI (per-feature), II (per-feature)
        Level 2: JMI greedy (only up to JMI_MAX_P — see note below),
                 MI joint LOO, II joint LOO, DII+L1, DII no-L1
  - Wall-clock time is recorded per method per p, producing the
    scalability evidence itself (not just accuracy).

Note on JMI greedy: its cost grows as O(p^2) CMI evaluations (Frenzel &
Pompe estimator, each with an O(N) Python-level neighbour count). This
becomes intractable well before DII or the LOO methods, which is itself
a relevant finding (sequential greedy conditional-MI selection does not
scale to high dimensions the way joint/gradient-based methods do). JMI
is therefore only run for p <= JMI_MAX_P and skipped (NaN) above that,
with the skip itself reported.
"""

import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial  import cKDTree
from scipy.special  import digamma
from scipy.stats    import rankdata, kendalltau, spearmanr
from sklearn.preprocessing      import StandardScaler
from sklearn.feature_selection  import mutual_info_regression
from sklearn.neighbors          import NearestNeighbors
from dadapy.diff_imbalance      import DiffImbalance

SEED       = 42
N          = 2000
K_CMI      = 5
P_VALUES   = [27, 50, 105]   # dimensionalities to sweep
JMI_MAX_P  = 60                   # JMI greedy skipped above this (see docstring)

# =============================================================================
# 1. CORE DATASET  (identical to v5 — do not change, keeps p=27 comparable)
# =============================================================================

def generate_core_dataset(seed=SEED, N=N):
    rng = np.random.default_rng(seed)
    t          = np.arange(N)
    cycle_fast = np.sin(2*np.pi*t/50)
    cycle_slow = np.sin(2*np.pi*t/200)
    eps        = rng.normal(0,1,N)

    z1 = np.zeros(N); z2 = np.zeros(N); z3 = np.zeros(N)
    for i in range(1,N):
        z1[i] =  0.72*z1[i-1] + 0.55*eps[i]  + rng.normal(0,0.25)
        z2[i] =  0.45*z2[i-1] + 0.30*z1[i-1] + rng.normal(0,0.35)
        z3[i] = -0.30*z3[i-1] + 0.25*z2[i-1] + rng.normal(0,0.30)

    z4 = rng.normal(0,1,N); z5 = rng.normal(0,1,N)
    z6 = np.zeros(N)
    for i in range(1,N): z6[i] = 0.65*z6[i-1] + rng.normal(0,0.50)

    vol = 0.25 + 0.18*np.abs(cycle_slow) + 0.12*(z1**2/(1+z1**2))

    x_lin_1  = z1                        + rng.normal(0,0.10,N)
    x_lin_2  = 0.8*z2                    + rng.normal(0,0.10,N)
    x_sym_1  = z1**2                     + rng.normal(0,0.08,N)
    x_sym_2  = np.abs(z2)                + rng.normal(0,0.08,N)
    x_dir_1  = np.maximum(z1,0)          + rng.normal(0,0.06,N)
    x_dir_2  = np.maximum(-z2,0)         + rng.normal(0,0.06,N)
    x_wave_1 = cycle_fast                + rng.normal(0,0.05,N)
    x_wave_2 = np.cos(2*np.pi*t/50)     + rng.normal(0,0.05,N)
    x_mix_1  = z1*z2                     + rng.normal(0,0.08,N)
    x_mix_2  = z1/(1+np.abs(z3))        + rng.normal(0,0.08,N)
    x_red_1  = x_lin_1                   + rng.normal(0,0.04,N)
    x_red_2  = x_sym_1                   + rng.normal(0,0.04,N)
    x_red_3  = 0.5*x_dir_1+0.5*x_mix_1  + rng.normal(0,0.05,N)
    regime   = (cycle_slow > 0).astype(float)
    momentum = pd.Series(z1).rolling(10,min_periods=1).mean().to_numpy()
    rvp      = pd.Series(z1**2).rolling(10,min_periods=1).mean().to_numpy()
    x_syn_1  = z4 + rng.normal(0,0.10,N)
    x_syn_2  = z5 + rng.normal(0,0.10,N)
    xor_sig  = np.sign(z4*z5)
    x_base   = z6 + rng.normal(0,0.05,N)
    x_dup_1  = x_base + rng.normal(0,0.02,N)
    x_dup_2  = x_base + rng.normal(0,0.02,N)
    x_dup_3  = x_base + rng.normal(0,0.03,N)
    x_dup_4  = x_base + rng.normal(0,0.03,N)
    noise_1  = rng.normal(0,1,N); noise_2 = rng.normal(0,1,N)
    noise_3  = rng.normal(0,1,N); noise_4 = rng.normal(0,1,N)

    Y = (  0.90*x_lin_1 + 0.70*x_lin_2
         + 0.85*x_sym_1 + 0.65*x_sym_2
         + 1.10*x_dir_1 + 0.95*x_dir_2
         + 0.35*x_wave_1 + 0.55*x_mix_1
         + 0.20*regime   + 0.80*xor_sig
         + 1.20*x_base   + rng.normal(0, vol, N))

    feature_names = np.array([
        'x_lin_1','x_lin_2','x_sym_1','x_sym_2',
        'x_dir_1','x_dir_2','x_wave_1','x_wave_2',
        'x_mix_1','x_mix_2','x_red_1','x_red_2','x_red_3',
        'momentum','rvp','regime',
        'x_syn_1','x_syn_2',
        'x_base','x_dup_1','x_dup_2','x_dup_3','x_dup_4',
        'noise_1','noise_2','noise_3','noise_4',
    ])
    feature_groups = np.array([
        'Linear','Linear','NonlinSym','NonlinSym',
        'Directional','Directional','Cyclical','Cyclical',
        'Interaction','Interaction','Redundant','Redundant','Redundant',
        'Derived','Derived','Derived',
        'Synergistic','Synergistic',
        'NearDup','NearDup','NearDup','NearDup','NearDup',
        'Noise','Noise','Noise','Noise',
    ])
    gt_binary = np.array([
        1,1, 1,1, 1,1, 1,0, 1,0, 1,1,1, 0,0,1, 1,1, 1,1,1,1,1, 0,0,0,0,
    ])
    gt_rank = np.array([
         4,  6,  5,  7,  2,  3,
         9, 20,  8, 21,
        13, 14, 15,
        22, 23, 10,
        11, 12,
         1, 16, 17, 18, 19,
        24, 25, 26, 27,
    ])
    X = np.column_stack([
        x_lin_1, x_lin_2, x_sym_1, x_sym_2,
        x_dir_1, x_dir_2, x_wave_1, x_wave_2,
        x_mix_1, x_mix_2, x_red_1, x_red_2, x_red_3,
        momentum, rvp, regime,
        x_syn_1, x_syn_2,
        x_base, x_dup_1, x_dup_2, x_dup_3, x_dup_4,
        noise_1, noise_2, noise_3, noise_4,
    ])
    return X, Y, feature_names, feature_groups, gt_binary, gt_rank


# =============================================================================
# 2. ESTIMATORS  (identical implementations to v5)
# =============================================================================

def compute_cmi(xi_1d, y_1d, xc_2d, k=K_CMI):
    N  = len(xi_1d)
    xi = xi_1d.reshape(-1,1)
    y  = y_1d.reshape(-1,1)
    if xc_2d.shape[1] == 0:
        return float(mutual_info_regression(
            xi, y_1d, n_neighbors=k, random_state=SEED)[0])
    xyz = np.hstack([xi, y, xc_2d])
    xz  = np.hstack([xi, xc_2d])
    yz  = np.hstack([y,  xc_2d])
    z   = xc_2d
    tree_xyz = cKDTree(xyz)
    dists, _ = tree_xyz.query(xyz, k=k+1, p=np.inf)
    eps = np.maximum(dists[:, k], 1e-15)

    def count_nbrs(data):
        tree = cKDTree(data)
        return np.array([
            len(tree.query_ball_point(data[i], eps[i], p=np.inf)) - 1
            for i in range(N)
        ], dtype=float)

    n_xz = count_nbrs(xz); n_yz = count_nbrs(yz); n_z = count_nbrs(z)
    cmi = (digamma(k)
           + np.mean(digamma(np.maximum(n_z,  0.0) + 1))
           - np.mean(digamma(np.maximum(n_xz, 0.0) + 1))
           - np.mean(digamma(np.maximum(n_yz, 0.0) + 1)))
    return float(max(0.0, cmi))


def compute_mi_joint(X_mat, y_1d, k=K_CMI):
    N = len(y_1d)
    Y = y_1d.reshape(-1, 1)
    XY = np.hstack([X_mat, Y])
    tree_XY = cKDTree(XY)
    dists, _ = tree_XY.query(XY, k=k+1, p=np.inf)
    eps = np.maximum(dists[:, k], 1e-15)
    tree_X = cKDTree(X_mat); tree_Y = cKDTree(Y)
    n_X = np.array([
        len(tree_X.query_ball_point(X_mat[i], eps[i], p=np.inf)) - 1
        for i in range(N)
    ], dtype=float)
    n_Y = np.array([
        len(tree_Y.query_ball_point(Y[i], eps[i], p=np.inf)) - 1
        for i in range(N)
    ], dtype=float)
    mi = (digamma(k) + digamma(N)
          - np.mean(digamma(np.maximum(n_X, 0.0) + 1))
          - np.mean(digamma(np.maximum(n_Y, 0.0) + 1)))
    return float(max(0.0, mi))


def jmi_greedy(X_sc, y_sc, k=K_CMI):
    N, d = X_sc.shape
    rng_jmi   = np.random.default_rng(SEED)
    remaining = list(rng_jmi.permutation(d))
    selected  = []
    cmi_vals  = np.zeros(d)
    for step in range(d):
        xc_2d = X_sc[:, selected] if selected else np.empty((N, 0))
        best_val, best_feat = -1.0, remaining[0]
        for fi in remaining:
            val = compute_cmi(X_sc[:, fi], y_sc, xc_2d, k=k)
            if val > best_val:
                best_val, best_feat = val, fi
        cmi_vals[best_feat] = best_val
        selected.append(best_feat)
        remaining.remove(best_feat)
    ranks = np.zeros(d, dtype=int)
    for pos, fi in enumerate(selected):
        ranks[fi] = pos + 1
    return ranks, cmi_vals


def compute_ii_pf(xi, ry):
    """ry = precomputed rank-of-Y-distance matrix (depends only on Y, which
    is fixed across the whole p-sweep — hoisted out and passed in so it is
    computed once instead of once per feature per p)."""
    N  = len(xi)
    xi = xi.reshape(-1,1)
    dx = np.abs(xi-xi.T); np.fill_diagonal(dx, np.inf)
    nn = np.argmin(dx, axis=1)
    return 2.0/N**2 * np.sum(ry[np.arange(N), nn])


def make_ii_joint(Y_scaled):
    dist_Y_mat = np.abs(Y_scaled[:,None] - Y_scaled[None,:])
    np.fill_diagonal(dist_Y_mat, np.inf)
    Y_ranks_mat = np.argsort(np.argsort(dist_Y_mat, axis=1), axis=1)

    def ii_joint(X_sub):
        N = len(X_sub)
        nb = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X_sub)
        _, idx = nb.kneighbors(X_sub)
        return 2.0/N**2 * float(np.sum(Y_ranks_mat[np.arange(N), idx[:,1]]))
    return ii_joint


def run_dii(X_scaled, Y_scaled, l1, label, N=N):
    k_init  = max(5, int(0.025*N))
    k_final = max(1, int(0.010*N))
    m = DiffImbalance(
        data_A=X_scaled.astype(np.float64),
        data_B=Y_scaled.reshape(-1,1).astype(np.float64),
        num_epochs=300, batches_per_epoch=1, seed=SEED,
        l1_strength=l1, point_adapt_lambda=True,
        k_init=k_init, k_final=k_final, lambda_factor=0.1,
        optimizer_name='adam', learning_rate=1e-2,
        learning_rate_decay='cos',
    )
    _, imbs = m.train(bar_label=label)
    w = np.array(m.params_final)
    return w, rankdata(-w).astype(int), imbs


def tau_rho(gt_rank, ranks):
    t,_ = kendalltau(gt_rank, ranks)
    r,_ = spearmanr(gt_rank,  ranks)
    return t, r

def topk(gt_binary, ranks, k):
    return float(gt_binary[ranks <= k].sum()) / k


if __name__ == "__main__":
    # =============================================================================
    # 3. BUILD NESTED FEATURE POOL  (core 27 + growing noise padding)
    # =============================================================================
    print("="*70)
    print("SIMULATION STUDY v6 — HIGH-DIMENSIONAL SCALABILITY SWEEP")
    print(f"P_VALUES = {P_VALUES}  |  JMI greedy run only for p <= {JMI_MAX_P}")
    print("="*70)

    X_core, Y, core_names, core_groups, core_binary, core_rank = generate_core_dataset()
    n_core = X_core.shape[1]
    assert n_core == 27

    max_extra = max(P_VALUES) - n_core
    rng_pad   = np.random.default_rng(SEED + 1)
    noise_pool = rng_pad.normal(0, 1, size=(N, max_extra))  # fixed pool, sliced per p

    Y_scaled = (Y - Y.mean()) / Y.std()
    ii_joint = make_ii_joint(Y_scaled)   # depends only on Y — build once

    # rank-of-Y-distance matrix for II per-feature — depends only on Y, so it is
    # identical at every p in the sweep. Built once here instead of inside
    # compute_ii_pf (which was previously rebuilding it on every single call).
    _dy_global = np.abs(Y_scaled.reshape(-1,1) - Y_scaled.reshape(1,-1))
    np.fill_diagonal(_dy_global, np.inf)
    ry_global = np.argsort(np.argsort(_dy_global, axis=1), axis=1)

    ks = [3, 5, 10, 16]
    syn_idx_core = [np.where(core_names==f)[0][0] for f in ['x_syn_1','x_syn_2']]

    rows = []       # scalability summary (one row per method per p)
    t_start_all = time.time()

    # =============================================================================
    # 4. SWEEP OVER p
    # =============================================================================
    for p in P_VALUES:
        n_extra = p - n_core
        print(f"\n{'='*70}\n  p = {p}  ({n_extra} extra noise features)\n{'='*70}")

        if n_extra > 0:
            X_p = np.column_stack([X_core, noise_pool[:, :n_extra]])
            extra_names  = np.array([f'hd_noise_{i+1}' for i in range(n_extra)])
            extra_groups = np.array(['Noise'] * n_extra)
            extra_binary = np.zeros(n_extra, dtype=int)
            # extra noise features are exchangeable (iid, uninformative): give them
            # a single tied rank rather than an arbitrary sequential order, so
            # tau/rho don't penalise methods for failing to guess an order that
            # doesn't actually exist in the data-generating process.
            tie_rank     = np.arange(n_core+1, n_core+1+n_extra).mean()
            extra_rank   = np.full(n_extra, tie_rank)
            feature_names = np.concatenate([core_names, extra_names])
            feature_groups= np.concatenate([core_groups, extra_groups])
            gt_binary     = np.concatenate([core_binary, extra_binary])
            gt_rank       = np.concatenate([core_rank.astype(float), extra_rank])
        else:
            X_p = X_core
            feature_names, feature_groups = core_names, core_groups
            gt_binary, gt_rank = core_binary, core_rank

        syn_idx = [np.where(feature_names==f)[0][0] for f in ['x_syn_1','x_syn_2']]

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_p)

        # ---- Level 1: MI, II per-feature -------------------------------------
        t0 = time.time()
        mi_scores = mutual_info_regression(X_scaled, Y_scaled, random_state=SEED)
        mi_ranks  = rankdata(-mi_scores).astype(int)
        t_mi = time.time() - t0
        print(f"  [MI per-feature]   {t_mi:6.1f}s")

        t0 = time.time()
        ii_scores = np.array([compute_ii_pf(X_scaled[:,i], ry_global) for i in range(p)])
        ii_ranks  = rankdata(ii_scores).astype(int)
        t_ii = time.time() - t0
        print(f"  [II per-feature]   {t_ii:6.1f}s")

        # ---- Level 2: JMI greedy (skipped above JMI_MAX_P) ---------------------
        if p <= JMI_MAX_P:
            t0 = time.time()
            jmi_ranks, _ = jmi_greedy(X_scaled, Y_scaled, k=K_CMI)
            t_jmi = time.time() - t0
            print(f"  [JMI greedy]       {t_jmi:6.1f}s")
        else:
            jmi_ranks, t_jmi = None, np.nan
            print(f"  [JMI greedy]       SKIPPED (p > {JMI_MAX_P}, O(p^2) intractable)")

        # ---- Level 2: MI joint LOO ---------------------------------------------
        t0 = time.time()
        mi_full = compute_mi_joint(X_scaled, Y_scaled)
        mi_loo  = np.array([compute_mi_joint(np.delete(X_scaled, i, axis=1), Y_scaled)
                            for i in range(p)])
        mi_joint_imp   = mi_full - mi_loo
        mi_joint_ranks = rankdata(-mi_joint_imp).astype(int)
        t_mij = time.time() - t0
        mi_degenerate = bool(np.ptp(mi_joint_imp) < 1e-9)
        if mi_degenerate:
            print(f"  [MI joint LOO]     {t_mij:6.1f}s  "
                  f"WARNING: importance scores are flat (mi_full_clipped={mi_full:.4f}) — "
                  f"Kraskov estimator's negative bias likely clipped every fold to 0 at this "
                  f"dimensionality; ranking carries no signal.")
        else:
            print(f"  [MI joint LOO]     {t_mij:6.1f}s")

        # ---- Level 2: II joint LOO ----------------------------------------------
        t0 = time.time()
        ii_full = ii_joint(X_scaled)
        ii_loo  = np.array([ii_joint(np.delete(X_scaled, i, axis=1)) for i in range(p)])
        ii_joint_imp   = ii_loo - ii_full
        ii_joint_ranks = rankdata(-ii_joint_imp).astype(int)
        t_iij = time.time() - t0
        print(f"  [II joint LOO]     {t_iij:6.1f}s")

        # ---- Level 2: DII --------------------------------------------------------
        t0 = time.time()
        dii_l1_w, dii_l1_r, dii_l1_imbs = run_dii(X_scaled, Y_scaled, 0.10, f"DII+L1 p={p}")
        t_l1 = time.time() - t0
        print(f"  [DII + L1]         {t_l1:6.1f}s  (final II={dii_l1_imbs[-1]:.4f})")

        t0 = time.time()
        dii_nol1_w, dii_nol1_r, dii_nol1_imbs = run_dii(X_scaled, Y_scaled, 0.00, f"DII noL1 p={p}")
        t_nl1 = time.time() - t0
        print(f"  [DII no L1]        {t_nl1:6.1f}s  (final II={dii_nol1_imbs[-1]:.4f})")

        # ---- Evaluate --------------------------------------------------------
        final_imb = {'DII_L1': float(dii_l1_imbs[-1]), 'DII_noL1': float(dii_nol1_imbs[-1])}
        method_results = [
            ('MI_perfeat',  mi_ranks,        t_mi),
            ('II_perfeat',  ii_ranks,        t_ii),
            ('JMI_greedy',  jmi_ranks,       t_jmi),
            ('MI_joint',    mi_joint_ranks,  t_mij),
            ('II_joint',    ii_joint_ranks,  t_iij),
            ('DII_L1',      dii_l1_r,        t_l1),
            ('DII_noL1',    dii_nol1_r,      t_nl1),
        ]
        for name, ranks, runtime in method_results:
            if ranks is None:
                rows.append(dict(p=p, method=name, tau=np.nan, rho=np.nan,
                                  top3=np.nan, top5=np.nan, top10=np.nan, top16=np.nan,
                                  syn_rank_1=np.nan, syn_rank_2=np.nan, runtime_s=runtime,
                                  final_imbalance=np.nan,
                                  note='SKIPPED (p > JMI_MAX_P, O(p^2) cost)'))
                continue
            t_, r_ = tau_rho(gt_rank, ranks)
            note = ''
            if name == 'MI_joint' and mi_degenerate:
                note = 'DEGENERATE (Kraskov MI estimator collapsed to 0 for every LOO fold)'
            elif pd.isna(t_):
                note = 'DEGENERATE (tied/constant importance scores)'
            rows.append(dict(
                p=p, method=name, tau=t_, rho=r_,
                top3=topk(gt_binary,ranks,3),  top5=topk(gt_binary,ranks,5),
                top10=topk(gt_binary,ranks,10), top16=topk(gt_binary,ranks,16),
                syn_rank_1=int(ranks[syn_idx[0]]), syn_rank_2=int(ranks[syn_idx[1]]),
                runtime_s=runtime, final_imbalance=final_imb.get(name, np.nan),
                note=note,
            ))

        # per-p full rankings (useful for auditing individual runs)
        per_p_df = pd.DataFrame({
            'Feature': feature_names, 'Group': feature_groups,
            'Informative': gt_binary, 'GT_Rank': gt_rank,
            'MI_Rank': mi_ranks, 'II_Rank': ii_ranks,
            'JMI_Rank': jmi_ranks if jmi_ranks is not None else np.nan,
            'MI_joint_Rank': mi_joint_ranks, 'II_joint_Rank': ii_joint_ranks,
            'DII_L1_Rank': dii_l1_r, 'DII_noL1_Rank': dii_nol1_r,
        })
        per_p_df.to_csv(f'simulation_study_v6_highdim_p{p}_rankings.csv', index=False)

    print(f"\nTotal sweep time: {(time.time()-t_start_all)/60:.1f} min")

    # =============================================================================
    # 5. SAVE SCALABILITY SUMMARY
    # =============================================================================
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv('simulation_study_v6_highdim_scalability.csv', index=False)
    print("\nSaved: simulation_study_v6_highdim_scalability.csv")

    with open('simulation_study_v6_highdim_summary.txt', 'w') as f:
        f.write("SIMULATION STUDY v6 — HIGH-DIMENSIONAL SCALABILITY SWEEP\n")
        f.write("="*60 + "\n")
        f.write(f"N={N} | P_VALUES={P_VALUES} | JMI run only for p<={JMI_MAX_P}\n\n")
        for p in P_VALUES:
            sub = summary_df[summary_df.p == p]
            f.write(f"--- p={p} ---\n")
            for _, r in sub.iterrows():
                note_str = f"  [{r['note']}]" if r['note'] else ""
                if pd.notna(r['tau']):
                    f.write(f"  {r['method']:<12} tau={r['tau']:.3f}  "
                            f"top10={r['top10']:.2f}  "
                            f"syn=({r['syn_rank_1']:.0f},{r['syn_rank_2']:.0f})  "
                            f"runtime={r['runtime_s']:.1f}s{note_str}\n")
                else:
                    f.write(f"  {r['method']:<12} runtime={r['runtime_s']:.1f}s"
                            f"{note_str or '  [NaN, unexplained]'}\n")
            f.write("\n")
    print("Saved: simulation_study_v6_highdim_summary.txt")

    # =============================================================================
    # 6. PLOTS: tau vs p, runtime vs p, top-10 vs p, synergy rank vs p
    # =============================================================================
    methods_plot = ['MI_perfeat','II_perfeat','JMI_greedy','MI_joint','II_joint','DII_L1','DII_noL1']
    colors = {'MI_perfeat':'#3498db','II_perfeat':'#2ecc71','JMI_greedy':'#f39c12',
              'MI_joint':'#e67e22','II_joint':'#9b59b6','DII_L1':'#e74c3c','DII_noL1':'#8e44ad'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0,0]
    for m in methods_plot:
        sub = summary_df[summary_df.method == m]
        ax.plot(sub.p, sub.tau, 'o-', color=colors[m], label=m)
    ax.set_xlabel('Number of features (p)'); ax.set_ylabel("Kendall's tau")
    ax.set_title('Ranking accuracy vs dimensionality'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[0,1]
    for m in methods_plot:
        sub = summary_df[summary_df.method == m]
        ax.plot(sub.p, sub.runtime_s, 'o-', color=colors[m], label=m)
    ax.set_xlabel('Number of features (p)'); ax.set_ylabel('Runtime (s, log scale)')
    ax.set_yscale('log'); ax.set_title('Computational cost vs dimensionality')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1,0]
    for m in methods_plot:
        sub = summary_df[summary_df.method == m]
        ax.plot(sub.p, sub.top10, 'o-', color=colors[m], label=m)
    ax.set_xlabel('Number of features (p)'); ax.set_ylabel('Top-10 precision')
    ax.set_title('Top-10 precision vs dimensionality'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1,1]
    for m in methods_plot:
        sub = summary_df[summary_df.method == m]
        avg_syn_rank = (sub.syn_rank_1 + sub.syn_rank_2) / 2
        ax.plot(sub.p, avg_syn_rank, 'o-', color=colors[m], label=m)
    ax.set_xlabel('Number of features (p)'); ax.set_ylabel('Avg. rank of XOR synergy pair')
    ax.set_title('Synergy-pair detection vs dimensionality\n(lower = still correctly detected)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.suptitle('Simulation Study v6 — High-Dimensional Scalability Sweep', fontweight='bold')
    plt.tight_layout()
    plt.savefig('simulation_study_v6_highdim_results.png', dpi=300, bbox_inches='tight')
    print("Saved: simulation_study_v6_highdim_results.png")

    print("\n" + "="*70)
    print("SIMULATION STUDY v6 COMPLETE")
    print("="*70)

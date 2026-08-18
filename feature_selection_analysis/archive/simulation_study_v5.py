"""
Simulation Study v5: Two-Level Fair Comparison
===============================================
Addresses the asymmetry in v4 (per-feature MI/II vs joint DII).

Two-level framework:
  LEVEL 1 — Per-feature:  MI(Xi;Y)          vs  II(Xi→Y)
  LEVEL 2 — Joint:        JMI greedy
                           MI joint (LOO backward)
                           II joint (LOO backward)
                           DII (gradient optimisation)

JMI greedy (Yang & Moody 1999):
  Step 1: i₁ = argmax I(Xi; Y)
  Step k: iₖ = argmax I(Xi; Y | X_{S_{k-1}})   [conditional MI]
  CMI estimated via Frenzel & Pompe (2007) k-NN estimator.

MI joint backward LOO (Kraskov et al. 2004):
  importance_i = I(X_full; Y) − I(X_{-i}; Y)
  I(X;Y) estimated via Kraskov k-NN estimator for multivariate X.

II joint backward LOO:
  importance_i = II(X_{-i}→Y) − II(X_full→Y)

DII (Wild et al. 2025):
  Gradient descent optimisation of continuous feature weights.

Level 2 is fair: MI joint LOO and II joint LOO share identical
backward LOO structure; only the measure differs (MI vs II).

Same dataset as v4 (N=2000, 27 features) for direct comparability.
"""

import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial  import cKDTree
from scipy.special  import digamma
from scipy.stats    import rankdata, kendalltau, spearmanr
from sklearn.preprocessing      import StandardScaler
from sklearn.feature_selection  import mutual_info_regression
from sklearn.neighbors          import NearestNeighbors
from dadapy.diff_imbalance      import DiffImbalance

SEED  = 42
N     = 2000
K_CMI = 5          # k-NN neighbours for CMI / MI-joint estimators
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# =============================================================================
# 1. DATASET  (identical to v4 — do not change)
# =============================================================================
print("="*70)
print("SIMULATION STUDY v5 — Two-Level Fair Comparison")
print("Level 1 (per-feature): MI  vs  II")
print("Level 2 (joint):       JMI greedy | MI joint LOO | II joint LOO | DII")
print("="*70)
print("\n[1/9] Generating dataset (same as v4)...")

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
n_features = X.shape[1]
print(f"  Shape: {X.shape} | Informative: {int(gt_binary.sum())} / {n_features}")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = (Y - Y.mean()) / Y.std()

# =============================================================================
# 2. CMI ESTIMATOR  (Frenzel & Pompe 2007, k-NN, Chebyshev norm)
#    Used by JMI greedy for conditional MI.
# =============================================================================

def compute_cmi(xi_1d, y_1d, xc_2d, k=K_CMI):
    """
    Estimate I(xi ; y | xc) via Frenzel-Pompe (2007) k-NN estimator.
    If xc_2d has 0 columns → falls back to sklearn MI (no conditioning).
    """
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

    n_xz = count_nbrs(xz)
    n_yz = count_nbrs(yz)
    n_z  = count_nbrs(z)

    cmi = (digamma(k)
           + np.mean(digamma(np.maximum(n_z,  0.0) + 1))
           - np.mean(digamma(np.maximum(n_xz, 0.0) + 1))
           - np.mean(digamma(np.maximum(n_yz, 0.0) + 1)))
    return float(max(0.0, cmi))


# =============================================================================
# 3. MULTIVARIATE MI ESTIMATOR  (Kraskov et al. 2004, k-NN, Chebyshev norm)
#    Used by MI joint LOO for I(X_matrix ; Y).
# =============================================================================

def compute_mi_joint(X_mat, y_1d, k=K_CMI):
    """
    Estimate I(X; Y) for multivariate X via Kraskov et al. (2004) Algorithm 1.

    Formula:
      I(X;Y) = ψ(k) + ψ(N) − <ψ(n_X+1)> − <ψ(n_Y+1)>

    Distances are Chebyshev (L∞) norm.
    n_X[i] = # points within the k-th-neighbour radius (in the joint XY space)
             when projected onto X-space.
    n_Y[i] = same projected onto Y-space.
    """
    N = len(y_1d)
    Y = y_1d.reshape(-1, 1)
    XY = np.hstack([X_mat, Y])

    tree_XY = cKDTree(XY)
    dists, _ = tree_XY.query(XY, k=k+1, p=np.inf)
    eps = np.maximum(dists[:, k], 1e-15)

    tree_X = cKDTree(X_mat)
    tree_Y = cKDTree(Y)

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


# =============================================================================
# 4. JMI GREEDY  (Yang & Moody 1999)
# =============================================================================

def jmi_greedy(X_sc, y_sc, k=K_CMI):
    """
    Greedy JMI: at each step, selects the remaining feature with the highest
    I(Xi ; Y | already_selected), estimated via Frenzel-Pompe CMI.

    Returns ranks (1 = selected first), cmi_at_selection, selection_order.
    """
    N, d = X_sc.shape
    # Shuffle initial order so tie-breaking among CMI=0 features is random,
    # not an artifact of the original feature index order.
    rng_jmi   = np.random.default_rng(SEED)
    remaining = list(rng_jmi.permutation(d))
    selected  = []
    cmi_vals  = np.zeros(d)
    t0 = time.time()
    print(f"  k_CMI={k}  |  {d} greedy steps")

    for step in range(d):
        xc_2d = X_sc[:, selected] if selected else np.empty((N, 0))
        if xc_2d.ndim == 1:
            xc_2d = xc_2d.reshape(-1, 1)

        best_val  = -1.0
        best_feat = remaining[0]

        for fi in remaining:
            val = compute_cmi(X_sc[:, fi], y_sc, xc_2d, k=k)
            if val > best_val:
                best_val  = val
                best_feat = fi

        cmi_vals[best_feat] = best_val
        selected.append(best_feat)
        remaining.remove(best_feat)

        elapsed = time.time() - t0
        print(f"  step {step+1:2d}/{d}: {feature_names[best_feat]:<12} "
              f"CMI={best_val:.4f}  [{elapsed:5.0f}s]")

    ranks = np.zeros(d, dtype=int)
    for pos, fi in enumerate(selected):
        ranks[fi] = pos + 1

    return ranks, cmi_vals, selected


# =============================================================================
# 5. LEVEL 1 — MI per-feature
# =============================================================================
print("\n[2/9] Level 1 — MI per-feature...")
mi_scores = mutual_info_regression(X_scaled, Y_scaled, random_state=SEED)
mi_ranks  = rankdata(-mi_scores).astype(int)

# =============================================================================
# 6. LEVEL 1 — II per-feature
# =============================================================================
print("\n[3/9] Level 1 — II per-feature...")

def compute_ii_pf(xi, y):
    N  = len(xi)
    xi = xi.reshape(-1,1); y = y.reshape(-1,1)
    dx = np.abs(xi-xi.T); np.fill_diagonal(dx, np.inf)
    dy = np.abs(y -y.T);  np.fill_diagonal(dy, np.inf)
    nn = np.argmin(dx, axis=1)
    ry = np.argsort(np.argsort(dy, axis=1), axis=1)
    return 2.0/N**2 * np.sum(ry[np.arange(N), nn])

ii_scores = np.array([compute_ii_pf(X_scaled[:,i], Y_scaled)
                      for i in range(n_features)])
ii_ranks  = rankdata(ii_scores).astype(int)

# =============================================================================
# 7. LEVEL 2 — JMI greedy
# =============================================================================
print("\n[4/9] Level 2 — JMI greedy (Yang & Moody 1999 + Frenzel-Pompe CMI)...")
print("  Estimated run time: ~20–60s for N=2000, 27 features.")
t_jmi = time.time()
jmi_ranks, jmi_cmi, jmi_order = jmi_greedy(X_scaled, Y_scaled, k=K_CMI)
print(f"  Total JMI time: {time.time()-t_jmi:.1f}s")

# =============================================================================
# 8. LEVEL 2 — MI joint backward LOO  (Kraskov 2004)
#
#   importance_i = I(X_full; Y) − I(X_{-i}; Y)
#   Positive importance → removing feature i hurts MI → feature is informative.
#   Same backward-LOO structure as II joint LOO.
# =============================================================================
print("\n[5/9] Level 2 — MI joint backward LOO (Kraskov 2004 k-NN estimator)...")
t_mij = time.time()
mi_full_jt = compute_mi_joint(X_scaled, Y_scaled)
print(f"  MI_full = {mi_full_jt:.5f}")
mi_jt_loo = np.array([
    compute_mi_joint(np.delete(X_scaled, i, axis=1), Y_scaled)
    for i in range(n_features)
])
mi_joint_imp   = mi_full_jt - mi_jt_loo      # drop when feature i is removed
mi_joint_ranks = rankdata(-mi_joint_imp).astype(int)
print(f"  MI joint LOO time: {time.time()-t_mij:.1f}s")

# =============================================================================
# 9. LEVEL 2 — II joint backward LOO
# =============================================================================
print("\n[6/9] Level 2 — II joint backward LOO...")

dist_Y_mat  = np.abs(Y_scaled[:,None] - Y_scaled[None,:])
np.fill_diagonal(dist_Y_mat, np.inf)
Y_ranks_mat = np.argsort(np.argsort(dist_Y_mat, axis=1), axis=1)

def ii_joint(X_sub):
    N = len(X_sub)
    nb = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X_sub)
    _, idx = nb.kneighbors(X_sub)
    return 2.0/N**2 * float(np.sum(Y_ranks_mat[np.arange(N), idx[:,1]]))

ii_full = ii_joint(X_scaled)
print(f"  II_full = {ii_full:.5f}")
ii_loo = np.array([ii_joint(np.delete(X_scaled, i, axis=1))
                   for i in range(n_features)])
ii_joint_imp   = ii_loo - ii_full
ii_joint_ranks = rankdata(-ii_joint_imp).astype(int)

# =============================================================================
# 10. LEVEL 2 — DII
# =============================================================================
k_init  = max(5, int(0.025*N))
k_final = max(1, int(0.010*N))

def run_dii(l1, label):
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

print("\n[7/9] Level 2 — DII + L1 (λ=0.1)...")
dii_l1_w,   dii_l1_r,   imbs_l1   = run_dii(0.10, "DII + L1")
print("\n[8/9] Level 2 — DII no L1...")
dii_nol1_w, dii_nol1_r, imbs_nol1 = run_dii(0.00, "DII no L1")

# =============================================================================
# 11. EVALUATION
# =============================================================================
print("\n[9/9] Evaluating...")

def tau_rho(ranks):
    t,_ = kendalltau(gt_rank, ranks)
    r,_ = spearmanr(gt_rank,  ranks)
    return t, r

def topk(ranks, k):
    return float(gt_binary[ranks <= k].sum()) / k

tau_mi,  rho_mi  = tau_rho(mi_ranks)
tau_ii,  rho_ii  = tau_rho(ii_ranks)
tau_jmi, rho_jmi = tau_rho(jmi_ranks)
tau_mij, rho_mij = tau_rho(mi_joint_ranks)
tau_iij, rho_iij = tau_rho(ii_joint_ranks)
tau_l1,  rho_l1  = tau_rho(dii_l1_r)
tau_nl1, rho_nl1 = tau_rho(dii_nol1_r)

ks      = [3, 5, 10, 16]
syn_idx = [np.where(feature_names==f)[0][0] for f in ['x_syn_1','x_syn_2']]
dup_idx = [np.where(feature_names==f)[0][0] for f in
           ['x_base','x_dup_1','x_dup_2','x_dup_3','x_dup_4']]

results = pd.DataFrame({
    'Feature'      : feature_names,
    'Group'        : feature_groups,
    'Informative'  : gt_binary,
    'GT_Rank'      : gt_rank,
    'MI_Score'     : mi_scores,     'MI_Rank'       : mi_ranks,
    'II_pf_Score'  : ii_scores,     'II_pf_Rank'    : ii_ranks,
    'JMI_CMI'      : jmi_cmi,       'JMI_Rank'      : jmi_ranks,
    'MI_jt_Imp'    : mi_joint_imp,  'MI_jt_Rank'    : mi_joint_ranks,
    'II_jt_Imp'    : ii_joint_imp,  'II_jt_Rank'    : ii_joint_ranks,
    'DII_L1_W'     : dii_l1_w,      'DII_L1_Rank'   : dii_l1_r,
    'DII_noL1_W'   : dii_nol1_w,    'DII_noL1_Rank' : dii_nol1_r,
})
results.to_csv('simulation_study_v5_rankings.csv', index=False)

# ---------- console output -----------------------------------------------
print("\n"+"="*70)
print("RESULTS — TWO-LEVEL FAIR COMPARISON")
print("="*70)
print(f"\n  {'Method':<26} {'Kendall τ':>10} {'Spearman ρ':>12}")
print(f"  {'-'*50}")
print("  ─── Level 1 (per-feature) ───────────────────────────────")
for nm,t,r in [('MI  (per-feat.)', tau_mi, rho_mi),
               ('II  (per-feat.)', tau_ii, rho_ii)]:
    print(f"  {nm:<26} {t:>10.3f} {r:>12.3f}")
print("  ─── Level 2 (joint) ─────────────────────────────────────")
for nm,t,r in [('JMI greedy',        tau_jmi, rho_jmi),
               ('MI  joint (LOO)',   tau_mij, rho_mij),
               ('II  joint (LOO)',   tau_iij, rho_iij),
               ('DII + L1',          tau_l1,  rho_l1),
               ('DII no L1',         tau_nl1, rho_nl1)]:
    print(f"  {nm:<26} {t:>10.3f} {r:>12.3f}")

print(f"\n  {'Method':<26}"+"".join(f"  Top-{k:2d}" for k in ks))
print("  "+"-"*54)
for nm,rk in [('MI  (per-feat.)',  mi_ranks),
              ('II  (per-feat.)',  ii_ranks),
              ('JMI greedy',       jmi_ranks),
              ('MI  joint (LOO)',  mi_joint_ranks),
              ('II  joint (LOO)',  ii_joint_ranks),
              ('DII + L1',         dii_l1_r),
              ('DII no L1',        dii_nol1_r)]:
    print(f"  {nm:<26}"+"".join(f"  {topk(rk,k):6.2f}" for k in ks))

print(f"\n  Synergistic pair — ranks (lower = more important):")
print(f"  {'Feat':<10}{'MI':>5}{'II_pf':>6}{'JMI':>5}"
      f"{'MI_jt':>6}{'II_jt':>6}{'DII+L1':>8}{'DII_nL1':>9}")
for i in syn_idx:
    print(f"  {feature_names[i]:<10}{mi_ranks[i]:>5}{ii_ranks[i]:>6}"
          f"{jmi_ranks[i]:>5}{mi_joint_ranks[i]:>6}{ii_joint_ranks[i]:>6}"
          f"{dii_l1_r[i]:>8}{dii_nol1_r[i]:>9}")

# =============================================================================
# 12. PLOTS
# =============================================================================
grp_colors = {
    'Linear':'#3498db','NonlinSym':'#2ecc71','Directional':'#e74c3c',
    'Cyclical':'#f39c12','Interaction':'#9b59b6','Redundant':'#1abc9c',
    'Derived':'#e67e22','Synergistic':'#e91e63','NearDup':'#795548',
    'Noise':'#bdc3c7',
}

# 7 methods (include DII no L1 for completeness)
method_cfg = [
    ('MI (per-feat.)',  mi_ranks,        tau_mi,  rho_mi,  '#3498db'),
    ('II (per-feat.)',  ii_ranks,        tau_ii,  rho_ii,  '#2ecc71'),
    ('JMI greedy',      jmi_ranks,       tau_jmi, rho_jmi, '#f39c12'),
    ('MI joint (LOO)',  mi_joint_ranks,  tau_mij, rho_mij, '#e67e22'),
    ('II joint (LOO)',  ii_joint_ranks,  tau_iij, rho_iij, '#9b59b6'),
    ('DII + L1',        dii_l1_r,        tau_l1,  rho_l1,  '#e74c3c'),
    ('DII no L1',       dii_nol1_r,      tau_nl1, rho_nl1, '#8e44ad'),
]
labels7 = ['MI\n(pf)','II\n(pf)','JMI\ngreedy','MI\njoint','II\njoint',
           'DII\n+L1','DII\nnL1']
colors7 = [c for *_,c in method_cfg]

fig = plt.figure(figsize=(28, 16))
gs  = gridspec.GridSpec(3, 7, figure=fig, hspace=0.5, wspace=0.4)

# Row 0: rank-scatter for 7 methods
for col,(nm,mranks,t,r,_) in enumerate(method_cfg):
    ax = fig.add_subplot(gs[0, col])
    for idx,grp in enumerate(feature_groups):
        ax.scatter(gt_rank[idx], mranks[idx], color=grp_colors[grp],
                   s=35, edgecolors='black', linewidth=0.3, zorder=3)
    ax.plot([1,n_features],[1,n_features],'k--',alpha=0.3,lw=1)
    ax.set_title(f'{nm}\nτ={t:.3f}  ρ={r:.3f}', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('GT Rank', fontsize=7); ax.set_ylabel('Method Rank', fontsize=7)
    ax.tick_params(labelsize=6)

# Row 1-left: Kendall τ bar chart with level separator
ax_tau = fig.add_subplot(gs[1, 0:2])
taus7 = [tau_mi, tau_ii, tau_jmi, tau_mij, tau_iij, tau_l1, tau_nl1]
bars = ax_tau.bar(labels7, taus7, color=colors7, edgecolor='black', linewidth=0.5)
for bar,v in zip(bars,taus7):
    ax_tau.text(bar.get_x()+bar.get_width()/2, v+0.005, f'{v:.3f}',
                ha='center', va='bottom', fontsize=7, fontweight='bold')
ax_tau.axvline(1.5, color='grey', ls='--', lw=1.5)
ax_tau.text(0.5, max(taus7)*1.07, 'Level 1\nper-feature', ha='center',
            fontsize=7, color='grey')
ax_tau.text(4.0, max(taus7)*1.07, 'Level 2\njoint', ha='center',
            fontsize=7, color='grey')
ax_tau.set_ylabel('Kendall τ')
ax_tau.set_title('Rank correlation τ by method', fontweight='bold')
ax_tau.grid(True, alpha=0.3, axis='y')

# Row 1-centre: Top-K bar chart
ax_topk = fig.add_subplot(gs[1, 2:6])
xp = np.arange(len(ks)); bw = 0.11
for k_off,(nm,rk,*_) in enumerate(method_cfg):
    ax_topk.bar(xp+(k_off-3)*bw,
                [topk(rk,k) for k in ks],
                bw, label=labels7[k_off], color=colors7[k_off],
                edgecolor='black', linewidth=0.3)
ax_topk.axhline(gt_binary.mean(), color='k', ls=':', lw=1, label='Baseline')
ax_topk.set_xticks(xp); ax_topk.set_xticklabels([f'Top-{k}' for k in ks])
ax_topk.set_ylabel('Precision'); ax_topk.set_ylim(0,1.15)
ax_topk.set_title('Top-K Precision — all methods', fontweight='bold')
ax_topk.legend(fontsize=7, ncol=4); ax_topk.grid(True, alpha=0.3, axis='y')

# Row 1-right: DII training curve
ax_conv = fig.add_subplot(gs[1, 6])
ax_conv.plot(imbs_l1,   color='#e74c3c', lw=1.5, label='DII+L1')
ax_conv.plot(imbs_nol1, color='#8e44ad', lw=1.5, label='DII noL1', ls='--')
ax_conv.set_xlabel('Epoch', fontsize=8); ax_conv.set_ylabel('II value', fontsize=8)
ax_conv.set_title('DII convergence', fontweight='bold')
ax_conv.legend(fontsize=8); ax_conv.grid(True, alpha=0.3)

# Row 2-left: Near-duplicate weights
ax_dup = fig.add_subplot(gs[2, 0:2])
dup_names = [feature_names[i] for i in dup_idx]
xp2 = np.arange(len(dup_idx)); bw2 = 0.35
ax_dup.bar(xp2-bw2/2, dii_l1_w[dup_idx],   bw2, label='DII+L1',
           color='#e74c3c', edgecolor='black', linewidth=0.5)
ax_dup.bar(xp2+bw2/2, dii_nol1_w[dup_idx], bw2, label='DII noL1',
           color='#8e44ad', edgecolor='black', linewidth=0.5)
ax_dup.set_xticks(xp2); ax_dup.set_xticklabels(dup_names, fontsize=9)
ax_dup.set_ylabel('DII weight')
ax_dup.set_title('Near-duplicate weights\n(L1 concentrates, no-L1 spreads)',
                 fontweight='bold')
ax_dup.legend(); ax_dup.grid(True, alpha=0.3, axis='y')

# Row 2-centre: Synergistic pair ranks — all 7 methods
ax_syn = fig.add_subplot(gs[2, 2:6])
syn_ranks_all = np.array([
    [mi_ranks[i], ii_ranks[i], jmi_ranks[i],
     mi_joint_ranks[i], ii_joint_ranks[i],
     dii_l1_r[i], dii_nol1_r[i]]
    for i in syn_idx
])  # (2, 7)
xp3 = np.arange(2); bw3 = 0.11
for k_off,(lb,col) in enumerate(zip(labels7, colors7)):
    ax_syn.bar(xp3+(k_off-3)*bw3, syn_ranks_all[:,k_off], bw3,
               label=lb, color=col, edgecolor='black', linewidth=0.3)
ax_syn.axhline(n_features/2, color='k', ls='--', lw=1, alpha=0.5)
ax_syn.set_xticks(xp3); ax_syn.set_xticklabels(['x_syn_1','x_syn_2'])
ax_syn.set_ylabel('Assigned rank (lower = more important)')
ax_syn.set_title('Synergistic pair (XOR)\n(joint methods should rank these LOW)',
                 fontweight='bold')
ax_syn.legend(fontsize=7, ncol=4); ax_syn.grid(True, alpha=0.3, axis='y')

# Row 2-right: JMI greedy CMI profile
ax_jmi = fig.add_subplot(gs[2, 6])
step_cmi = [jmi_cmi[jmi_order[s]] for s in range(n_features)]
ax_jmi.plot(range(1, n_features+1), step_cmi, 'o-', color='#f39c12',
            lw=1.5, ms=4)
ax_jmi.axvline(int(gt_binary.sum())+0.5, color='red', ls='--', lw=1,
               alpha=0.7, label=f'{int(gt_binary.sum())} informative')
ax_jmi.set_xlabel('Selection step'); ax_jmi.set_ylabel('CMI at selection')
ax_jmi.set_title('JMI greedy\nCMI profile', fontweight='bold')
ax_jmi.legend(fontsize=7); ax_jmi.grid(True, alpha=0.3)

plt.suptitle(
    'Simulation Study v5 — Two-Level Fair Comparison  |  N=2000, 27 features\n'
    'Level 1 (per-feature): MI vs II  ·  '
    'Level 2 (joint): JMI greedy | MI joint LOO | II joint LOO | DII',
    fontsize=11, fontweight='bold', y=1.01)
plt.savefig('simulation_study_v5_results.png', dpi=300, bbox_inches='tight')
print("Saved: simulation_study_v5_results.png")

# =============================================================================
# 13. SUMMARY FILE
# =============================================================================
syn_str = lambda ranks: (f"x_syn_1={ranks[syn_idx[0]]}, "
                         f"x_syn_2={ranks[syn_idx[1]]}")

summary = f"""
SIMULATION STUDY v5 — TWO-LEVEL FAIR COMPARISON
=================================================
N={N} | {n_features} features | {int(gt_binary.sum())} informative

LEVEL 1 — Per-feature (single variable vs target):
  MI  (per-feat.):   τ={tau_mi:.3f}  ρ={rho_mi:.3f}
  II  (per-feat.):   τ={tau_ii:.3f}  ρ={rho_ii:.3f}

LEVEL 2 — Joint (all features used simultaneously):
  JMI greedy:        τ={tau_jmi:.3f}  ρ={rho_jmi:.3f}   [forward greedy, cond. MI]
  MI  joint (LOO):   τ={tau_mij:.3f}  ρ={rho_mij:.3f}   [backward LOO, Kraskov k-NN]
  II  joint (LOO):   τ={tau_iij:.3f}  ρ={rho_iij:.3f}   [backward LOO, k-NN ranks]
  DII + L1 (λ=0.1):  τ={tau_l1:.3f}  ρ={rho_l1:.3f}   [global gradient optimisation]
  DII no L1:         τ={tau_nl1:.3f}  ρ={rho_nl1:.3f}

Top-K Precision:
  Method                   Top-3  Top-5  Top-10  Top-16
  MI  (per-feat.)          {topk(mi_ranks,3):.2f}   {topk(mi_ranks,5):.2f}   {topk(mi_ranks,10):.2f}    {topk(mi_ranks,16):.2f}
  II  (per-feat.)          {topk(ii_ranks,3):.2f}   {topk(ii_ranks,5):.2f}   {topk(ii_ranks,10):.2f}    {topk(ii_ranks,16):.2f}
  JMI greedy               {topk(jmi_ranks,3):.2f}   {topk(jmi_ranks,5):.2f}   {topk(jmi_ranks,10):.2f}    {topk(jmi_ranks,16):.2f}
  MI  joint (LOO)          {topk(mi_joint_ranks,3):.2f}   {topk(mi_joint_ranks,5):.2f}   {topk(mi_joint_ranks,10):.2f}    {topk(mi_joint_ranks,16):.2f}
  II  joint (LOO)          {topk(ii_joint_ranks,3):.2f}   {topk(ii_joint_ranks,5):.2f}   {topk(ii_joint_ranks,10):.2f}    {topk(ii_joint_ranks,16):.2f}
  DII + L1                 {topk(dii_l1_r,3):.2f}   {topk(dii_l1_r,5):.2f}   {topk(dii_l1_r,10):.2f}    {topk(dii_l1_r,16):.2f}
  DII no L1                {topk(dii_nol1_r,3):.2f}   {topk(dii_nol1_r,5):.2f}   {topk(dii_nol1_r,10):.2f}    {topk(dii_nol1_r,16):.2f}
  Baseline                 {gt_binary.mean():.2f}

Synergistic pair (XOR):
  MI  (per-feat.):   {syn_str(mi_ranks)}
  II  (per-feat.):   {syn_str(ii_ranks)}
  JMI greedy:        {syn_str(jmi_ranks)}
  MI  joint (LOO):   {syn_str(mi_joint_ranks)}
  II  joint (LOO):   {syn_str(ii_joint_ranks)}
  DII + L1:          {syn_str(dii_l1_r)}
  DII no L1:         {syn_str(dii_nol1_r)}

Near-duplicate DII weights (x_base + 4 copies):
""" + "".join(
    f"  {feature_names[i]:<12} DII+L1={dii_l1_w[i]:.4f}  DII_noL1={dii_nol1_w[i]:.4f}\n"
    for i in dup_idx)

with open('simulation_study_v5_summary.txt','w') as f:
    f.write(summary)

print("Saved: simulation_study_v5_summary.txt")
print("Saved: simulation_study_v5_rankings.csv")
print("\n"+"="*70)
print("SIMULATION STUDY v5 COMPLETE")
print("="*70)

"""
Simulation Study v4: MI vs II vs DII (full DADApy)
===================================================
Extends v3 with two new feature groups suggested by Daniele:

  GROUP H — Synergistic / XOR interactions
    Two features Z4, Z5 drawn independently. Each is individually
    uncorrelated with Y (MI ≈ 0 per feature alone), but JOINTLY they
    predict Y via  Y += 0.80 * sign(Z4 * Z5).
    Per-feature methods (MI, II) cannot detect this.
    DII joint optimisation should assign weight to BOTH.

  GROUP I — Near-duplicates  (LASSO test)
    A base signal X_base = Z6 + tiny noise.
    Four near-copies: X_dup_i = X_base + tiny noise.
    Y += 1.20 * X_base.
    Without L1: DII spreads weights across all 5 copies.
    With    L1: DII concentrates weight on one (sparsity).
    MI / II rank all 5 equally high (cannot detect redundancy).

All methods use target_y as reference space (fair comparison).
Reference: Glielmo et al. PNAS Nexus 2022 (II), Wild et al. Nat. Comm. 2025 (DII).
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, kendalltau, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from dadapy.diff_imbalance import DiffImbalance

# =============================================================================
# 1. DATASET
# =============================================================================
SEED = 42
N    = 2000
np.random.seed(SEED)
rng  = np.random.default_rng(SEED)

print("="*70)
print("SIMULATION STUDY v4: MI vs II vs DII (full DADApy)")
print("Additions: synergistic XOR group + near-duplicate LASSO group")
print("="*70)
print("\n[1/6] Generating dataset...")

t = np.arange(N)
cycle_fast = np.sin(2 * np.pi * t / 50)
cycle_slow = np.sin(2 * np.pi * t / 200)
eps = rng.normal(0, 1, N)

# Latent AR processes (z1, z2, z3 — same as v3)
z1 = np.zeros(N); z2 = np.zeros(N); z3 = np.zeros(N)
for i in range(1, N):
    z1[i] =  0.72 * z1[i-1] + 0.55 * eps[i]      + rng.normal(0, 0.25)
    z2[i] =  0.45 * z2[i-1] + 0.30 * z1[i-1]     + rng.normal(0, 0.35)
    z3[i] = -0.30 * z3[i-1] + 0.25 * z2[i-1]     + rng.normal(0, 0.30)

# New independent latent signals for the two new groups
z4 = rng.normal(0, 1, N)   # synergistic signal 1 (independent of z1-z3)
z5 = rng.normal(0, 1, N)   # synergistic signal 2 (independent of z1-z3)
z6 = np.zeros(N)            # near-duplicate base signal (AR)
for i in range(1, N):
    z6[i] = 0.65 * z6[i-1] + rng.normal(0, 0.50)

vol = 0.25 + 0.18 * np.abs(cycle_slow) + 0.12 * (z1**2 / (1 + z1**2))

# ---- Group A: Linear -------------------------------------------------------
x_lin_1 = z1                          + rng.normal(0, 0.10, N)
x_lin_2 = 0.8 * z2                    + rng.normal(0, 0.10, N)

# ---- Group B: Nonlinear symmetric ------------------------------------------
x_sym_1 = z1**2                        + rng.normal(0, 0.08, N)
x_sym_2 = np.abs(z2)                   + rng.normal(0, 0.08, N)

# ---- Group C: Directional --------------------------------------------------
x_dir_1 = np.maximum(z1, 0)           + rng.normal(0, 0.06, N)
x_dir_2 = np.maximum(-z2, 0)          + rng.normal(0, 0.06, N)

# ---- Group D: Cyclical (wave_2 NOT in target) ------------------------------
x_wave_1 = cycle_fast                  + rng.normal(0, 0.05, N)
x_wave_2 = np.cos(2*np.pi*t/50)       + rng.normal(0, 0.05, N)

# ---- Group E: Interactions -------------------------------------------------
x_mix_1  = z1 * z2                     + rng.normal(0, 0.08, N)
x_mix_2  = z1 / (1 + np.abs(z3))      + rng.normal(0, 0.08, N)  # NOT in target

# ---- Group F: Redundant (correlated with informative features) -------------
x_red_1  = x_lin_1                     + rng.normal(0, 0.04, N)
x_red_2  = x_sym_1                     + rng.normal(0, 0.04, N)
x_red_3  = 0.5*x_dir_1 + 0.5*x_mix_1  + rng.normal(0, 0.05, N)

# ---- Group G: Derived time-series ------------------------------------------
regime   = (cycle_slow > 0).astype(float)
momentum = pd.Series(z1).rolling(10, min_periods=1).mean().to_numpy()
rvp      = (pd.Series(np.abs(np.diff(np.r_[0.0, z1])))
              .rolling(15, min_periods=1).mean().to_numpy())

# ---- Group H: SYNERGISTIC / XOR  (NEW — Daniele's suggestion) -------------
# X_syn_1 and X_syn_2 are individually uncorrelated with Y.
# Together they determine Y through sign(Z4 * Z5):
#   if Z4 and Z5 have the same sign → positive contribution
#   if they have opposite signs     → negative contribution
# Per-feature methods (MI, II) assign them rank ≈ last.
# DII joint optimisation should give weight to BOTH simultaneously.
x_syn_1 = z4                           + rng.normal(0, 0.10, N)
x_syn_2 = z5                           + rng.normal(0, 0.10, N)
xor_signal = np.sign(z4 * z5)          # ∈ {-1, +1}

# ---- Group I: NEAR-DUPLICATES  (NEW — Daniele's suggestion) ---------------
# X_base is a clean version of z6.  The four copies add only tiny noise.
# Without L1: DII spreads weights across all 5 (~0.20 each).
# With    L1: DII concentrates on one (~1.0) and zeros the rest.
x_base  = z6                           + rng.normal(0, 0.05, N)
x_dup_1 = x_base                       + rng.normal(0, 0.02, N)
x_dup_2 = x_base                       + rng.normal(0, 0.02, N)
x_dup_3 = x_base                       + rng.normal(0, 0.03, N)
x_dup_4 = x_base                       + rng.normal(0, 0.03, N)

# ---- Group J: Pure noise ---------------------------------------------------
noise_1 = rng.normal(0, 1, N); noise_2 = rng.normal(0, 1, N)
noise_3 = rng.normal(0, 1, N); noise_4 = rng.normal(0, 1, N)

# ---- Target Y (known formula) ----------------------------------------------
Y = (
      0.90 * x_lin_1
    - 0.70 * x_lin_2
    + 0.85 * (z1**2 - np.mean(z1**2))
    + 0.65 * np.abs(z2)
    + 1.10 * np.maximum(z1, 0)
    - 0.95 * np.maximum(-z2, 0)
    + 0.55 * (z1 * z2)
    + 0.35 * cycle_fast
    + 0.20 * regime
    + 0.80 * xor_signal          # synergistic: needs BOTH z4 AND z5
    + 1.20 * x_base               # near-duplicate base signal
    + rng.normal(0, vol, N)
)

# ---- Feature matrix & metadata ---------------------------------------------
feature_names = [
    # Group A
    'x_lin_1',   'x_lin_2',
    # Group B
    'x_sym_1',   'x_sym_2',
    # Group C
    'x_dir_1',   'x_dir_2',
    # Group D
    'x_wave_1',  'x_wave_2',
    # Group E
    'x_mix_1',   'x_mix_2',
    # Group F
    'x_red_1',   'x_red_2',   'x_red_3',
    # Group G
    'momentum',  'rvp',       'regime',
    # Group H — synergistic
    'x_syn_1',   'x_syn_2',
    # Group I — near-duplicates
    'x_base',    'x_dup_1',   'x_dup_2',   'x_dup_3',   'x_dup_4',
    # Group J
    'noise_1',   'noise_2',   'noise_3',   'noise_4',
]
feature_groups = (
    ['Linear']*2 + ['Nonlin_Sym']*2 + ['Directional']*2 +
    ['Cyclical']*2 + ['Interaction']*2 + ['Redundant']*3 +
    ['Derived']*3 + ['Synergistic']*2 +
    ['NearDuplicate']*5 + ['Noise']*4
)
# 1 = directly informative (contributes to Y)
gt_binary = np.array([
    1, 1,          # A
    1, 1,          # B
    1, 1,          # C
    1, 0,          # D (wave_2 NOT in target)
    1, 0,          # E (mix_2 NOT in target)
    1, 1, 1,       # F (redundant but correlated)
    0, 0, 1,       # G (momentum, rvp not in target; regime yes)
    1, 1,          # H (synergistic — both needed)
    1, 1, 1, 1, 1, # I (near-duplicates — all correlated with x_base)
    0, 0, 0, 0,    # J
])

# GT rank by |coef| contribution to Y:
# x_base(1.20)>x_dir_1(1.10)>x_dir_2(0.95)>x_lin_1(0.90)>x_sym_1(0.85)>
# x_lin_2(0.70)>x_sym_2(0.65)>x_mix_1(0.55)>x_wave_1(0.35)>regime(0.20)>
# xor(0.80 split→ x_syn_1~rank11, x_syn_2~rank12)>
# redundant(10,13,14)>near-dups(15-18)>not-in-target>noise
gt_rank = np.array([
    4,  6,          # A: lin_1(0.90→4), lin_2(0.70→6)
    5,  7,          # B: sym_1(0.85→5), sym_2(0.65→7)
    2,  3,          # C: dir_1(1.10→2), dir_2(0.95→3)
    9,  20,         # D: wave_1(0.35→9), wave_2(not→20)
    8,  21,         # E: mix_1(0.55→8), mix_2(not→21)
    10, 13, 14,     # F: redundant
    22, 23, 10,     # G: momentum(not), rvp(not), regime(0.20→10) — fix below
    11, 12,         # H: syn_1, syn_2 (XOR, each ~0.40 effective)
    1,  15, 16, 17, 18,  # I: base(1.20→1), dups
    24, 25, 26, 27, # J: noise
])
# fix regime (index 15) rank — it should be 10
# re-assign ranks cleanly to avoid duplicates
gt_rank = np.array([
     4,  6,         # lin_1, lin_2
     5,  7,         # sym_1, sym_2
     2,  3,         # dir_1, dir_2
     9, 20,         # wave_1, wave_2
     8, 21,         # mix_1, mix_2
    13, 14, 15,     # red_1, red_2, red_3
    22, 23, 10,     # momentum, rvp, regime
    11, 12,         # syn_1, syn_2
     1, 16, 17, 18, 19,  # base, dup_1..4
    24, 25, 26, 27, # noise
])

X = np.column_stack([
    x_lin_1, x_lin_2,
    x_sym_1, x_sym_2,
    x_dir_1, x_dir_2,
    x_wave_1, x_wave_2,
    x_mix_1, x_mix_2,
    x_red_1, x_red_2, x_red_3,
    momentum, rvp, regime,
    x_syn_1, x_syn_2,
    x_base, x_dup_1, x_dup_2, x_dup_3, x_dup_4,
    noise_1, noise_2, noise_3, noise_4,
])
n_features = X.shape[1]

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = (Y - Y.mean()) / Y.std()

print(f"  Shape: {X_scaled.shape} | Informative: {gt_binary.sum()} / {n_features}")
print(f"  Groups: Linear, Nonlin_Sym, Directional, Cyclical, Interaction,")
print(f"          Redundant, Derived, Synergistic(NEW), NearDuplicate(NEW), Noise")

# =============================================================================
# 2. MUTUAL INFORMATION
# =============================================================================
print("\n[2/6] Computing Mutual Information...")
mi_scores = mutual_info_regression(X_scaled, Y_scaled, n_neighbors=5, random_state=SEED)
mi_ranks  = rankdata(-mi_scores).astype(int)

# =============================================================================
# 3. INFORMATION IMBALANCE (per-feature)
# =============================================================================
print("\n[3/6] Computing Information Imbalance (per-feature)...")

def compute_ii(X, y_ref):
    n = X.shape[0]
    dist_Y = squareform(pdist(y_ref.reshape(-1, 1)))
    np.fill_diagonal(dist_Y, dist_Y.max() + 1.0)
    ranks_Y = rankdata(dist_Y, method='average', axis=1).astype(int)
    scores  = np.empty(X.shape[1])
    for i in range(X.shape[1]):
        dist_X = squareform(pdist(X[:, i:i+1]))
        np.fill_diagonal(dist_X, dist_X.max() + 1.0)
        scores[i] = 2.0 * sum(
            ranks_Y[j, np.argmin(dist_X[j])] for j in range(n)
        ) / n**2
    return scores

ii_scores = compute_ii(X_scaled, Y_scaled)
ii_ranks  = rankdata(ii_scores).astype(int)

# =============================================================================
# 4 & 5. DII — full DADApy — with and without L1
# =============================================================================
k_init  = min(100, max(5, int(0.05 * N)))
k_final = max(1,   int(0.01 * N))

def run_dii(l1, label):
    model = DiffImbalance(
        data_A              = X_scaled.astype(np.float64),
        data_B              = Y_scaled.reshape(-1, 1).astype(np.float64),
        num_epochs          = 300,
        batches_per_epoch   = 1,
        seed                = SEED,
        l1_strength         = l1,
        point_adapt_lambda  = True,
        k_init              = k_init,
        k_final             = k_final,
        lambda_factor       = 0.1,
        optimizer_name      = 'adam',
        learning_rate       = 1e-2,
        learning_rate_decay = 'cos',
    )
    _, imbs = model.train(bar_label=label)
    w = np.array(model.params_final)
    return w, rankdata(-w).astype(int), imbs

print("\n[4/6] DII with L1 (l1=0.1)...")
dii_l1_w,   dii_l1_r,   imbs_l1   = run_dii(0.10, "DII + L1")
print("\n[5/6] DII without L1...")
dii_nol1_w, dii_nol1_r, imbs_nol1 = run_dii(0.00, "DII no L1")

# =============================================================================
# 6. EVALUATE
# =============================================================================
print("\n[6/6] Evaluating...")

results = pd.DataFrame({
    'Feature'       : feature_names,
    'Group'         : feature_groups,
    'Informative'   : gt_binary,
    'GT_Rank'       : gt_rank,
    'MI_Score'      : mi_scores,    'MI_Rank'       : mi_ranks,
    'II_Score'      : ii_scores,    'II_Rank'       : ii_ranks,
    'DII_L1_W'      : dii_l1_w,    'DII_L1_Rank'   : dii_l1_r,
    'DII_noL1_W'    : dii_nol1_w,  'DII_noL1_Rank' : dii_nol1_r,
})

def tau_rho(ranks):
    t, _ = kendalltau(gt_rank, ranks)
    r, _ = spearmanr(gt_rank,  ranks)
    return t, r

def topk(ranks, k):
    return gt_binary[np.where(ranks <= k)[0]].sum() / k

tau_mi,  rho_mi  = tau_rho(mi_ranks)
tau_ii,  rho_ii  = tau_rho(ii_ranks)
tau_l1,  rho_l1  = tau_rho(dii_l1_r)
tau_nl1, rho_nl1 = tau_rho(dii_nol1_r)

# =============================================================================
# PRINT RESULTS
# =============================================================================
print("\n" + "="*70)
print("RANK CORRELATION WITH GROUND TRUTH")
print("="*70)
print(f"\n  {'Method':<16} {'Kendall τ':>10} {'Spearman ρ':>12}")
print(f"  {'-'*40}")
for nm, t, r in [('MI', tau_mi, rho_mi), ('II', tau_ii, rho_ii),
                  ('DII + L1', tau_l1, rho_l1), ('DII no L1', tau_nl1, rho_nl1)]:
    print(f"  {nm:<16} {t:>10.3f} {r:>12.3f}")

print("\n" + "="*70)
print("TOP-K PRECISION")
print("="*70)
ks = [3, 5, 10, 16]
print(f"\n  {'Method':<16}", "  ".join(f"{'Top-'+str(k):>8}" for k in ks))
print(f"  {'-'*52}")
for nm, rk in [('MI',mi_ranks),('II',ii_ranks),
               ('DII + L1',dii_l1_r),('DII no L1',dii_nol1_r)]:
    vals = "  ".join(f"{topk(rk,k):>8.2f}" for k in ks)
    print(f"  {nm:<16} {vals}")
print(f"  {'Baseline':<16}", "  ".join(f"{gt_binary.mean():>8.2f}" for _ in ks))

print("\n" + "="*70)
print("SYNERGISTIC GROUP (H) — per-feature methods should miss these")
print("="*70)
syn_idx = [feature_names.index(f) for f in ['x_syn_1','x_syn_2']]
print(f"\n  {'Feature':<12} {'MI r':>6} {'II r':>6} {'DII+L1 r':>10} {'DII noL1 r':>12}")
print(f"  {'-'*48}")
for i in syn_idx:
    print(f"  {feature_names[i]:<12} {mi_ranks[i]:>6} {ii_ranks[i]:>6} "
          f"{dii_l1_r[i]:>10} {dii_nol1_r[i]:>12}")

print("\n" + "="*70)
print("NEAR-DUPLICATE GROUP (I) — LASSO concentrates, no-L1 spreads")
print("="*70)
dup_idx = [feature_names.index(f) for f in ['x_base','x_dup_1','x_dup_2','x_dup_3','x_dup_4']]
print(f"\n  {'Feature':<12} {'MI r':>6} {'II r':>6} "
      f"{'DII+L1 w':>10} {'DII+L1 r':>10} {'DII noL1 w':>12} {'DII noL1 r':>12}")
print(f"  {'-'*72}")
for i in dup_idx:
    print(f"  {feature_names[i]:<12} {mi_ranks[i]:>6} {ii_ranks[i]:>6} "
          f"{dii_l1_w[i]:>10.4f} {dii_l1_r[i]:>10} "
          f"{dii_nol1_w[i]:>12.4f} {dii_nol1_r[i]:>12}")

# =============================================================================
# VISUALISATIONS
# =============================================================================
print("\nCreating visualisations...")

group_colors = {
    'Linear'       : '#2ecc71', 'Nonlin_Sym'  : '#e74c3c',
    'Directional'  : '#3498db', 'Cyclical'    : '#9b59b6',
    'Interaction'  : '#f39c12', 'Redundant'   : '#e67e22',
    'Derived'      : '#1abc9c', 'Synergistic' : '#c0392b',
    'NearDuplicate': '#8e44ad', 'Noise'       : '#95a5a6',
}

fig = plt.figure(figsize=(22, 18))
gs  = gridspec.GridSpec(3, 4, hspace=0.55, wspace=0.35)

# Panels 0-3: scatter GT rank vs method rank
for col, (mname, mranks, tau, rho) in enumerate([
    ('MI',          mi_ranks,  tau_mi,  rho_mi),
    ('II',          ii_ranks,  tau_ii,  rho_ii),
    ('DII + L1',    dii_l1_r,  tau_l1,  rho_l1),
    ('DII no L1',   dii_nol1_r,tau_nl1, rho_nl1),
]):
    ax = fig.add_subplot(gs[0, col])
    for idx, grp in enumerate(feature_groups):
        ax.scatter(gt_rank[idx], mranks[idx],
                   color=group_colors[grp], s=55,
                   edgecolors='black', linewidth=0.4, zorder=3)
    ax.plot([1, n_features], [1, n_features], 'k--', alpha=0.3, lw=1)
    ax.set_xlabel('Ground Truth Rank', fontsize=8)
    ax.set_ylabel('Method Rank',       fontsize=8)
    ax.set_title(f'{mname}\nτ={tau:.3f}  ρ={rho:.3f}', fontweight='bold', fontsize=9)
    if col == 0:
        ax.legend(fontsize=5, loc='upper left',
                  handles=[plt.Line2D([0],[0], marker='o', color='w',
                            markerfacecolor=c, markersize=6, label=g)
                            for g, c in group_colors.items()])
    ax.grid(True, alpha=0.3)

# Panel: Top-K precision
ax4 = fig.add_subplot(gs[1, 0:2])
ks_plot = [3, 5, 10, 16]; x = np.arange(len(ks_plot)); w = 0.18
bar_c = {'MI':'#3498db','II':'#2ecc71','DII + L1':'#e74c3c','DII no L1':'#8e44ad'}
for off, (nm, rk) in zip([-1.5*w,-0.5*w,0.5*w,1.5*w],
    [('MI',mi_ranks),('II',ii_ranks),('DII + L1',dii_l1_r),('DII no L1',dii_nol1_r)]):
    ax4.bar(x+off, [topk(rk,k) for k in ks_plot], w, label=nm,
            color=bar_c[nm], edgecolor='black', linewidth=0.5)
ax4.axhline(gt_binary.mean(), color='k', ls='--', lw=1.5,
            label=f'Baseline ({gt_binary.mean():.2f})')
ax4.set_xticks(x); ax4.set_xticklabels([f'Top-{k}' for k in ks_plot])
ax4.set_ylabel('Precision'); ax4.set_ylim(0, 1.1)
ax4.set_title('Top-K Precision', fontweight='bold')
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis='y')

# Panel: DII training curves
ax5 = fig.add_subplot(gs[1, 2:4])
ax5.plot(imbs_l1,   color='#e74c3c', lw=1.5, label='DII + L1')
ax5.plot(imbs_nol1, color='#8e44ad', lw=1.5, label='DII no L1', ls='--')
ax5.set_xlabel('Epoch'); ax5.set_ylabel('DII value')
ax5.set_title('DII Training Curves\n(with vs without L1)', fontweight='bold')
ax5.legend(); ax5.grid(True, alpha=0.3)

# Panel: near-duplicate LASSO comparison
ax6 = fig.add_subplot(gs[2, 0:2])
dup_labels = ['x_base','x_dup_1','x_dup_2','x_dup_3','x_dup_4']
dup_idx2   = [feature_names.index(f) for f in dup_labels]
xp = np.arange(len(dup_labels)); bw = 0.35
ax6.bar(xp - bw/2, dii_l1_w[dup_idx2],   bw, label='DII + L1',
        color='#e74c3c', edgecolor='black', linewidth=0.5)
ax6.bar(xp + bw/2, dii_nol1_w[dup_idx2], bw, label='DII no L1',
        color='#8e44ad', edgecolor='black', linewidth=0.5, alpha=0.7)
ax6.set_xticks(xp); ax6.set_xticklabels(dup_labels, fontsize=9)
ax6.set_ylabel('DII Weight')
ax6.set_title('Near-Duplicates: LASSO concentrates, no-L1 spreads\n'
              '(all 5 carry same information)', fontweight='bold')
ax6.legend(); ax6.grid(True, alpha=0.3, axis='y')

# Panel: synergistic group ranks
ax7 = fig.add_subplot(gs[2, 2:4])
syn_labels = ['x_syn_1', 'x_syn_2']
methods    = ['MI', 'II', 'DII+L1', 'DII noL1']
syn_ranks  = np.array([
    [mi_ranks[feature_names.index(f)] for f in syn_labels],
    [ii_ranks[feature_names.index(f)] for f in syn_labels],
    [dii_l1_r[feature_names.index(f)] for f in syn_labels],
    [dii_nol1_r[feature_names.index(f)] for f in syn_labels],
])
xp2 = np.arange(len(syn_labels)); bw2 = 0.18
colors2 = ['#3498db','#2ecc71','#e74c3c','#8e44ad']
for i, (m, c) in enumerate(zip(methods, colors2)):
    ax7.bar(xp2 + (i-1.5)*bw2, syn_ranks[i], bw2, label=m,
            color=c, edgecolor='black', linewidth=0.5)
ax7.axhline(n_features/2, color='k', ls='--', lw=1, alpha=0.5, label='Random baseline')
ax7.set_xticks(xp2); ax7.set_xticklabels(syn_labels, fontsize=9)
ax7.set_ylabel('Assigned Rank (lower = more important)')
ax7.set_title('Synergistic features (XOR): rank by method\n'
              '(per-feature methods should rank these LOW)', fontweight='bold')
ax7.legend(fontsize=8); ax7.grid(True, alpha=0.3, axis='y')
ax7.invert_yaxis()

plt.suptitle(
    'Simulation Study v4  |  MI vs II vs DII (full DADApy)  |  N=2000, 27 features\n'
    'NEW: synergistic XOR group (H) + near-duplicate LASSO group (I)',
    fontsize=11, fontweight='bold', y=1.01
)
plt.savefig('simulation_study_v4_results.png', dpi=300, bbox_inches='tight')
print("Saved: simulation_study_v4_results.png")

# =============================================================================
# SAVE
# =============================================================================
results.to_csv('simulation_study_v4_rankings.csv', index=False)

summary = f"""
SIMULATION STUDY v4 – SUMMARY
================================
N={N} | {n_features} features | {int(gt_binary.sum())} informative
NEW: synergistic XOR (H) + near-duplicates (I)

Rank Correlation with Ground Truth (Kendall τ / Spearman ρ):
  MI           τ={tau_mi:.3f}  ρ={rho_mi:.3f}
  II           τ={tau_ii:.3f}  ρ={rho_ii:.3f}
  DII + L1     τ={tau_l1:.3f}  ρ={rho_l1:.3f}
  DII no L1    τ={tau_nl1:.3f}  ρ={rho_nl1:.3f}

Top-K Precision:
  Method           Top-3  Top-5  Top-10  Top-16
  MI               {topk(mi_ranks,3):.2f}   {topk(mi_ranks,5):.2f}   {topk(mi_ranks,10):.2f}    {topk(mi_ranks,16):.2f}
  II               {topk(ii_ranks,3):.2f}   {topk(ii_ranks,5):.2f}   {topk(ii_ranks,10):.2f}    {topk(ii_ranks,16):.2f}
  DII + L1         {topk(dii_l1_r,3):.2f}   {topk(dii_l1_r,5):.2f}   {topk(dii_l1_r,10):.2f}    {topk(dii_l1_r,16):.2f}
  DII no L1        {topk(dii_nol1_r,3):.2f}   {topk(dii_nol1_r,5):.2f}   {topk(dii_nol1_r,10):.2f}    {topk(dii_nol1_r,16):.2f}
  Baseline         {gt_binary.mean():.2f}

Synergistic group (x_syn_1, x_syn_2) — ranks assigned:
  MI:       x_syn_1={mi_ranks[feature_names.index('x_syn_1')]}, x_syn_2={mi_ranks[feature_names.index('x_syn_2')]}
  II:       x_syn_1={ii_ranks[feature_names.index('x_syn_1')]}, x_syn_2={ii_ranks[feature_names.index('x_syn_2')]}
  DII+L1:   x_syn_1={dii_l1_r[feature_names.index('x_syn_1')]}, x_syn_2={dii_l1_r[feature_names.index('x_syn_2')]}
  DII noL1: x_syn_1={dii_nol1_r[feature_names.index('x_syn_1')]}, x_syn_2={dii_nol1_r[feature_names.index('x_syn_2')]}
  (expected: MI/II rank HIGH number = unimportant; DII ranks LOW = important)

Near-duplicate weights (x_base + 4 copies):
  Feature      DII+L1 weight   DII noL1 weight
  x_base       {dii_l1_w[feature_names.index('x_base')]:.4f}         {dii_nol1_w[feature_names.index('x_base')]:.4f}
  x_dup_1      {dii_l1_w[feature_names.index('x_dup_1')]:.4f}         {dii_nol1_w[feature_names.index('x_dup_1')]:.4f}
  x_dup_2      {dii_l1_w[feature_names.index('x_dup_2')]:.4f}         {dii_nol1_w[feature_names.index('x_dup_2')]:.4f}
  x_dup_3      {dii_l1_w[feature_names.index('x_dup_3')]:.4f}         {dii_nol1_w[feature_names.index('x_dup_3')]:.4f}
  x_dup_4      {dii_l1_w[feature_names.index('x_dup_4')]:.4f}         {dii_nol1_w[feature_names.index('x_dup_4')]:.4f}
  (expected: +L1 concentrates on one; no-L1 spreads across all 5)
"""
print(summary)
with open('simulation_study_v4_summary.txt', 'w') as f:
    f.write(summary)
print("Saved: simulation_study_v4_summary.txt")
print("\n" + "="*70)
print("L1 TUNING — effect on near-duplicate weights")
print("="*70)
print("Testing l1_strength values: 0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3")

l1_values  = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3]
dup_names  = ['x_base','x_dup_1','x_dup_2','x_dup_3','x_dup_4']
dup_idx    = [feature_names.index(f) for f in dup_names]

# store weights for each l1
tuning_weights = {}
tuning_tau     = {}

for l1 in l1_values:
    w, r, _ = run_dii(l1, f"L1={l1}")
    tuning_weights[l1] = w
    t, _ = kendalltau(gt_rank, r)
    tuning_tau[l1] = t

# Print table
print(f"\n  {'l1':>8}  " + "  ".join(f"{n:>10}" for n in dup_names) +
      f"  {'max/sum':>8}  {'τ':>7}")
print("  " + "-"*85)
for l1 in l1_values:
    w = tuning_weights[l1]
    dup_w = w[dup_idx]
    ratio = dup_w.max() / (dup_w.sum() + 1e-10)
    print(f"  {l1:>8.3f}  " + "  ".join(f"{dup_w[i]:>10.4f}" for i in range(5)) +
          f"  {ratio:>8.3f}  {tuning_tau[l1]:>7.3f}")

print("\n  max/sum → 1.0 means weight fully concentrated on one feature")
print("  max/sum → 0.2 means weight evenly spread across 5 features")

# Plot tuning results
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: weight distribution across near-duplicates for each l1
ax_left = axes[0]
x_pos = np.arange(len(dup_names))
colors_l1 = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(l1_values)))
bw = 0.12
for k, l1 in enumerate(l1_values):
    dup_w = tuning_weights[l1][dup_idx]
    ax_left.bar(x_pos + (k - len(l1_values)/2 + 0.5)*bw, dup_w, bw,
                label=f'l1={l1}', color=colors_l1[k], edgecolor='black', linewidth=0.3)
ax_left.set_xticks(x_pos)
ax_left.set_xticklabels(dup_names, fontsize=9)
ax_left.set_ylabel('DII Weight')
ax_left.set_title('Near-duplicate weights by l1_strength\n(L1 concentrates weight on one feature)', fontweight='bold')
ax_left.legend(fontsize=7, loc='upper right')
ax_left.grid(True, alpha=0.3, axis='y')

# Right: max/sum ratio and tau vs l1
ax_right = axes[1]
ratios = [tuning_weights[l1][dup_idx].max() /
          (tuning_weights[l1][dup_idx].sum() + 1e-10) for l1 in l1_values]
taus   = [tuning_tau[l1] for l1 in l1_values]
l1_labels = [str(v) for v in l1_values]

ax_right.plot(range(len(l1_values)), ratios, 'o-', color='#e74c3c',
              lw=2, ms=7, label='max/sum (sparsity)')
ax_right2 = ax_right.twinx()
ax_right2.plot(range(len(l1_values)), taus, 's--', color='#3498db',
               lw=2, ms=7, label='Kendall τ')
ax_right.set_xticks(range(len(l1_values)))
ax_right.set_xticklabels(l1_labels, fontsize=9)
ax_right.set_xlabel('l1_strength')
ax_right.set_ylabel('Sparsity (max/sum)', color='#e74c3c')
ax_right2.set_ylabel('Kendall τ', color='#3498db')
ax_right.set_title('Sparsity vs accuracy trade-off\nas l1_strength increases', fontweight='bold')
ax_right.axhline(1/len(dup_names), color='grey', ls=':', label='uniform baseline')
lines1, labels1 = ax_right.get_legend_handles_labels()
lines2, labels2 = ax_right2.get_legend_handles_labels()
ax_right.legend(lines1+lines2, labels1+labels2, fontsize=8)
ax_right.grid(True, alpha=0.3)

plt.suptitle('L1 Tuning: effect on near-duplicate sparsity and overall performance',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('simulation_study_v4_l1_tuning.png', dpi=300, bbox_inches='tight')
print("\nSaved: simulation_study_v4_l1_tuning.png")

print("\n" + "="*70)
print("SIMULATION STUDY v4 COMPLETE")
print("="*70)

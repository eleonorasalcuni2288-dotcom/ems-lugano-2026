# Robustness of Information-Theoretic Feature Selection in High Dimensions: A Bootstrap-Validated Comparison Across Synthetic and Real Data

Information Imbalance (II), Differentiable Information Imbalance (DII), and
Mutual Information (MI) compared under a shared two-level fair-comparison
framework, extended to three baselines from outside the information-
imbalance family (MINE, Random Forest, LASSO), evaluated on a synthetic
benchmark with known ground truth and three different real-world
datasets. Prepared as a poster for the 35th European Meeting of
Statisticians (EMS 2026), Lugano.

## Overview

Feature selection methods can disagree substantially on which features
matter most, and this disagreement is not merely noise: it reflects
genuine differences in what each method measures (correlation, k-NN
geometry, or a learned neural bound) and in how each scales with
dimensionality. This project runs six methods (eight method-level
configurations, since MI and II are each evaluated per-feature and
jointly) head-to-head under one controlled experimental protocol:

1. **Synthetic benchmark** (p=27/50/105, known ground-truth ranking
   including a synergistic XOR-type feature pair invisible to per-feature
   methods), which lets each method's ranking be checked directly against
   the true feature importance.
2. **Three real datasets** (FRED-MD macro-financial panel, post-infarction
   clinical data, trading time series), where there is no ground truth, so
   methods are instead validated by whether their selected features actually
   improve downstream k-NN classification accuracy over a random baseline,
   with bootstrap-quantified uncertainty on that advantage.

## Methods compared

| Method | What it measures | Level(s) evaluated |
|---|---|---|
| **MI** (Mutual Information) | Statistical dependence, k-NN (Kraskov–Stögbauer–Grassberger) estimator | per-feature, joint |
| **II** (Information Imbalance) | Rank-based nearest-neighbour geometry: II(X→Y) = (2/N²) Σᵢ rank_Y(NN_X(i)) | per-feature, joint (leave-one-out) |
| **DII** (Differentiable Information Imbalance) | Gradient-optimised soft relaxation of II (softmax neighbourhoods), with L1 regularisation on feature weights for sparsity | joint, gradient-optimised |
| **MINE** | Neural MI estimator, Donsker–Varadhan bound with bias-corrected (EMA) gradient | joint |
| **RF** (Random Forest) | Permutation importance from a general-purpose ensemble model | joint |
| **LASSO** | Linear model, L1-penalized; ranks by \|coefficient\| magnitude, alpha auto-selected via 5-fold CV | joint |

**JMI** (greedy conditional-MI selection) was also run on the synthetic
benchmark as a scalability reference, not as a competing ranking method:
23.8s at p=27, 505.7s at p=50, and dropped above p=60 as its O(p²) cost
becomes intractable. This is itself part of the result, showing that
sequential greedy selection does not scale the way joint/gradient-based
methods do.

**InfoNCE** was explored early as a sixth candidate (p=27 only, single
point estimate) but was superseded by MINE for the final comparison:
unlike InfoNCE, which remains an unpublished arXiv preprint, MINE is
peer-reviewed (Belghazi et al., 2018, ICML). Its script and results stay
in `synthetic/` rather than `archive/`, since `infonce_synthetic.py` also
defines the MLP utilities (`init_mlp`, `mlp_apply`) reused by
`mine_synthetic_highdim.py` and `rf_synthetic_highdim.py`. The InfoNCE
experiment itself is not part of the reported comparison, but the file is
not purely historical.

## Datasets

| Dataset | N | p | Target | Role |
|---|---|---|---|---|
| Synthetic benchmark | 2,000 | 27 / 50 / 105 | continuous Y, known generative process incl. one XOR-type synergy pair | ground-truth check |
| Trading (`train2.csv`) | 5,000 (stratified subsample of 440,402) | 27 | next-day price direction (100 anonymized stocks, ~50.3% positive) | real-data domain, comparable dimensionality to the synthetic baseline (p=27) |
| FRED-MD | 794 | 120 | S&P 500 monthly return, median-split binarised | real-data domain (macro-financial, high-dimensional) |
| Post-infarction complications | 1,700 | 107 | any of 11 binary post-infarction complications | real-data domain (clinical, high-dimensional, no ground truth) |

Datasets are grouped by comparable dimensionality in the poster: synthetic
and trading (p≈27) side by side, FRED-MD and post-infarction complications (p≈100–120)
side by side.

## Methodology

- **Two-level fair comparison**: MI and II are evaluated both per-feature
  (independent scores) and jointly (leave-one-out backward elimination,
  or direct joint gradient optimisation for DII); DII, MINE, RF, and LASSO
  (inherently joint methods by construction) are evaluated jointly only,
  so no method is compared per-feature against another's joint result.
- **Bootstrap CI**: 80% subsampling *without* replacement (not classic
  with-replacement bootstrap, chosen to avoid zero-distance duplicate-
  neighbour artefacts in the distance-based methods, II and DII), ranking
  recomputed from scratch on every draw.
- **Ground-truth-free downstream validation** (the 3 real datasets): k-NN
  leave-one-out balanced accuracy using each method's top-K features,
  compared against K random features (n=100 random draws), reported as an
  accuracy advantage with a bootstrap 95% CI. An advantage is called
  *robust* only if its CI excludes zero.
- **Multiple-testing correction**: applied uniformly to all three real
  datasets.
  Post-infarction complications tests 11 complications × 4 methods × 4
  K values (176 tests): Benjamini–Hochberg correction leaves 7
  survivors, of which only
  3 are *also* bootstrap-robust. FRED-MD and trading each test 4 methods ×
  4 K values (16 tests): under the same BH correction, FRED-MD keeps 4/16
  (down from 6/16 raw p<0.05), and trading keeps **0/16** (down from 2/16
  raw p<0.05), trading's apparent signal does not survive correction for
  testing 16 hypotheses. BH significance and CI-excludes-zero robustness
  do not always agree, on any of the three datasets, which is reported
  honestly rather than picking whichever criterion looks stronger.
- **A stress-test of the uncertainty-quantification procedure itself**:
  the project's subsampling-based CIs are checked against the rescaled
  Politis-Romano-Wolf subsampling correction on the tightest-margin case
  in every dataset (14 configurations total: 1 synthetic, 4+4 FRED-MD
  MI/DII, 3 post-infarction, 2 trading), confirming the central claims
  are robust to this choice of procedure, while identifying which
  specific results are close enough to the zero boundary to be sensitive
  to it. A follow-up empirical coverage check found the project's own
  procedure achieves 97.5% coverage of the true value (close to the 95%
  nominal target, mildly conservative), while the Politis-Romano-Wolf
  procedure achieves only 70.0% here — under-covering, consistent with
  its asymptotic assumptions not holding at 80% subsampling. Full
  results in Scope, Design Choices & Robustness Checks.

## Key findings

**DII+L1 is the only method with a downstream advantage in all three real
datasets under the project's own bootstrap procedure**, and the only
method that is simultaneously accurate and robust to synergy detection on
the synthetic benchmark. A follow-up verification against the
Politis-Romano-Wolf resampling correction (see Scope, Design Choices &
Robustness Checks) confirmed this holds for FRED-MD and post-infarction complications, but
**not** for trading, whose result — already the weakest of the three by
several other measures — is best read as complementary rather than
confirmatory evidence:

- **Synthetic** (bootstrap τ / XOR-pair rank, p=27→105): DII+L1 stays at
  rank 1–3 on the synergy pair at every p (τ 0.370→0.466→0.420); II-joint
  detects it well at low p but collapses to rank 22–70 at p=105; MI-joint
  (Kraskov) degenerates entirely at p≥50; per-feature MI/II never detect
  it at any p (rank 21–88, by construction), confirmed not an artefact of
  MI's own `n_neighbors` default (3): varying it over [3,5,7,10] at p=27
  leaves the XOR pair stuck at rank 21–26 throughout, with the rest of the
  ranking stable (τ 0.89–0.90 agreement with the default). LASSO, the
  classical sparse-regularized baseline conceptually closest to DII+L1,
  both ranking via a sparsity-inducing penalty, is statistically
  indistinguishable from DII+L1 on raw τ at every p (0.354 vs. 0.370 at
  p=27, 0.417 vs. 0.466 at p=50, 0.422 vs. 0.420 at p=105; bootstrap CIs
  overlap substantially throughout), but **never detects the synergy
  pair** (rank 18–39 at p=27, worsening to 30–69 at p=105), a linear
  model structurally cannot represent a pure multiplicative/XOR
  interaction without explicit interaction terms, which LASSO here does
  not have. This sharpens what DII+L1's advantage actually is: not
  superior *raw* ranking accuracy (LASSO and Random Forest are both
  competitive there), but the ability to detect synergistic dependencies
  that per-feature, linear, and tree-ensemble methods alike structurally
  cannot see. Random Forest also has strong raw ranking accuracy
  (τ up to 0.50, CI never crossing zero, on par with MI per-feature's
  own peak of 0.51 at p=50) but never ranks the synergy pair above
  14th, and dilutes importance across near-duplicate features (23%
  concentration on the best copy vs. 43% for DII+L1). MINE's
  bootstrap CI crosses zero at every p (τ −0.081 to 0.073) and its
  own synergy detection collapses most abruptly of all methods, from
  rank ~2 to ~89 between p=50 and p=105, consistent with the modest
  sample size relative to what neural
  MI estimators typically require. A dedicated hyperparameter check on
  DII+L1 itself (mirroring RF's n_estimators test and MINE's convergence
  check) shows XOR-pair detection unchanged (rank 1–2) across l1_strength
  in [0.05, 0.20], though overall τ trends down as L1 grows (0.35 at the
  project's l1=0.10 to 0.25 at l1=0.20, ranking agreement τ 0.74–0.90
  vs. l1=0.10); across 3 random seeds at l1=0.10, results are identical
  (std=0.000), DII's training here is full-batch (`batches_per_epoch=1`)
  and effectively deterministic, not stochastic the way MINE's is.
- **Trading**: at full resolution (B=100 for MI/II, B=15 for DII, matching
  FRED-MD/post-infarction exactly), DII+L1's bootstrap CI excludes zero
  at K=10 (+0.018 [0.004, 0.026]) and K=16 (+0.015 [0.006, 0.025]), the
  *only* robust result on this dataset under the project's own procedure.
  II per-feature's apparent K=3 robustness at a smaller preliminary
  bootstrap (B=12) did not hold up under the full run (now +0.003
  [−0.019, 0.017], fragile), a genuine correction, not noise in the other
  direction: more draws revealed a false positive, not a false negative.
  **None of DII+L1's results survive BH correction** for the dataset's 16
  tests either, and neither K=10 nor K=16 survives the Politis-Romano-Wolf
  resampling CI (see Scope, Design Choices & Robustness Checks) — both
  margins were tight enough (ci_lo +0.004 and +0.006) that the
  correction's re-centring pushes them through zero. Trading's downstream
  signal is the weakest of the three real datasets by every criterion
  applied.
- **FRED-MD**: DII+L1's bootstrap CI excludes zero at all 4 K (advantage
  +0.03 to +0.13), independently re-confirmed under the Politis-Romano-Wolf
  resampling CI at all 4 K (see Scope, Design Choices &
  Robustness Checks); under BH correction, DII+L1 at K=5/10/16 and MI
  per-feature at K=10 remain significant (4/16 total), while DII+L1 at
  K=3 and MI per-feature at K=5, both bootstrap-robust under the
  project's procedure, do not
  survive BH correction — and MI per-feature at K=5 additionally does not
  survive the Politis-Romano-Wolf resampling CI either (K=10 does).
- **Post-infarction complications**: of 7 BH survivors, only 3 are also
  bootstrap-robust, all involving the ZSN (cardiac) complication, via MI
  per-feature (K=3, K=5) and DII+L1 (K=16).

**Future work**: the joint-detection capability demonstrated in the
present work, on synthetic and real data alike, suggests a natural
extension to multi-asset financial data, where signals weak in isolation
might combine into a stronger joint predictor. Preliminary tests already
implemented on a gold/cyclical-growth pair (see
`future_work_real_assets/`) show a promising signal; further
bootstrap-validated studies will be conducted, extending to a larger
sample and a broader set of assets.

## Project structure

```
feature_selection_analysis/
├── README.md
├── requirements.txt
│
├── common/
│   └── downstream_validation.py   # shared: knn_loo_accuracy, evaluate_method_vs_baseline (+ RF variants)
│
├── synthetic/                 # ground-truth benchmark, p=27/50/105
│   ├── simulation_study_v6_highdim.py      # core MI/II/DII/JMI sweep (also the shared library
│   │                                        #   for compute_ii_pf / make_ii_joint / run_dii)
│   ├── bootstrap_ci_synthetic.py           # bootstrap CI, p=27
│   ├── bootstrap_ci_synthetic_highdimensional.py  # bootstrap CI, p=50/105
│   ├── mine_synthetic_highdim.py           # MINE, p=27/50/105
│   ├── rf_synthetic_highdim.py             # Random Forest, p=27/50/105
│   ├── rf_diagnostics.py                   # RF stability + near-duplicate-feature checks
│   ├── infonce_synthetic.py                # InfoNCE, p=27 (also defines init_mlp/mlp_apply,
│   │                                        #   reused by MINE and RF above, not purely historical)
│   ├── infonce_synthetic_highdim.py        # InfoNCE p=50/105 extension (written, never completed)
│   ├── validate_downstream_synthetic.py    # sanity check: ranking-by-accuracy vs ranking-by-tau
│   ├── dii_diagnostics.py                  # DII+L1 sensitivity: L1-strength sweep + cross-seed check
│   ├── mi_diagnostics.py                   # MI sensitivity: n_neighbors sweep + cross-seed check
│   ├── lasso_synthetic_highdim.py          # LASSO baseline, p=27/50/105 (closest classical
│   │                                        #   competitor to DII+L1: both rank via sparsity penalty)
│   ├── subsampling_correction_check_synthetic.py  # Politis-Romano-Wolf spot-check, II-joint p=27
│   └── coverage_check.py                   # frequentist coverage check (97.5% vs 70.0%)
│
├── fredmd/                    # macro-financial, N=794, p=120
│   ├── 2026-07-MD.csv         # raw FRED-MD panel
│   ├── fredmd_analysis.py     # rankings + point-estimate downstream validation
│   ├── bootstrap_ci_fredmd.py
│   ├── bh_correction_fredmd.py    # Benjamini-Hochberg on the 16 method x K tests
│   ├── verify_fredmd_balanced.py
│   ├── subsampling_correction_check.py     # Politis-Romano-Wolf spot-check, MI_perfeat, all 4 K
│   └── subsampling_correction_check_dii.py # Politis-Romano-Wolf spot-check, DII+L1, all 4 K
│
├── post_infarction/          # clinical, N=1700, p=107
│   ├── MI.data                # raw UCI/Leicester dataset
│   ├── post_infarction_analysis.py
│   ├── post_infarction_per_target.py      # per-complication sweep (176 tests) + BH correction
│   ├── post_infarction_rf_check.py
│   ├── bootstrap_ci_mi_aggregate.py
│   ├── bootstrap_mi_survivors.py           # bootstrap CI on the 7 BH survivors
│   ├── verify_mi_rf_balanced.py
│   ├── reverify_balanced_accuracy.py
│   └── subsampling_correction_check.py     # Politis-Romano-Wolf spot-check, the 3 ZSN survivors
│
├── trading/                   # N=5000 (subsample of 440,402), p=27
│   ├── train2.csv              # gitignored (110MB, over GitHub's 100MB limit)
│   ├── real_data_analysis.py
│   ├── trading_downstream_validation.py
│   ├── bh_correction_trading.py   # Benjamini-Hochberg on the 16 method x K tests
│   └── subsampling_correction_check_dii.py # Politis-Romano-Wolf spot-check, DII+L1, K=10/16
│
├── figures/                   # poster figures (matplotlib, PDF+PNG) + the scripts that build them
│   ├── plot_poster_figures.py       # fig_scalability, fig_fredmd, fig_mi_survivors, fig_trading
│   └── plot_real_data_comparison.py # combined 3-panel real-data figure
│
├── future_work_real_assets/   # exploratory, NOT part of the poster or any reported result
│   ├── build_dataset.py       # real multi-asset panel (yfinance), interaction-pair design
│   ├── run_methods.py         # first-look MI/II/DII+L1/RF/LASSO point estimates, no bootstrap CI
│   ├── dataset_feature_selection_2023_2025.csv
│   └── method_rankings.csv
│
└── archive/                   # superseded early-stage scripts (simulation_study v1–v5, first-pass
                                # trading/DII scripts, paper_draft.tex), kept for history, not used
                                # by any current result
```


## Requirements

Python 3.12. Exact versions used for the results above (see
`requirements.txt`):

```
numpy==1.26.4
pandas==3.0.2
scikit-learn==1.8.0
scipy==1.17.1
matplotlib==3.10.8
dadapy==0.3.3      # DII (DiffImbalance)
jax==0.4.30        # MINE
jaxlib==0.4.30
optax==0.2.5        # MINE
```

## Installation

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Each dataset folder is self-contained: run its scripts with that folder as
the working directory (e.g. `cd synthetic && python3 bootstrap_ci_synthetic.py`).
Typical order per dataset: the `*_analysis.py` / `simulation_study_v6_highdim.py`
script first (produces rankings), then the `bootstrap_ci_*.py` scripts
(bootstrap CI on downstream advantage), then any `verify_*`/`*_diagnostics.py`
scripts. Once all four datasets' results exist, generate the poster figures:

```bash
cd figures
python3 plot_poster_figures.py       # fig_scalability, fig_fredmd, fig_mi_survivors, fig_trading
```

## Scope, Design Choices & Robustness Checks

Every design choice below was tested rather than assumed: each bullet
leads with what was checked and found, not just what could go wrong.

- **Uncertainty quantification**: the raw-percentile subsampling CI was
  validated against the rescaled Politis-Romano-Wolf method across 14
  tightest-margin configurations (1 synthetic, 4+4 FRED-MD MI/DII, 3
  post-infarction, 2 trading). The primary procedure achieves 97.5%
  empirical coverage against a 95% nominal target, versus 70.0% for the
  Politis-Romano-Wolf correction, and 12 of the 14 configurations keep
  the same robust/fragile classification either way. The only exceptions
  (2 of 14) are borderline results for
  per-feature MI at K=5 on FRED-MD and DII+L1 at K=10 and K=16 on the
  trading dataset.
- **Trading baseline signal**: next-day price direction sits near the
  efficient-market frontier, with baseline accuracy close to random
  chance. The lower downstream signal on trading relative to FRED-MD and
  post-infarction complications reflects the task's intrinsic difficulty.
- **Trading panel dependencies**: the 5,000-row subsample yields a
  same-stock k-NN neighbour rate of 4.2% versus 1.0% expected under
  i.i.d. sampling. This structure does not favour the proposed method:
  DII+L1's top-5 features show a lower same-stock neighbour rate (1.9%)
  than a random 5-feature baseline (2.7%).
- **Random Forest stability**: testing across n_estimators ∈
  {50, 100, 200, 500} trees shows strong rank agreement (Kendall's
  τ ≈ 0.95–0.97) across all levels, confirming feature rank stability
  despite a 6.7% drift in raw importance magnitude.
- **MINE convergence diagnostics**: a convergence audit revealed an
  18.3% relative change at 300 epochs against a 5% target threshold.
  This under-convergence directly accounts for MINE's wider bootstrap
  confidence intervals.
- **JMI scalability**: due to its O(p²) complexity, runtime scales from
  23.8 seconds at p=27 to 505.7 seconds at p=50, making it intractable
  beyond p=60. It is therefore excluded from the p=105 experiments.
- **Dimensionality scope**: trading is deliberately not high-dimensional
  (p=27), unlike FRED-MD and post-infarction complications (p=107–120). It is
  included precisely because its dimensionality matches the synthetic
  benchmark, isolating the real vs. synthetic comparison from the low
  vs. high-dimensional one: without it, any gap between synthetic and
  real-data behaviour could be confounded with dimensionality rather
  than "realness" alone. It is complementary evidence, not a third
  high-dimensional confirmation.

## Data provenance & licensing

- **`fredmd/2026-07-MD.csv`**: McCracken & Ng, FRED-MD monthly database
  (Federal Reserve Bank of St. Louis): public-domain or permission-granted
  series only. Redistributed here with attribution; cite McCracken & Ng
  (2016) when using this data.
- **`post_infarction/MI.data`**: Golovenkin et al. (2020), dataset DOI:
  10.25392/leicester.data.12045261.v3 (UCI Machine Learning Repository /
  University of Leicester), CC BY 4.0 licensed, sharing and adaptation
  permitted with attribution; cite the *GigaScience* paper above when
  using this data.
- **`trading/train2.csv`**: *Stock Market Signal: Predict Next-Day
  Returns* (Kaggle competition, created by J. Saleeby, 2026), CC BY-SA
  4.0 licensed, 100 anonymized US equities, 2000–2023. Gitignored here
  (110MB, exceeds GitHub's 100MB limit), not redistributed; download from
  the competition page and place at `trading/train2.csv` to reproduce.

## References

[1] Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating
    mutual information. *Physical Review E*, 69, 066138.

[2] Glielmo, A., Zeni, C., Cheng, B., Csányi, G., & Laio, A. (2022).
    Ranking the information content of distance measures. *PNAS Nexus*,
    1(2), pgac039.

[3] Glielmo, A., Macocco, I., Doimo, D., Carli, M., Zeni, C., Wild, R.,
    d'Errico, M., Rodriguez, A., & Laio, A. (2022). DADApy: Distance-based
    analysis of data-manifolds in Python. *Patterns*, 3(10), 100589.

[4] Wild, R., Wodaczek, F., Del Tatto, V., Cheng, B., & Laio, A. (2025).
    Automatic feature selection and weighting in molecular systems using
    Differentiable Information Imbalance. *Nature Communications*, 16, 270.

[5] McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for
    macroeconomic research. *Journal of Business & Economic Statistics*,
    34(4), 574–589.

[6] Golovenkin, S. E., Bac, J., Chervov, A., Mirkes, E. M., Orlova, Y. V.,
    Barillot, E., Gorban, A. N., & Zinovyev, A. (2020). Trajectories,
    bifurcations, and pseudo-time in large clinical datasets:
    applications to myocardial infarction and diabetes data.
    *GigaScience*, 9(11), giaa128.

[7] Saleeby, J. (2026). Stock Market Signal: Predict Next-Day Returns.
    Kaggle.

[8] Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
    rate: A practical and powerful approach to multiple testing. *Journal
    of the Royal Statistical Society, Series B*, 57(1), 289–300.

[9] Belghazi, M. I., Baratin, A., Rajeshwar, S., Ozair, S., Bengio, Y.,
    Courville, A., & Hjelm, R. D. (2018). MINE: Mutual information neural
    estimation. *Proceedings of the 35th International Conference on
    Machine Learning*, PMLR 80, 531–540.

[10] Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

[11] Tibshirani, R. (1996). Regression shrinkage and selection via the
     lasso. *Journal of the Royal Statistical Society, Series B*, 58(1),
     267–288.

[12] Politis, D. N., & Romano, J. P. (1994). Large sample confidence
     regions based on subsamples under minimal assumptions. *The Annals
     of Statistics*, 22(4), 2031–2050.

[13] Politis, D. N., Romano, J. P., & Wolf, M. (1999). *Subsampling*.
     Springer Series in Statistics. Springer, New York.

## Contact

**Eleonora Salcuni**: eleonora.salcuni@usi.ch or eleonorasalcuni2288@gmail.com

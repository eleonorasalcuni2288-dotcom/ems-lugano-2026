"""
FRED-MD — Benjamini-Hochberg Correction on the Point-Estimate Downstream Tests
================================================================================
FRED-MD's downstream validation (fredmd_analysis.py) runs 4 methods x 4 K
values = 16 significance tests (one empirical p-value per method/K, from
evaluate_method_vs_baseline's n_random=100 comparison), but until now no
multiple-testing correction was applied to them — unlike MI-complications,
whose 176 per-target tests already get BH correction
(post_infarction_per_target.py). This closes that asymmetry, applying the
identical procedure to FRED-MD's own 16 tests.

Uses the existing fredmd_downstream_results.csv (no retraining, no new
ranking computation) — this is a pure post-hoc statistical correction on
already-computed p-values.

BH procedure: sort p-values ascending, threshold(i) = i/n * alpha; find the
LARGEST rank i with p(i) <= threshold(i); reject (mark significant) all
ranks <= i. This is the correct BH step-up procedure (not a pointwise
per-rank comparison, which can under- or over-reject in edge cases).
"""
import pandas as pd
import numpy as np

ALPHA = 0.05

df = pd.read_csv('fredmd_downstream_results.csv')
d = df.dropna(subset=['p_value']).copy().sort_values('p_value', kind='mergesort').reset_index(drop=True)
n = len(d)
d['rank'] = np.arange(1, n + 1)
d['bh_threshold'] = d['rank'] / n * ALPHA
passes = d['p_value'] <= d['bh_threshold']
max_i = d.loc[passes, 'rank'].max() if passes.any() else 0
d['bh_significant'] = d['rank'] <= max_i

print("=" * 70)
print(f"FRED-MD — BENJAMINI-HOCHBERG CORRECTION ({n} tests: 4 methods x 4 K)")
print("=" * 70)
print(f"Raw p<0.05: {(d['p_value'] < 0.05).sum()}/{n} "
      f"(expect ~{0.05*n:.1f} by chance alone if no real effect)")
print(f"BH-significant: {d['bh_significant'].sum()}/{n}\n")
print("Surviving BH correction:")
print(d[d['bh_significant']].sort_values('p_value')[
    ['method', 'K', 'method_acc', 'baseline_mean', 'p_value']
].to_string(index=False))

d.to_csv('bh_correction_fredmd_results.csv', index=False)
print("\nSaved: bh_correction_fredmd_results.csv")

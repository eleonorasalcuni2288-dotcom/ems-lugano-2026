"""
Trading — Benjamini-Hochberg Correction on the Point-Estimate Downstream Tests
================================================================================
Same rationale and procedure as fredmd/bh_correction_fredmd.py: trading's
downstream validation runs 4 methods x 4 K values = 16 significance tests
(empirical p-value per method/K), never previously corrected for multiple
testing. Uses the existing trading_downstream_results.csv (no retraining).

BH procedure: sort p-values ascending, threshold(i) = i/n * alpha; find the
LARGEST rank i with p(i) <= threshold(i); reject (mark significant) all
ranks <= i.
"""
import pandas as pd
import numpy as np

ALPHA = 0.05

df = pd.read_csv('trading_downstream_results.csv')
d = df.dropna(subset=['p_value']).copy().sort_values('p_value', kind='mergesort').reset_index(drop=True)
n = len(d)
d['rank'] = np.arange(1, n + 1)
d['bh_threshold'] = d['rank'] / n * ALPHA
passes = d['p_value'] <= d['bh_threshold']
max_i = d.loc[passes, 'rank'].max() if passes.any() else 0
d['bh_significant'] = d['rank'] <= max_i

print("=" * 70)
print(f"TRADING — BENJAMINI-HOCHBERG CORRECTION ({n} tests: 4 methods x 4 K)")
print("=" * 70)
print(f"Raw p<0.05: {(d['p_value'] < 0.05).sum()}/{n} "
      f"(expect ~{0.05*n:.1f} by chance alone if no real effect)")
print(f"BH-significant: {d['bh_significant'].sum()}/{n}\n")
if d['bh_significant'].sum() > 0:
    print("Surviving BH correction:")
    print(d[d['bh_significant']].sort_values('p_value')[
        ['method', 'K', 'method_acc', 'baseline_mean', 'p_value']
    ].to_string(index=False))
else:
    print("No result survives BH correction — the raw p<0.05 hits (DII_L1 at "
          "K=3, K=5) do not hold up once corrected for testing 16 hypotheses.")

d.to_csv('bh_correction_trading_results.csv', index=False)
print("\nSaved: bh_correction_trading_results.csv")

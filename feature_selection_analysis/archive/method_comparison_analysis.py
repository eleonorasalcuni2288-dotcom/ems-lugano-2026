"""
Method Comparison Analysis
Compares II, MI, and DII to determine which method is best for feature selection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau

print("\n" + "="*80)
print("METHOD COMPARISON ANALYSIS")
print("Which method selects the best features?")
print("="*80)

# Load all results
print("\n[1/5] Loading results...")
consensus_df = pd.read_csv('consensus_ranking.csv')
dii_df = pd.read_csv('ii_vs_dii_comparison.csv')
complete_df = pd.read_csv('ii_vs_mi_vs_dii_complete_comparison.csv')

print("Loaded all ranking data")

# Step 2: Agreement Analysis
print("\n[2/5] Computing method agreement...")

ii_rank = complete_df['II_Rank'].values
mi_rank = complete_df['MI_Rank'].values
dii_rank = complete_df['DII_Rank'].values

# Correlation between methods
corr_ii_mi, _ = spearmanr(ii_rank, mi_rank)
corr_ii_dii, _ = spearmanr(ii_rank, dii_rank)
corr_mi_dii, _ = spearmanr(mi_rank, dii_rank)

print(f"Spearman correlation between methods:")
print(f"  II vs MI:  {corr_ii_mi:.4f}")
print(f"  II vs DII: {corr_ii_dii:.4f}")
print(f"  MI vs DII: {corr_mi_dii:.4f}")

# Step 3: Top-K Agreement
print("\n[3/5] Computing top-K agreement...")

for k in [5, 10, 15]:
    top_k_ii = set(complete_df.nsmallest(k, 'II_Rank')['Feature'].tolist())
    top_k_mi = set(complete_df.nsmallest(k, 'MI_Rank')['Feature'].tolist())
    top_k_dii = set(complete_df.nsmallest(k, 'DII_Rank')['Feature'].tolist())
    
    overlap_all = top_k_ii & top_k_mi & top_k_dii
    overlap_ii_mi = top_k_ii & top_k_mi
    overlap_ii_dii = top_k_ii & top_k_dii
    overlap_mi_dii = top_k_mi & top_k_dii
    
    print(f"\nTop-{k} Features Agreement:")
    print(f"  All three methods agree: {len(overlap_all)}/{k} ({100*len(overlap_all)/k:.1f}%)")
    print(f"  II & MI agree: {len(overlap_ii_mi)}/{k} ({100*len(overlap_ii_mi)/k:.1f}%)")
    print(f"  II & DII agree: {len(overlap_ii_dii)}/{k} ({100*len(overlap_ii_dii)/k:.1f}%)")
    print(f"  MI & DII agree: {len(overlap_mi_dii)}/{k} ({100*len(overlap_mi_dii)/k:.1f}%)")

# Step 4: Stability Analysis
print("\n[4/5] Computing stability metrics...")

std_ii = np.std(complete_df['II_Rank'])
std_mi = np.std(complete_df['MI_Rank'])
std_dii = np.std(complete_df['DII_Rank'])

cv_ii = std_ii / np.mean(complete_df['II_Rank'])
cv_mi = std_mi / np.mean(complete_df['MI_Rank'])
cv_dii = std_dii / np.mean(complete_df['DII_Rank'])

print(f"\nStandard Deviation (lower = more stable):")
print(f"  II:  {std_ii:.4f} (CV: {cv_ii:.4f})")
print(f"  MI:  {std_mi:.4f} (CV: {cv_mi:.4f})")
print(f"  DII: {std_dii:.4f} (CV: {cv_dii:.4f})")

# Step 5: Feature Robustness
print("\n[5/5] Computing feature robustness...")

complete_df['Rank_Variance'] = complete_df[['II_Rank', 'MI_Rank', 'DII_Rank']].var(axis=1)
robust_features = complete_df.nsmallest(10, 'Rank_Variance')

print("\nMost Robust Features (low variance across methods):")
print(robust_features[['Feature', 'II_Rank', 'MI_Rank', 'DII_Rank', 'Rank_Variance']].to_string(index=False))

robust_features.to_csv('robust_features.csv', index=False)
print("\nSaved: robust_features.csv")

# Create comparison visualization
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80 + "\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Method Agreement Heatmap
ax = axes[0, 0]
corr_matrix = np.array([
    [1.0, corr_ii_mi, corr_ii_dii],
    [corr_ii_mi, 1.0, corr_mi_dii],
    [corr_ii_dii, corr_mi_dii, 1.0]
])
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
            xticklabels=['II', 'MI', 'DII'], yticklabels=['II', 'MI', 'DII'],
            ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Method Agreement (Spearman Correlation)', fontweight='bold')

# Plot 2: Stability Comparison
ax = axes[0, 1]
methods = ['II', 'MI', 'DII']
cvs = [cv_ii, cv_mi, cv_dii]
colors = ['#ff7f0e', '#2ca02c', '#d62728']
bars = ax.bar(methods, cvs, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Coefficient of Variation (lower = better)', fontweight='bold')
ax.set_title('Method Stability Comparison', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, cv in zip(bars, cvs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{cv:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 3: Rank Distribution
ax = axes[1, 0]
ax.boxplot([ii_rank, mi_rank, dii_rank], labels=['II', 'MI', 'DII'])
ax.set_ylabel('Feature Rank (lower = better)', fontweight='bold')
ax.set_title('Ranking Distribution by Method', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Plot 4: Feature Robustness
ax = axes[1, 1]
top_robust = complete_df.nsmallest(10, 'Rank_Variance')
ax.barh(range(len(top_robust)), top_robust['Rank_Variance'], color='#1f77b4', edgecolor='black')
ax.set_yticks(range(len(top_robust)))
ax.set_yticklabels(top_robust['Feature'], fontsize=9)
ax.set_xlabel('Rank Variance (lower = more robust)', fontweight='bold')
ax.set_title('Top 10 Most Robust Features', fontweight='bold')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('method_comparison_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: method_comparison_analysis.png")

# Summary
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if corr_ii_mi > 0.8 and corr_ii_dii > 0.8 and corr_mi_dii > 0.8:
    print("\nAll three methods have HIGH agreement (r > 0.8)")
    print("Result: ROBUST - All methods agree on feature importance")
    print("Recommendation: Use consensus ranking as it combines all methods")
elif corr_ii_mi > 0.6 or corr_ii_dii > 0.6 or corr_mi_dii > 0.6:
    print("\nMethods have MODERATE agreement (0.6 < r < 0.8)")
    print("Result: CONSENSUS needed - Different methods capture different aspects")
    print("Recommendation: Use consensus ranking from all three methods")
else:
    print("\nMethods have LOW agreement (r < 0.6)")
    print("Result: DIVERGENT - Methods select different important features")
    print("Recommendation: Combine all three methods for comprehensive analysis")

print(f"\nBest Method by Stability: {['II', 'MI', 'DII'][np.argmin([cv_ii, cv_mi, cv_dii])]}")
print(f"Most Robust Feature: {complete_df.loc[complete_df['Rank_Variance'].idxmin(), 'Feature']}")

print("\n" + "="*80)


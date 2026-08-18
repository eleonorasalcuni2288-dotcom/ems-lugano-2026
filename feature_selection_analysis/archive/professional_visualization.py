"""
Professional Visualizations for Feature Selection Analysis
Generates publication-ready graphics for statistical conference
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

print("\n" + "="*80)
print("GENERATING PROFESSIONAL VISUALIZATIONS")
print("="*80)

# Load data
print("\nLoading data...")
complete_df = pd.read_csv('ii_vs_mi_vs_dii_complete_comparison.csv')
consensus_df = pd.read_csv('consensus_ranking.csv')

# ===== FIGURE 1: Ranking Comparison (Scatter + Histogram) =====
print("\n[1/4] Creating ranking comparison visualizations...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Scatter plots
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(complete_df['II_Rank'], complete_df['MI_Rank'], alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
ax1.set_xlabel('Information Imbalance Rank', fontweight='bold')
ax1.set_ylabel('Mutual Information Rank', fontweight='bold')
ax1.set_title('II vs MI\n(r = 0.21)', fontweight='bold')
for i, txt in enumerate(complete_df['Feature'].head(5)):
    ax1.annotate(txt, (complete_df['II_Rank'].iloc[i], complete_df['MI_Rank'].iloc[i]), fontsize=8, alpha=0.7)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(complete_df['II_Rank'], complete_df['DII_Rank'], alpha=0.6, s=80, edgecolors='black', linewidth=0.5, color='orange')
ax2.set_xlabel('Information Imbalance Rank', fontweight='bold')
ax2.set_ylabel('Differential II Rank', fontweight='bold')
ax2.set_title('II vs DII\n(r = -0.21)', fontweight='bold')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(complete_df['MI_Rank'], complete_df['DII_Rank'], alpha=0.6, s=80, edgecolors='black', linewidth=0.5, color='green')
ax3.set_xlabel('Mutual Information Rank', fontweight='bold')
ax3.set_ylabel('Differential II Rank', fontweight='bold')
ax3.set_title('MI vs DII\n(r = -0.19)', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Histograms
ax4 = fig.add_subplot(gs[1, :])
ax4.hist(complete_df['II_Rank'], bins=10, alpha=0.5, label='II', color='blue', edgecolor='black')
ax4.hist(complete_df['MI_Rank'], bins=10, alpha=0.5, label='MI', color='green', edgecolor='black')
ax4.hist(complete_df['DII_Rank'], bins=10, alpha=0.5, label='DII', color='orange', edgecolor='black')
ax4.set_xlabel('Feature Rank', fontweight='bold')
ax4.set_ylabel('Frequency', fontweight='bold')
ax4.set_title('Distribution of Rankings across Methods', fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

# Box plots
ax5 = fig.add_subplot(gs[2, :])
box_data = [complete_df['II_Rank'], complete_df['MI_Rank'], complete_df['DII_Rank']]
bp = ax5.boxplot(box_data, labels=['II', 'MI', 'DII'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['blue', 'green', 'orange']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax5.set_ylabel('Feature Rank', fontweight='bold')
ax5.set_title('Ranking Distribution Comparison', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

plt.suptitle('Figure 1: Method Ranking Comparison', fontsize=14, fontweight='bold', y=0.995)
plt.savefig('fig1_ranking_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: fig1_ranking_comparison.png")

# ===== FIGURE 2: Feature Robustness Analysis =====
print("[2/4] Creating feature robustness visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Variance by feature
ax = axes[0, 0]
top_features = complete_df.nsmallest(15, 'All_Three_Consensus')
colors_robust = plt.cm.RdYlGn(np.linspace(0, 1, len(top_features)))
ax.barh(range(len(top_features)), top_features['Rank_Variance'], color=colors_robust, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'], fontsize=9)
ax.set_xlabel('Rank Variance (lower = more robust)', fontweight='bold')
ax.set_title('Feature Robustness\n(Variance across methods)', fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Heatmap of rankings
ax = axes[0, 1]
heatmap_data = complete_df.nsmallest(15, 'All_Three_Consensus')[['Feature', 'II_Rank', 'MI_Rank', 'DII_Rank']].set_index('Feature')
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Rank'})
ax.set_title('Top 15 Features: Ranking Heatmap', fontweight='bold')

# Consensus ranking
ax = axes[1, 0]
top_12 = complete_df.nsmallest(12, 'All_Three_Consensus')
colors_consensus = plt.cm.RdYlGn(np.linspace(0, 1, len(top_12)))
ax.barh(range(len(top_12)), top_12['All_Three_Consensus'], color=colors_consensus, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(top_12)))
ax.set_yticklabels(top_12['Feature'], fontsize=9)
ax.set_xlabel('Consensus Rank (lower = better)', fontweight='bold')
ax.set_title('Top 12 Features: Consensus Ranking', fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Method agreement pie chart
ax = axes[1, 1]
top_15_ii = set(complete_df.nsmallest(15, 'II_Rank')['Feature'])
top_15_mi = set(complete_df.nsmallest(15, 'MI_Rank')['Feature'])
top_15_dii = set(complete_df.nsmallest(15, 'DII_Rank')['Feature'])

at_least_2 = (top_15_ii & top_15_mi) | (top_15_ii & top_15_dii) | (top_15_mi & top_15_dii)
all_three = len(top_15_ii & top_15_mi & top_15_dii)
two_methods = len(at_least_2) - all_three
one_method = len(top_15_ii | top_15_mi | top_15_dii) - len(at_least_2)

sizes = [all_three, two_methods, one_method]
labels = [f'All 3\n({all_three})', f'2 Methods\n({two_methods})', f'1 Method\n({one_method})']
colors = ['#2ecc71', '#f39c12', '#e74c3c']
ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax.set_title('Feature Agreement (Top-15)', fontweight='bold')

plt.suptitle('Figure 2: Feature Robustness Analysis', fontsize=14, fontweight='bold', y=0.995)
plt.savefig('fig2_feature_robustness.png', dpi=300, bbox_inches='tight')
print("Saved: fig2_feature_robustness.png")

# ===== FIGURE 3: Method Characteristics =====
print("[3/4] Creating method characteristics visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Statistics
ax = axes[0, 0]
stats = {
    'Mean Rank': [complete_df['II_Rank'].mean(), complete_df['MI_Rank'].mean(), complete_df['DII_Rank'].mean()],
    'Std Dev': [complete_df['II_Rank'].std(), complete_df['MI_Rank'].std(), complete_df['DII_Rank'].std()],
    'Median': [complete_df['II_Rank'].median(), complete_df['MI_Rank'].median(), complete_df['DII_Rank'].median()]
}
x = np.arange(3)
width = 0.25
for i, (label, values) in enumerate(stats.items()):
    ax.bar(x + i*width, values, width, label=label, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Value', fontweight='bold')
ax.set_title('Ranking Statistics by Method', fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(['II', 'MI', 'DII'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Top 5 features comparison
ax = axes[0, 1]
top_5_ii = complete_df.nsmallest(5, 'II_Rank')
top_5_mi = complete_df.nsmallest(5, 'MI_Rank')
top_5_dii = complete_df.nsmallest(5, 'DII_Rank')

x_pos = np.arange(5)
width = 0.25
ax.bar(x_pos - width, top_5_ii['II_Rank'].values, width, label='II Method', edgecolor='black', linewidth=0.5)
ax.bar(x_pos, top_5_mi['MI_Rank'].values, width, label='MI Method', edgecolor='black', linewidth=0.5)
ax.bar(x_pos + width, top_5_dii['DII_Rank'].values, width, label='DII Method', edgecolor='black', linewidth=0.5)
ax.set_ylabel('Rank', fontweight='bold')
ax.set_title('Top 5 Features by Each Method', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['#1', '#2', '#3', '#4', '#5'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Variance distribution
ax = axes[1, 0]
ax.hist(complete_df['Rank_Variance'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
ax.axvline(complete_df['Rank_Variance'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {complete_df["Rank_Variance"].mean():.2f}')
ax.axvline(complete_df['Rank_Variance'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {complete_df["Rank_Variance"].median():.2f}')
ax.set_xlabel('Rank Variance', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Distribution of Feature Robustness', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Correlation matrix
ax = axes[1, 1]
corr_matrix = complete_df[['II_Rank', 'MI_Rank', 'DII_Rank']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, vmin=-1, vmax=1, 
            ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Method Correlation Matrix', fontweight='bold')

plt.suptitle('Figure 3: Method Characteristics', fontsize=14, fontweight='bold', y=0.995)
plt.savefig('fig3_method_characteristics.png', dpi=300, bbox_inches='tight')
print("Saved: fig3_method_characteristics.png")

# ===== FIGURE 4: Summary Infographic =====
print("[4/4] Creating summary infographic...")

fig = plt.figure(figsize=(14, 10))
fig.suptitle('Figure 4: Feature Selection Analysis - Summary', fontsize=16, fontweight='bold', y=0.98)

# Remove axes
ax = fig.add_subplot(111)
ax.axis('off')

# Text summary
summary_text = f"""
DATASET INFORMATION
• Total Features Analyzed: {len(complete_df)}
• Sample Size: 3000 observations
• Target Variable: Binary classification

METHODS APPLIED
1. Information Imbalance (II)
   - Distance-based ranking method
   - Measures feature predictive power
   
2. Mutual Information (MI)
   - Statistical dependency measure
   - Captures feature-target associations
   
3. Differential Information Imbalance (DII)
   - Local density variation method
   - Captures manifold structure

KEY FINDINGS
• Method Agreement: LOW (r < 0.6)
  - II vs MI: r = 0.21
  - II vs DII: r = -0.21
  - MI vs DII: r = -0.19

• Interpretation: Methods capture DIFFERENT aspects of feature importance
  - No single method is superior
  - Consensus ranking recommended

TOP 3 ROBUST FEATURES (Consensus)
1. RSI_14 - Consensus Rank: 6.67
2. Volume_SMA_Ratio_10 - Consensus Rank: 7.00
3. SMA_Ratio_200 - Consensus Rank: 8.00

MOST ROBUST FEATURE ACROSS ALL METHODS
• Feature: ema_ratio_26
• Rank Variance: 1.00 (most consistent)

RECOMMENDATION
Combine all three methods for comprehensive feature selection.
Use consensus ranking for robust and reliable feature importance.
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11, verticalalignment='top',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('fig4_summary_infographic.png', dpi=300, bbox_inches='tight')
print("Saved: fig4_summary_infographic.png")

print("\n" + "="*80)
print("ALL PROFESSIONAL VISUALIZATIONS CREATED")
print("="*80)

print("\nGenerated Files:")
print("1. fig1_ranking_comparison.png - Scatter plots and distributions")
print("2. fig2_feature_robustness.png - Feature analysis and agreement")
print("3. fig3_method_characteristics.png - Method statistics")
print("4. fig4_summary_infographic.png - Results summary")

print("\nReady for conference presentation!")
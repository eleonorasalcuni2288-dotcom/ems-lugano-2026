"""
COMPARISON OF ALL THREE FEATURE SELECTION METHODS
Information Imbalance (II) vs Mutual Information (MI) vs Differential Information Imbalance (DII)

This script loads the results from the two main analysis scripts and performs a 
comprehensive comparison of the three methods.
"""

import numpy as np
import pandas as pd
import sys


def load_results():
    """Load results from both analysis methods."""
    print("\n[1/4] Loading results from analyses...")
    
    try:
        consensus_df = pd.read_csv('consensus_ranking.csv')
        print("Loaded: consensus_ranking.csv (II and MI results)")
    except FileNotFoundError:
        print("Error: consensus_ranking.csv not found")
        print("Please run: python main_analysis.py")
        sys.exit(1)
    
    try:
        dii_df = pd.read_csv('ii_vs_dii_comparison.csv')
        print("Loaded: ii_vs_dii_comparison.csv (II and DII results)")
    except FileNotFoundError:
        print("Error: ii_vs_dii_comparison.csv not found")
        print("Please run: python my_dadapy_dii_analysis.py")
        sys.exit(1)
    
    return consensus_df, dii_df


def merge_results(consensus_df, dii_df):
    """Merge results from both analyses."""
    print("\n[2/4] Merging results from both methods...")
    
    ii_mi_data = consensus_df[['Feature', 'II_Rank', 'MI_Rank']].copy()
    ii_dii_data = dii_df[['Feature', 'II_Rank', 'DII_Rank']].copy()
    
    comparison = pd.merge(
        ii_mi_data, 
        ii_dii_data[['Feature', 'DII_Rank']], 
        on='Feature', 
        how='outer'
    )
    
    print(f"Merged data for {len(comparison)} features")
    return comparison


def calculate_consensus(comparison):
    """Calculate consensus rankings across all three methods."""
    print("\n[3/4] Calculating consensus rankings...")
    
    comparison['II_MI_Consensus'] = (
        comparison['II_Rank'] + comparison['MI_Rank']
    ) / 2
    
    comparison['II_DII_Consensus'] = (
        comparison['II_Rank'] + comparison['DII_Rank']
    ) / 2
    
    comparison['All_Three_Consensus'] = (
        comparison['II_Rank'] + 
        comparison['MI_Rank'] + 
        comparison['DII_Rank']
    ) / 3
    
    comparison = comparison.sort_values('All_Three_Consensus').reset_index(drop=True)
    
    print("Consensus calculated")
    return comparison


def display_results(comparison):
    """Display comparison results."""
    print("\n[4/4] Displaying results...\n")
    
    print("=" * 80)
    print("TOP 12 FEATURES - CONSENSUS OF ALL THREE METHODS")
    print("(Information Imbalance, Mutual Information, Differential Information Imbalance)")
    print("=" * 80)
    
    display_df = comparison.head(12)[[
        'Feature', 'II_Rank', 'MI_Rank', 'DII_Rank', 'All_Three_Consensus'
    ]]
    
    print("\n" + display_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("METHOD COMPARISON STATISTICS")
    print("=" * 80)
    
    print("\nAverage Rank by Method:")
    print(f"Information Imbalance (II):                {comparison['II_Rank'].mean():.2f}")
    print(f"Mutual Information (MI):                   {comparison['MI_Rank'].mean():.2f}")
    print(f"Differential Information Imbalance (DII): {comparison['DII_Rank'].mean():.2f}")
    
    print("\nTop 5 Feature Agreement:")
    
    top_5_ii = set(comparison.nsmallest(5, 'II_Rank')['Feature'].tolist())
    top_5_mi = set(comparison.nsmallest(5, 'MI_Rank')['Feature'].tolist())
    top_5_dii = set(comparison.nsmallest(5, 'DII_Rank')['Feature'].tolist())
    
    overlap_all_three = top_5_ii & top_5_mi & top_5_dii
    overlap_ii_mi = top_5_ii & top_5_mi
    overlap_ii_dii = top_5_ii & top_5_dii
    overlap_mi_dii = top_5_mi & top_5_dii
    
    print(f"Features in top 5 of all three methods: {len(overlap_all_three)}/5")
    if overlap_all_three:
        print(f"Features: {sorted(list(overlap_all_three))}")
    
    print(f"Features in top 5 of II and MI: {len(overlap_ii_mi)}/5")
    print(f"Features in top 5 of II and DII: {len(overlap_ii_dii)}/5")
    print(f"Features in top 5 of MI and DII: {len(overlap_mi_dii)}/5")
    
    return comparison


def save_results(comparison):
    """Save comparison results to CSV files."""
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80 + "\n")
    
    comparison.to_csv('ii_vs_mi_vs_dii_complete_comparison.csv', index=False)
    print("Saved: ii_vs_mi_vs_dii_complete_comparison.csv")
    
    summary = comparison.head(12)[[
        'Feature', 'II_Rank', 'MI_Rank', 'DII_Rank', 'All_Three_Consensus'
    ]]
    summary.to_csv('ii_vs_mi_vs_dii_top12.csv', index=False)
    print("Saved: ii_vs_mi_vs_dii_top12.csv")


def print_summary():
    """Print summary of the analysis."""
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    print("\nGenerated Rankings:")
    print("1. II_Rank: Information Imbalance ranking")
    print("2. MI_Rank: Mutual Information ranking")
    print("3. DII_Rank: Differential Information Imbalance ranking")
    
    print("\nGenerated Consensus Rankings:")
    print("1. II_MI_Consensus: Average of II and MI rankings")
    print("2. II_DII_Consensus: Average of II and DII rankings")
    print("3. All_Three_Consensus: Average of II, MI, and DII rankings")
    
    print("\nOutput Files:")
    print("1. ii_vs_mi_vs_dii_complete_comparison.csv (all 27 features)")
    print("2. ii_vs_mi_vs_dii_top12.csv (top 12 features)")
    
    print("\nInterpretation:")
    print("Lower consensus rank indicates feature importance across all methods.")
    print("Features with low rank in all three methods are most robust.")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("COMPARISON OF FEATURE SELECTION METHODS")
    print("Information Imbalance vs Mutual Information vs Differential Information Imbalance")
    print("=" * 80)
    
    consensus_df, dii_df = load_results()
    comparison = merge_results(consensus_df, dii_df)
    comparison = calculate_consensus(comparison)
    comparison = display_results(comparison)
    save_results(comparison)
    print_summary()


if __name__ == "__main__":
    main()
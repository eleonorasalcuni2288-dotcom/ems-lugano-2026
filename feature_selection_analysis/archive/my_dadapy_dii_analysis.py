"""
Differential Information Imbalance Implementation
Aligned with DADApy (sissa-data-science/DADApy)

References:
  - Information Imbalance: Glielmo et al., PNAS Nexus (2022)
  - Differentiable Information Imbalance: Wild et al., Nature Communications (2025)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')


def compute_dii_scores(X, y_true, lambda_factor=0.1):
    """
    Compute Differential Information Imbalance scores.

    For each feature i, computes DII(A_i -> B) where A_i is the 1D space
    of feature i and B is the ground truth space (y_true).

    Formula (Wild et al. 2025 / DADApy):
        c_ij = exp(-(dist_A(i,j) - min_dist_A(i)) / lambda) / sum_k exp(...)
        DII(A->B) = 2/N^2 * sum_{i,j} c_ij * rank_B(i,j)

    A lower DII score means the feature better preserves the structure of B.

    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
        Feature matrix
    y_true : array, shape (n_samples, n_components)
        Ground truth space (e.g., from PCA)
    lambda_factor : float
        Fraction of mean 1st-NN distance used as softmax scale. Default: 0.1.

    Returns:
    --------
    dii_scores : array, shape (n_features,)
        DII score for each feature (lower = better)
    """
    n_samples, n_features = X.shape
    dii_scores = np.zeros(n_features)

    # Rank matrix for ground truth space B (diagonal = max+1 to exclude self)
    dist_y = squareform(pdist(y_true, metric='euclidean'))
    np.fill_diagonal(dist_y, np.max(dist_y) + 1)
    rank_y = rankdata(dist_y, method='average', axis=1).astype(int)

    for i in range(n_features):
        # Distance matrix for feature i (diagonal = max+1 to exclude self)
        dist_x = squareform(pdist(X[:, i:i+1], metric='euclidean'))
        np.fill_diagonal(dist_x, np.max(dist_x) + 1)

        # Adaptive lambda: fraction of mean 1st-NN distance
        nn_dists = np.min(dist_x, axis=1)
        lambd = lambda_factor * np.mean(nn_dists)

        # Softmax weights c_ij (subtract min per row for numerical stability)
        min_dists = nn_dists[:, np.newaxis]
        exp_matrix = np.exp(-(dist_x - min_dists) / (lambd + 1e-10))
        np.fill_diagonal(exp_matrix, 0)  # c_ii = 0 (no self-contribution)
        rowsums = np.sum(exp_matrix, axis=1)[:, np.newaxis]
        c_matrix = exp_matrix / (rowsums + 1e-10)

        # DII = 2/N^2 * sum_{i,j} c_ij * rank_B(i,j)
        dii_scores[i] = 2.0 / n_samples ** 2 * np.sum(rank_y * c_matrix)

    return dii_scores


def main():
    """Main analysis function."""

    print("\n" + "="*70)
    print("DIFFERENTIAL INFORMATION IMBALANCE (DII) ANALYSIS")
    print("Aligned with DADApy (Wild et al. 2025)")
    print("="*70)

    # Step 1: Load Data
    print("\n[1/6] Loading data...")
    try:
        df = pd.read_csv('train2.csv', nrows=5000)
        print(f"Loaded {len(df)} rows x {len(df.columns)} columns")
    except FileNotFoundError:
        print("Error: train2.csv not found")
        return

    # Step 2: Prepare Data
    print("\n[2/6] Preparing data...")
    numeric_df = df.select_dtypes(include=[np.number])
    target_col = 'target'
    feature_cols = [c for c in numeric_df.columns if c != target_col and c != 'id']

    X = numeric_df[feature_cols].values.astype(np.float64)
    y = numeric_df[target_col].values.astype(np.float64)

    X[np.isnan(X)] = 0.0
    y[np.isnan(y)] = 0.0

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"Data shape: {X.shape}")
    print(f"Features: {len(feature_cols)}")

    # Step 3: Create Ground Truth
    print("\n[3/6] Creating ground truth space...")
    pca = PCA(n_components=3)
    y_true = pca.fit_transform(X)
    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")

    # Step 4: Compute DII
    print("\n[4/6] Computing DII scores...")
    dii_scores = compute_dii_scores(X, y_true, lambda_factor=0.1)

    print(f"DII scores computed")
    print(f"  Min: {dii_scores.min():.6f}")
    print(f"  Max: {dii_scores.max():.6f}")
    print(f"  Mean: {dii_scores.mean():.6f}")

    # Step 5: Compute II
    # Formula (Glielmo et al. 2022 / DADApy):
    #   II(A->B) = 2/N^2 * sum_i rank_B(i, NN_A(i))
    # where NN_A(i) is the nearest neighbor of i in space A.
    print("\n[5/6] Computing Information Imbalance...")

    # Precompute rank matrix for ground truth space (shared across all features)
    dist_Y = squareform(pdist(y_true))
    np.fill_diagonal(dist_Y, np.max(dist_Y) + 1)
    ranks_Y = rankdata(dist_Y, method='average', axis=1).astype(int)

    ii_scores = []
    for i in range(X.shape[1]):
        dist_X = squareform(pdist(X[:, i:i+1]))
        np.fill_diagonal(dist_X, np.max(dist_X) + 1)  # exclude self

        score = 0.0
        for j in range(len(X)):
            nearest = np.argmin(dist_X[j])  # safe: diagonal is max+1
            score += ranks_Y[j, nearest]

        ii_scores.append(2.0 * score / (len(X) ** 2))

    print(f"II scores computed ({len(ii_scores)} features)")

    # Step 6: Create Comparison
    print("\n[6/6] Creating comparison...")

    comparison_df = pd.DataFrame({
        'Feature': feature_cols,
        'II_Score': ii_scores,
        'II_Rank': np.argsort(ii_scores) + 1,       # lower II = better
        'DII_Score': dii_scores,
        'DII_Rank': np.argsort(dii_scores) + 1,     # lower DII = better
    })

    comparison_df['Avg_Rank'] = (comparison_df['II_Rank'] + comparison_df['DII_Rank']) / 2
    comparison_df = comparison_df.sort_values('Avg_Rank')

    print("\n" + "="*70)
    print("COMPARISON: II vs DII Rankings")
    print("="*70)
    print("\nTop 12 Features (by Consensus of II + DII):")
    print(comparison_df.head(12)[['Feature', 'II_Rank', 'DII_Rank', 'Avg_Rank']].to_string(index=False))

    # Save results
    comparison_df.to_csv('ii_vs_dii_comparison.csv', index=False)
    print("\nSaved: ii_vs_dii_comparison.csv")

    print("\n" + "="*70)
    print("DII-Specific Top Features")
    print("="*70)
    dii_top = comparison_df.nsmallest(12, 'DII_Rank')[['Feature', 'DII_Score', 'DII_Rank']]
    print(dii_top.to_string(index=False))

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    print("\nOutput Files:")
    print("1. ii_vs_dii_comparison.csv - II vs DII comparison")

    print("\nMethod Summary:")
    print("1. Information Imbalance (II):             II(A->B) = 2/N^2 * sum_i rank_B(i, NN_A(i))")
    print("2. Differential Information Imbalance (DII): DII(A->B) = 2/N^2 * sum_{i,j} c_ij * rank_B(i,j)")
    print("   (softmax weights c_ij with scale lambda = lambda_factor * mean(1st-NN dist))")
    print("3. Consensus: Average ranking of II and DII (lower rank = better feature)")


if __name__ == "__main__":
    main()

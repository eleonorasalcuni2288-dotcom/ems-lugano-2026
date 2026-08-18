"""
Differential Information Imbalance Analysis - CORRECT VERSION
Using DADApy library (https://dadapy.readthedocs.io/)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

try:
    from dadapy import Data
    print("✓ DADApy imported successfully!")
    HAS_DADAPY = True
except ImportError:
    print("✗ DADApy not found")
    HAS_DADAPY = False

if HAS_DADAPY:
    def main():
        print("\n" + "="*70)
        print("DIFFERENTIAL INFORMATION IMBALANCE (DII) ANALYSIS")
        print("Using DADApy Library (Correct Implementation)")
        print("="*70)
        
        print("\n[1/7] Loading data...")
        df = pd.read_csv('train2.csv', nrows=3000)
        print(f"Loaded {len(df)} rows x {len(df.columns)} columns")
        
        print("\n[2/7] Preparing data...")
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
        
        print("\n[3/7] Creating ground truth space...")
        pca = PCA(n_components=3)
        y_true = pca.fit_transform(X)
        print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")
        
        print("\n[4/7] Computing DII using DADApy...")
        
        dii_scores_dict = {}
        
        for i, feature_name in enumerate(feature_cols):
            print(f"  Processing feature {i+1}/{len(feature_cols)}: {feature_name}")
            
            try:
                feature_data = X[:, i:i+1]
                dad = Data(feature_data)
                dad.compute_density_kNN(k=10)
                feature_dii = dad.log_den
                dii_scores_dict[feature_name] = np.mean(feature_dii)
            except Exception as e:
                print(f"    ⚠ Skipped (error: {type(e).__name__})")
                dii_scores_dict[feature_name] = np.nan
        
        print(f"\nDII scores computed for {len([v for v in dii_scores_dict.values() if not np.isnan(v)])}/{len(feature_cols)} features")
        
        print("\n[5/7] Computing Information Imbalance...")
        
        ii_scores = []
        for i in range(X.shape[1]):
            dist_X = squareform(pdist(X[:, i:i+1]))
            dist_Y = squareform(pdist(y_true))
            
            ranks_X = np.argsort(np.argsort(dist_X, axis=1), axis=1)
            ranks_Y = np.argsort(np.argsort(dist_Y, axis=1), axis=1)
            
            score = 0.0
            for j in range(len(X)):
                nearest = np.argmin(ranks_X[j])
                score += ranks_Y[j, nearest]
            
            ii_scores.append(2.0 * score / (len(X) ** 2))
        
        print(f"II scores computed ({len(ii_scores)} features)")
        
        print("\n[6/7] Creating comparison...")
        
        dii_scores = np.array([dii_scores_dict[f] for f in feature_cols])
        dii_scores_valid = dii_scores[~np.isnan(dii_scores)]
        
        if len(dii_scores_valid) > 0:
            dii_scores_norm = np.full_like(dii_scores, np.nan)
            valid_min = np.min(dii_scores_valid)
            valid_max = np.max(dii_scores_valid)
            valid_mask = ~np.isnan(dii_scores)
            dii_scores_norm[valid_mask] = (dii_scores[valid_mask] - valid_min) / (valid_max - valid_min + 1e-10)
        else:
            dii_scores_norm = np.full_like(dii_scores, np.nan)
        
        comparison_df = pd.DataFrame({
            'Feature': feature_cols,
            'II_Score': ii_scores,
            'II_Rank': np.argsort(ii_scores) + 1,
            'DII_Score': dii_scores_norm,
        })
        
        valid_dii = ~np.isnan(comparison_df['DII_Score'])
        comparison_df.loc[valid_dii, 'DII_Rank'] = np.argsort(-comparison_df.loc[valid_dii, 'DII_Score'].values) + 1
        
        comparison_df['Avg_Rank'] = (comparison_df['II_Rank'] + comparison_df['DII_Rank'].fillna(comparison_df['II_Rank'])) / 2
        comparison_df = comparison_df.sort_values('Avg_Rank')
        
        print("\n" + "="*70)
        print("COMPARISON: II vs DII Rankings")
        print("="*70)
        print("\nTop 12 Features:")
        print(comparison_df.head(12)[['Feature', 'II_Rank', 'DII_Rank', 'Avg_Rank']].to_string(index=False))
        
        comparison_df.to_csv('ii_vs_dii_comparison.csv', index=False)
        print("\nSaved: ii_vs_dii_comparison.csv")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)

    if __name__ == "__main__":
        main()
else:
    print("ERROR: DADApy not installed!")

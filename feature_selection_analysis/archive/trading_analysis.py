"""
MY TRADING ANALYSIS SCRIPT
Start here to run the complete feature selection analysis
Ready to use in VS Code - just copy this file to your project folder!
"""
 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
 
print("\n" + "="*70)
print("FEATURE SELECTION ANALYSIS - MY VERSION")
print("="*70)
 
# ===== STEP 1: Load Data =====
print("\n[1/5] Loading data...")
try:
    df = pd.read_csv('train2.csv', nrows=5000)
    print(f"✓ Loaded {len(df)} rows × {len(df.columns)} columns")
except FileNotFoundError:
    print("✗ Error: train2.csv not found!")
    print("  Make sure train2.csv is in the same folder as this script")
    exit(1)
 
# ===== STEP 2: Prepare Data =====
print("\n[2/5] Preparing data...")
numeric_df = df.select_dtypes(include=[np.number])
target_col = 'target'
feature_cols = [c for c in numeric_df.columns if c != target_col and c != 'id']
 
X = numeric_df[feature_cols].values.astype(np.float64)
y = numeric_df[target_col].values.astype(np.float64)
 
# Handle NaN
X[np.isnan(X)] = 0.0
y[np.isnan(y)] = 0.0
 
# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = (y - y.mean()) / (y.std() + 1e-10)
 
print(f"✓ Data shape: {X.shape}")
print(f"✓ Features: {len(feature_cols)} total")
print(f"  Samples: {feature_cols[:5]}...")
 
# ===== STEP 3: Consensus Ranking =====
print("\n[3/5] Reading consensus ranking results...")
 
try:
    consensus = pd.read_csv('consensus_ranking.csv')
    print(f"✓ Read consensus_ranking.csv")
    
    print("\n📊 Top 10 Features (Consensus Ranking):")
    print(consensus.head(10)[['Feature', 'Avg_Rank']].to_string(index=False))
    
except FileNotFoundError:
    print("✗ consensus_ranking.csv not found")
    print("  Make sure you have all output CSV files")
    consensus = None
 
# ===== STEP 4: Visualize Results =====
print("\n[4/5] Creating visualization...")
 
if consensus is not None:
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        top_12 = consensus.head(12).sort_values('Avg_Rank', ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(top_12)))
 
        ax.barh(range(len(top_12)), top_12['Avg_Rank'], 
               color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(top_12)))
        ax.set_yticklabels(top_12['Feature'], fontsize=10)
        ax.set_xlabel('Average Rank (Lower = Better)', fontsize=11)
        ax.set_title('Top 12 Features - Consensus Ranking\n(Information Imbalance + Mutual Information)', 
                    fontsize=12, fontweight='bold')
        ax.invert_xaxis()
        ax.grid(axis='x', alpha=0.3)
 
        plt.tight_layout()
        plt.savefig('my_consensus_ranking.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: my_consensus_ranking.png")
        plt.show()
        
    except Exception as e:
        print(f"✗ Error creating visualization: {e}")
 
# ===== STEP 5: Strategy Preview =====
print("\n[5/5] Strategy signal preview...")
 
def generate_basic_signals(df):
    """Generate simple trading signals from features"""
    signals = np.zeros(len(df))
    
    # Check if required features exist
    if 'bb_position' in df.columns and 'rsi_14' in df.columns:
        # Oversold signal
        oversold = (df['bb_position'] < -0.6)
        signals[oversold] = 1.0
        
        # Overbought signal
        overbought = (df['bb_position'] > 0.6)
        signals[overbought] = -1.0
    
    return signals
 
try:
    signals = generate_basic_signals(numeric_df)
    
    buy_count = np.sum(signals > 0)
    sell_count = np.sum(signals < 0)
    neutral_count = np.sum(signals == 0)
    
    print(f"✓ Generated {len(signals)} signal values")
    print(f"  - Buy signals:   {buy_count:6d} ({100*buy_count/len(signals):5.1f}%)")
    print(f"  - Sell signals:  {sell_count:6d} ({100*sell_count/len(signals):5.1f}%)")
    print(f"  - Neutral:       {neutral_count:6d} ({100*neutral_count/len(signals):5.1f}%)")
    
except Exception as e:
    print(f"✗ Error generating signals: {e}")
 
print("\n" + "="*70)
print("✓ ANALYSIS COMPLETE!")
print("="*70)
 
print("\n📚 Next steps:")
print("  1. View my_consensus_ranking.png for top features")
print("  2. Read QUICK_START.md for strategy implementation")
print("  3. Copy my_trading_strategy.py for backtesting")
print("  4. Run my_walking_validation.py for validation")
 
print("\n💡 Key findings:")
if consensus is not None:
    top_5 = consensus.head(5)['Feature'].tolist()
    print(f"  Top 5 features: {', '.join(top_5)}")
 
print("\n🚀 You can now:")
print("  - Modify this script to test different features")
print("  - Read the documentation in QUICK_START.md")
print("  - Implement your strategy with BACKTESTING_FRAMEWORK.py")
 
print("\n" + "="*70)
 
import pandas as pd

df = pd.read_csv('ii_vs_mi_vs_dii_complete_comparison.csv')
df['Rank_Variance'] = df[['II_Rank', 'MI_Rank', 'DII_Rank']].var(axis=1)
df.to_csv('ii_vs_mi_vs_dii_complete_comparison.csv', index=False)
print("Added Rank_Variance to CSV")

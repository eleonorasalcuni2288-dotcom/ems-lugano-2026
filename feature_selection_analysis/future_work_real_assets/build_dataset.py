"""
Real-asset feature selection dataset -- exploratory future work, not part
of the poster. Builds a daily panel of asset returns (features) and next-day
VIX direction (target), designed around interaction/divergence pairs (the
real-market analog of the synthetic benchmark's XOR synergy pair), not
comovement pairs -- see run_methods.py for why that distinction matters:
two features that trend together are redundant, not synergistic; the pairs
below are chosen because they are expected to move in OPPOSITE directions,
and it is that divergence, not either asset alone, that should carry signal.
"""
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. Asset list (p ~ 28) ---
tickers = {
    # Target (excluded from features)
    "VIX": "^VIX",
    # Interaction/divergence pairs (8 assets, 4 pairs)
    "GC=F": "GC=F",         # Gold (safe-haven)
    "HD": "HD",              # Home Depot (cyclical growth, risk-on counterpart to gold)
    "CL=F": "CL=F",          # Oil (input cost)
    "DAL": "DAL",             # Delta Air Lines (cost-sensitive to oil)
    "EURUSD=X": "EURUSD=X",  # EUR/USD
    "SIEGY": "SIEGY",         # Siemens (export exposure to EUR strength)
    "GS": "GS",               # Goldman Sachs (rate-regime sensitive)
    "CRM": "CRM",             # Salesforce (long-duration growth, rate-sensitive the other way)
    # Control assets, no designed interaction with each other (20 assets)
    "KO": "KO", "JNJ": "JNJ", "NEE": "NEE", "VZ": "VZ", "PG": "PG",
    "AAPL": "AAPL", "MSFT": "MSFT", "AMZN": "AMZN", "GOOGL": "GOOGL",
    "META": "META", "NVDA": "NVDA", "XOM": "XOM", "JPM": "JPM",
    "WMT": "WMT", "DIS": "DIS", "PEP": "PEP", "UNH": "UNH",
    "MA": "MA", "V": "V", "T": "T",
}

start_date = "2023-01-01"
end_date = "2025-12-31"

print("Downloading data from Yahoo Finance...")
data = {}
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if df.empty:
            print(f"  no data for {name} ({ticker})")
            continue
        # yfinance sometimes returns MultiIndex columns even for a single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Close" in df.columns:
            df = df[["Close"]].rename(columns={"Close": name})
        else:
            print(f"  Close column not found for {name} ({ticker})")
            continue
        data[name] = df
    except Exception as e:
        print(f"  download error {name} ({ticker}): {e}")

print("Merging assets on shared date index...")
combined = pd.concat(data.values(), axis=1)
combined = combined.dropna(how="all")

print("Computing daily returns...")
returns = combined.pct_change().dropna()
returns.columns = [f"r_{col}" for col in returns.columns]
returns = returns.drop(columns=["r_VIX"], errors="ignore")  # VIX is the target, not a feature

print("Building target: VIX direction at t+1 (0/1 encoding, matching the rest of the project)...")
vix = combined["VIX"]
vix_t1 = vix.shift(-1)
target = (vix_t1 > vix).astype(int).rename("VIX_t+1")
target = target.dropna()

print("Aligning target and features (rows with any missing asset are dropped)...")
final_data = pd.concat([returns, target], axis=1).dropna()
final_data.index.name = "date"
final_data = final_data.reset_index()

print(f"Final size: {len(final_data)} rows (target range: 500-750)")
if len(final_data) < 500:
    print("  WARNING: below the minimum target size -- check trading-calendar mismatches across assets.")

output_filename = "dataset_feature_selection_2023_2025.csv"
final_data.to_csv(output_filename, index=False)
print(f"Saved: {output_filename}")

feat_cols = [c for c in final_data.columns if c.startswith("r_")]
print(f"\nFeatures included (p = {len(feat_cols)}):")
print(feat_cols)
print(f"\nTarget balance: {final_data['VIX_t+1'].value_counts(normalize=True).to_dict()}")
print("\n--- Preview ---")
print(final_data.head())

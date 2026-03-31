import sys; sys.path.insert(0, '.')
from data.bloomberg_fetcher import OfflineDataLoader
from data.cot_fetcher import COTFetcher
from features.indicators import compute_all_features

# Load cached data from Step 1 & 2
loader = OfflineDataLoader()
df = loader.load("vix_strategy_data.parquet")
cot_df = COTFetcher().fetch_all(start_year=2022)

# Compute all features
df_feat = compute_all_features(df, cot_df=cot_df)

print(f"Shape: {df_feat.shape}")
print(f"\nSample features (last 5 rows):")
cols = ["VIX_Spot", "TS_VIX_VIX3M_Ratio", "TS_Slope_UX2_UX1", "VIX_ZScore", 
        "VRP_Simple", "VVIX", "VVIX_Divergence", "COT_AM_Net", "COT_AM_Pctl_3yr"]
available = [c for c in cols if c in df_feat.columns]
print(df_feat[available].tail())
df_feat.to_parquet("outputs/features_dataset.parquet")
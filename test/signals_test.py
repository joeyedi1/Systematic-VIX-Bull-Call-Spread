import sys; sys.path.insert(0, '.')
import pandas as pd
from signals.composite_score import CompositeSignal

df_regime = pd.read_parquet("outputs/features_dataset.parquet")

# If regime columns aren't saved yet, re-run HMM first:
if "Regime" not in df_regime.columns:
    from regime.hmm_classifier import RegimeClassifier
    clf = RegimeClassifier()
    df_regime = clf.fit_predict(df_regime)

df_regime.to_parquet("outputs/signals_dataset.parquet")

signal = CompositeSignal()
df_signals = signal.compute(df_regime)

print(f"Entry signals: {df_signals['Signal_Entry'].sum()} days")
print(f"\nScore distribution:")
print(df_signals["Signal_Score"].describe())

print(f"\nSub-scores:")
sub_cols = [c for c in df_signals.columns if c.startswith("SubScore_")]
print(df_signals[sub_cols].describe())

print(f"\nEntry signal dates (first 10):")
entries = df_signals[df_signals["Signal_Entry"] == True]
print(entries[["VIX_Spot", "Signal_Score", "Regime"]].head(10))
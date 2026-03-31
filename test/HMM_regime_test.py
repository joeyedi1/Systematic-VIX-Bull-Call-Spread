import sys; sys.path.insert(0, '.')
import pandas as pd
from regime.hmm_classifier import RegimeClassifier

df_feat = pd.read_parquet("outputs/features_dataset.parquet")

clf = RegimeClassifier()
df_regime = clf.fit_predict(df_feat)

print("Regime distribution:")
print(df_regime["Regime"].value_counts().sort_index())

print("\nTransition matrix:")
print(clf.get_transition_matrix())

print("\nRegime parameters:")
params = clf.get_regime_params()
for name, p in params.items():
    print(f"  {name}: duration={p['expected_duration_days']:.0f}d, self_prob={p['self_transition_prob']:.3f}")

print("\nSample (last 10 rows):")
print(df_regime[["VIX_Spot", "Regime", "Regime_Prob_LowVol", "Regime_Prob_HighVol"]].tail(10))
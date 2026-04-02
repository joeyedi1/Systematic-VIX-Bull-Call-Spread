import sys; sys.path.insert(0, '.')
import pandas as pd
from pathlib import Path

from features.indicators import compute_all_features
from regime.hmm_classifier import RegimeClassifier
from signals.composite_score import CompositeSignal
from backtest.engine import BacktestEngine

# Load cached raw data
df = pd.read_parquet("outputs/cache/vix_strategy_data.parquet")
cot_path = Path("outputs/cache/cot_vix_data.parquet")
cot_df = pd.read_parquet(cot_path) if cot_path.exists() else None

# Recompute features (picks up EMA fix)
df = compute_all_features(df, cot_df=cot_df)

# Recompute regime
clf = RegimeClassifier()
df = clf.fit_predict(df)

# Recompute signals (picks up weight + threshold changes)
signal = CompositeSignal()
df = signal.compute(df)

# Save
df.to_parquet("outputs/signals_dataset.parquet")

# Run backtest (picks up unconditional exit + cooldown)
engine = BacktestEngine()
result = engine.run(df)
engine.print_summary(result)

# Save trades
output_dir = Path("outputs")
trades_df = pd.DataFrame(result.trades)
if not trades_df.empty:
    trades_df.to_csv(output_dir / "backtest_trades.csv", index=False)

print("\nDone. Now run: python notebooks/hedge_effectiveness.py")
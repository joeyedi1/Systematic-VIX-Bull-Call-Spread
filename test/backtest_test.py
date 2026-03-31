import sys; sys.path.insert(0, '.')
import pandas as pd
from signals.composite_score import CompositeSignal
from regime.hmm_classifier import RegimeClassifier
from backtest.engine import BacktestEngine

# Load features
df = pd.read_parquet("outputs/features_dataset.parquet")

# Re-run regime + signals (to ensure columns are present)
clf = RegimeClassifier()
df = clf.fit_predict(df)

signal = CompositeSignal()
df = signal.compute(df)

# Save complete dataset
df.to_parquet("outputs/signals_dataset.parquet")

# Run backtest
engine = BacktestEngine()
result = engine.run(df)
engine.print_summary(result)

# Save trades
trades_df = pd.DataFrame(result.trades)
if not trades_df.empty:
    trades_df.to_csv("outputs/backtest_trades.csv", index=False)
    print(f"\nTrades saved to outputs/backtest_trades.csv")
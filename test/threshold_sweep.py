"""
test/threshold_sweep.py
=======================
Sweep VIX momentum threshold values to find the best post-spike recovery filter.

Thresholds tested: [None (baseline), -0.03, -0.05, -0.08, -0.10]
Metric: (VIX_SMA10_today - VIX_SMA10_5d_ago) / VIX_SMA10_5d_ago
        If < threshold → block entry that day

Usage:
    python test/threshold_sweep.py
"""

import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from pathlib import Path

from features.indicators import compute_all_features
from regime.hmm_classifier import RegimeClassifier
from signals.composite_score import CompositeSignal
from backtest.engine import BacktestEngine
from config.settings import BacktestConfig, SIGNAL

# ------------------------------------------------------------------
# 1. Build the full signals dataset (identical to full_rerun.py)
# ------------------------------------------------------------------
print("Loading raw data and computing features...")
df = pd.read_parquet("outputs/cache/vix_strategy_data.parquet")
cot_path = Path("outputs/cache/cot_vix_data.parquet")
cot_df = pd.read_parquet(cot_path) if cot_path.exists() else None

df = compute_all_features(df, cot_df=cot_df)

clf = RegimeClassifier()
df = clf.fit_predict(df)

signal = CompositeSignal()
df = signal.compute(df)

# Verify the slope column was produced
if "VIX_SMA10_Slope_5d" not in df.columns:
    raise RuntimeError("VIX_SMA10_Slope_5d missing from features — check indicators.py")

print(f"Dataset ready: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
print()

# ------------------------------------------------------------------
# 2. Run backtest for each threshold
# ------------------------------------------------------------------
THRESHOLDS = [None, -0.03, -0.05, -0.08, -0.10]
rows = []

for thresh in THRESHOLDS:
    label = "baseline" if thresh is None else f"{thresh:+.2f}"
    config = BacktestConfig(vix_momentum_threshold=thresh)
    engine = BacktestEngine(config=config, signal_config=SIGNAL)
    result = engine.run(df)

    rows.append({
        "threshold":    label,
        "trades":       result.total_trades,
        "win_pct":      result.win_rate,
        "avg_win":      result.avg_win,
        "avg_loss":     result.avg_loss,
        "pf":           result.profit_factor,
        "total_pnl":    result.total_pnl,
        "max_dd":       result.max_drawdown,
        "sharpe":       result.sharpe_ratio,
        "avg_hold":     result.avg_holding_days,
        "trades_list":  result.trades,
    })

# ------------------------------------------------------------------
# 3. Print comparison table
# ------------------------------------------------------------------
COL = 11
SEP = "-" * 100

print("=" * 100)
print("VIX MOMENTUM FILTER THRESHOLD SWEEP")
print("=" * 100)
print(
    f"{'Threshold':>10} | {'Trades':>6} | {'Win%':>6} | "
    f"{'AvgWin':>8} | {'AvgLoss':>9} | {'PF':>5} | "
    f"{'TotalPnL':>10} | {'MaxDD':>10} | {'Sharpe':>7} | {'Hold':>5}"
)
print(SEP)

for r in rows:
    pf_str = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "  inf"
    print(
        f"{r['threshold']:>10} | {r['trades']:>6} | {r['win_pct']:>5.1f}% | "
        f"+${r['avg_win']:>6.2f} | ${r['avg_loss']:>8.2f} | {pf_str:>5} | "
        f"${r['total_pnl']:>9.2f} | ${r['max_dd']:>9.2f} | {r['sharpe']:>7.2f} | {r['avg_hold']:>4.0f}d"
    )

print("=" * 100)

# ------------------------------------------------------------------
# 4. Per-threshold trade logs
# ------------------------------------------------------------------
for r in rows:
    print(f"\n--- Trade Log: threshold = {r['threshold']} ---")
    for t in r["trades_list"]:
        sign = "+" if t["pnl"] >= 0 else ""
        print(
            f"  #{t['id']:3d} | {t['entry_date']} -> {t['exit_date']} | "
            f"C{t['long_strike']}/C{t['short_strike']} | "
            f"${t['entry_price']:.2f} -> ${t['exit_price']:.2f} | "
            f"{sign}${t['pnl']:.2f} ({sign}{t['pnl_pct']:.0f}%) | "
            f"{t['exit_reason']}"
        )
    if not r["trades_list"]:
        print("  (no trades)")

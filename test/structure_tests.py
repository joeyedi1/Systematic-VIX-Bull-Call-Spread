"""
test/structure_tests.py
=======================
Structure tests: compare spread width, scale-out, and outright call configs.

Baseline: full system with hybrid pricing (mid entry, bid/ask exit), -0.03 momentum filter.

Test 1: 5-point spread, NO scale-out, 50% max-profit exit
Test 2: 10-point spread, NO scale-out, 50% max-profit exit
Test 3: Outright 30-delta call, 100% gain target exit

All tests use:
  - HMM + composite score + -0.03 momentum filter
  - Hybrid pricing (mid entry, bid/ask exit)
  - Same signal_config (exit thresholds stay at 30% / 60% where used)
  - first_exit_pct overridden to 0.50 for Tests 1+2 (single full exit at 50%)

Usage:
    python test/structure_tests.py
"""

import sys; sys.path.insert(0, '.')
import copy
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import replace

from features.indicators import compute_all_features
from regime.hmm_classifier import RegimeClassifier
from signals.composite_score import CompositeSignal
from backtest.engine import BacktestEngine
from config.settings import (
    BacktestConfig, SignalConfig, StrikeConfig, SIGNAL, STRIKE
)

# ============================================================
# 1.  Build signals dataset (once)
# ============================================================
print("Building signals dataset ...")
df = pd.read_parquet("outputs/cache/vix_strategy_data.parquet")
cot_path = Path("outputs/cache/cot_vix_data.parquet")
cot_df   = pd.read_parquet(cot_path) if cot_path.exists() else None

df = compute_all_features(df, cot_df=cot_df)
df = RegimeClassifier().fit_predict(df)
df = CompositeSignal().compute(df)

print(f"Ready: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
print()

# ============================================================
# 2.  Shared base config
# ============================================================
BASE_ENG = dict(
    vix_momentum_threshold=-0.03,
    execution_mode="hybrid",
    scale_out=True,
    position_type="spread",
)

# signal_config with 50% single-exit (replaces 30%/60% scale-out for Tests 1+2)
SIGNAL_50 = replace(SIGNAL,
    first_exit_pct=0.50,
    second_exit_pct=0.50,   # irrelevant when scale_out=False, but set cleanly
)

# signal_config for outright call: 100% gain target
# max_profit is set to entry_cost * 1.0 in engine, so first_exit_pct=1.0 triggers
# when current_price >= entry_price + 1.0 * max_profit = 2 * entry_price
SIGNAL_CALL = replace(SIGNAL,
    first_exit_pct=1.0,
    second_exit_pct=1.0,
    stop_loss_pct=0.70,
    time_stop_dte=10,
)

# ============================================================
# 3.  Config table
# ============================================================
CONFIGS = [
    {
        "name": "Baseline: 5pt spread, scale-out 30/60",
        "eng": BacktestConfig(**BASE_ENG),
        "sig": SIGNAL,
        "str": STRIKE,
    },
    {
        "name": "Test 1: 5pt spread, no scale-out, 50% exit",
        "eng": BacktestConfig(**{**BASE_ENG, "scale_out": False}),
        "sig": SIGNAL_50,
        "str": STRIKE,
    },
    {
        "name": "Test 2: 10pt spread, no scale-out, 50% exit",
        "eng": BacktestConfig(**{**BASE_ENG, "scale_out": False}),
        "sig": SIGNAL_50,
        "str": replace(STRIKE, spread_width_override=10, max_spread_cost=6.00),
    },
    {
        "name": "Test 3: Outright 30d call, 100% gain",
        "eng": BacktestConfig(**{**BASE_ENG,
                                 "scale_out": False,
                                 "position_type": "call",
                                 "call_profit_target_pct": 1.0}),
        "sig": SIGNAL_CALL,
        "str": STRIKE,
    },
]

# ============================================================
# 4.  Run
# ============================================================
results = []
for cfg in CONFIGS:
    print(f"Running: {cfg['name']} ...")
    engine = BacktestEngine(
        config=cfg["eng"],
        signal_config=cfg["sig"],
        strike_config=cfg["str"],
    )
    result = engine.run(df)
    results.append({
        "name":       cfg["name"],
        "trades":     result.total_trades,
        "win_pct":    result.win_rate,
        "avg_win":    result.avg_win,
        "avg_loss":   result.avg_loss,
        "pf":         result.profit_factor,
        "total_pnl":  result.total_pnl,
        "max_dd":     result.max_drawdown,
        "sharpe":     result.sharpe_ratio,
        "avg_hold":   result.avg_holding_days,
        "trade_list": result.trades,
    })
    print(f"  -> {result.total_trades} trades, {result.win_rate:.1f}% win, "
          f"PF={result.profit_factor:.2f}, PnL=${result.total_pnl:.2f}")

print()

# ============================================================
# 5.  Summary table
# ============================================================
W = 110
print("=" * W)
print("STRUCTURE TEST RESULTS  |  Hybrid pricing, -0.03 momentum filter, HMM+score")
print("=" * W)
print(
    f"{'Config':42} | {'Trades':>6} | {'Win%':>6} | "
    f"{'AvgWin':>8} | {'AvgLoss':>9} | {'PF':>5} | "
    f"{'TotalPnL':>10} | {'MaxDD':>10} | {'Sharpe':>7} | {'Hold':>5}"
)
print("-" * W)
for r in results:
    pf_str = f"{r['pf']:.2f}" if r["pf"] != float("inf") else "   inf"
    print(
        f"{r['name']:42} | {r['trades']:>6} | {r['win_pct']:>5.1f}% | "
        f"+${r['avg_win']:>6.2f} | ${r['avg_loss']:>8.2f} | {pf_str:>5} | "
        f"${r['total_pnl']:>9.2f} | ${r['max_dd']:>9.2f} | "
        f"{r['sharpe']:>7.2f} | {r['avg_hold']:>4.0f}d"
    )
print("=" * W)

# ============================================================
# 6.  Per-config trade logs
# ============================================================
for r in results:
    print(f"\n{'='*75}")
    print(f"  {r['name']}")
    print(f"{'='*75}")
    for t in r["trade_list"]:
        sign = "+" if t["pnl"] >= 0 else ""
        print(
            f"  #{t['id']:3d} | {t['entry_date']} -> {t['exit_date']} | "
            f"C{t['long_strike']}/C{t['short_strike']} | "
            f"${t['entry_price']:.2f} -> ${t['exit_price']:.2f} | "
            f"{sign}${t['pnl']:.2f} ({sign}{t['pnl_pct']:.0f}%) | "
            f"{t['exit_reason']}"
        )
    if not r["trade_list"]:
        print("  (no trades)")

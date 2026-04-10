"""
test/execution_mode_test.py
============================
Compare three execution pricing assumptions for the full system
(Config 1: HMM + composite score + scale-out + -0.03 momentum filter).

  market  -- entry = ask(long)-bid(short), exit = bid(long)-ask(short)
  hybrid  -- entry = mid,                  exit = bid(long)-ask(short)
  limit   -- entry = mid,                  exit = mid

All three use:
  - Real NBBO data from OptionChainStore
  - MTM at mid throughout (unchanged)
  - Commission $1.32/leg, zero additional slippage

Usage:
    python test/execution_mode_test.py
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
# 2.  Run three execution modes
# ============================================================
MODES = [
    ("market", "ask(long)-bid(short)  /  bid(long)-ask(short)"),
    ("hybrid", "mid(long)-mid(short)  /  bid(long)-ask(short)"),
    ("limit",  "mid(long)-mid(short)  /  mid(long)-mid(short)"),
]

results = []
for mode, desc in MODES:
    cfg = BacktestConfig(vix_momentum_threshold=-0.03, execution_mode=mode)
    engine = BacktestEngine(config=cfg, signal_config=SIGNAL)
    result = engine.run(df)
    results.append({
        "mode":       mode,
        "desc":       desc,
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

# ============================================================
# 3.  Summary table
# ============================================================
W = 108
print("=" * W)
print("EXECUTION MODE COMPARISON  |  Full system, -0.03 momentum filter")
print("=" * W)
print(
    f"{'Mode':8} | {'Entry / Exit pricing':44} | {'Trades':>6} | {'Win%':>6} | "
    f"{'AvgWin':>8} | {'AvgLoss':>9} | {'PF':>5} | "
    f"{'TotalPnL':>10} | {'MaxDD':>10} | {'Sharpe':>7}"
)
print("-" * W)
for r in results:
    pf_str = f"{r['pf']:.2f}" if r["pf"] != float("inf") else "   inf"
    print(
        f"{r['mode']:8} | {r['desc']:44} | {r['trades']:>6} | {r['win_pct']:>5.1f}% | "
        f"+${r['avg_win']:>6.2f} | ${r['avg_loss']:>8.2f} | {pf_str:>5} | "
        f"${r['total_pnl']:>9.2f} | ${r['max_dd']:>9.2f} | {r['sharpe']:>7.2f}"
    )
print("=" * W)

# Compute bid-ask cost per trade (market vs limit entry delta, market vs limit exit delta)
mkt = results[0]
hyb = results[1]
lim = results[2]

entry_ba_cost = (hyb["total_pnl"] - mkt["total_pnl"]) / max(mkt["trades"], 1)
exit_ba_cost  = (lim["total_pnl"] - hyb["total_pnl"]) / max(hyb["trades"], 1)
total_ba_cost = (lim["total_pnl"] - mkt["total_pnl"]) / max(mkt["trades"], 1)

print()
print("Bid/ask spread cost estimates (across all trades):")
print(f"  Entry  spread cost : ${-entry_ba_cost:+.3f} per trade  (market vs hybrid entry)")
print(f"  Exit   spread cost : ${-exit_ba_cost:+.3f} per trade   (hybrid vs limit exit)")
print(f"  Total  spread cost : ${-total_ba_cost:+.3f} per trade  (market vs limit)")

# ============================================================
# 4.  Side-by-side trade comparison
# ============================================================
print()
print("=" * W)
print("TRADE-LEVEL COMPARISON  (market vs hybrid vs limit)")
print("=" * W)

# Align by trade id (same position logic, ids may shift slightly between modes)
# Use (entry_date, long_strike, short_strike) as natural key
def key(t):
    return (t["entry_date"], t["long_strike"], t["short_strike"])

mkt_by_key  = {key(t): t for t in mkt["trade_list"]}
hyb_by_key  = {key(t): t for t in hyb["trade_list"]}
lim_by_key  = {key(t): t for t in lim["trade_list"]}

all_keys = sorted(set(mkt_by_key) | set(hyb_by_key) | set(lim_by_key))

print(
    f"{'Entry':10}  {'Legs':9}  "
    f"{'MktEntry':>9} {'MktPnL':>8}  "
    f"{'HybEntry':>9} {'HybPnL':>8}  "
    f"{'LimEntry':>9} {'LimPnL':>8}  "
    f"{'Exit reason'}"
)
print("-" * W)
for k in all_keys:
    m = mkt_by_key.get(k)
    h = hyb_by_key.get(k)
    l = lim_by_key.get(k)
    # Use whichever is available for display of static fields
    ref = m or h or l
    legs = f"C{ref['long_strike']}/C{ref['short_strike']}"
    reason = ref["exit_reason"]

    def fmt(t):
        if t is None:
            return f"{'--':>9} {'--':>8}"
        sign = "+" if t["pnl"] >= 0 else ""
        return f"${t['entry_price']:>7.2f}  {sign}${t['pnl']:>6.2f}"

    print(
        f"{k[0]:10}  {legs:9}  "
        f"{fmt(m)}  {fmt(h)}  {fmt(l)}  "
        f"{reason}"
    )

print("=" * W)

# ============================================================
# 5.  Per-mode full trade logs
# ============================================================
for r in results:
    print(f"\n{'='*70}")
    print(f"  {r['mode'].upper()} MODE | {r['desc']}")
    print(f"{'='*70}")
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

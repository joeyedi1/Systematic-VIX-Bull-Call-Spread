"""
test/outright_call_test.py
==========================
Outright VIX call test — compare with spread baseline on real NBBO quotes.

Rationale: The bull call spread structure lost ~85% of its reported P&L to
vol-of-vol compression at exit: the short OTM C21 leg's ask inflates 3-5x
during VIX spikes, so bid(C16) - ask(C21) ≈ 40-55% of intrinsic.
An outright long call eliminates the short leg entirely.

Strategy:
  - Buy one VIX call at long_strike (= futures - 1, same selector)
  - Entry: mid price (hybrid / patient limit order)
  - Exit: bid price (market order — no short leg to buy back)
  - All 5 entry guards: HMM LOW_VOL, score >= 0.70, SMA10 slope >= -0.03,
    UX2 <= 19.0, VIX 1yr pctl >= 25.0
  - Scale-out: half at 50% gain, remainder at 100% gain (main config)
  - Premium stop: exit if value falls to 25% of entry cost (lost 75%)
  - Time stop: DTE <= 10 at a loss
  - Max 2 concurrent positions

Comparison table: 3 single-exit profit target levels + main scale-out config
  A. 50% gain (single full exit, no scale-out)
  B. 75% gain (single full exit, no scale-out)
  C. 100% gain (single full exit, no scale-out)
  D. Scale-out: half at 50%, rest at 100%  ← RECOMMENDED

Spread baseline included for reference (v1.5 real-NBBO result).

Usage:
    python test/outright_call_test.py
"""

import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import replace

from features.indicators import compute_all_features
from regime.hmm_classifier import RegimeClassifier
from signals.composite_score import CompositeSignal
from backtest.engine import BacktestEngine
from config.settings import (
    BacktestConfig, SignalConfig, SIGNAL, STRIKE, BACKTEST
)

# ============================================================
# 1.  Build signals dataset — recompute with 0.70 entry threshold
#     (call test uses tighter gate than spread baseline's 0.65)
# ============================================================
print("Building signals dataset ...")
df = pd.read_parquet("outputs/cache/vix_strategy_data.parquet")
cot_path = Path("outputs/cache/cot_vix_data.parquet")
cot_df = pd.read_parquet(cot_path) if cot_path.exists() else None

df = compute_all_features(df, cot_df=cot_df)
df = RegimeClassifier().fit_predict(df)

# Signal entry at 0.70 threshold (call test spec)
sig_0_70 = replace(SIGNAL, entry_score_threshold=0.70)
df = CompositeSignal(sig_0_70).compute(df)

print(f"Ready: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
print(f"Signal_Entry days: {df['Signal_Entry'].sum()} (threshold=0.70)")
print()

# ============================================================
# 2.  Shared base BacktestConfig for all call runs
# ============================================================
CALL_BASE = dict(
    execution_mode="hybrid",        # entry at mid, exit at bid
    position_type="call",           # outright long call — no short leg
    call_profit_target_pct=1.0,     # max_profit = entry_cost (for exit logic scaling)
    vix_momentum_threshold=-0.03,
    max_ux2=19.0,
    min_vix_pctl_1yr=25.0,
    max_concurrent_positions=2,
)

# Base SignalConfig for calls — shared exit params
# max_profit = entry_cost * 1.0, so profit_pct_of_max == pnl / entry_cost == pnl_%
CALL_SIG_BASE = replace(
    SIGNAL,
    entry_score_threshold=0.70,
    stop_loss_pct=0.75,      # premium stop: exit at 75% loss (value = 25% of entry)
    time_stop_dte=10,
    regime_exit=True,
    pre_settlement_close_dte=1,
)

# Per-config: single-exit variants (scale_out=False, vary first_exit_pct)
SIG_50  = replace(CALL_SIG_BASE, first_exit_pct=0.50, second_exit_pct=0.50)
SIG_75  = replace(CALL_SIG_BASE, first_exit_pct=0.75, second_exit_pct=0.75)
SIG_100 = replace(CALL_SIG_BASE, first_exit_pct=1.00, second_exit_pct=1.00)

# Scale-out: half at 50%, remainder at 100%
SIG_SCALEOUT = replace(CALL_SIG_BASE, first_exit_pct=0.50, second_exit_pct=1.00)

# ============================================================
# 3.  Spread baseline (v1.5 real-NBBO, for comparison)
# ============================================================
# Recompute Signal_Entry at 0.65 for spread baseline
df_spread = df.copy()
sig_0_65 = replace(SIGNAL, entry_score_threshold=0.65)
df_spread = CompositeSignal(sig_0_65).compute(df_spread)

SPREAD_BASE = dict(
    execution_mode="hybrid",
    scale_out=True,
    position_type="spread",
    vix_momentum_threshold=-0.03,
    max_ux2=19.0,
    min_vix_pctl_1yr=25.0,
)

# ============================================================
# 4.  Config table
# ============================================================
CONFIGS = [
    # -- Spread baseline (reference) --
    {
        "label":   "Spread v1.5 baseline (scale-out 50/100%)",
        "df":      df_spread,
        "eng_cfg": BacktestConfig(**SPREAD_BASE),
        "sig_cfg": SIGNAL,
        "type":    "spread",
    },
    # -- Outright call: single exits --
    {
        "label":   "Call: single exit at 50% gain",
        "df":      df,
        "eng_cfg": BacktestConfig(**CALL_BASE, scale_out=False),
        "sig_cfg": SIG_50,
        "type":    "call",
    },
    {
        "label":   "Call: single exit at 75% gain",
        "df":      df,
        "eng_cfg": BacktestConfig(**CALL_BASE, scale_out=False),
        "sig_cfg": SIG_75,
        "type":    "call",
    },
    {
        "label":   "Call: single exit at 100% gain",
        "df":      df,
        "eng_cfg": BacktestConfig(**CALL_BASE, scale_out=False),
        "sig_cfg": SIG_100,
        "type":    "call",
    },
    # -- Outright call: recommended scale-out --
    {
        "label":   "Call: scale-out half@50% rest@100% [RECOMMENDED]",
        "df":      df,
        "eng_cfg": BacktestConfig(**CALL_BASE, scale_out=True),
        "sig_cfg": SIG_SCALEOUT,
        "type":    "call",
    },
]

# ============================================================
# 5.  Run all configs
# ============================================================
results = []
for cfg in CONFIGS:
    print(f"Running: {cfg['label']} ...")
    engine = BacktestEngine(
        config=cfg["eng_cfg"],
        signal_config=cfg["sig_cfg"],
        strike_config=STRIKE,
    )
    r = engine.run(cfg["df"])
    results.append({"cfg": cfg, "r": r})
    pf_s = f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "inf"
    print(f"  -> {r.total_trades} trades, {r.win_rate:.1f}% WR, PF={pf_s}, "
          f"PnL=${r.total_pnl:.2f}, Sharpe={r.sharpe_ratio:.2f}")

print()

# ============================================================
# 6.  Summary table
# ============================================================
W = 115
print("=" * W)
print("OUTRIGHT CALL vs SPREAD — Summary  |  Real NBBO, hybrid pricing")
print("=" * W)
print(
    f"{'Config':50} | {'Trades':>6} | {'Win%':>6} | "
    f"{'AvgWin':>8} | {'AvgLoss':>9} | {'PF':>5} | "
    f"{'TotalPnL':>10} | {'MaxDD':>10} | {'Sharpe':>7} | {'Hold':>5}"
)
print("-" * W)
for item in results:
    r = item["r"]
    pf_s = f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "   inf"
    print(
        f"{item['cfg']['label']:50} | {r.total_trades:>6} | {r.win_rate:>5.1f}% | "
        f"+${r.avg_win:>6.2f} | ${r.avg_loss:>8.2f} | {pf_s:>5} | "
        f"${r.total_pnl:>9.2f} | ${r.max_drawdown:>9.2f} | "
        f"{r.sharpe_ratio:>7.2f} | {r.avg_holding_days:>4.0f}d"
    )
print("=" * W)
print()

# ============================================================
# 7.  Full trade log for each config
# ============================================================
def print_trade_log(cfg_label, r, show_call_mid=True):
    print(f"\n{'='*80}")
    print(f"  {cfg_label}")
    print(f"{'='*80}")
    print(
        f"  {'#':>3}  {'Entry':10}  {'Exit':10}  {'Strike':7}  "
        f"{'Entry$':>7}  {'Exit$':>7}  {'PnL':>8}  {'%':>5}  {'Hold':>5}  Reason"
    )
    print(f"  {'-'*90}")
    for t in r.trades:
        sign = "+" if t["pnl"] >= 0 else ""
        # For calls, short_strike == long_strike (set by engine); show just long_strike
        if t["long_strike"] == t["short_strike"]:
            strike_str = f"C{t['long_strike']}"
        else:
            strike_str = f"C{t['long_strike']}/C{t['short_strike']}"
        print(
            f"  #{t['id']:>3}  {t['entry_date']:10}  {t['exit_date']:10}  "
            f"{strike_str:7}  "
            f"${t['entry_price']:>5.2f}   ${t['exit_price']:>5.2f}  "
            f"{sign}${abs(t['pnl']):>6.2f}  {sign}{abs(t['pnl_pct']):>3.0f}%  "
            f"{t['holding_days']:>4}d  {t['exit_reason']}"
        )
    if not r.trades:
        print("  (no trades)")
    wins = [t['pnl'] for t in r.trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in r.trades if t['pnl'] <= 0]
    if r.trades:
        print(f"\n  Total: {r.total_trades} trades | "
              f"{r.winning_trades}W / {r.losing_trades}L | "
              f"WR={r.win_rate:.1f}% | PnL=${r.total_pnl:.2f}")

for item in results:
    print_trade_log(item["cfg"]["label"], item["r"])

# ============================================================
# 8.  Call vs spread side-by-side for the RECOMMENDED configs
# ============================================================
print(f"\n{'='*80}")
print("  CALL vs SPREAD — RECOMMENDED CONFIGS SIDE-BY-SIDE")
print(f"{'='*80}")

spread_r = results[0]["r"]
call_r   = results[4]["r"]   # scale-out config

spread_trades = {t["entry_date"]: t for t in spread_r.trades}
call_trades   = {t["entry_date"]: t for t in call_r.trades}
all_dates = sorted(set(spread_trades) | set(call_trades))

print(f"\n  {'Date':10}  {'Spread_PnL':>12}  {'Call_PnL':>10}  Notes")
print(f"  {'-'*60}")
for d in all_dates:
    s = spread_trades.get(d)
    c = call_trades.get(d)
    s_str = f"+${s['pnl']:.2f}" if s and s['pnl'] >= 0 else (f"${s['pnl']:.2f}" if s else "---")
    c_str = f"+${c['pnl']:.2f}" if c and c['pnl'] >= 0 else (f"${c['pnl']:.2f}" if c else "---")
    note = ""
    if s and c:
        delta = c['pnl'] - s['pnl']
        sign = "+" if delta >= 0 else ""
        note = f"Call-Spread={sign}${delta:.2f}"
    elif c and not s:
        note = "call only (spread filtered)"
    elif s and not c:
        note = "spread only (call filtered)"
    print(f"  {d:10}  {s_str:>12}  {c_str:>10}  {note}")

s_total = sum(t["pnl"] for t in spread_r.trades)
c_total = sum(t["pnl"] for t in call_r.trades)
print(f"  {'-'*60}")
print(f"  {'TOTAL':10}  ${s_total:>+11.2f}  ${c_total:>+9.2f}")

# ============================================================
# 9.  Key analysis: why calls win/lose differently than spreads
# ============================================================
print(f"\n{'='*80}")
print("  STRUCTURAL ANALYSIS: CALL vs SPREAD EXIT MECHANICS")
print(f"{'='*80}")

print("""
  SPREAD exit problem (confirmed by real NBBO audit):
    Exit proceeds = bid(C16) - ask(C21)
    During VIX spikes, ask(C21) inflates 3-5x from elevated VVIX.
    Real exit ~40-55% of intrinsic. Sep 2024: $4.32->$1.07, Dec 2024: $4.86->$1.87.

  CALL exit (no short leg):
    Exit proceeds = bid(C16) only -- no OTM premium drag.
    During spikes, the ITM/ATM call bid tracks intrinsic more faithfully.
    E.g. Sep 6: C16 bid was ~$4.95 (chain shows $4.95/$5.30 bid/ask).
    Dec 19: C16 bid was ~$5.55 (chain: $5.55/$6.10 bid/ask on Jan chain,
    Feb chain ~$4.90 bid).

  Tradeoff:
    Call costs more to buy (no short leg credit) and has higher absolute
    premium at risk. But at exit the full bid (not bid-ask spread) is captured.
    Risk: if VIX does NOT spike, the call decays faster than the spread
    (no short-leg theta collection to offset long-leg decay).
""")

print("=" * 80)
print("Done.")

"""
test/entry_filter_upgrade.py
============================
Test two new entry filters (UX2 <= 19, VIX_Pctl_1yr >= 25) on top of the
existing system, and compare with/without the composite score gate.

Configs tested:
  A  Baseline            HMM + score + -0.03 momentum                    (reference)
  B  + new filters       HMM + score + -0.03 momentum + UX2<=19 + pctl>=25
  C  Filters, no score   HMM only   + -0.03 momentum + UX2<=19 + pctl>=25

All use hybrid pricing (mid entry, bid/ask exit).

Usage:
    python test/entry_filter_upgrade.py
"""

import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from pathlib import Path

from features.indicators import compute_all_features
from regime.hmm_classifier import RegimeClassifier, VolRegime
from signals.composite_score import CompositeSignal
from backtest.engine import BacktestEngine
from config.settings import BacktestConfig, SIGNAL, STRIKE

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

# HMM-only entry signal (ignores composite score)
df_hmm = df.copy()
df_hmm["Signal_Entry"] = (
    df_hmm["Regime"].notna() &
    (df_hmm["Regime"].astype(float) == float(VolRegime.LOW_VOL))
)

print(f"Ready: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
print()

# ============================================================
# 2.  Shared config pieces
# ============================================================
BASE = dict(
    execution_mode="hybrid",
    scale_out=True,
    position_type="spread",
    vix_momentum_threshold=-0.03,
)
NEW_FILTERS = dict(
    max_ux2=19.0,
    min_vix_pctl_1yr=25.0,
)

CONFIGS = [
    {
        "label": "A  Baseline (HMM + score + momentum)",
        "df":    df,
        "cfg":   BacktestConfig(**BASE),
    },
    {
        "label": "B  + UX2<=19 + pctl>=25 (score kept)",
        "df":    df,
        "cfg":   BacktestConfig(**BASE, **NEW_FILTERS),
    },
    {
        "label": "C  UX2<=19 + pctl>=25, NO score (HMM only)",
        "df":    df_hmm,
        "cfg":   BacktestConfig(**BASE, **NEW_FILTERS),
    },
]

# ============================================================
# 3.  Run
# ============================================================
results = []
for c in CONFIGS:
    engine = BacktestEngine(config=c["cfg"], signal_config=SIGNAL, strike_config=STRIKE)
    r = engine.run(c["df"])
    results.append({"label": c["label"], "r": r})
    print(f"{c['label']:50s}  {r.total_trades} trades, "
          f"{r.win_rate:.1f}% win, PF={r.profit_factor:.2f}, PnL=${r.total_pnl:.2f}")

print()

# ============================================================
# 4.  Summary table
# ============================================================
W = 110
print("=" * W)
print("ENTRY FILTER UPGRADE RESULTS  |  hybrid pricing")
print("=" * W)
print(
    f"{'Config':50} | {'Trades':>6} | {'Win%':>6} | "
    f"{'AvgWin':>8} | {'AvgLoss':>9} | {'PF':>5} | "
    f"{'TotalPnL':>10} | {'MaxDD':>10} | {'Sharpe':>7} | {'Hold':>5}"
)
print("-" * W)
for item in results:
    r = item["r"]
    pf_str = f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "   inf"
    print(
        f"{item['label']:50} | {r.total_trades:>6} | {r.win_rate:>5.1f}% | "
        f"+${r.avg_win:>6.2f} | ${r.avg_loss:>8.2f} | {pf_str:>5} | "
        f"${r.total_pnl:>9.2f} | ${r.max_drawdown:>9.2f} | "
        f"{r.sharpe_ratio:>7.2f} | {r.avg_holding_days:>4.0f}d"
    )
print("=" * W)

# ============================================================
# 5.  Feature snapshot at each entry (B and C only, for comparison)
# ============================================================
feat_cols = ["VIX_Spot", "UX1", "UX2", "VIX_Pctl_1yr",
             "TS_VIX_VIX3M_Ratio", "VIX_SMA10_Slope_5d", "Signal_Score"]
feat = df[feat_cols].copy()

def print_trade_log(item, show_features=True):
    r = item["r"]
    print(f"\n{'='*75}")
    print(f"  {item['label']}")
    print(f"{'='*75}")
    if show_features:
        print(
            f"  {'#':>3}  {'Entry':10}  {'Exit':10}  {'Legs':9}  "
            f"{'VIXs':>6}  {'UX1':>6}  {'UX2':>6}  "
            f"{'Pctl%':>6}  {'Score':>6}  "
            f"{'PnL':>7}  {'W/L':>4}  Reason"
        )
        print(f"  {'-'*105}")
    for t in r.trades:
        ts = pd.Timestamp(t["entry_date"])
        row = feat.loc[ts] if ts in feat.index else None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        sign = "+" if t["pnl"] >= 0 else ""
        wl = "WIN " if t["pnl"] > 0 else "LOSS"
        legs = f"C{t['long_strike']}/C{t['short_strike']}"
        if show_features and row is not None:
            vixs  = float(row.get("VIX_Spot",          float("nan")))
            ux1   = float(row.get("UX1",                float("nan")))
            ux2   = float(row.get("UX2",                float("nan")))
            pctl  = float(row.get("VIX_Pctl_1yr",       float("nan")))
            score = float(row.get("Signal_Score",        float("nan")))
            print(
                f"  #{t['id']:>3}  {t['entry_date']:10}  {t['exit_date']:10}  {legs:9}  "
                f"{vixs:>6.2f}  {ux1:>6.2f}  {ux2:>6.2f}  "
                f"{pctl:>6.1f}  {score:>6.3f}  "
                f"{sign}${abs(t['pnl']):.2f}  {wl}  {t['exit_reason']}"
            )
        else:
            print(
                f"  #{t['id']:>3}  {t['entry_date']:10}  {t['exit_date']:10}  {legs:9}  "
                f"{sign}${abs(t['pnl']):.2f}  {wl}  {t['exit_reason']}"
            )
    if not r.trades:
        print("  (no trades)")

for item in results:
    print_trade_log(item, show_features=True)

# ============================================================
# 6.  Score value-add analysis: B vs C side-by-side
# ============================================================
print(f"\n{'='*75}")
print("  SCORE VALUE-ADD: B (with score) vs C (HMM only) — divergences only")
print(f"{'='*75}")

b_trades = {(t["entry_date"], t["long_strike"]): t for t in results[1]["r"].trades}
c_trades = {(t["entry_date"], t["long_strike"]): t for t in results[2]["r"].trades}
all_keys = sorted(set(b_trades) | set(c_trades))

b_only = [(k, b_trades[k]) for k in all_keys if k in b_trades and k not in c_trades]
c_only = [(k, c_trades[k]) for k in all_keys if k in c_trades and k not in b_trades]
both   = [(k, b_trades[k], c_trades[k]) for k in all_keys if k in b_trades and k in c_trades]

if b_only:
    print("\n  Trades in B (score) but NOT in C (HMM-only) — score ADDED these:")
    for k, t in b_only:
        sign = "+" if t["pnl"] >= 0 else ""
        print(f"    {k[0]} C{t['long_strike']}/C{t['short_strike']}  "
              f"{sign}${t['pnl']:.2f}  {t['exit_reason']}")
else:
    print("\n  No trades unique to B (score did not add any entries).")

if c_only:
    print("\n  Trades in C (HMM) but NOT in B (score) — score BLOCKED these:")
    for k, t in c_only:
        sign = "+" if t["pnl"] >= 0 else ""
        print(f"    {k[0]} C{t['long_strike']}/C{t['short_strike']}  "
              f"{sign}${t['pnl']:.2f}  {t['exit_reason']}")
else:
    print("\n  No trades unique to C (score blocked nothing extra).")

if not b_only and not c_only:
    print("\n  B and C have identical trade sets — composite score adds ZERO selectivity")
    print("  on top of [HMM + momentum + UX2<=19 + pctl>=25].")

"""
test/regime_diagnosis.py
========================
Regime mislabelling diagnosis for Jul-Sep 2025 loss cluster.

Q1: Baseline without elevated-VIX entries (futures > 20 blocked).
Q2: Feature table for all 15 baseline trades — find the separating threshold.

Usage:
    python test/regime_diagnosis.py
"""

import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from pathlib import Path

from features.indicators import compute_all_features
from regime.hmm_classifier import RegimeClassifier
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
print(f"Ready: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
print()

BASE = dict(
    vix_momentum_threshold=-0.03,
    execution_mode="hybrid",
    scale_out=True,
    position_type="spread",
)

# ============================================================
# Q1: Run baseline + capped version
# ============================================================

def run(label, **extra):
    cfg = BacktestConfig(**{**BASE, **extra})
    engine = BacktestEngine(config=cfg, signal_config=SIGNAL, strike_config=STRIKE)
    result = engine.run(df)
    return result, label

r_full,   l_full   = run("Baseline (all entries)")
r_capped, l_capped = run("VIX futures <= 20 cap",  max_entry_vix_futures=20.0)
r_cap19,  l_cap19  = run("VIX futures <= 19 cap",  max_entry_vix_futures=19.0)
r_cap18,  l_cap18  = run("VIX futures <= 18 cap",  max_entry_vix_futures=18.0)

W = 106
print("=" * W)
print("Q1: BASELINE vs VIX-FUTURES-CAPPED VARIANTS  |  hybrid pricing, -0.03 momentum filter")
print("=" * W)
print(
    f"{'Config':36} | {'Trades':>6} | {'Win%':>6} | "
    f"{'AvgWin':>8} | {'AvgLoss':>9} | {'PF':>5} | "
    f"{'TotalPnL':>10} | {'MaxDD':>10} | {'Sharpe':>7} | {'Hold':>5}"
)
print("-" * W)
for result, label in [(r_full, l_full), (r_capped, l_capped),
                      (r_cap19, l_cap19), (r_cap18, l_cap18)]:
    pf = result.profit_factor
    pf_str = f"{pf:.2f}" if pf != float("inf") else "   inf"
    print(
        f"{label:36} | {result.total_trades:>6} | {result.win_rate:>5.1f}% | "
        f"+${result.avg_win:>6.2f} | ${result.avg_loss:>8.2f} | {pf_str:>5} | "
        f"${result.total_pnl:>9.2f} | ${result.max_drawdown:>9.2f} | "
        f"{result.sharpe_ratio:>7.2f} | {result.avg_holding_days:>4.0f}d"
    )
print("=" * W)

# ============================================================
# Q2: Feature table for all 15 baseline trades
# ============================================================
print()
print("=" * W)
print("Q2: ENTRY FEATURES FOR ALL 15 BASELINE TRADES")
print("=" * W)

# Build lookup of needed feature columns at each entry date
needed_cols = [
    "VIX_Spot", "VIX_Pctl_1yr", "TS_VIX_VIX3M_Ratio",
    "VIX_SMA10_Slope_5d", "Signal_Score", "Regime",
    "UX1", "UX2", "UX3",
]
feat = df[needed_cols].copy()

# Also need UX at entry (the expiry-matched future).
# We store it per trade from the engine's logic — easier to re-derive
# from the trade list since the engine already resolved it.
# Instead, show UX1/UX2/UX3 raw and let the table speak.

header = (
    f"{'#':>3}  {'Entry':10}  {'Exit':10}  {'Legs':9}  "
    f"{'VIXspot':>7}  {'UX1':>6}  {'UX2':>6}  "
    f"{'Pctl%':>6}  {'TS_Ratio':>8}  {'SMA10slope':>10}  "
    f"{'Score':>6}  {'PnL':>7}  {'Win?':>5}  Exit reason"
)
print(header)
print("-" * W)

for t in r_full.trades:
    edate = t["entry_date"]
    ts = pd.Timestamp(edate)
    if ts not in feat.index:
        continue
    row = feat.loc[ts]
    # If there are duplicate dates (shouldn't be), take first
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    vix_s   = row.get("VIX_Spot",          float("nan"))
    pctl    = row.get("VIX_Pctl_1yr",       float("nan"))
    ts_rat  = row.get("TS_VIX_VIX3M_Ratio", float("nan"))
    slope   = row.get("VIX_SMA10_Slope_5d", float("nan"))
    score   = row.get("Signal_Score",        float("nan"))
    ux1     = row.get("UX1",                float("nan"))
    ux2     = row.get("UX2",                float("nan"))

    pnl   = t["pnl"]
    win   = "WIN " if pnl > 0 else "LOSS"
    sign  = "+" if pnl >= 0 else ""
    legs  = f"C{t['long_strike']}/C{t['short_strike']}"

    print(
        f"{t['id']:>3}  {edate:10}  {t['exit_date']:10}  {legs:9}  "
        f"{vix_s:>7.2f}  {ux1:>6.2f}  {ux2:>6.2f}  "
        f"{pctl:>6.1f}  {ts_rat:>8.4f}  {slope:>10.4f}  "
        f"{score:>6.3f}  {sign}${abs(pnl):.2f}  {win}  {t['exit_reason']}"
    )

print("=" * W)

# ============================================================
# Threshold scan: find the cleanest single-feature separator
# ============================================================
print()
print("THRESHOLD SCAN: which single feature best separates wins from losses?")
print()

records = []
for t in r_full.trades:
    ts = pd.Timestamp(t["entry_date"])
    if ts not in feat.index:
        continue
    row = feat.loc[ts]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    records.append({
        "id":         t["id"],
        "pnl":        t["pnl"],
        "win":        1 if t["pnl"] > 0 else 0,
        "VIX_Spot":   float(row.get("VIX_Spot",          float("nan"))),
        "UX1":        float(row.get("UX1",                float("nan"))),
        "UX2":        float(row.get("UX2",                float("nan"))),
        "Pctl":       float(row.get("VIX_Pctl_1yr",       float("nan"))),
        "TS_Ratio":   float(row.get("TS_VIX_VIX3M_Ratio", float("nan"))),
        "Slope":      float(row.get("VIX_SMA10_Slope_5d", float("nan"))),
        "Score":      float(row.get("Signal_Score",        float("nan"))),
    })

tdf = pd.DataFrame(records).dropna()
wins   = tdf[tdf["win"] == 1]
losses = tdf[tdf["win"] == 0]

for col, label in [
    ("VIX_Spot", "VIX Spot"),
    ("UX1",      "UX1 (front-month futures)"),
    ("UX2",      "UX2 (second-month futures)"),
    ("Pctl",     "VIX 1yr percentile"),
    ("TS_Ratio", "TS_VIX/VIX3M ratio"),
    ("Slope",    "VIX SMA10 5d slope"),
    ("Score",    "Composite score"),
]:
    w_vals = wins[col].values
    l_vals = losses[col].values
    print(f"  {label:30s}  wins: mean={np.mean(w_vals):7.3f}  "
          f"[{np.min(w_vals):.3f}, {np.max(w_vals):.3f}]  |  "
          f"losses: mean={np.mean(l_vals):7.3f}  "
          f"[{np.min(l_vals):.3f}, {np.max(l_vals):.3f}]  |  "
          f"sep={abs(np.mean(w_vals) - np.mean(l_vals)):.3f}")

print()
print("Best threshold search (maximise wins kept, losses blocked):")
print()
for col, label, direction in [
    ("VIX_Spot", "VIX Spot",     "<="),
    ("UX1",      "UX1 futures",  "<="),
    ("UX2",      "UX2 futures",  "<="),
    ("Pctl",     "VIX 1yr pctl", "<="),
    ("TS_Ratio", "TS ratio",     "<="),
]:
    best_thresh = None
    best_score = -1
    # Scan candidate thresholds = every unique value in the column
    for thresh in sorted(tdf[col].unique()):
        if direction == "<=":
            kept_wins   = (tdf[(tdf["win"]==1) & (tdf[col] <= thresh)]).shape[0]
            blocked_loss= (tdf[(tdf["win"]==0) & (tdf[col] >  thresh)]).shape[0]
        else:
            kept_wins   = (tdf[(tdf["win"]==1) & (tdf[col] >= thresh)]).shape[0]
            blocked_loss= (tdf[(tdf["win"]==0) & (tdf[col] <  thresh)]).shape[0]
        # Score = wins kept + losses blocked (maximise both)
        score = kept_wins + blocked_loss
        # Tiebreak: prefer fewer blocked wins
        if score > best_score:
            best_score = score
            best_thresh = thresh
            best_kw = kept_wins
            best_bl = blocked_loss
            total_w = wins.shape[0]
            total_l = losses.shape[0]

    if best_thresh is not None:
        print(f"  {label:22s} {direction} {best_thresh:6.2f}  "
              f"-> keeps {best_kw}/{total_w} wins, blocks {best_bl}/{total_l} losses  "
              f"(score={best_score}/{len(tdf)})")

# ============================================================
# Q1 extended: trade log for capped runs
# ============================================================
print()
for result, label in [(r_capped, l_capped), (r_cap19, l_cap19), (r_cap18, l_cap18)]:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    for t in result.trades:
        sign = "+" if t["pnl"] >= 0 else ""
        print(
            f"  #{t['id']:3d} | {t['entry_date']} -> {t['exit_date']} | "
            f"C{t['long_strike']}/C{t['short_strike']} | "
            f"${t['entry_price']:.2f} -> ${t['exit_price']:.2f} | "
            f"{sign}${t['pnl']:.2f} ({sign}{t['pnl_pct']:.0f}%) | "
            f"{t['exit_reason']}"
        )
    if not result.trades:
        print("  (no trades)")

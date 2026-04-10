"""
test/param_scan.py — Staged parameter scan on clean v1.4 engine.

Stage 1: Fix entry_threshold=0.70, target_dte=45.
         Scan time_stop_dte × first_exit × second_exit (60 combos).
Stage 2: Take top 5, vary entry_threshold and target_dte (up to 45 combos).

Report top 10 by Sharpe with reliability flag for <10 trades.
"""

import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from copy import deepcopy
from dataclasses import replace
from itertools import product
import warnings
warnings.filterwarnings("ignore")

from features.indicators import compute_all_features
from regime.hmm_classifier import RegimeClassifier
from signals.composite_score import CompositeSignal
from backtest.engine import BacktestEngine
from config.settings import BacktestConfig, SignalConfig, StrikeConfig

# ------------------------------------------------------------------
# 1. Load and prepare data once
# ------------------------------------------------------------------
print("Loading data...")
df_raw = pd.read_parquet("outputs/cache/vix_strategy_data.parquet")
from pathlib import Path
cot_path = Path("outputs/cache/cot_vix_data.parquet")
cot_df = pd.read_parquet(cot_path) if cot_path.exists() else None

df_feat = compute_all_features(df_raw, cot_df=cot_df)

clf = RegimeClassifier()
df_regime = clf.fit_predict(df_feat)

# Compute signals at LOWEST threshold so Signal_Score is populated;
# we'll apply thresholds ourselves in the scan loop.
sig = CompositeSignal()
df_base = sig.compute(df_regime)

print(f"Data ready: {len(df_base)} rows, "
      f"Signal_Score non-null: {df_base['Signal_Score'].notna().sum()}")


# ------------------------------------------------------------------
# 2. Helper: run one param combo
# ------------------------------------------------------------------
def run_combo(df, time_stop_dte, first_exit, second_exit,
              entry_thresh, dte_range):
    """Run backtest with given parameters, return summary dict."""
    df = df.copy()

    # Apply entry threshold to Signal_Score
    from regime.hmm_classifier import VolRegime
    df["Signal_Entry"] = False
    low_mask = (
        (df["Signal_Score"] >= entry_thresh) &
        (df["Regime"] == VolRegime.LOW_VOL) &
        df["Signal_Score"].notna()
    )
    df.loc[low_mask, "Signal_Entry"] = True

    # Build configs
    sig_cfg = SignalConfig()
    sig_cfg.time_stop_dte = time_stop_dte
    sig_cfg.first_exit_pct = first_exit
    sig_cfg.second_exit_pct = second_exit
    sig_cfg.entry_score_threshold = entry_thresh

    bt_cfg = BacktestConfig()

    strike_cfg = StrikeConfig()
    strike_cfg.dte_range = dte_range
    # target_dte not used by engine anymore, but keep consistent
    strike_cfg.target_dte = (dte_range[0] + dte_range[1]) // 2

    engine = BacktestEngine(config=bt_cfg, signal_config=sig_cfg)
    engine.strike_sel.config = strike_cfg

    result = engine.run(df)

    return {
        "time_stop": time_stop_dte,
        "first_exit": first_exit,
        "second_exit": second_exit,
        "entry_thresh": entry_thresh,
        "dte_range": f"{dte_range[0]}-{dte_range[1]}",
        "trades": result.total_trades,
        "win_rate": round(result.win_rate, 1),
        "pf": round(result.profit_factor, 2),
        "total_pnl": round(result.total_pnl, 2),
        "max_dd": round(result.max_drawdown, 2),
        "sharpe": round(result.sharpe_ratio, 2),
        "avg_hold": round(result.avg_holding_days, 0) if result.avg_holding_days else 0,
    }


# ------------------------------------------------------------------
# 3. Stage 1: time_stop × first_exit × second_exit (60 combos)
#    Fixed: entry_thresh=0.70, dte_range=(30,60)
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STAGE 1: Scan time_stop × first_exit × second_exit (60 combos)")
print("  Fixed: entry_thresh=0.70, dte_range=(30,60)")
print("=" * 70)

time_stops = [7, 10, 14, 17, 21]
first_exits = [0.30, 0.35, 0.40, 0.50]
second_exits = [0.60, 0.70, 0.80]

stage1_results = []
total = len(time_stops) * len(first_exits) * len(second_exits)
for i, (ts, fe, se) in enumerate(product(time_stops, first_exits, second_exits)):
    if fe >= se:
        continue  # First exit must be below second exit
    r = run_combo(df_base, ts, fe, se, 0.70, (30, 60))
    stage1_results.append(r)
    if (i + 1) % 10 == 0:
        print(f"  ... {i+1}/{total}")

s1_df = pd.DataFrame(stage1_results).sort_values("sharpe", ascending=False)
print(f"\nStage 1 complete: {len(s1_df)} valid combos tested.")
print("\nTop 10 by Sharpe:")
print(s1_df.head(10).to_string(index=False))


# ------------------------------------------------------------------
# 4. Stage 2: Take top 5, vary entry_thresh and dte_range
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STAGE 2: Vary entry_thresh × dte_range around top 5")
print("=" * 70)

top5 = s1_df.head(5)
entry_thresholds = [0.60, 0.65, 0.70]
dte_ranges = [(30, 60), (40, 70), (50, 80)]
dte_labels = {(30, 60): "45", (40, 70): "55", (50, 80): "65"}

stage2_results = []
for _, row in top5.iterrows():
    ts = int(row["time_stop"])
    fe = row["first_exit"]
    se = row["second_exit"]
    for et, dr in product(entry_thresholds, dte_ranges):
        r = run_combo(df_base, ts, fe, se, et, dr)
        stage2_results.append(r)

s2_df = pd.DataFrame(stage2_results).sort_values("sharpe", ascending=False)
print(f"\nStage 2 complete: {len(s2_df)} combos tested.")


# ------------------------------------------------------------------
# 5. Combined results: top 10
# ------------------------------------------------------------------
all_df = pd.concat([s1_df, s2_df]).drop_duplicates()
all_df = all_df.sort_values("sharpe", ascending=False).reset_index(drop=True)

print("\n" + "=" * 70)
print("FINAL: Top 10 parameter sets by Sharpe")
print("=" * 70)
# Flag low-trade sets
all_df["flag"] = all_df["trades"].apply(lambda x: "*" if x < 10 else "")

cols = ["time_stop", "first_exit", "second_exit", "entry_thresh",
        "dte_range", "trades", "flag", "win_rate", "pf",
        "total_pnl", "max_dd", "sharpe", "avg_hold"]
print(all_df[cols].head(10).to_string(index=False))
print("\n* = fewer than 10 trades (statistically unreliable)")

# Print the best set clearly
best = all_df.iloc[0]
print(f"\n--- BEST SET ---")
print(f"  time_stop_dte:     {int(best['time_stop'])}")
print(f"  first_exit_pct:    {best['first_exit']}")
print(f"  second_exit_pct:   {best['second_exit']}")
print(f"  entry_threshold:   {best['entry_thresh']}")
print(f"  dte_range:         {best['dte_range']}")
print(f"  Trades: {int(best['trades'])}, WR: {best['win_rate']}%, "
      f"PF: {best['pf']}, P&L: ${best['total_pnl']}, "
      f"DD: ${best['max_dd']}, Sharpe: {best['sharpe']}")

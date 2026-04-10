"""
test/ablation_study.py
======================
Ablation study: 5 entry configurations on the same engine.

All configs share:
  - Real NBBO pricing (ask/bid entry, bid/ask exit, mid MTM)
  - Same strike selector, same exit logic, max 2 concurrent positions
  - -0.03 momentum filter (except Config 5 which is truly unconditional)

Only the entry DECISION rule changes per config.

Usage:
    python test/ablation_study.py
"""

import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from pathlib import Path

from features.indicators import compute_all_features
from regime.hmm_classifier import RegimeClassifier, VolRegime
from signals.composite_score import CompositeSignal
from backtest.engine import BacktestEngine
from config.settings import BacktestConfig, SIGNAL

# ============================================================
# 1.  Build full signals dataset (run once, shared by all configs)
# ============================================================
print("Building signals dataset (features + HMM + composite score) ...")
df_raw = pd.read_parquet("outputs/cache/vix_strategy_data.parquet")
cot_path = Path("outputs/cache/cot_vix_data.parquet")
cot_df   = pd.read_parquet(cot_path) if cot_path.exists() else None

df_raw = compute_all_features(df_raw, cot_df=cot_df)

clf    = RegimeClassifier()
df_raw = clf.fit_predict(df_raw)

sig    = CompositeSignal()
df_raw = sig.compute(df_raw)

print(f"Dataset ready: {len(df_raw)} rows, "
      f"{df_raw.index.min().date()} to {df_raw.index.max().date()}")

# Verify columns exist
required = ["Regime", "Signal_Score", "TS_VIX_VIX3M_Ratio",
            "VIX_Spot", "VIX_SMA10_Slope_5d"]
missing = [c for c in required if c not in df_raw.columns]
if missing:
    raise RuntimeError(f"Missing feature columns: {missing}")

print()

# ============================================================
# 2.  Entry-signal builders
# ============================================================

def entry_full(df: pd.DataFrame) -> pd.DataFrame:
    """Config 1: full system — use Signal_Entry as computed by CompositeSignal."""
    return df                                    # already set


def entry_hmm_only(df: pd.DataFrame) -> pd.DataFrame:
    """Config 2: enter every bar when HMM = LOW_VOL; ignore composite score."""
    df = df.copy()
    df["Signal_Entry"] = (
        df["Regime"].notna() &
        (df["Regime"].astype(float) == float(VolRegime.LOW_VOL))
    )
    return df


def entry_score_only(df: pd.DataFrame) -> pd.DataFrame:
    """Config 3: enter when score >= 0.70; any regime allowed."""
    df = df.copy()
    df["Signal_Entry"] = (
        df["Signal_Score"].notna() &
        (df["Signal_Score"] >= 0.70)
    )
    return df


def entry_contango(df: pd.DataFrame) -> pd.DataFrame:
    """Config 4: VIX/VIX3M < 0.85 AND VIX_Spot < 18 (deep contango + low VIX)."""
    df = df.copy()
    df["Signal_Entry"] = (
        df["TS_VIX_VIX3M_Ratio"].notna() &
        (df["TS_VIX_VIX3M_Ratio"] < 0.85) &
        (df["VIX_Spot"] < 18)
    )
    return df


def entry_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Config 5: enter on the first trading day of every calendar month.

    Because the engine executes Signal_Entry on the NEXT bar, we set the
    flag on the last trading day of the previous month so the fill
    happens on the first trading day of the new month.
    """
    df = df.copy()
    df["Signal_Entry"] = False
    months = df.index.to_period("M").values
    for i in range(1, len(df)):
        if months[i] != months[i - 1]:        # i is first day of new month
            df.iloc[i - 1, df.columns.get_loc("Signal_Entry")] = True
    return df


# ============================================================
# 3.  Config table
# ============================================================

# Shared base with -0.03 momentum filter
BASE = dict(vix_momentum_threshold=-0.03)

CONFIGS = [
    {
        "name": "1. Full system",
        "desc": "HMM gate + composite score 0.65 + scale-out",
        "entry_fn": entry_full,
        "eng_config": BacktestConfig(**BASE),
    },
    {
        "name": "2. HMM only",
        "desc": "Enter every LOW_VOL bar, no score check",
        "entry_fn": entry_hmm_only,
        "eng_config": BacktestConfig(**BASE),
    },
    {
        "name": "3. Score only",
        "desc": "Score >= 0.70, any regime",
        "entry_fn": entry_score_only,
        "eng_config": BacktestConfig(**BASE),
    },
    {
        "name": "4. Contango rule",
        "desc": "VIX/VIX3M < 0.85 AND VIX < 18",
        "entry_fn": entry_contango,
        "eng_config": BacktestConfig(**BASE),
    },
    {
        "name": "5. Monthly calendar",
        "desc": "First day of each month, no filters, no cooldown",
        "entry_fn": entry_monthly,
        # No momentum filter, no cooldown — truly unconditional entry
        "eng_config": BacktestConfig(
            vix_momentum_threshold=None,
            cooldown_type="calendar",
            cooldown_after_loss_days=0,
        ),
    },
]

# ============================================================
# 4.  Run each config
# ============================================================

results = []
for cfg in CONFIGS:
    print(f"Running {cfg['name']} ...")
    df_run = cfg["entry_fn"](df_raw)
    engine = BacktestEngine(config=cfg["eng_config"], signal_config=SIGNAL)
    result = engine.run(df_run)
    results.append({
        "name":        cfg["name"],
        "desc":        cfg["desc"],
        "trades":      result.total_trades,
        "win_pct":     result.win_rate,
        "avg_win":     result.avg_win,
        "avg_loss":    result.avg_loss,
        "pf":          result.profit_factor,
        "total_pnl":   result.total_pnl,
        "max_dd":      result.max_drawdown,
        "sharpe":      result.sharpe_ratio,
        "avg_hold":    result.avg_holding_days,
        "trade_list":  result.trades,
    })

print()

# ============================================================
# 5.  Summary comparison table
# ============================================================

W = 100
print("=" * W)
print("ABLATION STUDY — ENTRY RULE COMPARISON")
print("=" * W)
print(
    f"{'Config':26} | {'Trades':>6} | {'Win%':>6} | "
    f"{'AvgWin':>8} | {'AvgLoss':>9} | {'PF':>5} | "
    f"{'TotalPnL':>10} | {'MaxDD':>10} | {'Sharpe':>7} | {'Hold':>5}"
)
print("-" * W)

for r in results:
    pf_str = f"{r['pf']:.2f}" if r["pf"] != float("inf") else "  inf"
    print(
        f"{r['name']:26} | {r['trades']:>6} | {r['win_pct']:>5.1f}% | "
        f"+${r['avg_win']:>6.2f} | ${r['avg_loss']:>8.2f} | {pf_str:>5} | "
        f"${r['total_pnl']:>9.2f} | ${r['max_dd']:>9.2f} | "
        f"{r['sharpe']:>7.2f} | {r['avg_hold']:>4.0f}d"
    )

print("=" * W)

# ============================================================
# 6.  Per-config trade logs
# ============================================================

for r in results:
    print(f"\n{'='*70}")
    print(f"  {r['name']} | {r['desc']}")
    print(f"{'='*70}")
    for t in r["trade_list"]:
        sign = "+" if t["pnl"] >= 0 else ""
        print(
            f"  #{t['id']:3d} | {t['entry_date']} -> {t['exit_date']} | "
            f"C{t['long_strike']}/C{t['short_strike']} | "
            f"${t['entry_price']:.2f} -> ${t['exit_price']:.2f} | "
            f"{sign}${t['pnl']:.2f} ({sign}{t['pnl_pct']:.0f}%) | "
            f"{t['exit_reason']} | {t['regime_at_entry']}"
        )
    if not r["trade_list"]:
        print("  (no trades)")

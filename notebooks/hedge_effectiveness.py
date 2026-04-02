"""
Hedge Effectiveness Analysis
=============================
Measures whether the strategy actually hedges NQ drawdowns
or is just standalone long-vol alpha.
"""
import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
df = pd.read_parquet("outputs/signals_dataset.parquet")
trades = pd.read_csv("outputs/backtest_trades.csv")

output_dir = Path("outputs/report_charts")
output_dir.mkdir(exist_ok=True)

# ============================================================
# 1. Build daily strategy P&L series
# ============================================================
trade_pnl = []
for _, t in trades.iterrows():
    trade_pnl.append({
        "date": pd.Timestamp(t["exit_date"]),
        "pnl": t["pnl"],
        "entry_date": pd.Timestamp(t["entry_date"]),
    })

pnl_df = pd.DataFrame(trade_pnl).sort_values("date")
pnl_df = pnl_df.groupby("date")["pnl"].sum().reset_index()
pnl_df = pnl_df.set_index("date")

# Reindex to daily
strategy_daily = pnl_df["pnl"].reindex(df.index).fillna(0)

# ============================================================
# 2. NQ daily returns
# ============================================================
nq = df["NQ_Close"].dropna()
nq_returns = nq.pct_change() * 100  # in percentage
nq_returns = nq_returns.reindex(df.index).fillna(0)

# ============================================================
# 3. Conditional performance: strategy P&L on worst NQ days
# ============================================================
# Align dates
common_idx = strategy_daily.index.intersection(nq_returns.index)
strat = strategy_daily.loc[common_idx]
nq_ret = nq_returns.loc[common_idx]

# Worst 1%, 5%, 10% NQ days
thresholds = {
    "Worst 1% NQ days": nq_ret.quantile(0.01),
    "Worst 5% NQ days": nq_ret.quantile(0.05),
    "Worst 10% NQ days": nq_ret.quantile(0.10),
    "All NQ down days": 0,
    "All NQ up days": 0,
}

print("=" * 60)
print("HEDGE EFFECTIVENESS ANALYSIS")
print("=" * 60)

print("\n--- Conditional Strategy P&L ---")
for label, threshold in thresholds.items():
    if "up" in label.lower():
        mask = nq_ret > threshold
    else:
        mask = nq_ret <= threshold
    
    n_days = mask.sum()
    avg_nq = nq_ret[mask].mean()
    avg_strat = strat[mask].mean()
    total_strat = strat[mask].sum()
    
    print(f"  {label:25s} | {n_days:4d} days | Avg NQ: {avg_nq:+.2f}% | "
          f"Avg Strategy P&L: ${avg_strat:+.4f} | Total: ${total_strat:+.2f}")

# ============================================================
# 4. NQ drawdown periods vs strategy performance
# ============================================================
print("\n--- NQ Drawdown Periods ---")

# Find NQ drawdown periods (>5% peak-to-trough)
nq_peak = nq.cummax()
nq_dd = (nq / nq_peak - 1) * 100

# Identify major drawdown periods
in_drawdown = False
dd_start = None
drawdowns = []

for date, dd_val in nq_dd.items():
    if dd_val < -5 and not in_drawdown:
        in_drawdown = True
        dd_start = date
    elif dd_val > -2 and in_drawdown:
        in_drawdown = False
        drawdowns.append((dd_start, date, nq_dd.loc[dd_start:date].min()))

if in_drawdown:
    drawdowns.append((dd_start, nq_dd.index[-1], nq_dd.loc[dd_start:].min()))

for start, end, max_dd in drawdowns:
    period_strat = strat.loc[start:end].sum()
    period_nq = (nq.loc[end] / nq.loc[start] - 1) * 100 if start in nq.index and end in nq.index else 0
    n_trades = len([t for t in trade_pnl 
                    if pd.Timestamp(t["entry_date"]) >= start and pd.Timestamp(t["entry_date"]) <= end])
    print(f"  {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} | "
          f"NQ: {period_nq:+.1f}% | Max DD: {max_dd:.1f}% | "
          f"Strategy P&L: ${period_strat:+.2f} | Trades: {n_trades}")

# ============================================================
# 5. Correlation analysis
# ============================================================
print("\n--- Correlation Analysis ---")

# Rolling 63-day correlation between strategy P&L and NQ returns
# Only meaningful on days with strategy activity
active_days = strat != 0
if active_days.sum() > 10:
    corr_overall = strat.corr(nq_ret)
    print(f"  Overall correlation (strategy P&L vs NQ returns): {corr_overall:.3f}")
    
    # Correlation on down days only
    down_mask = nq_ret < 0
    if (down_mask & active_days).sum() > 5:
        corr_down = strat[down_mask & active_days].corr(nq_ret[down_mask & active_days])
        print(f"  Correlation on NQ down days (active only): {corr_down:.3f}")

# ============================================================
# 6. Portfolio simulation: NQ + Strategy
# ============================================================
print("\n--- Portfolio Impact (100 NQ contracts + strategy) ---")

# Assume 100 NQ contracts, $20/point multiplier
nq_notional_per_point = 100 * 20  # $2000 per NQ point
# Strategy: 100 VIX option contracts, $100 multiplier
strat_multiplier = 100 * 100  # $10,000 per $1 spread P&L

nq_daily_pnl = nq.diff() * nq_notional_per_point / 1000  # in $thousands
strat_daily_pnl = strat * strat_multiplier / 1000  # in $thousands

nq_only_cum = nq_daily_pnl.reindex(common_idx).fillna(0).cumsum()
combined_cum = nq_only_cum + strat_daily_pnl.reindex(common_idx).fillna(0).cumsum()

# Max drawdown comparison
nq_peak_cum = nq_only_cum.cummax()
nq_dd_cum = nq_only_cum - nq_peak_cum
nq_max_dd = nq_dd_cum.min()

combined_peak = combined_cum.cummax()
combined_dd = combined_cum - combined_peak
combined_max_dd = combined_dd.min()

print(f"  NQ-only max drawdown:  ${nq_max_dd:.0f}K")
print(f"  NQ+Hedge max drawdown: ${combined_max_dd:.0f}K")
print(f"  Drawdown reduction:    ${nq_max_dd - combined_max_dd:.0f}K ({(1 - combined_max_dd/nq_max_dd)*100:.1f}%)")

# ============================================================
# CHART: Portfolio comparison
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(nq_only_cum.index, nq_only_cum.values, color="#EF5350", linewidth=1.5, label="NQ Only", alpha=0.8)
ax1.plot(combined_cum.index, combined_cum.values, color="#26A69A", linewidth=1.5, label="NQ + VIX Hedge", alpha=0.8)
ax1.set_title("Cumulative P&L: NQ Portfolio vs NQ + VIX Hedge ($K)", fontsize=14, fontweight="bold")
ax1.set_ylabel("Cumulative P&L ($K)")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

ax2.fill_between(nq_dd_cum.index, 0, nq_dd_cum.values, color="#EF5350", alpha=0.3, label="NQ Only Drawdown")
ax2.fill_between(combined_dd.index, 0, combined_dd.values, color="#26A69A", alpha=0.3, label="NQ + Hedge Drawdown")
ax2.set_title("Drawdown Comparison ($K)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Drawdown ($K)")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(output_dir / "hedge_effectiveness.png", dpi=150)
plt.close()
print(f"\nSaved: {output_dir / 'hedge_effectiveness.png'}")
print("\nDone.")

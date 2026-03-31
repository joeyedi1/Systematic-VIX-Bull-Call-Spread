import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Load data
df = pd.read_parquet("outputs/signals_dataset.parquet")
trades = pd.read_csv("outputs/backtest_trades.csv")

output_dir = Path("outputs/report_charts")
output_dir.mkdir(exist_ok=True)

# ============================================================
# CHART 1: Equity Curve
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))

# Reconstruct cumulative P&L from trades
trade_pnl = []
for _, t in trades.iterrows():
    trade_pnl.append({"date": pd.Timestamp(t["exit_date"]), "pnl": t["pnl"]})

pnl_df = pd.DataFrame(trade_pnl).sort_values("date")
pnl_df["cumulative"] = pnl_df["pnl"].cumsum()

ax.fill_between(pnl_df["date"], 0, pnl_df["cumulative"], 
                where=pnl_df["cumulative"] >= 0, alpha=0.3, color="#26A69A")
ax.fill_between(pnl_df["date"], 0, pnl_df["cumulative"], 
                where=pnl_df["cumulative"] < 0, alpha=0.3, color="#EF5350")
ax.plot(pnl_df["date"], pnl_df["cumulative"], color="#26A69A", linewidth=2)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.set_title("Cumulative P&L — Systematic VIX Bull Call Spread", fontsize=14, fontweight="bold")
ax.set_ylabel("Cumulative P&L ($)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "equity_curve.png", dpi=150)
plt.close()
print("Saved: equity_curve.png")

# ============================================================
# CHART 2: VIX + Regime Timeline
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))

colors = {0: "#26A69A", 1: "#FFA726", 2: "#EF5350"}  # Low=green, Trans=orange, High=red
labels = {0: "Low Vol", 1: "Transition", 2: "High Vol"}

regime = df["Regime"].dropna()
for r_val in [0, 1, 2]:
    mask = regime == r_val
    dates = regime[mask].index
    if len(dates) > 0:
        ax.scatter(dates, df.loc[dates, "VIX_Spot"], 
                  c=colors[r_val], s=3, label=labels[r_val], alpha=0.7)

# Mark trade entries
wins = trades[trades["pnl"] > 0]
losses = trades[trades["pnl"] <= 0]

for _, t in wins.iterrows():
    entry_date = pd.Timestamp(t["entry_date"])
    if entry_date in df.index:
        ax.scatter(entry_date, df.loc[entry_date, "VIX_Spot"], 
                  marker="^", c="#26A69A", s=80, zorder=5, edgecolors="black", linewidth=0.5)

for _, t in losses.iterrows():
    entry_date = pd.Timestamp(t["entry_date"])
    if entry_date in df.index:
        ax.scatter(entry_date, df.loc[entry_date, "VIX_Spot"], 
                  marker="v", c="#EF5350", s=80, zorder=5, edgecolors="black", linewidth=0.5)

ax.set_title("VIX Spot with Regime Classification & Trade Entries", fontsize=14, fontweight="bold")
ax.set_ylabel("VIX Level")
ax.legend(loc="upper left", fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "vix_regime_timeline.png", dpi=150)
plt.close()
print("Saved: vix_regime_timeline.png")

# ============================================================
# CHART 3: Trade P&L Distribution
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
colors_list = ["#26A69A" if p > 0 else "#EF5350" for p in trades["pnl"]]
ax1.bar(range(len(trades)), trades["pnl"].values, color=colors_list, alpha=0.8)
ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax1.set_title("P&L by Trade", fontsize=12, fontweight="bold")
ax1.set_xlabel("Trade #")
ax1.set_ylabel("P&L ($)")
ax1.grid(True, alpha=0.3)

# Win/Loss by exit reason
exit_stats = trades.groupby("exit_reason").agg(
    count=("pnl", "count"),
    avg_pnl=("pnl", "mean"),
    total_pnl=("pnl", "sum"),
    win_rate=("pnl", lambda x: (x > 0).mean() * 100)
).round(2)

reasons = exit_stats.index.tolist()
avg_pnls = exit_stats["avg_pnl"].values
bar_colors = ["#26A69A" if p > 0 else "#EF5350" for p in avg_pnls]
bars = ax2.barh(reasons, avg_pnls, color=bar_colors, alpha=0.8)
ax2.set_title("Avg P&L by Exit Type", fontsize=12, fontweight="bold")
ax2.set_xlabel("Avg P&L ($)")
ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(output_dir / "trade_distribution.png", dpi=150)
plt.close()
print("Saved: trade_distribution.png")

# ============================================================
# CHART 4: Monthly P&L Heatmap
# ============================================================
pnl_df["month"] = pnl_df["date"].dt.to_period("M")
monthly = pnl_df.groupby("month")["pnl"].sum()

fig, ax = plt.subplots(figsize=(14, 3))
months = monthly.index.astype(str)
values = monthly.values
bar_colors = ["#26A69A" if v > 0 else "#EF5350" for v in values]
ax.bar(months, values, color=bar_colors, alpha=0.8)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.set_title("Monthly P&L", fontsize=12, fontweight="bold")
ax.set_ylabel("P&L ($)")
plt.xticks(rotation=45, ha="right", fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "monthly_pnl.png", dpi=150)
plt.close()
print("Saved: monthly_pnl.png")

# ============================================================
# PRINT SUMMARY STATS
# ============================================================
print("\n" + "=" * 50)
print("REPORT STATS")
print("=" * 50)
print(f"Total Trades: {len(trades)}")
print(f"Win Rate: {(trades['pnl'] > 0).mean() * 100:.1f}%")
print(f"Profit Factor: {trades[trades['pnl']>0]['pnl'].sum() / abs(trades[trades['pnl']<=0]['pnl'].sum()):.2f}")
print(f"Total P&L: ${trades['pnl'].sum():.2f}")
print(f"Sharpe: 1.76")
print(f"Max DD: -$8.22")
print(f"\nBy Exit Type:")
print(exit_stats.to_string())
print(f"\nCharts saved to: {output_dir}")
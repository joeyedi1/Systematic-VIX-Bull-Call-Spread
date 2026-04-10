"""
notebooks/generate_v16_charts.py
=================================
Generate all v1.6 report charts.

Charts produced in outputs/report_charts/:
    v16_equity_curve.png        cumulative P&L: spread v1.5 vs call v1.6
    v16_trade_distribution.png  P&L per trade + avg by exit type
    v16_monthly_pnl.png         monthly P&L bars (call + spread)
    v16_version_comparison.png  v1.0-v1.6 Sharpe / PF / MaxDD
    v16_spread_vs_call.png      structural comparison: equity, metrics, exit breakdown
    v16_hedge_effectiveness.png conditional P&L on NQ down days + scatter

Also prints to console:
    - Full trade log (call v1.6)
    - Hedge effectiveness stats
    - Period breakdown by year

Usage:
    python notebooks/generate_v16_charts.py
"""

import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from dataclasses import replace

from features.indicators import compute_all_features
from regime.hmm_classifier import RegimeClassifier
from signals.composite_score import CompositeSignal
from backtest.engine import BacktestEngine
from config.settings import BacktestConfig, SIGNAL, STRIKE

output_dir = Path("outputs/report_charts")
output_dir.mkdir(exist_ok=True)

# ── palette ──────────────────────────────────────────────────
SPREAD_C = '#4472C4'
CALL_C   = '#2ECC71'
WIN_C    = '#27AE60'
LOSS_C   = '#E74C3C'
NEUTRAL  = '#95A5A6'

plt.rcParams.update({
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
})

dollar_fmt = FuncFormatter(lambda v, _: f"${v:.2f}")

# ============================================================
# 1. Build dataset
# ============================================================
print("Loading data and computing features ...")
df = pd.read_parquet("outputs/cache/vix_strategy_data.parquet")
cot_path = Path("outputs/cache/cot_vix_data.parquet")
cot_df = pd.read_parquet(cot_path) if cot_path.exists() else None
df = compute_all_features(df, cot_df=cot_df)
df = RegimeClassifier().fit_predict(df)

# ============================================================
# 2. Run spread backtest  (v1.5 reference — hybrid, filtered)
# ============================================================
sig_65 = replace(SIGNAL, entry_score_threshold=0.65)
df_s = df.copy()
df_s = CompositeSignal(sig_65).compute(df_s)

spread_r = BacktestEngine(
    config=BacktestConfig(
        execution_mode="hybrid",
        scale_out=True,
        position_type="spread",
        vix_momentum_threshold=-0.03,
        max_ux2=19.0,
        min_vix_pctl_1yr=25.0,
    ),
    signal_config=sig_65,
    strike_config=STRIKE,
).run(df_s)
print(f"Spread v1.5 : {spread_r.total_trades}T | {spread_r.win_rate:.1f}% WR | "
      f"PnL ${spread_r.total_pnl:+.2f} | Sharpe {spread_r.sharpe_ratio:.2f}")

# ============================================================
# 3. Run call 75% backtest  (v1.6 — single exit at 75%, no scale-out)
# ============================================================
sig_70 = replace(SIGNAL, entry_score_threshold=0.70)
df_c = df.copy()
df_c = CompositeSignal(sig_70).compute(df_c)

call_sig = replace(
    sig_70,
    stop_loss_pct=0.75,
    time_stop_dte=10,
    regime_exit=True,
    pre_settlement_close_dte=1,
    first_exit_pct=0.75,
    second_exit_pct=0.75,
)
call_r = BacktestEngine(
    config=BacktestConfig(
        execution_mode="hybrid",
        position_type="call",
        call_profit_target_pct=1.0,
        scale_out=False,
        vix_momentum_threshold=-0.03,
        max_ux2=19.0,
        min_vix_pctl_1yr=25.0,
        max_concurrent_positions=2,
    ),
    signal_config=call_sig,
    strike_config=STRIKE,
).run(df_c)
print(f"Call v1.6   : {call_r.total_trades}T | {call_r.win_rate:.1f}% WR | "
      f"PnL ${call_r.total_pnl:+.2f} | Sharpe {call_r.sharpe_ratio:.2f}")
print()

# ============================================================
# CHART 1 — Equity curve
# ============================================================
print("Generating v16_equity_curve.png ...")
fig, ax = plt.subplots(figsize=(14, 5))

s_cum = spread_r.cumulative_pnl
c_cum = call_r.cumulative_pnl

ax.plot(s_cum.index, s_cum.values, color=SPREAD_C, linewidth=2,
        label=f"Spread v1.5  (${spread_r.total_pnl:+.2f}, Sharpe {spread_r.sharpe_ratio:.2f})", zorder=3)
ax.plot(c_cum.index, c_cum.values, color=CALL_C, linewidth=2,
        label=f"Call v1.6    (${call_r.total_pnl:+.2f}, Sharpe {call_r.sharpe_ratio:.2f})", zorder=3)

# Shade under each curve
ax.fill_between(s_cum.index, s_cum.values, 0, where=s_cum.values >= 0,
                alpha=0.07, color=SPREAD_C)
ax.fill_between(s_cum.index, s_cum.values, 0, where=s_cum.values < 0,
                alpha=0.07, color=LOSS_C)
ax.fill_between(c_cum.index, c_cum.values, 0, where=c_cum.values >= 0,
                alpha=0.07, color=CALL_C)
ax.fill_between(c_cum.index, c_cum.values, 0, where=c_cum.values < 0,
                alpha=0.07, color=LOSS_C)

# Trade exit dots
for t in spread_r.trades:
    dt = pd.Timestamp(t["exit_date"])
    if dt in s_cum.index:
        col = WIN_C if t['pnl'] > 0 else LOSS_C
        ax.scatter(dt, s_cum.loc[dt], marker='o', color=col, s=55, zorder=5, edgecolors=SPREAD_C, linewidths=1)

for t in call_r.trades:
    dt = pd.Timestamp(t["exit_date"])
    if dt in c_cum.index:
        col = WIN_C if t['pnl'] > 0 else LOSS_C
        ax.scatter(dt, c_cum.loc[dt], marker='^', color=col, s=60, zorder=5, edgecolors=CALL_C, linewidths=1)

ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.set_title("Cumulative P&L: Spread v1.5 vs Outright Call v1.6  |  Real NBBO, Hybrid Pricing",
             fontsize=11, fontweight='bold')
ax.set_ylabel("Cumulative P&L ($)")
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(dollar_fmt)
fig.tight_layout()
fig.savefig(output_dir / "v16_equity_curve.png", dpi=150)
plt.close()
print("  Saved v16_equity_curve.png")

# ============================================================
# CHART 2 — Trade distribution
# ============================================================
print("Generating v16_trade_distribution.png ...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                gridspec_kw={'width_ratios': [2, 1]})

trades = call_r.trades
pnls = [t['pnl'] for t in trades]
bar_colors = [WIN_C if p > 0 else LOSS_C for p in pnls]
tick_labels = [f"#{t['id']} {t['entry_date'][5:]}\n{t['exit_reason'].replace('EXIT_','')[:10]}" for t in trades]

bars = ax1.bar(range(len(trades)), pnls, color=bar_colors, edgecolor='white', linewidth=0.5, zorder=3)
ax1.set_xticks(range(len(trades)))
ax1.set_xticklabels(tick_labels, fontsize=8)
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax1.set_title("Call v1.6 — Individual Trade P&L", fontsize=11, fontweight='bold')
ax1.set_ylabel("P&L ($)")
ax1.yaxis.set_major_formatter(dollar_fmt)
for bar, val in zip(bars, pnls):
    sign = '+' if val >= 0 else ''
    yoff = 0.04 if val >= 0 else -0.05
    va   = 'bottom' if val >= 0 else 'top'
    ax1.text(bar.get_x() + bar.get_width()/2, val + yoff, f"{sign}${val:.2f}",
             ha='center', va=va, fontsize=8, fontweight='bold')

# Avg P&L by exit type
exit_groups: dict = {}
for t in trades:
    key = t['exit_reason'].replace('EXIT_', '')
    exit_groups.setdefault(key, []).append(t['pnl'])

g_labels  = list(exit_groups.keys())
g_avgs    = [np.mean(v) for v in exit_groups.values()]
g_totals  = [sum(v)     for v in exit_groups.values()]
g_counts  = [len(v)     for v in exit_groups.values()]
g_colors  = [WIN_C if a > 0 else LOSS_C for a in g_avgs]

bars2 = ax2.barh(g_labels, g_avgs, color=g_colors, edgecolor='white', linewidth=0.5, zorder=3)
ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8)
ax2.set_title("Avg P&L by Exit Type", fontsize=11, fontweight='bold')
ax2.set_xlabel("Avg P&L ($)")
ax2.xaxis.set_major_formatter(dollar_fmt)
for bar, avg, cnt, tot in zip(bars2, g_avgs, g_counts, g_totals):
    sign = '+' if avg >= 0 else ''
    xoff = 0.03 if avg >= 0 else -0.03
    ha   = 'left'  if avg >= 0 else 'right'
    ax2.text(avg + xoff, bar.get_y() + bar.get_height()/2,
             f"{sign}${avg:.2f}  (n={cnt}, tot={sign}${tot:.2f})",
             va='center', ha=ha, fontsize=8, fontweight='bold')

fig.suptitle("Call v1.6 Trade Analysis  |  75% Profit Target, No Scale-Out, Real NBBO",
             fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig(output_dir / "v16_trade_distribution.png", dpi=150)
plt.close()
print("  Saved v16_trade_distribution.png")

# ============================================================
# CHART 3 — Monthly P&L
# ============================================================
print("Generating v16_monthly_pnl.png ...")

def _monthly(tlist):
    d = {}
    for t in tlist:
        ym = t['exit_date'][:7]
        d[ym] = d.get(ym, 0) + t['pnl']
    return d

s_mo = _monthly(spread_r.trades)
c_mo = _monthly(call_r.trades)
all_months = sorted(set(s_mo) | set(c_mo))
x = np.arange(len(all_months))

s_vals = [s_mo.get(m, 0) for m in all_months]
c_vals = [c_mo.get(m, 0) for m in all_months]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharey=False)

c_colors = [WIN_C if v > 0 else (LOSS_C if v < 0 else NEUTRAL) for v in c_vals]
bars1 = ax1.bar(x, c_vals, color=c_colors, edgecolor='white', linewidth=0.4, zorder=3)
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax1.set_title("Call v1.6 — Monthly P&L", fontsize=11, fontweight='bold')
ax1.set_ylabel("P&L ($)")
ax1.set_xticks(x)
ax1.set_xticklabels(all_months, rotation=45, ha='right', fontsize=8)
ax1.yaxis.set_major_formatter(dollar_fmt)
for bar, val in zip(bars1, c_vals):
    if val != 0:
        sign = '+' if val >= 0 else ''
        yoff = 0.02 if val >= 0 else -0.03
        va = 'bottom' if val >= 0 else 'top'
        ax1.text(bar.get_x() + bar.get_width()/2, val + yoff,
                 f"{sign}${val:.2f}", ha='center', va=va, fontsize=7, fontweight='bold')

s_colors = [WIN_C if v > 0 else (LOSS_C if v < 0 else NEUTRAL) for v in s_vals]
bars2 = ax2.bar(x, s_vals, color=s_colors, edgecolor='white', linewidth=0.4, zorder=3)
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax2.set_title("Spread v1.5 — Monthly P&L (reference)", fontsize=11, fontweight='bold')
ax2.set_ylabel("P&L ($)")
ax2.set_xticks(x)
ax2.set_xticklabels(all_months, rotation=45, ha='right', fontsize=8)
ax2.yaxis.set_major_formatter(dollar_fmt)
for bar, val in zip(bars2, s_vals):
    if val != 0:
        sign = '+' if val >= 0 else ''
        yoff = 0.01 if val >= 0 else -0.02
        va = 'bottom' if val >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2, val + yoff,
                 f"{sign}${val:.2f}", ha='center', va=va, fontsize=7, fontweight='bold')

fig.tight_layout()
fig.savefig(output_dir / "v16_monthly_pnl.png", dpi=150)
plt.close()
print("  Saved v16_monthly_pnl.png")

# ============================================================
# CHART 4 — Version comparison
# ============================================================
print("Generating v16_version_comparison.png ...")

# Historical stats from review packages (synthetic pricing for v1.0-v1.3)
VERSION_STATS = [
    # label   sharpe   pf     max_dd   pricing
    ("v1.0",   1.41,   2.66,  -14.28,  "synth"),
    ("v1.1",   1.76,   3.31,   -8.22,  "synth"),
    ("v1.2",   2.02,   3.62,   -4.77,  "synth"),
    ("v1.3",   1.98,   5.34,   -4.98,  "synth"),
    ("v1.4",  -0.94,   0.57,   -5.50,  "real"),
    ("v1.5",  spread_r.sharpe_ratio, spread_r.profit_factor, spread_r.max_drawdown, "real"),
    ("v1.6",  call_r.sharpe_ratio,   call_r.profit_factor,   call_r.max_drawdown,   "real"),
]
v_labels  = [v[0] for v in VERSION_STATS]
v_sharpes = [v[1] for v in VERSION_STATS]
v_pfs     = [v[2] for v in VERSION_STATS]
v_dds     = [abs(v[3]) for v in VERSION_STATS]
v_pricing = [v[4] for v in VERSION_STATS]

def _bar_color(pricing, is_dd=False):
    if pricing == 'real':
        return LOSS_C if is_dd else CALL_C
    return '#A0A0A0'

real_colors  = [_bar_color(p) for p in v_pricing]
dd_colors    = [_bar_color(p, is_dd=True) for p in v_pricing]

x = np.arange(len(v_labels))

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# — Sharpe —
for i, (v, p) in enumerate(zip(v_sharpes, v_pricing)):
    axes[0].bar(i, v, color=real_colors[i], edgecolor='white', linewidth=0.5)
    sign = '+' if v >= 0 else ''
    yoff = 0.05 if v >= 0 else -0.12
    va   = 'bottom' if v >= 0 else 'top'
    axes[0].text(i, v + yoff, f"{sign}{v:.2f}", ha='center', va=va, fontsize=8, fontweight='bold')
    if p == 'synth':
        axes[0].text(i, min(v_sharpes) - 0.15, 'synth', ha='center', fontsize=6,
                     color='#999', rotation=90, va='top')
axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
axes[0].set_xticks(x); axes[0].set_xticklabels(v_labels)
axes[0].set_title("Sharpe Ratio", fontsize=11, fontweight='bold')
axes[0].set_ylabel("Sharpe")

# — Profit Factor (log scale for readability) —
for i, v in enumerate(v_pfs):
    axes[1].bar(i, v, color=real_colors[i], edgecolor='white', linewidth=0.5)
    axes[1].text(i, v + 0.05, f"{v:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
axes[1].axhline(1.0, color='orange', linestyle='--', linewidth=1.0, alpha=0.7, label='PF = 1.0 (breakeven)')
axes[1].set_xticks(x); axes[1].set_xticklabels(v_labels)
axes[1].set_title("Profit Factor", fontsize=11, fontweight='bold')
axes[1].set_ylabel("Profit Factor")
axes[1].legend(fontsize=8)

# — Max Drawdown —
for i, v in enumerate(v_dds):
    axes[2].bar(i, v, color=dd_colors[i], edgecolor='white', linewidth=0.5)
    axes[2].text(i, v + 0.05, f"${v:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
axes[2].set_xticks(x); axes[2].set_xticklabels(v_labels)
axes[2].set_title("Max Drawdown ($, absolute)", fontsize=11, fontweight='bold')
axes[2].set_ylabel("Max Drawdown ($)")
axes[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.0f}"))

fig.suptitle(
    "Strategy Version Comparison: v1.0 → v1.6\n"
    "Gray bars = synthetic pricing  |  Colored bars = real NBBO",
    fontsize=12, fontweight='bold'
)
fig.tight_layout()
fig.savefig(output_dir / "v16_version_comparison.png", dpi=150)
plt.close()
print("  Saved v16_version_comparison.png")

# ============================================================
# CHART 5 — Spread vs Call structural comparison
# ============================================================
print("Generating v16_spread_vs_call.png ...")
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.35)

# ── Panel A: cumulative P&L (full width) ──────────────────
ax_a = fig.add_subplot(gs[0, :])
ax_a.plot(s_cum.index, s_cum.values, color=SPREAD_C, linewidth=2,
          label=f"Spread v1.5  ${spread_r.total_pnl:+.2f}", zorder=3)
ax_a.plot(c_cum.index, c_cum.values, color=CALL_C, linewidth=2,
          label=f"Call v1.6    ${call_r.total_pnl:+.2f}", zorder=3)
ax_a.fill_between(s_cum.index, s_cum.values, 0, alpha=0.07, color=SPREAD_C)
ax_a.fill_between(c_cum.index, c_cum.values, 0, alpha=0.07, color=CALL_C)
ax_a.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax_a.set_title("Cumulative P&L: Spread v1.5 vs Call v1.6  |  Real NBBO", fontsize=11, fontweight='bold')
ax_a.legend(fontsize=10)
ax_a.yaxis.set_major_formatter(dollar_fmt)

# ── Panel B: summary metrics table ────────────────────────
ax_b = fig.add_subplot(gs[1, 0])
ax_b.axis('off')
_pf_s  = f"{spread_r.profit_factor:.2f}" if spread_r.profit_factor != float('inf') else "inf"
_pf_c  = f"{call_r.profit_factor:.2f}"   if call_r.profit_factor   != float('inf') else "inf"
rows = [
    ("",              "Spread v1.5",                       "Call v1.6",                         "Edge"),
    ("Trades",        str(spread_r.total_trades),          str(call_r.total_trades),             "—"),
    ("Win Rate",      f"{spread_r.win_rate:.1f}%",         f"{call_r.win_rate:.1f}%",            "~tie"),
    ("Avg Win",       f"+${spread_r.avg_win:.2f}",         f"+${call_r.avg_win:.2f}",
     "Call" if call_r.avg_win > spread_r.avg_win else "Spread"),
    ("Avg Loss",      f"${spread_r.avg_loss:.2f}",         f"${call_r.avg_loss:.2f}",
     "Spread" if abs(call_r.avg_loss) > abs(spread_r.avg_loss) else "Call"),
    ("Prof Factor",   _pf_s,                               _pf_c,
     "Call" if call_r.profit_factor > spread_r.profit_factor else "Spread"),
    ("Sharpe",        f"{spread_r.sharpe_ratio:.2f}",      f"{call_r.sharpe_ratio:.2f}",
     "Call" if call_r.sharpe_ratio > spread_r.sharpe_ratio else "Spread"),
    ("Total PnL",     f"${spread_r.total_pnl:+.2f}",      f"${call_r.total_pnl:+.2f}",
     "Call" if call_r.total_pnl > spread_r.total_pnl else "Spread"),
    ("Max DD",        f"${spread_r.max_drawdown:.2f}",     f"${call_r.max_drawdown:.2f}",
     "Spread" if abs(call_r.max_drawdown) > abs(spread_r.max_drawdown) else "Call"),
    ("Hold Days",     f"{spread_r.avg_holding_days:.0f}d", f"{call_r.avg_holding_days:.0f}d",   "—"),
]
col_xs = [0.01, 0.30, 0.56, 0.80]
for ri, row in enumerate(rows):
    is_header = ri == 0
    for ci, (val, cx) in enumerate(zip(row, col_xs)):
        wt = 'bold' if is_header else 'normal'
        fg = 'black'
        if not is_header:
            if val in ("Call",):
                fg = CALL_C
            elif val in ("Spread",):
                fg = SPREAD_C
        ax_b.text(cx, 1.0 - ri * 0.09, val, transform=ax_b.transAxes,
                  ha='left', va='top', fontsize=9, fontweight=wt, color=fg)
ax_b.set_title("Key Metrics Comparison", fontsize=11, fontweight='bold')

# ── Panel C: avg P&L by exit reason ───────────────────────
ax_c = fig.add_subplot(gs[1, 1])
EXIT_TYPES = ['REGIME_CHANGE', 'PROFIT_TARGET', 'TIME_STOP', 'STOP_LOSS']
SHORT_LABS  = ['Regime\nChange', 'Profit\nTarget', 'Time\nStop', 'Stop\nLoss']

def _avg_by_exit(result, exit_types):
    groups = {e: [] for e in exit_types}
    for t in result.trades:
        k = t['exit_reason'].replace('EXIT_', '')
        if k in groups:
            groups[k].append(t['pnl'])
    return [np.mean(groups[e]) if groups[e] else 0 for e in exit_types]

s_avgs = _avg_by_exit(spread_r, EXIT_TYPES)
c_avgs = _avg_by_exit(call_r,   EXIT_TYPES)
x2 = np.arange(len(EXIT_TYPES))
w2 = 0.38

ax_c.bar(x2 - w2/2, s_avgs, w2, color=SPREAD_C, label='Spread v1.5', edgecolor='white', linewidth=0.5)
ax_c.bar(x2 + w2/2, c_avgs, w2, color=CALL_C,   label='Call v1.6',   edgecolor='white', linewidth=0.5)
ax_c.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax_c.set_xticks(x2); ax_c.set_xticklabels(SHORT_LABS, fontsize=9)
ax_c.set_title("Avg P&L by Exit Type", fontsize=11, fontweight='bold')
ax_c.set_ylabel("Avg P&L ($)")
ax_c.legend(fontsize=9)
ax_c.yaxis.set_major_formatter(dollar_fmt)
for xi, (sv, cv) in enumerate(zip(s_avgs, c_avgs)):
    if sv != 0:
        ax_c.text(xi - w2/2, sv + (0.03 if sv >= 0 else -0.05),
                  f"${sv:.2f}", ha='center', va='bottom' if sv >= 0 else 'top', fontsize=7)
    if cv != 0:
        ax_c.text(xi + w2/2, cv + (0.03 if cv >= 0 else -0.05),
                  f"${cv:.2f}", ha='center', va='bottom' if cv >= 0 else 'top', fontsize=7)

fig.suptitle("Spread v1.5 vs Outright Call v1.6 — Structural Comparison  |  Real NBBO",
             fontsize=12, fontweight='bold')
fig.savefig(output_dir / "v16_spread_vs_call.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved v16_spread_vs_call.png")

# ============================================================
# CHART 6 — Hedge effectiveness (call v1.6)
# ============================================================
print("Generating v16_hedge_effectiveness.png ...")

# Build exit-day P&L series
pnl_rows = [{"date": pd.Timestamp(t["exit_date"]), "pnl": t["pnl"]} for t in call_r.trades]
if pnl_rows:
    pnl_df = pd.DataFrame(pnl_rows).groupby("date")["pnl"].sum()
else:
    pnl_df = pd.Series(dtype=float)
strategy_daily = pnl_df.reindex(df.index).fillna(0)

nq = df["NQ_Close"].dropna()
nq_ret = (nq.pct_change() * 100).reindex(df.index).fillna(0)
common_idx = strategy_daily.index.intersection(nq_ret.index)
strat = strategy_daily.loc[common_idx]
nq_r  = nq_ret.loc[common_idx]

# Conditional buckets
buckets = [
    ("Worst 1%\nNQ days",  nq_r.quantile(0.01), False),
    ("Worst 5%\nNQ days",  nq_r.quantile(0.05), False),
    ("Worst 10%\nNQ days", nq_r.quantile(0.10), False),
    ("All NQ\ndown days",  0.0,                 False),
    ("All NQ\nup days",    0.0,                 True),
]
b_avgs = []; b_tots = []; b_ns = []; b_avgnq = []
for _, thresh, is_up in buckets:
    mask = nq_r > thresh if is_up else nq_r <= thresh
    b_avgs.append(strat[mask].mean())
    b_tots.append(strat[mask].sum())
    b_ns.append(int(mask.sum()))
    b_avgnq.append(nq_r[mask].mean())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: conditional avg P&L
b_labs  = [b[0] for b in buckets]
bc_cols = [WIN_C if v > 0 else (LOSS_C if v < 0 else NEUTRAL) for v in b_avgs]
bars = ax1.bar(b_labs, b_avgs, color=bc_cols, edgecolor='white', linewidth=0.5, zorder=3)
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax1.set_title("Call v1.6 — Avg P&L by NQ Regime\n(Exit-day attribution)", fontsize=10, fontweight='bold')
ax1.set_ylabel("Avg Strategy P&L ($)")
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.4f}"))
for bar, avg, tot, n in zip(bars, b_avgs, b_tots, b_ns):
    sign = '+' if tot >= 0 else ''
    yoff = 0.00005 if avg >= 0 else -0.0001
    va   = 'bottom' if avg >= 0 else 'top'
    ax1.text(bar.get_x() + bar.get_width()/2, avg + yoff,
             f"${avg:+.4f}\n(n={n}, tot={sign}${tot:.2f})",
             ha='center', va=va, fontsize=7, fontweight='bold')

# Right: scatter — strategy P&L vs NQ return on active exit days
active = strat != 0
if active.sum() > 0:
    x_s = nq_r[active].values
    y_s = strat[active].values
    pt_cols = [WIN_C if p > 0 else LOSS_C for p in y_s]
    ax2.scatter(x_s, y_s, c=pt_cols, s=65, alpha=0.8,
                edgecolors='white', linewidths=0.5, zorder=3)
    if len(x_s) > 2:
        z = np.polyfit(x_s, y_s, 1)
        p = np.poly1d(z)
        xl = np.linspace(x_s.min(), x_s.max(), 60)
        ax2.plot(xl, p(xl), color='darkorange', linestyle='--', linewidth=1.2,
                 label=f"Trend  slope={z[0]:.3f}", zorder=2)
        corr = float(np.corrcoef(x_s, y_s)[0, 1])
        ax2.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax2.transAxes,
                 ha='left', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_title("Strategy P&L vs NQ Return\n(Active exit days only)", fontsize=10, fontweight='bold')
    ax2.set_xlabel("NQ Daily Return (%)")
    ax2.set_ylabel("Strategy P&L ($)")
    ax2.yaxis.set_major_formatter(dollar_fmt)
    ax2.legend(fontsize=9)
else:
    ax2.text(0.5, 0.5, "No active exit days\nin dataset", ha='center', va='center',
             transform=ax2.transAxes, fontsize=11)

fig.suptitle("Hedge Effectiveness: Call v1.6  |  NQ Conditional P&L + Correlation",
             fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(output_dir / "v16_hedge_effectiveness.png", dpi=150)
plt.close()
print("  Saved v16_hedge_effectiveness.png")

# ============================================================
# Console output: trade log + hedge stats + period breakdown
# ============================================================
W = 105
print()
print("=" * W)
print("FULL TRADE LOG — CALL v1.6  (outright long call, 75% profit target, real NBBO, hybrid pricing)")
print("=" * W)
print(f"  {'#':>3}  {'Entry':10}  {'Exit':10}  {'Strike':6}  "
      f"{'Entry$':>7}  {'Exit$':>7}  {'PnL':>9}  {'%':>5}  {'Hold':>5}  Reason")
print(f"  {'-'*95}")
for t in call_r.trades:
    sign = "+" if t['pnl'] >= 0 else ""
    print(f"  #{t['id']:>3}  {t['entry_date']:10}  {t['exit_date']:10}  "
          f"C{t['long_strike']:<5}  "
          f"${t['entry_price']:>5.2f}   ${t['exit_price']:>5.2f}  "
          f"{sign}${abs(t['pnl']):>6.2f}  "
          f"{sign}{abs(t['pnl_pct']):>3.0f}%  "
          f"{t['holding_days']:>4}d  {t['exit_reason']}")
print()
pf_s = f"{call_r.profit_factor:.2f}" if call_r.profit_factor != float('inf') else "inf"
print(f"  Totals: {call_r.total_trades} trades | "
      f"{call_r.winning_trades}W / {call_r.losing_trades}L | "
      f"WR={call_r.win_rate:.1f}% | PF={pf_s} | "
      f"PnL=${call_r.total_pnl:.2f} | MaxDD=${call_r.max_drawdown:.2f} | "
      f"Sharpe={call_r.sharpe_ratio:.2f}")

print()
print("=" * W)
print("HEDGE EFFECTIVENESS — CALL v1.6  (exit-day attribution vs NQ daily return)")
print("=" * W)
print(f"\n  {'Bucket':<28}  {'N':>4}  {'Avg NQ%':>8}  {'Avg Strat$':>12}  {'Total Strat$':>13}")
print(f"  {'-'*72}")
for (label, thresh, is_up), avg, tot, n, avgnq in zip(buckets, b_avgs, b_tots, b_ns, b_avgnq):
    label_clean = label.replace('\n', ' ')
    sign = '+' if tot >= 0 else ''
    print(f"  {label_clean:<28}  {n:>4}  {avgnq:>+7.2f}%  ${avg:>+10.4f}  ${tot:>+11.2f}")

corr_overall = float(strat.corr(nq_r))
print(f"\n  Overall correlation (strategy P&L vs NQ return): {corr_overall:.3f}")
down_active = (nq_r < 0) & (strat != 0)
if down_active.sum() > 0:
    corr_down = float(strat[down_active].corr(nq_r[down_active]))
    print(f"  Correlation on NQ down days (active exits only): {corr_down:.3f}")

print()
print("=" * W)
print("PERIOD BREAKDOWN — CALL v1.6  (by calendar year, exit date)")
print("=" * W)
print(f"\n  {'Year':<6}  {'Trades':>6}  {'Wins':>4}  {'Win%':>6}  {'PnL':>9}  Trades")
print(f"  {'-'*85}")
year_map: dict = {}
for t in call_r.trades:
    yr = t['exit_date'][:4]
    year_map.setdefault(yr, []).append(t)
for yr in sorted(year_map):
    yr_t = year_map[yr]
    wins  = [t for t in yr_t if t['pnl'] > 0]
    total = sum(t['pnl'] for t in yr_t)
    wr    = len(wins) / len(yr_t) * 100
    sign  = '+' if total >= 0 else ''
    details = '  '.join(
        f"#{t['id']} {t['entry_date'][5:]}->{t['exit_date'][5:]} "
        f"C{t['long_strike']} {('+' if t['pnl']>=0 else '')}${t['pnl']:.2f}"
        for t in yr_t
    )
    print(f"  {yr:<6}  {len(yr_t):>6}  {len(wins):>4}  {wr:>5.1f}%  "
          f"{sign}${abs(total):>6.2f}  {details}")

print()
print(f"Charts saved to: {output_dir}/")
print()
print("=" * W)
print("Done.")

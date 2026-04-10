"""
test/quote_coverage_audit2.py
=============================
Deep validation of synthetic exit prices for trades 6, 7, 8 —
the three winning trades where both entry AND exit were synthetic.

Method:
  1. Load vix_strategy_data.parquet for actual VIX futures values
  2. For each trade, show the UX column values on entry and exit date
  3. Verify synthetic pricing formula against actual futures
  4. For trade 6 (Sep 2-6): check the Aug-21 chain (earlier expiry) to see
     if VVIX or other data can validate the spike magnitude
  5. Check if any chain parquet covers the critical exit dates via a
     different expiry file (cross-check)
  6. Show what the corrected P&L would be if real mid was used at entry
     (using first available chain price as a proxy)

Usage:
    python test/quote_coverage_audit2.py
"""

import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime, timedelta
import calendar

# ============================================================
# Replicate expiry calendar (same as audit script 1)
# ============================================================
def third_friday(year, month):
    cal = calendar.monthcalendar(year, month)
    fridays = [week[4] for week in cal if week[4] != 0]
    return date(year, month, fridays[2])

def build_vix_expiry_calendar(start_year, end_year):
    expiries = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            next_m = m + 1 if m < 12 else 1
            next_y = y if m < 12 else y + 1
            tf = third_friday(next_y, next_m)
            expiries.append(tf - timedelta(days=30))
    return sorted(expiries)

EXPIRY_CAL = build_vix_expiry_calendar(2022, 2027)
DTE_MIN, DTE_MAX = 50, 80

def find_target_expiry(entry_date):
    for exp in EXPIRY_CAL:
        dte = (exp - entry_date).days
        if dte < DTE_MIN: continue
        if dte > DTE_MAX: return None
        return exp
    return None

def get_ux_col(current_date, target_expiry):
    n = 0
    for exp in EXPIRY_CAL:
        if exp <= current_date: continue
        n += 1
        if exp == target_expiry:
            return f"UX{n}" if n <= 9 else None
    return None

# ============================================================
# Load strategy data (contains actual UX futures prices)
# ============================================================
print("Loading strategy data ...")
df = pd.read_parquet("outputs/cache/vix_strategy_data.parquet")
df.index = pd.to_datetime(df.index)
print(f"Data: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
print()

# ============================================================
# The three synthetic-exit winning trades
# ============================================================
WINNING_TRADES = [
    # (label, entry_date, exit_date, long_k, short_k, eng_entry, eng_exit, pnl)
    ("T6 Sep 2024", "2024-09-02", "2024-09-06", 16, 21, 1.78, 4.32, 2.00),
    ("T7 Dec 2024", "2024-12-12", "2024-12-19", 16, 21, 1.81, 4.86, 2.61),
    ("T8 Dec 2024", "2024-12-13", "2024-12-19", 16, 21, 1.80, 4.86, 2.61),
]

# ============================================================
# For each synthetic-exit winner, validate pricing
# ============================================================
print("=" * 100)
print("PART A: SYNTHETIC EXIT PRICE VALIDATION")
print("VIX futures at entry and exit vs synthetic model price")
print("=" * 100)

def synthetic_spread_price(vix_futures, long_k, short_k, entry_price,
                            current_dte, entry_dte):
    """Replicate the engine's synthetic fallback model."""
    intrinsic = max(vix_futures - long_k, 0) - max(vix_futures - short_k, 0)
    if current_dte > 0:
        time_ratio = np.sqrt(current_dte / entry_dte) if entry_dte > 0 else 0
        initial_time_value = max(entry_price - (max(vix_futures - long_k, 0) -
                                                 max(vix_futures - short_k, 0)), 0)
        # NOTE: at entry, intrinsic is computed vs entry-day futures, not exit-day
        # The model uses initial_time_value = max(entry_price - intrinsic_at_entry, 0)
        # But at exit we don't have entry-day intrinsic here -- approximate
        time_value = initial_time_value * time_ratio * 0.5
        return intrinsic + time_value
    else:
        return intrinsic

COMM = 1.32 * 2 / 100  # $0.0264

for label, entry_str, exit_str, lk, sk, eng_entry, eng_exit, pnl in WINNING_TRADES:
    entry_dt = datetime.strptime(entry_str, "%Y-%m-%d").date()
    exit_dt  = datetime.strptime(exit_str,  "%Y-%m-%d").date()
    target_exp = find_target_expiry(entry_dt)
    expiry_str = target_exp.strftime("%Y-%m-%d")
    entry_dte = (target_exp - entry_dt).days
    exit_dte  = (target_exp - exit_dt).days

    entry_ts = pd.Timestamp(entry_str)
    exit_ts  = pd.Timestamp(exit_str)

    # Get the expiry-matched UX column for entry and exit dates
    ux_at_entry = get_ux_col(entry_dt, target_exp)
    ux_at_exit  = get_ux_col(exit_dt,  target_exp)

    ux_entry_val = float(df.loc[entry_ts, ux_at_entry]) if entry_ts in df.index and ux_at_entry else float("nan")
    ux_exit_val  = float(df.loc[exit_ts,  ux_at_exit])  if exit_ts  in df.index and ux_at_exit  else float("nan")

    vix_spot_entry = float(df.loc[entry_ts, "VIX_Spot"]) if "VIX_Spot" in df.columns and entry_ts in df.index else float("nan")
    vix_spot_exit  = float(df.loc[exit_ts,  "VIX_Spot"]) if "VIX_Spot" in df.columns and exit_ts  in df.index else float("nan")

    # Intrinsic values
    intrinsic_entry = max(ux_entry_val - lk, 0) - max(ux_entry_val - sk, 0)
    intrinsic_exit  = max(ux_exit_val  - lk, 0) - max(ux_exit_val  - sk, 0)

    # Expected synthetic exit price (model)
    # Need entry intrinsic (based on entry futures level) for time value calc
    initial_tv = max(eng_entry - intrinsic_entry, 0)
    time_ratio = np.sqrt(exit_dte / entry_dte) if entry_dte > 0 else 0
    synth_exit_computed = intrinsic_exit + initial_tv * time_ratio * 0.5

    # Is chain available for exit date?
    fname = f"vix_options_{expiry_str[:4]}_{expiry_str[5:7]}.parquet"
    fpath = Path(f"outputs/cache/vix_option_chains/{fname}")
    chain_start = None
    first_avail_mid = None
    if fpath.exists():
        chain = pd.read_parquet(fpath)
        chain.index = pd.to_datetime(chain.index)
        chain = chain.sort_index()
        chain_start = chain.index[0].date()

    print(f"\n{'='*70}")
    print(f"  {label}  C{lk}/C{sk}  expiry={expiry_str}")
    print(f"{'='*70}")
    print(f"  Entry {entry_str}  DTE={entry_dte}  {ux_at_entry}={ux_entry_val:.2f}  VIX_Spot={vix_spot_entry:.2f}")
    print(f"  Exit  {exit_str}   DTE={exit_dte}  {ux_at_exit}={ux_exit_val:.2f}  VIX_Spot={vix_spot_exit:.2f}")
    print()
    print(f"  Intrinsic at entry: ${intrinsic_entry:.3f}  (futures={ux_entry_val:.2f}, C{lk} intrinsic=${max(ux_entry_val-lk,0):.2f}, C{sk} intrinsic=${max(ux_entry_val-sk,0):.2f})")
    print(f"  Intrinsic at exit:  ${intrinsic_exit:.3f}  (futures={ux_exit_val:.2f}, C{lk} intrinsic=${max(ux_exit_val-lk,0):.2f}, C{sk} intrinsic=${max(ux_exit_val-sk,0):.2f})")
    print()
    print(f"  Engine entry price:  ${eng_entry:.3f}  (synthetic)")
    print(f"  Engine exit price:   ${eng_exit:.3f}  (synthetic)")
    print(f"  Reported P&L:        ${pnl:+.3f}")
    print()
    print(f"  Synthetic exit model reconstruction:")
    print(f"    initial_time_value = max(eng_entry - intrinsic_entry, 0) = max({eng_entry:.3f} - {intrinsic_entry:.3f}, 0) = ${initial_tv:.3f}")
    print(f"    time_ratio = sqrt({exit_dte}/{entry_dte}) = {time_ratio:.4f}")
    print(f"    synth_exit = {intrinsic_exit:.3f} + {initial_tv:.3f} * {time_ratio:.4f} * 0.5 = ${synth_exit_computed:.3f}")
    print(f"    engine exit = ${eng_exit:.3f}  vs reconstructed = ${synth_exit_computed:.3f}")

    if chain_start:
        print(f"\n  Chain for {expiry_str}: starts {chain_start}  (exit was {exit_str} — {(chain_start - exit_dt).days} days AFTER exit)")

    # What would P&L be if entry was at real NBBO (first available mid)?
    # First available mid was: T6=$1.17, T7=$1.125, T8=$1.125
    real_entry_mids = {
        "T6 Sep 2024": 1.170,
        "T7 Dec 2024": 1.125,
        "T8 Dec 2024": 1.125,
    }
    real_mid_entry = real_entry_mids[label]
    real_entry_cost = real_mid_entry + COMM  # mid + commission

    # Cannot directly validate real exit — but can bound it:
    # Max spread value = spread_width = 5
    # Intrinsic at exit = already computed above
    print(f"\n  CORRECTED ENTRY ANALYSIS:")
    print(f"    Real NBBO mid (first chain date): ${real_mid_entry:.3f}")
    print(f"    Real entry cost (mid + commission): ${real_entry_cost:.3f}")
    print(f"    Synthetic entry was: ${eng_entry:.3f}")
    print(f"    Synthetic entry overstatement: ${eng_entry - real_entry_cost:+.3f}")
    print()
    # If exit price (synthetic) is accurate, and we correct the entry:
    corrected_pnl = pnl + (eng_entry - real_entry_cost)
    print(f"    If exit price is correct & entry replaced with real NBBO:")
    print(f"      Corrected P&L = reported ${pnl:.3f} + overstatement ${eng_entry - real_entry_cost:+.3f} = ${corrected_pnl:.3f}")
    print(f"      (HIGHER than reported — synthetic entry overpriced = start from lower basis = larger gain)")

# ============================================================
# PART B: VIX spot and futures path around critical dates
# ============================================================
print()
print("=" * 100)
print("PART B: VIX FUTURES PATH AROUND CRITICAL TRADE DATES")
print("=" * 100)

WINDOWS = [
    ("Sep 2024 winner (T6)", "2024-08-26", "2024-09-13", "2024-09-02", "2024-09-06"),
    ("Dec 2024 winners (T7/T8)", "2024-12-06", "2024-12-27", "2024-12-12", "2024-12-19"),
]

for (label, start, end, entry_str, exit_str) in WINDOWS:
    entry_dt = datetime.strptime(entry_str, "%Y-%m-%d").date()
    target_exp = find_target_expiry(entry_dt)

    print(f"\n  {label}")
    print(f"  Expiry used: {target_exp}  (chain: vix_options_{target_exp.strftime('%Y_%m')}.parquet)")
    print()

    subset = df.loc[start:end]
    cols = ["VIX_Spot"]
    # Add UX1-UX5 if available
    for col in ["UX1", "UX2", "UX3", "UX4", "UX5"]:
        if col in df.columns:
            cols.append(col)

    print(f"  {'Date':12}  {'VIX_Spot':>8}", end="")
    for col in cols[1:]:
        print(f"  {col:>6}", end="")

    # Show which UX column is the expiry-matched one for each date
    print(f"  {'MatchedUX':>9}  {'FuturesVal':>10}  Marker")
    print(f"  {'-'*85}")

    for ts, row in subset.iterrows():
        d = ts.date()
        ux_col = get_ux_col(d, target_exp)
        ux_val = float(row[ux_col]) if ux_col and ux_col in row.index else float("nan")
        vix_s  = float(row.get("VIX_Spot", float("nan")))

        marker = ""
        if str(d) == entry_str:  marker = " <-- ENTRY"
        elif str(d) == exit_str: marker = " <-- EXIT"

        print(f"  {str(d):12}  {vix_s:>8.2f}", end="")
        for col in cols[1:]:
            v = float(row.get(col, float("nan")))
            print(f"  {v:>6.2f}", end="")
        print(f"  {str(ux_col):>9}  {ux_val:>10.2f}{marker}")

# ============================================================
# PART C: Cross-check using an adjacent chain
# ============================================================
print()
print("=" * 100)
print("PART C: ADJACENT CHAIN CROSS-CHECK (can any other chain validate exit pricing?)")
print("=" * 100)

# For T6 (Sep 2 entry, Sep 6 exit, Nov expiry):
# The Aug 2024 chain would cover up to ~Aug 21 expiry. Not helpful for Sep 6.
# But the Oct 2024 chain might start around Sep 2024.
# For T7/T8 (Dec 12-19, Feb expiry):
# The Jan 2025 chain might cover Dec 2024.

chains_to_check = [
    ("Oct 2024 chain (for Sep 2024 context)", "outputs/cache/vix_option_chains/vix_options_2024_10.parquet", 16, 21, "2024-09-02", "2024-09-08"),
    ("Nov 2024 chain (T6 expiry — when does it start?)", "outputs/cache/vix_option_chains/vix_options_2024_11.parquet", 16, 21, "2024-09-01", "2024-09-30"),
    ("Jan 2025 chain (for Dec 2024 context)", "outputs/cache/vix_option_chains/vix_options_2025_01.parquet", 16, 21, "2024-12-09", "2024-12-23"),
    ("Feb 2025 chain (T7/T8 expiry — when does it start?)", "outputs/cache/vix_option_chains/vix_options_2025_02.parquet", 16, 21, "2024-12-09", "2024-12-30"),
]

for label, fpath_str, lk, sk, start, end in chains_to_check:
    fpath = Path(fpath_str)
    print(f"\n  {label}")
    if not fpath.exists():
        print(f"    FILE NOT FOUND: {fpath_str}")
        continue
    chain = pd.read_parquet(fpath)
    chain.index = pd.to_datetime(chain.index)
    chain = chain.sort_index()
    chain_start = chain.index[0].date()
    chain_end   = chain.index[-1].date()
    print(f"    Coverage: {chain_start} to {chain_end}  ({len(chain)} rows)")

    # Show rows within window
    subset = chain.loc[start:end]
    if subset.empty:
        print(f"    (no rows in window {start} to {end})")
        continue

    print(f"    {'Date':12}  {'C{lk}_Bid':>9}  {'C{lk}_Mid':>9}  {'C{lk}_Ask':>9}  {'C{sk}_Bid':>9}  {'C{sk}_Mid':>9}  {'SpreadMid':>9}")
    print(f"    {'-'*80}")
    for ts, row in subset.iterrows():
        lb = row.get(f"C{lk}_Bid", np.nan)
        lm = row.get(f"C{lk}_Mid", np.nan)
        la = row.get(f"C{lk}_Ask", np.nan)
        sb = row.get(f"C{sk}_Bid", np.nan)
        sm = row.get(f"C{sk}_Mid", np.nan)
        spread_mid = (float(lm) - float(sm)) if (not pd.isna(lm) and not pd.isna(sm)) else float("nan")
        def fmt(v): return f"{float(v):>9.3f}" if not pd.isna(v) else f"{'N/A':>9}"
        spread_str = f"{spread_mid:>9.3f}" if not np.isnan(spread_mid) else f"{'N/A':>9}"
        print(f"    {str(ts.date()):12}  {fmt(lb)}  {fmt(lm)}  {fmt(la)}  {fmt(sb)}  {fmt(sm)}  {spread_str}")

print()
print("=" * 100)
print("SUMMARY: WHAT THIS MEANS FOR THE +$4.71 RESULT")
print("=" * 100)
print()
print("  Trades with REAL entry AND real exit:  T2, T3 (2 trades, losing)")
print("  Trades with REAL entry, synthetic MTM: T2 is real exit, T3 is real exit (regime change caught in chain)")
print("  Trades with SYNTHETIC entry AND exit:  T6, T7, T8 (3 winners — these are the big 3)")
print()
print("  The +$4.71 result depends critically on trades T6, T7, T8 being correctly priced.")
print("  Their exits occurred BEFORE the chain parquets started, so there are NO real NBBO quotes")
print("  for the exit dates. The exit prices ($4.32, $4.86, $4.86) are ALL synthetic.")
print()
print("  The synthetic model at exit: spread_price = intrinsic(VIX_futures) + time_value")
print("  Time value component is conservative (x0.5 factor), but intrinsic depends on actual futures prices.")
print("  See Part B above for actual futures values on those exit dates.")
print()

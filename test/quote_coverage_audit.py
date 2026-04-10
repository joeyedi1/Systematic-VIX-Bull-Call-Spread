"""
test/quote_coverage_audit.py
============================
Quote coverage audit for v1.5 trades.

For each of the 8 v1.5 trades, report:
  - Entry date
  - Expiry date selected by the engine
  - Entry DTE
  - First date in the option chain parquet for that expiry
  - Whether entry price was real NBBO or synthetic fallback
  - If synthetic: what was the real NBBO mid on the first available chain date
  - Synthetic entry price vs real NBBO at first available date (discrepancy)

Usage:
    python test/quote_coverage_audit.py
"""

import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime, timedelta
import calendar

from data.option_chain_store import OptionChainStore

# ============================================================
# 1. Replicate engine expiry logic exactly
# ============================================================

def third_friday(year: int, month: int) -> date:
    cal = calendar.monthcalendar(year, month)
    fridays = [week[4] for week in cal if week[4] != 0]
    return date(year, month, fridays[2])

def build_vix_expiry_calendar(start_year: int, end_year: int):
    expiries = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            next_m = m + 1 if m < 12 else 1
            next_y = y if m < 12 else y + 1
            tf = third_friday(next_y, next_m)
            expiry = tf - timedelta(days=30)
            expiries.append(expiry)
    return sorted(expiries)

EXPIRY_CAL = build_vix_expiry_calendar(2022, 2027)
DTE_MIN, DTE_MAX = 50, 80  # v1.5 dte_range

def find_target_expiry(entry_date: date):
    for exp in EXPIRY_CAL:
        dte = (exp - entry_date).days
        if dte < DTE_MIN:
            continue
        if dte > DTE_MAX:
            return None
        return exp
    return None

# ============================================================
# 2. v1.5 trade list (from review package / test run)
# ============================================================
# Entry date, long_strike, short_strike, entry_price (from v1.5 run)
V15_TRADES = [
    # (entry_date,   long_k, short_k, engine_entry_price)
    ("2024-02-16",  15, 20, 1.71),
    ("2024-02-19",  15, 20, 1.06),
    ("2024-04-02",  15, 20, 1.03),
    ("2024-06-20",  15, 20, 1.63),
    ("2024-06-21",  14, 19, 1.63),
    ("2024-09-02",  16, 21, 1.78),
    ("2024-12-12",  16, 21, 1.81),
    ("2024-12-13",  16, 21, 1.80),
]

# ============================================================
# 3. Load chain store and audit each trade
# ============================================================
store = OptionChainStore()

print("Loading option chains ...")
print()

COMM = 1.32 * 2 / 100  # $0.0264 for 2-leg spread

W = 130
print("=" * W)
print("QUOTE COVERAGE AUDIT — v1.5 TRADES (hybrid pricing)")
print("=" * W)
print(
    f"{'#':>2}  {'Entry':10}  {'Expiry':10}  {'DTE':>4}  "
    f"{'Chain_Start':11}  {'Entry_Gap':>10}  "
    f"{'Engine_$':>9}  {'NBBO_mid_entry':>14}  "
    f"{'1st_avail_mid':>13}  {'Pricing':11}  {'Discrepancy':>12}"
)
print("-" * W)

results = []
for i, (entry_str, lk, sk, eng_price) in enumerate(V15_TRADES, 1):
    entry_dt = datetime.strptime(entry_str, "%Y-%m-%d").date()
    target_exp = find_target_expiry(entry_dt)
    if target_exp is None:
        print(f" {i:2d}  {entry_str}  *** NO EXPIRY IN DTE WINDOW ***")
        continue

    dte = (target_exp - entry_dt).days
    expiry_str = target_exp.strftime("%Y-%m-%d")

    # Load the chain for this expiry
    chain_key = expiry_str[:7]  # "YYYY-MM"
    fname = f"vix_options_{chain_key[:4]}_{chain_key[5:7]}.parquet"
    fpath = Path(f"outputs/cache/vix_option_chains/{fname}")

    if not fpath.exists():
        print(f" {i:2d}  {entry_str}  {expiry_str}  {dte:>4}  FILE MISSING: {fname}")
        continue

    chain = pd.read_parquet(fpath)
    chain.index = pd.to_datetime(chain.index)
    chain = chain.sort_index()

    chain_start = chain.index[0].date()
    chain_end   = chain.index[-1].date()

    # Days between entry and first available chain date
    entry_gap = (chain_start - entry_dt).days  # positive = chain starts AFTER entry

    # 3a. Was the entry date in the chain?
    entry_ts = pd.Timestamp(entry_str)
    entry_in_chain = entry_ts in chain.index

    # 3b. Real mid at entry date (if available)
    nbbo_mid_at_entry = None
    if entry_in_chain:
        row = chain.loc[entry_ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        lm = row.get(f"C{lk}_Mid", np.nan)
        sm = row.get(f"C{sk}_Mid", np.nan)
        if not pd.isna(lm) and not pd.isna(sm) and float(lm) > 0:
            nbbo_mid_at_entry = float(lm) - float(sm)

    # 3c. Real mid at first available chain date
    first_avail_mid = None
    first_avail_date = None
    for dt in chain.index:
        if dt.date() >= entry_dt:
            row = chain.loc[dt]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            lm = row.get(f"C{lk}_Mid", np.nan)
            sm = row.get(f"C{sk}_Mid", np.nan)
            if not pd.isna(lm) and not pd.isna(sm) and float(lm) > 0 and float(sm) >= 0:
                first_avail_mid = float(lm) - float(sm)
                first_avail_date = dt.date()
                break

    # Pricing determination
    if entry_in_chain and nbbo_mid_at_entry is not None:
        pricing = "REAL_NBBO"
        discrepancy = 0.0
    elif first_avail_date is not None and first_avail_date == entry_dt:
        pricing = "REAL_NBBO"   # same day but Mid column was NaN on entry date; ask mid
        discrepancy = 0.0
    else:
        pricing = "SYNTHETIC"
        # Discrepancy = engine price vs first real NBBO mid (both net of commission)
        real_ref = (first_avail_mid + COMM) if first_avail_mid is not None else None
        discrepancy = (eng_price - real_ref) if real_ref is not None else float("nan")

    # Format
    gap_str = f"+{entry_gap}d" if entry_gap > 0 else (f"{entry_gap}d" if entry_gap < 0 else "0d")
    nbbo_str = f"${nbbo_mid_at_entry:.3f}" if nbbo_mid_at_entry is not None else "N/A"
    first_str = f"${first_avail_mid:.3f}" if first_avail_mid is not None else "N/A"
    disc_str  = (f"${discrepancy:+.3f}" if not np.isnan(discrepancy) else "N/A")

    print(
        f" {i:2d}  {entry_str}  {expiry_str}  {dte:>4}  "
        f"{str(chain_start):11}  {gap_str:>10}  "
        f"${eng_price:>7.3f}  {nbbo_str:>14}  "
        f"{first_str:>13}  {pricing:11}  {disc_str:>12}"
    )

    results.append({
        "trade": i,
        "entry": entry_str,
        "expiry": expiry_str,
        "dte": dte,
        "chain_start": str(chain_start),
        "entry_gap": entry_gap,
        "engine_price": eng_price,
        "nbbo_mid_entry": nbbo_mid_at_entry,
        "first_avail_mid": first_avail_mid,
        "first_avail_date": str(first_avail_date) if first_avail_date else None,
        "pricing": pricing,
        "discrepancy": discrepancy,
    })

print("=" * W)

# ============================================================
# 4. Summary
# ============================================================
print()
real_trades  = [r for r in results if r["pricing"] == "REAL_NBBO"]
synth_trades = [r for r in results if r["pricing"] == "SYNTHETIC"]

print(f"Trades with real NBBO at entry:  {len(real_trades)}/{len(results)}")
print(f"Trades with synthetic at entry:  {len(synth_trades)}/{len(results)}")
print()

if synth_trades:
    print("SYNTHETIC ENTRY TRADES — detail:")
    for r in synth_trades:
        disc = r["discrepancy"]
        disc_str = f"${disc:+.3f}" if not np.isnan(disc) else "N/A"
        print(
            f"  Trade {r['trade']}  {r['entry']}  expiry={r['expiry']}  DTE={r['dte']}"
            f"  chain_starts={r['chain_start']}  gap={r['entry_gap']}d"
            f"  engine=${r['engine_price']:.3f}  first_real_mid=${r['first_avail_mid']:.3f}"
            f" on {r['first_avail_date']}  discrepancy={disc_str}"
        )
    print()

# ============================================================
# 5. Deep-dive: full chain coverage around each entry date
# ============================================================
print("=" * W)
print("CHAIN COVERAGE DETAIL — first 5 dates in chain vs entry date")
print("=" * W)

for i, (entry_str, lk, sk, eng_price) in enumerate(V15_TRADES, 1):
    entry_dt = datetime.strptime(entry_str, "%Y-%m-%d").date()
    target_exp = find_target_expiry(entry_dt)
    if target_exp is None:
        continue
    expiry_str = target_exp.strftime("%Y-%m-%d")
    chain_key  = expiry_str[:7]
    fname = f"vix_options_{chain_key[:4]}_{chain_key[5:7]}.parquet"
    fpath = Path(f"outputs/cache/vix_option_chains/{fname}")
    if not fpath.exists():
        continue

    chain = pd.read_parquet(fpath)
    chain.index = pd.to_datetime(chain.index)
    chain = chain.sort_index()

    print(f"\n  Trade {i}  entry={entry_str}  expiry={expiry_str}")
    print(f"  {'Date':12}  {'C{lk}_Bid':>9}  {'C{lk}_Mid':>9}  {'C{lk}_Ask':>9}  "
          f"{'C{sk}_Bid':>9}  {'C{sk}_Mid':>9}  {'C{sk}_Ask':>9}  "
          f"{'SpreadMid':>9}  In_Chain")
    print(f"  {'-'*95}")

    # Print 2 rows before entry date (if available) + entry date + 3 rows after
    entry_ts = pd.Timestamp(entry_str)
    nearby_idx = chain.index[
        (chain.index >= entry_ts - pd.Timedelta(days=5)) &
        (chain.index <= entry_ts + pd.Timedelta(days=5))
    ]

    for dt in nearby_idx:
        row = chain.loc[dt]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        lb = row.get(f"C{lk}_Bid", np.nan)
        lm = row.get(f"C{lk}_Mid", np.nan)
        la = row.get(f"C{lk}_Ask", np.nan)
        sb = row.get(f"C{sk}_Bid", np.nan)
        sm = row.get(f"C{sk}_Mid", np.nan)
        sa = row.get(f"C{sk}_Ask", np.nan)
        spread_mid = (float(lm) - float(sm)) if (not pd.isna(lm) and not pd.isna(sm)) else float("nan")
        marker = " <-- ENTRY" if dt.date() == entry_dt else ""
        def fmt(v): return f"{float(v):>9.3f}" if not pd.isna(v) else f"{'N/A':>9}"
        spread_str = f"{spread_mid:>9.3f}" if not np.isnan(spread_mid) else f"{'N/A':>9}"
        print(f"  {str(dt.date()):12}  {fmt(lb)}  {fmt(lm)}  {fmt(la)}  "
              f"{fmt(sb)}  {fmt(sm)}  {fmt(sa)}  {spread_str}{marker}")

print()
print("=" * W)
print("INTERPRETATION GUIDE")
print("=" * W)
print("  'REAL_NBBO'  = entry date IN the chain; get_spread_mid() returned a real mid price")
print("  'SYNTHETIC'  = entry date NOT in chain; engine fell back to StrikeSelector.estimated_cost")
print("  'Discrepancy'= engine_entry_price - first_real_NBBO_mid (positive = synthetic overpriced)")
print("  'Entry_Gap'  = chain_start - entry_date  (positive = chain starts AFTER entry)")
print()

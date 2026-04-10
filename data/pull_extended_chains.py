"""
data/pull_extended_chains.py
============================
Re-pull VIX option chain parquets with the extended 80-day window.

Previously chains started ~60 calendar days before expiry; the v1.5 audit
found that 6 of 8 entries and all 3 winning exits fell OUTSIDE that window
and used synthetic pricing.  80 days covers every entry and exit in v1.5.

What this script does:
  1. Compute the required start date (expiry - 80 days) for each of the 38
     expiry months.
  2. For each existing parquet, check whether its first row is LATER than the
     required start date.  If so, mark it as stale.
  3. Delete all stale files.
  4. Re-pull the stale files via Bloomberg with days_before_expiry=80.
  5. Print a before/after coverage summary for the 8 v1.5 trades.

Usage:
    python data/pull_extended_chains.py              # dry run — shows what would change
    python data/pull_extended_chains.py --execute    # delete stale files and re-pull

Bloomberg terminal must be running and blpapi installed for --execute.
"""

import sys; sys.path.insert(0, '.')
import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
import calendar
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHAIN_DIR  = Path("outputs/cache/vix_option_chains")
START_EXP  = "2023-01"
END_EXP    = "2026-03"
DAYS_WINDOW = 80       # new target — covers all v1.5 entries and exits
DAYS_OLD    = 60       # previous window — was the source of the gap

# ============================================================
# Expiry calendar (same logic as engine)
# ============================================================

def _third_friday(year: int, month: int) -> date:
    cal = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0]
    return date(year, month, fridays[2])


def build_expiry_list(start_ym: str, end_ym: str):
    """Return list of (expiry_date, tag) for every month in range."""
    sy, sm = int(start_ym[:4]), int(start_ym[5:7])
    ey, em = int(end_ym[:4]),   int(end_ym[5:7])
    expiries = []
    y, m = sy, sm
    while (y < ey) or (y == ey and m <= em):
        next_m = m + 1 if m < 12 else 1
        next_y = y     if m < 12 else y + 1
        tf     = _third_friday(next_y, next_m)
        exp    = tf - timedelta(days=30)
        tag    = f"{y:04d}_{m:02d}"
        expiries.append((exp, tag))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return expiries


# ============================================================
# v1.5 trades for before/after coverage check
# ============================================================

V15_TRADES = [
    # (label, entry_date, exit_date, long_k, short_k)
    ("T1", "2024-02-16", "2024-03-25", 15, 20),
    ("T2", "2024-02-19", "2024-04-08", 15, 20),
    ("T3", "2024-04-02", "2024-04-17", 15, 20),
    ("T4", "2024-06-20", "2024-07-26", 15, 20),
    ("T5", "2024-06-21", "2024-07-26", 14, 19),
    ("T6", "2024-09-02", "2024-09-06", 16, 21),
    ("T7", "2024-12-12", "2024-12-19", 16, 21),
    ("T8", "2024-12-13", "2024-12-19", 16, 21),
]

def find_expiry_for_entry(entry_str: str, all_expiries: list) -> date | None:
    """Same logic as BacktestEngine._find_target_expiry (DTE range 50-80)."""
    entry_dt = datetime.strptime(entry_str, "%Y-%m-%d").date()
    for exp, _ in all_expiries:
        dte = (exp - entry_dt).days
        if dte < 50: continue
        if dte > 80: return None
        return exp
    return None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Re-pull VIX option chains with 80-day window.")
    parser.add_argument("--execute", action="store_true",
                        help="Actually delete stale files and re-pull from Bloomberg.")
    args = parser.parse_args()

    all_expiries = build_expiry_list(START_EXP, END_EXP)

    # ----------------------------------------------------------
    # Step 1: Identify stale files
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"STALE FILE ANALYSIS  (window: {DAYS_OLD}d -> {DAYS_WINDOW}d)")
    print(f"{'='*70}")
    print(f"{'Expiry':12}  {'File':35}  {'Current_Start':13}  {'Required_Start':14}  {'Gap':>6}  Status")
    print(f"{'-'*90}")

    stale = []
    for exp_dt, tag in all_expiries:
        fname  = f"vix_options_{tag}.parquet"
        fpath  = CHAIN_DIR / fname
        req_start = exp_dt - timedelta(days=DAYS_WINDOW)

        if not fpath.exists():
            print(f"  {str(exp_dt):12}  {fname:35}  {'MISSING':13}  {str(req_start):14}  {'N/A':>6}  MISSING")
            stale.append((exp_dt, tag, fpath, None))
            continue

        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index)
        current_start = df.index.min().date()
        gap = (current_start - req_start).days  # positive = current is LATER than needed

        status = "OK" if gap <= 2 else f"STALE (+{gap}d)"  # 2-day tolerance for weekends
        if gap > 2:
            stale.append((exp_dt, tag, fpath, current_start))

        print(f"  {str(exp_dt):12}  {fname:35}  {str(current_start):13}  {str(req_start):14}  {gap:>+6}d  {status}")

    print(f"\n  {len(stale)} file(s) need updating out of {len(all_expiries)} total.")

    if not stale:
        print("\n  All files already have 80-day coverage. Nothing to do.")
        _print_coverage_summary(all_expiries)
        return

    print(f"\n  Files to update:")
    for exp_dt, tag, fpath, cs in stale:
        req = exp_dt - timedelta(days=DAYS_WINDOW)
        print(f"    {str(exp_dt):12}  vix_options_{tag}.parquet  (starts {cs}, needs {req})")

    # ----------------------------------------------------------
    # Step 2: Coverage check before (current state)
    # ----------------------------------------------------------
    _print_coverage_summary(all_expiries, label="CURRENT")

    if not args.execute:
        print("\n" + "="*70)
        print("DRY RUN — no files changed. Re-run with --execute to proceed.")
        print("Requires: Bloomberg Terminal running + blpapi installed.")
        print("="*70)
        return

    # ----------------------------------------------------------
    # Step 3: Delete stale files
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("DELETING STALE FILES")
    print(f"{'='*70}")
    for exp_dt, tag, fpath, cs in stale:
        if fpath.exists():
            fpath.unlink()
            print(f"  Deleted: {fpath.name}")

    # ----------------------------------------------------------
    # Step 4: Re-pull via Bloomberg
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"PULLING WITH days_before_expiry={DAYS_WINDOW}")
    print(f"{'='*70}")

    from data.bloomberg_fetcher import BloombergDataPipeline
    pipeline = BloombergDataPipeline(use_cache=False)

    # Only pull the stale months
    stale_tags = {tag for _, tag, _, _ in stale}
    stale_start = min(f"{tag[:4]}-{tag[5:7]}" for tag in stale_tags)
    stale_end   = max(f"{tag[:4]}-{tag[5:7]}" for tag in stale_tags)

    logger.info(f"Pulling expiries {stale_start} to {stale_end} with {DAYS_WINDOW}-day window ...")
    written = pipeline.fetch_option_chains(
        start_expiry=stale_start,
        end_expiry=stale_end,
        days_before_expiry=DAYS_WINDOW,
        force_refresh=False,   # we already deleted the stale ones; fresh files will be created
    )
    pipeline.close()

    print(f"\n  {len(written)} file(s) written.")

    # ----------------------------------------------------------
    # Step 5: Coverage check after
    # ----------------------------------------------------------
    _print_coverage_summary(all_expiries, label="AFTER PULL")


def _print_coverage_summary(all_expiries: list, label: str = "CURRENT"):
    """Show entry/exit chain coverage for each of the 8 v1.5 trades."""
    print(f"\n{'='*90}")
    print(f"v1.5 QUOTE COVERAGE ({label})")
    print(f"{'='*90}")
    print(f"  {'Trade':6}  {'Entry':10}  {'Exit':10}  {'Expiry':10}  "
          f"{'Chain_Start':12}  {'Entry_In':9}  {'Exit_In':8}  Notes")
    print(f"  {'-'*85}")

    for label_t, entry_str, exit_str, lk, sk in V15_TRADES:
        entry_dt = datetime.strptime(entry_str, "%Y-%m-%d").date()
        exit_dt  = datetime.strptime(exit_str,  "%Y-%m-%d").date()
        target_exp = find_expiry_for_entry(entry_str, all_expiries)
        if target_exp is None:
            print(f"  {label_t:6}  {entry_str}  {exit_str}  NO EXPIRY IN DTE WINDOW")
            continue

        tag   = f"{target_exp.year:04d}_{target_exp.month:02d}"
        fname = f"vix_options_{tag}.parquet"
        fpath = CHAIN_DIR / fname

        if not fpath.exists():
            print(f"  {label_t:6}  {entry_str}  {exit_str}  {str(target_exp):10}  {'FILE MISSING':12}  {'?':>9}  {'?':>8}")
            continue

        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index)
        chain_start = df.index.min().date()
        chain_end   = df.index.max().date()

        entry_ts = pd.Timestamp(entry_str)
        exit_ts  = pd.Timestamp(exit_str)

        entry_in_chain = entry_ts in df.index
        exit_in_chain  = exit_ts  in df.index

        # Also check real mid available (not just date present)
        entry_real = False
        if entry_in_chain:
            row = df.loc[entry_ts]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            import numpy as np
            lm = row.get(f"C{lk}_Mid", float("nan"))
            sm = row.get(f"C{sk}_Mid", float("nan"))
            if not pd.isna(lm) and not pd.isna(sm) and float(lm) > 0:
                entry_real = True

        exit_real = False
        if exit_in_chain:
            row = df.loc[exit_ts]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            lm = row.get(f"C{lk}_Mid", float("nan"))
            sm = row.get(f"C{sk}_Mid", float("nan"))
            if not pd.isna(lm) and not pd.isna(sm) and float(lm) > 0:
                exit_real = True

        entry_status = "REAL   " if entry_real else ("DATE_OK" if entry_in_chain else "MISSING")
        exit_status  = "REAL   " if exit_real  else ("DATE_OK" if exit_in_chain  else "MISSING")

        note = ""
        if not entry_real:
            gap = (chain_start - entry_dt).days
            note = f"entry gap={gap:+d}d"

        print(f"  {label_t:6}  {entry_str}  {exit_str}  {str(target_exp):10}  "
              f"{str(chain_start):12}  {entry_status:9}  {exit_status:8}  {note}")


if __name__ == "__main__":
    main()

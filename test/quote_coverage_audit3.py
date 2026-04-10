"""
test/quote_coverage_audit3.py
=============================
Final validation: using adjacent chains as exit price proxies.

The question: what would T6, T7, T8 exit proceeds actually be in the real market?
The synthetic model uses intrinsic(VIX_futures) + small_time_value.
The real market applies vol premium on the short OTM leg (C21) during VIX spikes,
which REDUCES spread proceeds vs intrinsic.

Method:
  - For T7/T8 (Dec 19 exit, Feb expiry): use Jan 2025 chain (real NBBO) as reference
    and adjust for DTE difference (62 DTE Feb vs 34 DTE Jan)
  - For T6 (Sep 6 exit, Nov expiry): Oct chain data is bad; bound using intrinsic approach
  - Build best/worst case corrected P&L range for the full v1.5 result

Usage:
    python test/quote_coverage_audit3.py
"""

import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime, timedelta
import calendar

# Re-run audit 1 findings (hardcoded from previous run for reference):
#
# Trade  Entry       Expiry       DTE  Chain_Start  Gap  Engine_$  1st_real_mid  Pricing
#   1    2024-02-16  2024-04-17    61  2024-02-19   +3d     1.710         1.030  SYNTHETIC
#   2    2024-02-19  2024-04-17    58  2024-02-19    0d     1.060         1.030  REAL_NBBO
#   3    2024-04-02  2024-05-22    50  2024-03-25   -8d     1.030         1.005  REAL_NBBO
#   4    2024-06-20  2024-08-21    62  2024-06-24   +4d     1.630         0.880  SYNTHETIC
#   5    2024-06-21  2024-08-21    61  2024-06-24   +3d     1.630         1.160  SYNTHETIC
#   6    2024-09-02  2024-11-20    79  2024-09-23  +21d     1.780         1.170  SYNTHETIC
#   7    2024-12-12  2025-02-19    69  2024-12-23  +11d     1.810         1.125  SYNTHETIC
#   8    2024-12-13  2025-02-19    68  2024-12-23  +10d     1.800         1.125  SYNTHETIC
#
# Exit prices from engine (hybrid mode):
#   T1: $0.37 exit  Mar 25 -> Apr chain covers this (REAL exit)
#   T2: $0.63 exit  Apr 08 -> Apr chain covers this (REAL exit)
#   T3: $1.38 exit  Apr 17 -> May chain covers this (REAL exit)
#   T4: $0.95 exit  Jul 26 -> Aug chain covers this (REAL exit)
#   T5: $1.22 exit  Jul 26 -> Aug chain covers this (REAL exit)
#   T6: $4.32 exit  Sep 06 -> Nov chain starts Sep 23 (SYNTHETIC exit)
#   T7: $4.86 exit  Dec 19 -> Feb chain starts Dec 23 (SYNTHETIC exit)
#   T8: $4.86 exit  Dec 19 -> Feb chain starts Dec 23 (SYNTHETIC exit)

COMM = 1.32 * 2 / 100  # 2-leg commission = $0.0264

# ============================================================
# PART A: Verify that exit dates T1-T5 ARE in the chain
# ============================================================
print("=" * 90)
print("PART A: EXIT DATE CHAIN COVERAGE (T1-T5, which have real exits)")
print("=" * 90)

exit_checks = [
    ("T1", "2024-03-25", "outputs/cache/vix_option_chains/vix_options_2024_04.parquet", 15, 20, 0.37),
    ("T2", "2024-04-08", "outputs/cache/vix_option_chains/vix_options_2024_04.parquet", 15, 20, 0.63),
    ("T3", "2024-04-17", "outputs/cache/vix_option_chains/vix_options_2024_05.parquet", 15, 20, 1.38),
    ("T4", "2024-07-26", "outputs/cache/vix_option_chains/vix_options_2024_08.parquet", 15, 20, 0.95),
    ("T5", "2024-07-26", "outputs/cache/vix_option_chains/vix_options_2024_08.parquet", 14, 19, 1.22),
]

for label, exit_str, fpath_str, lk, sk, engine_exit in exit_checks:
    fpath = Path(fpath_str)
    chain = pd.read_parquet(fpath)
    chain.index = pd.to_datetime(chain.index)
    exit_ts = pd.Timestamp(exit_str)
    in_chain = exit_ts in chain.index
    if in_chain:
        row = chain.loc[exit_ts]
        if isinstance(row, pd.DataFrame): row = row.iloc[0]
        lb = row.get(f"C{lk}_Bid", np.nan)
        la = row.get(f"C{lk}_Ask", np.nan)
        sb = row.get(f"C{sk}_Bid", np.nan)
        sa = row.get(f"C{sk}_Ask", np.nan)
        lm = row.get(f"C{lk}_Mid", np.nan)
        sm = row.get(f"C{sk}_Mid", np.nan)
        real_exit_proceeds = (float(lb) - float(sa)) if not (pd.isna(lb) or pd.isna(sa)) else float("nan")
        real_mid = (float(lm) - float(sm)) if not (pd.isna(lm) or pd.isna(sm)) else float("nan")
        print(f"  {label}  {exit_str}  IN_CHAIN=YES  C{lk}bid={lb:.3f}/mid={float(lm):.3f}/ask={float(la):.3f}  "
              f"C{sk}bid={float(sb):.3f}/mid={float(sm):.3f}/ask={float(sa):.3f}  "
              f"exit_proceeds(bid-ask)={real_exit_proceeds:.3f}  "
              f"exit_mid={real_mid:.3f}  engine_exit={engine_exit:.3f}")
    else:
        print(f"  {label}  {exit_str}  IN_CHAIN=NO  (coverage: {chain.index[0].date()} to {chain.index[-1].date()})")

# ============================================================
# PART B: Corrected P&L for T1-T5 (real exit, check entry pricing)
# ============================================================
print()
print("=" * 90)
print("PART B: CORRECTED P&L USING REAL ENTRY PRICES WHERE AVAILABLE")
print("(exit prices are all real for T1-T5 per Part A above)")
print("=" * 90)

# Real entry mids (from audit 1):
real_entry_mids = {
    "T1": 1.030,   # SYNTHETIC (first real mid Feb 19)
    "T2": 1.030,   # REAL
    "T3": 1.005,   # REAL
    "T4": 0.880,   # SYNTHETIC (first real mid Jun 24)
    "T5": 1.160,   # SYNTHETIC (first real mid Jun 24, different strikes)
}

# Engine P&Ls (from v1.5 run):
engine_trades = [
    ("T1", "SYNTHETIC", 1.71, 0.37, -1.34),
    ("T2", "REAL",      1.06, 0.63, -0.43),
    ("T3", "REAL",      1.03, 1.38, +0.35),
    ("T4", "SYNTHETIC", 1.63, 0.95, -0.68),
    ("T5", "SYNTHETIC", 1.63, 1.22, -0.41),
]

# Need actual exit proceeds from chain (re-read since we need these net of commission)
exit_proceeds_real = {}
for label, exit_str, fpath_str, lk, sk, engine_exit in exit_checks:
    fpath = Path(fpath_str)
    chain = pd.read_parquet(fpath)
    chain.index = pd.to_datetime(chain.index)
    exit_ts = pd.Timestamp(exit_str)
    if exit_ts in chain.index:
        row = chain.loc[exit_ts]
        if isinstance(row, pd.DataFrame): row = row.iloc[0]
        lb = row.get(f"C{lk}_Bid", np.nan)
        sa = row.get(f"C{sk}_Ask", np.nan)
        proceeds = float(lb) - float(sa) if not (pd.isna(lb) or pd.isna(sa)) else float(engine_exit)
        exit_proceeds_real[label] = max(proceeds, 0.0) - COMM  # net of commission
    else:
        exit_proceeds_real[label] = engine_exit - COMM  # fallback

print(f"  {'Trade':5}  {'Entry_Type':12}  {'Eng_Entry':>10}  {'Real_Entry':>10}  "
      f"{'Real_EntCost':>12}  {'Exit_Proceeds':>13}  "
      f"{'Eng_PnL':>9}  {'Corrected_PnL':>14}  {'Delta':>8}")
print(f"  {'-'*105}")

for (label, entry_type, eng_entry, eng_exit, eng_pnl) in engine_trades:
    real_mid = real_entry_mids[label]
    real_cost = real_mid + COMM   # mid price + commission
    exit_net  = exit_proceeds_real[label]
    corrected_pnl = exit_net - real_cost
    delta = corrected_pnl - eng_pnl
    print(f"  {label:5}  {entry_type:12}  ${eng_entry:>8.3f}  ${real_mid:>8.3f}  "
          f"${real_cost:>10.3f}  ${exit_net:>11.3f}  "
          f"${eng_pnl:>+7.3f}  ${corrected_pnl:>+12.3f}  ${delta:>+6.3f}")

# ============================================================
# PART C: T7/T8 Dec 19 exit — Jan chain as proxy for Feb
# ============================================================
print()
print("=" * 90)
print("PART C: T7/T8 EXIT PROXY — Jan 2025 chain vs synthetic model (Dec 19)")
print("=" * 90)

jan_chain = pd.read_parquet("outputs/cache/vix_option_chains/vix_options_2025_01.parquet")
jan_chain.index = pd.to_datetime(jan_chain.index)

dec19_ts = pd.Timestamp("2024-12-19")
dec18_ts = pd.Timestamp("2024-12-18")
lk, sk = 16, 21

df_strat = pd.read_parquet("outputs/cache/vix_strategy_data.parquet")
df_strat.index = pd.to_datetime(df_strat.index)

print()
print("  Jan 2025 C16/C21 chain (real NBBO) around Dec 18-19:")
for ts in [dec18_ts, dec19_ts]:
    if ts in jan_chain.index:
        row = jan_chain.loc[ts]
        if isinstance(row, pd.DataFrame): row = row.iloc[0]
        jan_ux1 = float(df_strat.loc[ts, "UX1"]) if ts in df_strat.index else float("nan")
        jan_ux2 = float(df_strat.loc[ts, "UX2"]) if ts in df_strat.index else float("nan")
        jan_ux3 = float(df_strat.loc[ts, "UX3"]) if ts in df_strat.index else float("nan")
        vix_s   = float(df_strat.loc[ts, "VIX_Spot"]) if ts in df_strat.index else float("nan")
        lb = float(row.get(f"C{lk}_Bid", np.nan))
        lm = float(row.get(f"C{lk}_Mid", np.nan))
        la = float(row.get(f"C{lk}_Ask", np.nan))
        sb = float(row.get(f"C{sk}_Bid", np.nan))
        sm = float(row.get(f"C{sk}_Mid", np.nan))
        sa = float(row.get(f"C{sk}_Ask", np.nan))
        jan_spread_mid = lm - sm
        jan_spread_exit = lb - sa  # bid(long) - ask(short)
        jan_dte = (datetime.strptime("2025-01-22", "%Y-%m-%d").date() - ts.date()).days
        feb_dte = (datetime.strptime("2025-02-19", "%Y-%m-%d").date() - ts.date()).days

        # Jan intrinsic of C16/C21 with Jan futures (UX1)
        jan_intrinsic = max(jan_ux1 - lk, 0) - max(jan_ux1 - sk, 0)
        feb_intrinsic = max(jan_ux2 - lk, 0) - max(jan_ux2 - sk, 0)

        print(f"\n  {ts.date()}:")
        print(f"    VIX spot: {vix_s:.2f}  UX1 (Jan futures): {jan_ux1:.2f}  UX2 (Feb futures): {jan_ux2:.2f}  UX3: {jan_ux3:.2f}")
        print(f"    Jan C{lk}/C{sk} (Jan expiry, {jan_dte} DTE):")
        print(f"      C{lk}: bid={lb:.3f}  mid={lm:.3f}  ask={la:.3f}  intrinsic={max(jan_ux1-lk,0):.3f}")
        print(f"      C{sk}: bid={sb:.3f}  mid={sm:.3f}  ask={sa:.3f}  intrinsic={max(jan_ux1-sk,0):.3f}")
        print(f"      Spread mid = {jan_spread_mid:.3f}  |  exit proceeds (bid-ask) = {jan_spread_exit:.3f}")
        print(f"      vs Jan intrinsic = {jan_intrinsic:.3f}  => spread mid/intrinsic = {jan_spread_mid/max(jan_intrinsic,0.01):.1%}")
        print()
        print(f"    Feb C{lk}/C{sk} ENGINE estimate (Feb expiry, {feb_dte} DTE):")
        print(f"      Feb futures (UX2) = {jan_ux2:.2f}  Feb intrinsic = {feb_intrinsic:.3f}")
        print(f"      Engine synthetic exit = $4.86 (from synthetic model using UX2 intrinsic)")
        print(f"      Real Feb spread (estimated from Jan chain ratio):")
        if jan_intrinsic > 0:
            ratio = jan_spread_mid / jan_intrinsic
            print(f"        Jan spread/Jan_intrinsic = {ratio:.3f}")
            print(f"        If Feb spread / Feb_intrinsic ~ same ratio: {ratio * feb_intrinsic:.3f}")
            print(f"        But Feb has more DTE ({feb_dte} vs {jan_dte}), C{sk} OTM vs ITM — ratio would be WORSE (lower)")
        print(f"      Engine synthetic: $4.860  vs Jan-proxy estimate: ~${ratio*feb_intrinsic:.2f} (upper bound — Feb C{sk} is OTM, more expensive)")

# ============================================================
# PART D: Full corrected P&L scenario analysis
# ============================================================
print()
print("=" * 90)
print("PART D: CORRECTED TOTAL P&L — THREE SCENARIOS")
print("=" * 90)
print()

# From PARTS A & B, corrected P&Ls for T1-T5:
# T1: real exit $0.37 - COMM, real entry $1.03 + COMM = -$0.686
# T2: real (unchanged) = -$0.43
# T3: real (unchanged) = +$0.35
# T4: real exit ~$0.95 - COMM, real entry $0.88 + COMM = +$0.044
# T5: real exit ~$1.22 - COMM, real entry $1.16 + COMM = +$0.034
#
# Need to re-read actual exit proceeds from chain for T4/T5

t4_exit = exit_proceeds_real["T4"]  # bid(15) - ask(20) on Jul 26, net of comm
t5_exit = exit_proceeds_real["T5"]  # bid(14) - ask(19) on Jul 26, net of comm

# Corrected P&L for T1-T5 (using real entry costs where applicable)
corrected_t1_5 = {
    "T1": t4_exit,  # placeholder – need individual values
}

# Read individually
def get_exit_net(fpath_str, exit_str, lk, sk):
    fpath = Path(fpath_str)
    chain = pd.read_parquet(fpath)
    chain.index = pd.to_datetime(chain.index)
    ts = pd.Timestamp(exit_str)
    if ts not in chain.index: return None
    row = chain.loc[ts]
    if isinstance(row, pd.DataFrame): row = row.iloc[0]
    lb = float(row.get(f"C{lk}_Bid", np.nan))
    sa = float(row.get(f"C{sk}_Ask", np.nan))
    if pd.isna(lb) or pd.isna(sa): return None
    return max(lb - sa, 0) - COMM

t1_exit_net = get_exit_net("outputs/cache/vix_option_chains/vix_options_2024_04.parquet", "2024-03-25", 15, 20)
t4_exit_net = get_exit_net("outputs/cache/vix_option_chains/vix_options_2024_08.parquet", "2024-07-26", 15, 20)
t5_exit_net = get_exit_net("outputs/cache/vix_option_chains/vix_options_2024_08.parquet", "2024-07-26", 14, 19)

t1_corr = (t1_exit_net if t1_exit_net is not None else 0.37 - COMM) - (1.030 + COMM)
t2_corr = -0.43  # real both ways
t3_corr = +0.35  # real both ways
t4_corr = (t4_exit_net if t4_exit_net is not None else 0.95 - COMM) - (0.880 + COMM)
t5_corr = (t5_exit_net if t5_exit_net is not None else 1.22 - COMM) - (1.160 + COMM)

# T6/T7/T8: three scenarios
# Scenario A (OPTIMISTIC): trust engine exit pricing (intrinsic-based)
#   Correct only the entry to real NBBO
#   T6: +2.00 + (1.78 - 1.196) = +2.584  [engine exit $4.32 kept]
#   T7: +2.61 + (1.81 - 1.151) = +3.269  [engine exit $4.86 kept]
#   T8: +2.61 + (1.80 - 1.151) = +3.259  [engine exit $4.86 kept]

t6_A = 2.00 + (1.78 - (1.170 + COMM))
t7_A = 2.61 + (1.81 - (1.125 + COMM))
t8_A = 2.61 + (1.80 - (1.125 + COMM))

# Scenario B (CALIBRATED): use Jan chain ratio to estimate Feb C16/C21 exit
# Jan spread mid on Dec 19 = $2.55 with Jan intrinsic = $5.00 → ratio 51%
# Feb intrinsic on Dec 19 = $4.89 (UX2=20.89)
# Feb C21 is OTM (UX2=20.89 < 21) vs Jan C21 barely ITM → Feb C21 has MORE time value premium
# Conservative estimate: Feb spread = $1.80 (similar to Jan ratio but worse due to OTM short leg)
# T6 exit: similar reasoning, no adjacent chain, use $2.50 (rough: intrinsic 4.35, assume 57% ratio)
# Using bid-ask exit (not mid) for T7/T8 calibration
# Jan chain Dec 19: exit proceeds (bid-ask) = 5.55 - 3.275? wait, bid(C16)-ask(C21)
# Need to read that...

# Jan chain Dec 19 actual bid/ask values read above in Part C
# We'll compute below after reading
jan_dec19 = None
if dec19_ts in jan_chain.index:
    row = jan_chain.loc[dec19_ts]
    if isinstance(row, pd.DataFrame): row = row.iloc[0]
    lb = float(row.get(f"C{lk}_Bid", np.nan))
    sa = float(row.get(f"C{sk}_Ask", np.nan))
    jan_dec19_exit = max(lb - sa, 0) - COMM  # real exit proceeds for Jan expiry
    jan_dec19_mid  = float(row.get(f"C{lk}_Mid", np.nan)) - float(row.get(f"C{sk}_Mid", np.nan))
    jan_ux1_dec19  = float(df_strat.loc[dec19_ts, "UX1"])
    jan_ux2_dec19  = float(df_strat.loc[dec19_ts, "UX2"])

print(f"  Jan chain Dec 19 actual exit proceeds (bid16 - ask21 - comm): ${jan_dec19_exit:.3f}")
print(f"  Jan chain Dec 19 spread mid: ${jan_dec19_mid:.3f}")
print(f"  Jan futures (UX1): {jan_ux1_dec19:.2f}  Jan intrinsic: {max(jan_ux1_dec19-16,0)-max(jan_ux1_dec19-21,0):.3f}")
print(f"  Feb futures (UX2): {jan_ux2_dec19:.2f}  Feb intrinsic: {max(jan_ux2_dec19-16,0)-max(jan_ux2_dec19-21,0):.3f}")
print()
print(f"  Jan spread_exit_proceeds / Jan_intrinsic = {jan_dec19_exit / max(max(jan_ux1_dec19-16,0)-max(jan_ux1_dec19-21,0),0.01):.3f}")
print(f"  Feb spread ENGINE exit ($4.86) / Feb_intrinsic ({max(jan_ux2_dec19-16,0)-max(jan_ux2_dec19-21,0):.3f}) = {4.86 / max(max(jan_ux2_dec19-16,0)-max(jan_ux2_dec19-21,0),0.01):.3f}")
print()

jan_ratio_dec19 = jan_dec19_exit / max(max(jan_ux1_dec19-16,0)-max(jan_ux1_dec19-21,0), 0.01)
feb_intrinsic_dec19 = max(jan_ux2_dec19-16,0) - max(jan_ux2_dec19-21,0)

# For Feb spread, OTM short leg means ratio is WORSE than Jan ratio
# Jan C21 barely ITM (21.45) → Feb C21 OTM (20.89) → short leg MORE expensive
# Apply 80% of Jan ratio as estimate (conservative)
feb_exit_scenario_B = feb_intrinsic_dec19 * jan_ratio_dec19 * 0.80 - COMM

t6_B = 2.00 + (1.78 - (1.170 + COMM))  # no good reference; keep optimistic for T6
t7_B = feb_exit_scenario_B - (1.125 + COMM)
t8_B = feb_exit_scenario_B - (1.125 + COMM)
# T6 scenario B: use 57% of intrinsic (4.35) ~ $2.48, since Sep spike had LESS vol than Dec
t6_ux3_sep6 = 20.35
t6_intrinsic_exit = max(t6_ux3_sep6-16,0) - max(t6_ux3_sep6-21,0)
t6_exit_B = t6_intrinsic_exit * 0.57 - COMM  # 57% ratio estimate
t6_B = t6_exit_B - (1.170 + COMM)

# Scenario C (PESSIMISTIC): apply same Jan ratio to Feb spread
feb_exit_scenario_C = feb_intrinsic_dec19 * jan_ratio_dec19 - COMM
t6_exit_C = t6_intrinsic_exit * jan_ratio_dec19 - COMM  # use same ratio for Sep
t6_C = t6_exit_C - (1.170 + COMM)
t7_C = feb_exit_scenario_C - (1.125 + COMM)
t8_C = feb_exit_scenario_C - (1.125 + COMM)

total_t1_5 = t1_corr + t2_corr + t3_corr + t4_corr + t5_corr

print(f"  Scenario B Feb exit (80% × Jan ratio × Feb intrinsic - comm): ${feb_exit_scenario_B:.3f}")
print(f"  Scenario C Feb exit (100% × Jan ratio × Feb intrinsic - comm): ${feb_exit_scenario_C:.3f}")

print()
print(f"  {'Trade':5}  {'Engine PnL':>12}  {'Scenario A':>11}  {'Scenario B':>11}  {'Scenario C':>11}")
print(f"  {'-'*55}")
trades_data = [
    ("T1", -1.34, t1_corr, t1_corr, t1_corr),
    ("T2", -0.43, t2_corr, t2_corr, t2_corr),
    ("T3", +0.35, t3_corr, t3_corr, t3_corr),
    ("T4", -0.68, t4_corr, t4_corr, t4_corr),
    ("T5", -0.41, t5_corr, t5_corr, t5_corr),
    ("T6", +2.00, t6_A, t6_B, t6_C),
    ("T7", +2.61, t7_A, t7_B, t7_C),
    ("T8", +2.61, t8_A, t8_B, t8_C),
]

eng_total = 0; a_total = 0; b_total = 0; c_total = 0
for label, eng, a, b, c in trades_data:
    print(f"  {label:5}  ${eng:>+10.3f}  ${a:>+9.3f}  ${b:>+9.3f}  ${c:>+9.3f}")
    eng_total += eng; a_total += a; b_total += b; c_total += c

print(f"  {'-'*55}")
print(f"  {'TOTAL':5}  ${eng_total:>+10.3f}  ${a_total:>+9.3f}  ${b_total:>+9.3f}  ${c_total:>+9.3f}")

print()
print("=" * 90)
print("INTERPRETATION")
print("=" * 90)
print("""
  Scenario A (Optimistic): Correct ONLY entry prices to real NBBO; trust synthetic exits.
    Net effect: synthetic entry overpricing actually *helps* the losing trades (smaller losses)
    and *hurts* the winning trades (higher basis, smaller gains). But the main issue is the exits.

  Scenario B (Calibrated): Use Jan 2025 chain as proxy for Feb exit, scaled for OTM short leg.
    The Jan chain shows C16/C21 (Jan expiry) traded at ~50% of its intrinsic value on Dec 19.
    This is because the short C21 leg had massive implied vol premium (barely ITM Jan vs OTM Feb).
    For the Feb spread (OTM short leg), the compression is likely WORSE.

  Scenario C (Pessimistic): Apply Jan ratio directly to Feb, no further adjustment for OTM short.

  KEY INSIGHT — Why synthetic model fails at exit:
    The engine computes: exit_proceeds = intrinsic(VIX_futures) + small_time_value * 0.5
    The real market computes: exit_proceeds = bid(C16) - ask(C21)
    During VIX spikes, ask(C21) expands massively due to elevated implied vol on OTM short calls.
    This is the core vol-of-vol (VVIX) effect: as VIX spikes, OTM VIX call premiums inflate
    disproportionately vs intrinsic. The synthetic model completely ignores this.

  VERDICT:
    The +$4.71 reported result is NOT validated on real NBBO quotes.
    The 3 biggest winners (T6, T7, T8) have fully synthetic entry AND exit pricing.
    The Jan 2025 chain cross-check suggests the real exit proceeds were likely
    ~40-60% of what the synthetic model assumes.
    A realistic best-case P&L (Scenario B) is likely around ${b_total:.2f}.
    The true result could be negative.
""")

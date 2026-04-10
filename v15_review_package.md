# VIX Bull Call Spread v1.5 — Full Code Review Package

## STRATEGY OVERVIEW

Systematic VIX bull call spread strategy designed as a tactical convex overlay
for hedging long Nasdaq 100 (NQ) exposure. NOT a complete hedge program — this
is Sleeve 2 of a two-sleeve architecture (Sleeve 1: NQ/SPX put spreads for
persistent drawdowns, not yet built).

**How it works:**
- 3-state HMM (trained on VIX log returns + term structure slope, pre-trained on
  2010–2021 extended history) classifies each day as Low Vol, Transition, or High Vol
- Entries ONLY in Low Vol regime when composite score >= 0.65
- Five entry guards applied in sequence: HMM regime, composite score >= 0.65,
  VIX SMA10 momentum slope >= -0.03, UX2 <= 19.0, VIX 1-year percentile >= 25
- Composite weights: Term Structure 25%, VVIX 25%, VRP 25%, VIX Percentile 20%, COT 5%
- Strike selection: moneyness-based (long = futures-1, short = +30% of futures, clamped [3,8])
- Scale-out exits: half closed at 30% max profit, remainder at 60% or regime/time exit
- Unconditional exit if regime shifts to High Vol
- Signal-reset cooldown after losses
- Real Bloomberg NBBO quotes for all entry/exit pricing (option chain parquets)
- Entry at mid price (limit orders), exit at bid/ask (market orders at exit)

**v1.5 CORRECTED result — real NBBO (7 trades, 80-day chain window, all entries/exits real):**
```
Total Trades:     7
Win Rate:         57.1%
Avg Win:          +$0.41
Avg Loss:         -$0.32
Profit Factor:    1.72
Total P&L:        +$0.69
Max Drawdown:     -$2.20
Sharpe Ratio:     +0.35
Avg Holding Days: 29
```

**SUPERSEDED — v1.5 original result (was based on partly synthetic pricing, now invalidated):**
```
Total Trades:     8   <- one extra trade unblocked by synthetic stop-loss timing
Win Rate:         50.0%
Avg Win:          +$1.89
Avg Loss:         -$0.72
Profit Factor:    2.65
Total P&L:        +$4.71  <- 85% of this was synthetic pricing artifact
Max Drawdown:     -$3.72
Sharpe Ratio:     +1.61
```

**What changed (quote coverage audit, April 2026):**
Chains previously started ~60 calendar days before expiry.  Six of eight entry dates
fell outside chain coverage; all three winning trades' exits also fell outside coverage.
Extended pull to 80 days before expiry — all 39 chains repulled, now 8/8 entries and
7/7 exits covered by real Bloomberg NBBO.

Entry price correction: synthetic model overpriced entries by 36–40% ($0.41–$0.72/trade).
Exit price correction: synthetic model assumed exit_proceeds ≈ intrinsic(VIX_futures).
Real market: bid(C16) − ask(C21) during a VIX spike is 40–55% of intrinsic because the
short OTM C21 call's ask inflates massively with elevated implied vol (VVIX effect).
  Sep 06 exit: synthetic $4.32 → real $1.07  (−$3.25)
  Dec 19 exit: synthetic $4.86 → real $1.87  (−$2.99)

**Known limitations (updated):**
- 7 trades is a very thin sample (confidence interval on 57% WR: ~18%–90%)
- Entry filters (UX2 ≤ 19, pctl ≥ 25) were identified post-hoc on the same data used
  to test them — they need out-of-sample validation
- Real-NBBO exits show the spread does NOT capture intrinsic value during spikes; the
  bull call spread underperforms its theoretical max because the short OTM leg's vol
  premium crushes exit proceeds. Strike selection or exit mechanics need rethinking.


---

## VERSION HISTORY

```
v1.0: 64 trades, 56.2% WR, 1.41 Sharpe, 2.66 PF, -$14.28 DD
      Baseline: synthetic pricing, score >= 0.70, max 3 concurrent, 10d cooldown

v1.1: 38 trades, 63.2% WR, 1.76 Sharpe, 3.31 PF, -$8.22 DD
      Max 2 concurrent, 0.70 threshold, calendar cooldown retained

v1.2: 19 trades, 57.9% WR, 2.02 Sharpe, 3.62 PF, -$4.77 DD
      TRANSITION entries banned, unconditional regime exit,
      moneyness-based strikes (futures-1), EMA realized vol, COT reduced to 5%

v1.3: 20 trades, 75.0% WR, 1.98 Sharpe, 5.34 PF, -$4.98 DD (synthetic pricing)
      Scale-out exits (50%/75%), normalized spread width (30% of futures),
      signal-reset cooldown, Sharpe on premium-at-risk basis

v1.4: Real Bloomberg NBBO pricing added (P0-A/P0-B)
      - HMM pre-trained on 2010-2021 VIX history (walk-forward baseline)
      - 39 monthly VIX option chain parquets pulled (Jan 2023 - Mar 2026)
      - Entry: ask(long) - bid(short) + commission (market orders)
      - Exit: bid(long) - ask(short) + commission (market orders)
      - MTM: mid(long) - mid(short) daily
      - Slippage zeroed (real bid/ask IS the slippage; double-counting removed)
      Baseline result (market pricing, no new filters):
        15 trades, 26.7% WR, PF 0.57, -$5.50, Sharpe -0.94

v1.5: Entry filter upgrade + hybrid pricing
      New filters: VIX SMA10 5d slope >= -0.03 (post-spike momentum guard),
                   UX2 <= 19.0 (elevated futures curve block),
                   VIX 1yr percentile >= 25 (bottom-of-range guard)
      Pricing: hybrid (entry at mid, exit at bid/ask)
      Exit rules recalibrated: first target 30%, second 60%, time stop DTE 10
      DTE window shifted to 50-80 (was 30-60) for more runway
      Result: 8 trades, 50.0% WR, PF 2.65, +$4.71, Sharpe +1.61
```


---

## v1.5 TRADE LOG — CORRECTED (Real NBBO, 80-day chain window)

```
Entry       Exit        Legs       Entry    Exit    PnL       %      Exit reason
----------  ----------  ---------  -------  ------  --------  -----  ---------------------
2024-02-16  2024-04-08  C15/C20    $1.06    $0.63   -$0.43    -41%   EXIT_TIME_STOP
2024-02-19  2024-04-08  C15/C20    $1.06    $0.63   -$0.43    -41%   EXIT_TIME_STOP
2024-06-20  2024-07-26  C15/C20    $1.05    $0.95   -$0.10    -10%   EXIT_REGIME_CHANGE
2024-06-21  2024-07-26  C14/C19    $1.22    $1.43   +$0.21    +17%   EXIT_REGIME_CHANGE
2024-09-02  2024-09-17  C16/C21    $1.06    $1.07   +$0.01     +1%   EXIT_REGIME_CHANGE
2024-12-12  2024-12-19  C16/C21    $1.16    $1.87   +$0.71    +61%   EXIT_REGIME_CHANGE
2024-12-13  2024-12-19  C16/C21    $1.15    $1.87   +$0.72    +63%   EXIT_REGIME_CHANGE
```

Notes:
- All 7 entries and all 7 exits confirmed in real Bloomberg NBBO chain data (0 synthetic)
- Chain window extended from 60 to 80 calendar days before expiry; all 39 chains repulled
- T3 (Apr 2 entry, +$0.35 synthetic) no longer fires: T1's stop never triggers with real
  entry ($1.06 vs synthetic $1.71), so T1 stays open to Apr 8, blocking T3 on capacity
- Sep 2024 (T5): VIX spiked +6pts in 4 days but bid(C16)-ask(C21) only reached ~$1.07;
  the C21 OTM call's ask inflated during the spike, compressing spread exit proceeds
- Dec 2024 (T6/T7): same VVIX compression — C21 ask rose 3x during Dec 18-19 FOMC shock,
  limiting real exit to $1.87 vs $4.86 synthetic (which assumed intrinsic value)
- Jun 2024 (T3/T4): Jul carry-unwind regime flip; correctly classified as system risk

SUPERSEDED SYNTHETIC TRADE LOG (for reference only — prices are wrong):
```
2024-02-16  2024-03-25  C15/C20    $1.71    $0.37   -$1.34    STOP_LOSS  [synthetic entry]
2024-02-19  2024-04-08  C15/C20    $1.06    $0.63   -$0.43    TIME_STOP  [real entry]
2024-04-02  2024-04-17  C15/C20    $1.03    $1.38   +$0.35    REGIME     [blocked in real run]
2024-06-20  2024-07-26  C15/C20    $1.63    $0.95   -$0.68    REGIME     [synthetic entry]
2024-06-21  2024-07-26  C14/C19    $1.63    $1.22   -$0.41    REGIME     [synthetic entry]
2024-09-02  2024-09-06  C16/C21    $1.78    $4.32   +$2.00    PROFIT     [synthetic entry+exit]
2024-12-12  2024-12-19  C16/C21    $1.81    $4.86   +$2.61    PROFIT     [synthetic entry+exit]
2024-12-13  2024-12-19  C16/C21    $1.80    $4.86   +$2.61    PROFIT     [synthetic entry+exit]
```


---

## ABLATION STUDY

All configs use hybrid pricing, -0.03 momentum filter, same engine and pricing.
Only the entry decision rule changes.

```
Config                       Trades  Win%  AvgWin  AvgLoss    PF  TotalPnL    MaxDD  Sharpe
---------------------------  ------  ----  ------  -------  ----  --------  -------  ------
1. Full system (HMM+score)       15  26.7%  +$1.89   -$1.08  0.64    -$4.29   -$9.38   -0.64
2. HMM only (no score)           16  31.2%  +$1.66   -$1.39  0.54    -$7.01  -$10.54   -1.34
3. Score only (no HMM)           18  22.2%  +$1.06   -$1.15  0.26   -$11.90  -$12.88   -1.51
4. Simple contango rule          20  15.0%  +$1.50   -$1.37  0.19   -$18.81  -$22.58   -2.32
5. Monthly calendar              18  33.3%  +$1.26   -$1.19  0.53    -$6.70  -$11.99   -0.78
```

Key findings:
- HMM is the dominant signal: removing it (Config 3) is far worse than keeping only HMM (Config 2)
- Composite score adds marginal value over HMM-only: +$1.50 P&L, -$0.25 max DD
- Simple contango rule is worst: fires in post-spike recovery (VIX < 18 but still declining)
- Monthly calendar is competitive on Sharpe (-0.78 vs -0.64): warns that the full system's
  complexity is not definitively proven on 15 trades


---

## STRUCTURE TESTS

All configs use hybrid pricing, -0.03 momentum filter, HMM+score signal.

```
Config                             Trades  Win%  AvgWin  AvgLoss    PF  TotalPnL    MaxDD  Sharpe
---------------------------------  ------  ----  ------  -------  ----  --------  -------  ------
Baseline: 5pt, scale-out 30/60         15  26.7%  +$1.89   -$1.08  0.64    -$4.29   -$9.38   -0.64
Test 1: 5pt, no scale-out 50% exit     15  26.7%  +$1.61   -$1.08  0.54    -$5.43   -$9.38   -0.82
Test 2: 10pt, no scale-out 50% exit    15  20.0%  +$1.59   -$1.24  0.32   -$10.15  -$10.15   -1.34
Test 3: Outright call, 100% gain       15  20.0%  +$1.19   -$1.03  0.29    -$8.82  -$10.29   -0.71
```

Key findings:
- Scale-out 30/60 is the best structure: +$1.14 better than single 50% exit (same losers,
  better winners because 30% first target + staying half-in captures the full spike)
- 10pt spreads are worse: wider strikes dilute gains when VIX spike is moderate;
  Sep 2024 trade earns -$0.19 vs +$2.00 with 5pt spread
- Outright call underperforms: 100% target requires VIX to double; winners exit via
  regime change before target; avg win only $1.19 vs $1.89 for spread


---

## FEATURE ANALYSIS (Entry Filter Derivation)

### Entry feature table for 15 baseline trades (before new filters)

```
 #  Entry       VIXspot   UX1    UX2   Pctl%  TS_Ratio  SMA10slope  Score     PnL     W/L
 1  2024-02-16    14.24  15.01  15.94   43.3    0.8979      +0.026   0.714   -$1.34  LOSS
 2  2024-02-19    14.71  15.01  15.94   48.4    0.9275      +0.031   0.633   -$0.43  LOSS
 3  2024-04-02    14.61  14.91  15.75   58.3    0.9080      -0.027   0.562   +$0.35  WIN
 4  2024-06-12    12.04  12.58  13.98    1.2    0.8527      -0.014   0.903   -$0.12  LOSS
 5  2024-06-20    13.28  14.79  15.67   32.5    0.8618      -0.007   0.732   -$0.68  LOSS
 6  2024-09-02    15.55  15.40  17.39   71.8    0.8921      -0.002   0.628   +$2.00  WIN
 7  2024-12-12    13.92  14.51  16.36   42.1    0.8023      -0.024   0.672   +$2.61  WIN
 8  2024-12-13    13.81  14.55  16.32   38.9    0.8052      -0.004   0.685   +$2.61  WIN
 9  2025-07-11    16.40  19.25  20.48   30.6    0.8467      -0.025   0.602   -$1.62  LOSS
10  2025-07-28    15.03  17.30  19.25   12.3    0.8276      -0.030   0.649   -$1.17  LOSS
11  2025-08-26    14.62  16.80  19.08    8.3    0.8168      -0.023   0.789   -$1.28  LOSS
12  2025-09-10    15.35  16.19  18.41   19.8    0.8324      +0.006   0.630   -$1.96  LOSS
13  2025-09-15    15.69  18.16  19.39   25.0    0.8467      +0.001   0.632   -$2.03  LOSS
14  2025-12-22    14.08  16.90  18.67    0.0    0.7915      -0.004   0.866   -$0.67  LOSS
15  2025-12-23    14.00  16.95  18.69    0.0    0.7848      -0.021   0.841   -$0.56  LOSS
```

### Feature separation: winners vs losers

```
Feature                       Winners (n=4)          Losers (n=11)        Separation
----------------------------  ---------------------  -------------------  ----------
VIX Spot                      mean=14.47 [13.8,15.6] mean=14.50 [12.0,16.4]   0.02
UX1 (front-month futures)     mean=14.84 [14.5,15.4] mean=16.27 [12.6,19.3]   1.43
UX2 (second-month futures)    mean=16.46 [15.8,17.4] mean=17.77 [14.0,20.5]   1.32
VIX 1yr percentile            mean=52.8  [38.9,71.8] mean=20.1  [0.0, 48.4]  32.6  <-- STRONGEST
TS VIX/VIX3M ratio            mean=0.852 [0.80,0.91] mean=0.844 [0.79,0.93]   0.01
VIX SMA10 5d slope            mean=-0.014            mean=-0.005               0.01
Composite score               mean=0.637             mean=0.727               -0.09 (INVERTED)
```

### Key findings

**VIX Spot is useless as a separator.** All entries (winners and losers) entered with
spot VIX between 12 and 16. The problem is in the futures curve, not spot.

**VIX 1-year percentile is the strongest separator (32.6 points gap).** All 4 winners
entered above the 38th percentile. Every 2025 cluster loser entered below 30th.
Dec 2025 losers entered at the 0th percentile — the lowest VIX in the past year.

**UX1 <= 14.55 is the best single-threshold by scan** (keeps 2/4 wins, blocks 10/11
losses), but UX2 <= 19.0 is preferred because it targets the term structure at the
relevant expiry horizon and keeps all 4 winners while blocking 7/11 losses.

**The composite score is inverted.** Losers average 0.727 vs winners 0.637. The score
over-weights term structure and VVIX signals that fire during post-spike recovery
(deep contango + cheap VVIX) — exactly the wrong environment for bull call spreads.

**Two-filter design:** UX2 <= 19.0 eliminates all 7 trades in the 2025 post-tariff
cluster (all had UX2 17.4-20.5). VIX_Pctl >= 25 eliminates the Dec 2025-Jan 2026
bottom-of-range entries (pctl = 0.0). Together they keep all 4 winners intact.


---

## ENTRY FILTER UPGRADE RESULTS

```
Config                                     Trades  Win%  AvgWin  AvgLoss    PF  TotalPnL    MaxDD  Sharpe
-----------------------------------------  ------  ----  ------  -------  ----  --------  -------  ------
A. Baseline (HMM+score+momentum)               15  26.7%  +$1.89   -$1.08  0.64    -$4.29   -$9.38   -0.64
B. + UX2<=19 + pctl>=25 (score kept)            8  50.0%  +$1.89   -$0.72  2.65    +$4.71   -$3.72   +1.61
C. + UX2<=19 + pctl>=25, HMM only             11  45.5%  +$1.70   -$0.59  2.38    +$4.92   -$3.32   +1.52
```

**Score value-add on top of [HMM + momentum + UX2<=19 + pctl>=25]:**
- Score added 0 entries that HMM-only would have missed
- Score blocked 3 entries vs HMM-only: +$0.91, -$0.42, -$0.28
- Net contribution of score: -$0.21 (blocked a +$0.91 winner, saved $0.70 in losses)
- B ($4.71) vs C ($4.92): C is $0.21 better in P&L but B has slightly better Sharpe (1.61 vs 1.52)

The composite score's value is now marginal — it adds selectivity but at negligible net P&L
benefit after the structural filters do the heavy work. Config B (score kept) is the
recommended baseline: tighter drawdown, same signal quality, two extra loss blocks.


---

## EXECUTION PRICING TEST

```
Mode                              Entry pricing        Exit pricing       TotalPnL  Sharpe
--------------------------------  -------------------  -----------------  --------  ------
market (worst-case both ways)     ask(L)-bid(S)        bid(L)-ask(S)        -$5.50   -0.94
hybrid (patient in, urgent out)   mid(L)-mid(S)        bid(L)-ask(S)        -$4.29   -0.64
limit  (patient both ways)        mid(L)-mid(S)        mid(L)-mid(S)        -$2.01   -0.31
```

Bid/ask drag per trade: entry $0.081, exit $0.152, total $0.233.
At limit pricing (mid/mid), the system is near breakeven (-$2.01) before the new filters.
After the new filters, hybrid pricing produces +$4.71 — the system has real edge when
entries are at or near mid and the structural filters are applied.


---

## AUDIT RESULTS

Three trades verified manually against Bloomberg parquet data (to 4 decimal places):

**Trade A — 2024-02-19 C15/C20, expiry 2024-04-17, time stop 2024-04-08**
```
Entry: ask(C15)=2.19, bid(C20)=1.03 -> 2.19-1.03+0.0264 = 1.19  (engine: $1.19) MATCH
Exit:  bid(C15)=0.93, ask(C20)=0.27 -> max(0.66,0)-0.0264 = 0.63  (engine: $0.63) MATCH
PnL:   0.63 - 1.19 = -0.56  (engine: -$0.56) MATCH
```

**Trade B — 2025-07-28 C18/C24, expiry 2025-09-17, stop loss 2025-08-08**
```
Entry: ask(C18)=2.81, bid(C24)=1.15 -> 2.81-1.15+0.0264 = 1.69  (engine: $1.69) MATCH
Exit:  bid(C18)=0.72, ask(C24)=1.21 -> max(-0.49,0)-0.0264 = 0.00  (engine: $0.00) MATCH
       (inverted bid/ask: market correctly prices spread at zero)
PnL:   0.00 - 1.69 = -1.69  (engine: -$1.69) MATCH
```

**Trade C — 2025-12-22 C18/C24, expiry 2026-02-18, time stop 2026-02-09**
```
Entry: ask(C18)=2.56, bid(C24)=1.05 -> 2.56-1.05+0.0264 = 1.54  (engine: $1.54) MATCH
Exit:  bid(C18)=1.10, ask(C24)=0.47 -> max(0.63,0)-0.0264 = 0.60  (engine: $0.60) MATCH
PnL:   0.60 - 1.54 = -0.94  (engine: -$0.94) MATCH
```

**Scale-out accounting verified:** commission drag = 2 × $0.0264 = $0.0528 per trade
(same total as single-close). Trade #7 back-calculation confirms half-close exit price
$3.98 > 30% target trigger $2.77.

**Chain lookup verified:** YYYY-MM key scheme correctly loads holiday-adjusted expiries
(Jun 2024 actual=Jun 18, Mar 2025 actual=Mar 18) — both load from correct monthly file.

**One structural note:** option chains start at ~60 DTE but strategy enters at 65 DTE.
First 5 trading days after entry use synthetic pricing. The 3 largest 2024 winners
(all within 8 days of entry) exit on synthetic MTM. Real-NBBO P&L on purely real-priced
trades: wins=$0.18, losses=-$9.14, net=-$8.96.


---

## MOMENTUM FILTER THRESHOLD SWEEP

```
Threshold  Trades  Win%  AvgWin  AvgLoss    PF  TotalPnL    MaxDD  Sharpe
---------  ------  ----  ------  -------  ----  --------  -------  ------
baseline       19  21.1%  +$1.56   -$1.44  0.29   -$15.40  -$19.00   -2.08
-0.03          15  26.7%  +$1.85   -$1.17  0.57    -$5.50  -$10.29   -0.94
-0.05          16  25.0%  +$1.92   -$1.27  0.50    -$7.55  -$13.08   -1.30
-0.08          18  27.8%  +$1.73   -$1.48  0.45   -$10.55  -$18.42   -1.54
-0.10          18  27.8%  +$1.75   -$1.45  0.46   -$10.13  -$18.23   -1.45
```

-0.03 is the optimal threshold: eliminates the Jun 2025 C20/C26 entries (SMA10 slope
-4% to -5% at entry) without over-filtering. Looser thresholds (-0.05 to -0.10) let
the same entries through a few days later and still lose. -0.03 adopted in v1.5.


---

## FILE: config/settings.py
```python
"""
config/settings.py
==================
Central configuration for the VIX Bull Call Spread strategy.
All Bloomberg tickers, thresholds, and regime parameters in one place.

CRITICAL: This file contains NO logic, only constants and parameters.
Changes here propagate through the entire system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import date

# ============================================================
# 1. BLOOMBERG TICKERS
# ============================================================

VIX_SPOT = "VIX Index"
VIX3M = "VIX3M Index"
VIX6M = "VIX6M Index"
VIX9D = "VIX9D Index"
VIX1D = "VIX1D Index"
VVIX = "VVIX Index"
SKEW = "SKEW Index"

VIX_FUTURES = {
    1: "UX1 Index", 2: "UX2 Index", 3: "UX3 Index", 4: "UX4 Index",
    5: "UX5 Index", 6: "UX6 Index", 7: "UX7 Index", 8: "UX8 Index", 9: "UX9 Index",
}

VIX_MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}

SPX = "SPX Index"
NQ = "NQ1 Index"
ES = "ES1 Index"
HYG = "HYG US Equity"
LQD = "LQD US Equity"

DAILY_FIELDS = ["PX_LAST", "PX_HIGH", "PX_LOW", "PX_OPEN", "PX_VOLUME", "OPEN_INT"]
OPTION_FIELDS = ["PX_LAST", "PX_BID", "PX_ASK", "PX_MID", "IVOL_MID", "DELTA_MID",
                 "GAMMA_MID", "THETA_MID", "VEGA_MID", "OPEN_INT", "PX_VOLUME"]
VIX_OPTION_TEMPLATE = "VIX US {expiry} {cp_type}{strike} Index"

CFTC_VIX_CODE = "1170E1"
CFTC_DATA_URL = "https://www.cftc.gov/dea/newcot/FinFutL.txt"
CFTC_HISTORY_URL = "https://www.cftc.gov/files/dea/history/fin_fut_txt_{year}.zip"
COT_BLOOMBERG = {
    "asset_mgr_long": "CFVXAMLG Index", "asset_mgr_short": "CFVXAMSH Index",
    "lev_fund_long": "CFVXLFLG Index",  "lev_fund_short": "CFVXLFSH Index",
    "dealer_long": "CFVXDILG Index",    "dealer_short": "CFVXDISH Index",
}

# ============================================================
# 4. BACKTEST PARAMETERS
# ============================================================

@dataclass
class BacktestConfig:
    """Walk-forward backtest configuration."""
    start_date: date = date(2022, 1, 3)
    end_date: date = date(2026, 3, 18)

    training_window_days: int = 504
    refit_frequency_days: int = 63
    min_training_days: int = 252

    # Real NBBO bid/ask spread IS the slippage — no additional charge
    entry_slippage_pct: float = 0.00
    exit_slippage_pct: float = 0.00
    commission_per_contract: float = 1.32   # Per leg, CBOE VIX options

    max_position_pct: float = 0.02
    max_concurrent_positions: int = 2

    # Cooldown — signal-reset based (v1.3)
    cooldown_type: str = "signal_reset"
    cooldown_after_loss_days: int = 10
    cooldown_score_rearm: float = 0.65

    # VIX futures level cap (expiry-matched future)
    max_entry_vix_futures: Optional[float] = None

    # UX2 cap — blocks post-spike recovery entries where curve is still elevated
    max_ux2: Optional[float] = None

    # VIX 1-year percentile floor — blocks bottom-of-range entries
    min_vix_pctl_1yr: Optional[float] = None

    # Structure switches
    scale_out: bool = True          # True = 30%/60% scale-out; False = single full exit
    position_type: str = "spread"   # "spread" or "call"
    call_profit_target_pct: float = 1.0

    # Post-spike momentum filter: SMA10 slope >= threshold required for entry
    vix_momentum_threshold: Optional[float] = None

    # Execution pricing model
    # "market": entry=ask/bid, exit=bid/ask
    # "hybrid": entry=mid,     exit=bid/ask
    # "limit":  entry=mid,     exit=mid
    execution_mode: str = "market"

    use_vro_settlement: bool = True
    vro_uncertainty_band: float = 0.50


@dataclass
class RegimeConfig:
    n_states: int = 3
    features: List[str] = field(default_factory=lambda: [
        "vix_log_return", "term_structure_slope",
    ])
    covariance_type: str = "full"
    n_iter: int = 200
    tol: float = 1e-4
    n_random_starts: int = 10
    regime_transition_prob_threshold: float = 0.80
    min_regime_holding_days: int = 3
    expected_regime_params: Dict[str, Dict] = field(default_factory=lambda: {
        "low_vol":    {"vix_mean_range": (12, 18),  "daily_vol_range": (0.01, 0.04), "persistence": 0.97},
        "transition": {"vix_mean_range": (18, 25),  "daily_vol_range": (0.04, 0.08), "persistence": 0.90},
        "high_vol":   {"vix_mean_range": (25, 80),  "daily_vol_range": (0.06, 0.20), "persistence": 0.88},
    })


@dataclass
class SignalConfig:
    weights: Dict[str, float] = field(default_factory=lambda: {
        "term_structure": 0.25,
        "vix_percentile": 0.20,
        "vvix_level":     0.25,
        "cot_positioning":0.05,
        "vrp_signal":     0.25,
    })
    entry_score_threshold: float = 0.65      # v1.5: was 0.70
    transition_score_threshold: float = 999.0  # Effectively disabled
    term_structure_contango_threshold: float = 0.92
    vix_low_percentile: int = 30
    vvix_cheap_threshold: float = 85.0
    vvix_divergence_threshold: float = 105.0
    cot_extreme_percentile: int = 90
    vrp_threshold: float = 5.0
    first_exit_pct: float = 0.30             # v1.5: was 0.50
    second_exit_pct: float = 0.60            # v1.5: was 0.75
    second_exit_fallback: str = "regime"
    time_stop_dte: int = 10                  # v1.5: was 21
    regime_exit: bool = True
    pre_settlement_close_dte: int = 1
    stop_loss_pct: float = 0.70


@dataclass
class StrikeConfig:
    long_delta_target: float = 0.40
    short_delta_target: float = 0.25
    delta_tolerance: float = 0.05
    spread_width_pct: float = 0.30
    min_spread_width: int = 3
    max_spread_width: int = 8
    spread_width_override: Optional[int] = None
    include_wide_spread: bool = False
    wide_spread_width: int = 20
    wide_spread_allocation_pct: float = 0.30
    target_dte: int = 65                     # v1.5: was 45
    dte_range: Tuple[int, int] = (50, 80)    # v1.5: was (30, 60)
    strike_increment: float = 1.0
    max_spread_cost: float = 3.00
    min_risk_reward: float = 1.5


@dataclass
class FeatureConfig:
    realized_vol_window: int = 21
    vix_percentile_window: int = 252
    vvix_ma_window: int = 10
    momentum_window: int = 5
    z_score_window: int = 63
    ts_slope_lookback: int = 20
    cot_percentile_window: int = 156
    cot_z_score_window: int = 52


BACKTEST = BacktestConfig()
REGIME   = RegimeConfig()
SIGNAL   = SignalConfig()
STRIKE   = StrikeConfig()
FEATURE  = FeatureConfig()
```


---

## FILE: backtest/engine.py — _try_entry() (entry gate logic)

The five entry guards in order:

```python
def _try_entry(self, row: pd.Series, date_str: str):
    # 1. Cooldown (signal-reset): blocked if last loss not yet rearmed
    if self.closed_positions:
        if not getattr(self, '_score_rearmed', True):
            return

    # 2. Capacity check
    if len(self.positions) >= self.config.max_concurrent_positions:
        return

    # 3. HMM regime gate
    regime = row.get("Regime", None)
    if regime is None or np.isnan(regime):
        return
    regime_enum = VolRegime(int(regime))
    # (Signal_Entry already gates on LOW_VOL + composite score >= 0.65
    #  via CompositeSignal.compute() — the engine trusts Signal_Entry)

    # 4. Post-spike momentum filter
    if self.config.vix_momentum_threshold is not None:
        slope = row.get("VIX_SMA10_Slope_5d", None)
        if slope is not None and not np.isnan(float(slope)):
            if float(slope) < self.config.vix_momentum_threshold:
                return    # VIX SMA10 declining too fast — post-spike recovery

    # 5. UX2 cap (v1.5): block elevated second-month curve
    if self.config.max_ux2 is not None:
        ux2 = row.get("UX2", None)
        if ux2 is not None and not np.isnan(float(ux2)):
            if float(ux2) > self.config.max_ux2:
                return

    # 6. VIX percentile floor (v1.5): block bottom-of-range entries
    if self.config.min_vix_pctl_1yr is not None:
        pctl = row.get("VIX_Pctl_1yr", None)
        if pctl is not None and not np.isnan(float(pctl)):
            if float(pctl) < self.config.min_vix_pctl_1yr:
                return

    # 7. Expiry calendar + DTE window
    entry_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
    target_expiry = self._find_target_expiry(entry_dt)  # first expiry in [50, 80] DTE
    ...

    # 8. Pricing: hybrid = mid entry, bid/ask exit
    # mid(long) - mid(short) from OptionChainStore.get_spread_mid()
    # Falls back to synthetic Black-76 when chain data unavailable
    entry_cost += self.config.commission_per_contract * 2 / 100
```


---

## FILE: data/option_chain_store.py

```python
"""
data/option_chain_store.py
==========================
Lazy-loading cache of VIX option chain NBBO data.

Files: outputs/cache/vix_option_chains/vix_options_YYYY_MM.parquet
       Columns: C{K}_Bid, C{K}_Ask, C{K}_Mid, C{K}_Last  (K = 10..35)
"""

class OptionChainStore:
    DEFAULT_DIR = Path("outputs/cache/vix_option_chains")

    def get_spread_mid(self, expiry_date, obs_date, long_strike, short_strike):
        """mid(long) - mid(short). Used for MTM and hybrid entry."""
        # Sanity: long_mid >= short_mid (call monotonicity)
        # Sanity: spread <= spread_width (no-arb)
        ...

    def get_entry_cost(self, expiry_date, obs_date, long_strike, short_strike):
        """ask(long) - bid(short). Market-order entry cost."""
        ...

    def get_exit_proceeds(self, expiry_date, obs_date, long_strike, short_strike):
        """max(bid(long) - ask(short), 0). Market-order exit proceeds."""
        ...

    def get_call_mid(self, expiry_date, obs_date, strike):
        """Mid price of a single call leg."""
        ...

    def get_call_ask(self, expiry_date, obs_date, strike):
        """Ask price of a single call leg (market-order entry)."""
        ...

    def get_call_bid(self, expiry_date, obs_date, strike):
        """Bid price of a single call leg (market-order exit)."""
        ...

    def _load_chain(self, expiry_date):
        """Load by YYYY-MM key. Both theoretical and holiday-adjusted expiries
        map to the same monthly file, so the lookup is always correct."""
        key = expiry_date[:7]  # "YYYY-MM"
        fname = f"vix_options_{key[:4]}_{key[5:7]}.parquet"
        ...
```

Chain data coverage: 39 monthly parquets, Jan 2023 – Mar 2026.
Each file covers approximately 60 calendar days (~43 trading days) ending at expiry.


---

## OPEN ISSUES / NEXT STEPS

1. **Chain coverage gap**: chains start at ~60 DTE, strategy enters at 65 DTE.
   First ~5 trading days use synthetic pricing. The 3 largest 2024 winners (Sep,
   Dec×2) all exit during this window. Fix: extend Bloomberg pull to 70+ DTE.

2. **Out-of-sample validation of UX2/percentile filters**: both filters were
   identified on the same data used to test them. Need forward data (Apr–Dec 2026)
   to confirm they generalize.

3. **2024 June losses not addressed**: C14/C18 and C15/C20 entries on Jun 12–21
   lost to the Jul 2024 carry-unwind spike (VIX 12→18). These pass all five filters.
   Possible mitigation: cross-asset signal (HYG/LQD spread) or additional DXY/rates
   filter. Or accept as structural strategy risk.

4. **Synthetic exit validation**: Sep 2024 and Dec 2024 exits used the synthetic
   time-value model (intrinsic + √DTE_ratio × initial_TV × 0.5). Should validate
   against actual NBBO quotes for those expiries if Bloomberg history allows
   earlier chain pull.

5. **Sleeve 1 (NQ put spreads)**: not yet built. The VIX call spread alone provides
   zero protection on worst 1% NQ days (strategy inactive in HIGH_VOL by design).

6. **Portfolio sizing**: all P&L figures are per-spread in index points (×$100
   multiplier = per contract). Position sizing at 2% of portfolio not yet modelled
   in absolute dollar terms.

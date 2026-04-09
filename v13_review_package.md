# VIX Bull Call Spread v1.3 — Full Code for Review

## STRATEGY OVERVIEW

Systematic VIX bull call spread strategy designed as a tactical convex overlay 
for hedging long Nasdaq 100 (NQ) exposure. NOT a complete hedge program — this 
is Sleeve 2 of a two-sleeve architecture (Sleeve 1: NQ/SPX put spreads for 
persistent drawdowns, not yet built).

How it works:
- 3-state HMM (trained on VIX log returns + term structure slope) classifies 
  each day as Low Vol, Transition, or High Vol
- Entries ONLY in Low Vol regime when composite score >= 0.70
- Composite: Term Structure 25%, VVIX 25%, VRP 25%, VIX Percentile 20%, COT 5%
- Strike selection: moneyness-based (long = futures-1, short = +30% of futures width, clamped [3,8])
- Scale-out exits: half closed at 50% max profit, remainder at 75% or regime/time exit
- Unconditional exit if regime shifts to High Vol
- Signal-reset cooldown after losses (entry signal must turn off then re-arm)
- Backtest uses estimated spread prices, NOT historical VIX option quotes

Known limitations:
- 20 trades is statistically thin (95% CI on 75% WR: ~53%-89%)
- All parameters are in-sample optimized (v1.0→v1.3)
- Synthetic pricing, no actual bid/ask history
- $0 hedge contribution on worst 1% NQ days (strategy inactive in HIGH_VOL)
- 0% portfolio drawdown reduction at current sizing (proof-of-concept scale)

## BACKTEST RESULTS (v1.3)
```
Total Trades:     20
Win Rate:         75.0%
Avg Win:          +$1.48
Avg Loss:         $-0.83
Profit Factor:    5.34
Total P&L:        $18.00
Max Drawdown:     $-4.98
Sharpe Ratio:     1.98
Avg Holding Days: 17

Trade Log:
  #1  2023-12-08→2023-12-20 C12/C16 +$1.45 EXIT_PROFIT_TARGET
  #2  2023-12-11→2023-12-20 C12/C16 +$1.61 EXIT_PROFIT_TARGET
  #3  2023-12-26→2024-01-19 C14/C18 -$0.65 EXIT_TIME_STOP
  #4  2023-12-27→2024-01-22 C13/C17 -$0.39 EXIT_TIME_STOP
  #5  2024-06-07→2024-07-01 C12/C16 +$0.48 EXIT_TIME_STOP [scaled-out]
  #6  2024-06-10→2024-07-04 C12/C16 +$0.43 EXIT_TIME_STOP [scaled-out]
  #7  2024-07-05→2024-07-19 C12/C16 +$1.76 EXIT_PROFIT_TARGET
  #8  2024-07-08→2024-07-19 C12/C16 +$1.84 EXIT_PROFIT_TARGET
  #9  2024-08-15→2024-09-04 C15/C20 +$2.33 EXIT_PROFIT_TARGET
  #10 2024-08-16→2024-09-04 C15/C20 +$2.33 EXIT_PROFIT_TARGET
  #11 2024-12-02→2024-12-18 C14/C18 +$1.82 EXIT_PROFIT_TARGET
  #12 2024-12-03→2024-12-18 C14/C18 +$1.90 EXIT_PROFIT_TARGET
  #13 2025-06-09→2025-06-16 C17/C23 +$1.54 EXIT_REGIME_CHANGE
  #14 2025-06-10→2025-06-16 C17/C22 +$1.82 EXIT_REGIME_CHANGE
  #15 2025-08-08→2025-09-01 C16/C21 -$0.53 EXIT_TIME_STOP
  #16 2025-08-12→2025-09-08 C15/C20 +$0.57 EXIT_TIME_STOP [scaled-out]
  #17 2025-10-27→2025-11-04 C17/C22 +$0.73 EXIT_REGIME_CHANGE
  #18 2025-11-28→2025-12-22 C17/C22 -$1.32 EXIT_TIME_STOP
  #19 2025-12-03→2025-12-29 C17/C22 -$1.26 EXIT_TIME_STOP
  #20 2026-01-09→2026-01-19 C15/C20 +$1.54 EXIT_REGIME_CHANGE
```


## HEDGE EFFECTIVENESS
```
Worst 1% NQ days:  +$0.00 (strategy inactive)
Worst 5% NQ days:  +$3.72
Worst 10% NQ days: +$4.45
All NQ down days:  +$15.56
All NQ up days:    +$2.44
Down-day correlation (active): -0.326
NQ-only max drawdown:  -$11,534K
NQ+Hedge max drawdown: -$11,534K (0% reduction at current sizing)
```


## VERSION HISTORY
```
v1.0: 64 trades, 56.2% WR, 1.41 Sharpe, 2.66 PF, -$14.28 DD (baseline)
v1.1: 38 trades, 63.2% WR, 1.76 Sharpe, 3.31 PF, -$8.22 DD (tuned: max 2 concurrent, 0.70 threshold, 10d cooldown)
v1.2: 19 trades, 57.9% WR, 2.02 Sharpe, 3.62 PF, -$4.77 DD (banned TRANSITION, unconditional regime exit, moneyness strikes, EMA vol, reduced COT to 5%)
v1.3: 20 trades, 75.0% WR, 1.98 Sharpe, 5.34 PF, -$4.98 DD (scale-out exits, normalized spread width, signal-reset cooldown, Sharpe on premium-at-risk)
```


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
from typing import Dict, List, Tuple
from datetime import date

# ============================================================
# 1. BLOOMBERG TICKERS
# ============================================================

# VIX Spot & Term Structure
VIX_SPOT = "VIX Index"
VIX3M = "VIX3M Index"          # 3-month VIX (formerly VXV)
VIX6M = "VIX6M Index"          # 6-month VIX
VIX9D = "VIX9D Index"          # 9-day VIX
VIX1D = "VIX1D Index"          # 1-day VIX
VVIX = "VVIX Index"            # Volatility of VIX
SKEW = "SKEW Index"            # S&P 500 Skew Index

# VIX Generic Futures (rolling contracts)
VIX_FUTURES = {
    1: "UX1 Index",    # Front month
    2: "UX2 Index",    # Second month
    3: "UX3 Index",    # Third month
    4: "UX4 Index",    # Fourth month
    5: "UX5 Index",    # Fifth month
    6: "UX6 Index",    # Sixth month
    7: "UX7 Index",    # Seventh month
    8: "UX8 Index",    # Eighth month
    9: "UX9 Index",    # Ninth month (for deep curve analysis)
}

# Specific VIX Futures Contracts (for backtest settlement)
# Format: UX + month_code + year_digits + " Index"
# Month codes: F=Jan G=Feb H=Mar J=Apr K=May M=Jun N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec
VIX_MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}

# Equity Benchmarks (for hedge effectiveness measurement)
SPX = "SPX Index"
NQ = "NQ1 Index"               # Nasdaq 100 E-mini front month
ES = "ES1 Index"               # S&P 500 E-mini front month

# Credit (cross-asset regime confirmation)
HYG = "HYG US Equity"          # High Yield Corporate Bond ETF
LQD = "LQD US Equity"          # Investment Grade Corporate Bond ETF

# ============================================================
# 2. BLOOMBERG FIELDS
# ============================================================

# Daily fields for historical data
DAILY_FIELDS = [
    "PX_LAST",        # Close price
    "PX_HIGH",        # Intraday high
    "PX_LOW",         # Intraday low
    "PX_OPEN",        # Open price
    "PX_VOLUME",      # Volume
    "OPEN_INT",        # Open interest (futures)
]

# Option-specific fields
OPTION_FIELDS = [
    "PX_LAST",
    "PX_BID",
    "PX_ASK",
    "PX_MID",
    "IVOL_MID",        # Mid implied volatility
    "DELTA_MID",       # Mid delta
    "GAMMA_MID",       # Mid gamma
    "THETA_MID",       # Mid theta
    "VEGA_MID",        # Mid vega
    "OPEN_INT",
    "PX_VOLUME",
]

# VIX Option ticker format for Bloomberg
# Example: "VIX US 03/18/26 C20 Index"
VIX_OPTION_TEMPLATE = "VIX US {expiry} {cp_type}{strike} Index"

# ============================================================
# 3. CFTC COT CONFIGURATION
# ============================================================

# CFTC COT data for VIX futures
CFTC_VIX_CODE = "1170E1"       # CBOE VIX Futures
CFTC_DATA_URL = "https://www.cftc.gov/dea/newcot/FinFutL.txt"  # TFF Futures Only
CFTC_HISTORY_URL = "https://www.cftc.gov/files/dea/history/fin_fut_txt_{year}.zip"

# Bloomberg COT tickers (if available on your terminal)
# Format: CFTC_{commodity_code}_{category}_{metric}
COT_BLOOMBERG = {
    "asset_mgr_long": "CFVXAMLG Index",   # Asset Manager Long
    "asset_mgr_short": "CFVXAMSH Index",   # Asset Manager Short
    "lev_fund_long": "CFVXLFLG Index",     # Leveraged Fund Long
    "lev_fund_short": "CFVXLFSH Index",    # Leveraged Fund Short
    "dealer_long": "CFVXDILG Index",       # Dealer Long
    "dealer_short": "CFVXDISH Index",      # Dealer Short
}

# ============================================================
# 4. BACKTEST PARAMETERS
# ============================================================

@dataclass
class BacktestConfig:
    """Walk-forward backtest configuration."""
    start_date: date = date(2022, 1, 3)    # First trading day 2022
    end_date: date = date(2026, 3, 18)     # Mar 2026 VIX expiry
    
    # Walk-forward
    training_window_days: int = 504         # ~2 years of trading days for HMM training
    refit_frequency_days: int = 63          # Refit HMM quarterly (~63 trading days)
    min_training_days: int = 252            # Minimum 1 year before first signal
    
    # Execution assumptions
    entry_slippage_pct: float = 0.05        # 5% of spread price
    exit_slippage_pct: float = 0.05
    commission_per_contract: float = 1.32   # Per leg, per contract (CBOE VIX options)
    
    # Position sizing
    max_position_pct: float = 0.02          # Max 2% of portfolio per spread
    # max_concurrent_positions: int = 3       # Max 3 open spreads at once
    max_concurrent_positions: int = 2       # Was 3 — reduces correlated drawdowns
    
    # Cooldown — signal-reset based (v1.3)
    cooldown_type: str = "signal_reset"     # "calendar" or "signal_reset"
    cooldown_after_loss_days: int = 10      # Only used if cooldown_type = "calendar"
    cooldown_score_rearm: float = 0.65      # Was 0.50 — too strict, score rarely drops that low in LOW_VOL

    # Settlement
    use_vro_settlement: bool = True         # Use VRO when available, else VIX futures proxy
    vro_uncertainty_band: float = 0.50      # ±$0.50 for stress testing narrow spreads

# ============================================================
# 5. REGIME CLASSIFIER PARAMETERS
# ============================================================

@dataclass
class RegimeConfig:
    """3-state HMM regime classifier configuration."""
    n_states: int = 3                       # Low vol, Transition, High vol
    
    # HMM observation features
    features: List[str] = field(default_factory=lambda: [
        "vix_log_return",                   # Primary: VIX daily log changes
        "term_structure_slope",             # Secondary: VIX/VIX3M ratio
    ])
    
    # Covariance type
    covariance_type: str = "full"           # "full", "diag", "tied", "spherical"
    
    # Training
    n_iter: int = 200                       # Max EM iterations
    tol: float = 1e-4                       # Convergence tolerance
    n_random_starts: int = 10               # Multiple initializations to avoid local optima
    
    # Real-time filtering thresholds
    regime_transition_prob_threshold: float = 0.80   # Min posterior prob to declare regime change
    min_regime_holding_days: int = 3                 # Debounce: stay in regime for 3 days
    
    # Expected regime characteristics (for label assignment post-training)
    # Based on Hardy (2001) and Li (2016) calibrations
    expected_regime_params: Dict[str, Dict] = field(default_factory=lambda: {
        "low_vol": {
            "vix_mean_range": (12, 18),     # VIX level when in this regime
            "daily_vol_range": (0.01, 0.04),# Daily VIX volatility
            "persistence": 0.97,            # Self-transition probability
        },
        "transition": {
            "vix_mean_range": (18, 25),
            "daily_vol_range": (0.04, 0.08),
            "persistence": 0.90,
        },
        "high_vol": {
            "vix_mean_range": (25, 80),
            "daily_vol_range": (0.06, 0.20),
            "persistence": 0.88,
        },
    })

# ============================================================
# 6. SIGNAL COMPOSITE PARAMETERS
# ============================================================

@dataclass
class SignalConfig:
    """Composite entry/exit signal configuration."""
    
    # Signal weights (must sum to 1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "term_structure": 0.25,     # Was 0.30 — reduced to avoid double-counting with HMM
        "vix_percentile": 0.20,     
        "vvix_level": 0.25,         # Was 0.20 — increased, strongest independent signal
        "cot_positioning": 0.05,    # Was 0.15 — weekly lagged data, reduced to backdrop
        "vrp_signal": 0.25,         # Was 0.15 — increased to balance
    })
    
    # Entry thresholds — regime-dependent
    entry_score_threshold: float = 0.70          # LOW_VOL entry threshold
    transition_score_threshold: float = 999.0    # Effectively disabled — LOW_VOL only
    
    # Individual indicator thresholds
    term_structure_contango_threshold: float = 0.92   # VIX/VIX3M < 0.92 = strong contango
    vix_low_percentile: int = 30                      # VIX below 30th percentile of 1yr = cheap
    vvix_cheap_threshold: float = 85.0                # VVIX < 85 = cheap VIX options
    vvix_divergence_threshold: float = 105.0          # High VVIX + low VIX = institutional tell
    cot_extreme_percentile: int = 90                  # Net short above 90th pctl = extreme
    vrp_threshold: float = 5.0                        # VIX - RV > 5 = large headwind
    
    # Exit rules — scale-out (NEW in v1.3)
    first_exit_pct: float = 0.50            # Close HALF at 50% of max profit
    second_exit_pct: float = 0.75           # Close remainder at 75% of max profit
    second_exit_fallback: str = "regime"    # If 75% not hit, exit on regime change or time stop
    time_stop_dte: int = 21                 # Close remaining if DTE <= 21 and at a loss
    regime_exit: bool = True                # Close remaining if HIGH_VOL detected
    pre_settlement_close_dte: int = 1
    stop_loss_pct: float = 0.70

# ============================================================
# 7. STRIKE SELECTION PARAMETERS
# ============================================================

@dataclass
class StrikeConfig:
    """Delta-based dynamic strike selection."""
    
    # Target deltas
    long_delta_target: float = 0.40         # Buy 40-delta call
    short_delta_target: float = 0.25        # Sell 25-delta call
    delta_tolerance: float = 0.05           # Accept ±0.05 from target
    
    # Regime-dependent spread widths — as percentage of VIX futures (v1.3)
    spread_width_pct: float = 0.30          # 30% of VIX futures level
    min_spread_width: int = 3               # Floor: never less than 3 points
    max_spread_width: int = 8               # Ceiling: never more than 8 points
    
    # Wide spread (tail risk) overlay
    include_wide_spread: bool = True
    wide_spread_width: int = 20             # C20/C40 style wide spread
    wide_spread_allocation_pct: float = 0.30  # 30% of position in wide spread
    
    # DTE targeting
    target_dte: int = 45                    # Enter 45 days before expiry
    dte_range: Tuple[int, int] = (30, 60)   # Acceptable DTE window
    
    # Strike rounding
    strike_increment: float = 1.0           # VIX options have $1 strike increments
    
    # Max cost constraints
    max_spread_cost: float = 3.00           # Never pay more than $3.00 for a spread
    min_risk_reward: float = 1.5            # Minimum R:R ratio to enter

# ============================================================
# 8. FEATURE ENGINEERING PARAMETERS
# ============================================================

@dataclass
class FeatureConfig:
    """Parameters for indicator calculation."""
    
    # Rolling windows
    realized_vol_window: int = 21           # 21-day realized vol (1 month)
    vix_percentile_window: int = 252        # 1-year lookback for percentile
    vvix_ma_window: int = 10                # 10-day VVIX moving average
    momentum_window: int = 5                # 5-day VIX momentum
    z_score_window: int = 63                # Quarterly Z-score lookback
    
    # Term structure
    ts_slope_lookback: int = 20             # 20-day MA of term structure slope
    
    # COT normalization
    cot_percentile_window: int = 156        # 3-year (156 weeks) percentile window
    cot_z_score_window: int = 52            # 1-year Z-score window

# ============================================================
# 9. CONVENIENCE: DEFAULT INSTANCES
# ============================================================

BACKTEST = BacktestConfig()
REGIME = RegimeConfig()
SIGNAL = SignalConfig()
STRIKE = StrikeConfig()
FEATURE = FeatureConfig()


```


## FILE: signals/composite_score.py
```python

"""
signals/composite_score.py
==========================
Weighted composite scoring for entry/exit signal generation.

Each individual indicator is normalized to [0, 1] where:
    1.0 = strongly favors entering a long VIX call spread
    0.0 = strongly against entry

The composite score is a weighted average. Entry when score > threshold.

IMPORTANT: All signals use only backward-looking data. No look-ahead.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from enum import Enum
import logging

from config.settings import SIGNAL
from regime.hmm_classifier import VolRegime

logger = logging.getLogger(__name__)


class SignalDecision(Enum):
    """Signal output decisions."""
    NO_SIGNAL = 0
    ENTRY = 1
    EXIT_PROFIT_TARGET = 2
    EXIT_TIME_STOP = 3
    EXIT_REGIME_CHANGE = 4
    EXIT_STOP_LOSS = 5
    EXIT_PRE_SETTLEMENT = 6


class CompositeSignal:
    """
    Generates entry and exit signals based on weighted indicator composite.
    
    Usage:
        signal = CompositeSignal()
        df = signal.compute(df)
        # df now has: Signal_Score, Signal_Decision, and individual sub-scores
    """
    
    def __init__(self, config: object = SIGNAL):
        self.config = config
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute composite signal score and decisions.
        
        Args:
            df: DataFrame with feature columns and Regime column
            
        Returns:
            df with Signal_Score, Signal_Decision, and sub-score columns
        """
        df = df.copy()
        
        # 1. Compute individual sub-scores (each normalized to [0, 1])
        df = self._score_term_structure(df)
        df = self._score_vix_percentile(df)
        df = self._score_vvix(df)
        df = self._score_cot_positioning(df)
        df = self._score_vrp(df)
        
        # 2. Compute weighted composite
        weights = self.config.weights
        score_cols = {
            "term_structure": "SubScore_TermStructure",
            "vix_percentile": "SubScore_VIXPctl",
            "vvix_level": "SubScore_VVIX",
            "cot_positioning": "SubScore_COT",
            "vrp_signal": "SubScore_VRP",
        }
        
        # Weighted sum (handles NaN gracefully)
        df["Signal_Score"] = 0.0
        total_weight = 0.0
        
        for signal_name, col_name in score_cols.items():
            if col_name in df.columns:
                w = weights.get(signal_name, 0.0)
                mask = df[col_name].notna()
                df.loc[mask, "Signal_Score"] += df.loc[mask, col_name] * w
                total_weight += w
        
        # Normalize by actual available weight
        if total_weight > 0:
            df["Signal_Score"] /= total_weight
        
        # 3. Generate entry decisions (regime-gated, regime-dependent thresholds)
        df["Signal_Entry"] = False
        
        # LOW_VOL: standard threshold
        low_vol_mask = (
            (df["Signal_Score"] >= self.config.entry_score_threshold) &
            (df["Regime"] == VolRegime.LOW_VOL) &
            (df["Signal_Score"].notna())
        )
        df.loc[low_vol_mask, "Signal_Entry"] = True
        
        # TRANSITION: higher bar required
        transition_threshold = getattr(self.config, 'transition_score_threshold', 0.80)
        transition_mask = (
            (df["Signal_Score"] >= transition_threshold) &
            (df["Regime"] == VolRegime.TRANSITION) &
            (df["Signal_Score"].notna())
        )
        df.loc[transition_mask, "Signal_Entry"] = True
        
        logger.info(f"Entry signals: {df['Signal_Entry'].sum()} days out of {len(df)}")
        
        return df
    
    def _score_term_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Term structure score.
        Steep contango = high score (cheap insurance, good entry).
        Backwardation = low score (expensive, avoid).
        """
        if "TS_VIX_VIX3M_Ratio" not in df.columns:
            return df
        
        ratio = df["TS_VIX_VIX3M_Ratio"]
        threshold = self.config.term_structure_contango_threshold
        
        # Score: linear mapping where ratio < 0.85 → 1.0, ratio > 1.05 → 0.0
        # Below threshold = strong contango = high score
        score = np.clip(1.0 - (ratio - 0.85) / (1.05 - 0.85), 0.0, 1.0)
        df["SubScore_TermStructure"] = score
        
        return df
    
    def _score_vix_percentile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        VIX level score.
        Low VIX percentile = high score (cheap options, max asymmetry).
        High VIX = low score (expensive entry).
        """
        if "VIX_Pctl_1yr" not in df.columns:
            return df
        
        pctl = df["VIX_Pctl_1yr"]
        threshold = self.config.vix_low_percentile
        
        # Score: VIX at 0th percentile → 1.0, VIX at 80th percentile → 0.0
        score = np.clip(1.0 - pctl / 80.0, 0.0, 1.0)
        df["SubScore_VIXPctl"] = score
        
        return df
    
    def _score_vvix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        VVIX score.
        Low VVIX = cheap VIX options = high score.
        High VVIX + Low VIX = institutional divergence = bonus score.
        """
        if "VVIX" not in df.columns:
            return df
        
        vvix = df["VVIX"]
        cheap = self.config.vvix_cheap_threshold
        divergence = self.config.vvix_divergence_threshold
        
        # Base score: VVIX < 80 → 1.0, VVIX > 120 → 0.0
        base_score = np.clip(1.0 - (vvix - 75) / (120 - 75), 0.0, 1.0)
        
        # Divergence bonus: high VVIX + low VIX = institutional tell
        if "VVIX_Divergence" in df.columns:
            div = df["VVIX_Divergence"]
            # Positive divergence (VVIX elevated vs VIX) → bonus
            bonus = np.clip(div / 2.0, 0.0, 0.3)  # Max 0.3 bonus
            base_score = np.clip(base_score + bonus, 0.0, 1.0)
        
        df["SubScore_VVIX"] = base_score
        
        return df
    
    def _score_cot_positioning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        COT positioning score.
        Extreme net short = high score (fragile positioning, contrarian bullish).
        """
        if "COT_AM_Pctl_3yr" not in df.columns:
            # If no COT data, return neutral score
            df["SubScore_COT"] = 0.5
            return df
        
        pctl = df["COT_AM_Pctl_3yr"]
        threshold = self.config.cot_extreme_percentile
        
        # AM percentile is: high = more long (less bearish on VIX)
        #                    low = more short (bearish on VIX = contrarian bullish)
        # Invert: low AM pctl (= very short) → high score
        score = np.clip(1.0 - pctl / 100.0, 0.0, 1.0)
        
        # Amplify at extremes: below 10th pctl → boost to 1.0
        extreme_mask = pctl < 10
        score = score.copy()
        score[extreme_mask] = 1.0
        
        df["SubScore_COT"] = score
        
        return df
    
    def _score_vrp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Variance Risk Premium score.
        Large VRP = options expensive relative to realized = headwind.
        Small/negative VRP = cheap options = favorable.
        
        Note: VRP is a HEADWIND indicator. High VRP → low score.
        """
        if "VRP_Simple" not in df.columns:
            df["SubScore_VRP"] = 0.5
            return df
        
        vrp = df["VRP_Simple"]
        threshold = self.config.vrp_threshold
        
        # Score: VRP < 0 (realized > implied, rare) → 1.0
        #        VRP > 8 (very expensive) → 0.0
        score = np.clip(1.0 - vrp / 8.0, 0.0, 1.0)
        df["SubScore_VRP"] = score
        
        return df


class ExitSignalGenerator:
    """
    Generates exit signals for open positions.
    Checks profit targets, time stops, regime changes, and stop losses.
    """
    
    def __init__(self, config: object = SIGNAL):
        self.config = config
    
    def check_exit(
        self,
        current_spread_price: float,
        entry_price: float,
        max_profit: float,
        dte: int,
        current_regime: int,
        entry_regime: int,
        half_closed: bool = False,
    ) -> SignalDecision:
        current_pnl = current_spread_price - entry_price
        pnl_pct = current_pnl / entry_price if entry_price > 0 else 0
        profit_pct_of_max = current_pnl / max_profit if max_profit > 0 else 0
        
        # Scale-out logic (v1.3)
        if not half_closed:
            # First exit: close half at 50% of max profit
            first_target = getattr(self.config, 'first_exit_pct', 0.50)
            if profit_pct_of_max >= first_target:
                return SignalDecision.EXIT_PROFIT_TARGET
        else:
            # Second exit: close remainder at 75% of max profit
            second_target = getattr(self.config, 'second_exit_pct', 0.75)
            if profit_pct_of_max >= second_target:
                return SignalDecision.EXIT_PROFIT_TARGET
        
        # Pre-settlement
        if dte <= self.config.pre_settlement_close_dte:
            return SignalDecision.EXIT_PRE_SETTLEMENT
        
        # Time stop (only on remaining half if half already closed)
        if dte <= self.config.time_stop_dte and current_pnl <= 0:
            return SignalDecision.EXIT_TIME_STOP
        
        # Regime change — unconditional on HIGH_VOL
        if (self.config.regime_exit and 
            current_regime == VolRegime.HIGH_VOL):
            return SignalDecision.EXIT_REGIME_CHANGE
        
        # Stop loss
        if pnl_pct <= -self.config.stop_loss_pct:
            return SignalDecision.EXIT_STOP_LOSS
        
        return SignalDecision.NO_SIGNAL


```


## FILE: backtest/engine.py
```python

"""
backtest/engine.py
==================
Walk-forward backtest engine for VIX bull call spread strategy.

Handles:
    - Position lifecycle: entry -> management -> exit
    - VRO settlement mechanics (or VIX futures proxy)
    - Transaction costs and slippage
    - Performance analytics
    
CRITICAL: No look-ahead bias. All decisions use only data available at that point.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from config.settings import BACKTEST, SIGNAL
from regime.hmm_classifier import VolRegime
from signals.composite_score import SignalDecision, ExitSignalGenerator
from strikes.selector import StrikeSelector, SpreadSelection

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open bull call spread position."""
    id: int
    entry_date: str
    expiry_date: str
    long_strike: int
    short_strike: int
    entry_price: float           # Debit paid
    spread_width: int
    max_profit: float
    entry_regime: int
    dte_at_entry: int
    
    # Optional wide spread
    has_wide: bool = False
    wide_short_strike: int = 0
    wide_entry_price: float = 0.0
    wide_max_profit: float = 0.0
    
    # Tracking
    current_price: float = 0.0
    current_pnl: float = 0.0
    current_dte: int = 0
    # Scale-out tracking (v1.3)
    half_closed: bool = False
    half_closed_price: float = 0.0
    half_closed_pnl: float = 0.0
    exit_date: Optional[str] = None
    exit_price: float = 0.0
    exit_reason: str = ""
    
    @property
    def is_open(self) -> bool:
        return self.exit_date is None


@dataclass 
class BacktestResult:
    """Complete backtest output."""
    trades: List[Dict]
    daily_pnl: pd.Series
    cumulative_pnl: pd.Series
    
    # Summary statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_holding_days: float = 0.0
    
    # By regime
    regime_stats: Dict = field(default_factory=dict)


class BacktestEngine:
    """
    Walk-forward backtest for VIX bull call spread strategy.
    
    Usage:
        engine = BacktestEngine()
        result = engine.run(df)
        engine.print_summary(result)
    """
    
    def __init__(
        self,
        config: object = BACKTEST,
        signal_config: object = SIGNAL,
    ):
        self.config = config
        self.exit_gen = ExitSignalGenerator(signal_config)
        self.strike_sel = StrikeSelector()
        
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.next_id = 1
    
    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Execute the full backtest.
        
        Args:
            df: DataFrame with all features, regime, and signal columns.
                Required: Signal_Entry, Regime, VIX_Spot, UX1, UX2
                
        Returns:
            BacktestResult with all trades and performance metrics
        """
        logger.info("=" * 60)
        logger.info("STARTING BACKTEST")
        logger.info(f"Period: {df.index.min().date()} to {df.index.max().date()}")
        logger.info(f"Trading days: {len(df)}")
        logger.info("=" * 60)
        
        daily_pnl = pd.Series(0.0, index=df.index, name="Daily_PnL")
        prev_portfolio_value = 0.0
        
        for i, (date, row) in enumerate(df.iterrows()):
            date_str = date.strftime("%Y-%m-%d")

            # 1. Update open positions with current prices
            self._update_positions(row, date_str)

            # 2. Check exit signals for open positions
            self._check_exits(row, date_str)

            # 3. Update cooldown rearm (must run every bar, not just entry days)
            self._update_cooldown_rearm(row)

            # 4. Check entry signals (if capacity available)
            if row.get("Signal_Entry", False):
                self._try_entry(row, date_str)
            
            # 5. Record daily P&L (change in total portfolio value)
            # Portfolio value = unrealized P&L on open + cumulative realized P&L
            unrealized = sum(p.current_pnl for p in self.positions if p.is_open)
            realized = sum(p.current_pnl for p in self.closed_positions)
            portfolio_value = unrealized + realized
            daily_pnl.loc[date] = portfolio_value - prev_portfolio_value
            prev_portfolio_value = portfolio_value
        
        # Force-close any remaining open positions
        self._close_all_remaining(df.iloc[-1], df.index[-1].strftime("%Y-%m-%d"))
        
        # Build result
        result = self._build_result(daily_pnl)
        
        return result
    
    def _update_cooldown_rearm(self, row: pd.Series):
        """
        Track signal-reset cooldown on every bar.

        Rearm when Signal_Entry turns off after a loss. This must run
        every bar (not just entry days) so the system sees the signal
        go False before it goes True again.
        """
        if getattr(self, '_score_rearmed', True):
            return  # Already rearmed or no cooldown active
        if not row.get("Signal_Entry", False):
            self._score_rearmed = True

    def _try_entry(self, row: pd.Series, date_str: str):
        """Attempt to open a new position if capacity allows."""
        # Cooldown logic (v1.3: signal-reset based)
        if self.closed_positions:
            last_closed = self.closed_positions[-1]
            if last_closed.current_pnl < 0:
                cooldown_type = getattr(self.config, 'cooldown_type', 'calendar')

                if cooldown_type == "signal_reset":
                    if not getattr(self, '_score_rearmed', True):
                        return

                else:  # calendar
                    last_exit = datetime.strptime(last_closed.exit_date, "%Y-%m-%d")
                    current = datetime.strptime(date_str, "%Y-%m-%d")
                    if (current - last_exit).days < self.config.cooldown_after_loss_days:
                        return

        if len(self.positions) >= self.config.max_concurrent_positions:
            return
        
        regime = row.get("Regime", None)
        if regime is None or np.isnan(regime):
            return
        
        regime_enum = VolRegime(int(regime))
        
        # Get VIX futures for strike selection
        vix_futures = row.get("UX1", row.get("VIX_Spot", None))
        if vix_futures is None or np.isnan(vix_futures) or vix_futures <= 0:
            return
        
        # Get VVIX for IV proxy (VVIX ≈ IV of VIX options)
        vix_iv = row.get("VVIX", None)
        if vix_iv is not None and (np.isnan(vix_iv) or vix_iv <= 0):
            vix_iv = None
        
        # Select strikes
        # Estimate DTE: target the next monthly VIX expiry (~30-45 days out)
        target_dte = self.strike_sel.config.target_dte
        
        selection = self.strike_sel.select(
            vix_futures=vix_futures,
            regime=regime_enum,
            vix_iv=vix_iv,
            dte=target_dte,
        )
        
        if selection is None:
            return
        
        # Apply slippage to entry cost
        entry_cost = selection.estimated_cost * (1 + self.config.entry_slippage_pct)
        entry_cost += self.config.commission_per_contract * 2 / 100  # 2 legs, per $100 multiplier
        
        # Estimate expiry date (next monthly VIX expiry, ~45 days out)
        entry_dt = datetime.strptime(date_str, "%Y-%m-%d")
        expiry_dt = entry_dt + timedelta(days=target_dte)
        expiry_str = expiry_dt.strftime("%Y-%m-%d")
        
        # Open position
        pos = Position(
            id=self.next_id,
            entry_date=date_str,
            expiry_date=expiry_str,
            long_strike=selection.long_strike,
            short_strike=selection.short_strike,
            entry_price=round(entry_cost, 2),
            spread_width=selection.spread_width,
            max_profit=round(selection.spread_width - entry_cost, 2),
            entry_regime=int(regime),
            dte_at_entry=target_dte,
        )
        
        # Add wide spread if selected
        if selection.wide_short_strike is not None and selection.wide_estimated_cost is not None:
            wide_cost = selection.wide_estimated_cost * (1 + self.config.entry_slippage_pct)
            pos.has_wide = True
            pos.wide_short_strike = selection.wide_short_strike
            pos.wide_entry_price = round(wide_cost, 2)
            pos.wide_max_profit = round(selection.wide_short_strike - selection.long_strike - wide_cost, 2)
        
        self.positions.append(pos)
        self.next_id += 1
        
        logger.debug(
            f"ENTRY #{pos.id} on {date_str}: "
            f"C{pos.long_strike}/C{pos.short_strike} @ ${pos.entry_price:.2f}"
        )
    
    def _update_positions(self, row: pd.Series, date_str: str):
        """Update current price and P&L for all open positions."""
        vix_futures = row.get("UX1", row.get("VIX_Spot", 0))
        if vix_futures is None or np.isnan(vix_futures):
            return
        
        for pos in self.positions:
            if not pos.is_open:
                continue
            
            # Estimate current spread value from VIX futures
            # At expiry: intrinsic value
            # Before expiry: approximate with intrinsic + time value decay
            entry_dt = datetime.strptime(pos.entry_date, "%Y-%m-%d")
            current_dt = datetime.strptime(date_str, "%Y-%m-%d")
            expiry_dt = datetime.strptime(pos.expiry_date, "%Y-%m-%d")
            
            pos.current_dte = max(0, (expiry_dt - current_dt).days)
            
            # Simple valuation: intrinsic + declining time value
            intrinsic = self._calc_intrinsic(vix_futures, pos.long_strike, pos.short_strike)
            
            if pos.current_dte > 0:
                # Time value decays with sqrt(time)
                time_ratio = np.sqrt(pos.current_dte / pos.dte_at_entry) if pos.dte_at_entry > 0 else 0
                initial_time_value = max(pos.entry_price - self._calc_intrinsic(
                    vix_futures, pos.long_strike, pos.short_strike
                ), 0)
                time_value = initial_time_value * time_ratio * 0.5  # Conservative
                pos.current_price = intrinsic + time_value
            else:
                # At expiry: intrinsic only
                pos.current_price = intrinsic
            
            pos.current_pnl = pos.current_price - pos.entry_price
    
    def _check_exits(self, row: pd.Series, date_str: str):
        """Check exit conditions — supports scale-out (v1.3)."""
        regime = row.get("Regime", None)
        if regime is not None and not np.isnan(regime):
            current_regime = int(regime)
        else:
            current_regime = VolRegime.TRANSITION
        
        positions_to_act = []
        
        for pos in self.positions:
            if not pos.is_open:
                continue
            
            decision = self.exit_gen.check_exit(
                current_spread_price=pos.current_price,
                entry_price=pos.entry_price,
                max_profit=pos.max_profit,
                dte=pos.current_dte,
                current_regime=current_regime,
                entry_regime=pos.entry_regime,
                half_closed=pos.half_closed,
            )
            
            if decision != SignalDecision.NO_SIGNAL:
                positions_to_act.append((pos, decision, date_str))
        
        for pos, decision, date_str in positions_to_act:
            if decision == SignalDecision.EXIT_PROFIT_TARGET and not pos.half_closed:
                # Scale-out: close first half, keep position open
                self._half_close_position(pos, date_str)
            else:
                # Full close (second target, time stop, regime change, etc.)
                self._close_position(pos, date_str, decision.name)
    
    def _close_position(self, pos: Position, date_str: str, reason: str):
        """Close position (or remaining half if partially closed)."""
        exit_price = pos.current_price * (1 - self.config.exit_slippage_pct)
        exit_price -= self.config.commission_per_contract * 2 / 100
        
        pos.exit_date = date_str
        pos.exit_price = round(max(exit_price, 0), 2)
        
        if pos.half_closed:
            # Second half P&L
            second_half_pnl = (pos.exit_price - pos.entry_price) / 2
            pos.current_pnl = round(pos.half_closed_pnl + second_half_pnl, 2)
        else:
            pos.current_pnl = round(pos.exit_price - pos.entry_price, 2)
        
        pos.exit_reason = reason
        
        self.closed_positions.append(pos)
        self.positions.remove(pos)

        # Reset cooldown rearm flag on new loss
        if pos.current_pnl < 0:
            self._score_rearmed = False
        
        pnl_sign = "+" if pos.current_pnl >= 0 else ""
        half_tag = " [scaled-out]" if pos.half_closed else ""
        logger.debug(
            f"EXIT #{pos.id} on {date_str} ({reason}{half_tag}): "
            f"C{pos.long_strike}/C{pos.short_strike} "
            f"PnL: {pnl_sign}${pos.current_pnl:.2f}"
        )
    
    def _half_close_position(self, pos: Position, date_str: str):
        """Close first half of position at profit target. Keep remainder open."""
        exit_price = pos.current_price * (1 - self.config.exit_slippage_pct)
        exit_price -= self.config.commission_per_contract * 2 / 100
        
        pos.half_closed = True
        pos.half_closed_price = round(max(exit_price, 0), 2)
        pos.half_closed_pnl = round((pos.half_closed_price - pos.entry_price) / 2, 2)  # Half the P&L
        
        logger.debug(
            f"HALF-CLOSE #{pos.id} on {date_str}: "
            f"C{pos.long_strike}/C{pos.short_strike} "
            f"Half P&L: +${pos.half_closed_pnl:.2f}"
        )
    
    def _close_all_remaining(self, row: pd.Series, date_str: str):
        """Force-close all open positions at end of backtest."""
        for pos in list(self.positions):
            if pos.is_open:
                self._close_position(pos, date_str, "BACKTEST_END")
    
    @staticmethod
    def _calc_intrinsic(futures: float, long_k: int, short_k: int) -> float:
        """Calculate intrinsic value of bull call spread at a given futures level."""
        long_val = max(futures - long_k, 0)
        short_val = max(futures - short_k, 0)
        return long_val - short_val
    
    def _get_daily_pnl(self) -> float:
        """Sum of daily P&L changes across all open positions."""
        return sum(pos.current_pnl for pos in self.positions if pos.is_open)
    
    def _build_result(self, daily_pnl: pd.Series) -> BacktestResult:
        """Compile backtest results and compute statistics."""
        trades = []
        for pos in self.closed_positions:
            trades.append({
                "id": pos.id,
                "entry_date": pos.entry_date,
                "exit_date": pos.exit_date,
                "long_strike": pos.long_strike,
                "short_strike": pos.short_strike,
                "entry_price": pos.entry_price,
                "exit_price": pos.exit_price,
                "pnl": round(pos.current_pnl, 2),
                "pnl_pct": round(pos.current_pnl / pos.entry_price * 100, 1) if pos.entry_price > 0 else 0,
                "exit_reason": pos.exit_reason,
                "regime_at_entry": VolRegime(pos.entry_regime).name if pos.entry_regime in [0,1,2] else "UNKNOWN",
                "dte_at_entry": pos.dte_at_entry,
                "holding_days": (
                    datetime.strptime(pos.exit_date, "%Y-%m-%d") -
                    datetime.strptime(pos.entry_date, "%Y-%m-%d")
                ).days if pos.exit_date else 0,
            })
        
        trades_df = pd.DataFrame(trades)
        cumulative_pnl = daily_pnl.cumsum()
        
        # Summary stats
        result = BacktestResult(
            trades=trades,
            daily_pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
        )
        
        if trades:
            pnls = [t["pnl"] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            
            result.total_trades = len(trades)
            result.winning_trades = len(wins)
            result.losing_trades = len(losses)
            result.win_rate = len(wins) / len(trades) * 100
            result.avg_win = np.mean(wins) if wins else 0
            result.avg_loss = np.mean(losses) if losses else 0
            result.profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')
            result.total_pnl = sum(pnls)
            result.avg_holding_days = np.mean([t["holding_days"] for t in trades])
            
            # Max drawdown
            cum = cumulative_pnl
            peak = cum.cummax()
            dd = cum - peak
            result.max_drawdown = dd.min()
            
            # Sharpe on premium-at-risk basis (v1.3)
            # Denominator: average capital deployed (entry price * holding days)
            total_premium_days = sum(
                t["entry_price"] * t["holding_days"] for t in trades if t["holding_days"] > 0
            )
            trading_days = len(daily_pnl[daily_pnl != 0])
            avg_premium_at_risk = total_premium_days / max(trading_days, 1)
            
            if avg_premium_at_risk > 0:
                daily_return = daily_pnl / avg_premium_at_risk
                active_returns = daily_return[daily_return != 0]
                if len(active_returns) > 1 and active_returns.std() > 0:
                    result.sharpe_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252)
        
        return result
    
    @staticmethod
    def print_summary(result: BacktestResult):
        """Print formatted backtest summary."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        
        print(f"\nTotal Trades:     {result.total_trades}")
        print(f"Win Rate:         {result.win_rate:.1f}%")
        print(f"Avg Win:          +${result.avg_win:.2f}")
        print(f"Avg Loss:         ${result.avg_loss:.2f}")
        print(f"Profit Factor:    {result.profit_factor:.2f}")
        print(f"Total P&L:        ${result.total_pnl:.2f}")
        print(f"Max Drawdown:     ${result.max_drawdown:.2f}")
        print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        print(f"Avg Holding Days: {result.avg_holding_days:.0f}")
        
        print("\n--- Trade Log ---")
        for t in result.trades:
            sign = "+" if t["pnl"] >= 0 else ""
            print(
                f"  #{t['id']:3d} | {t['entry_date']} -> {t['exit_date']} | "
                f"C{t['long_strike']}/C{t['short_strike']} | "
                f"${t['entry_price']:.2f} -> ${t['exit_price']:.2f} | "
                f"{sign}${t['pnl']:.2f} ({sign}{t['pnl_pct']:.0f}%) | "
                f"{t['exit_reason']} | {t['regime_at_entry']}"
            )
        
        print("=" * 60)


```


## FILE: strikes/selector.py
```python

"""
strikes/selector.py
===================
Dynamic strike selection for VIX bull call spreads.

Uses delta-based targeting with regime-dependent spread widths.
Strikes auto-adjust for VIX level and implied volatility.

Design rationale:
    - Delta-based > fixed strikes: auto-adjusts for vol surface
    - 40Δ long / 25Δ short: standard institutional risk/reward
    - Wider spreads in low vol (cheap insurance)
    - No new entries in high vol (mean reversion headwind)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import logging

from config.settings import STRIKE
from regime.hmm_classifier import VolRegime

logger = logging.getLogger(__name__)


@dataclass
class SpreadSelection:
    """Result of strike selection."""
    long_strike: int
    short_strike: int
    spread_width: int
    estimated_cost: float        # Estimated debit
    max_profit: float            # spread_width - estimated_cost
    risk_reward: float           # max_profit / estimated_cost
    breakeven: float             # long_strike + estimated_cost
    long_delta: float
    short_delta: float
    dte: int
    regime: str
    
    # Optional: wide spread overlay
    wide_short_strike: Optional[int] = None
    wide_estimated_cost: Optional[float] = None
    wide_max_profit: Optional[float] = None
    wide_risk_reward: Optional[float] = None


class StrikeSelector:
    """
    Select optimal strikes for VIX bull call spreads.
    
    Strategy:
        1. Determine target expiry (45 DTE, within 30-60 window)
        2. Get VIX futures price for that expiry
        3. Select long strike: nearest to 40-delta
        4. Select short strike: regime-dependent width above long
        5. Validate cost and risk/reward constraints
        6. Optionally add wide spread overlay
    """
    
    def __init__(self, config: object = STRIKE):
        self.config = config
    
    def select(
        self,
        vix_futures: float,
        regime: VolRegime,
        vix_iv: float = None,
        dte: int = 45,
        available_strikes: Optional[List[int]] = None,
    ) -> Optional[SpreadSelection]:
        """
        Select optimal spread given current market conditions.
        
        Args:
            vix_futures: Current VIX futures price for target expiry
            regime: Current volatility regime
            vix_iv: Implied volatility of VIX options (for delta estimation)
            dte: Days to expiry
            available_strikes: List of available strike prices
            
        Returns:
            SpreadSelection or None if no valid spread found
        """
        # No entries in high vol regime
        if regime == VolRegime.HIGH_VOL:
            logger.info("HIGH_VOL regime: no new entries")
            return None
        
        # Check DTE window
        dte_min, dte_max = self.config.dte_range
        if dte < dte_min or dte > dte_max:
            logger.info(f"DTE {dte} outside window [{dte_min}, {dte_max}]")
            return None
        
        # Scale-normalized spread width (v1.3): percentage of VIX futures, clamped
        raw_width = round(vix_futures * self.config.spread_width_pct)
        spread_width = max(self.config.min_spread_width, min(self.config.max_spread_width, raw_width))
        
        if spread_width == 0:
            return None
        
        # Generate available strikes if not provided
        if available_strikes is None:
            # VIX options have $1 increments
            min_strike = max(10, int(vix_futures - 10))
            max_strike = int(vix_futures + 30)
            available_strikes = list(range(min_strike, max_strike + 1))
        
        # Select long strike (target: 40-delta, approximately ATM or slightly OTM)
        long_strike = self._select_long_strike(vix_futures, vix_iv, dte, available_strikes)
        
        # Select short strike (regime-dependent width above long)
        short_strike = long_strike + spread_width
        
        # Validate short strike exists
        if short_strike not in available_strikes:
            short_strike = min(available_strikes, key=lambda x: abs(x - short_strike))
        
        # Estimate cost using simple Black-76 approximation
        estimated_cost = self._estimate_spread_cost(
            vix_futures, long_strike, short_strike, vix_iv, dte
        )
        
        # Validate constraints
        if estimated_cost > self.config.max_spread_cost:
            logger.info(f"Spread cost ${estimated_cost:.2f} exceeds max ${self.config.max_spread_cost:.2f}")
            return None
        
        actual_width = short_strike - long_strike
        max_profit = actual_width - estimated_cost
        risk_reward = max_profit / estimated_cost if estimated_cost > 0 else 0
        
        if risk_reward < self.config.min_risk_reward:
            logger.info(f"R:R {risk_reward:.1f} below minimum {self.config.min_risk_reward}")
            return None
        
        # Build base selection
        selection = SpreadSelection(
            long_strike=long_strike,
            short_strike=short_strike,
            spread_width=actual_width,
            estimated_cost=round(estimated_cost, 2),
            max_profit=round(max_profit, 2),
            risk_reward=round(risk_reward, 1),
            breakeven=round(long_strike + estimated_cost, 2),
            long_delta=self._estimate_delta(vix_futures, long_strike, vix_iv, dte),
            short_delta=self._estimate_delta(vix_futures, short_strike, vix_iv, dte),
            dte=dte,
            regime=regime.name,
        )
        
        # Optionally add wide spread overlay
        if self.config.include_wide_spread and regime == VolRegime.LOW_VOL:
            wide_short = long_strike + self.config.wide_spread_width
            if wide_short in available_strikes or wide_short <= max(available_strikes):
                wide_cost = self._estimate_spread_cost(
                    vix_futures, long_strike, wide_short, vix_iv, dte
                )
                selection.wide_short_strike = wide_short
                selection.wide_estimated_cost = round(wide_cost, 2)
                selection.wide_max_profit = round(wide_short - long_strike - wide_cost, 2)
                selection.wide_risk_reward = round(
                    (wide_short - long_strike - wide_cost) / wide_cost, 1
                ) if wide_cost > 0 else 0
        
        logger.info(
            f"Selected: C{long_strike}/C{short_strike} ({actual_width}pt) "
            f"@ ${estimated_cost:.2f}, R:R 1:{risk_reward:.1f}, "
            f"BE {long_strike + estimated_cost:.2f}, Regime={regime.name}"
        )
        
        return selection
    
    def _select_long_strike(
        self,
        futures: float,
        iv: Optional[float],
        dte: int,
        available: List[int],
    ) -> int:
        """
        Select long strike using moneyness-based targeting.
        
        For VIX options, delta-based targeting with Black-76 is unreliable 
        because VIX IV is 80-130% (vs 15-30% for equities). Standard 
        40-delta targeting pushes strikes too far OTM.
        
        Instead, we target slightly OTM: long strike = futures - 1 to 
        futures + 0 (at-the-money to 1 strike below). This matches 
        practitioner convention for VIX call spreads and produces spreads 
        with 1.5:1 to 3:1 risk/reward.
        
        Wang & Daigler (2011) found Black-like models work best for ITM 
        VIX options but misprice OTM — confirming that delta targeting 
        is unreliable for strike selection in the VIX complex.
        """
        # Target: 1 strike below futures (slightly ITM, captures most of the move)
        target = futures - 1
        
        # Round to nearest available strike
        long_strike = min(available, key=lambda x: abs(x - target))
        
        return long_strike
    
    def _estimate_spread_cost(
        self,
        futures: float,
        long_strike: int,
        short_strike: int,
        iv: Optional[float],
        dte: int,
    ) -> float:
        """
        Estimate spread cost using simplified Black-76 pricing.
        
        This is an approximation for strike selection — actual execution
        prices come from Bloomberg bid/ask quotes.
        """
        if iv is None:
            # Rough heuristic when IV is not available
            # VIX call spread cost ≈ intrinsic + time value
            intrinsic = max(futures - long_strike, 0) - max(futures - short_strike, 0)
            time_factor = np.sqrt(dte / 365) * 0.5  # Rough time value multiplier
            return max(intrinsic + time_factor, 0.20)  # Floor at $0.20
        
        # Simplified Black-76 for VIX options
        from scipy.stats import norm
        
        t = dte / 365
        sigma = iv / 100
        
        def black76_call(F, K, sigma, t):
            if t <= 0 or sigma <= 0:
                return max(F - K, 0)
            d1 = (np.log(F / K) + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
            d2 = d1 - sigma * np.sqrt(t)
            return F * norm.cdf(d1) - K * norm.cdf(d2)
        
        long_price = black76_call(futures, long_strike, sigma, t)
        short_price = black76_call(futures, short_strike, sigma, t)
        
        return max(long_price - short_price, 0.10)
    
    def _estimate_delta(
        self,
        futures: float,
        strike: int,
        iv: Optional[float],
        dte: int,
    ) -> float:
        """Estimate call option delta using Black-76."""
        if iv is None or iv <= 0 or dte <= 0:
            # Rough approximation
            if futures > strike:
                return min(0.95, 0.5 + (futures - strike) / futures)
            else:
                return max(0.05, 0.5 - (strike - futures) / futures)
        
        from scipy.stats import norm
        
        t = dte / 365
        sigma = iv / 100
        
        if t <= 0 or sigma <= 0:
            return 1.0 if futures > strike else 0.0
        
        d1 = (np.log(futures / strike) + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
        return norm.cdf(d1)


```


## FILE: features/indicators.py
```python

"""
features/indicators.py
======================
Feature engineering for VIX strategy signals.
Computes all derived indicators from raw Bloomberg + COT data.

All functions are pure: DataFrame in → DataFrame out, no side effects.
All indicators use ONLY backward-looking data (no look-ahead bias).
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from config.settings import FEATURE

logger = logging.getLogger(__name__)


def compute_all_features(
    df: pd.DataFrame,
    cot_df: Optional[pd.DataFrame] = None,
    config: object = FEATURE,
) -> pd.DataFrame:
    """
    Master function: compute all features from raw data.
    
    Args:
        df: Daily data with VIX_Spot, VIX3M, UX1..UX9, VVIX, SPX_Close, etc.
        cot_df: Weekly COT data (optional, forward-filled to daily)
        config: FeatureConfig instance
        
    Returns:
        df with additional feature columns appended
    """
    df = df.copy()
    
    # 1. Term Structure Features
    df = add_term_structure_features(df, config)
    
    # 2. VIX Level & Momentum Features
    df = add_vix_level_features(df, config)
    
    # 3. Variance Risk Premium
    df = add_vrp_features(df, config)
    
    # 4. VVIX Features
    df = add_vvix_features(df, config)
    
    # 5. COT Positioning Features (if available)
    if cot_df is not None and not cot_df.empty:
        df = add_cot_features(df, cot_df, config)
    
    # 6. Cross-Asset Features
    df = add_cross_asset_features(df, config)
    
    # 7. HMM Input Features (specific transformations for the regime classifier)
    df = add_hmm_features(df)
    
    logger.info(f"Feature engineering complete. {len(df.columns)} total columns.")
    return df


# ============================================================
# TERM STRUCTURE FEATURES
# ============================================================

def add_term_structure_features(df: pd.DataFrame, config: object = FEATURE) -> pd.DataFrame:
    """
    VIX term structure indicators.
    
    Key insight from research: term structure slope predicts FUTURES returns
    (not spot direction). Steep contango = cheap insurance = good entry for calls.
    """
    # 1. VIX/VIX3M ratio (IVTS - Implied Volatility Term Structure)
    # < 0.90 = strong contango (bullish for call spreads)
    # > 1.00 = backwardation (caution)
    if "VIX3M" in df.columns:
        df["TS_VIX_VIX3M_Ratio"] = df["VIX_Spot"] / df["VIX3M"]
    
    # 2. Front-month futures basis: (UX1 - VIX_Spot) / VIX_Spot
    # Positive = contango, Negative = backwardation
    if "UX1" in df.columns:
        df["TS_Basis_Pct"] = (df["UX1"] - df["VIX_Spot"]) / df["VIX_Spot"] * 100
    
    # 3. Term structure slope: UX2 - UX1 (absolute points)
    if "UX1" in df.columns and "UX2" in df.columns:
        df["TS_Slope_UX2_UX1"] = df["UX2"] - df["UX1"]
        
        # Normalized slope: (UX2 - UX1) / UX1
        df["TS_Slope_Pct"] = df["TS_Slope_UX2_UX1"] / df["UX1"] * 100
    
    # 4. Full curve slope: UX4 - UX1 (captures longer-term expectations)
    if "UX1" in df.columns and "UX4" in df.columns:
        df["TS_Slope_UX4_UX1"] = df["UX4"] - df["UX1"]
    
    # 5. Term structure curvature: (UX3 - UX2) - (UX2 - UX1)
    # Positive = convex (normal), Negative = concave (stress building in front end)
    if all(f"UX{i}" in df.columns for i in [1, 2, 3]):
        df["TS_Curvature"] = (df["UX3"] - df["UX2"]) - (df["UX2"] - df["UX1"])
    
    # 6. Term structure momentum: change in slope over N days
    if "TS_Slope_UX2_UX1" in df.columns:
        df["TS_Slope_Momentum_5d"] = df["TS_Slope_UX2_UX1"].diff(5)
        df["TS_Slope_Momentum_20d"] = df["TS_Slope_UX2_UX1"].diff(20)
    
    # 7. Contango percentile (rolling 1-year)
    if "TS_Slope_Pct" in df.columns:
        df["TS_Contango_Pctl"] = df["TS_Slope_Pct"].rolling(
            window=252, min_periods=63
        ).apply(lambda x: (x < x.iloc[-1]).mean() * 100, raw=False)
    
    # 8. Roll yield proxy: front-month decay rate
    if "UX1" in df.columns:
        df["TS_Roll_Yield_Daily"] = -df["UX1"].diff(1) / df["UX1"].shift(1) * 100
    
    return df


# ============================================================
# VIX LEVEL & MOMENTUM FEATURES
# ============================================================

def add_vix_level_features(df: pd.DataFrame, config: object = FEATURE) -> pd.DataFrame:
    """
    VIX level, momentum, and mean-reversion indicators.
    
    Key insight: VIX has a half-life of ~11 trading days (OU process).
    Low VIX + high momentum toward lower levels = spring coiling.
    """
    vix = df["VIX_Spot"]
    
    # 1. VIX log returns (primary HMM input)
    df["VIX_LogReturn"] = np.log(vix / vix.shift(1))
    
    # 2. VIX daily change (raw)
    df["VIX_DailyChange"] = vix.diff(1)
    
    # 3. VIX spike indicator: today vs yesterday
    df["VIX_Spike_1d"] = df["VIX_DailyChange"]
    
    # 4. VIX momentum (5-day)
    df["VIX_Momentum_5d"] = vix.pct_change(5) * 100
    
    # 5. VIX percentile rank (rolling 1-year)
    window = config.vix_percentile_window
    df["VIX_Pctl_1yr"] = vix.rolling(window=window, min_periods=63).apply(
        lambda x: (x < x.iloc[-1]).mean() * 100, raw=False
    )
    
    # 6. VIX Z-score (rolling quarterly)
    z_window = config.z_score_window
    df["VIX_ZScore"] = (
        (vix - vix.rolling(z_window, min_periods=21).mean()) / 
        vix.rolling(z_window, min_periods=21).std()
    )
    
    # 7. VIX distance from 200-day MA
    df["VIX_MA200"] = vix.rolling(200, min_periods=100).mean()
    df["VIX_Dist_MA200_Pct"] = (vix - df["VIX_MA200"]) / df["VIX_MA200"] * 100
    
    # 8. Realized volatility of VIX (vol of vol, from returns)
    df["VIX_RealizedVol_21d"] = df["VIX_LogReturn"].rolling(21, min_periods=10).std() * np.sqrt(252)
    
    # 9. VIX 5-day and 20-day moving averages
    df["VIX_MA5"] = vix.rolling(5).mean()
    df["VIX_MA20"] = vix.rolling(20).mean()
    
    # 10. VIX regime indicator: VIX9D/VIX ratio (short-term stress)
    if "VIX9D" in df.columns:
        df["VIX_ShortTermStress"] = df["VIX9D"] / vix
    
    return df


# ============================================================
# VARIANCE RISK PREMIUM
# ============================================================

def add_vrp_features(df: pd.DataFrame, config: object = FEATURE) -> pd.DataFrame:
    """
    Variance Risk Premium: VIX² - Realized Variance.
    
    Positive VRP = implied > realized = structural headwind for long vol.
    Large VRP = expensive options = harder for call spreads to profit.
    
    Key insight from Bollerslev et al. (2009): VRP predicts equity returns
    and represents the compensation investors demand for bearing vol risk.
    """
    if "SPX_Close" not in df.columns:
        logger.warning("SPX_Close not available. Skipping VRP features.")
        return df
    
    spx = df["SPX_Close"]
    vix = df["VIX_Spot"]
    
    # 1. Realized volatility of SPX (EMA-weighted, annualized)
    # EMA weights recent days more heavily, reducing lag during sudden shocks
    # This addresses the VRP lag problem: when VIX spikes instantly but 
    # backward-looking RV is still low, EMA catches up faster than SMA
    spx_returns = np.log(spx / spx.shift(1))
    rv_window = config.realized_vol_window
    
    # EMA of squared returns, then annualize
    squared_returns = spx_returns ** 2
    ema_var = squared_returns.ewm(span=rv_window, min_periods=10).mean()
    df["SPX_RealizedVol_21d"] = np.sqrt(ema_var * 252) * 100
    
    # 2. VRP = VIX - Realized Vol (simple, in vol points)
    df["VRP_Simple"] = vix - df["SPX_RealizedVol_21d"]
    
    # 3. VRP Z-score (is current VRP unusually high or low?)
    z_window = config.z_score_window
    df["VRP_ZScore"] = (
        (df["VRP_Simple"] - df["VRP_Simple"].rolling(z_window, min_periods=21).mean()) /
        df["VRP_Simple"].rolling(z_window, min_periods=21).std()
    )
    
    # 4. VRP percentile (rolling 1-year)
    df["VRP_Pctl_1yr"] = df["VRP_Simple"].rolling(252, min_periods=63).apply(
        lambda x: (x < x.iloc[-1]).mean() * 100, raw=False
    )
    
    # 5. 5-day VRP change (VRP expanding or contracting?)
    df["VRP_Momentum_5d"] = df["VRP_Simple"].diff(5)
    
    return df


# ============================================================
# VVIX FEATURES
# ============================================================

def add_vvix_features(df: pd.DataFrame, config: object = FEATURE) -> pd.DataFrame:
    """
    VVIX (Volatility of VIX) indicators.
    
    Key insight from Park (2013): VVIX predicts VIX call returns at 99% confidence
    even after controlling for VIX level. It captures tail risk demand that VIX alone misses.
    
    Critical divergence: High VVIX + Low VIX = institutional pre-positioning for spike.
    """
    if "VVIX" not in df.columns:
        logger.warning("VVIX not available. Skipping VVIX features.")
        return df
    
    vvix = df["VVIX"]
    vix = df["VIX_Spot"]
    
    # 1. VVIX level (raw)
    # < 80: complacency, cheap VIX options
    # 80-100: normal
    # 100-120: elevated hedging demand
    # > 125: acute stress
    
    # 2. VVIX moving average (smoothed)
    ma_window = config.vvix_ma_window
    df["VVIX_MA10"] = vvix.rolling(ma_window).mean()
    
    # 3. VVIX percentile (1-year)
    df["VVIX_Pctl_1yr"] = vvix.rolling(252, min_periods=63).apply(
        lambda x: (x < x.iloc[-1]).mean() * 100, raw=False
    )
    
    # 4. VVIX/VIX ratio (normalized vol-of-vol)
    df["VVIX_VIX_Ratio"] = vvix / vix
    
    # 5. VVIX divergence score: high VVIX + low VIX = institutional tell
    # Standardize both, then take the difference
    vvix_z = (vvix - vvix.rolling(63).mean()) / vvix.rolling(63).std()
    vix_z = (vix - vix.rolling(63).mean()) / vix.rolling(63).std()
    df["VVIX_Divergence"] = vvix_z - vix_z  # Positive = VVIX elevated relative to VIX
    
    # 6. VVIX Z-score
    df["VVIX_ZScore"] = (
        (vvix - vvix.rolling(63, min_periods=21).mean()) /
        vvix.rolling(63, min_periods=21).std()
    )
    
    return df


# ============================================================
# COT POSITIONING FEATURES
# ============================================================

def add_cot_features(
    df: pd.DataFrame,
    cot_df: pd.DataFrame,
    config: object = FEATURE,
) -> pd.DataFrame:
    """
    CFTC COT positioning indicators.
    
    Key insight: COT signals FRAGILITY, not timing.
    Extreme net short = coiled spring. But can persist for weeks.
    Weekly data → forward-fill to daily for alignment.
    """
    # Forward-fill weekly COT data to daily
    cot_daily = cot_df.reindex(df.index, method="ffill")
    
    # 1. Asset Manager Net Position (raw, forward-filled)
    if "AM_Net" in cot_daily.columns:
        df["COT_AM_Net"] = cot_daily["AM_Net"]
        
        # OI-normalized
        if "AM_Net_OI_Pct" in cot_daily.columns:
            df["COT_AM_Net_OI_Pct"] = cot_daily["AM_Net_OI_Pct"]
    
    # 2. Leveraged Funds Net Position
    if "LF_Net" in cot_daily.columns:
        df["COT_LF_Net"] = cot_daily["LF_Net"]
        if "LF_Net_OI_Pct" in cot_daily.columns:
            df["COT_LF_Net_OI_Pct"] = cot_daily["LF_Net_OI_Pct"]
    
    # 3. AM Net percentile (rolling 3-year)
    pctl_window = config.cot_percentile_window  # 156 weeks ≈ 3 years
    # Since COT is weekly, we compute percentile on the weekly data then forward-fill
    if "AM_Net" in cot_df.columns:
        am_pctl = cot_df["AM_Net"].rolling(pctl_window, min_periods=26).apply(
            lambda x: (x < x.iloc[-1]).mean() * 100, raw=False
        )
        df["COT_AM_Pctl_3yr"] = am_pctl.reindex(df.index, method="ffill")
    
    # 4. AM Net Z-score (rolling 1-year)
    z_window = config.cot_z_score_window  # 52 weeks
    if "AM_Net" in cot_df.columns:
        am_z = (
            (cot_df["AM_Net"] - cot_df["AM_Net"].rolling(z_window, min_periods=13).mean()) /
            cot_df["AM_Net"].rolling(z_window, min_periods=13).std()
        )
        df["COT_AM_ZScore"] = am_z.reindex(df.index, method="ffill")
    
    # 5. Week-over-week change in AM Net (momentum of positioning)
    if "AM_Net" in cot_df.columns:
        am_wow = cot_df["AM_Net"].diff(1)
        df["COT_AM_WoW_Change"] = am_wow.reindex(df.index, method="ffill")
    
    # 6. Total Open Interest
    if "Total_OI" in cot_daily.columns:
        df["COT_Total_OI"] = cot_daily["Total_OI"]
    
    return df


# ============================================================
# CROSS-ASSET FEATURES
# ============================================================

def add_cross_asset_features(df: pd.DataFrame, config: object = FEATURE) -> pd.DataFrame:
    """
    Cross-asset indicators for regime confirmation.
    Equity drawdown and credit stress provide independent regime signals.
    """
    # 1. SPX drawdown from 52-week high
    if "SPX_Close" in df.columns:
        spx = df["SPX_Close"]
        df["SPX_Drawdown_Pct"] = (spx / spx.rolling(252, min_periods=63).max() - 1) * 100
        
        # SPX 5-day return (short-term equity momentum)
        df["SPX_Return_5d"] = spx.pct_change(5) * 100
        
        # SPX 20-day return
        df["SPX_Return_20d"] = spx.pct_change(20) * 100
    
    # 2. NQ drawdown (our actual hedge target)
    if "NQ_Close" in df.columns:
        nq = df["NQ_Close"]
        df["NQ_Drawdown_Pct"] = (nq / nq.rolling(252, min_periods=63).max() - 1) * 100
        df["NQ_Return_5d"] = nq.pct_change(5) * 100
    
    # 3. VIX-SPX correlation (rolling 21-day)
    if "SPX_Close" in df.columns:
        spx_ret = np.log(df["SPX_Close"] / df["SPX_Close"].shift(1))
        vix_ret = df.get("VIX_LogReturn", np.log(df["VIX_Spot"] / df["VIX_Spot"].shift(1)))
        df["VIX_SPX_Corr_21d"] = vix_ret.rolling(21, min_periods=10).corr(spx_ret)
    
    return df


# ============================================================
# HMM INPUT FEATURES
# ============================================================

def add_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare specific features optimized for the 3-state HMM.
    
    Based on literature review:
    - Primary: VIX log returns (captures volatility clustering)
    - Secondary: Term structure slope (captures forward expectations)
    
    Both are standardized to zero mean, unit variance for HMM stability.
    """
    # Primary: VIX log return (already computed, just standardize)
    if "VIX_LogReturn" in df.columns:
        lr = df["VIX_LogReturn"]
        # Expanding standardization (backward-looking only)
        lr_mean = lr.expanding(min_periods=21).mean()
        lr_std = lr.expanding(min_periods=21).std()
        df["HMM_VIX_LogReturn_Std"] = (lr - lr_mean) / lr_std
    
    # Secondary: Term structure slope
    if "TS_Slope_Pct" in df.columns:
        ts = df["TS_Slope_Pct"]
        ts_mean = ts.expanding(min_periods=21).mean()
        ts_std = ts.expanding(min_periods=21).std()
        df["HMM_TS_Slope_Std"] = (ts - ts_mean) / ts_std
    
    return df


```


## FILE: regime/hmm_classifier.py
```python

"""
regime/hmm_classifier.py
========================
3-state Gaussian HMM for volatility regime detection.

States:
    0 = LOW_VOL:    VIX ~12-18, low daily vol, steep contango
    1 = TRANSITION:  VIX ~18-25, moderate vol, flattening curve
    2 = HIGH_VOL:    VIX ~25+, high vol, backwardation

Key design decisions (from literature):
    - 3 states (Goutte et al. 2017) — better than 2 for VIX specifically
    - Full covariance (captures cross-feature dynamics)
    - Walk-forward: retrain quarterly on expanding window
    - Label assignment: post-hoc by variance (resolves label-switching)
    - Debounce: 3-day minimum holding to avoid whipsaw (Nystrup et al. 2020)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from enum import IntEnum
import logging
import warnings

logger = logging.getLogger(__name__)

# Try importing hmmlearn
try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    logger.warning("hmmlearn not installed. pip install hmmlearn")

from config.settings import REGIME


class VolRegime(IntEnum):
    """Volatility regime labels."""
    LOW_VOL = 0
    TRANSITION = 1
    HIGH_VOL = 2


class RegimeClassifier:
    """
    Walk-forward 3-state Gaussian HMM for volatility regime detection.
    
    Usage:
        clf = RegimeClassifier()
        df = clf.fit_predict(df)  # Walk-forward: train → predict → retrain → predict
        
        # Or for real-time inference:
        clf.fit(training_df)
        regime, probs = clf.predict_realtime(new_observation)
    """
    
    def __init__(self, config: object = REGIME):
        self.config = config
        self.model = None
        self.label_map = None  # Maps HMM state IDs → VolRegime labels
        self._fitted = False
    
    def fit_predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Walk-forward regime classification.
        
        1. Use first `training_window_days` for initial fit
        2. Predict forward `refit_frequency_days`
        3. Expand training window, refit, predict next chunk
        4. Repeat until end of data
        
        Args:
            df: DataFrame with HMM feature columns
            feature_cols: Columns to use as HMM observations.
                         Default: ["HMM_VIX_LogReturn_Std", "HMM_TS_Slope_Std"]
        
        Returns:
            df with added columns: Regime, Regime_Prob_LowVol, Regime_Prob_Transition, Regime_Prob_HighVol
        """
        if not HMMLEARN_AVAILABLE:
            raise RuntimeError("hmmlearn required. pip install hmmlearn")
        
        if feature_cols is None:
            feature_cols = ["HMM_VIX_LogReturn_Std", "HMM_TS_Slope_Std"]
        
        # Validate features exist
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        
        df = df.copy()
        n = len(df)
        
        # Initialize output columns
        df["Regime"] = np.nan
        df["Regime_Prob_LowVol"] = np.nan
        df["Regime_Prob_Transition"] = np.nan
        df["Regime_Prob_HighVol"] = np.nan
        df["Regime_Raw"] = np.nan  # Before debouncing
        
        from config.settings import BACKTEST
        train_window = BACKTEST.training_window_days
        refit_freq = BACKTEST.refit_frequency_days
        min_train = BACKTEST.min_training_days
        
        if n < min_train:
            logger.warning(f"Not enough data ({n} rows) for minimum training ({min_train}). Returning NaN regimes.")
            return df
        
        # Walk-forward loop
        fit_start = 0
        predict_start = max(min_train, train_window)
        refit_count = 0
        
        while predict_start < n:
            predict_end = min(predict_start + refit_freq, n)
            
            # Extract training data (expanding window from start)
            train_data = df.iloc[fit_start:predict_start][feature_cols].dropna()
            
            if len(train_data) < min_train:
                predict_start = predict_end
                continue
            
            # Fit HMM
            X_train = train_data.values
            self._fit_hmm(X_train)
            refit_count += 1
            
            # Predict on the next chunk (and all prior data for consistency)
            predict_data = df.iloc[:predict_end][feature_cols].dropna()
            X_predict = predict_data.values
            
            if len(X_predict) == 0:
                predict_start = predict_end
                continue
            
            states, posteriors = self._predict(X_predict)
            
            # Map HMM states to VolRegime labels
            mapped_states = self._map_labels(states, X_train)
            
            # Write predictions for the current chunk only
            chunk_indices = df.iloc[predict_start:predict_end].index
            chunk_mask = predict_data.index.isin(chunk_indices)
            
            for i, idx in enumerate(predict_data.index):
                if idx in chunk_indices:
                    row_idx = df.index.get_loc(idx)
                    df.iloc[row_idx, df.columns.get_loc("Regime_Raw")] = mapped_states[i]
                    df.iloc[row_idx, df.columns.get_loc("Regime_Prob_LowVol")] = posteriors[i, self._get_hmm_state(VolRegime.LOW_VOL)]
                    df.iloc[row_idx, df.columns.get_loc("Regime_Prob_Transition")] = posteriors[i, self._get_hmm_state(VolRegime.TRANSITION)]
                    df.iloc[row_idx, df.columns.get_loc("Regime_Prob_HighVol")] = posteriors[i, self._get_hmm_state(VolRegime.HIGH_VOL)]
            
            predict_start = predict_end
        
        # Apply debouncing
        df["Regime"] = self._debounce_regimes(
            df["Regime_Raw"].values,
            min_holding=self.config.min_regime_holding_days,
        )
        
        logger.info(f"Walk-forward complete: {refit_count} refits, {n - min_train} predictions")
        
        # Log regime distribution
        regime_counts = df["Regime"].dropna().value_counts()
        for regime_val, count in regime_counts.items():
            regime_name = VolRegime(int(regime_val)).name if not np.isnan(regime_val) else "NaN"
            logger.info(f"  {regime_name}: {count} days ({count/len(df)*100:.1f}%)")
        
        return df
    
    def _fit_hmm(self, X: np.ndarray):
        """
        Fit 3-state Gaussian HMM with multiple random restarts.
        Select the model with highest log-likelihood.
        """
        best_score = -np.inf
        best_model = None
        
        n_starts = self.config.n_random_starts
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for i in range(n_starts):
                model = GaussianHMM(
                    n_components=self.config.n_states,
                    covariance_type=self.config.covariance_type,
                    n_iter=self.config.n_iter,
                    tol=self.config.tol,
                    random_state=i * 42,
                )
                
                try:
                    model.fit(X)
                    score = model.score(X)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                except Exception as e:
                    continue
        
        if best_model is None:
            raise RuntimeError("HMM fitting failed on all random starts")
        
        self.model = best_model
        self._fitted = True
    
    def _predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict states and posterior probabilities.
        
        Returns:
            states: (n_samples,) array of state indices
            posteriors: (n_samples, n_states) array of posterior probabilities
        """
        states = self.model.predict(X)
        posteriors = self.model.predict_proba(X)
        return states, posteriors
    
    def _map_labels(self, states: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """
        Map HMM state indices to VolRegime labels.
        
        Strategy: Assign labels by the VARIANCE of VIX log returns in each state.
        - Lowest variance → LOW_VOL
        - Highest variance → HIGH_VOL
        - Middle → TRANSITION
        
        This resolves the label-switching problem inherent in HMMs.
        """
        state_variances = {}
        for s in range(self.config.n_states):
            mask = states[:len(X_train)] == s
            if mask.sum() > 0:
                # Use variance of first feature (VIX log return)
                state_variances[s] = X_train[mask, 0].var()
            else:
                state_variances[s] = 0.0
        
        # Sort states by variance
        sorted_states = sorted(state_variances.keys(), key=lambda s: state_variances[s])
        
        # Build mapping: HMM state → VolRegime
        self.label_map = {
            sorted_states[0]: VolRegime.LOW_VOL,
            sorted_states[1]: VolRegime.TRANSITION,
            sorted_states[2]: VolRegime.HIGH_VOL,
        }
        
        # Apply mapping
        mapped = np.array([self.label_map.get(s, VolRegime.TRANSITION) for s in states])
        return mapped
    
    def _get_hmm_state(self, regime: VolRegime) -> int:
        """Get the HMM state index for a given VolRegime label."""
        if self.label_map is None:
            return int(regime)
        
        reverse_map = {v: k for k, v in self.label_map.items()}
        return reverse_map.get(regime, int(regime))
    
    @staticmethod
    def _debounce_regimes(
        raw_regimes: np.ndarray,
        min_holding: int = 3,
    ) -> np.ndarray:
        """
        Debounce regime transitions to avoid whipsaw.
        
        If a regime change lasts fewer than `min_holding` days, 
        revert to the previous regime.
        
        Based on Nystrup et al. (2020) finding that real-time HMMs
        produce ~2x as many switches as in-sample.
        """
        result = raw_regimes.copy()
        n = len(result)
        
        if n == 0:
            return result
        
        # Find regime change points
        i = 0
        while i < n:
            if np.isnan(result[i]):
                i += 1
                continue
            
            # Find the end of this regime run
            j = i + 1
            while j < n and (result[j] == result[i] or np.isnan(result[j])):
                j += 1
            
            # If this run is too short, revert to previous regime
            run_length = j - i
            if run_length < min_holding and i > 0:
                # Find the previous non-NaN regime
                prev_regime = np.nan
                for k in range(i - 1, -1, -1):
                    if not np.isnan(result[k]):
                        prev_regime = result[k]
                        break
                
                if not np.isnan(prev_regime):
                    result[i:j] = prev_regime
            
            i = j
        
        return result
    
    def predict_realtime(self, observation: np.ndarray) -> Tuple[VolRegime, np.ndarray]:
        """
        Single-observation real-time prediction.
        Uses the forward algorithm for efficient online updating.
        
        Args:
            observation: (n_features,) array for one day
            
        Returns:
            regime: VolRegime enum value
            probs: (n_states,) posterior probabilities
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit_predict() first.")
        
        X = observation.reshape(1, -1)
        state = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        
        regime = self.label_map.get(state, VolRegime.TRANSITION)
        return regime, probs
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """Return the estimated transition matrix with readable labels."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        
        transmat = self.model.transmat_
        labels = [VolRegime(self.label_map[i]).name for i in range(self.config.n_states)]
        
        return pd.DataFrame(transmat, index=labels, columns=labels)
    
    def get_regime_params(self) -> Dict:
        """Return estimated parameters for each regime."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        
        params = {}
        for hmm_state in range(self.config.n_states):
            regime = self.label_map[hmm_state]
            params[VolRegime(regime).name] = {
                "mean": self.model.means_[hmm_state].tolist(),
                "covariance": self.model.covars_[hmm_state].tolist(),
                "stationary_prob": self.model.get_stationary_distribution()[hmm_state],
                "self_transition_prob": self.model.transmat_[hmm_state, hmm_state],
                "expected_duration_days": 1 / (1 - self.model.transmat_[hmm_state, hmm_state]),
            }
        
        return params


```


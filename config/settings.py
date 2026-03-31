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
    max_concurrent_positions: int = 3       # Max 3 open spreads at once
    
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
        "term_structure": 0.30,     # VIX/VIX3M ratio → steep contango = bullish
        "vix_percentile": 0.20,     # VIX level vs 1yr percentile → low = bullish
        "vvix_level": 0.20,         # VVIX → low = cheap options
        "cot_positioning": 0.15,    # COT net short → extreme = bullish contrarian
        "vrp_signal": 0.15,         # Variance risk premium → large = headwind
    })
    
    # Entry threshold
    entry_score_threshold: float = 0.65     # Composite score > 0.65 → consider entry
    
    # Individual indicator thresholds
    term_structure_contango_threshold: float = 0.92   # VIX/VIX3M < 0.92 = strong contango
    vix_low_percentile: int = 30                      # VIX below 30th percentile of 1yr = cheap
    vvix_cheap_threshold: float = 85.0                # VVIX < 85 = cheap VIX options
    vvix_divergence_threshold: float = 105.0          # High VVIX + low VIX = institutional tell
    cot_extreme_percentile: int = 90                  # Net short above 90th pctl = extreme
    vrp_threshold: float = 5.0                        # VIX - RV > 5 = large headwind
    
    # Exit rules
    profit_target_pct: float = 0.50         # Close at 50% of max profit
    time_stop_dte: int = 21                 # Close if DTE <= 21 and not profitable
    regime_exit: bool = True                # Close if regime shifts to high_vol (already profitable)
    pre_settlement_close_dte: int = 1       # Close 1 day before settlement
    stop_loss_pct: float = 0.70             # Close if spread loses 70% of entry value

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
    
    # Regime-dependent spread widths
    spread_widths: Dict[str, int] = field(default_factory=lambda: {
        "low_vol": 5,           # 5-point spread in low vol
        "transition": 5,        # 5-point spread in transition
        "high_vol": 0,          # No new entries in high vol
    })
    
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

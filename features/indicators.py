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
    
    # 9. VIX 5-day, 10-day and 20-day moving averages
    df["VIX_MA5"] = vix.rolling(5).mean()
    df["VIX_SMA10"] = vix.rolling(10, min_periods=5).mean()
    df["VIX_MA20"] = vix.rolling(20).mean()

    # 10. VIX SMA10 momentum slope: (SMA10_today - SMA10_5d_ago) / SMA10_5d_ago
    # Negative = VIX declining on a smoothed basis; used for post-spike recovery filter
    sma10_lag = df["VIX_SMA10"].shift(5)
    df["VIX_SMA10_Slope_5d"] = (df["VIX_SMA10"] - sma10_lag) / sma10_lag
    
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
    Weekly data -> forward-fill to daily for alignment.

    CFTC release lag: positions are measured Tuesday, published Friday
    3:30 PM ET. Shift the index from Tuesday to Friday so that ffill
    makes data available no earlier than the release date.
    """
    # Enforce CFTC release lag: Tuesday as-of -> Friday publication
    cot_df = cot_df.copy()
    # Shift each Tuesday date forward to the following Friday (+3 calendar days).
    # If the as-of date is not a Tuesday (e.g. holiday-adjusted), shift to the
    # next Friday anyway (offset to weekday 4 = Friday).
    shifted_dates = cot_df.index + pd.offsets.Week(weekday=4)
    cot_df.index = shifted_dates

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

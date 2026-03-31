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
        
        # Get regime-specific spread width
        regime_name = regime.name.lower()
        spread_width = self.config.spread_widths.get(regime_name, 5)
        
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
        Select long strike targeting ~40 delta.
        
        Approximation: For VIX options, 40-delta is roughly ATM to slightly OTM.
        When IV is not available, use futures level as ATM proxy.
        """
        if iv is not None and iv > 0:
            # Delta-based: ATM ≈ 50Δ, 40Δ ≈ slightly OTM
            # Rough approximation: strike = futures + IV * sqrt(T/365) * delta_offset
            t = dte / 365
            # 40Δ is approximately 0.25 standard deviations OTM
            offset = iv / 100 * np.sqrt(t) * futures * 0.25
            target = futures - offset  # OTM = below futures for calls
        else:
            # Without IV, use futures as ATM, go 1-2 strikes OTM
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
        """Estimate option delta using Black-76."""
        if iv is None or iv <= 0 or dte <= 0:
            # Rough approximation
            moneyness = (futures - strike) / futures
            return np.clip(0.5 + moneyness * 2, 0.05, 0.95)
        
        from scipy.stats import norm
        
        t = dte / 365
        sigma = iv / 100
        
        if t <= 0 or sigma <= 0:
            return 1.0 if futures > strike else 0.0
        
        d1 = (np.log(futures / strike) + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
        return norm.cdf(d1)

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
        
        # Weighted sum with per-row normalization.
        # Rows where a sub-score is NaN should be divided only by the
        # weight of the non-NaN signals on that row, not the global total.
        df["Signal_Score"] = 0.0
        df["_row_weight"] = 0.0

        for signal_name, col_name in score_cols.items():
            if col_name in df.columns:
                w = weights.get(signal_name, 0.0)
                mask = df[col_name].notna()
                df.loc[mask, "Signal_Score"] += df.loc[mask, col_name] * w
                df.loc[mask, "_row_weight"] += w

        # Normalize by per-row available weight
        valid = df["_row_weight"] > 0
        df.loc[valid, "Signal_Score"] /= df.loc[valid, "_row_weight"]
        df.loc[~valid, "Signal_Score"] = np.nan
        df.drop(columns=["_row_weight"], inplace=True)
        
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

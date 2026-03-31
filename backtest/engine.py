"""
backtest/engine.py
==================
Walk-forward backtest engine for VIX bull call spread strategy.

Handles:
    - Position lifecycle: entry → management → exit
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
            
            # 3. Check entry signals (if capacity available)
            if row.get("Signal_Entry", False):
                self._try_entry(row, date_str)
            
            # 4. Record daily P&L (change in total portfolio value)
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
    
    def _try_entry(self, row: pd.Series, date_str: str):
        """Attempt to open a new position if capacity allows."""
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
        """Check exit conditions for all open positions."""
        regime = row.get("Regime", None)
        if regime is not None and not np.isnan(regime):
            current_regime = int(regime)
        else:
            current_regime = VolRegime.TRANSITION
        
        positions_to_close = []
        
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
            )
            
            if decision != SignalDecision.NO_SIGNAL:
                positions_to_close.append((pos, decision, date_str))
        
        for pos, decision, date_str in positions_to_close:
            self._close_position(pos, date_str, decision.name)
    
    def _close_position(self, pos: Position, date_str: str, reason: str):
        """Close a position and move to closed list."""
        # Apply exit slippage
        exit_price = pos.current_price * (1 - self.config.exit_slippage_pct)
        exit_price -= self.config.commission_per_contract * 2 / 100
        
        pos.exit_date = date_str
        pos.exit_price = round(max(exit_price, 0), 2)
        pos.exit_reason = reason
        pos.current_pnl = pos.exit_price - pos.entry_price
        
        self.closed_positions.append(pos)
        self.positions.remove(pos)
        
        pnl_sign = "+" if pos.current_pnl >= 0 else ""
        logger.debug(
            f"EXIT #{pos.id} on {date_str} ({reason}): "
            f"C{pos.long_strike}/C{pos.short_strike} "
            f"PnL: {pnl_sign}${pos.current_pnl:.2f}"
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
            
            # Sharpe (annualized, daily P&L is already in change form)
            active_pnl = daily_pnl[daily_pnl != 0]
            if len(active_pnl) > 1 and active_pnl.std() > 0:
                result.sharpe_ratio = active_pnl.mean() / active_pnl.std() * np.sqrt(252)
        
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
                f"  #{t['id']:3d} | {t['entry_date']} → {t['exit_date']} | "
                f"C{t['long_strike']}/C{t['short_strike']} | "
                f"${t['entry_price']:.2f} → ${t['exit_price']:.2f} | "
                f"{sign}${t['pnl']:.2f} ({sign}{t['pnl_pct']:.0f}%) | "
                f"{t['exit_reason']} | {t['regime_at_entry']}"
            )
        
        print("=" * 60)

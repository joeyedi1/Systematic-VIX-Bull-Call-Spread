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
from datetime import datetime, timedelta, date
import logging
import calendar

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
        self._pending_entry = False  # Latched on signal day, executed next bar
        self._expiry_calendar: List[date] = []  # Built on first run()

    # ------------------------------------------------------------------
    # VIX expiry calendar helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _third_friday(year: int, month: int) -> date:
        """Third Friday of a given month."""
        # First day of the month, find first Friday, then add 2 weeks
        cal = calendar.monthcalendar(year, month)
        # calendar weeks: Mon=0 .. Sun=6.  Friday = 4
        fridays = [week[4] for week in cal if week[4] != 0]
        return date(year, month, fridays[2])

    @classmethod
    def _build_vix_expiry_calendar(cls, start_year: int, end_year: int) -> List[date]:
        """
        Generate monthly VIX expiry dates.

        CBOE rule: VIX expiry = Wednesday that is exactly 30 calendar days
        before the third Friday of the *following* calendar month.
        (30 days before a Friday is always a Wednesday.)
        """
        expiries = []
        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                # "Following month" for month m
                next_m = m + 1 if m < 12 else 1
                next_y = y if m < 12 else y + 1
                third_fri = cls._third_friday(next_y, next_m)
                expiry = third_fri - timedelta(days=30)
                expiries.append(expiry)
        return sorted(expiries)

    def _find_target_expiry(self, entry_date: date) -> Optional[date]:
        """
        Find the first VIX expiry within the configured DTE window.

        Returns the expiry date or None if nothing falls in [dte_min, dte_max].
        """
        dte_min, dte_max = self.strike_sel.config.dte_range
        for exp in self._expiry_calendar:
            dte = (exp - entry_date).days
            if dte < dte_min:
                continue
            if dte > dte_max:
                return None      # All subsequent expiries are further out
            return exp
        return None

    def _get_ux_column(self, current_date: date, target_expiry: date) -> Optional[str]:
        """
        Determine which generic UXn column matches a target expiry.

        On any trading day, UX1 = first expiry strictly after today,
        UX2 = second expiry after today, etc.  Returns e.g. "UX2".
        """
        n = 0
        for exp in self._expiry_calendar:
            if exp <= current_date:
                continue          # Already expired / expiring today
            n += 1
            if exp == target_expiry:
                return f"UX{n}" if n <= 9 else None
        return None               # Target expiry not found in calendar
    
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

        # Build VIX expiry calendar covering the full backtest window + margin
        start_yr = df.index.min().year
        end_yr = df.index.max().year + 1
        self._expiry_calendar = self._build_vix_expiry_calendar(start_yr, end_yr)

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

            # 4. Execute pending entry from PREVIOUS day's signal at TODAY's prices
            #    All guards (capacity, regime, cooldown) checked at execution time.
            if self._pending_entry:
                self._pending_entry = False
                self._try_entry(row, date_str)

            # 5. Latch new entry signal for NEXT bar execution
            if row.get("Signal_Entry", False):
                self._pending_entry = True
            
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
        # Cooldown logic — scan ALL recent closes, not just the last one.
        # With max_concurrent_positions=2, two positions can close on the
        # same bar.  If the winner closes last, checking only [-1] would
        # miss the loser and bypass cooldown.
        if self.closed_positions:
            cooldown_type = getattr(self.config, 'cooldown_type', 'calendar')

            if cooldown_type == "signal_reset":
                # Blocked if any loss has not yet been rearmed.
                # _score_rearmed is set False on every loss close
                # and True only when Signal_Entry goes False on a
                # subsequent bar (tracked in _update_cooldown_rearm).
                if not getattr(self, '_score_rearmed', True):
                    return

            else:  # calendar
                current = datetime.strptime(date_str, "%Y-%m-%d")
                cooldown_days = self.config.cooldown_after_loss_days
                # Check every loss within the calendar window
                for pos in reversed(self.closed_positions):
                    exit_dt = datetime.strptime(pos.exit_date, "%Y-%m-%d")
                    gap = (current - exit_dt).days
                    if gap >= cooldown_days:
                        break  # Older closes are outside the window
                    if pos.current_pnl < 0:
                        return

        if len(self.positions) >= self.config.max_concurrent_positions:
            return
        
        regime = row.get("Regime", None)
        if regime is None or np.isnan(regime):
            return
        
        regime_enum = VolRegime(int(regime))
        
        # Find actual VIX expiry within the DTE window
        entry_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
        target_expiry = self._find_target_expiry(entry_dt)
        if target_expiry is None:
            return
        actual_dte = (target_expiry - entry_dt).days

        # Resolve the expiry-matched VIX future (UXn)
        ux_col = self._get_ux_column(entry_dt, target_expiry)
        if ux_col is None:
            return
        vix_futures = row.get(ux_col, None)
        if vix_futures is None or np.isnan(vix_futures) or vix_futures <= 0:
            return

        # Get VVIX for IV proxy (VVIX ~ IV of VIX options)
        vix_iv = row.get("VVIX", None)
        if vix_iv is not None and (np.isnan(vix_iv) or vix_iv <= 0):
            vix_iv = None

        # Select strikes using real DTE and expiry-matched future
        selection = self.strike_sel.select(
            vix_futures=vix_futures,
            regime=regime_enum,
            vix_iv=vix_iv,
            dte=actual_dte,
        )

        if selection is None:
            return

        # Apply slippage to entry cost
        entry_cost = selection.estimated_cost * (1 + self.config.entry_slippage_pct)
        entry_cost += self.config.commission_per_contract * 2 / 100  # 2 legs, per $100 multiplier

        expiry_str = target_expiry.strftime("%Y-%m-%d")

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
            dte_at_entry=actual_dte,
        )
        
        self.positions.append(pos)
        self.next_id += 1
        
        logger.debug(
            f"ENTRY #{pos.id} on {date_str}: "
            f"C{pos.long_strike}/C{pos.short_strike} @ ${pos.entry_price:.2f}"
        )
    
    def _update_positions(self, row: pd.Series, date_str: str):
        """Update current price and P&L for all open positions."""
        current_dt = datetime.strptime(date_str, "%Y-%m-%d").date()

        for pos in self.positions:
            if not pos.is_open:
                continue

            expiry_dt = datetime.strptime(pos.expiry_date, "%Y-%m-%d").date()
            pos.current_dte = max(0, (expiry_dt - current_dt).days)

            # Resolve the expiry-matched VIX future for this position
            ux_col = self._get_ux_column(current_dt, expiry_dt)
            if ux_col is not None:
                vix_futures = row.get(ux_col, None)
            else:
                vix_futures = None

            # Fallback: if expiry-matched future unavailable (e.g. past
            # last calendar entry), use UX1 then VIX_Spot
            if vix_futures is None or np.isnan(vix_futures):
                vix_futures = row.get("UX1", row.get("VIX_Spot", None))
            if vix_futures is None or np.isnan(vix_futures):
                continue

            # Simple valuation: intrinsic + declining time value
            intrinsic = self._calc_intrinsic(vix_futures, pos.long_strike, pos.short_strike)

            if pos.current_dte > 0:
                time_ratio = np.sqrt(pos.current_dte / pos.dte_at_entry) if pos.dte_at_entry > 0 else 0
                initial_time_value = max(pos.entry_price - self._calc_intrinsic(
                    vix_futures, pos.long_strike, pos.short_strike
                ), 0)
                time_value = initial_time_value * time_ratio * 0.5  # Conservative
                pos.current_price = intrinsic + time_value
            else:
                # At expiry: intrinsic only
                pos.current_price = intrinsic

            if pos.half_closed:
                pos.current_pnl = pos.half_closed_pnl + (pos.current_price - pos.entry_price) / 2
            else:
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

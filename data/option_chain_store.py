"""
data/option_chain_store.py
==========================
Lazy-loading cache of VIX option chain NBBO data for backtest MTM pricing.

Data files: outputs/cache/vix_option_chains/vix_options_YYYY_MM.parquet
Columns:    C{K}_Bid, C{K}_Ask, C{K}_Mid, C{K}_Last   (K = 10..35)

Usage:
    store = OptionChainStore()
    if store.available():
        mid = store.get_spread_mid("2024-03-20", "2024-02-15", 14, 18)
        # Returns float spread mid, or None → caller falls back to synthetic model
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class OptionChainStore:
    """
    Manages loading and lookup of VIX option chain parquet files.

    Each file covers one monthly expiry.  Files are loaded on first access
    and held in memory for the lifetime of the store instance.

    Designed for no-fail operation: any missing file or missing data returns
    None, and the caller (BacktestEngine) falls back to the synthetic model.
    """

    DEFAULT_DIR = Path("outputs/cache/vix_option_chains")

    def __init__(self, chain_dir: Optional[str] = None):
        self.chain_dir = Path(chain_dir) if chain_dir else self.DEFAULT_DIR
        # key: "YYYY-MM", value: DataFrame or None (None = file checked, not found)
        self._cache: Dict[str, Optional[pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def available(self) -> bool:
        """True if at least one chain parquet exists in the chain directory."""
        return (
            self.chain_dir.exists()
            and any(self.chain_dir.glob("vix_options_*.parquet"))
        )

    def get_spread_mid(
        self,
        expiry_date: str,   # "YYYY-MM-DD"
        obs_date: str,      # "YYYY-MM-DD"
        long_strike: int,
        short_strike: int,
    ) -> Optional[float]:
        """
        Return the mid-price of the bull call spread on obs_date.

        Spread mid = C{long_strike}_Mid  −  C{short_strike}_Mid.
        Used for daily unrealised mark-to-market.

        Returns None when real data is unavailable → caller uses synthetic.
        """
        row = self._get_row(expiry_date, obs_date)
        if row is None:
            return None

        long_mid  = row.get(f"C{long_strike}_Mid",  np.nan)
        short_mid = row.get(f"C{short_strike}_Mid", np.nan)

        if pd.isna(long_mid) or pd.isna(short_mid):
            return None
        if float(long_mid) <= 0:
            return None

        long_mid, short_mid = float(long_mid), float(short_mid)

        if long_mid < short_mid:
            return None                              # Inverted — bad data
        spread = long_mid - short_mid
        if spread > (short_strike - long_strike):
            return None                              # Exceeds theoretical max

        return max(spread, 0.0)

    def get_entry_cost(
        self,
        expiry_date: str,   # "YYYY-MM-DD"
        obs_date: str,      # "YYYY-MM-DD"
        long_strike: int,
        short_strike: int,
    ) -> Optional[float]:
        """
        Return the real market entry cost for a bull call spread.

        Entry cost = ask(long_strike) − bid(short_strike).
        This is the worst-case fill when BUYING the spread.

        Returns None when real quotes are unavailable → caller falls back to
        the synthetic StrikeSelector estimate.  Never raises.
        """
        row = self._get_row(expiry_date, obs_date)
        if row is None:
            return None

        long_ask  = row.get(f"C{long_strike}_Ask",  np.nan)
        short_bid = row.get(f"C{short_strike}_Bid", np.nan)

        if pd.isna(long_ask) or pd.isna(short_bid):
            return None
        long_ask, short_bid = float(long_ask), float(short_bid)

        if long_ask <= 0:
            return None   # Long leg has no offer — can't enter

        cost = long_ask - short_bid
        if cost <= 0:
            return None   # Would be a credit — not this strategy
        if cost > (short_strike - long_strike):
            return None   # Exceeds spread width — bad data

        return cost

    def get_exit_proceeds(
        self,
        expiry_date: str,   # "YYYY-MM-DD"
        obs_date: str,      # "YYYY-MM-DD"
        long_strike: int,
        short_strike: int,
    ) -> Optional[float]:
        """
        Return the real market proceeds when CLOSING a bull call spread.

        Proceeds = bid(long_strike) − ask(short_strike).
        This is the worst-case fill when SELLING the spread.

        Proceeds can legitimately be near zero for deeply OTM spreads.
        Returns None when real quotes are unavailable → caller uses
        pos.current_price (the last mid mark) as fallback.  Never raises.
        """
        row = self._get_row(expiry_date, obs_date)
        if row is None:
            return None

        long_bid  = row.get(f"C{long_strike}_Bid",  np.nan)
        short_ask = row.get(f"C{short_strike}_Ask", np.nan)

        if pd.isna(long_bid) or pd.isna(short_ask):
            return None
        long_bid, short_ask = float(long_bid), float(short_ask)

        proceeds = long_bid - short_ask
        if proceeds > (short_strike - long_strike):
            return None   # Exceeds spread width — bad data

        return max(proceeds, 0.0)

    def get_call_ask(
        self,
        expiry_date: str,
        obs_date: str,
        strike: int,
    ) -> Optional[float]:
        """Ask price of a single call leg — used for market-order entry."""
        row = self._get_row(expiry_date, obs_date)
        if row is None:
            return None
        val = row.get(f"C{strike}_Ask", np.nan)
        if pd.isna(val) or float(val) <= 0:
            return None
        return float(val)

    def get_call_bid(
        self,
        expiry_date: str,
        obs_date: str,
        strike: int,
    ) -> Optional[float]:
        """Bid price of a single call leg — used for market-order exit."""
        row = self._get_row(expiry_date, obs_date)
        if row is None:
            return None
        val = row.get(f"C{strike}_Bid", np.nan)
        if pd.isna(val):
            return None
        return max(float(val), 0.0)

    def get_call_mid(
        self,
        expiry_date: str,
        obs_date: str,
        strike: int,
    ) -> Optional[float]:
        """Mid price of a single call leg — used for MTM and limit-order pricing."""
        row = self._get_row(expiry_date, obs_date)
        if row is None:
            return None
        bid_val = row.get(f"C{strike}_Bid", np.nan)
        ask_val = row.get(f"C{strike}_Ask", np.nan)
        mid_val = row.get(f"C{strike}_Mid", np.nan)
        if pd.isna(mid_val) or float(mid_val) <= 0:
            return None
        mid = float(mid_val)
        # Sanity check: reject if ask is pathologically wide relative to bid
        # (stale/stuck Bloomberg quote — e.g. Feb 27 2024: bid=1.54, ask=25.00 → mid=13.27)
        if not pd.isna(bid_val) and not pd.isna(ask_val):
            bid, ask = float(bid_val), float(ask_val)
            if bid > 0 and ask > 0:
                if ask > bid * 6:   # ask > 6x bid signals stale data
                    return None
                if not (bid - 0.01 <= mid <= ask + 0.01):
                    return None
        return mid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_row(self, expiry_date: str, obs_date: str) -> Optional[pd.Series]:
        """Load chain and return the row for obs_date. Returns None if missing."""
        chain = self._load_chain(expiry_date)
        if chain is None:
            return None

        ts = pd.Timestamp(obs_date)
        if ts not in chain.index:
            return None

        row = chain.loc[ts]
        # Guard against duplicate index (shouldn't happen for daily data)
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    def _load_chain(self, expiry_date: str) -> Optional[pd.DataFrame]:
        """Load (and memory-cache) the chain parquet for a given expiry."""
        key = expiry_date[:7]   # "YYYY-MM"

        if key in self._cache:
            return self._cache[key]

        fname = f"vix_options_{key[:4]}_{key[5:7]}.parquet"
        fpath = self.chain_dir / fname

        if not fpath.exists():
            self._cache[key] = None
            return None

        try:
            df = pd.read_parquet(fpath)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            self._cache[key] = df
            logger.debug(f"Loaded option chain: {fname} ({len(df)} rows)")
            return df
        except Exception as exc:
            logger.warning(f"Failed to load option chain {fname}: {exc}")
            self._cache[key] = None
            return None

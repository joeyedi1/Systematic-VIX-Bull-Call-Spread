"""
data/bloomberg_fetcher.py
=========================
Bloomberg historical data pipeline via blpapi.
Fetches VIX spot, futures curve, VVIX, VIX3M, and VIX options data.

Usage:
    from data.bloomberg_fetcher import BloombergDataPipeline
    pipeline = BloombergDataPipeline()
    df = pipeline.fetch_all(start_date="20220103", end_date="20260318")
    pipeline.close()
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try importing blpapi — fail gracefully for offline development
try:
    import blpapi
    BLOOMBERG_AVAILABLE = True
except ImportError:
    BLOOMBERG_AVAILABLE = False
    logger.warning("blpapi not installed. Bloomberg functions will use cached data only.")

from config.settings import (
    VIX_SPOT, VIX3M, VIX6M, VIX9D, VVIX, SKEW,
    VIX_FUTURES, DAILY_FIELDS, SPX, NQ,
    VIX_MONTH_CODES,
)


class BloombergSession:
    """Manages Bloomberg API connection lifecycle."""
    
    def __init__(self):
        self.session = None
        
    def connect(self):
        if not BLOOMBERG_AVAILABLE:
            raise RuntimeError("blpapi not installed. Cannot connect to Bloomberg.")
        
        logger.info("Connecting to Bloomberg Terminal...")
        options = blpapi.SessionOptions()
        options.setServerHost("localhost")
        options.setServerPort(8194)
        
        self.session = blpapi.Session(options)
        if not self.session.start():
            raise ConnectionError("Failed to start Bloomberg session")
        if not self.session.openService("//blp/refdata"):
            raise ConnectionError("Failed to open //blp/refdata service")
        
        logger.info("Connected to Bloomberg.")
        return self
    
    def close(self):
        if self.session:
            self.session.stop()
            self.session = None
            logger.info("Bloomberg session closed.")
    
    def __enter__(self):
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class BloombergDataPipeline:
    """
    Fetches and structures all data needed for the VIX strategy.
    
    Output: A single DataFrame with DatetimeIndex and columns:
        - VIX_Spot, VIX3M, VIX6M, VIX9D, VVIX, SKEW
        - UX1..UX9 (generic VIX futures)
        - SPX_Close, NQ_Close (equity benchmarks)
    """
    
    CACHE_DIR = Path("outputs/cache")
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.bbg = None
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def fetch_all(
        self,
        start_date: str = "20220103",
        end_date: str = None,
        cache_file: str = "vix_strategy_data.parquet",
    ) -> pd.DataFrame:
        """
        Fetch all required data from Bloomberg.
        
        Args:
            start_date: YYYYMMDD format
            end_date: YYYYMMDD format (default: today)
            cache_file: Name of parquet cache file
            
        Returns:
            DataFrame with all indicators, DatetimeIndex
        """
        cache_path = self.CACHE_DIR / cache_file
        
        # Check cache first
        if self.use_cache and cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            df = pd.read_parquet(cache_path)
            
            # Check if cache is fresh enough (within 1 day)
            last_date = df.index.max()
            if end_date:
                target = pd.Timestamp(datetime.strptime(end_date, "%Y%m%d"))
            else:
                target = pd.Timestamp.now().normalize()
            
            if last_date >= target - pd.Timedelta(days=3):
                logger.info(f"Cache is fresh (last: {last_date.date()}). Using cached data.")
                return df
            else:
                logger.info(f"Cache is stale (last: {last_date.date()}, target: {target.date()}). Refreshing...")
        
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        
        # Fetch from Bloomberg
        self.bbg = BloombergSession()
        self.bbg.connect()
        
        try:
            df = self._fetch_and_merge(start_date, end_date)
            
            # Save to cache
            df.to_parquet(cache_path)
            logger.info(f"Data cached to {cache_path} ({len(df)} rows)")
            
            return df
        finally:
            self.bbg.close()
    
    def _fetch_and_merge(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch all tickers and merge into single DataFrame."""
        
        # === 1. VIX Spot & Related Indices ===
        spot_tickers = {
            "VIX_Spot": VIX_SPOT,
            "VIX3M": VIX3M,
            "VIX6M": VIX6M,
            "VIX9D": VIX9D,
            "VVIX": VVIX,
            "SKEW": SKEW,
        }
        
        logger.info("Fetching VIX spot & related indices...")
        spot_df = self._fetch_historical(
            list(spot_tickers.values()),
            ["PX_LAST"],
            start_date, end_date
        )
        spot_df = self._pivot_single_field(spot_df, spot_tickers)
        
        # === 2. VIX Futures Curve ===
        futures_tickers = {f"UX{i}": ticker for i, ticker in VIX_FUTURES.items()}
        
        logger.info("Fetching VIX futures curve (UX1-UX9)...")
        futures_df = self._fetch_historical(
            list(futures_tickers.values()),
            ["PX_LAST", "PX_VOLUME", "OPEN_INT"],
            start_date, end_date
        )
        futures_price_df = self._pivot_single_field(futures_df, futures_tickers, field="PX_LAST")
        futures_vol_df = self._pivot_single_field(futures_df, futures_tickers, field="PX_VOLUME", suffix="_Volume")
        futures_oi_df = self._pivot_single_field(futures_df, futures_tickers, field="OPEN_INT", suffix="_OI")
        
        # === 3. Equity Benchmarks ===
        equity_tickers = {"SPX_Close": SPX, "NQ_Close": NQ}
        
        logger.info("Fetching equity benchmarks...")
        equity_df = self._fetch_historical(
            list(equity_tickers.values()),
            ["PX_LAST"],
            start_date, end_date
        )
        equity_df = self._pivot_single_field(equity_df, equity_tickers)
        
        # === 4. Merge all ===
        logger.info("Merging all data...")
        dfs = [spot_df, futures_price_df, futures_vol_df, futures_oi_df, equity_df]
        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, how="outer")
        
        # Forward-fill small gaps (weekends, holidays already handled by Bloomberg)
        result = result.ffill(limit=3)
        
        # Drop rows where VIX_Spot is missing (non-trading days)
        result = result.dropna(subset=["VIX_Spot"])
        
        logger.info(f"Final dataset: {len(result)} rows, {len(result.columns)} columns")
        logger.info(f"Date range: {result.index.min().date()} to {result.index.max().date()}")
        
        return result
    
    def _fetch_historical(
        self,
        tickers: List[str],
        fields: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Core Bloomberg BDH (historical data) request.
        
        Returns long-format DataFrame: Date, Ticker, Field1, Field2, ...
        """
        service = self.bbg.session.getService("//blp/refdata")
        request = service.createRequest("HistoricalDataRequest")
        
        for ticker in tickers:
            request.append("securities", ticker)
        for f in fields:
            request.append("fields", f)
        
        request.set("startDate", start_date)
        request.set("endDate", end_date)
        request.set("periodicitySelection", "DAILY")
        request.set("nonTradingDayFillOption", "NON_TRADING_WEEKDAYS")
        request.set("nonTradingDayFillMethod", "PREVIOUS_VALUE")
        
        self.bbg.session.sendRequest(request)
        
        records = []
        while True:
            event = self.bbg.session.nextEvent(5000)
            for msg in event:
                if msg.hasElement("securityData"):
                    sec_data = msg.getElement("securityData")
                    ticker = sec_data.getElementAsString("security")
                    
                    if sec_data.hasElement("fieldData"):
                        field_data = sec_data.getElement("fieldData")
                        for i in range(field_data.numValues()):
                            point = field_data.getValueAsElement(i)
                            raw_date = point.getElementAsDatetime("date")
                            
                            row = {
                                "Date": pd.Timestamp(raw_date).strftime("%Y-%m-%d"),
                                "Ticker": ticker,
                            }
                            for f in fields:
                                if point.hasElement(f):
                                    try:
                                        row[f] = point.getElementAsFloat(f)
                                    except Exception:
                                        row[f] = np.nan
                                else:
                                    row[f] = np.nan
                            records.append(row)
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        df = pd.DataFrame(records)
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"])
        
        logger.info(f"  Fetched {len(df)} records for {len(tickers)} tickers")
        return df
    
    def _pivot_single_field(
        self,
        df: pd.DataFrame,
        name_map: Dict[str, str],
        field: str = "PX_LAST",
        suffix: str = "",
    ) -> pd.DataFrame:
        """Pivot long-format Bloomberg data into wide format with readable column names."""
        if df.empty:
            return pd.DataFrame()
        
        # Reverse the name map: Bloomberg ticker → readable name
        reverse_map = {v: k for k, v in name_map.items()}
        
        # Filter to the requested field
        pivot_df = df[["Date", "Ticker", field]].copy()
        pivot_df["Name"] = pivot_df["Ticker"].map(reverse_map)
        pivot_df = pivot_df.dropna(subset=["Name"])
        
        # Pivot
        result = pivot_df.pivot(index="Date", columns="Name", values=field)
        result.index = pd.to_datetime(result.index)
        
        # Add suffix if needed
        if suffix:
            result.columns = [f"{c}{suffix}" for c in result.columns]
        
        return result
    
    def fetch_vix_options_history(
        self,
        expiry_date: str,
        strikes: List[int],
        start_date: str,
        end_date: str = None,
        option_type: str = "C",
    ) -> pd.DataFrame:
        """
        Fetch historical data for specific VIX option contracts.
        
        Args:
            expiry_date: "MM/DD/YY" format (Bloomberg convention)
            strikes: List of strike prices [20, 25, 30, 40]
            start_date: "YYYYMMDD"
            end_date: "YYYYMMDD"
            option_type: "C" for calls, "P" for puts
            
        Returns:
            DataFrame with columns: Date, Strike, PX_LAST, PX_BID, PX_ASK, IVOL_MID, DELTA_MID
        """
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        
        if not self.bbg or not self.bbg.session:
            self.bbg = BloombergSession()
            self.bbg.connect()
        
        tickers = {}
        for strike in strikes:
            ticker = f"VIX US {expiry_date} {option_type}{strike} Index"
            tickers[f"C{strike}" if option_type == "C" else f"P{strike}"] = ticker
        
        fields = ["PX_LAST", "PX_BID", "PX_ASK", "PX_MID", "IVOL_MID", "DELTA_MID",
                   "THETA_MID", "VEGA_MID", "OPEN_INT", "PX_VOLUME"]
        
        logger.info(f"Fetching VIX options: {expiry_date} {option_type}{strikes}")
        raw_df = self._fetch_historical(list(tickers.values()), fields, start_date, end_date)
        
        if raw_df.empty:
            return pd.DataFrame()
        
        # Add strike info
        reverse_map = {v: k for k, v in tickers.items()}
        raw_df["Strike_Label"] = raw_df["Ticker"].map(reverse_map)
        raw_df["Strike"] = raw_df["Strike_Label"].str.extract(r"(\d+)").astype(float)
        
        return raw_df
    
    def close(self):
        """Clean up Bloomberg connection."""
        if self.bbg:
            self.bbg.close()


class OfflineDataLoader:
    """
    Load data from CSV/Parquet files for offline development and backtesting.
    Mirrors the BloombergDataPipeline output format.
    """
    
    def __init__(self, data_dir: str = "outputs/cache"):
        self.data_dir = Path(data_dir)
    
    def load(self, filename: str = "vix_strategy_data.parquet") -> pd.DataFrame:
        """Load cached data."""
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"No cached data at {path}. Run BloombergDataPipeline.fetch_all() first."
            )
        return pd.read_parquet(path)
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load from CSV with Date as index."""
        path = self.data_dir / filename
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        return df
    
    @staticmethod
    def from_existing_dashboard_csv(csv_path: str) -> pd.DataFrame:
        """
        Load data from the existing VIX dashboard CSV format
        and reshape it for the strategy pipeline.
        
        This bridges your existing vix_spread_data.csv into the new system.
        """
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        
        # Map existing columns to new naming convention
        column_map = {}
        for col in df.columns:
            if "VIX_Futures" in col:
                # Extract month prefix
                if col.startswith("Feb_2026"):
                    column_map[col] = "UXG26_Last"
                elif col.startswith("Mar_2026") and "20-40" not in col:
                    column_map[col] = "UXH26_Last"
        
        # Don't rename - keep original columns and add metadata
        df.attrs["source"] = "dashboard_csv"
        df.attrs["original_path"] = csv_path
        
        return df

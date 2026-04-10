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
import calendar
import time
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
    
    def fetch_extended_history(
        self,
        start_date: str = "20100101",
        end_date: str = "20211231",
        cache_file: str = "vix_extended_history_2010_2021.parquet",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch 12-year VIX history (2010-2021) for HMM pre-training.

        Tickers pulled (BDH daily, PX_LAST):
            VIX Index, VIX3M Index (VXV fallback for pre-2018),
            VIX9D Index (~2016+), VVIX Index (~2012+),
            UX1/UX2/UX3 Index, SPX Index.

        Missing-data handling:
          - VIX9D: available from ~2016, NaN-filled before.
          - VVIX:  available from ~2012, NaN-filled before.
          - VIX3M: Bloomberg back-fills VXV history under VIX3M Index;
                   falls back to VXV Index if VIX3M returns empty.

        Saves to: outputs/cache/<cache_file>
        """
        cache_path = self.CACHE_DIR / cache_file

        if not force_refresh and cache_path.exists():
            logger.info(f"Loading extended history cache: {cache_path}")
            return pd.read_parquet(cache_path)

        self.bbg = BloombergSession()
        self.bbg.connect()

        try:
            df = self._fetch_extended_and_merge(start_date, end_date)
            df.to_parquet(cache_path)
            logger.info(
                f"Extended history saved: {cache_path} "
                f"({len(df)} rows, {df.index.min().date()} – {df.index.max().date()})"
            )
            return df
        finally:
            self.bbg.close()

    def _fetch_extended_and_merge(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch all extended-history tickers and merge into one DataFrame."""

        # 1. Core tickers — available for the full 2010-2021 window
        core_map = {
            "VIX_Spot":  "VIX Index",
            "UX1":       "UX1 Index",
            "UX2":       "UX2 Index",
            "UX3":       "UX3 Index",
            "SPX_Close": "SPX Index",
        }
        logger.info("Extended history — fetching core tickers (VIX, UX1-3, SPX)...")
        core_raw = self._fetch_historical(
            list(core_map.values()), ["PX_LAST"], start_date, end_date
        )
        result = self._pivot_single_field(core_raw, core_map)

        # 2. VIX3M — Bloomberg back-fills VXV as VIX3M Index; fall back to VXV if empty
        logger.info("Extended history — fetching VIX3M (VXV fallback if needed)...")
        vix3m_raw = self._fetch_historical(
            ["VIX3M Index"], ["PX_LAST"], start_date, end_date
        )
        if not vix3m_raw.empty:
            vix3m_wide = self._pivot_single_field(vix3m_raw, {"VIX3M": "VIX3M Index"})
        else:
            logger.info("  VIX3M Index returned empty — retrying with VXV Index...")
            vxv_raw = self._fetch_historical(
                ["VXV Index"], ["PX_LAST"], start_date, end_date
            )
            vix3m_wide = (
                self._pivot_single_field(vxv_raw, {"VIX3M": "VXV Index"})
                if not vxv_raw.empty
                else pd.DataFrame()
            )
        if not vix3m_wide.empty:
            result = result.join(vix3m_wide, how="left")
        else:
            logger.warning("  VIX3M/VXV both unavailable — column will be NaN")
            result["VIX3M"] = np.nan

        # 3. Optional tickers — graceful NaN fill when unavailable for early dates
        optional_map = {
            "VIX9D": "VIX9D Index",  # Available from ~2016; NaN before
            "VVIX":  "VVIX Index",   # Available from ~2012; NaN before
        }
        for col, ticker in optional_map.items():
            logger.info(f"Extended history — fetching {col} ({ticker})...")
            opt_raw = self._fetch_historical([ticker], ["PX_LAST"], start_date, end_date)
            if not opt_raw.empty:
                opt_wide = self._pivot_single_field(opt_raw, {col: ticker})
                if not opt_wide.empty:
                    result = result.join(opt_wide, how="left")
                    non_null = int(opt_wide[col].notna().sum()) if col in opt_wide.columns else 0
                    logger.info(f"  {col}: {non_null} non-null rows")
                else:
                    result[col] = np.nan
            else:
                logger.info(f"  {col}: no data returned — filling with NaN")
                result[col] = np.nan

        # 4. Forward-fill small gaps (weekends / holidays already handled by Bloomberg)
        result = result.ffill(limit=3)

        # 5. Drop rows where VIX_Spot is missing (non-trading days)
        result = result.dropna(subset=["VIX_Spot"])

        logger.info(
            f"Extended history: {len(result)} rows, {len(result.columns)} cols | "
            f"{result.index.min().date()} → {result.index.max().date()}"
        )
        return result

    def fetch_option_chains(
        self,
        start_expiry: str = "2023-01",
        end_expiry: str = "2026-03",
        strikes: Optional[List[int]] = None,
        days_before_expiry: int = 60,
        output_dir: str = "outputs/cache/vix_option_chains",
        batch_delay_seconds: float = 0.5,
        test_mode_n: int = 0,
    ) -> List[str]:
        """
        Fetch daily VIX call option NBBO for every monthly expiry in range.

        For each expiry pulls PX_BID / PX_ASK / PX_LAST for calls at
        strikes 10-35, covering the 60 calendar days before expiry.

        Output: one parquet per expiry in output_dir:
            vix_options_YYYY_MM.parquet
        Columns: C{K}_Bid, C{K}_Ask, C{K}_Mid, C{K}_Last  (K = 10..35).
        Existing files are skipped (idempotent).

        Args:
            start_expiry:         First expiry month as "YYYY-MM".
            end_expiry:           Last  expiry month as "YYYY-MM".
            strikes:              Integer strikes to pull (default 10-35).
            days_before_expiry:   Calendar-day look-back per expiry (default 60).
            batch_delay_seconds:  Sleep between expiry requests (rate-limit guard).
            test_mode_n:          If > 0, stop after this many expiries.

        Returns:
            List of parquet file paths written (skipped files excluded).
        """
        if strikes is None:
            strikes = list(range(10, 36))   # C10 … C35

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        sy, sm = int(start_expiry[:4]), int(start_expiry[5:7])
        ey, em = int(end_expiry[:4]),   int(end_expiry[5:7])
        expiries = self._generate_expiry_dates_in_range(sy, sm, ey, em)

        if test_mode_n > 0:
            expiries = expiries[:test_mode_n]
            logger.info(f"TEST MODE: fetching {test_mode_n} of {len(expiries)} expiries")

        logger.info(
            f"Option chain fetch: {len(expiries)} expiries × {len(strikes)} strikes "
            f"(C{strikes[0]}–C{strikes[-1]})"
        )

        self.bbg = BloombergSession()
        self.bbg.connect()

        written: List[str] = []
        try:
            for i, exp_dt in enumerate(expiries):
                tag   = f"{exp_dt.year:04d}_{exp_dt.month:02d}"
                fname = f"vix_options_{tag}.parquet"
                fpath = out_dir / fname

                if fpath.exists():
                    logger.info(f"  [{i+1}/{len(expiries)}] {exp_dt.date()} — cached, skip")
                    continue

                start_dt  = exp_dt - timedelta(days=days_before_expiry)
                bbg_expiry = exp_dt.strftime("%m/%d/%y")   # Bloomberg: MM/DD/YY
                start_str  = start_dt.strftime("%Y%m%d")
                end_str    = exp_dt.strftime("%Y%m%d")

                logger.info(
                    f"  [{i+1}/{len(expiries)}] {exp_dt.date()} ({bbg_expiry}) "
                    f"| {start_str}–{end_str} | {len(strikes)} strikes"
                )

                try:
                    df = self._fetch_one_expiry_chain(
                        bbg_expiry, strikes, start_str, end_str
                    )
                    if df.empty:
                        logger.warning(f"    No data returned for {exp_dt.date()}")
                    else:
                        df.to_parquet(fpath)
                        logger.info(
                            f"    Saved {fname}: {len(df)} rows × {len(df.columns)} cols"
                        )
                        written.append(str(fpath))
                except Exception as exc:
                    logger.error(f"    Fetch failed for {exp_dt.date()}: {exc}")

                if batch_delay_seconds > 0 and i < len(expiries) - 1:
                    time.sleep(batch_delay_seconds)

        finally:
            self.bbg.close()

        logger.info(f"Option chain fetch complete: {len(written)} new files in {out_dir}")
        return written

    def _fetch_one_expiry_chain(
        self,
        bbg_expiry: str,    # "MM/DD/YY"
        strikes: List[int],
        start_date: str,    # "YYYYMMDD"
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch all strikes for one expiry and return wide-format DataFrame.
        Columns: C{K}_Bid, C{K}_Ask, C{K}_Mid, C{K}_Last.
        Missing / illiquid strikes produce NaN columns — no error raised.
        """
        ticker_map: Dict[str, str] = {
            f"C{k}": f"VIX US {bbg_expiry} C{k} Index"
            for k in strikes
        }

        raw = self._fetch_historical(
            list(ticker_map.values()),
            ["PX_BID", "PX_ASK", "PX_LAST"],
            start_date,
            end_date,
        )

        if raw.empty:
            return pd.DataFrame()

        # Build wide format: one series per (strike, field)
        series_dict: Dict[str, pd.Series] = {}
        for label, ticker in ticker_map.items():
            sub = raw[raw["Ticker"] == ticker].copy()
            if sub.empty:
                continue
            sub = sub.set_index("Date").sort_index()
            for field, suffix in [("PX_BID", "Bid"), ("PX_ASK", "Ask"), ("PX_LAST", "Last")]:
                if field in sub.columns:
                    series_dict[f"{label}_{suffix}"] = sub[field]

        if not series_dict:
            return pd.DataFrame()

        result = pd.DataFrame(series_dict)
        result.index = pd.to_datetime(result.index)
        result = result.sort_index()

        # Compute mid = (Bid + Ask) / 2; fill any gaps from PX_LAST
        for k in strikes:
            bid  = result.get(f"C{k}_Bid")
            ask  = result.get(f"C{k}_Ask")
            last = result.get(f"C{k}_Last")

            if bid is not None and ask is not None:
                mid = (bid + ask) / 2.0
                if last is not None:
                    mid = mid.fillna(last)
            elif last is not None:
                mid = last.copy()
            else:
                continue   # No price data for this strike

            result[f"C{k}_Mid"] = mid

        # Canonical column order: C10_Bid, C10_Ask, C10_Mid, C10_Last, C11_Bid …
        ordered = [
            f"C{k}_{suf}"
            for k in strikes
            for suf in ("Bid", "Ask", "Mid", "Last")
            if f"C{k}_{suf}" in result.columns
        ]
        return result[ordered]

    @staticmethod
    def _generate_expiry_dates_in_range(
        start_year: int, start_month: int,
        end_year: int,   end_month: int,
    ) -> List[datetime]:
        """
        Generate VIX monthly expiry dates (inclusive) from start to end.

        CBOE rule: VIX expiry = 3rd Friday of the FOLLOWING calendar month
        minus 30 calendar days.  This always lands on a Wednesday.
        """
        expiries: List[datetime] = []
        y, m = start_year, start_month

        while (y < end_year) or (y == end_year and m <= end_month):
            next_m = m + 1 if m < 12 else 1
            next_y = y     if m < 12 else y + 1

            month_cal = calendar.monthcalendar(next_y, next_m)
            fridays   = [week[calendar.FRIDAY] for week in month_cal
                         if week[calendar.FRIDAY] != 0]
            third_fri = datetime(next_y, next_m, fridays[2])
            expiries.append(third_fri - timedelta(days=30))

            m += 1
            if m > 12:
                m = 1
                y += 1

        return expiries

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

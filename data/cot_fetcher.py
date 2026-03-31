"""
data/cot_fetcher.py
===================
CFTC Commitments of Traders data for VIX futures.
Uses free public CSV downloads — no Bloomberg or paid data required.

The TFF (Traders in Financial Futures) report breaks positioning into:
  - Dealer/Intermediary
  - Asset Manager/Institutional  ← Primary signal
  - Leveraged Funds             ← Secondary signal
  - Other Reportables

Published weekly: positions as of Tuesday, released Friday 3:30 PM ET.
"""

import pandas as pd
import numpy as np
import io
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Try requests for downloads
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from config.settings import CFTC_VIX_CODE, CFTC_DATA_URL, CFTC_HISTORY_URL


class COTFetcher:
    """
    Fetch and process CFTC Commitments of Traders data for VIX futures.
    
    Output columns:
        - AM_Long, AM_Short, AM_Net     (Asset Manager)
        - LF_Long, LF_Short, LF_Net     (Leveraged Funds)
        - Dealer_Long, Dealer_Short, Dealer_Net
        - Total_OI                        (Total Open Interest)
        - AM_Net_OI_Pct                  (AM Net as % of OI)
        - LF_Net_OI_Pct                  (LF Net as % of OI)
    """
    
    CACHE_DIR = Path("outputs/cache")
    VIX_CFTC_CODE = "1170E1"   # CBOE VIX Futures code in TFF report
    
    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def fetch_all(
        self,
        start_year: int = 2022,
        end_year: int = None,
        cache_file: str = "cot_vix_data.parquet",
    ) -> pd.DataFrame:
        """
        Fetch COT data from CFTC public archives + current year.
        
        Returns:
            Weekly DataFrame with positioning data, DatetimeIndex (Tuesdays)
        """
        if not end_year:
            end_year = datetime.now().year
        
        cache_path = self.CACHE_DIR / cache_file
        
        # Check cache
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            last_date = df.index.max()
            if (pd.Timestamp.now() - last_date).days < 7:
                logger.info(f"COT cache fresh (last: {last_date.date()})")
                return df
        
        if not REQUESTS_AVAILABLE:
            if cache_path.exists():
                logger.warning("requests not installed. Using cached COT data.")
                return pd.read_parquet(cache_path)
            raise RuntimeError("requests package required for COT download. pip install requests")
        
        all_dfs = []
        
        # Fetch historical years
        for year in range(start_year, end_year):
            logger.info(f"Fetching COT data for {year}...")
            df = self._fetch_year(year)
            if df is not None and not df.empty:
                all_dfs.append(df)
        
        # Fetch current year (uses different URL)
        logger.info(f"Fetching COT data for {end_year} (current)...")
        df_current = self._fetch_current()
        if df_current is not None and not df_current.empty:
            all_dfs.append(df_current)
        
        if not all_dfs:
            raise RuntimeError("No COT data fetched from any source")
        
        result = pd.concat(all_dfs).sort_index()
        result = result[~result.index.duplicated(keep="last")]
        
        # Cache
        result.to_parquet(cache_path)
        logger.info(f"COT data cached: {len(result)} weeks, {result.index.min().date()} to {result.index.max().date()}")
        
        return result
    
    def _fetch_year(self, year: int) -> Optional[pd.DataFrame]:
        """Fetch historical year from CFTC zip archive."""
        url = CFTC_HISTORY_URL.format(year=year)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                # Find the TFF file in the archive
                csv_names = [n for n in z.namelist() if n.endswith(".txt") or n.endswith(".csv")]
                if not csv_names:
                    logger.warning(f"No CSV found in {url}")
                    return None
                
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f)
            
            return self._process_tff(df)
            
        except Exception as e:
            logger.warning(f"Failed to fetch COT for {year}: {e}")
            return None
    
    def _fetch_current(self) -> Optional[pd.DataFrame]:
        """Fetch current year TFF report."""
        try:
            resp = requests.get(CFTC_DATA_URL, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            return self._process_tff(df)
        except Exception as e:
            logger.warning(f"Failed to fetch current COT: {e}")
            return None
    
    def _process_tff(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract VIX futures positioning from raw TFF report.
        
        The TFF columns we need:
        - "Asset Mgr Positions-Long (All)" / "Asset Mgr Positions-Short (All)"
        - "Lev Money Positions-Long (All)" / "Lev Money Positions-Short (All)"  
        - "Dealer Positions-Long (All)" / "Dealer Positions-Short (All)"
        - "Open Interest (All)"
        """
        # Filter to VIX futures only
        # The CFTC code column varies by file format
        code_cols = ["CFTC_Contract_Market_Code", "CFTC Contract Market Code"]
        code_col = None
        for c in code_cols:
            if c in df.columns:
                code_col = c
                break
        
        if code_col is None:
            # Try filtering by name
            name_cols = ["Market_and_Exchange_Names", "Market and Exchange Names"]
            for c in name_cols:
                if c in df.columns:
                    df = df[df[c].str.contains("VIX", case=False, na=False)]
                    break
        else:
            df = df[df[code_col].astype(str).str.strip() == self.VIX_CFTC_CODE]
        
        if df.empty:
            return pd.DataFrame()
        
        # Parse report date
        date_cols = ["Report_Date_as_YYYY-MM-DD", "As of Date in Form YYYY-MM-DD"]
        date_col = None
        for c in date_cols:
            if c in df.columns:
                date_col = c
                break
        
        if date_col is None:
            logger.warning("Could not find date column in COT data")
            return pd.DataFrame()
        
        # Build output
        result = pd.DataFrame()
        result.index = pd.to_datetime(df[date_col].values)
        result.index.name = "Date"
        
        # Asset Manager positioning
        am_long_cols = [c for c in df.columns if "asset" in c.lower() and "long" in c.lower() and "all" in c.lower()]
        am_short_cols = [c for c in df.columns if "asset" in c.lower() and "short" in c.lower() and "all" in c.lower()]
        
        if am_long_cols and am_short_cols:
            result["AM_Long"] = df[am_long_cols[0]].values
            result["AM_Short"] = df[am_short_cols[0]].values
            result["AM_Net"] = result["AM_Long"] - result["AM_Short"]
        
        # Leveraged Funds
        lf_long_cols = [c for c in df.columns if "lev" in c.lower() and "long" in c.lower() and "all" in c.lower()]
        lf_short_cols = [c for c in df.columns if "lev" in c.lower() and "short" in c.lower() and "all" in c.lower()]
        
        if lf_long_cols and lf_short_cols:
            result["LF_Long"] = df[lf_long_cols[0]].values
            result["LF_Short"] = df[lf_short_cols[0]].values
            result["LF_Net"] = result["LF_Long"] - result["LF_Short"]
        
        # Dealer
        dealer_long_cols = [c for c in df.columns if "dealer" in c.lower() and "long" in c.lower() and "all" in c.lower()]
        dealer_short_cols = [c for c in df.columns if "dealer" in c.lower() and "short" in c.lower() and "all" in c.lower()]
        
        if dealer_long_cols and dealer_short_cols:
            result["Dealer_Long"] = df[dealer_long_cols[0]].values
            result["Dealer_Short"] = df[dealer_short_cols[0]].values
            result["Dealer_Net"] = result["Dealer_Long"] - result["Dealer_Short"]
        
        # Total Open Interest
        oi_cols = [c for c in df.columns if "open" in c.lower() and "interest" in c.lower() and "all" in c.lower()]
        if oi_cols:
            result["Total_OI"] = df[oi_cols[0]].values
            
            # Normalize by OI
            if "AM_Net" in result.columns:
                result["AM_Net_OI_Pct"] = result["AM_Net"] / result["Total_OI"] * 100
            if "LF_Net" in result.columns:
                result["LF_Net_OI_Pct"] = result["LF_Net"] / result["Total_OI"] * 100
        
        # Convert to numeric
        for col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
        
        result = result.sort_index()
        return result

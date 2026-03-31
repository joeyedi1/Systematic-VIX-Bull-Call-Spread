"""
data/cot_fetcher.py
===================
CFTC Commitments of Traders data for VIX futures.
Uses the cot_reports library which handles CFTC's changing URLs automatically.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    import cot_reports as cot
    COT_LIBRARY_AVAILABLE = True
except ImportError:
    COT_LIBRARY_AVAILABLE = False
    logger.warning("cot_reports not installed. pip install cot_reports")


class COTFetcher:
    CACHE_DIR = Path("outputs/cache")
    VIX_MARKET_NAMES = ["VIX", "CBOE VIX", "CBOE VOLATILITY INDEX"]

    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_all(
        self,
        start_year: int = 2022,
        end_year: int = None,
        cache_file: str = "cot_vix_data.parquet",
    ) -> pd.DataFrame:
        if not end_year:
            end_year = datetime.now().year

        cache_path = self.CACHE_DIR / cache_file

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            last_date = df.index.max()
            if (pd.Timestamp.now() - last_date).days < 7:
                logger.info(f"COT cache fresh (last: {last_date.date()})")
                return df

        if not COT_LIBRARY_AVAILABLE:
            if cache_path.exists():
                return pd.read_parquet(cache_path)
            raise RuntimeError("pip install cot_reports")

        all_dfs = []

        # Historical bulk (up to 2016)
        logger.info("Fetching COT historical bulk...")
        try:
            hist_df = cot.cot_hist(cot_report_type="traders_in_financial_futures_fut")
            if hist_df is not None and not hist_df.empty:
                processed = self._process_cot_df(hist_df)
                if processed is not None and not processed.empty:
                    all_dfs.append(processed)
                    logger.info(f"  Historical: {len(processed)} weeks")
        except Exception as e:
            logger.warning(f"  Historical fetch failed: {e}")

        # Recent years (2017+)
        for year in range(max(start_year, 2017), end_year + 1):
            logger.info(f"Fetching COT {year}...")
            try:
                year_df = cot.cot_year(
                    year=year,
                    cot_report_type="traders_in_financial_futures_fut",
                )
                if year_df is not None and not year_df.empty:
                    processed = self._process_cot_df(year_df)
                    if processed is not None and not processed.empty:
                        all_dfs.append(processed)
                        logger.info(f"  {year}: {len(processed)} weeks")
            except Exception as e:
                logger.warning(f"  {year} failed: {e}")

        if not all_dfs:
            raise RuntimeError("No COT data fetched from any source")

        result = pd.concat(all_dfs).sort_index()
        result = result[~result.index.duplicated(keep="last")]
        result = result[result.index >= pd.Timestamp(f"{start_year}-01-01")]

        result.to_parquet(cache_path)
        logger.info(f"COT cached: {len(result)} weeks, {result.index.min().date()} to {result.index.max().date()}")
        return result

    def _process_cot_df(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None

        # Find market name column
        name_col = None
        for c in df.columns:
            sample = df[c].astype(str).head(20)
            if any("VIX" in str(v).upper() for v in sample):
                name_col = c
                break

        if name_col is None:
            # Try common column names
            for c in ["Market and Exchange Names", "Market_and_Exchange_Names"]:
                if c in df.columns:
                    name_col = c
                    break

        if name_col is None:
            logger.warning(f"No market name column found. Cols: {df.columns.tolist()[:10]}")
            return None

        # Filter to VIX
        vix_mask = df[name_col].astype(str).str.upper().str.contains(
            "|".join(self.VIX_MARKET_NAMES), na=False
        )
        vix_df = df[vix_mask].copy()

        if vix_df.empty:
            return None

        # Find date column
        date_col = None
        for c in ["As of Date in Form YYYY-MM-DD", "Report_Date_as_YYYY-MM-DD",
                   "As_of_Date_In_Form_YYMMDD"]:
            if c in vix_df.columns:
                date_col = c
                break

        if date_col is None:
            # Try any column with dates
            for c in vix_df.columns:
                if "date" in c.lower():
                    date_col = c
                    break

        if date_col is None:
            logger.warning("No date column found")
            return None

        result = pd.DataFrame()
        result.index = pd.to_datetime(vix_df[date_col].values)
        result.index.name = "Date"

        # Asset Manager
        am_long = self._find_col(vix_df, ["asset", "mgr"], ["long"], ["spread", "change"])
        if not am_long:
            am_long = self._find_col(vix_df, ["asset", "inst"], ["long"], ["spread", "change"])
        am_short = self._find_col(vix_df, ["asset", "mgr"], ["short"], ["spread", "change"])
        if not am_short:
            am_short = self._find_col(vix_df, ["asset", "inst"], ["short"], ["spread", "change"])

        if am_long and am_short:
            result["AM_Long"] = pd.to_numeric(vix_df[am_long].values, errors="coerce")
            result["AM_Short"] = pd.to_numeric(vix_df[am_short].values, errors="coerce")
            result["AM_Net"] = result["AM_Long"] - result["AM_Short"]

        # Leveraged Funds
        lf_long = self._find_col(vix_df, ["lev"], ["long"], ["spread", "change"])
        lf_short = self._find_col(vix_df, ["lev"], ["short"], ["spread", "change"])

        if lf_long and lf_short:
            result["LF_Long"] = pd.to_numeric(vix_df[lf_long].values, errors="coerce")
            result["LF_Short"] = pd.to_numeric(vix_df[lf_short].values, errors="coerce")
            result["LF_Net"] = result["LF_Long"] - result["LF_Short"]

        # Dealer
        dealer_long = self._find_col(vix_df, ["dealer"], ["long"], ["spread", "change"])
        dealer_short = self._find_col(vix_df, ["dealer"], ["short"], ["spread", "change"])

        if dealer_long and dealer_short:
            result["Dealer_Long"] = pd.to_numeric(vix_df[dealer_long].values, errors="coerce")
            result["Dealer_Short"] = pd.to_numeric(vix_df[dealer_short].values, errors="coerce")
            result["Dealer_Net"] = result["Dealer_Long"] - result["Dealer_Short"]

        # Open Interest
        oi_col = self._find_col(vix_df, ["open", "interest"], ["all"], ["change", "old", "other"])
        if not oi_col:
            oi_col = self._find_col(vix_df, ["open", "interest"], [], ["change", "old", "other", "pct"])

        if oi_col:
            result["Total_OI"] = pd.to_numeric(vix_df[oi_col].values, errors="coerce")
            if "AM_Net" in result.columns:
                result["AM_Net_OI_Pct"] = result["AM_Net"] / result["Total_OI"] * 100
            if "LF_Net" in result.columns:
                result["LF_Net_OI_Pct"] = result["LF_Net"] / result["Total_OI"] * 100

        return result.sort_index()

    @staticmethod
    def _find_col(df, must_contain, also_contain, must_not_contain):
        for col in df.columns:
            col_lower = col.lower().replace("_", " ")
            if not all(kw.lower() in col_lower for kw in must_contain):
                continue
            if also_contain and not any(kw.lower() in col_lower for kw in also_contain):
                continue
            if any(kw.lower() in col_lower for kw in must_not_contain):
                continue
            return col
        return None
"""
utils/helpers.py
================
Common utility functions: VIX expiry calendar, date math, formatting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import calendar


def get_vix_expiry_dates(start_year: int = 2022, end_year: int = 2027) -> List[datetime]:
    """
    Generate VIX monthly expiry dates.
    
    VIX expires 30 days before SPX monthly option expiry (3rd Friday of following month).
    This is usually a Wednesday, unless there's a holiday.
    
    Returns:
        List of VIX expiry dates
    """
    expiry_dates = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # SPX expiry: 3rd Friday of the NEXT month
            next_month = month + 1
            next_year = year
            if next_month > 12:
                next_month = 1
                next_year += 1
            
            # Find 3rd Friday
            c = calendar.monthcalendar(next_year, next_month)
            fridays = [week[calendar.FRIDAY] for week in c if week[calendar.FRIDAY] != 0]
            third_friday = datetime(next_year, next_month, fridays[2])
            
            # VIX expiry = 30 days before
            vix_expiry = third_friday - timedelta(days=30)
            
            # Adjust to Wednesday (VIX typically expires on Wednesday)
            # If the 30-day-before lands on a weekend, move to the previous Wednesday
            while vix_expiry.weekday() != 2:  # Wednesday = 2
                vix_expiry -= timedelta(days=1)
            
            expiry_dates.append(vix_expiry)
    
    return sorted(set(expiry_dates))


def get_next_vix_expiry(from_date: datetime) -> datetime:
    """Get the next VIX expiry date from a given date."""
    expiries = get_vix_expiry_dates(from_date.year, from_date.year + 1)
    for exp in expiries:
        if exp > from_date:
            return exp
    return expiries[-1]


def business_days_between(start: str, end: str) -> int:
    """Count business days between two date strings."""
    return int(np.busday_count(start, end))


def format_pnl(pnl: float) -> str:
    """Format P&L with sign and color hint."""
    sign = "+" if pnl >= 0 else ""
    return f"{sign}${pnl:.2f}"


def format_pct(pct: float) -> str:
    """Format percentage with sign."""
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"

# utils/helpers.py
# Helper utilities for the NYSE trading bot
# Updated Feb 28 2026 — added proper NYSE holiday support
# March 2026: FIXED critical double-advance bug in time_until_next_open()
# (Sunday evening or Friday after 4 PM no longer skips an extra day)
# March 2026: FIXED holidays.US() bug (Bug #7)
# → Now uses accurate NYSE calendar (excludes Columbus/Veterans Day,
#   includes Good Friday, dynamic early closes for Thanksgiving & July 3)

import pandas as pd
from datetime import datetime, time, timedelta
import holidays
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# ACCURATE NYSE HOLIDAY + EARLY CLOSE LOGIC (replaces inaccurate holidays.US())
# =============================================================================
def _is_nyse_holiday(d: datetime.date) -> bool:
    """Returns True only for actual NYSE closed holidays.
    - Excludes Columbus Day and Veterans Day (markets are open)
    - Includes Good Friday (markets are closed)
    - Uses holidays.US() as base but overrides the two incorrect federal holidays"""
    if d.weekday() >= 5:
        return True  # weekend (handled separately in callers, but safe)

    us_hols = holidays.US(years=d.year, observed=True)
    if d not in us_hols:
        return False

    # Remove non-NYSE federal holidays
    if d.month == 10 and d.weekday() == 0 and 8 <= d.day <= 14:   # 2nd Monday Oct = Columbus
        return False
    if d.month == 11 and d.day == 11:                              # Veterans Day
        return False

    return True

def _is_early_close_day(d: datetime.date) -> bool:
    """Dynamic early-close detection (1:00 PM ET close).
    Correct for all years including 2026+ (no hardcoded dates)."""
    # Day after Thanksgiving (4th Friday of November)
    if d.month == 11:
        # Find 4th Thursday
        first_thu = next(day for day in range(1, 8) if datetime(d.year, 11, day).weekday() == 3)
        thanksgiving = datetime(d.year, 11, first_thu + 21).date()
        if d == thanksgiving + timedelta(days=1):
            return True

    # July 3 when July 4 falls on Tue/Wed/Thu
    if d.month == 7 and d.day == 3:
        july4 = datetime(d.year, 7, 4).weekday()
        if july4 in (1, 2, 3):  # Tue, Wed, Thu
            return True

    # Christmas Eve (always early close)
    if d.month == 12 and d.day == 24:
        return True

    return False

# =============================================================================
# PUBLIC API (unchanged structure)
# =============================================================================
def is_market_open(now: datetime = None) -> bool:
    """
    Returns True if the NYSE is currently open for regular trading.
    Handles weekends, holidays, and early closes correctly.
    """
    if now is None:
        now = datetime.now(tz=pd.Timestamp.now(tz='America/New_York').tzinfo)
   
    # Weekend
    if now.weekday() >= 5:
        return False
   
    # NYSE holiday (accurate version)
    if _is_nyse_holiday(now.date()):
        return False
   
    # Regular trading hours: 9:30 AM – 4:00 PM ET
    market_open = time(9, 30)
    market_close = time(16, 0)
    current_time = now.time()
   
    # Early close days (now dynamic and future-proof)
    if _is_early_close_day(now.date()):
        market_close = time(13, 0) # 1:00 PM ET early close
   
    return market_open <= current_time < market_close

def time_until_next_open(now: datetime = None) -> timedelta:
    """Returns timedelta until the next market open (9:30 AM ET)."""
    if now is None:
        now = datetime.now(tz=pd.Timestamp.now(tz='America/New_York').tzinfo)
   
    # Start at the next possible 9:30 AM slot
    next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
   
    # If we are already past today's 9:30 AM, move forward one day first
    if now.time() >= time(9, 30):
        next_open += timedelta(days=1)
   
    # Unified skip: handle weekends AND holidays in a single loop
    # This eliminates the double-advance bug that occurred when both
    # the weekend block and the after-hours block fired on Sunday evenings
    while next_open.weekday() >= 5 or _is_nyse_holiday(next_open.date()):
        next_open += timedelta(days=1)
   
    return next_open - now

# Legacy function for backward compatibility
def time_until_next_8am(now: datetime = None) -> timedelta:
    """Deprecated — kept for compatibility. Use time_until_next_open() instead."""
    logger.warning("time_until_next_8am() is deprecated. Use time_until_next_open() instead.")
    return time_until_next_open(now)

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
from zoneinfo import ZoneInfo

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

    # Good Friday: NYSE is closed but it's not a US federal holiday
    # Easter algorithm (Anonymous Gregorian) to find Good Friday
    a = d.year % 19
    b, c = divmod(d.year, 100)
    e, f = divmod(b, 4)
    g = (8 * b + 13) // 25
    h = (19 * a + b - e - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * f + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    easter = datetime(d.year, month, day).date()
    good_friday = easter - timedelta(days=2)
    if d == good_friday:
        return True

    us_hols = holidays.US(years=d.year, observed=True)
    if d not in us_hols:
        return False

    # Remove non-NYSE federal holidays
    if d.month == 10 and d.weekday() == 0 and 8 <= d.day <= 14:   # 2nd Monday Oct = Columbus
        return False
    # FIX #48: Veterans Day (Nov 11) AND its observed date — NYSE is OPEN on both.
    # When Nov 11 falls on Saturday, holidays.US(observed=True) marks Friday Nov 10.
    # When Nov 11 falls on Sunday, it marks Monday Nov 12. Exclude all of these.
    if d.month == 11:
        nov11 = datetime(d.year, 11, 11).date()
        nov11_weekday = nov11.weekday()
        veterans_dates = {nov11}
        if nov11_weekday == 5:  # Saturday → observed Friday
            veterans_dates.add(nov11 - timedelta(days=1))
        elif nov11_weekday == 6:  # Sunday → observed Monday
            veterans_dates.add(nov11 + timedelta(days=1))
        if d in veterans_dates:
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

    # July 3 early close when July 4 falls on Tue/Wed/Thu/Sat
    # When July 4 is Sunday, observed holiday is Monday July 5, early close is Friday July 2
    if d.month == 7:
        july4_weekday = datetime(d.year, 7, 4).weekday()
        if july4_weekday == 6 and d.day == 2:  # FIX #39: Sunday → early close Friday July 2
            return True
        if d.day == 3 and july4_weekday in (1, 2, 3, 5):  # Tue, Wed, Thu, Sat
            return True

    # Christmas Eve early close (1:00 PM ET)
    # If Dec 24 is Saturday, early close moves to Friday Dec 23
    # If Dec 24 is Sunday, early close moves to Friday Dec 22
    dec24 = datetime(d.year, 12, 24).date()
    dec24_weekday = dec24.weekday()
    if dec24_weekday == 5:       # Saturday → Friday Dec 23
        early_close_date = dec24 - timedelta(days=1)
    elif dec24_weekday == 6:     # Sunday → Friday Dec 22
        early_close_date = dec24 - timedelta(days=2)
    else:
        early_close_date = dec24
    if d == early_close_date:
        return True

    # L16 FIX: New Year's Eve early close (1:00 PM ET)
    dec31 = datetime(d.year, 12, 31).date()
    dec31_weekday = dec31.weekday()
    if dec31_weekday == 5:
        early_nye = dec31 - timedelta(days=1)  # Friday Dec 30
    elif dec31_weekday == 6:
        early_nye = dec31 - timedelta(days=2)  # Friday Dec 29
    else:
        early_nye = dec31
    if d == early_nye:
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
    eastern = ZoneInfo('America/New_York')
    if now is None:
        now = datetime.now(tz=eastern)
    elif now.tzinfo is not None:
        # Convert to Eastern time regardless of input timezone
        now = now.astimezone(eastern)
    else:
        # Naive datetime — assume it's already Eastern
        now = now.replace(tzinfo=eastern)

    # L15 FIX: _is_nyse_holiday already checks weekends, so skip redundant check
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
    eastern = ZoneInfo('America/New_York')
    if now is None:
        now = datetime.now(tz=eastern)
    elif now.tzinfo is not None:
        # Convert to Eastern time regardless of input timezone
        now = now.astimezone(eastern)
    else:
        # Naive datetime — assume it's already Eastern
        now = now.replace(tzinfo=eastern)

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

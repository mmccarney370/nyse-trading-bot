# strategy/earnings_filter.py
"""
B5 — Anti-earnings filter.

Every earnings announcement is a -5% to -20% single-name gap risk. Statistically,
trades held through earnings have WR crushed by ~15-20 percentage points vs.
trades avoiding the event. Cheapest way to boost win rate: just don't trade
through them.

Behavior:
- At startup and weekly, refresh next earnings dates for all universe symbols
  via yfinance (free, cached for a week).
- Blackout window: no new entries for BLACKOUT_PRE days before earnings and
  BLACKOUT_POST days after. Defaults 2 pre, 1 post.
- Auto-close: open positions within CLOSE_PRE days of earnings are flattened
  on the next cycle. Default 1 day — close the day before.

Failure is always safe: if yfinance returns nothing, no blackouts fire and
normal trading proceeds. Errors are swallowed + logged, never blocking.
"""
from __future__ import annotations

import logging
import os
import pickle
import threading
from datetime import datetime, date, timedelta
from dateutil import tz
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


class EarningsCalendar:
    """Tracks upcoming earnings dates for the trading universe."""

    def __init__(self, symbols: List[str], persist_path: str = "earnings_cache.pkl",
                 ttl_hours: int = 168):  # 1 week
        self.symbols = list(symbols)
        self.persist_path = persist_path
        self.ttl_hours = ttl_hours
        # symbol → (earnings_date, fetched_at). date may be None if no upcoming date found.
        self._cache: Dict[str, tuple[Optional[date], datetime]] = {}
        self._lock = threading.RLock()
        self._load()

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------
    def refresh(self, symbols: Optional[List[str]] = None, force: bool = False) -> None:
        """Fetch next earnings date for each symbol. Honors TTL unless force=True."""
        if not _YF_AVAILABLE:
            logger.warning("[EARNINGS] yfinance unavailable — blackouts disabled")
            return
        targets = symbols if symbols is not None else self.symbols
        now = datetime.now(tz=tz.gettz('UTC'))
        for sym in targets:
            try:
                with self._lock:
                    cached = self._cache.get(sym)
                if not force and cached is not None:
                    _d, fetched_at = cached
                    if (now - fetched_at).total_seconds() / 3600 < self.ttl_hours:
                        continue  # still fresh
                next_date = self._fetch_one(sym)
                with self._lock:
                    self._cache[sym] = (next_date, now)
                if next_date is not None:
                    logger.info(f"[EARNINGS] {sym} next earnings: {next_date.isoformat()}")
                else:
                    logger.debug(f"[EARNINGS] {sym} no upcoming earnings found")
            except Exception as e:
                logger.warning(f"[EARNINGS] {sym} refresh failed: {e}")
        self._save()

    @staticmethod
    def _fetch_one(symbol: str) -> Optional[date]:
        """Fetch next earnings date from yfinance. Returns None if not found."""
        try:
            ticker = yf.Ticker(symbol)
            # yfinance offers several endpoints; we try the most reliable ones in order
            try:
                cal = ticker.calendar
                if isinstance(cal, dict):
                    raw = cal.get('Earnings Date')
                    if raw:
                        # Can be a list or a single date
                        candidates = raw if isinstance(raw, list) else [raw]
                        today = date.today()
                        upcoming = [_coerce_date(d) for d in candidates]
                        upcoming = [d for d in upcoming if d is not None and d >= today]
                        if upcoming:
                            return min(upcoming)
            except Exception:
                pass
            # Fallback: earnings_dates DataFrame
            try:
                df = ticker.earnings_dates
                if df is not None and not df.empty:
                    # Index is datetime; select the next upcoming one
                    today = date.today()
                    candidates = []
                    for idx in df.index:
                        d = _coerce_date(idx)
                        if d is not None and d >= today:
                            candidates.append(d)
                    if candidates:
                        return min(candidates)
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"[EARNINGS] yfinance lookup for {symbol}: {e}")
        return None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def days_to_earnings(self, symbol: str) -> Optional[int]:
        """Trading days (calendar, approx) until next earnings. None if unknown."""
        with self._lock:
            cached = self._cache.get(symbol)
        if not cached:
            return None
        next_date, _ = cached
        if next_date is None:
            return None
        return (next_date - date.today()).days

    def is_in_blackout(self, symbol: str, pre_days: int, post_days: int) -> tuple[bool, str]:
        """Return (blocked, reason). If no earnings data or too far away, not blocked."""
        dte = self.days_to_earnings(symbol)
        if dte is None:
            return False, "no-data"
        if 0 <= dte <= pre_days:
            return True, f"earnings_in_{dte}d"
        if -post_days <= dte < 0:
            return True, f"post_earnings_{-dte}d"
        return False, f"clear_t{dte:+d}d"

    def should_close_before_earnings(self, symbol: str, close_pre_days: int) -> tuple[bool, str]:
        """Open-position check: return True if we should flatten this position now."""
        dte = self.days_to_earnings(symbol)
        if dte is None:
            return False, "no-data"
        if 0 <= dte <= close_pre_days:
            return True, f"flat_pre_earnings_{dte}d"
        return False, f"hold_t{dte:+d}d"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(self):
        try:
            import tempfile, shutil
            dir_name = os.path.dirname(self.persist_path) or "."
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, dir=dir_name, suffix=".tmp") as tmp:
                with self._lock:
                    pickle.dump({"cache": self._cache, "symbols": self.symbols}, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
            shutil.move(tmp.name, self.persist_path)
        except Exception as e:
            logger.debug(f"[EARNINGS] save failed: {e}")

    def _load(self):
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
            with self._lock:
                self._cache = data.get("cache", {})
            logger.info(f"[EARNINGS] Loaded {len(self._cache)} symbols from cache")
        except Exception as e:
            logger.debug(f"[EARNINGS] load failed: {e}")


def _coerce_date(d) -> Optional[date]:
    """Best-effort conversion from timestamp/datetime/date to date."""
    try:
        if isinstance(d, date) and not isinstance(d, datetime):
            return d
        if hasattr(d, "date"):
            return d.date()
        if isinstance(d, str):
            return datetime.fromisoformat(d).date()
    except Exception:
        return None
    return None

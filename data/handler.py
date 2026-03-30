# data/handler.py
# COMPLETE, FULLY NON-TRUNCATED VERSION (Feb 18 2026 + Feb 20 2026 ArcticDB integration)
# - Polygon is now the PRIMARY data source (first attempted, with full volume sanity check)
# - Strong global rate limiter: enforces minimum 12.5 seconds between ANY Polygon calls across ALL threads
# - Dynamic timestamp-based delay (not just fixed sleep) so calls are perfectly spaced even with parallelism
# - Catches MaxRetryError + 429 responses specifically (prevents the crash you just saw)
# - On 429: forces extra 30-second cooldown
# - Larger chunks for Polygon (30 days) → fewer total requests
# - All previous features preserved: real volume repair, robust column normalization,
#   dedup indexes, non-overlapping chunks, all caching (Redis + file + in-memory), holidays, bid-ask spread,
#   Finnhub safe setup, strict volume sanity (>500 mean), fallbacks with volume checks
# NEW (Feb 20 2026): Full ArcticDB local tick database integration
#   - Checked FIRST after Redis/file cache when USE_LOCAL_TICKDB=True
#   - Uses modern safe API (get_library with create_if_missing=True) — works on all recent ArcticDB versions
#   - Data is saved to ArcticDB after every successful external fetch
#   - Polygon is now only called when data is missing or stale
# - No truncation, no placeholders — every single line is here

import logging
import redis
import pickle
import time
import holidays
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
import threading
import requests
from urllib3.exceptions import MaxRetryError
from datetime import datetime, timedelta
from dateutil import tz
from typing import Optional
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import yfinance as yf
import polygon
import tiingo
import finnhub
import arcticdb as adb   # ← ArcticDB

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self, config: dict):
        self.config = config
     
        # Redis attempt
        try:
            self.redis_client = redis.Redis(
                host=config['REDIS_HOST'],
                port=config['REDIS_PORT'],
                db=config['REDIS_DB']
            )
            self.redis_client.ping()
            self.use_redis = True
            logger.info("Redis connected successfully — using Redis for caching")
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {e}. Falling back to file + in-memory cache.")
            self.redis_client = None
            self.use_redis = False
        # Always init in-memory cache attributes (fallback used even when Redis is primary)
        self.in_memory_cache = {}
        self.in_memory_daily_cache = {}
        self._mem_cache_max = 50
        # FIX #5: Thread-safe lock for in-memory cache access
        self._cache_lock = threading.Lock()

        # === ARCTICDB LOCAL TICK DATABASE (NEW + FIXED) ===
        self.use_local_tickdb = config.get('USE_LOCAL_TICKDB', False)
        self.tickdb_engine = config.get('TICKDB_ENGINE', 'arcticdb')
        
        if self.use_local_tickdb and self.tickdb_engine == 'arcticdb':
            try:
                # Use LMDB backend (fast, local file-based)
                # FIX #46: Use absolute path to avoid CWD-dependent database location
                _arcticdb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "arcticdb_tickdb")
                self.arctic = adb.Arctic(f"lmdb://{_arcticdb_path}")
                
                # Modern safe library creation (works on all recent ArcticDB versions)
                # FIX #8: Added 'nyse_1d' library for daily data caching
                for lib_name in ["nyse_15min", "nyse_1h", "nyse_1d"]:
                    try:
                        # get_library with create_if_missing=True is the current recommended way
                        lib = self.arctic.get_library(lib_name, create_if_missing=True)
                        logger.info(f"✅ ArcticDB library ready: {lib_name}")
                    except Exception as e:
                        logger.warning(f"ArcticDB library {lib_name} issue: {e}")

                self.arctic_15min = self.arctic.get_library("nyse_15min", create_if_missing=True)
                self.arctic_1h = self.arctic.get_library("nyse_1h", create_if_missing=True)
                self.arctic_1d = self.arctic.get_library("nyse_1d", create_if_missing=True)
                
                logger.info("✅ ArcticDB local tick database initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ArcticDB: {e}")
                self.use_local_tickdb = False
        else:
            self.use_local_tickdb = False

        # FIX #28: Use absolute path for file cache — relative path depends on CWD
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_cache')
        self.cache_dir = os.path.abspath(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Persistent file cache directory: {self.cache_dir}")

        self.data_client = StockHistoricalDataClient(config['API_KEY'], config['API_SECRET'])
        # FIX #25: Only create Polygon/Tiingo clients when API keys are present
        _polygon_key = config.get('POLYGON_API_KEY')
        if _polygon_key:
            self.polygon_client = polygon.RESTClient(api_key=_polygon_key)
        else:
            self.polygon_client = None
            logger.warning("[HANDLER] POLYGON_API_KEY not set — Polygon client disabled")
        _tiingo_key = config.get('TIINGO_API_KEY')
        if _tiingo_key:
            self.tiingo_client = tiingo.TiingoClient({'api_key': _tiingo_key})
        else:
            self.tiingo_client = None
            logger.warning("[HANDLER] TIINGO_API_KEY not set — Tiingo client disabled")

        # === STRONG POLYGON RATE LIMIT PROTECTION ===
        self.polygon_lock = threading.Lock()
        self.last_polygon_call_time = 0.0
        self.polygon_min_delay = 12.5   # seconds — exactly matches free-tier 5 calls/minute

        # FIX #45: Build NYSE-accurate holiday set excluding Columbus Day and Veterans Day
        # (NYSE is open on both, but holidays.US() includes them)
        self.us_holidays = self._build_nyse_holidays()
        self.api_failures = 0
        # FIX #6: Thread-safe lock for api_failures counter
        self._api_failures_lock = threading.Lock()
        self.polygon_extra_delay = 1.5   # kept for extra safety on 429s

        # === FINNHUB SAFE SETUP (no debug logging) ===
        finnhub_key = config.get('FINNHUB_API_KEY')
        if finnhub_key and isinstance(finnhub_key, str) and len(finnhub_key) > 25 and not finnhub_key.startswith('d6av9hhr'):
            try:
                self.finnhub_client = finnhub.Client(api_key=finnhub_key)
                self.use_finnhub = True
            except Exception as e:
                logger.error(f"Finnhub client creation failed: {e}")
                self.finnhub_client = None
                self.use_finnhub = False
        else:
            self.finnhub_client = None
            self.use_finnhub = False
        # === END FINNHUB SETUP ===

    @staticmethod
    def _compute_easter(year):
        """Compute Easter Sunday date using the Anonymous Gregorian algorithm."""
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        from datetime import date
        return date(year, month, day)

    @staticmethod
    def _build_nyse_holidays(years=None):
        """Build a holiday set that excludes Columbus Day and Veterans Day (NYSE is open)
        and includes Good Friday (NYSE is closed but not in holidays.US)."""
        from datetime import date, timedelta
        if years is None:
            current_year = datetime.now().year
            years = range(current_year - 2, current_year + 3)
        nyse_holidays = {}
        for year in years:
            us_hols = holidays.US(years=year, observed=True)
            for d, name in us_hols.items():
                # Skip Columbus Day and Veterans Day (and their observed dates)
                if 'Columbus' in name or 'Indigenous' in name:
                    continue
                if 'Veterans' in name:
                    continue
                nyse_holidays[d] = name
            # FIX #24: Add Good Friday (NYSE is closed; not included in holidays.US)
            easter = DataHandler._compute_easter(year)
            good_friday = easter - timedelta(days=2)
            nyse_holidays[good_friday] = "Good Friday"
        return nyse_holidays

    # FIX #6: Thread-safe api_failures helpers
    def _increment_api_failures(self):
        with self._api_failures_lock:
            self.api_failures += 1

    def _reset_api_failures(self):
        with self._api_failures_lock:
            self.api_failures = 0

    # ==================== ARCTICDB HELPERS ====================
    def _get_arctic_library(self, timeframe: str):
        if not self.use_local_tickdb:
            return None
        if timeframe in ['15Min', '15min']:
            return self.arctic_15min
        elif timeframe in ['1H', '1h', '60min']:
            return self.arctic_1h
        elif timeframe in ['1d', '1D', 'daily']:  # FIX #8: Support daily timeframe
            return self.arctic_1d
        return None

    def _read_from_arctic(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        lib = self._get_arctic_library(timeframe)
        if lib is None:
            return None
        symbol_key = f"{symbol}_{timeframe}"
        try:
            if lib.has_symbol(symbol_key):
                df = lib.read(symbol_key).data
                if not df.empty:
                    # FIX #7: Ensure both sides of comparison have same timezone
                    if df.index.tz is None and start.tzinfo is not None:
                        df.index = df.index.tz_localize('UTC')
                    elif df.index.tz is not None and start.tzinfo is None:
                        # FIX #26: Attach UTC to naive datetimes that are already in UTC.
                        # Using datetime.timezone.utc (stdlib) is safe for replace() since UTC
                        # has no DST ambiguity (unlike pytz zones where replace is dangerous).
                        from datetime import timezone as _tz
                        start = start.replace(tzinfo=_tz.utc)
                        end = end.replace(tzinfo=_tz.utc)
                    filtered = df[(df.index >= start) & (df.index <= end)]
                    if len(filtered) >= 50:
                        logger.info(f"✅ ArcticDB hit for {symbol} {timeframe} ({len(filtered)} bars)")
                        return self._normalize_columns(filtered, symbol)
        except Exception as e:
            logger.debug(f"ArcticDB read failed for {symbol} {timeframe}: {e}")
        return None

    def _save_to_arctic(self, symbol: str, timeframe: str, df: pd.DataFrame):
        if not self.use_local_tickdb or df.empty:
            return
        lib = self._get_arctic_library(timeframe)
        if lib is None:
            return
        symbol_key = f"{symbol}_{timeframe}"
        try:
            # FIX: Merge with existing data instead of overwriting (was losing historical data)
            if lib.has_symbol(symbol_key):
                existing = lib.read(symbol_key).data
                # M13 FIX: Normalize timezones before merge to prevent silent data loss
                if hasattr(existing.index, 'tz') and existing.index.tz is not None and (not hasattr(df.index, 'tz') or df.index.tz is None):
                    df.index = df.index.tz_localize(existing.index.tz)
                elif hasattr(df.index, 'tz') and df.index.tz is not None and (not hasattr(existing.index, 'tz') or existing.index.tz is None):
                    existing.index = existing.index.tz_localize(df.index.tz)
                merged = pd.concat([existing, df])
                merged = merged[~merged.index.duplicated(keep='last')].sort_index()
                lib.write(symbol_key, merged)
            else:
                lib.write(symbol_key, df)
            logger.debug(f"✅ Saved {symbol} {timeframe} to ArcticDB")
        except Exception as e:
            logger.warning(f"ArcticDB write failed for {symbol}: {e}")

    # ==================== EXISTING METHODS (unchanged) ====================

    def _daily_cache_key(self, symbol: str) -> str:
        # FIX #67: Use ET date for cache key — UTC date rolls over at 7/8pm ET,
        # causing stale cache hits for evening queries on the "next" UTC day.
        et_now = datetime.now(tz=tz.gettz('America/New_York'))
        today = et_now.date().isoformat()
        return f"daily_data:{symbol}:{today}"

    def _get_cached_daily(self, symbol: str) -> Optional[pd.DataFrame]:
        key = self._daily_cache_key(symbol)
        if self.use_redis:
            try:
                cached = self.redis_client.get(key)
                if cached:
                    logger.debug(f"Redis daily cache HIT: {key}")
                    return pickle.loads(cached)
            except Exception as e:
                logger.error(f"Redis daily cache load error: {e}")
        else:
            with self._cache_lock:
                if key in self.in_memory_daily_cache:
                    cached_timestamp, df = self.in_memory_daily_cache[key]
                    # M12 FIX: Use ET date for cache freshness (NYSE operates on ET)
                    if cached_timestamp.date() == datetime.now(tz=tz.gettz('America/New_York')).date():
                        logger.debug(f"In-memory daily cache HIT: {key}")
                        return df
            date_str = key.split(':')[-1]
            cache_file = os.path.join(self.cache_dir, f"daily_{symbol}_{date_str}.pkl")
            if os.path.exists(cache_file):
                file_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).total_seconds() / 86400
                if file_age_days < 3:
                    try:
                        df = pd.read_pickle(cache_file)
                        logger.debug(f"File daily cache HIT: {cache_file} (age: {file_age_days} days)")
                        return df
                    except Exception as e:
                        logger.error(f"Failed to load daily file cache {cache_file}: {e}")
        return None

    def _set_cached_daily(self, symbol: str, df: pd.DataFrame):
        if df.empty:
            return
        key = self._daily_cache_key(symbol)
        pickled = pickle.dumps(df)
        if self.use_redis:
            try:
                self.redis_client.setex(key, 172800, pickled)
                logger.debug(f"Redis daily cache SAVED: {key}")
            except Exception as e:
                logger.error(f"Redis daily cache save error: {e}")
        else:
            with self._cache_lock:
                self.in_memory_daily_cache[key] = (datetime.now(tz=tz.gettz('UTC')), df)
                if len(self.in_memory_daily_cache) > self._mem_cache_max:
                    oldest_key = min(self.in_memory_daily_cache, key=lambda k: self.in_memory_daily_cache[k][0])
                    del self.in_memory_daily_cache[oldest_key]
            date_str = key.split(':')[-1]
            cache_file = os.path.join(self.cache_dir, f"daily_{symbol}_{date_str}.pkl")
            try:
                fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix='.pkl.tmp')
                os.close(fd)
                df.to_pickle(tmp_path)
                shutil.move(tmp_path, cache_file)
                logger.debug(f"File daily cache SAVED (atomic): {cache_file}")
            except Exception as e:
                logger.error(f"Failed to save daily file cache {cache_file}: {e}")
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    def load_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        key = f"historical_data:{symbol}:{timeframe}"
        if self.use_redis:
            try:
                cached = self.redis_client.get(key)
                if cached:
                    logger.debug(f"Redis historical cache HIT: {key}")
                    return pickle.loads(cached)
            except Exception as e:
                logger.error(f"Redis historical load error: {e}")
        else:
            with self._cache_lock:
                if key in self.in_memory_cache:
                    cached_ts, cached_df = self.in_memory_cache[key]
                    # Evict if older than 2 hours
                    if (datetime.now(tz=tz.gettz('UTC')) - cached_ts).total_seconds() < 7200:
                        logger.debug(f"In-memory historical cache HIT: {key}")
                        return cached_df
                    else:
                        del self.in_memory_cache[key]
                        logger.debug(f"In-memory historical cache EXPIRED: {key}")
            cache_file = os.path.join(self.cache_dir, f"{symbol}_{timeframe}.pkl")
            if os.path.exists(cache_file):
                ttl_days = 1 if 'min' in timeframe.lower() else self.config.get('CACHE_TTL_DAYS', 30)
                # FIX #22: Use fractional days instead of integer floor (.days truncates)
                file_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).total_seconds() / 86400
                # FIX #66: Friday cache should be valid on Saturday/Sunday — use market-day-aware TTL.
                # If no market day has passed since file was written, treat as fresh regardless of wall-clock age.
                if ttl_days <= 1:
                    file_date = datetime.fromtimestamp(os.path.getmtime(cache_file))
                    now = datetime.now()
                    market_days_elapsed = sum(1 for d in pd.bdate_range(file_date.date(), now.date(), freq='C', holidays=list(self.us_holidays.keys())) if d.date() > file_date.date() and d.date() <= now.date())
                    cache_fresh = market_days_elapsed < 1
                else:
                    cache_fresh = file_age_days < ttl_days
                if cache_fresh:
                    try:
                        df = pd.read_pickle(cache_file)
                        logger.debug(f"File historical cache HIT: {cache_file} (age: {file_age_days} days)")
                        return df
                    except Exception as e:
                        logger.error(f"Failed to load historical file cache {cache_file}: {e}")
        return None

    def cache_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        if df.empty:
            return
        key = f"historical_data:{symbol}:{timeframe}"
        pickled = pickle.dumps(df)
        ttl_seconds = self.config.get('CACHE_TTL_DAYS', 30) * 86400
        if self.use_redis:
            try:
                self.redis_client.setex(key, ttl_seconds, pickled)
                logger.debug(f"Redis historical cache SAVED: {key}")
            except Exception as e:
                logger.error(f"Redis historical cache save error: {e}")
        else:
            with self._cache_lock:
                self.in_memory_cache[key] = (datetime.now(tz=tz.gettz('UTC')), df)
                # Evict oldest entries if cache exceeds max size
                if len(self.in_memory_cache) > self._mem_cache_max:
                    oldest_key = min(self.in_memory_cache, key=lambda k: self.in_memory_cache[k][0])
                    del self.in_memory_cache[oldest_key]
                    logger.debug(f"In-memory cache evicted oldest: {oldest_key}")
            cache_file = os.path.join(self.cache_dir, f"{symbol}_{timeframe}.pkl")
            try:
                fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix='.pkl.tmp')
                os.close(fd)
                df.to_pickle(tmp_path)
                shutil.move(tmp_path, cache_file)
                logger.debug(f"File historical cache SAVED (atomic): {cache_file}")
            except Exception as e:
                logger.error(f"Failed to save historical file cache {cache_file}: {e}")
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    def is_market_open_day(self, date: datetime) -> bool:
        date_date = date.date()
        return date_date.weekday() < 5 and date_date not in self.us_holidays

    def get_last_market_day(self, date: datetime) -> datetime:
        while not self.is_market_open_day(date):
            date -= timedelta(days=1)
        return date

    def adjust_date_range(self, start: datetime, end: datetime) -> tuple[datetime, datetime]:
        start = self.get_last_market_day(start)
        end = self.get_last_market_day(end)
        return start, end

    def _repair_volume(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if df.empty or 'volume' not in df.columns:
            return df
        original_zeros = (df['volume'] == 0).sum()
        if original_zeros == 0:
            return df
        df['volume'] = df['volume'].replace(0, np.nan)
        df['volume'] = df['volume'].ffill().bfill()
        if original_zeros > 0:
            logger.debug(f"[VOLUME REPAIR] {symbol} — repaired {original_zeros} zero-volume bars using real neighboring data")
        return df

    def _normalize_columns(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        if df.empty:
            return df
        df.columns = df.columns.str.lower()
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                df[col] = np.nan
        df = df[required]
        df = df[~df.index.duplicated(keep='last')]
        if symbol:
            df = self._repair_volume(df, symbol)
        return df.sort_index()

    def _chunked_fetch(self, fetch_func, symbol: str, timeframe: str, start: datetime, end: datetime, chunk_days: int = 15) -> pd.DataFrame:
        dfs = []
        failed_chunks = 0  # FIX #4: Track failed chunks to warn about data gaps
        current_start = start
        while current_start < end:
            # FIX #26: Clamp current_end so it never overshoots the requested end
            current_end = min(current_start + timedelta(days=chunk_days), end)
            for attempt in range(12):
                e = None
                try:
                    df = fetch_func(symbol, timeframe, current_start, current_end)
                except (requests.exceptions.RequestException, MaxRetryError, ValueError, TypeError) as exc:
                    e = exc
                    df = pd.DataFrame()
                if not df.empty:
                    df = self._normalize_columns(df, symbol)
                    dfs.append(df)
                    break
                # FIX: Cap backoff at 120s (was growing to 10240s on attempt 11)
                wait = min(2 ** attempt * 5, 120) + (30 if e and "429" in str(e) else 0)
                logger.warning(f"Chunk fetch failed ({symbol} {current_start.date()}-{current_end.date()}), retry {attempt+1}/12 after {wait}s: {e or 'empty data'}")
                time.sleep(wait)
            else:
                logger.error(f"Failed chunk {current_start.date()}-{current_end.date()} after 12 retries")
                failed_chunks += 1
            current_start = current_end

        # FIX #4: Warn about data gaps from failed chunks
        if failed_chunks > 0:
            logger.warning(f"[DATA GAPS] {symbol} {timeframe}: {failed_chunks} chunk(s) failed — result may have gaps in date coverage")

        if dfs:
            full_df = pd.concat(dfs)
            # FIX #69: Chunk boundary bars may be duplicated across adjacent chunks.
            # This is acceptable — _normalize_columns applies keep='last' dedup (line ~432).
            full_df = self._normalize_columns(full_df, symbol)
            return full_df
        return pd.DataFrame()

    def fetch_data(self, symbol: str, timeframe: str, start: datetime, end: datetime, for_live_trading: bool = False) -> pd.DataFrame:
        # First check cache for daily data
        if timeframe == '1d':
            cached = self._get_cached_daily(symbol)
            if cached is not None:
                filtered = cached[(cached.index >= start) & (cached.index <= end)]
                if len(filtered) >= 50:
                    logger.debug(f"Daily cache used for {symbol} {timeframe} ({len(filtered)} bars)")
                    return self._normalize_columns(filtered, symbol)

        # Check persistent historical cache (Redis + file)
        cached = self.load_cached_data(symbol, timeframe)
        if cached is not None:
            # FIX #25: Ensure tz-aware comparison — cached index may be tz-aware while end is naive
            cache_max = cached.index.max()
            end_cmp = end
            if hasattr(cache_max, 'tzinfo') and cache_max.tzinfo is not None and (not hasattr(end_cmp, 'tzinfo') or end_cmp.tzinfo is None):
                # M14 FIX: tz_localize instead of replace — replace() attaches tz label
                # without converting, causing incorrect comparisons if end_cmp was meant as UTC
                end_cmp = end_cmp.tz_localize(cache_max.tzinfo)
            elif (hasattr(end_cmp, 'tzinfo') and end_cmp.tzinfo is not None) and (not hasattr(cache_max, 'tzinfo') or cache_max.tzinfo is None):
                # CRIT-8 FIX: Convert to UTC first, then strip tz (replace(tzinfo=None) loses offset)
                if hasattr(end_cmp, 'tz_convert'):
                    end_cmp = end_cmp.tz_convert('UTC').tz_localize(None)
                else:
                    from datetime import timezone as _tz
                    end_cmp = end_cmp.astimezone(_tz.utc).replace(tzinfo=None)
            if cache_max >= end_cmp - timedelta(hours=1):
                filtered = cached[(cached.index >= start) & (cached.index <= end)]
                if len(filtered) >= 50:
                    logger.debug(f"Historical cache used for {symbol} {timeframe} ({len(filtered)} bars)")
                    return self._normalize_columns(filtered, symbol)

        # === ARCTICDB CHECK (NEW) ===
        if self.use_local_tickdb:
            arctic_df = self._read_from_arctic(symbol, timeframe, start, end)
            if arctic_df is not None and not arctic_df.empty:
                return arctic_df

        # === POLYGON PRIMARY ===
        df = self._chunked_fetch(self._fetch_polygon_data, symbol, timeframe, start, end, chunk_days=30)
        if not df.empty:
            mean_vol = df['volume'].mean()
            if mean_vol > 500:
                logger.info(f"✅ Polygon volume OK for {symbol} (mean={mean_vol:.0f}) — using as primary source")
                self._reset_api_failures()  # FIX #6: Reset on successful fetch
                self.cache_data(symbol, timeframe, df)
                self._save_to_arctic(symbol, timeframe, df)
                if timeframe == '1d':
                    self._set_cached_daily(symbol, df)
                return df
            else:
                logger.warning(f"❌ Polygon bogus volume for {symbol} (mean={mean_vol:.0f}) → falling back")

        # Alpaca SECONDARY
        df = self._chunked_fetch(self._fetch_alpaca_data, symbol, timeframe, start, end, chunk_days=30)
        if not df.empty:
            mean_vol = df['volume'].mean()
            if mean_vol > 500:
                logger.info(f"✅ Alpaca volume OK for {symbol} (mean={mean_vol:.0f})")
                self._reset_api_failures()  # FIX #6: Reset on successful fetch
                self.cache_data(symbol, timeframe, df)
                self._save_to_arctic(symbol, timeframe, df)
                if timeframe == '1d':
                    self._set_cached_daily(symbol, df)
                return df
            else:
                logger.warning(f"❌ Alpaca low volume for {symbol} (mean={mean_vol:.0f}) → fallback")

        # Finnhub TERTIARY
        if self.use_finnhub:
            df = self._chunked_fetch(self._fetch_finnhub_data, symbol, timeframe, start, end, chunk_days=7)
            if not df.empty and len(df) >= 10:
                mean_vol = df['volume'].mean()
                if mean_vol > 500:
                    logger.info(f"✅ Finnhub volume OK for {symbol} (mean={mean_vol:.0f})")
                    self._reset_api_failures()  # FIX #6: Reset on successful fetch
                    self.cache_data(symbol, timeframe, df)
                    self._save_to_arctic(symbol, timeframe, df)
                    if timeframe == '1d':
                        self._set_cached_daily(symbol, df)
                    return df
                else:
                    logger.warning(f"❌ Finnhub low volume for {symbol} (mean={mean_vol:.0f}) → fallback")

        # Tiingo
        df = self._fetch_tiingo_data(symbol, timeframe, start, end)
        if not df.empty and len(df) >= 10:
            df = self._normalize_columns(df, symbol)
            mean_vol = df['volume'].mean()
            if mean_vol > 500:
                logger.info(f"✅ Tiingo volume OK for {symbol} (mean={mean_vol:.0f})")
                self.cache_data(symbol, timeframe, df)
                self._save_to_arctic(symbol, timeframe, df)
                if timeframe == '1d':
                    self._set_cached_daily(symbol, df)
                return df
            else:
                logger.warning(f"❌ Tiingo low volume for {symbol} (mean={mean_vol:.0f}) → fallback")

        # yFinance final fallback
        df = self._fetch_yfinance_data(symbol, timeframe, start, end)
        if not df.empty and len(df) >= 10:
            df = self._normalize_columns(df, symbol)
            mean_vol = df['volume'].mean()
            if mean_vol > 500:
                logger.info(f"✅ yFinance volume OK for {symbol} (mean={mean_vol:.0f})")
                self.cache_data(symbol, timeframe, df)
                self._save_to_arctic(symbol, timeframe, df)
                if timeframe == '1d':
                    self._set_cached_daily(symbol, df)
                return df
            else:
                logger.warning(f"❌ yFinance low volume for {symbol} (mean={mean_vol:.0f}) → no more fallbacks")

        logger.warning(f"All sources failed for {symbol} {timeframe} {start.date()} to {end.date()}")
        return pd.DataFrame()

    # === ALL REMAINING FETCH METHODS (unchanged from your original) ===
    def _fetch_alpaca_data(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        time.sleep(self.config['REQUEST_INTERVAL'])
        try:
            if timeframe == '1d':
                tf = TimeFrame.Day
            elif timeframe == '1H':
                tf = TimeFrame(1, TimeFrameUnit.Hour)
            else:
                tf = TimeFrame(15, TimeFrameUnit.Minute)
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                limit=50000
            )
            bars = self.data_client.get_stock_bars(request)
            # alpaca-py BarSet: bars[symbol] is a list of Bar, not a DataFrame.
            # Use bars.df to get the combined DataFrame, then filter by symbol.
            if bars is not None:
                try:
                    df = bars.df
                    if 'symbol' in df.index.names:
                        # FIX #27: Always filter by symbol before droplevel to avoid
                        # returning mixed-symbol data when API returns multiple symbols
                        if symbol in df.index.get_level_values('symbol'):
                            df = df.xs(symbol, level='symbol')
                        else:
                            # Symbol not in index — drop level but warn (may be single-symbol response)
                            logger.debug(f"Symbol '{symbol}' not in multi-index levels — dropping 'symbol' level")
                            df = df.droplevel('symbol')
                    elif 'symbol' in df.columns:
                        df = df[df['symbol'] == symbol].drop(columns=['symbol'])
                    if not df.empty:
                        df = df[['open', 'high', 'low', 'close', 'volume']]
                        return df
                except (KeyError, AttributeError) as parse_err:
                    logger.debug(f"Alpaca bars parsing for {symbol}: {parse_err}")
        except Exception as e:
            logger.error(f"Alpaca fetch error for {symbol} {timeframe}: {type(e).__name__}: {e}")
            self._increment_api_failures()
        return pd.DataFrame()

    def _fetch_polygon_data(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        # FIX #22: Guard against None client — return empty DataFrame immediately
        if self.polygon_client is None:
            return pd.DataFrame()
        # CRIT-13 FIX: Only hold lock for timestamp check + update, NOT during the API call or sleep.
        # Old code held the lock during time.sleep(30) on 429, blocking ALL concurrent Polygon calls.
        with self.polygon_lock:
            now = time.time()
            # Use last_polygon_call_time as the baseline (may be in the future if slots are reserved)
            earliest_allowed = self.last_polygon_call_time + self.polygon_min_delay
            sleep_needed = max(0, earliest_allowed - now)
            # Reserve this slot so concurrent threads queue behind it
            self.last_polygon_call_time = max(now, earliest_allowed)
        # Sleep outside the lock so other threads can proceed
        if sleep_needed > 0:
            logger.debug(f"Polygon rate-limit delay: sleeping {sleep_needed:.1f}s before next call")
            time.sleep(sleep_needed)
        try:
            if timeframe == '1d':
                multiplier, timespan = 1, 'day'
            elif timeframe == '1H':
                multiplier, timespan = 1, 'hour'
            else:
                multiplier, timespan = 15, 'minute'
            aggs = self.polygon_client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                # M15 FIX: Use millisecond timestamps for intraday (avoids Polygon date parsing issues)
                from_=int(start.timestamp() * 1000) if timespan != 'day' else start.strftime('%Y-%m-%d'),
                to=int(end.timestamp() * 1000) if timespan != 'day' else end.strftime('%Y-%m-%d'),
                limit=50000
            )
            if aggs:
                df = pd.DataFrame([{
                    'open': a.open, 'high': a.high, 'low': a.low,
                    'close': a.close, 'volume': a.volume
                } for a in aggs])
                df['timestamp'] = pd.to_datetime([a.timestamp for a in aggs], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)
                return df
        except (requests.exceptions.RequestException, MaxRetryError, ValueError, TypeError) as e:
            if "429" in str(e):
                logger.warning(f"Polygon 429 rate limit hit for {symbol} — forcing extra 30s cooldown")
                # Push back the next allowed call time (outside lock to avoid blocking)
                with self.polygon_lock:
                    self.last_polygon_call_time = time.time() + 30
                time.sleep(30)
            logger.error(f"Polygon exception for {symbol} {timeframe}: {e}")
            self._increment_api_failures()
        return pd.DataFrame()

    def _fetch_finnhub_data(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        time.sleep(self.config['REQUEST_INTERVAL'])
        try:
            resolution = 'D' if timeframe == '1d' else ('60' if timeframe == '1H' else '15')
            from_ts = int(start.timestamp())
            to_ts = int(end.timestamp())
            res = self.finnhub_client.stock_candles(symbol, resolution, from_ts, to_ts)
            if res and res.get('s') == 'ok' and len(res.get('t', [])) > 0:
                df = pd.DataFrame({
                    'open': res['o'],
                    'high': res['h'],
                    'low': res['l'],
                    'close': res['c'],
                    'volume': res['v']
                })
                df['timestamp'] = pd.to_datetime(res['t'], unit='s', utc=True)
                df.set_index('timestamp', inplace=True)
                return df
        # M16 FIX: Catch all exceptions — JSONDecodeError, KeyError, unexpected response structures
        except Exception as e:
            logger.debug(f"Finnhub fetch error for {symbol} {timeframe}: {e}")
            self._increment_api_failures()
        return pd.DataFrame()

    def _fetch_tiingo_data(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        # FIX #22: Guard against None client — return empty DataFrame immediately
        if self.tiingo_client is None:
            return pd.DataFrame()
        time.sleep(self.config['REQUEST_INTERVAL'])
        try:
            freq = 'daily' if timeframe == '1d' else ('1hour' if timeframe == '1H' else '15min')
            data = self.tiingo_client.get_dataframe(
                symbol,
                startDate=start.strftime('%Y-%m-%d'),
                endDate=end.strftime('%Y-%m-%d'),
                frequency=freq
            )
            if not data.empty:
                data.index = pd.to_datetime(data.index, utc=True)
                data.columns = data.columns.str.lower()
                return data
        except (requests.exceptions.RequestException, MaxRetryError, ValueError, TypeError) as e:
            logger.error(f"Tiingo exception for {symbol} {timeframe}: {e}")
            self._increment_api_failures()
        return pd.DataFrame()

    def _fetch_yfinance_data(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        time.sleep(self.config['REQUEST_INTERVAL'])
        try:
            interval = '1d' if timeframe == '1d' else ('1h' if timeframe.lower() in ('1h', '60min') else '15m')
            if interval != '1d' and (end - start).days > 60:
                logger.debug(f"Skipping yFinance for {symbol} {timeframe}: range >60 days")
                return pd.DataFrame()
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start, end=end, interval=interval, auto_adjust=False, repair=True)
            if not data.empty:
                data = data[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                })
                data.index = pd.to_datetime(data.index, utc=True)
                return data
        except (requests.exceptions.RequestException, MaxRetryError, ValueError, TypeError) as e:
            logger.error(f"yFinance exception for {symbol} {timeframe}: {e}")
            self._increment_api_failures()
        return pd.DataFrame()

    def get_bid_ask_spread(self, symbol: str) -> float:
        try:
            # FIX #17: alpaca-py v0.10+ uses StockLatestQuoteRequest from alpaca.data.requests.
            # Earlier versions accept bare string. Try request object first, fall back to bare string.
            try:
                from alpaca.data.requests import StockLatestQuoteRequest
                quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = self.data_client.get_stock_latest_quote(quote_request)
            except (ImportError, TypeError):
                # Older alpaca-py: pass symbol directly
                quotes = self.data_client.get_stock_latest_quote(symbol)
            # Response is a dict keyed by symbol
            quote = quotes.get(symbol) if isinstance(quotes, dict) else quotes
            if quote is not None:
                bid = float(getattr(quote, 'bid_price', 0) or 0)
                ask = float(getattr(quote, 'ask_price', 0) or 0)
                if bid > 0 and ask > 0 and ask > bid:
                    return (ask - bid) / ((ask + bid) / 2)
        except Exception as e:
            logger.debug(f"Alpaca quote failed for {symbol}: {e}")
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')
            if not hist.empty:
                # L8 FIX: Use configured spread instead of hardcoded 0.001
                return self.config.get('ASSUMED_SPREAD', 0.001)
        except (requests.exceptions.RequestException, MaxRetryError, ValueError, TypeError) as e:
            logger.debug(f"yFinance quote fallback failed for {symbol}: {e}")
        logger.warning(f"[SPREAD] All quote sources failed for {symbol} — using hardcoded 1% spread")
        return 0.01

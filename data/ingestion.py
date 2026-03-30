# data/ingestion.py
import logging
import asyncio
import threading
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz
from alpaca.data.live import StockDataStream
from .handler import DataHandler
from newsapi import NewsApiClient  # ← Added for full causal LLM refinement
import time  # ← For time.time() in polling

logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, config, symbols, timeframes):
        self.config = config
        self.symbols = symbols
        self.timeframes = timeframes
        self.data_handler = DataHandler(config)
        # NOTE: stream_client is now created FRESH inside stream_data() on every reconnect attempt
        # Do NOT create it here — prevents reusing broken client after crash
        # FIX #23: Don't create NewsApiClient with api_key=None — every call silently fails.
        _news_key = config.get('NEWS_API_KEY')
        if _news_key:
            self.news_api = NewsApiClient(api_key=_news_key)
        else:
            self.news_api = None
            logger.warning("[INGESTION] NEWS_API_KEY not set — NewsApiClient disabled. "
                           "Sentiment features will use fallback values.")
        
        # Ensure '1d' is always in data_store so UniverseManager and daily cache work
        all_timeframes = set(timeframes + ['1d'])
        self.data_store = {
            symbol: {
                tf: pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                for tf in all_timeframes
            }
            for symbol in symbols
        }
        self._data_store_lock = threading.RLock()  # Protects data_store mutations across threads
        self.last_bar_time = {symbol: None for symbol in symbols}
        # FIX #44: Track the active stream client so we can resubscribe after universe rotation
        self._active_stream_client = None
        self._subscribed_symbols = set(symbols)

    def _ensure_symbol_in_store(self, symbol: str):
        """Lazily initialize data_store entry for a symbol added via universe rotation."""
        with self._data_store_lock:
            if symbol not in self.data_store:
                all_timeframes = set(self.timeframes + ['1d'])
                self.data_store[symbol] = {
                    tf: pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                    for tf in all_timeframes
                }
                self.last_bar_time[symbol] = None
                logger.info(f"[LAZY INIT] Created data_store entry for new symbol {symbol}")

    def update_symbols(self, new_symbols: list):
        """Update symbols list and resubscribe the stream for any newly added symbols.
        Called after universe rotation adds/removes symbols."""
        added = set(new_symbols) - self._subscribed_symbols
        removed = self._subscribed_symbols - set(new_symbols)
        self.symbols = list(new_symbols)
        # Ensure data_store entries exist for new symbols
        for sym in added:
            self._ensure_symbol_in_store(sym)
        # Resubscribe stream for new symbols
        if added and self._active_stream_client is not None:
            try:
                self._active_stream_client.subscribe_bars(self.handle_alpaca_bar, *added)
                self._subscribed_symbols.update(added)
                logger.info(f"[STREAM RESUBSCRIBE] Subscribed to {len(added)} new symbols: {', '.join(sorted(added))}")
            except Exception as e:
                logger.error(f"[STREAM RESUBSCRIBE] Failed to subscribe new symbols: {e}")
        if removed:
            # FIX #33: Unsubscribe removed symbols from the stream to stop receiving stale bars
            if self._active_stream_client is not None:
                try:
                    self._active_stream_client.unsubscribe_bars(*removed)
                    logger.info(f"[STREAM UNSUBSCRIBE] Unsubscribed {len(removed)} removed symbols: {', '.join(sorted(removed))}")
                except Exception as e:
                    # NOTE: If the Alpaca SDK version doesn't support unsubscribe_bars,
                    # removed symbols will continue streaming until next reconnect.
                    logger.warning(f"[STREAM UNSUBSCRIBE] Failed to unsubscribe removed symbols: {e}")
            logger.info(f"[UNIVERSE ROTATION] Removed {len(removed)} symbols: {', '.join(sorted(removed))}")
            self._subscribed_symbols -= removed

    async def handle_alpaca_bar(self, bar):
        symbol = bar.symbol
        self._ensure_symbol_in_store(symbol)
        # FIX #71: Compute cutoff once per bar event (not per timeframe iteration)
        self._stream_cutoff = datetime.now(tz=tz.gettz('UTC')) - timedelta(days=180)
        # FIX #1: Don't floor timestamp — Alpaca provides correct bar timestamps.
        # Resampling on lines below handles aggregation per timeframe.
        timestamp = pd.to_datetime(bar.timestamp, utc=True)
        data = {
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
        for timeframe in self.timeframes:
            # FIX #32: Skip daily timeframe for stream resampling — intraday bars produce
            # incomplete daily candles. Daily data should only come from historical fetches.
            if timeframe in ('1d', '1D', 'day', 'daily'):
                continue
            # FIX #25: Read under lock, process outside lock, write under lock.
            # Resample+ffill are CPU-intensive — holding the lock blocks all other data access.
            with self._data_store_lock:
                df = self.data_store[symbol][timeframe].copy()
            # --- Append new bar (outside lock) ---
            # FIX #35: Use try/except with get_loc (O(1) for sorted unique index)
            try:
                df.index.get_loc(timestamp)
                _ts_exists = True
            except KeyError:
                _ts_exists = False
            if _ts_exists:
                df.loc[timestamp] = list(data.values())
            else:
                new_row = pd.DataFrame([data], index=[timestamp])
                df = pd.concat([df, new_row])
            # Deduplicate and sort
            df = df[~df.index.duplicated(keep='last')].sort_index()
            # Resample, ffill, trim — CPU-intensive, done outside lock
            # FIX #70: Extended resample mapping for additional timeframe aliases
            resample_rule = {
                '15Min': '15min', '5Min': '5min', '30Min': '30min',
                '1H': '1h', '60min': '1h', '4H': '4h', '1d': '1D'
            }.get(timeframe, '15min')
            df = df.resample(resample_rule).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            })
            # FIX #40: Drop rows where all OHLCV are NaN (gap periods after reconnect)
            # before ffill, so 'open' doesn't pick up stale data from previous period
            df = df.dropna(how='all')
            df = df.ffill()
            # FIX #71: Cutoff computed per-iteration was wasteful; now uses outer cutoff
            cutoff = self._stream_cutoff
            df = df[df.index >= cutoff]
            # --- Write back under lock ---
            with self._data_store_lock:
                self.data_store[symbol][timeframe] = df
            # FIX #2: cache_data does file I/O — fire-and-forget in a thread to avoid blocking event loop
            # FIX #26: Add done_callback to log exceptions instead of silently swallowing them
            def _cache_done_callback(fut, _sym=symbol, _tf=timeframe):
                exc = fut.exception()
                if exc is not None:
                    logger.error(f"[CACHE] cache_data failed for {_sym} {_tf}: {type(exc).__name__}: {exc}")
            # FIX #19: Pass df.copy() to prevent race if another bar mutates df before thread serializes
            cache_fut = asyncio.ensure_future(asyncio.to_thread(self.data_handler.cache_data, symbol, timeframe, df.copy()))
            cache_fut.add_done_callback(_cache_done_callback)
        self.last_bar_time[symbol] = timestamp
        # M17 FIX: Use actual timeframe key (was hardcoded '15Min' which crashes if not in data_store)
        tf_key = timeframe if timeframe in self.data_store.get(symbol, {}) else list(self.data_store.get(symbol, {}).keys())[0] if self.data_store.get(symbol) else '15Min'
        logger.debug(f"Stream bar added {symbol} | {tf_key} len={len(self.data_store.get(symbol, {}).get(tf_key, []))} | last={timestamp}")

    async def _close_stream_client(self, client):
        """Cleanly stop a StockDataStream client and drain any dangling coroutines."""
        if client is None:
            return
        try:
            client.stop()
            logger.debug("Stopped old StockDataStream client")
        except Exception as e:
            logger.debug(f"stop() on old client raised (safe to ignore): {e}")
        await asyncio.sleep(0)

    async def stream_data(self):
        logger.info("Starting live data stream (async)")
        
        MAX_RETRIES = 20
        BASE_BACKOFF_SEC = 5
        
        retry_count = 0
        while retry_count < MAX_RETRIES:
            logger.info(f"Stream connection attempt {retry_count + 1}/{MAX_RETRIES}")
            
            client = None
            stream_task = None
            try:
                # Create fresh client — NO 'paper' argument
                logger.info("Creating fresh Alpaca StockDataStream instance")
                logger.debug(f"Using API key (first 5 chars): {self.config['API_KEY'][:5]}... | "
                             f"PAPER mode expected: {self.config.get('PAPER', True)} "
                             f"(environment determined by keys)")
                
                client = StockDataStream(
                    self.config['API_KEY'],
                    self.config['API_SECRET']
                )
                
                # Subscribe to bars (queued until connected)
                logger.info(f"Subscribing to bars for {len(self.symbols)} symbols: {', '.join(self.symbols)}")
                client.subscribe_bars(self.handle_alpaca_bar, *self.symbols)
                self._active_stream_client = client
                self._subscribed_symbols = set(self.symbols)
                
                # FIX: Do NOT call client.run() — it uses asyncio.run() internally which fails
                # Instead, directly run the internal coroutine as a task in this event loop
                # WARNING #20: _run_forever() is an alpaca-py private API. No public coroutine alternative exists.
                # If alpaca-py internals change, this will raise AttributeError and fall through to retry.
                logger.info("Starting internal stream coroutine via create_task (avoids nested asyncio.run)")
                if not hasattr(client, '_run_forever'):
                    raise AttributeError("alpaca-py StockDataStream._run_forever() no longer exists — library update may have broken internal API")
                stream_task = asyncio.create_task(client._run_forever())
                
                # Give it a moment to start connecting and log progress
                for _ in range(30):  # up to ~30 seconds initial wait to observe
                    await asyncio.sleep(1)
                    if stream_task.done():
                        # If it finished already (likely error), let exception handler catch it
                        break
                    logger.debug("Stream task running — waiting for connection/auth...")
                
                # If task is still running after initial wait → assume it's healthy and let it continue forever
                if not stream_task.done():
                    logger.info("Stream task appears healthy — awaiting indefinitely")
                    # FIX #3: Reset retry_count on healthy connection (survived initial health check)
                    retry_count = 0
                    await stream_task  # this line blocks forever unless exception or cancellation
                
                # FIX #44: Non-exception stream completion still needs retry increment
                retry_count += 1
                logger.warning(f"Stream task completed unexpectedly (clean disconnect) — retry_count={retry_count}")

            except Exception as e:
                logger.error(f"Stream fatal error: {type(e).__name__}: {e}", exc_info=True)
                retry_count += 1
                backoff = BASE_BACKOFF_SEC * (2 ** min(retry_count, 6))  # cap ~320s
                logger.info(f"Retrying in {backoff} seconds...")
                await asyncio.sleep(backoff)
            
            finally:
                # Proper cleanup
                if stream_task is not None and not stream_task.done():
                    stream_task.cancel()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        logger.debug("Stream task cancelled cleanly")
                    except Exception as cancel_err:
                        logger.debug(f"Error during task cancel: {cancel_err}")
                
                if client is not None:
                    await self._close_stream_client(client)
                    client = None

        logger.critical(
            f"Failed to establish Alpaca stream after {MAX_RETRIES} attempts — "
            f"giving up. Check API keys (must be PAPER keys if PAPER=True), network, or Alpaca status."
        )

    def get_latest_data(self, symbol: str, timeframe: str = '15Min', lookback_days: int = 60) -> pd.DataFrame:
        """Fetch latest data for a symbol. Returns cached data immediately if available.
        Only performs blocking HTTP fetch when cache is empty/short — callers in async
        context should wrap with asyncio.to_thread() for that case."""
        self._ensure_symbol_in_store(symbol)
        with self._data_store_lock:
            df = self.data_store[symbol].get(timeframe, pd.DataFrame())
        logger.debug(f"get_latest_data {symbol} {timeframe}: cache len={len(df)}, last_bar={df.index[-1] if not df.empty else 'empty'}")
        if not df.empty and len(df) >= 100:
            # Cache hit — no blocking I/O, return immediately
            return df.ffill()
        # M18: Cache miss — blocking HTTP fetch. All async call sites MUST wrap this
        # in asyncio.to_thread(). Warning logged once per session as a safety net.
        try:
            asyncio.get_running_loop()
            if not getattr(self, '_event_loop_warned', False):
                logger.warning(
                    f"[M18] get_latest_data() blocking HTTP fetch for {symbol} {timeframe} on event loop — "
                    "caller should use asyncio.to_thread()"
                )
                self._event_loop_warned = True
        except RuntimeError:
            pass  # Synchronous context — blocking is fine
        logger.info(f"Cache short/empty for {symbol} {timeframe} — fetching hist")
        end = datetime.now(tz=tz.gettz('UTC'))
        start = end - timedelta(days=lookback_days + 90)
        df = self.data_handler.fetch_data(symbol, timeframe, start, end, for_live_trading=True)
        if not df.empty:
            df = df[~df.index.duplicated(keep='last')].sort_index().ffill()
            with self._data_store_lock:
                self.data_store[symbol][timeframe] = df
            self.data_handler.cache_data(symbol, timeframe, df)
            logger.info(f"Fetched hist for {symbol} {timeframe}: len={len(df)}")
        df = df.ffill()
        return df

    def initialize_data(self):
        end_date = datetime.now(tz=tz.gettz('UTC'))
        start_date = end_date - timedelta(days=365)
        for symbol in self.symbols:
            # M19 FIX: Use self.timeframes with config fallback (prevents KeyError)
            for timeframe in self.config.get('TIMEFRAMES', getattr(self, 'timeframes', ['15Min', '1H'])):
                cached = self.data_handler.load_cached_data(symbol, timeframe)
                if cached is not None and len(cached) >= 100:
                    cached = cached[~cached.index.duplicated(keep='last')].sort_index().ffill()
                    with self._data_store_lock:
                        self.data_store[symbol][timeframe] = cached
                    logger.info(f"Loaded cached {symbol} {timeframe} len={len(cached)}")
                    continue
                data = self.data_handler.fetch_data(symbol, timeframe, start_date, end_date)
                if not data.empty:
                    data = data[~data.index.duplicated(keep='last')].sort_index().ffill()
                    with self._data_store_lock:
                        self.data_store[symbol][timeframe] = data
                    self.data_handler.cache_data(symbol, timeframe, data)
                    logger.info(f"Initialized {symbol} {timeframe} len={len(data)}")

    def get_recent_news(self, symbol: str, days: int = 10) -> list:
        # FIX #23: Guard against None news_api — return empty list immediately
        if self.news_api is None:
            return []
        try:
            now = datetime.now(tz=tz.gettz('UTC'))
            from_date = (now - timedelta(days=days)).strftime('%Y-%m-%d')
            to_date = now.strftime('%Y-%m-%d')
            articles = self.news_api.get_everything(
                q=symbol,
                from_param=from_date,
                to=to_date,
                language='en',
                sort_by='relevancy',
                page_size=15
            )['articles']
            news_texts = [
                a['title'] + " " + (a.get('description') or "")
                for a in articles
                if a.get('title') or a.get('description')
            ]
            logger.info(f"[CAUSAL NEWS] Fetched {len(news_texts)} real news items for {symbol}")
            return news_texts
        except Exception as e:
            logger.warning(f"[CAUSAL NEWS] News fetch failed for {symbol}: {e} — returning empty list")
            return []

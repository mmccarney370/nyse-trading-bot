# data/ingestion.py
import logging
import asyncio
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
        self.news_api = NewsApiClient(api_key=config['NEWS_API_KEY'])  # ← Added for causal wrapper
        
        # Ensure '1d' is always in data_store so UniverseManager and daily cache work
        all_timeframes = set(timeframes + ['1d'])
        self.data_store = {
            symbol: {
                tf: pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                for tf in all_timeframes
            }
            for symbol in symbols
        }
        self.last_bar_time = {symbol: None for symbol in symbols}

    def _ensure_symbol_in_store(self, symbol: str):
        """Lazily initialize data_store entry for a symbol added via universe rotation."""
        if symbol not in self.data_store:
            all_timeframes = set(self.timeframes + ['1d'])
            self.data_store[symbol] = {
                tf: pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                for tf in all_timeframes
            }
            self.last_bar_time[symbol] = None
            logger.info(f"[LAZY INIT] Created data_store entry for new symbol {symbol}")

    async def handle_alpaca_bar(self, bar):
        symbol = bar.symbol
        self._ensure_symbol_in_store(symbol)
        timestamp = pd.to_datetime(bar.timestamp, utc=True).floor('15min')
        data = {
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
        for timeframe in self.timeframes:
            df = self.data_store[symbol][timeframe]
            if timestamp in df.index:
                df.loc[timestamp] = list(data.values())
            else:
                new_row = pd.DataFrame([data], index=[timestamp])
                df = pd.concat([df, new_row])
            # Deduplicate and sort
            df = df[~df.index.duplicated(keep='last')].sort_index()
        
            # CRIT-05 FIX: Proper resample rule per timeframe (was always '15min')
            resample_rule = {
                '15Min': '15min',
                '1H': '1h',
                '60min': '1h',
                '1d': '1D'
            }.get(timeframe, '15min')
            df = df.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        
            # FIX: Ffill gaps to help indicators
            df = df.ffill()
            # Keep recent
            cutoff = datetime.now(tz=tz.gettz('UTC')) - timedelta(days=180)
            df = df[df.index >= cutoff]
            self.data_store[symbol][timeframe] = df
            self.data_handler.cache_data(symbol, timeframe, df)
        self.last_bar_time[symbol] = timestamp
        logger.debug(f"Stream bar added {symbol} | 15min len={len(self.data_store[symbol]['15Min'])} | last={timestamp}")

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
                
                # FIX: Do NOT call client.run() — it uses asyncio.run() internally which fails
                # Instead, directly run the internal coroutine as a task in this event loop
                logger.info("Starting internal stream coroutine via create_task (avoids nested asyncio.run)")
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
                    await stream_task  # this line blocks forever unless exception or cancellation
                
                logger.info("Stream task completed unexpectedly — will retry")
                
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
        self._ensure_symbol_in_store(symbol)
        df = self.data_store[symbol].get(timeframe, pd.DataFrame())
        logger.debug(f"get_latest_data {symbol} {timeframe}: cache len={len(df)}, last_bar={df.index[-1] if not df.empty else 'empty'}")
        if df.empty or len(df) < 100:
            logger.info(f"Cache short/empty for {symbol} {timeframe} — fetching hist")
            end = datetime.now(tz=tz.gettz('UTC'))
            start = end - timedelta(days=lookback_days + 90)
            df = self.data_handler.fetch_data(symbol, timeframe, start, end, for_live_trading=True)
            if not df.empty:
                df = df[~df.index.duplicated(keep='last')].sort_index().ffill()
                self.data_store[symbol][timeframe] = df
                self.data_handler.cache_data(symbol, timeframe, df)
                logger.info(f"Fetched hist for {symbol} {timeframe}: len={len(df)}")
        df = df.ffill()
        return df

    def initialize_data(self):
        end_date = datetime.now(tz=tz.gettz('UTC'))
        start_date = end_date - timedelta(days=365)
        for symbol in self.symbols:
            for timeframe in self.config['TIMEFRAMES']:
                cached = self.data_handler.load_cached_data(symbol, timeframe)
                if cached is not None and len(cached) >= 100:
                    cached = cached[~cached.index.duplicated(keep='last')].sort_index().ffill()
                    self.data_store[symbol][timeframe] = cached
                    logger.info(f"Loaded cached {symbol} {timeframe} len={len(cached)}")
                    continue
                data = self.data_handler.fetch_data(symbol, timeframe, start_date, end_date)
                if not data.empty:
                    data = data[~data.index.duplicated(keep='last')].sort_index().ffill()
                    self.data_store[symbol][timeframe] = data
                    self.data_handler.cache_data(symbol, timeframe, data)
                    logger.info(f"Initialized {symbol} {timeframe} len={len(data)}")

    def get_recent_news(self, symbol: str, days: int = 10) -> list:
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

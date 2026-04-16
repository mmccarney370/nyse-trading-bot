# bot.py
# =====================================================================
import logging
import asyncio
from datetime import datetime, timedelta
from dateutil import tz
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor # ← C-4 FIX: Changed to ThreadPoolExecutor (no pickling issues)
import threading
import json
import os
import time # ← Added for correct background thread sleep
from collections import Counter
import tempfile # ← Priority 1: for atomic writes
import shutil # ← Priority 1: for atomic rename
from config import CONFIG
from data.ingestion import DataIngestion
from broker.alpaca import AlpacaBroker
from broker.stream import TradeStreamHandler
from models.trainer import Trainer
from strategy.signals import SignalGenerator # CausalSignalWrapper removed (Phase 1 modularization)
from strategy.risk import RiskManager
from strategy.portfolio_rebalancer import PortfolioRebalancer # ← NEW IMPORT (Phase 1)
from models.causal_rl_manager import CausalRLManager # ← NEW IMPORT (Phase 2)
from models.bot_initializer import BotInitializer # ← NEW IMPORT (Phase 3)
from models.causal_signal_manager import CausalSignalManager # ← NEW: Phase 1 modularization
from backtest import Backtester
from utils.helpers import is_market_open, time_until_next_open # ← UPDATED IMPORT (Bug #1 fix)
from strategy.regime import detect_regime, is_trending
from strategy.universe import UniverseManager
from gemini_tuner import query_gemini_for_tuning, load_dynamic_config
from models.features import generate_features
from models.portfolio_env import PortfolioEnv # B-12: required for real portfolio obs
# NEW IMPORT FOR SHUTDOWN FIX (matches exactly how GetOrdersRequest is used in alpaca.py)
from alpaca.trading.requests import GetOrdersRequest
# ─── CLEAN LOGGING SETUP (fixes double logging in nyse_bot.log) ───────────────────────────────
# NOTE: This runs at import time (before __main__.py calls setup_logging()).
# It configures the bot.py module logger specifically to prevent duplicate handlers.
# setup_logging() in __main__.py configures the root logger for all other modules.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Remove any existing handlers to prevent duplication
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
# File handler (single, clean log file)
file_handler = logging.FileHandler("nyse_bot.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s'))
logger.addHandler(file_handler)
# Console handler (only warnings+ for less spam)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(console_handler)
# Prevent propagation to root logger (main cause of duplicates)
logger.propagate = False
# Silence noisy third-party loggers
logging.getLogger("alpaca").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
# Path for persistent regime cache
REGIME_CACHE_FILE = "regime_cache.json"
# Path for persistent live signal history (B-17)
HISTORY_FILE = "live_signal_history.json"
# NEW: Path for persistent last_entry_times (Bug #6 fix)
LAST_ENTRY_FILE = "last_entry_times.json"
# NEW: Path for persistent portfolio causal ReplayBuffer (Bug #22 fix)
PORTFOLIO_REPLAY_BUFFER_FILE = "portfolio_replay_buffer.json"
# Path for persistent starting equity (survives restarts for accurate Gemini return_pct)
STARTING_EQUITY_FILE = "starting_equity.json"
# ==================== CONFIGURABLE SCHEDULE TIMES ====================
GEMINI_HOUR = 3
GEMINI_MINUTE = 30
REGIME_HOUR = 4
REGIME_MINUTE = 0
# =====================================================================
# ==================== TOP-LEVEL HELPER FOR BACKGROUND COMPUTATION ====================
def compute_regime(sym, lookback, config_dict=None):
    """Heavy computation — used ONLY in background 4 AM task.
    CRITICAL FIX: Now receives live config_dict from main process so Gemini updates are seen."""
    if config_dict is None:
        from config import CONFIG
        config_dict = CONFIG
    start = datetime.now()
    try:
        temp_ingestion = DataIngestion(config_dict, [sym], config_dict['TIMEFRAMES'])
        data = temp_ingestion.get_latest_data(sym, timeframe='15Min')
        if len(data) < 50:
            return sym, ('mean_reverting', 0.5)
        regime_tuple = detect_regime(
            data=data,
            symbol=sym,
            data_ingestion=temp_ingestion,
            lookback=lookback,
            verbose=False
        )
        # Defensive unpack — ensure we always store a clean (regime, persistence) tuple
        if isinstance(regime_tuple, (list, tuple)) and len(regime_tuple) == 2:
            regime, persistence = regime_tuple
        else:
            regime, persistence = str(regime_tuple), 0.5
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Regime computed for {sym} in {duration:.1f}s → {regime} (persistence={persistence:.3f})")
        return sym, (regime, persistence)
    except Exception as e:
        logger.error(f"[REGIME ERROR] Failed to compute regime for {sym}: {e}", exc_info=True)
        return sym, ('mean_reverting', 0.5) # safe fallback
class TradingBot:
    def __init__(self, config):
        self.config = config
        self.data_ingestion = DataIngestion(config, config['SYMBOLS'], config['TIMEFRAMES'])
        self.broker = AlpacaBroker(config, self.data_ingestion, bot=self)
        self.trainer = Trainer(config, self.data_ingestion)
        # Persistent regime cache — MOVED UP so it exists before SignalGenerator (BUG #4 FIX)
        self._regime_lock = threading.Lock()
        self.regime_cache = self._load_regime_cache()
        self._cleanup_old_regimes()
        # BUG #4 FIX: Shared regime cache with bot.py (4AM precompute + _get_all_regimes now syncs live to signal generation)
        self.signal_gen = SignalGenerator(config, self.data_ingestion, self.trainer, regime_cache=self.regime_cache)
        self.risk_manager = RiskManager(config, self.data_ingestion)
        self.backtester = Backtester(config, self.data_ingestion, self.trainer, self.signal_gen, self.risk_manager)
        # === NEW: PortfolioRebalancer (Phase 1 extraction) ===
        self.rebalancer = PortfolioRebalancer(config, self.signal_gen, self.risk_manager)
        # === NEW: CausalRLManager (Phase 2 extraction) ===
        self.live_signal_history = {}
        self._signal_history_lock = threading.Lock()  # protects live_signal_history mutations (shared with stream thread)
        self.causal_manager = CausalRLManager(self.signal_gen, self.config, signal_history_lock=self._signal_history_lock)
        # === NEW: BotInitializer (Phase 3 extraction) ===
        self.initializer = BotInitializer(self)
        self.daily_equity = {}
        self.equity_history = {}
        self._signal_history_max = 1000  # M50 FIX: Match disk cap (was 500 vs disk 1000)
        # BUG-13 FIX: Instance-level latest_prices cache — must be initialized BEFORE sharing with SignalGenerator
        self.latest_prices = {}
        self._prices_lock = threading.Lock()  # H21 FIX: protects latest_prices from concurrent mutation
        # BUG-12 FIX + ISSUE #6 PATCH: Persistent PortfolioEnv instance (created once, reused every cycle)
        self.portfolio_env = None
        # CRIT-1/2 FIX: Share live_signal_history and latest_prices with SignalGenerator
        # so sync_daily_rewards(), _monitor_oos_decay(), and causal warmup operate on real data
        self.signal_gen.live_signal_history = self.live_signal_history
        self.signal_gen.latest_prices = self.latest_prices
        self.cycle_count = 0
        self.performance_check_interval = 10
        self.portfolio_ppo = config.get('PORTFOLIO_PPO', False)
        # ISSUE #5 FIX: Pass CONFIG to UniverseManager (required after universe.py update)
        self.universe_manager = UniverseManager(self.data_ingestion, self.live_signal_history, CONFIG)
        self.last_universe_update = datetime.now(tz=tz.gettz('America/New_York'))
        # S1: Meta-labeling filter (refit nightly from live_signal_history)
        from strategy.meta_filter import MetaLabeler
        self.meta_labeler = MetaLabeler(symbols=config['SYMBOLS'])
        # Share with signal_gen so generate_portfolio_actions can consult it
        self.signal_gen.meta_labeler = self.meta_labeler
        # B5: Anti-earnings filter — auto-flat before earnings, block new entries
        from strategy.earnings_filter import EarningsCalendar
        self.earnings_calendar = EarningsCalendar(symbols=config['SYMBOLS'])
        self.signal_gen.earnings_calendar = self.earnings_calendar
        # B2: Adverse selection detector — track post-fill drift, flag toxic fills
        from strategy.adverse_selection import AdverseSelectionDetector
        self.adverse_selection = AdverseSelectionDetector()
        self.signal_gen.adverse_selection = self.adverse_selection
        self.broker.adverse_selection = self.adverse_selection  # broker will record fills
        # BPS: Bayesian per-symbol sizing (size by posterior P(win))
        from strategy.bayesian_sizing import BayesianSymbolSizer
        self.bayesian_sizer = BayesianSymbolSizer(symbols=config['SYMBOLS'])
        self.signal_gen.bayesian_sizer = self.bayesian_sizer
        # ESP: Execution slippage predictor (skip entries with predicted > alpha)
        from strategy.slippage_predictor import SlippagePredictor
        self.slippage_predictor = SlippagePredictor()
        self.signal_gen.slippage_predictor = self.slippage_predictor
        self.broker.slippage_predictor = self.slippage_predictor  # broker records on fill
        # EQ-SCORE: Daily execution-quality scorecard (aggregates gate/reject telemetry)
        from strategy.execution_scorecard import ExecutionScorecard
        self.execution_scorecard = ExecutionScorecard()
        self.signal_gen.execution_scorecard = self.execution_scorecard
        self.broker.execution_scorecard = self.execution_scorecard
        # ALPHA-ATTR: Per-closed-trade attribution logger
        from strategy.alpha_attribution import AlphaAttribution
        self.alpha_attribution = AlphaAttribution()
        self.signal_gen.alpha_attribution = self.alpha_attribution
        self.broker.alpha_attribution = self.alpha_attribution
    def _load_regime_cache(self):
        # FIX (Apr 16 gap #3): Regime detection code now uses S2 ensemble (slope +
        # ADX + autocorr + range + HMM). Old cache entries were produced by the
        # legacy HMM-only heuristic. Version-bump the cache so stale entries are
        # discarded — first call to _get_all_regimes will repopulate via the new
        # detect_regime which exercises the S2 voters.
        EXPECTED_VERSION = 2
        # Also honor a config flag to force refresh (useful for development)
        if CONFIG.get('REGIME_FORCE_REFRESH_ON_STARTUP', False):
            logger.info("[REGIME CACHE] REGIME_FORCE_REFRESH_ON_STARTUP=True — starting fresh")
            return {}
        if os.path.exists(REGIME_CACHE_FILE):
            try:
                with open(REGIME_CACHE_FILE, 'r') as f:
                    data = json.load(f)
                # Version check: if missing/mismatched, discard (triggers S2 recompute)
                file_version = data.pop('__version__', 1)
                if file_version != EXPECTED_VERSION:
                    logger.info(f"[REGIME CACHE] Version {file_version} != expected {EXPECTED_VERSION} — "
                                f"discarding {len(data)} stale entries, will re-detect via S2 ensemble")
                    return {}
                # Validate types: ensure persistence values are float (JSON may deserialize as int)
                for key, value in data.items():
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        data[key] = (value[0], float(value[1]))
                    elif isinstance(value, str):
                        data[key] = (value, 0.5)
                logger.info(f"Loaded persistent regime cache with {len(data)} symbols (v{EXPECTED_VERSION})")
                return data
            except Exception as e:
                logger.warning(f"Failed to load regime cache: {e}")
        return {}
    def _save_regime_cache(self):
        """Atomic save for regime cache (Priority 1 fix).
        Embeds __version__=2 so startup can discard stale pre-S2 entries."""
        try:
            path = REGIME_CACHE_FILE
            dir_name = os.path.dirname(path) or '.'
            snapshot = {'__version__': 2, **self.regime_cache}
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=dir_name) as tmp:
                json.dump(snapshot, tmp, default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else (x.tolist() if isinstance(x, np.ndarray) else str(x)))
                tmp.flush()
                os.fsync(tmp.fileno())
            shutil.move(tmp.name, path)
            logger.debug(f"[ATOMIC SAVE] Saved regime cache with {len(self.regime_cache)} symbols")
        except Exception as e:
            logger.warning(f"Failed to save regime cache: {e}")
    # FIX: _save_last_entry_times removed — use self.broker._save_last_entry_times() to avoid
    # race condition with broker's own saves (both were writing to the same file)
    def _save_live_signal_history(self):
        """B-28: Save pruned live_signal_history atomically (Priority 1 fix)"""
        try:
            # Snapshot under lock to avoid racing with stream thread mutations
            with self._signal_history_lock:
                serializable_history = {}
                for sym, entries in self.live_signal_history.items():
                    serializable_entries = []
                    for entry in entries:
                        serial_entry = entry.copy()
                        if 'timestamp' in serial_entry and isinstance(serial_entry['timestamp'], datetime):
                            serial_entry['timestamp'] = serial_entry['timestamp'].isoformat()
                        serializable_entries.append(serial_entry)
                    # Keep last 1000 per symbol (consistent with shutdown cap)
                    serializable_history[sym] = serializable_entries[-1000:]

            path = HISTORY_FILE
            dir_name = os.path.dirname(path) or '.'
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=dir_name) as tmp:
                json.dump(serializable_history, tmp, indent=2, default=str)
                tmp.flush()
                os.fsync(tmp.fileno())
            shutil.move(tmp.name, path)
            logger.debug(f"[ATOMIC SAVE B-28] Saved pruned live_signal_history to disk ({sum(len(v) for v in serializable_history.values())} entries)")
        except Exception as e:
            logger.warning(f"[B-28] Failed atomic save live_signal_history: {e}")
    def _load_starting_equity(self, fallback: float) -> float:
        """Load persistent starting equity — survives restarts so Gemini return_pct is accurate."""
        if os.path.exists(STARTING_EQUITY_FILE):
            try:
                with open(STARTING_EQUITY_FILE, 'r') as f:
                    data = json.load(f)
                return float(data.get('starting_equity', fallback))
            except Exception:
                pass
        # First run — save current equity as starting point
        self._save_starting_equity(fallback)
        return fallback

    def _save_starting_equity(self, equity: float):
        """Save starting equity atomically."""
        try:
            path = STARTING_EQUITY_FILE
            dir_name = os.path.dirname(path) or '.'
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=dir_name) as tmp:
                json.dump({'starting_equity': equity, 'saved_at': datetime.now(tz=tz.gettz('UTC')).isoformat()}, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
            shutil.move(tmp.name, path)
        except Exception as e:
            logger.warning(f"Failed to save starting equity: {e}")

    def _emergency_save_all(self):
        """B-33: Centralized emergency save — now using atomic writes where possible"""
        saved = 0
        try:
            self._save_live_signal_history()
            saved += 1
        except Exception as e:
            logger.error(f"[B-33] Failed atomic save live_signal_history: {e}")
        try:
            self.broker._save_last_entry_times()
            saved += 1
        except Exception as e:
            logger.error(f"[B-33] Failed atomic save last_entry_times: {e}")
        try:
            self._save_regime_cache()
            saved += 1
        except Exception as e:
            logger.error(f"[B-33] Failed atomic save regime_cache: {e}")
        try:
            # ISSUE P12 / BUG-09 FIX: Call save_buffer on the correct object (portfolio_causal_manager)
            if hasattr(self.signal_gen, 'portfolio_causal_manager') and self.signal_gen.portfolio_causal_manager is not None:
                self.signal_gen.portfolio_causal_manager.save_buffer()
                saved += 1
            else:
                logger.debug("[B-33] No portfolio_causal_manager — skipping buffer save")
        except Exception as e:
            logger.error(f"[B-33] Failed to save portfolio ReplayBuffer: {e}")
        logger.info(f"[B-33] Emergency atomic save attempt complete — {saved}/4 succeeded")
    def _cleanup_old_regimes(self):
        if not self.regime_cache:
            return
        cleaned = {}
        for key, value in self.regime_cache.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                cleaned[key] = value
            elif isinstance(value, str):
                cleaned[key] = (value, 0.5)
            else:
                cleaned[key] = ('mean_reverting', 0.5)
        with self._regime_lock:
            self.regime_cache.clear()
            self.regime_cache.update(cleaned)
        logger.debug(f"Regime cache cleaned — now keeping only latest entry per symbol ({len(cleaned)} entries)")
    def time_until_market_open(self):
        """Delegates to the fixed version in helpers.py (Bug #1 patched)."""
        delta = time_until_next_open()
        return delta.total_seconds()
    async def _universe_update_task(self):
        logger.info("Universe update task started — will now run only on Fridays at 8:00 PM ET")
        while True:
            now = datetime.now(tz=tz.gettz('America/New_York'))
            # Calculate next Friday 8:00 PM
            days_ahead = (4 - now.weekday()) % 7 # 4 = Friday
            if days_ahead == 0 and now.hour >= 20:
                days_ahead = 7
            target = now + timedelta(days=days_ahead)
            target = target.replace(hour=20, minute=0, second=0, microsecond=0)
            sleep_seconds = (target - now).total_seconds()
            logger.info(f"Next universe rotation scheduled for {target.strftime('%Y-%m-%d %H:%M:%S %Z')} "
                        f"({sleep_seconds/3600:.1f} hours from now)")
            await asyncio.sleep(sleep_seconds)
            try:
                logger.info("Running dynamic universe evaluation (Friday 8PM rotation)...")
                # HIGH-23 FIX: evaluate_universe() is blocking (34 symbols × 5 HMMs) — run in thread
                new_symbols = await asyncio.to_thread(self.universe_manager.evaluate_universe)
                if set(new_symbols) != set(self.config['SYMBOLS']):
                    await self._perform_universe_rotation(new_symbols)
                else:
                    logger.info("Universe unchanged — no rotation needed")
            except Exception as e:
                logger.error(f"Universe update failed: {e}", exc_info=True)
    async def _perform_universe_rotation(self, new_symbols: list):
        """Shared rotation logic — called from both Friday task and Gemini intra-week trigger."""
        old_symbols = set(self.config['SYMBOLS'])
        logger.info(f"Universe rotation triggered: {self.config['SYMBOLS']} → {new_symbols}")
        self.config['SYMBOLS'] = new_symbols
        self.data_ingestion.symbols = new_symbols
        await asyncio.to_thread(self.data_ingestion.initialize_data)
        # CRIT-1 FIX: Mutate in-place to preserve shared reference with signal_gen
        with self._signal_history_lock:
            pruned_history = {s: self.live_signal_history.get(s, []) for s in new_symbols}
            self.live_signal_history.clear()
            self.live_signal_history.update(pruned_history)
        # B-24 FIX: Also prune last_entry_times and save file to prevent stale timestamps/bloat for removed symbols
        self.broker.last_entry_times = {s: self.broker.last_entry_times.get(s) for s in new_symbols if s in self.broker.last_entry_times}
        await asyncio.to_thread(self.broker._save_last_entry_times)
        # B-26 FIX: Also prune regime_cache for removed symbols and save to prevent stale regimes
        with self._regime_lock:
            pruned = {s: self.regime_cache.get(s) for s in new_symbols if s in self.regime_cache}
            self.regime_cache.clear()
            self.regime_cache.update(pruned)
        self._save_regime_cache()
        # H21 FIX: Protect latest_prices mutation with lock (shared with regime thread)
        with self._prices_lock:
            pruned_prices = {s: self.latest_prices.get(s) for s in new_symbols if s in self.latest_prices}
            self.latest_prices.clear()
            self.latest_prices.update(pruned_prices)
        # ISSUE #6 PATCH: Update persistent portfolio_env with new symbols' data (lightweight)
        if self.portfolio_env is not None:
            self.portfolio_env.data_dict = {sym: self.data_ingestion.get_latest_data(sym, timeframe='15Min') for sym in new_symbols}
            logger.debug("[PORTFOLIO ENV] Updated data_dict after universe rotation (persistent env refreshed)")
        logger.info("Starting full retrain on new universe...")
        await asyncio.to_thread(self.trainer.train_symbols_parallel, new_symbols, True)
        # BUG #19 PATCH: Rebuild causal wrappers for the new universe (new symbols must get causal graph + ReplayBuffer)
        # M53 FIX: Use self.config consistently (CONFIG is the module-level dict, self.config is the instance ref)
        if self.config.get('USE_CAUSAL_RL', False):
            await asyncio.to_thread(self.signal_gen.refresh_causal_wrappers)
            logger.info("Causal wrappers refreshed for new universe symbols after rotation")
            # C-3 SAFETY: Ensure graphs are ready right after refresh/warmup
            if self.signal_gen.portfolio_causal_manager:
                self.signal_gen.portfolio_causal_manager._ensure_graph_exists()
                logger.debug("[C-3] Ensured portfolio causal graph ready after rotation refresh")
            for sym, mgr in self.signal_gen.causal_manager.items():
                mgr._ensure_graph_exists()
            # BUG #20 + #23: Full rotation reset now delegated to CausalRLManager
            self.causal_manager.reset_on_rotation(new_symbols)
        logger.info("Universe rotation complete")
    # ==================== HELPER: SAFE CLOSE VIA RISK MANAGER ====================
    # C-2 FUTURE-PROOFING: If you ever refactor to use risk_manager.safe_close_position instead of direct broker call,
    # use this helper — it handles the sync call correctly (no await needed after risk.py patch)
    async def safe_close_via_manager(self, symbol: str) -> bool:
        """Wrapper around risk_manager.safe_close_position (async after C-2 patch)."""
        if not hasattr(self, 'risk_manager') or self.risk_manager is None:
            logger.error("RiskManager not initialized — cannot safely close")
            return False
        success = await self.risk_manager.safe_close_position(symbol)
        if success:
            logger.info(f"Safely closed {symbol} via RiskManager")
        else:
            logger.error(f"Failed to safely close {symbol} via RiskManager")
        return success
    # ==================== TRADING LOOP (now proper class method — Bug #5 fixed) ====================
    async def trading_loop(self):
        while True:
            if not is_market_open():
                sleep_seconds = max(self.time_until_market_open(), 60)
                logger.info(f"Market closed — sleeping {sleep_seconds / 3600:.1f} hours until next open")
                await asyncio.sleep(sleep_seconds)
                continue
            current_equity = await asyncio.to_thread(self.broker.get_equity)
            # FIX #4: Use ET timezone for daily equity key — NYSE operates on ET
            today = datetime.now(tz=tz.gettz('America/New_York')).date()
            if today not in self.daily_equity:
                self.daily_equity[today] = current_equity
            self.equity_history[today] = current_equity
            # Prune equity history to last 90 trading days to prevent memory bloat
            if len(self.equity_history) > 90:
                sorted_dates = sorted(self.equity_history.keys())
                for d in sorted_dates[:-90]:
                    del self.equity_history[d]
            if len(self.daily_equity) > 90:
                sorted_dates = sorted(self.daily_equity.keys())
                for d in sorted_dates[:-90]:
                    del self.daily_equity[d]
            paused = self.risk_manager.check_pause_conditions(
                current_equity, self.daily_equity, self.equity_history
            )
            if paused:
                logger.warning("Trading paused due to risk conditions")
                await asyncio.sleep(300)
                continue
            regimes = await asyncio.to_thread(self._get_all_regimes)
            positions = await asyncio.to_thread(self.broker.get_positions_dict)
            # === B2: Update adverse-selection post-fill drift samples ===
            if (CONFIG.get('ADVERSE_SELECTION_ENABLED', True)
                    and hasattr(self, 'adverse_selection') and self.latest_prices):
                try:
                    await asyncio.to_thread(self.adverse_selection.update_prices,
                                              dict(self.latest_prices))
                except Exception as e:
                    logger.debug(f"[ADVERSE-SEL] update_prices failed: {e}")
            # === B5: Pre-earnings auto-close ===
            # If we're holding a position and earnings is within N days, flatten.
            # Runs BEFORE signal generation so the closed slot can be reused.
            if (CONFIG.get('EARNINGS_FILTER_ENABLED', True)
                    and hasattr(self, 'earnings_calendar')
                    and positions):
                close_pre = CONFIG.get('EARNINGS_CLOSE_PRE_DAYS', 1)
                for sym, qty in list(positions.items()):
                    if qty == 0:
                        continue
                    should_close, reason = self.earnings_calendar.should_close_before_earnings(
                        sym, close_pre_days=close_pre
                    )
                    if should_close:
                        logger.warning(f"[EARNINGS AUTO-CLOSE] {sym} — {reason} — flattening position")
                        try:
                            await asyncio.to_thread(self.broker.close_position_safely, sym)
                        except Exception as e:
                            logger.error(f"[EARNINGS AUTO-CLOSE] {sym} close failed: {e}")
                # Re-read positions after any closures
                positions = await asyncio.to_thread(self.broker.get_positions_dict)
            if self.portfolio_ppo and self.trainer.portfolio_ppo_model is not None:
                logger.debug("Generating portfolio-level actions via multi-asset PPO")
                try:
                    # BUG-11 FIX: data_dict + prices fetched ONCE here and reused everywhere below (no redundant get_latest_data calls)
                    data_dict = {}
                    prices = {}
                    for sym in self.config['SYMBOLS']:
                        df = self.data_ingestion.get_latest_data(sym, timeframe='15Min')
                        if len(df) >= 200:
                            data_dict[sym] = df
                            prices[sym] = df['close'].iloc[-1]
                    # BUG-13 FIX: Update persistent instance cache so gemini_scheduled_task / _monitor_oos_decay can reuse it
                    # CRIT-2 FIX: Mutate in-place to preserve shared reference with signal_gen
                    # CRIT-4 FIX: Atomic update under lock — clear+update was racy (other threads see empty dict)
                    with self._signal_history_lock:
                        self.latest_prices.clear()
                        self.latest_prices.update(prices)
                    if len(data_dict) < 2:
                        logger.warning(f"Only {len(data_dict)} symbol(s) have sufficient data — need at least 2, skipping portfolio inference")
                        await asyncio.sleep(self.config['TRADING_INTERVAL'])
                        continue
                    if len(data_dict) < len(self.config['SYMBOLS']):
                        missing = set(self.config['SYMBOLS']) - set(data_dict.keys())
                        logger.info(f"Proceeding with {len(data_dict)}/{len(self.config['SYMBOLS'])} symbols (excluded {missing} — insufficient bars)")
                    # C-3 SAFETY: Ensure portfolio causal graph is ready before rebalance/inference
                    if self.signal_gen.portfolio_causal_manager:
                        self.signal_gen.portfolio_causal_manager._ensure_graph_exists()
                        if self.signal_gen.portfolio_causal_manager.causal_graph is not None:
                            logger.debug("[C-3] Portfolio causal graph ready for inference")
                        else:
                            logger.debug("[C-3] Portfolio causal graph deferred — neutral penalty until built")
                    # ISSUE #6 PATCH: Pass persistent self.portfolio_env to rebalance_portfolio
                    # NEW: also pass daily_equity + live_signal_history for intraday risk pacing
                    target_weights_dict = await self.rebalancer.rebalance_portfolio(
                        current_equity=current_equity,
                        data_dict=data_dict,
                        prices=prices,
                        regimes=regimes,
                        positions=positions,
                        precomputed_env=self.portfolio_env, # Persistent env (ISSUE #6)
                        daily_equity=self.daily_equity,
                        live_signal_history=self.live_signal_history,
                    )
                    # ====================== END REBALANCER DELEGATION ======================
                    # Only the final order execution loop remains here (min-hold, close, bracket)
                    for sym in self.config['SYMBOLS']:
                        target_weight = target_weights_dict.get(sym, 0.0)
                        price = prices.get(sym)
                        if price is None or price <= 0.0:
                            continue
                        raw_qty = (target_weight * current_equity) / price
                        # Use fractional shares if enabled, otherwise truncate to int
                        if self.config.get('FRACTIONAL_SHARES', True):
                            target_qty = round(raw_qty, 4)  # Alpaca supports up to 9 decimals
                        else:
                            target_qty = int(raw_qty)
                        current_qty = positions.get(sym, 0)
                        # CRIT-9 FIX: Treat dust positions (< 0.01 shares) as zero — partial fills
                        # or fractional close can leave tiny remnants that block new entries.
                        dust_threshold = self.config.get('DUST_THRESHOLD', 0.01)
                        if abs(current_qty) < dust_threshold:
                            current_qty = 0
                        direction = 1 if target_qty > 0 else (-1 if target_qty < 0 else 0)
                        # === MIN-HOLD CHECK (Gemini-tuned, consistent with signals.py) ===
                        # Only enforce min-hold when we HAVE a position — no position means no hold to enforce
                        last_entry = self.broker.last_entry_times.get(sym)
                        if last_entry and current_qty != 0:
                            # FIX #24: Market-hours-aware bars_since (26 fifteen-min bars per trading day)
                            elapsed = datetime.now(tz=tz.gettz('UTC')) - last_entry
                            elapsed_days = elapsed.total_seconds() / 86400
                            if elapsed_days <= 1:
                                bars_since = elapsed / pd.Timedelta(minutes=15)
                            else:
                                full_days = int(elapsed_days)
                                partial_frac = elapsed_days - full_days
                                bars_since = full_days * 26 + partial_frac * 26
                            regime = regimes.get(sym, ('mean_reverting', 0.5))
                            # Defensive tuple unpack for regime
                            if isinstance(regime, (list, tuple)):
                                regime = regime[0]
                            # FIX: Add fallback defaults — config.get() returns None if key missing, causing TypeError on < comparison
                            if is_trending(regime):
                                min_hold = self.config.get('MIN_HOLD_BARS_TRENDING', 6)
                            else:
                                min_hold = self.config.get('MIN_HOLD_BARS_MEAN_REVERTING', 3)
                            if bars_since < min_hold:
                                # HIGH-1 FIX: Use tolerance-based float comparison instead of exact equality
                                # and only block direction changes, not size adjustments
                                qty_diff = abs(abs(target_qty) - abs(current_qty))
                                same_direction = np.sign(target_qty) == np.sign(current_qty)
                                if not same_direction or qty_diff > max(0.01, abs(current_qty) * 0.05):
                                    logger.debug(f"MIN-HOLD ACTIVE {sym} → skipping rebalance (Gemini-tuned {min_hold} bars)")
                                    continue
                        # Respect existing positions
                        if abs(current_qty) > 0 and np.sign(current_qty) == np.sign(target_qty) and abs(target_qty) > 0:
                            logger.debug(f"{sym} already has position in correct direction — skipping new bracket")
                            continue
                        if (current_qty * direction <= 0 and current_qty != 0) or (target_qty == 0 and current_qty != 0):
                            try:
                                # B-02 / B-10 FIX: Use safe async method (awaited) instead of raw client.close_position
                                # C-2 NOTE: If you ever refactor this to use risk_manager.safe_close_position,
                                # do NOT add 'await' — after patch it's synchronous.
                                # Current direct broker call is correct and unaffected.
                                logger.debug(f"Preparing safe close for {sym} (awaiting close_position_safely)")
                                success = await self.broker.close_position_safely(sym)
                                if success:
                                    logger.info(f"PORTFOLIO PPO SAFE CLOSE {sym} completed")
                                    # Update local qty so the entry block below doesn't use stale value
                                    current_qty = 0
                                    if self.broker.tracker.groups.get(sym):
                                        self.broker.tracker.mark_closed(sym)
                                        self.broker.tracker.remove_group(sym)
                                        logger.debug(f"[TRACKER CLEANUP] Removed tracker group for {sym} after safe close")
                                else:
                                    logger.error(f"PORTFOLIO PPO SAFE CLOSE {sym} failed")
                                    continue  # Don't attempt entry if close failed
                                if sym in self.broker.last_entry_times:
                                    del self.broker.last_entry_times[sym]
                                    logger.debug(f"[B-22] Cleared last_entry_time for closed {sym}")
                                    await asyncio.to_thread(self.broker._save_last_entry_times)
                            except Exception as e:
                                logger.error(f"Failed to safely close {sym}: {e}")
                                continue  # Don't attempt entry if close raised
                        min_qty = 0.01 if self.config.get('FRACTIONAL_SHARES', True) else 1
                        if target_qty != 0 and current_qty * direction <= 0 and abs(target_qty) >= min_qty:
                            # H22 FIX: Read existing_positions under lock to prevent race with sync thread
                            with self.broker._positions_lock:
                                n_positions = len(self.broker.existing_positions)
                            if n_positions >= self.config.get('MAX_POSITIONS', 5):
                                logger.info(f"Max positions reached ({n_positions}/{self.config.get('MAX_POSITIONS', 5)}) — skipping entry for {sym}")
                                continue
                            size = abs(target_qty)
                            regime = regimes.get(sym, ('mean_reverting', 0.5))
                            # Defensive tuple unpack for regime
                            if isinstance(regime, (list, tuple)):
                                regime = regime[0]
                            # ====================== BUG #1 FIX START ======================
                            # HIGH-2 FIX: confidence should be the PPO signal strength (0-1), not the portfolio weight.
                            # target_weights_dict contains allocation weights (e.g. 0.15), not signal confidence.
                            # Use abs(weight) normalized to 0-1 range as a proxy for PPO confidence.
                            raw_weight = target_weights_dict.get(sym, 0.0)
                            ppo_strength = min(1.0, abs(raw_weight) / max(0.01, max(abs(w) for w in target_weights_dict.values())))
                            confidence = ppo_strength
                            conviction = ppo_strength
                            # ====================== BUG #1 FIX END ======================
                            # M-5 FIX: Cap size by real-time buying power before placing order
                            buying_power = await asyncio.to_thread(self.broker.get_buying_power)
                            safety_factor = self.config.get('MAX_ORDER_NOTIONAL_PCT', 0.85)
                            if self.config.get('FRACTIONAL_SHARES', True):
                                max_affordable = round(buying_power * safety_factor / price, 4) if price > 0 else 0
                            else:
                                max_affordable = int(buying_power * safety_factor / price) if price > 0 else 0
                            if size > max_affordable:
                                logger.warning(f"[M-5 BUYING POWER CAP] {sym}: requested {size} shares → reduced to {max_affordable} "
                                               f"(buying_power=${buying_power:,.0f}, safety_factor={safety_factor})")
                                size = max(max_affordable, 0)
                            if size < (0.001 if self.config.get('FRACTIONAL_SHARES', True) else 1):
                                logger.warning(f"[BUYING POWER] {sym}: insufficient buying power — skipping order")
                                continue
                            order = await asyncio.to_thread(
                                self.broker.place_bracket_order,
                                symbol=sym,
                                size=size,
                                current_price=price,
                                data=data_dict[sym],
                                direction=direction
                            )
                            if order:
                                logger.info(f"PORTFOLIO PPO {'LONG' if direction > 0 else 'SHORT'} {sym} {size} shares @ {price:.2f} (regime {regime})")
                                # ====================== BUG #1 FIX (symbol → sym) ======================
                                self.broker.last_entry_times[sym] = datetime.now(tz=tz.gettz('UTC'))
                                # B-32 FIX: Store original observation for accurate causal reward push later
                                # FIX #6: Pass full history from data_ingestion as full_hist_df (not windowed slice)
                                full_hist = self.data_ingestion.get_latest_data(sym, timeframe='15Min')
                                features = await asyncio.to_thread(generate_features, data_dict[sym], regime, sym, full_hist)
                                obs_for_storage = features[-1:].astype(np.float32).flatten().tolist() if features is not None and features.shape[0] > 0 else []
                                with self._signal_history_lock:
                                    hist = self.live_signal_history.setdefault(sym, [])
                                    hist.append({
                                        'timestamp': datetime.now(tz=tz.gettz('UTC')),
                                        'direction': direction,
                                        'price': price,
                                        'confidence': confidence,
                                        'ppo_strength': ppo_strength,
                                        'conviction': conviction,
                                        'realized_return': None,
                                        'size': size, # B-15: store entry size for accurate dollar P&L later
                                        'obs': obs_for_storage # ← BUG-2 FIX
                                    })
                                    if len(hist) > self._signal_history_max:
                                        self.live_signal_history[sym] = hist[-self._signal_history_max:]
                                await asyncio.to_thread(self.broker._save_last_entry_times) # B-23: immediate save after new entry
                                await asyncio.to_thread(self._save_live_signal_history) # FIX: persist signal history off async loop
                except Exception as e:
                    logger.error(f"Portfolio PPO inference/rebalance failed: {e}", exc_info=True)
            else:
                for symbol in self.config['SYMBOLS']:
                    signal_data = self.data_ingestion.get_latest_data(symbol, timeframe='15Min')
                    if len(signal_data) < 200:
                        continue
                    timestamp = signal_data.index[-1]
                    price = signal_data['close'].iloc[-1]
                    self.latest_prices[symbol] = price # BUG #13 PATCH: populate cache in non-portfolio mode so gemini_scheduled_task + _monitor_oos_decay can reuse it
                    regime_tuple = regimes.get(symbol, ('mean_reverting', 0.5))
                    regime = regime_tuple[0] if isinstance(regime_tuple, (list, tuple)) else regime_tuple
                    persistence = regime_tuple[1] if isinstance(regime_tuple, (list, tuple)) else 0.5
                    # P-10 / Critical #1 FIX: Added await — method is async, must be awaited
                    direction, confidence, ppo_strength, _ = await self.signal_gen.generate_signal(
                        symbol=symbol,
                        data=signal_data,
                        timestamp=timestamp,
                        live_mode=True
                    )
                    # B-09 FIX: Populate ReplayBuffer with real transition for causal RL
                    if symbol in self.signal_gen.causal_manager and self.signal_gen.causal_manager[symbol] is not None: # ← UPDATED for Phase 1
                        # FIX #6: Pass full history from data_ingestion as full_hist_df (not windowed slice)
                        full_hist = self.data_ingestion.get_latest_data(symbol, timeframe='15Min')
                        features = await asyncio.to_thread(generate_features, signal_data, regime, symbol, full_hist)
                        if features is not None and features.shape[0] > 0:
                            obs = features[-1:].astype(np.float32).reshape(1, -1)
                            action_for_buffer = direction * confidence # signed final action
                            self.signal_gen.causal_manager[symbol].add_transition(obs, action_for_buffer, 0.0) # reward updated later in OOS monitor
                    prev_direction = self.signal_gen.prev_signals.get(symbol, 0)
                    conviction = 0.3 * confidence + 0.7 * ppo_strength
                    conviction = np.clip(conviction, 0.0, 1.0)
                    # FIX: Block shorts in strong trending regimes
                    regime_override_threshold = self.config.get('REGIME_OVERRIDE_PERSISTENCE', 0.80)
                    if direction == -1 and is_trending(regime) and persistence >= regime_override_threshold:
                        logger.warning(f"[REGIME OVERRIDE] {symbol} SHORT blocked — strong trending regime "
                                       f"(persistence={persistence:.3f})")
                        direction = 0
                    if direction != 0 and prev_direction == 0:
                        # MAX_POSITIONS enforcement: skip new entries when at capacity
                        if len(self.broker.existing_positions) >= self.config.get('MAX_POSITIONS', 5):
                            logger.info(f"Max positions reached ({len(self.broker.existing_positions)}/{self.config.get('MAX_POSITIONS', 5)}) — skipping entry for {symbol}")
                            continue
                        # CRIT-7 FIX: Pre-fetch buying power and cache on risk_manager
                        # so calculate_position_size doesn't make a blocking sync API call
                        buying_power = await asyncio.to_thread(self.broker.get_buying_power)
                        self.risk_manager._cached_buying_power = buying_power
                        size = self.risk_manager.calculate_position_size(
                            equity=current_equity,
                            price=price,
                            symbol=symbol,
                            data=signal_data,
                            conviction=conviction,
                            regime=regime,
                            persistence=persistence # ← NEW: pass regime confidence for dynamic sizing
                        )
                        # FIX: If fractional shares enabled, allow sizes as small as 0.01 (Alpaca minimum)
                        min_size = 0.01 if self.config.get('FRACTIONAL_SHARES', True) else 1
                        if size >= min_size:
                            # M-5 FIX: Cap size by real-time buying power before placing order
                            safety_factor = self.config.get('MAX_ORDER_NOTIONAL_PCT', 0.85)
                            if self.config.get('FRACTIONAL_SHARES', True):
                                max_affordable = round(buying_power * safety_factor / price, 4) if price > 0 else 0
                            else:
                                max_affordable = int(buying_power * safety_factor / price) if price > 0 else 0
                            if size > max_affordable:
                                logger.warning(f"[M-5 BUYING POWER CAP] {symbol}: requested {size} shares → reduced to {max_affordable} "
                                               f"(buying_power=${buying_power:,.0f}, safety_factor={safety_factor})")
                                size = max(max_affordable, 0)
                            if size < (0.001 if self.config.get('FRACTIONAL_SHARES', True) else 1):
                                logger.warning(f"[BUYING POWER] {symbol}: insufficient buying power — skipping order")
                                continue
                            order = await asyncio.to_thread(
                                self.broker.place_bracket_order,
                                symbol=symbol,
                                size=size,
                                current_price=price,
                                data=signal_data,
                                direction=direction
                            )
                            if order:
                                logger.info(f"Entered {'LONG' if direction > 0 else 'SHORT'} {symbol} {size} shares (regime {regime})")
                                self.broker.last_entry_times[symbol] = datetime.now(tz=tz.gettz('UTC'))
                                # B-32 FIX: Store original observation for accurate causal reward push later
                                # FIX #6: Pass full history from data_ingestion as full_hist_df (not windowed slice)
                                full_hist = self.data_ingestion.get_latest_data(symbol, timeframe='15Min')
                                features = await asyncio.to_thread(generate_features, signal_data, regime, symbol, full_hist)
                                obs_for_storage = features[-1:].astype(np.float32).flatten().tolist() if features is not None and features.shape[0] > 0 else []
                                with self._signal_history_lock:
                                    hist = self.live_signal_history.setdefault(symbol, [])
                                    hist.append({
                                        'timestamp': datetime.now(tz=tz.gettz('UTC')),
                                        'direction': direction,
                                        'price': price,
                                        'confidence': confidence,
                                        'ppo_strength': ppo_strength,
                                        'conviction': conviction,
                                        'realized_return': None,
                                        'size': size, # B-15: store entry size for accurate dollar P&L later
                                        'obs': obs_for_storage # ← BUG-2 FIX
                                    })
                                    if len(hist) > self._signal_history_max:
                                        self.live_signal_history[symbol] = hist[-self._signal_history_max:]
                                await asyncio.to_thread(self.broker._save_last_entry_times) # B-23: immediate save after new entry
                                await asyncio.to_thread(self._save_live_signal_history) # FIX: persist signal history off async loop
                    # HIGH-4 FIX: Close on flat signal OR signal reversal (e.g. long→short)
                    # direction==0 means flat; direction*current_qty<0 means reversal
                    current_pos = positions.get(symbol, 0)
                    should_close = (
                        (direction == 0 and current_pos != 0) or
                        (direction != 0 and current_pos != 0 and direction * current_pos < 0)
                    )
                    if should_close:
                        close_reason = "flat signal" if direction == 0 else f"signal reversal ({'long→short' if current_pos > 0 else 'short→long'})"
                        try:
                            logger.debug(f"Preparing safe close for {symbol} ({close_reason}, awaiting close_position_safely)")
                            success = await self.broker.close_position_safely(symbol)
                            if success:
                                logger.info(f"Closed position in {symbol} ({close_reason}) — safe close completed")
                                if self.broker.tracker.groups.get(symbol):
                                    self.broker.tracker.mark_closed(symbol)
                                    self.broker.tracker.remove_group(symbol)
                                    logger.debug(f"[TRACKER CLEANUP] Removed tracker group for {symbol} after safe close")
                            else:
                                logger.error(f"Safe close failed for {symbol} — position may still be open")
                            # B-22 FIX: Clear timestamp when position is closed so min-hold resets for next signal
                            if symbol in self.broker.last_entry_times:
                                del self.broker.last_entry_times[symbol]
                                logger.debug(f"[B-22] Cleared last_entry_time for closed {symbol}")
                                await asyncio.to_thread(self.broker._save_last_entry_times)
                        except Exception as e:
                            logger.error(f"Failed to safely close position in {symbol}: {e}")
            self.cycle_count += 1
            if self.cycle_count % self.performance_check_interval == 0:
                await asyncio.to_thread(self.signal_gen._monitor_oos_decay)
            await asyncio.sleep(self.config['TRADING_INTERVAL'])
    async def run(self):
        """Now extremely clean — only orchestration (Phase 3)"""
        logger.info("=== Starting Bot Initialization ===")
        await asyncio.to_thread(self.initializer.perform_full_startup)
        # ISSUE #6 PATCH: Create persistent PortfolioEnv once after data init
        if self.portfolio_ppo and self.trainer.portfolio_ppo_model is not None:
            logger.info("Creating persistent PortfolioEnv instance (ISSUE #6)")
            data_dict = {sym: self.data_ingestion.get_latest_data(sym, timeframe='15Min') for sym in self.config['SYMBOLS']}
            self.portfolio_env = PortfolioEnv(
                data_dict=data_dict,
                symbols=self.config['SYMBOLS'],
                initial_balance=await asyncio.to_thread(self.broker.get_equity),
                max_leverage=self.config.get('MAX_LEVERAGE', 2.0)
            )
            logger.info(f"Persistent PortfolioEnv created with {len(self.portfolio_env.timeline)} steps")
            # CAUSAL BOOTSTRAP: seed the portfolio replay buffer by replaying recent
            # historical bars through the env with PPO-predicted actions. Without this,
            # the portfolio causal graph stays in fallback until ≥100 live trades
            # accumulate organically (can take weeks at current trade cadence).
            if (CONFIG.get('USE_CAUSAL_RL', False)
                    and self.signal_gen.portfolio_causal_manager is not None):
                pcm = self.signal_gen.portfolio_causal_manager
                bootstrap_n = CONFIG.get('CAUSAL_BOOTSTRAP_STEPS', 1500)
                noise_sigma = CONFIG.get('CAUSAL_BOOTSTRAP_NOISE_SIGMA', 0.4)
                if bootstrap_n > 0:
                    vec_norm = getattr(self.trainer, 'portfolio_vec_norm', None)
                    added = await asyncio.to_thread(
                        pcm.bootstrap_from_env,
                        self.portfolio_env,
                        self.trainer.portfolio_ppo_model,
                        bootstrap_n,
                        vec_norm,
                        noise_sigma,
                    )
                    if added > 0:
                        pcm.save_buffer()
        # C-3 SAFETY: After full startup (including warmup_causal_buffers), ensure causal graphs are ready
        if CONFIG.get('USE_CAUSAL_RL', False):
            for sym, mgr in self.signal_gen.causal_manager.items():
                mgr._ensure_graph_exists()
            if self.signal_gen.portfolio_causal_manager:
                self.signal_gen.portfolio_causal_manager._ensure_graph_exists()
            logger.info("[C-3 STARTUP] Ensured all causal graphs are ready or deferred safely after initialization")
        threading.Thread(target=self._background_regime_precompute, daemon=True).start()
        async def data_stream_task():
            while True:
                try:
                    await self.data_ingestion.stream_data()
                except asyncio.CancelledError:
                    # HIGH-4 FIX: Let CancelledError propagate for clean shutdown
                    logger.info("[DATA STREAM] Task cancelled — shutting down cleanly")
                    raise
                except Exception as e:
                    logger.error(f"Live data stream crashed: {e}. Restarting in 10 seconds...")
                    await asyncio.sleep(10)
        # Trade updates websocket (fill-driven OCO, slippage, causal push)
        self.trade_stream_handler = TradeStreamHandler(self.broker)
        # ==================== SCHEDULED TASKS ====================
        # HIGH-5 FIX: Keep individual task references so we can cancel each one
        all_tasks = [
            asyncio.create_task(data_stream_task(), name='data_stream'),
            asyncio.create_task(self.trading_loop(), name='trading'),
            asyncio.create_task(self.broker.monitor_positions(), name='monitor'),
            asyncio.create_task(self.trade_stream_handler.run(), name='trade_stream'),
            asyncio.create_task(self._universe_update_task(), name='universe'),
            asyncio.create_task(self.gemini_scheduled_task(), name='gemini'),
            asyncio.create_task(self.ppo_nightly_retrain_task(), name='ppo_nightly'),
            asyncio.create_task(self._execution_scorecard_task(), name='eq_score'),
        ]
        # Critical tasks that must be restarted if they crash
        critical_task_names = {'trading', 'monitor', 'data_stream', 'trade_stream'}
        try:
            while True:
                done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)
                restart_needed = False
                for task in done:
                    name = task.get_name()
                    exc = task.exception() if not task.cancelled() else None
                    if exc:
                        logger.error(f"Task '{name}' crashed: {exc}")
                        if name in critical_task_names:
                            logger.warning(f"Restarting critical task '{name}' in 5s...")
                            await asyncio.sleep(5)
                            # Recreate the failed task
                            task_factories = {
                                'data_stream': lambda: asyncio.create_task(data_stream_task(), name='data_stream'),
                                'trading': lambda: asyncio.create_task(self.trading_loop(), name='trading'),
                                'monitor': lambda: asyncio.create_task(self.broker.monitor_positions(), name='monitor'),
                                'trade_stream': lambda: asyncio.create_task(self.trade_stream_handler.run(), name='trade_stream'),
                            }
                            if name in task_factories:
                                new_task = task_factories[name]()
                                all_tasks = [t for t in all_tasks if t is not task] + [new_task]
                                restart_needed = True
                                logger.info(f"Critical task '{name}' restarted")
                    elif task.cancelled():
                        logger.debug(f"Task '{name}' was cancelled")
                if not restart_needed and not pending:
                    break  # All tasks completed normally
        except asyncio.CancelledError:
            logger.info("Tasks cancelled during shutdown (Ctrl+C or external cancellation)")
        finally:
            logger.info("Shutdown signal received. Cleaning up...")
            try:
                self._emergency_save_all()
            except Exception as e:
                logger.error(f"Emergency save failed: {e}")
            try:
                open_orders = await asyncio.to_thread(self.broker.client.get_orders, GetOrdersRequest(status='open'))
                for order in open_orders:
                    await asyncio.to_thread(self.broker.client.cancel_order_by_id, order.id)
                logger.info(f"All {len(open_orders)} open orders cancelled on shutdown")
            except Exception as e:
                logger.error(f"Failed to cancel orders on shutdown: {e}")
            # HIGH-5 FIX: Cancel individual tasks, not just the gather future
            for task in all_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*all_tasks, return_exceptions=True)
    # ==================== BACKGROUND REGIME PRECOMPUTE ====================
    def _background_regime_precompute(self):
        logger.info("Background regime precompute thread started")
        while True:
            try:
                tz_et = tz.gettz('America/New_York')
                now = datetime.now(tz_et)
                target = now.replace(hour=REGIME_HOUR, minute=REGIME_MINUTE, second=0, microsecond=0)
                if now >= target:
                    target += timedelta(days=1)
                while target.weekday() >= 5:
                    target += timedelta(days=1)
                sleep_seconds = (target - now).total_seconds()
                logger.info(f"Next background regime precompute scheduled for {target.strftime('%Y-%m-%d %H:%M:%S %Z')} "
                            f"({sleep_seconds/3600:.1f} hours from now)")
                time.sleep(sleep_seconds) # ← Fixed: correct for threading context
                logger.info(f"=== Starting Background Regime Precomputation at {REGIME_HOUR:02d}:{REGIME_MINUTE:02d} AM ET ===")
                regimes = self._compute_full_regimes()
                logger.info(f"✅ Background regime precomputation completed for {len(regimes)} symbols")
            except Exception as e:
                logger.error(f"Background regime precompute failed: {e}", exc_info=True)
                time.sleep(300) # ← Fixed: correct for threading context
    def _compute_full_regimes(self):
        regimes = {}
        symbols = self.config['SYMBOLS']
        # C-4 FIX: Use ThreadPoolExecutor (no pickling required — threads share memory)
        with ThreadPoolExecutor(max_workers=min(16, len(symbols))) as executor:
            # Pass self.config directly (safe in threads) + lookback
            results = list(executor.map(
                lambda sym: compute_regime(sym, self.config.get('LOOKBACK', 900), self.config),
                symbols
            ))
        with self._regime_lock:
            for sym, regime_tuple in results:
                # Always store clean (regime, persistence) tuple under plain symbol key
                if isinstance(regime_tuple, (list, tuple)) and len(regime_tuple) == 2:
                    self.regime_cache[sym] = regime_tuple
                    regimes[sym] = regime_tuple[0]
                else:
                    self.regime_cache[sym] = (str(regime_tuple), 0.5)
                    regimes[sym] = str(regime_tuple)
        # B-07 REGIME PROPAGATION FIX: Set global CURRENT_REGIME so RiskManager (CVaR), min-hold, etc. always see the latest regime
        if regimes:
            regime_list = [r[0] if isinstance(r, (list,tuple)) else r for r in regimes.values()]
            dominant_regime = Counter(regime_list).most_common(1)[0][0]
            # FIX #30: Protect CURRENT_REGIME writes with _regime_lock — this runs from
            # _background_regime_precompute thread and would otherwise race with main thread reads
            with self._regime_lock:
                self.config['CURRENT_REGIME'] = dominant_regime
                CONFIG['CURRENT_REGIME'] = dominant_regime  # BUG #11 PATCH: also sync the imported global CONFIG
            logger.info(f"[REGIME PROPAGATION] Set CURRENT_REGIME = {dominant_regime} (dominant across {len(regimes)} symbols)")
        self._save_regime_cache()
        return regimes
    def _get_all_regimes(self):
        regimes = {}
        symbols = self.config['SYMBOLS']
        cache_misses = []
        # Phase 1: Read from cache (lock held briefly)
        with self._regime_lock:
            for sym in symbols:
                if sym in self.regime_cache:
                    value = self.regime_cache[sym]
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        regime = value[0]
                        persistence = value[1]
                    else:
                        regime = str(value)
                        persistence = 0.5
                    regimes[sym] = (regime, persistence)
                else:
                    cache_misses.append(sym)
        # Phase 2: Fetch data outside lock (no I/O under lock)
        # FIX (Apr 16 gap #3): Previously used an inline heuristic that bypassed
        # S2's regime ensemble entirely. Now delegate to detect_regime so the
        # slope/ADX/autocorr/range + HMM ensemble voters actually exercise.
        # Falls back to the simple heuristic on any error to avoid blocking trades.
        from strategy.regime import detect_regime as _detect_regime_full
        new_prices = {}
        for sym in cache_misses:
            data = self.data_ingestion.get_latest_data(sym, timeframe='15Min')
            if data is not None and len(data) > 0:
                new_prices[sym] = data['close'].iloc[-1]
            if data is None or len(data) < 50:
                regime = 'mean_reverting'
                persistence = 0.5
            else:
                try:
                    regime, persistence = _detect_regime_full(
                        data=data,
                        symbol=sym,
                        data_ingestion=self.data_ingestion,
                        lookback=self.config.get('LOOKBACK', 900),
                        is_backtest=False,
                        verbose=False,
                    )
                except Exception as e:
                    logger.debug(f"[_get_all_regimes] {sym} detect_regime failed ({e}) — heuristic fallback")
                    recent_return = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
                    returns = data['close'].pct_change().dropna().iloc[-20:]
                    rolling_vol = returns.std() if len(returns) > 1 else 0.01
                    directional_strength = abs(recent_return) / max(rolling_vol * np.sqrt(20), 1e-6)
                    if directional_strength > 1.5:
                        regime = 'trending_up' if recent_return >= 0 else 'trending_down'
                    elif directional_strength < 0.5:
                        regime = 'mean_reverting'
                    else:
                        regime = 'trending_up' if recent_return >= 0.015 else ('trending_down' if recent_return <= -0.015 else 'mean_reverting')
                    persistence = float(np.clip(0.3 + 0.5 * min(directional_strength / 2.0, 1.0), 0.3, 0.8))
            regimes[sym] = (regime, persistence)
        # Phase 3: Update cache (lock held briefly)
        if cache_misses:
            with self._regime_lock:
                for sym in cache_misses:
                    self.regime_cache[sym] = regimes[sym]
            # FIX: Update latest_prices via in-place mutation to preserve shared reference
            # and avoid torn reads from other threads
            self.latest_prices.update(new_prices)
        return regimes
    # ==================== GEMINI + CAUSAL REFRESH (3:30 AM) ====================
    async def gemini_scheduled_task(self):
        logger.info(f"Gemini tuning + Causal Refresh task scheduled — will run daily at {GEMINI_HOUR:02d}:{GEMINI_MINUTE:02d} AM ET")
        while True:
            try:
                tz_et = tz.gettz('America/New_York')
                now = datetime.now(tz_et)
                target = now.replace(hour=GEMINI_HOUR, minute=GEMINI_MINUTE, second=0, microsecond=0)
                if now >= target:
                    target += timedelta(days=1)
                while target.weekday() >= 5:
                    target += timedelta(days=1)
                sleep_seconds = (target - now).total_seconds()
                logger.info(f"Gemini + Causal Refresh next run scheduled for {target.strftime('%Y-%m-%d %H:%M:%S %Z')} "
                            f"({sleep_seconds/3600:.1f} hours from now)")
                await asyncio.sleep(sleep_seconds)
                logger.info(f"=== Starting Daily Gemini Tuning + Causal Refresh at {GEMINI_HOUR:02d}:{GEMINI_MINUTE:02d} AM ET ===")
                # Gemini tuning (enhanced with real dollar P&L)
                for attempt in range(3):
                    try:
                        current_equity = await asyncio.to_thread(self.broker.get_equity)
                        buying_power = await asyncio.to_thread(self.broker.get_buying_power)
                        # H24 FIX: _load_starting_equity does blocking file I/O — wrap in to_thread
                        starting_equity = await asyncio.to_thread(self._load_starting_equity, current_equity)
                        positions = await asyncio.to_thread(self.broker.get_positions_dict)
                        # === Aggregate trade statistics ===
                        # FIX #7: Filter to recent trades only (configurable window, default 30 days)
                        # Using all-time history biases Gemini toward stale regime conditions
                        gemini_lookback_days = self.config.get('GEMINI_HISTORY_DAYS', 30)
                        cutoff_time = datetime.now(tz=tz.gettz('UTC')) - timedelta(days=gemini_lookback_days)
                        all_closed = []
                        # HIGH-6 FIX: Snapshot live_signal_history under lock to prevent
                        # RuntimeError from concurrent modification by stream thread
                        with self._signal_history_lock:
                            history_snapshot = {sym: list(entries) for sym, entries in self.live_signal_history.items()}
                        for hist in history_snapshot.values():
                            for e in hist:
                                if e.get('realized_return') is None:
                                    continue
                                ts = e.get('timestamp', cutoff_time)
                                # Handle string timestamps from JSON-loaded history
                                if isinstance(ts, str):
                                    try:
                                        ts = datetime.fromisoformat(ts)
                                        if ts.tzinfo is None:
                                            ts = ts.replace(tzinfo=tz.gettz('UTC'))
                                    except (ValueError, TypeError):
                                        continue
                                if ts >= cutoff_time:
                                    all_closed.append(e)
                        total_closed = len(all_closed)
                        wins = sum(1 for e in all_closed if e.get('realized_return', 0) > 0)
                        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
                        # === Sharpe, Sortino, Max Drawdown from realized returns ===
                        realized_returns = [e['realized_return'] for e in all_closed if 'realized_return' in e]
                        sharpe_ratio = 0.0
                        sortino_ratio = 0.0
                        max_drawdown_pct = 0.0
                        avg_return = 0.0
                        avg_win = 0.0
                        avg_loss = 0.0
                        profit_factor = 0.0
                        avg_hold_bars = 0.0
                        if len(realized_returns) > 2:
                            avg_return = np.mean(realized_returns)
                            ret_std = np.std(realized_returns)
                            sharpe_ratio = round((avg_return / ret_std * np.sqrt(252)) if ret_std > 1e-8 else 0.0, 3)
                            downside = [r for r in realized_returns if r < 0]
                            downside_std = np.std(downside) if len(downside) > 1 else 1e-8
                            sortino_ratio = round((avg_return / downside_std * np.sqrt(252)) if downside_std > 1e-8 else 0.0, 3)
                            winners = [r for r in realized_returns if r > 0]
                            losers = [abs(r) for r in realized_returns if r < 0]
                            avg_win = round(np.mean(winners), 5) if winners else 0.0
                            avg_loss = round(np.mean(losers), 5) if losers else 0.0
                            profit_factor = round(sum(winners) / sum(losers), 3) if sum(losers) > 0 else 99.0
                            # Max drawdown from equity history (need >= 2 entries for meaningful drawdown)
                            if self.equity_history and len(self.equity_history) >= 2:
                                eq_vals = list(self.equity_history.values())
                                peak = eq_vals[0]
                                max_dd = 0.0
                                for eq in eq_vals:
                                    peak = max(peak, eq)
                                    dd = (peak - eq) / peak if peak > 0 else 0
                                    max_dd = max(max_dd, dd)
                                max_drawdown_pct = round(max_dd * 100, 2)
                        # Average hold time
                        hold_times = []
                        for e in all_closed:
                            ts = e.get('timestamp')
                            closed_ts = e.get('closed_at')
                            if ts and closed_ts:
                                try:
                                    hold_times.append((datetime.fromisoformat(str(closed_ts)) - datetime.fromisoformat(str(ts))).total_seconds() / 900)
                                except Exception:
                                    pass
                        avg_hold_bars = round(np.mean(hold_times), 1) if hold_times else 0.0
                        # === Per-symbol performance with regime breakdown ===
                        symbol_performance = {}
                        regimes = await asyncio.to_thread(self._get_all_regimes)
                        # H23 FIX: Snapshot live_signal_history under lock (shared with stream thread)
                        with self._signal_history_lock:
                            _sym_history_snap = {sym: list(self.live_signal_history.get(sym, [])) for sym in self.config['SYMBOLS']}
                        for sym in self.config['SYMBOLS']:
                            history = _sym_history_snap.get(sym, [])
                            closed = [e for e in history if e.get('realized_return') is not None]
                            current_price = self.latest_prices.get(sym)
                            sym_regime = regimes.get(sym, ('mixed', 0.5)) if regimes else ('mixed', 0.5)
                            regime_str = sym_regime[0] if isinstance(sym_regime, (list, tuple)) else sym_regime
                            persistence = sym_regime[1] if isinstance(sym_regime, (list, tuple)) and len(sym_regime) >= 2 else 0.5
                            if closed:
                                win_rate_sym = sum(1 for e in closed if e['realized_return'] > 0) / len(closed)
                                recent_win = sum(1 for e in closed[-10:] if e['realized_return'] > 0) / len(closed[-10:]) if len(closed) >= 10 else win_rate_sym
                                total_pnl_dollars = 0.0
                                recent_pnl_dollars = 0.0
                                sym_returns = []
                                # FIX #21: Pre-compute recent set to avoid O(n^2) re-slicing
                                recent_set = set(id(e) for e in closed[-10:])
                                for e in closed:
                                    size = e.get('size', 1)
                                    pnl = e['realized_return'] * e['price'] * size
                                    total_pnl_dollars += pnl
                                    sym_returns.append(e['realized_return'])
                                    if id(e) in recent_set:
                                        recent_pnl_dollars += pnl
                                sym_sharpe = 0.0
                                if len(sym_returns) > 2:
                                    s_std = np.std(sym_returns)
                                    sym_sharpe = round((np.mean(sym_returns) / s_std * np.sqrt(252)) if s_std > 1e-8 else 0.0, 3)
                            else:
                                unrealized_pnl = 0.0
                                total_pnl_dollars = 0.0
                                recent_pnl_dollars = 0.0
                                win_rate_sym = 0.5
                                recent_win = win_rate_sym
                                sym_sharpe = 0.0
                                open_entry = next((e for e in reversed(history) if e.get('realized_return') is None and e.get('direction', 0) != 0), None)
                                if open_entry and current_price is not None:
                                    direction = open_entry.get('direction', 1)
                                    entry_price = open_entry.get('price', current_price)
                                    size = open_entry.get('size', 1)
                                    unrealized = (current_price - entry_price) / entry_price * direction
                                    unrealized_pnl = unrealized * entry_price * size
                                    win_rate_sym = 1.0 if unrealized > 0 else 0.0
                                    recent_win = win_rate_sym
                                total_pnl_dollars = unrealized_pnl
                                recent_pnl_dollars = unrealized_pnl
                            symbol_performance[sym] = {
                                'win_rate': round(win_rate_sym, 3),
                                'recent_10_win_rate': round(recent_win, 3),
                                'total_pnl_dollars': round(total_pnl_dollars, 2),
                                'recent_pnl_dollars': round(recent_pnl_dollars, 2),
                                'trades': len(closed),
                                'regime': regime_str,
                                'persistence': round(persistence, 3),
                                'sharpe': sym_sharpe,
                                'current_price': round(current_price, 2) if current_price else None,
                            }
                        # === Regime breakdown ===
                        if regimes:
                            regime_list = [r[0] if isinstance(r, (list, tuple)) else r for r in regimes.values()]
                            dominant_regime = Counter(regime_list).most_common(1)[0][0] if regime_list else 'mixed'
                            regime_counts = dict(Counter(regime_list))
                            avg_persistence = round(np.mean([
                                r[1] if isinstance(r, (list, tuple)) and len(r) >= 2 else 0.5
                                for r in regimes.values()
                            ]), 3)
                        else:
                            dominant_regime = self.config.get('CURRENT_REGIME', 'mixed')
                            regime_counts = {}
                            avg_persistence = 0.5
                        # === Broker metrics ===
                        slippage_offset = self.broker.limit_price_offset
                        tracked_groups = len(self.broker.tracker.get_open_groups()) if hasattr(self.broker, 'tracker') else 0
                        # === Build full context ===
                        context = {
                            'pnl_summary': f"${current_equity:,.0f} ({(current_equity / starting_equity - 1)*100:+.2f}%)",
                            'trade_summary': str(len(all_closed)),
                            'win_rate': f"{win_rate:.1f}",
                            'regime': dominant_regime,
                            'symbol_performance': symbol_performance,
                            # New comprehensive metrics
                            'equity': round(current_equity, 2),
                            'buying_power': round(buying_power, 2),
                            'starting_equity': round(starting_equity, 2),
                            'return_pct': round((current_equity / starting_equity - 1) * 100, 3),
                            'open_positions': len(positions),
                            'tracked_orders': tracked_groups,
                            'sharpe_ratio': sharpe_ratio,
                            'sortino_ratio': sortino_ratio,
                            'max_drawdown_pct': max_drawdown_pct,
                            'profit_factor': profit_factor,
                            'avg_win': avg_win,
                            'avg_loss': avg_loss,
                            'avg_return': round(avg_return, 6),
                            'total_trades': total_closed,
                            'wins': wins,
                            'losses': total_closed - wins,
                            'avg_hold_bars': avg_hold_bars,
                            'regime_counts': regime_counts,
                            'avg_persistence': avg_persistence,
                            'current_slippage_offset': round(slippage_offset, 5),
                            'symbols_in_universe': len(self.config['SYMBOLS']),
                        }
                        logger.info(
                            f"[GEMINI] Sharpe={sharpe_ratio} Sortino={sortino_ratio} MaxDD={max_drawdown_pct}% "
                            f"PF={profit_factor} WR={win_rate:.1f}% Trades={total_closed} Regime={dominant_regime}"
                        )
                        applied = query_gemini_for_tuning(context, self.config, symbol_performance)
                        if applied:
                            logger.info(f"[GEMINI TUNER] Live config updated with {len(applied)} changes")
                            # ISSUE #5 PATCH: Check for intra-week rotation trigger from Gemini
                            if applied.get("rotate_now", False) and "SYMBOLS" in applied:
                                new_symbols = applied["SYMBOLS"]
                                logger.warning(f"[GEMINI TRIGGER] Intra-week rotation activated — poor performance detected")
                                await self._perform_universe_rotation(new_symbols)
                            # ==================== NEW: STRUCTURED BATCH SUMMARY LOG ====================
                            batch_summary = {
                                "event": "gemini_tuning_batch",
                                "timestamp": datetime.now(tz=tz.gettz('UTC')).isoformat(),
                                "changes_count": len(applied),
                                "parameters_changed": list(applied.keys()),
                                "pnl_context": context
                            }
                            logger.info(json.dumps(batch_summary, default=str))
                            # ==================== END STRUCTURED BATCH LOG ====================
                        break
                    except Exception as e:
                        # HIGH-7 FIX: Retry on all transient API errors (429, 500, 502, 503, timeout),
                        # not just 503. Non-transient errors still break immediately.
                        err_str = str(e).lower()
                        transient_indicators = ("503", "429", "500", "502", "unavailable",
                                                "timeout", "timed out", "deadline exceeded",
                                                "rate limit", "resource exhausted", "internal")
                        is_transient = any(indicator in err_str for indicator in transient_indicators)
                        if is_transient:
                            delay = 30 * (2 ** attempt)
                            logger.warning(f"[GEMINI TUNER] Attempt {attempt+1}/3 failed (transient: {type(e).__name__}). Retrying in {delay}s...")
                            await asyncio.sleep(delay)
                        else:
                            logger.error(f"[GEMINI TUNER] Non-retryable error on attempt {attempt+1}: {e}")
                            break
                # Causal graph refresh (daily at 3:30 AM)
                if CONFIG.get('USE_CAUSAL_RL', False):
                    try:
                        logger.info("Refreshing causal graph with latest full data...")
                        await asyncio.to_thread(self.signal_gen.refresh_causal_wrappers)
                        # C-3 SAFETY: Ensure graphs are ready right after daily refresh
                        if self.signal_gen.portfolio_causal_manager:
                            self.signal_gen.portfolio_causal_manager._ensure_graph_exists()
                            logger.debug("[C-3] Ensured portfolio causal graph ready after daily refresh")
                        for sym, mgr in self.signal_gen.causal_manager.items():
                            mgr._ensure_graph_exists()
                        logger.info("✅ Causal graph refresh completed")
                        # BUG #21 PATCH: After daily graph refresh, force-push any pending/unpushed realized rewards
                        # from live_signal_history into the ReplayBuffer(s) so overnight/after-hours closes are learned
                        if CONFIG.get('USE_CAUSAL_RL', False):
                            self.causal_manager.sync_daily_rewards()
                    except Exception as e:
                        logger.error(f"Causal graph refresh failed: {e}", exc_info=True)
                # B5: Earnings calendar refresh (daily at 3:30 AM, internal TTL=weekly)
                if CONFIG.get('EARNINGS_FILTER_ENABLED', True) and hasattr(self, 'earnings_calendar'):
                    try:
                        await asyncio.to_thread(
                            self.earnings_calendar.refresh, self.config['SYMBOLS']
                        )
                    except Exception as e:
                        logger.warning(f"[EARNINGS] Daily refresh failed: {e}")
                # S1: Meta-labeler refresh (daily at 3:30 AM alongside Gemini/causal)
                if CONFIG.get('META_FILTER_ENABLED', True):
                    try:
                        logger.info("Refitting meta-label filter from live_signal_history...")
                        stats = await asyncio.to_thread(
                            self.meta_labeler.fit_from_history,
                            self.live_signal_history,
                            CONFIG.get('META_FILTER_MIN_TRAIN', 30),
                            float(self.broker.get_equity() or 30000.0),
                        )
                        logger.info(f"[META-FILTER] Daily refit result: {stats}")
                    except Exception as e:
                        logger.error(f"Meta-filter refit failed: {e}", exc_info=True)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Gemini + Causal task failed: {e}", exc_info=True)
                await asyncio.sleep(300)
    async def _execution_scorecard_task(self):
        """EQ-SCORE: Emit the daily execution-quality scorecard shortly after market close.
        Runs at 16:30 ET on trading days, ~30 min after the bell. Also flushes
        on shutdown via __aexit__ semantics (naturally cleaned by task cancel)."""
        logger.info("Execution-scorecard daily emit scheduled — 16:30 ET on trading days")
        while True:
            try:
                tz_et = tz.gettz('America/New_York')
                now = datetime.now(tz_et)
                target = now.replace(hour=16, minute=30, second=0, microsecond=0)
                if now >= target:
                    target += timedelta(days=1)
                while target.weekday() >= 5:
                    target += timedelta(days=1)
                sleep_seconds = (target - now).total_seconds()
                await asyncio.sleep(sleep_seconds)
                logger.info("=== Emitting EQ-SCORE daily scorecard ===")
                # Build a snapshot of current live state
                try:
                    snapshot = {
                        "equity": float(await asyncio.to_thread(self.broker.get_equity) or 0.0),
                        "open_positions": len(await asyncio.to_thread(self.broker.get_positions_dict) or {}),
                        "portfolio_size": len(self.config.get('SYMBOLS', [])),
                    }
                    # Append Bayesian posterior snapshot if available
                    if hasattr(self, 'bayesian_sizer') and self.bayesian_sizer:
                        snapshot["bayesian_posterior"] = self.bayesian_sizer.snapshot()
                except Exception as e:
                    snapshot = {"snapshot_error": str(e)}
                if hasattr(self, 'execution_scorecard') and self.execution_scorecard:
                    try:
                        await asyncio.to_thread(self.execution_scorecard.emit, snapshot)
                    except Exception as e:
                        logger.error(f"[EQ-SCORE] emit failed: {e}", exc_info=True)
            except asyncio.CancelledError:
                # Flush one last time on clean shutdown
                try:
                    if hasattr(self, 'execution_scorecard') and self.execution_scorecard:
                        self.execution_scorecard.emit({"reason": "shutdown"})
                except Exception:
                    pass
                break
            except Exception as e:
                logger.error(f"[EQ-SCORE] task failed: {e}", exc_info=True)
                await asyncio.sleep(300)

    async def ppo_nightly_retrain_task(self):
        logger.info("PPO nightly retrain task scheduled — will run daily at 6:00 PM ET")
        while True:
            try:
                tz_et = tz.gettz('America/New_York')
                now = datetime.now(tz_et)
                target = now.replace(hour=18, minute=0, second=0, microsecond=0)
                if now >= target:
                    target += timedelta(days=1)
                while target.weekday() >= 5:
                    target += timedelta(days=1)
                sleep_seconds = (target - now).total_seconds()
                logger.info(f"PPO retrain next run scheduled for {target.strftime('%Y-%m-%d %H:%M:%S %Z')} ({sleep_seconds/3600:.1f} hours from now)")
                await asyncio.sleep(sleep_seconds)
                logger.info("=== Starting PPO Nightly Retrain at 6:00 PM ET ===")
                if self.portfolio_ppo:
                    await asyncio.to_thread(self.trainer.update_portfolio_weights, self.config.get('PORTFOLIO_ONLINE_TIMESTEPS', 100_000))
                else:
                    for symbol in self.config['SYMBOLS']:
                        latest_data = self.data_ingestion.get_latest_data(symbol)
                        if len(latest_data) >= 500:
                            await asyncio.to_thread(self.trainer.update_model_weights, symbol, latest_data)
                logger.info("PPO nightly retrain completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"PPO nightly retrain task failed: {e}", exc_info=True)
                await asyncio.sleep(300)

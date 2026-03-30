# models/bot_initializer.py
"""
BotInitializer — Phase 3 extraction
Handles the entire startup sequence from run() so the main run() method stays clean (~30 lines)
All previous patches, comments, and logs preserved exactly.
"""
import logging
import numpy as np
import os
import json
import pandas as pd
from datetime import datetime
from dateutil import tz
from config import CONFIG # ← FIXED: was missing → NameError on CONFIG.get('USE_CAUSAL_RL')
# Define missing constants used in this file (copied from bot.py)
HISTORY_FILE = "live_signal_history.json"
LAST_ENTRY_FILE = "last_entry_times.json"
logger = logging.getLogger(__name__)

class BotInitializer:
    def __init__(self, bot):
        self.bot = bot # reference to TradingBot instance

    def perform_full_startup(self):
        """All startup logic that used to live in run()"""
        from gemini_tuner import load_dynamic_config
        load_dynamic_config()
        # B-17 FIX: Load persistent live_signal_history EARLY (before causal warmup needs it)
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    loaded_history = json.load(f)
                # FIX #13: Parse timestamps per-entry so one bad entry doesn't kill all history
                bad_entries = 0
                for sym, entries in loaded_history.items():
                    valid_entries = []
                    for entry in entries:
                        try:
                            if 'timestamp' in entry and isinstance(entry['timestamp'], str):
                                entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                            valid_entries.append(entry)
                        except (ValueError, TypeError) as parse_err:
                            bad_entries += 1
                            logger.warning(f"[HISTORY PERSISTENCE] Skipping bad entry for {sym}: {parse_err}")
                    loaded_history[sym] = valid_entries
                if bad_entries > 0:
                    logger.warning(f"[HISTORY PERSISTENCE] Skipped {bad_entries} entries with unparseable timestamps")
                # Update in-place to preserve shared reference with signal_gen (CRIT-1 fix)
                self.bot.live_signal_history.clear()
                self.bot.live_signal_history.update(loaded_history)
                # FIX #11: Trim loaded entries to _signal_history_max to prevent unbounded memory
                max_entries = getattr(self.bot, '_signal_history_max', 500)
                for sym in self.bot.live_signal_history:
                    if len(self.bot.live_signal_history[sym]) > max_entries:
                        self.bot.live_signal_history[sym] = self.bot.live_signal_history[sym][-max_entries:]
                logger.info(f"[HISTORY PERSISTENCE] Loaded {sum(len(v) for v in self.bot.live_signal_history.values())} historical entries from disk (trimmed to max {max_entries}/symbol)")
            except Exception as e:
                logger.warning(f"[HISTORY PERSISTENCE] Failed to load history file: {e} — starting fresh")
        else:
            logger.info("[HISTORY PERSISTENCE] No previous history file found — starting fresh")
        # HIGH-33 FIX: Initialize data BEFORE regime pre-warm so HMMs have real data
        logger.info("Initializing data feeds before regime detection...")
        self.bot.data_ingestion.initialize_data()
        # Pre-warm cache before training
        logger.info("Pre-warming regime cache before training...")
        self.bot._get_all_regimes()
        logger.info(f"Cache pre-warm complete — {len(self.bot.regime_cache)} symbols ready")
        # FIX #10: NOTE — PPO training runs before causal warmup intentionally.
        # Models must exist before causal wrappers can be initialized and warmed up.
        # Causal penalties will be zero during this initial training, which is acceptable.
        logger.info("Starting parallel training for {} symbols".format(len(self.bot.config['SYMBOLS'])))
        # FIX #21: Training failure should not kill startup — fall back to cached models
        try:
            self.bot.trainer.train_symbols_parallel(self.bot.config['SYMBOLS'], full_ppo=True)
        except Exception as e:
            logger.error(f"Training failed during startup: {e} — falling back to cached models", exc_info=True)
            # Attempt to load any existing cached models for each symbol
            from models.ppo_utils import load_ppo_model
            for sym in self.bot.config['SYMBOLS']:
                try:
                    load_ppo_model(self.bot.trainer, sym)
                    if self.bot.trainer.ppo_models.get(sym):
                        logger.info(f"[FALLBACK] Loaded cached model for {sym}")
                    else:
                        logger.warning(f"[FALLBACK] No cached model available for {sym}")
                except Exception as load_err:
                    logger.warning(f"[FALLBACK] Failed to load cached model for {sym}: {load_err}")
        if self.bot.portfolio_ppo:
            from models.ppo_utils import load_ppo_model
            load_ppo_model(self.bot.trainer, "portfolio")
            if self.bot.trainer.portfolio_ppo_model is None:
                logger.warning("Portfolio PPO enabled but no saved model found — falling back")
                self.bot.portfolio_ppo = False
            # === FIX #3: Prevent VecNormalize stat drift in live inference ===
            if hasattr(self.bot.trainer, 'portfolio_vec_norm') and self.bot.trainer.portfolio_vec_norm is not None:
                self.bot.trainer.portfolio_vec_norm.training = False
                logger.info("Portfolio VecNormalize locked to inference mode (training=False)")
            # === SMART CAUSAL REBUILD LOGIC ===
            # FIX #65: Wrap rebuild_causal in try/except so exception doesn't kill startup
            try:
                if CONFIG.get('USE_CAUSAL_RL', False) and self.bot.trainer.portfolio_ppo_model is not None:
                    portfolio_model_path = os.path.join("ppo_checkpoints", "portfolio", "ppo_model.zip")
                    if not os.path.exists(portfolio_model_path):
                        logger.info("No portfolio model file found — forcing full causal rebuild (cache will be deleted)")
                        self.bot.signal_gen.refresh_causal_wrappers()
                    else:
                        mtime = datetime.fromtimestamp(os.path.getmtime(portfolio_model_path))
                        today = datetime.now().date()
                        if mtime.date() == today:
                            logger.info("Portfolio model is from today — forcing full causal rebuild (cache will be deleted)")
                            self.bot.signal_gen.refresh_causal_wrappers()
                        else:
                            logger.info("Portfolio model is from previous day — using cached causal graph (safe rebuild, ReplayBuffer preserved)")
                            self.bot.signal_gen.rebuild_causal_wrappers_without_deleting_cache()
                    logger.info("Causal wrappers rebuilt on startup with smart cache logic")
            except Exception as e:
                logger.error(f"[CAUSAL STARTUP] Causal wrapper rebuild failed: {e} — continuing without causal penalties", exc_info=True)
            # Bug #22: Load persisted portfolio ReplayBuffer
            if hasattr(self.bot.signal_gen, 'portfolio_causal_manager') and self.bot.signal_gen.portfolio_causal_manager is not None:
                self.bot.signal_gen.portfolio_causal_manager.load_buffer()
            else:
                logger.warning("[CAUSAL STARTUP] Portfolio causal manager not available — buffer load skipped")
            # ISSUE #4 PATCH: Warm up ALL causal ReplayBuffers from live_signal_history
            # History is now guaranteed loaded (moved above)
            # FIX #30: Skip warmup if buffer was already loaded from disk (prevents duplicates)
            if self.bot.live_signal_history:
                logger.info("[CAUSAL WARMUP] Replaying last closed trades into ReplayBuffers...")
                for sym, manager in self.bot.signal_gen.causal_manager.items():
                    if len(manager.replay_buffer.buffer) == 0:
                        manager.warmup_from_history(self.bot.live_signal_history.get(sym, []))
                    else:
                        logger.debug(f"[CAUSAL WARMUP] {sym} buffer already has {len(manager.replay_buffer.buffer)} entries — skipping warmup")
                if hasattr(self.bot.signal_gen, 'portfolio_causal_manager') and self.bot.signal_gen.portfolio_causal_manager:
                    pcm = self.bot.signal_gen.portfolio_causal_manager
                    if len(pcm.replay_buffer.buffer) == 0:
                        pcm.warmup_from_history(
                            [e for hist in self.bot.live_signal_history.values() for e in hist]
                        )
                    else:
                        logger.debug(f"[CAUSAL WARMUP] portfolio buffer already has {len(pcm.replay_buffer.buffer)} entries — skipping warmup")
                logger.info("[CAUSAL WARMUP] Replay complete — causal penalties now active from startup")
            else:
                logger.info("[CAUSAL WARMUP] No live_signal_history found — ReplayBuffers remain empty")
            if hasattr(self.bot.signal_gen, 'portfolio_causal_manager') and self.bot.signal_gen.portfolio_causal_manager is not None:
                buffer_size = len(self.bot.signal_gen.portfolio_causal_manager.replay_buffer.buffer)
                logger.info(f"[BUFFER RESTORE ON STARTUP] Loaded {buffer_size} samples")
            else:
                logger.info("[BUFFER RESTORE ON STARTUP] No portfolio buffer to log")
        # === CAUSAL RL: Also handle non-portfolio mode ===
        if not self.bot.portfolio_ppo and CONFIG.get('USE_CAUSAL_RL', False):
            logger.info("[CAUSAL STARTUP] Portfolio PPO disabled but Causal RL enabled — warming up per-symbol causal buffers")
            per_symbol_managers = getattr(self.bot.signal_gen, 'causal_manager', {})
            if not per_symbol_managers:
                logger.warning("[CAUSAL STARTUP] Per-symbol causal managers not yet populated — "
                               "causal warmup skipped. Managers are created during train_symbol(); "
                               "if training was skipped or failed, causal penalties will be inactive.")
            elif self.bot.live_signal_history:
                for sym, manager in per_symbol_managers.items():
                    # FIX #38: Skip warmup if buffer already has entries (same guard as portfolio section)
                    if len(manager.replay_buffer.buffer) == 0:
                        manager.warmup_from_history(self.bot.live_signal_history.get(sym, []))
                    else:
                        logger.debug(f"[CAUSAL WARMUP] {sym} buffer already has {len(manager.replay_buffer.buffer)} entries — skipping warmup")
                logger.info("[CAUSAL WARMUP] Per-symbol replay complete")
        # === CAUSAL RL STATUS ===
        if CONFIG.get('USE_CAUSAL_RL', False):
            logger.info("CAUSAL RL ENABLED — full stacked feature matrix built in SignalGenerator at startup + daily 3:30 AM refresh")
        else:
            logger.warning("CAUSAL RL DISABLED (USE_CAUSAL_RL=False in config)")
        if self.bot.config.get('RUN_BACKTEST_ON_STARTUP', False):
            logger.info("Running initial backtest validation")
            self.bot.backtester.run_backtest()
        else:
            logger.info("Skipping backtest — going straight to live paper trading")
        # HIGH-33: initialize_data() already called above before regime pre-warm
        # NOTE #50: get_equity() and get_positions_dict() are synchronous Alpaca API calls,
        # but this is safe because perform_full_startup() is called via asyncio.to_thread()
        # from bot.py's async run() method, so we're already in a worker thread, not the event loop.
        starting_equity = self.bot.broker.get_equity()
        logger.info(f"Starting equity: ${starting_equity:.2f}")
        today = datetime.now(tz=tz.gettz('UTC')).date()
        self.bot.daily_equity[today] = starting_equity
        self.bot.equity_history[today] = starting_equity
        # B-16 FIX: One-time startup migration
        logger.info("[MIGRATION B-16] Backfilling 'size' for existing live_signal_history entries...")
        positions = self.bot.broker.get_positions_dict()  # Safe: runs in worker thread (see NOTE #50 above)
        for symbol in self.bot.config['SYMBOLS']:
            history = self.bot.live_signal_history.get(symbol, [])
            for entry in history:
                if 'size' not in entry or entry['size'] is None:
                    # FIX #12: Only backfill size for open entries — closed entries would get made-up values
                    if entry.get('realized_return') is not None:
                        continue  # Skip closed entries — guessed size would be misleading
                    current_size = positions.get(symbol, 0)
                    if current_size != 0 and entry.get('direction', 0) == np.sign(current_size):
                        entry['size'] = abs(current_size)
                    else:
                        entry_price = entry.get('price', 1.0)
                        approx_size = max(1, int((starting_equity * 0.05) / entry_price))
                        entry['size'] = approx_size
        logger.info(f"[MIGRATION B-16] Size backfill complete — {len([e for h in self.bot.live_signal_history.values() for e in h if 'size' in e])} entries now have size")
        # ====================== BUG #5 FIX START ======================
        logger.info("[B-20 MIN-HOLD RESTORE] Rebuilding last_entry_times from live_signal_history...")
        for symbol in self.bot.config.get('SYMBOLS', []):
            history = self.bot.live_signal_history.get(symbol, [])
            if history:
                open_entries = [e for e in history if e.get('realized_return') is None and e.get('direction', 0) != 0]
                if open_entries:
                    latest_open = max(open_entries, key=lambda e: e['timestamp'])
                    ts = latest_open['timestamp']
                    # FIX #14: Ensure timestamp is timezone-aware (broker expects UTC)
                    if isinstance(ts, datetime) and ts.tzinfo is None:
                        ts = ts.replace(tzinfo=tz.gettz('UTC'))
                    self.bot.broker.last_entry_times[symbol] = ts
        logger.info(f"[B-20 MIN-HOLD RESTORE] Restored last_entry_times for {len(self.bot.broker.last_entry_times)} symbols")
        # ====================== BUG #5 FIX END ======================
        # ====================== BUG #7 FIX START ======================
        # HIGH-32 FIX: Only load from file for symbols NOT already restored from live_signal_history
        # (Bug #5 restores accurate timestamps from actual trade entries; file may have stale data)
        if os.path.exists(LAST_ENTRY_FILE):
            try:
                with open(LAST_ENTRY_FILE, 'r') as f:
                    loaded_times = json.load(f)
                restored_count = 0
                for sym, ts_str in loaded_times.items():
                    if isinstance(ts_str, str) and sym not in self.bot.broker.last_entry_times:
                        # M42 FIX: Ensure tz-aware (file may have naive timestamps)
                        parsed_dt = datetime.fromisoformat(ts_str)
                        if parsed_dt.tzinfo is None:
                            from dateutil import tz as _tz
                            parsed_dt = parsed_dt.replace(tzinfo=_tz.gettz('UTC'))
                        self.bot.broker.last_entry_times[sym] = parsed_dt
                        restored_count += 1
                logger.info(f"[B-21 FAST LOAD] Restored {restored_count} last_entry_times from file (skipped symbols already restored from history)")
            except Exception as e:
                logger.warning(f"[B-21] Failed to load LAST_ENTRY_FILE: {e}")
        # ====================== BUG #7 FIX END ======================
        logger.info("✅ Full bot initialization completed successfully")

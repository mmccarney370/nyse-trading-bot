# models/causal_rl_manager.py
"""
Centralized Causal RL Manager
Extracted from bot.py in Phase 2 (Bugs #19–#23)
Handles:
- Portfolio ReplayBuffer persistence (save/load) — Bug #22
- Rotation reset (portfolio buffer clear + per-symbol prune/init) — Bugs #20 + #23
- Daily reward sync after graph refresh — Bug #21
All previous patch comments preserved exactly.
March 2026 Update: Removed obsolete save/load_portfolio_buffer methods
→ These were duplicates; real persistence now handled in causal_signal_manager.py
"""
import logging
import json
import os
import pickle
import tempfile
import shutil
import numpy as np
from datetime import datetime
from typing import Dict, Any
from config import CONFIG
from .causal_signal_manager import CausalSignalManager  # CRIT-07 FIX: required for rotation init
logger = logging.getLogger(__name__)
class CausalRLManager:
    def __init__(self, signal_gen, config, signal_history_lock=None):
        self.signal_gen = signal_gen
        self.config = config
        self.replay_buffer_file = "portfolio_replay_buffer.json"
        self._signal_history_lock = signal_history_lock  # FIX #39: shared lock for live_signal_history
    def reset_on_rotation(self, new_symbols):
        """Bugs #20 + #23: Full reset on universe rotation.
        - Clear portfolio ReplayBuffer (Bug #20)
        - Prune removed per-symbol wrappers
        - Initialize new per-symbol wrappers for added symbols (Bug #23)
        """
        # FIX #38: Save portfolio buffer before clearing on rotation
        if hasattr(self.signal_gen, 'portfolio_causal_manager') and self.signal_gen.portfolio_causal_manager is not None:
            if hasattr(self.signal_gen.portfolio_causal_manager, 'save_buffer'):
                try:
                    self.signal_gen.portfolio_causal_manager.save_buffer()
                    logger.info("[B-20] Saved portfolio causal buffer before rotation reset")
                except Exception as e:
                    logger.warning(f"[B-20] Failed to save buffer before reset: {e}")
            # BUG #20 PATCH: Force ReplayBuffer reset on rotation
            if hasattr(self.signal_gen.portfolio_causal_manager, 'replay_buffer'):
                self.signal_gen.portfolio_causal_manager.replay_buffer.buffer.clear()
            logger.info("Portfolio causal ReplayBuffer reset for new universe (stale transitions cleared)")
        # BUG #23 PATCH: Prune per-symbol causal wrappers for removed symbols and init new ones for added symbols
        causal_mgr = getattr(self.signal_gen, 'causal_manager', {})
        old_symbols = set(causal_mgr.keys())
        removed = old_symbols - set(new_symbols)
        # FIX #37: Iterate over a copy of keys to avoid dict mutation during iteration
        for sym in list(removed):
            if sym in causal_mgr:
                del causal_mgr[sym]
                logger.debug(f"[B-23] Removed stale per-symbol causal wrapper for {sym}")
        added = set(new_symbols) - old_symbols
        trainer = getattr(self.signal_gen, 'trainer', None)
        for sym in added:
            if sym not in getattr(self.signal_gen, 'causal_manager', {}):
                # FIX #37: Only create causal manager if a trained PPO model exists for this symbol.
                # base_model=None would crash on predict().
                ppo_model = trainer.ppo_models.get(sym) if trainer else None
                if ppo_model is None:
                    logger.warning(f"[B-23] Skipping causal wrapper for {sym} — no trained PPO model")
                    continue
                data_ing = getattr(self.signal_gen, 'data_ingestion', None)
                self.signal_gen.causal_manager[sym] = CausalSignalManager(base_model=ppo_model, symbol=sym, data_ingestion=data_ing)
                logger.info(f"[B-23] Initialized new per-symbol causal wrapper for added symbol {sym}")
    def sync_daily_rewards(self):
        """Bug #21: After daily graph refresh, force-push any pending realized rewards from live_signal_history."""
        if not CONFIG.get('USE_CAUSAL_RL', False):
            return
        try:
            # CRIT-07 FIX: Real implementation (was stub) — pushes only unpushed closes
            if not hasattr(self.signal_gen, 'live_signal_history'):
                logger.warning("No live_signal_history found on signal_gen for daily reward sync")
                return
            history = self.signal_gen.live_signal_history
            pushed_per_symbol = 0
            pushed_portfolio = 0
            pushed = 0
            # FIX #39: Acquire signal_history_lock to avoid race with main thread mutations
            # FIX #20: Use timeout to prevent indefinite blocking if main thread holds lock
            lock = self._signal_history_lock
            if lock:
                if not lock.acquire(timeout=10):
                    logger.warning("[CAUSAL SYNC] Could not acquire signal_history_lock within 10s — skipping sync")
                    return False
            try:
                for symbol, entries in history.items():
                    for entry in entries:
                        if entry.get('realized_return') is not None and not entry.get('reward_pushed', False):
                            reward = entry['realized_return']
                            obs_raw = entry.get('obs', [])
                            if not obs_raw or (isinstance(obs_raw, (list, np.ndarray)) and len(obs_raw) == 0):
                                continue  # skip empty obs — would crash np.vstack in sample()
                            obs = np.array(obs_raw, dtype=np.float32) if isinstance(obs_raw, list) else obs_raw
                            # HIGH-30 FIX: Reconstruct actual action from stored direction + confidence
                            # instead of using action=0 placeholder (was corrupting causal analysis)
                            action_val = entry.get('direction', 0) * entry.get('confidence', 0.5)
                            # FIX #21: Only push obs to the buffer that matches its dimension.
                            # Per-symbol obs have different dimensions than portfolio features —
                            # cross-pushing corrupts the causal graph build.
                            obs_dim = obs.shape[-1] if hasattr(obs, 'shape') else len(obs)
                            # FIX #42: Track per-symbol and portfolio push separately
                            per_symbol_pushed = False
                            # Push to per-symbol causal manager if exists AND dimension matches
                            if symbol in getattr(self.signal_gen, 'causal_manager', {}):
                                wrapper = self.signal_gen.causal_manager[symbol]
                                if hasattr(wrapper, 'add_transition'):
                                    expected_dim = None
                                    if hasattr(wrapper, 'base_model') and wrapper.base_model is not None and hasattr(wrapper.base_model, 'observation_space'):
                                        expected_dim = wrapper.base_model.observation_space.shape[-1]
                                    if expected_dim is None or obs_dim == expected_dim:
                                        wrapper.add_transition(obs, action_val, reward)
                                        per_symbol_pushed = True
                                        pushed_per_symbol += 1
                                    else:
                                        logger.debug(f"[CAUSAL SYNC] {symbol}: obs dim {obs_dim} != per-symbol expected {expected_dim} — skipped")
                            # Push to portfolio causal wrapper if exists AND dimension matches
                            portfolio_pushed = False
                            if hasattr(self.signal_gen, 'portfolio_causal_manager') and self.signal_gen.portfolio_causal_manager is not None:
                                port_wrapper = self.signal_gen.portfolio_causal_manager
                                if hasattr(port_wrapper, 'add_transition'):
                                    expected_dim = None
                                    if hasattr(port_wrapper, 'base_model') and port_wrapper.base_model is not None and hasattr(port_wrapper.base_model, 'observation_space'):
                                        expected_dim = port_wrapper.base_model.observation_space.shape[-1]
                                    if expected_dim is None or obs_dim == expected_dim:
                                        port_wrapper.add_transition(obs, action_val, reward)
                                        portfolio_pushed = True
                                        pushed_portfolio += 1
                                    else:
                                        logger.debug(f"[CAUSAL SYNC] {symbol}: obs dim {obs_dim} != portfolio expected {expected_dim} — skipped")
                            # FIX #42: Only mark reward_pushed after BOTH pushes succeed (or were skipped)
                            if per_symbol_pushed or portfolio_pushed:
                                entry['reward_pushed'] = True
            finally:
                if lock:
                    lock.release()
            # FIX #63: Report per-symbol and portfolio push counts separately
            pushed = pushed_per_symbol + pushed_portfolio
            logger.info(f"Daily causal reward sync completed — pushed {pushed} realized rewards "
                        f"(per_symbol={pushed_per_symbol}, portfolio={pushed_portfolio})")
            return True
        except Exception as e:
            logger.error(f"Daily causal reward sync failed: {e}", exc_info=True)
            return False
    def save_buffer(self):
        """B-33 / BUG-09: Emergency save support for portfolio causal buffer.
        Called from TradingBot._emergency_save_all() — CRIT-07 FIX"""
        try:
            if hasattr(self.signal_gen, 'portfolio_causal_manager') and self.signal_gen.portfolio_causal_manager is not None:
                if hasattr(self.signal_gen.portfolio_causal_manager, 'save_buffer'):
                    self.signal_gen.portfolio_causal_manager.save_buffer()
                    logger.debug("[B-33] Portfolio causal ReplayBuffer saved via CausalRLManager")
                elif hasattr(self.signal_gen.portfolio_causal_manager, 'replay_buffer'):
                    # FIX #39/#43: Fallback save uses pickle with filename matching CausalSignalManager.load_buffer()
                    path = "replay_buffer_portfolio.pkl"
                    rb = self.signal_gen.portfolio_causal_manager.replay_buffer
                    buf_data = list(getattr(rb, 'buffer', []))
                    dir_name = os.path.dirname(path) or '.'
                    with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=dir_name) as tmp:
                        pickle.dump(buf_data, tmp)
                        tmp.flush()
                        os.fsync(tmp.fileno())
                    shutil.move(tmp.name, path)
                    logger.debug("[B-33] Portfolio causal buffer saved via fallback pickle")
            else:
                logger.debug("[B-33] No portfolio_causal_wrapper found — skipping buffer save")
        except Exception as e:
            logger.error(f"[B-33] Failed to save portfolio causal buffer: {e}")

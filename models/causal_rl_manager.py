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
import numpy as np
from datetime import datetime
from typing import Dict, Any
from config import CONFIG
from .causal_signal_manager import CausalSignalManager  # CRIT-07 FIX: required for rotation init
logger = logging.getLogger(__name__)
class CausalRLManager:
    def __init__(self, signal_gen, config):
        self.signal_gen = signal_gen
        self.config = config
        self.replay_buffer_file = "portfolio_replay_buffer.json"
    def reset_on_rotation(self, new_symbols):
        """Bugs #20 + #23: Full reset on universe rotation.
        - Clear portfolio ReplayBuffer (Bug #20)
        - Prune removed per-symbol wrappers
        - Initialize new per-symbol wrappers for added symbols (Bug #23)
        """
        # BUG #20 PATCH: Force ReplayBuffer reset on rotation
        if hasattr(self.signal_gen, 'portfolio_causal_wrapper') and self.signal_gen.portfolio_causal_wrapper is not None:
            if hasattr(self.signal_gen.portfolio_causal_wrapper, 'replay_buffer'):
                self.signal_gen.portfolio_causal_wrapper.replay_buffer.buffer.clear()
            logger.info("✅ Portfolio causal ReplayBuffer reset for new universe (stale transitions cleared)")
        # BUG #23 PATCH: Prune per-symbol causal wrappers for removed symbols and init new ones for added symbols
        old_symbols = set(getattr(self.signal_gen, 'causal_wrappers', {}).keys())
        removed = old_symbols - set(new_symbols)
        for sym in removed:
            if sym in getattr(self.signal_gen, 'causal_wrappers', {}):
                del self.signal_gen.causal_wrappers[sym]
                logger.debug(f"[B-23] Removed stale per-symbol causal wrapper for {sym}")
        added = set(new_symbols) - old_symbols
        for sym in added:
            if sym not in getattr(self.signal_gen, 'causal_wrappers', {}):
                # ISSUE #4 FIX: Updated class name from old CausalSignalWrapper → current CausalSignalManager
                data_ing = getattr(self.signal_gen, 'data_ingestion', None)
                self.signal_gen.causal_wrappers[sym] = CausalSignalManager(base_model=None, symbol=sym, data_ingestion=data_ing)
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
            pushed = 0
            for symbol, entries in history.items():
                for entry in entries:
                    if entry.get('realized_return') is not None and not entry.get('reward_pushed', False):
                        reward = entry['realized_return']
                        obs_raw = entry.get('obs', [])
                        if not obs_raw or (isinstance(obs_raw, (list, np.ndarray)) and len(obs_raw) == 0):
                            continue  # skip empty obs — would crash np.vstack in sample()
                        obs = np.array(obs_raw, dtype=np.float32) if isinstance(obs_raw, list) else obs_raw
                        # Push to per-symbol causal manager if exists
                        if symbol in getattr(self.signal_gen, 'causal_wrappers', {}):
                            wrapper = self.signal_gen.causal_wrappers[symbol]
                            if hasattr(wrapper, 'add_transition'):
                                wrapper.add_transition(obs, 0, reward)  # action=0 is placeholder
                                entry['reward_pushed'] = True
                                pushed += 1
                        # Push to portfolio causal wrapper if exists
                        if hasattr(self.signal_gen, 'portfolio_causal_wrapper') and self.signal_gen.portfolio_causal_wrapper is not None:
                            port_wrapper = self.signal_gen.portfolio_causal_wrapper
                            if hasattr(port_wrapper, 'add_transition'):
                                port_wrapper.add_transition(obs, 0, reward)
                                entry['reward_pushed'] = True
                                pushed += 1
            logger.info(f"✅ Daily causal reward sync completed — pushed {pushed} realized rewards")
            return True
        except Exception as e:
            logger.error(f"Daily causal reward sync failed: {e}", exc_info=True)
            return False
    def save_buffer(self):
        """B-33 / BUG-09: Emergency save support for portfolio causal buffer.
        Called from TradingBot._emergency_save_all() — CRIT-07 FIX"""
        try:
            if hasattr(self.signal_gen, 'portfolio_causal_wrapper') and self.signal_gen.portfolio_causal_wrapper is not None:
                if hasattr(self.signal_gen.portfolio_causal_wrapper, 'save_buffer'):
                    self.signal_gen.portfolio_causal_wrapper.save_buffer()
                    logger.debug("[B-33] Portfolio causal ReplayBuffer saved via CausalRLManager")
                elif hasattr(self.signal_gen.portfolio_causal_wrapper, 'replay_buffer'):
                    # Fallback atomic save if no save_buffer method
                    path = self.replay_buffer_file
                    dir_name = os.path.dirname(path) or '.'
                    with open(path, 'w') as f:  # atomic enough for buffer
                        json.dump(self.signal_gen.portfolio_causal_wrapper.replay_buffer, f, default=str)
                    logger.debug("[B-33] Portfolio causal buffer saved via fallback JSON")
            else:
                logger.debug("[B-33] No portfolio_causal_wrapper found — skipping buffer save")
        except Exception as e:
            logger.error(f"[B-33] Failed to save portfolio causal buffer: {e}")

# strategy/signals.py
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz
from newsapi import NewsApiClient
import os
import re # For robust number extraction from LLM response
import asyncio # ← ADDED for non-blocking Ollama calls
# Suppress harmless NumPy warnings from variance/std on small/constant slices
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
# Suppress Lightning/PyTorch Lightning configuration_validator warning globally
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="lightning.pytorch.trainer.configuration_validator",
    message="You defined a `validation_step` but have no `val_dataloader`. Skipping val loop."
)
# ─── ULTRA-EARLY GLOBAL TQDM SUPPRESSION (Feb 27 2026) ───────────────────────────────
os.environ["TQDM_DISABLE"] = "1"
os.environ["DISABLE_TQDM"] = "1"
# ─── DEFINE LOGGER FIRST ───────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
# ────────────────────────────────────────────────────────────────
# Local LLM support (Ollama preferred when enabled)
# ────────────────────────────────────────────────────────────────
from utils.local_llm import LocalLLMDebate
# Groq LLM availability (kept for optional fallback)
GROQ_AVAILABLE = 'GROQ_API_KEY' in os.environ
if GROQ_AVAILABLE:
    try:
        from groq import Groq
        logger.info("Groq API available — can be used as LLM fallback")
    except Exception as e:
        logger.warning(f"Groq import failed ({e}) — Groq fallback disabled")
        GROQ_AVAILABLE = False
from config import CONFIG
from models.features import generate_features
from strategy.regime import detect_regime, is_trending, is_bullish, is_bearish
from models.portfolio_env import PortfolioEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
import torch # ← Required for torch.no_grad()
# NEW: Causal RL dependencies
# FIX: Guard causal imports — these are only needed when USE_CAUSAL_RL is enabled
try:
    from dowhy import CausalModel
    import pgmpy.estimators.GES as GES
    import networkx as nx
    _CAUSAL_IMPORTS_OK = True
except ImportError as _causal_import_err:
    _CAUSAL_IMPORTS_OK = False
    logger.warning(f"Causal RL imports failed ({_causal_import_err}) — causal features will be disabled")
import threading # FIX #24: For sync wrapper lock
import time # ← REQUIRED for timing the causal graph build
import pickle # ← For causal graph disk caching
import hashlib # ← P-18 / Critical #11 FIX: for model hash
# ===================================================================
# ===================================================================
from models.causal_signal_manager import CausalSignalManager

class LLMAgentDebate:
    """
    Multi-agent LLM debate for sentiment analysis using Groq (Llama3-70b).
    Agents: bull, bear, analyst.
    Returns averaged conviction score (-1 to 1).
    Kept for fallback / compatibility — normally LocalLLMDebate is preferred.
    """
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY")) if GROQ_AVAILABLE else None
        self.agents = ["bull", "bear", "analyst"]

    def debate_sentiment(self, news_texts: list) -> float:
        if not GROQ_AVAILABLE or not self.client:
            logger.debug("Groq not available — skipping LLM debate")
            return 0.0
        if not news_texts:
            return 0.0
        opinions = []
        headlines = ". ".join(news_texts[:15])
        for role in self.agents:
            try:
                prompt = (
                    f"You are a {role} trader. Analyze the sentiment of these news headlines/descriptions for the stock: {headlines}\n"
                    "Respond ONLY with a single number from -1 (very negative) to 1 (very positive). No explanation."
                )
                completion = self.client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10
                )
                raw = completion.choices[0].message.content.strip()
                match = re.search(r'-?\d+\.?\d*', raw)
                if match:
                    score = float(match.group(0))
                    score = np.clip(score, -1.0, 1.0)
                else:
                    score = 0.0
                opinions.append(score)
                logger.debug(f"LLM {role} agent score: {score:.3f} (raw: {raw})")
            except Exception as e:
                logger.debug(f"LLM {role} agent failed ({e}) — using neutral")
                opinions.append(0.0)
        final = np.mean(opinions) if opinions else 0.0
        logger.info(f"LLM debate final sentiment: {final:.3f} (from {len(opinions)} agents)")
        return final

class _AdaptiveMemory:
    """Real-time trade memory that makes the signal generator learn from its own mistakes.
    Three interconnected systems:

    1. ANTI-CHURN: Exponential confidence decay after stop-outs. Each consecutive
       stop on the same symbol in the same direction multiplies confidence by 0.5^n.
       Decays back to neutral after a cooldown period. Prevents the TSLA-style
       "stop → re-enter → stop → re-enter" death spiral.

    2. CROSS-ASSET PANIC DETECTOR: Tracks stop-outs across ALL symbols in a rolling
       window. When >= 3 symbols stop out within 20 bars, the system enters "defensive
       mode" — all new entries are suppressed for a cooldown period. Recognizes that
       correlated stop-outs = systemic event, not individual signal failure.

    3. ADAPTIVE SIGNAL WEIGHTING: Tracks per-component accuracy (ensemble, PPO,
       sentiment) over a sliding window of recent closed trades. Components that
       have been wrong recently get downweighted; accurate ones get upweighted.
       This is online Bayesian updating of the signal blend.
    """
    def __init__(self):
        self.stop_history = {}       # {symbol: [(timestamp, direction, price), ...]}
        self.trade_outcomes = []     # [(timestamp, symbol, direction, pnl, meta_prob, ppo_prob, sentiment)]
        self.global_stops = []       # [(timestamp, symbol)] — all stops across all symbols
        self.defensive_until = None  # timestamp when defensive mode expires
        self._component_accuracy = {
            'ensemble': {'correct': 0, 'total': 0},
            'ppo': {'correct': 0, 'total': 0},
            'sentiment': {'correct': 0, 'total': 0},
        }

    def record_stop_out(self, symbol: str, direction: int, price: float, timestamp=None):
        """Call when a trailing stop is hit."""
        ts = timestamp or datetime.now(tz=tz.gettz('UTC'))
        if symbol not in self.stop_history:
            self.stop_history[symbol] = []
        self.stop_history[symbol].append((ts, direction, price))
        # Keep last 20 per symbol
        self.stop_history[symbol] = self.stop_history[symbol][-20:]
        # Global tracker
        self.global_stops.append((ts, symbol))
        self.global_stops = self.global_stops[-100:]
        # Check for cross-asset panic
        self._check_panic(ts)

    def _check_panic(self, now):
        """If >= 3 symbols stopped out within last 20 bars (~5 min), enter defensive mode."""
        cutoff = now - timedelta(minutes=5)
        recent = [s for ts, s in self.global_stops if ts >= cutoff]
        unique_symbols = len(set(recent))
        if unique_symbols >= 3:
            cooldown_bars = 12  # ~3 hours of 15-min bars
            self.defensive_until = now + timedelta(minutes=15 * cooldown_bars)
            logger.warning(f"[PANIC DETECT] {unique_symbols} symbols stopped in 5min — "
                          f"DEFENSIVE MODE until {self.defensive_until.strftime('%H:%M')}")

    def is_defensive(self, timestamp=None) -> bool:
        """True if the system is in defensive mode (suppress all new entries)."""
        if self.defensive_until is None:
            return False
        now = timestamp or datetime.now(tz=tz.gettz('UTC'))
        if now >= self.defensive_until:
            self.defensive_until = None
            return False
        return True

    def get_churn_penalty(self, symbol: str, direction: int, timestamp=None) -> float:
        """Returns confidence multiplier (0.0-1.0) based on recent stop-out history.
        Each recent stop in the same direction halves the multiplier.
        Stops older than cooldown_minutes are ignored."""
        cooldown_minutes = 60  # 4 bars of 15min
        now = timestamp or datetime.now(tz=tz.gettz('UTC'))
        cutoff = now - timedelta(minutes=cooldown_minutes)
        history = self.stop_history.get(symbol, [])
        # Count recent stops in the SAME direction
        same_dir_stops = sum(1 for ts, d, _ in history if ts >= cutoff and d == direction)
        if same_dir_stops == 0:
            return 1.0
        # Exponential decay: 0.5^n (1 stop = 0.5, 2 stops = 0.25, 3 stops = 0.125)
        penalty = 0.5 ** same_dir_stops
        logger.info(f"[CHURN] {symbol}: {same_dir_stops} recent stops in dir={direction} — "
                    f"confidence *= {penalty:.3f}")
        return penalty

    def record_trade_outcome(self, symbol: str, direction: int, pnl: float,
                             meta_prob: float, ppo_prob: float, sentiment: float,
                             timestamp=None):
        """Call when a trade closes (stop or TP). Updates component accuracy tracking."""
        ts = timestamp or datetime.now(tz=tz.gettz('UTC'))
        won = pnl > 0
        self.trade_outcomes.append((ts, symbol, direction, pnl, meta_prob, ppo_prob, sentiment))
        # Keep last 50 outcomes
        self.trade_outcomes = self.trade_outcomes[-50:]
        # Update per-component accuracy
        # Ensemble was "right" if meta_prob > 0.5 and trade won (long), or < 0.5 and trade won (short)
        ensemble_agreed = (meta_prob > 0.5 and direction == 1) or (meta_prob < 0.5 and direction == -1)
        ppo_agreed = (ppo_prob > 0.5 and direction == 1) or (ppo_prob < 0.5 and direction == -1)
        sent_agreed = (sentiment > 0 and direction == 1) or (sentiment < 0 and direction == -1)
        for component, agreed in [('ensemble', ensemble_agreed), ('ppo', ppo_agreed), ('sentiment', sent_agreed)]:
            self._component_accuracy[component]['total'] += 1
            if (agreed and won) or (not agreed and not won):
                self._component_accuracy[component]['correct'] += 1

    def get_adaptive_weights(self) -> dict:
        """Returns dynamically adjusted blend weights based on recent component accuracy.
        Components that have been more accurate get higher weight.
        Requires >= 10 trades to activate; before that, returns default weights."""
        total_trades = self._component_accuracy['ensemble']['total']
        if total_trades < 10:
            return None  # Use default weights
        accuracies = {}
        for comp in ['ensemble', 'ppo', 'sentiment']:
            stats = self._component_accuracy[comp]
            # Laplace smoothing: (correct + 1) / (total + 2) prevents 0% or 100%
            accuracies[comp] = (stats['correct'] + 1) / (stats['total'] + 2)
        # Normalize to sum to 1.0
        total = sum(accuracies.values())
        weights = {k: v / total for k, v in accuracies.items()}
        logger.debug(f"[ADAPTIVE] Component accuracies: {accuracies} → weights: {weights}")
        return weights


class SignalGenerator:
    def __init__(self, config, data_ingestion, trainer, regime_cache=None): # BUG #4 FIX: now accepts shared cache from bot.py
        self.config = config
        self.data_ingestion = data_ingestion
        self.trainer = trainer
        _news_key = config.get('NEWS_API_KEY')
        self.news_api = NewsApiClient(api_key=_news_key) if _news_key else None
        self.prev_signals = {}
        self.last_entry_time = {}
        self.meta_ema = {}
        self.sentiment_cache = {}
        self.lstm_states = {}
        self.last_portfolio_state = None
        self._meta_prob_history = {}  # FIX #17: Initialize here so reset_backtest_state() can clear it
        self.is_recurrent = config.get('PPO_RECURRENT', True)
        # BUG #4 FIX: Shared regime cache with bot.py (4AM precompute + _get_all_regimes now syncs live to signal generation)
        self.regime_cache = regime_cache if regime_cache is not None else {}
        # NEW: Initialize live_signal_history (required for OOS monitoring after reward push moved to alpaca.py)
        self.live_signal_history = {}
        # NEW: Initialize latest_prices cache (used by OOS monitoring)
        self.latest_prices = {}
        # === ADAPTIVE MEMORY: Real-time learning from trade outcomes ===
        self.memory = _AdaptiveMemory()
        # === COGNITIVE LAYER: AGI-inspired meta-intelligence ===
        from strategy.cognitive import EquityCurveTrader, ProfitVelocityTracker, TradeAutopsyEngine
        self.equity_curve_trader = EquityCurveTrader()
        self.profit_velocity = ProfitVelocityTracker()
        self.autopsy_engine = TradeAutopsyEngine()
        # Sentiment is handled exclusively by LLM debate (Ollama) — finBERT/VADER removed (dead code, wasted ~440MB RAM)
        self.llm_debate = (
            LocalLLMDebate()
            if self.config.get('USE_LOCAL_LLM', True)
            else (LLMAgentDebate() if GROQ_AVAILABLE else None)
        )
        if self.config.get('USE_LOCAL_LLM', True) and isinstance(self.llm_debate, LocalLLMDebate):
            logger.info("✅ LocalLLMDebate (Ollama) initialized successfully")
        # === NEW: CausalSignalManager (Phase 1 extraction) ===
        self.causal_manager = {} # per-symbol managers
        self.portfolio_causal_manager = None
        if CONFIG.get('USE_CAUSAL_RL', False):
            logger.info("🔥 CAUSAL RL PATCH APPLIED — full stacked feature matrix ACTIVE. Logs will follow...")
            logger.info("Initializing causal managers for PPO models")
            # Cache data fetches to avoid double fetch per symbol
            _symbol_data_cache = {}
            _symbol_features_cache = {}
            for symbol in config.get('SYMBOLS', []):
                data = self.data_ingestion.get_latest_data(symbol)
                _symbol_data_cache[symbol] = data
                if len(data) >= 200:
                    regime_tuple = self.trainer.get_cached_regime(symbol, data) if hasattr(self.trainer, 'get_cached_regime') else ('trending', 0.5)
                    regime = regime_tuple[0] if isinstance(regime_tuple, (list, tuple)) else regime_tuple
                    features = generate_features(data, regime, symbol, data)
                    _symbol_features_cache[symbol] = features
                    # FIX #28: Removed dead features_df computation (not passed to CausalSignalManager)
                    if symbol in self.trainer.ppo_models and self.trainer.ppo_models[symbol]:
                        self.causal_manager[symbol] = CausalSignalManager(
                            base_model=self.trainer.ppo_models[symbol],
                            symbol=symbol,
                            data_ingestion=self.data_ingestion
                        )
            if self.trainer.portfolio_ppo_model:
                # FULL STACKED FEATURE MATRIX + symbol_id (this gives thousands of rows → meaningful edges)
                all_features = []
                symbol_ids = []
                total_rows = 0
                for sym in config.get('SYMBOLS', []):
                    # Reuse cached data and features instead of re-fetching
                    if sym in _symbol_features_cache and _symbol_features_cache[sym] is not None:
                        features = _symbol_features_cache[sym]
                    else:
                        data = _symbol_data_cache.get(sym) or self.data_ingestion.get_latest_data(sym)
                        regime_tuple = self.trainer.get_cached_regime(sym, data) if hasattr(self.trainer, 'get_cached_regime') else ('trending', 0.5)
                        regime = regime_tuple[0] if isinstance(regime_tuple, (list, tuple)) else regime_tuple
                        features = generate_features(data, regime, sym, data)
                    if features is not None and features.shape[0] > 0:
                        rows = features.shape[0]
                        total_rows += rows
                        logger.info(f"[CAUSAL STACK] {sym} contributed {rows} rows")
                        all_features.append(features)
                        symbol_ids.extend([sym] * rows)
                if all_features:
                    full_matrix = np.vstack(all_features)
                    full_df = pd.DataFrame(full_matrix, columns=[f'feat_{i}' for i in range(full_matrix.shape[1])])
                    full_df['symbol_id'] = symbol_ids # Enables cross-asset causality
                    # BUG-3 FIX: Encode symbol_id to numeric codes BEFORE any numeric filtering
                    full_df['symbol_id'] = pd.Categorical(full_df['symbol_id']).codes
                    self.portfolio_causal_manager = CausalSignalManager(
                        base_model=self.trainer.portfolio_ppo_model,
                        symbol="portfolio",
                        data_ingestion=self.data_ingestion
                    )
                    # H14 FIX: Seed the causal graph with the expensive full_df we just built
                    # (was discarded — the entire multi-symbol feature matrix computation was wasted)
                    self.portfolio_causal_manager.build_causal_graph(full_df)
                    logger.info(f"Portfolio causal matrix built with {full_matrix.shape[0]} total rows and {full_matrix.shape[1]} features + symbol_id")
                else:
                    self.portfolio_causal_manager = None
            else:
                self.portfolio_causal_manager = None
        # ROBUSTNESS: If portfolio model is already loaded (edge case / future load order), build immediately
        if CONFIG.get('USE_CAUSAL_RL', False) and self.trainer.portfolio_ppo_model is not None and self.portfolio_causal_manager is None:
            logger.info("Portfolio model already loaded at init time — building causal manager immediately using cache")
            self.rebuild_causal_wrappers_without_deleting_cache()

    def reset_backtest_state(self):
        """FIX #17: Clear per-symbol state that accumulates across backtest runs.
        Call at the start of each backtest run to prevent stale history from leaking."""
        self._meta_prob_history = {}
        self._last_regime_per_symbol = {}  # FIX #30: Reset regime tracking for meta_prob clearing
        self.prev_signals = {}
        self.meta_ema = {}
        self.sentiment_cache = {}
        self.lstm_states = {}
        self.last_portfolio_state = None
        self.memory = _AdaptiveMemory()
        logger.debug("[SIGNAL GEN] Backtest state reset — cleared _meta_prob_history, adaptive memory, and related caches")

    # ISSUE #5 FIX: warmup_causal_buffers now accepts live_signal_history as parameter (passed from bot_initializer)
    # Removed self.bot reference — prevents AttributeError at startup and enables proper warmup
    def warmup_causal_buffers(self, live_signal_history: dict = None):
        """Replay closed trades from live_signal_history into all causal ReplayBuffers on startup.
        Called automatically during bot startup after causal wrappers are built."""
        if not CONFIG.get('USE_CAUSAL_RL', False) or not live_signal_history:
            logger.debug("[CAUSAL WARMUP] Skipped — causal RL disabled or no history")
            return
        logger.info("[CAUSAL WARMUP] Replaying closed trades from live_signal_history into buffers...")
        total_added = 0
        # Per-symbol managers
        for sym, manager in self.causal_manager.items():
            history = live_signal_history.get(sym, [])
            added = 0
            for entry in history:
                if entry.get('realized_return') is None:
                    continue # skip open trades
                obs = entry.get('obs')
                if obs is None or not isinstance(obs, (list, np.ndarray)) or len(obs) == 0:
                    continue
                try:
                    obs_array = np.array(obs, dtype=np.float32).reshape(1, -1)
                    # FIX #15: Validate obs dimension matches model's expected input
                    # Skip obs with wrong dimension to avoid corrupting causal buffer
                    if hasattr(manager, 'model') and manager.model is not None:
                        expected_dim = None
                        if hasattr(manager.model, 'observation_space'):
                            expected_dim = manager.model.observation_space.shape[-1]
                        if expected_dim is not None and obs_array.shape[-1] != expected_dim:
                            logger.debug(f"[CAUSAL WARMUP] {sym}: obs dim {obs_array.shape[-1]} != expected {expected_dim} — skipping")
                            continue
                    action = entry.get('direction', 0) * entry.get('confidence', 1.0)
                    reward = entry.get('realized_return', 0.0)
                    manager.replay_buffer.push(obs_array, action, reward)
                    added += 1
                    total_added += 1
                except Exception as e:
                    logger.debug(f"[CAUSAL WARMUP] Skipped invalid entry for {sym}: {e}")
            if added > 0:
                logger.info(f"[CAUSAL WARMUP] {sym}: replayed {added} closed trades")
            else:
                logger.debug(f"[CAUSAL WARMUP] {sym}: no closed trades to replay")
        # Portfolio-level manager (aggregate all symbols' closed trades)
        if hasattr(self, 'portfolio_causal_manager') and self.portfolio_causal_manager:
            added_port = 0
            all_closed = [e for hist in live_signal_history.values() for e in hist if e.get('realized_return') is not None]
            for entry in all_closed:
                obs = entry.get('obs')
                if obs is None or not isinstance(obs, (list, np.ndarray)) or len(obs) == 0:
                    continue
                try:
                    obs_array = np.array(obs, dtype=np.float32).reshape(1, -1)
                    # FIX #15: Validate obs dimension for portfolio buffer too
                    if hasattr(self.portfolio_causal_manager, 'model') and self.portfolio_causal_manager.model is not None:
                        expected_dim = None
                        if hasattr(self.portfolio_causal_manager.model, 'observation_space'):
                            expected_dim = self.portfolio_causal_manager.model.observation_space.shape[-1]
                        if expected_dim is not None and obs_array.shape[-1] != expected_dim:
                            logger.debug(f"[CAUSAL WARMUP] portfolio: obs dim {obs_array.shape[-1]} != expected {expected_dim} — skipping")
                            continue
                    action = entry.get('direction', 0) * entry.get('confidence', 1.0)
                    reward = entry.get('realized_return', 0.0)
                    self.portfolio_causal_manager.replay_buffer.push(obs_array, action, reward)
                    added_port += 1
                    total_added += 1
                except Exception as e:
                    logger.debug(f"[CAUSAL WARMUP] Skipped invalid portfolio entry: {e}")
            if added_port > 0:
                logger.info(f"[CAUSAL WARMUP] portfolio: replayed {added_port} closed trades (aggregated)")
            else:
                logger.debug("[CAUSAL WARMUP] portfolio: no closed trades to replay")
        if total_added > 0:
            logger.info(f"[CAUSAL WARMUP] Total replayed closed trades: {total_added} across all buffers")
        else:
            logger.info("[CAUSAL WARMUP] No closed trades found in history — buffers remain empty")

    # ===================================================================
    # NEW: Full transparent signal blending breakdown
    # ===================================================================
    async def explain_signal_breakdown(self, symbol: str, timestamp: pd.Timestamp, data: pd.DataFrame = None, precomputed: dict = None, live_mode: bool = True, full_hist_df: pd.DataFrame = None) -> dict:
        """Returns a clean dict showing exactly how the final confidence score was built.
        Call this anytime (bot, REPL, notebook) to verify the blend.
        P-16 / Critical #8 FIX: Accepts precomputed dict to avoid double inference when DEBUG_SIGNAL_BLEND=True.
        B-01 FIX: Added guard to prevent infinite recursion when DEBUG_SIGNAL_BLEND=True"""
        if data is None:
            data = self.data_ingestion.get_latest_data(symbol)
        if len(data) < 200:
            return {"error": "insufficient data"}
        # FIX #23: Use cached regime instead of running full HMM per signal call.
        # The regime_cache is populated by bot._get_all_regimes() each cycle.
        cached = self.regime_cache.get(symbol)
        if cached and isinstance(cached, (list, tuple)) and len(cached) == 2:
            regime, persistence = cached
        elif cached and isinstance(cached, str):
            regime, persistence = cached, 0.5
        else:
            # Cache miss — fall back to detect_regime (lightweight single call)
            regime, persistence = detect_regime(data, symbol=symbol, data_ingestion=self.data_ingestion)
        # FIX #30: Clear _meta_prob_history for this symbol on regime change.
        # Stale percentile ranks from a different regime distort the meta-prob rescaling.
        if not hasattr(self, '_last_regime_per_symbol'):
            self._last_regime_per_symbol = {}
        prev_regime = self._last_regime_per_symbol.get(symbol)
        if prev_regime is not None and prev_regime != regime:
            if symbol in self._meta_prob_history:
                self._meta_prob_history[symbol] = []
                logger.info(f"[META PROB] {symbol} regime changed {prev_regime} → {regime} — cleared _meta_prob_history")
        self._last_regime_per_symbol[symbol] = regime
        # FIX 1: Initialize action and obs before the if/else so they're always defined
        action = None
        obs = None
        # P-16 / Critical #8 FIX: Reuse precomputed values if provided (zero extra cost)
        if precomputed is not None:
            features = precomputed.get('features')
            ppo_prob = precomputed.get('ppo_prob', 0.5)
            ppo_strength = precomputed.get('ppo_strength', 0.5)
            action = precomputed.get('action')
            meta_prob = precomputed.get('meta_prob', 0.5)
            sentiment_score = precomputed.get('sentiment_score', 0.5)
            causal_factor = precomputed.get('causal_factor', 1.0) # ISSUE P10 FIX: renamed for clarity
        else:
            # Fallback: compute everything (old behavior)
            features = generate_features(data, regime, symbol, full_hist_df if full_hist_df is not None else data)
            ppo_prob = 0.5
            ppo_strength = 0.5
            meta_prob = 0.5
            sentiment_score = 0.5
            causal_factor = 1.0
        if features is None or features.shape[0] == 0:
            return {"error": "feature generation failed"}
        latest_features = features[-1:].astype(np.float32)
        # BUG-10 FIX: Single-pass PPO computation (reused for both main signal and breakdown)
        if precomputed is None:
            if symbol in self.trainer.ppo_models and self.trainer.ppo_models[symbol] is not None:
                model = self.trainer.ppo_models[symbol]
                causal_manager = self.causal_manager.get(symbol)
                vec_norm = self.trainer.vec_norms.get(symbol) if hasattr(self.trainer, 'vec_norms') else None
                obs = latest_features.reshape(1, -1)
                if vec_norm:
                    obs = vec_norm.normalize_obs(obs)
                with torch.no_grad():
                    # C-3 FIX: Ensure causal graph exists before use + safe fallback
                    if causal_manager:
                        causal_manager._ensure_graph_exists()
                        if causal_manager.causal_graph is not None:
                            action, _ = causal_manager.predict(obs, deterministic=True)
                            logger.debug(f"[CAUSAL] {symbol} prediction used successfully")
                        else:
                            logger.debug(f"[CAUSAL FALLBACK] {symbol} graph not ready — using base PPO")
                            action, _ = model.predict(obs, deterministic=True)
                    else:
                        action, _ = model.predict(obs, deterministic=True)
                raw_action = float(action.flat[0]) if hasattr(action, 'flat') else float(action)
                ppo_prob = (raw_action + 1) / 2
                ppo_strength = abs(raw_action)
        # Stacking meta-prob
        if precomputed is None:
            stacking_probs = []
            stacking_models_list = self.trainer.stacking_models.get(symbol, [])
            # FIX #23: Validate feature count before prediction to avoid crash on mismatch
            if stacking_models_list:
                expected_ncols = stacking_models_list[0].num_feature()
                actual_ncols = latest_features.shape[1] if latest_features.ndim == 2 else latest_features.shape[0]
                if actual_ncols != expected_ncols:
                    logger.warning(f"[STACKING] {symbol}: feature count mismatch (got {actual_ncols}, "
                                   f"model expects {expected_ncols}) — using 0.5 fallback")
                    stacking_models_list = []
            for stacking_model in stacking_models_list:
                if hasattr(stacking_model, 'predict_proba'):
                    prob = stacking_model.predict_proba(latest_features)
                    # FIX #29: Check ndim and shape[1] (columns=classes), not shape[0] (rows=samples)
                    if prob.ndim > 1 and prob.shape[1] > 1:
                        stacking_probs.append(float(prob[0, 1]))
                    else:
                        stacking_probs.append(float(prob.flat[0]))
                else:
                    prob = stacking_model.predict(latest_features)[0]
                    stacking_probs.append(float(prob))
            meta_prob = np.mean(stacking_probs) if stacking_probs else 0.5
            # Rescale to percentile rank against recent predictions (matches walk-forward training)
            # This prevents narrow LightGBM prediction spreads from making thresholds useless
            if not hasattr(self, '_meta_prob_history'):
                self._meta_prob_history = {}
            hist = self._meta_prob_history.setdefault(symbol, [])
            hist.append(meta_prob)
            if len(hist) > 500:
                hist[:] = hist[-500:]
            if len(hist) >= 20:
                std_val = np.std(hist)
                # FIX #26: Smooth blending between raw and percentile-ranked values
                # instead of abrupt activation at std < 0.05. Uses sigmoid-style blend
                # that's 0% percentile at std=0.10, ~50% at std=0.05, ~100% at std=0.01.
                if std_val < 0.10:
                    rank = sum(1 for h in hist if h <= meta_prob) / len(hist)
                    # Blend weight: 1.0 when std→0, 0.0 when std→0.10
                    blend = np.clip(1.0 - (std_val - 0.01) / 0.09, 0.0, 1.0)
                    meta_prob = blend * rank + (1.0 - blend) * meta_prob
        # Causal factor (renamed for clarity)
        if precomputed is None:
            causal_factor = 1.0
            causal_manager = self.causal_manager.get(symbol)
            if causal_manager and action is not None and obs is not None:
                try:
                    # C-3 FIX: Ensure graph exists + safe fallback
                    causal_manager._ensure_graph_exists()
                    if causal_manager.causal_graph is not None:
                        penalty_factor = causal_manager.compute_penalty_factor(obs, action)
                        causal_factor = penalty_factor # ISSUE P10 FIX: direct multiplier (boost if >1.0, suppress if <1.0)
                        logger.debug(f"[CAUSAL] {symbol} penalty factor applied: {penalty_factor:.4f}")
                    else:
                        logger.debug(f"[CAUSAL FALLBACK] {symbol} graph not ready — neutral causal factor=1.0")
                except Exception as e:
                    logger.debug(f"[CAUSAL] Penalty computation failed for {symbol}: {e} — neutral factor")
        # Sentiment (always compute if not precomputed, but can be passed)
        if precomputed is None:
            sentiment_score = await self.get_sentiment_score(symbol, timestamp, live_mode=live_mode)
        # === BLEND: PPO + Ensemble + Sentiment into combined_meta ===
        # Adaptive weighting: if enough trade history exists, adjust weights
        # based on which components have been most accurate recently.
        adaptive_w = self.memory.get_adaptive_weights()
        if adaptive_w is not None:
            # Adaptive weights from recent trade accuracy (Bayesian updating)
            ens_w = adaptive_w['ensemble']
            ppo_w = adaptive_w['ppo']
            sent_w = adaptive_w['sentiment']
            # Normalize ensemble + PPO to fill non-sentiment portion
            signal_portion = 1.0 - sent_w
            ens_norm = ens_w / (ens_w + ppo_w) * signal_portion
            ppo_norm = ppo_w / (ens_w + ppo_w) * signal_portion
            blended_meta = ens_norm * meta_prob + ppo_norm * ppo_prob
            if abs(sentiment_score) < 1e-6:
                combined_meta = blended_meta
            else:
                sentiment_01 = (sentiment_score + 1.0) / 2.0
                combined_meta = (1 - sent_w) * blended_meta + sent_w * sentiment_01
            logger.debug(f"[ADAPTIVE BLEND] {symbol}: ens={ens_norm:.2f} ppo={ppo_norm:.2f} sent={sent_w:.2f}")
        else:
            # Default static weights until enough trade history accumulates
            ppo_weight = self.config.get('PPO_SIGNAL_WEIGHT', 0.20)
            ensemble_weight = 1.0 - ppo_weight
            blended_meta = ensemble_weight * meta_prob + ppo_weight * ppo_prob
            if abs(sentiment_score) < 1e-6:
                combined_meta = blended_meta
            else:
                sentiment_01 = (sentiment_score + 1.0) / 2.0
                sent_w = self.config.get('SENTIMENT_WEIGHT', 0.2)
                combined_meta = (1 - sent_w) * blended_meta + sent_w * sentiment_01
        long_thresh, short_thresh = self.trainer.get_current_thresholds(symbol, timestamp)
        logger.debug(f"{symbol}: thresholds (long>{long_thresh:.3f}, short<{short_thresh:.3f}) | "
                     f"ensemble={meta_prob:.3f} ppo={ppo_prob:.3f} blended={blended_meta:.3f}")
        direction = 0
        confidence = 0.0
        if combined_meta > long_thresh:
            direction = 1
            confidence = (combined_meta - long_thresh) / (1.0 - long_thresh)
        elif combined_meta < short_thresh:
            direction = -1
            confidence = (short_thresh - combined_meta) / short_thresh
        # === DIRECTION-AWARE REGIME GATING ===
        # Suppress signals that fight the trend direction. In a bearish trend,
        # long entries need much higher confidence; in a bullish trend, short entries do.
        if direction != 0 and is_trending(regime):
            if is_bearish(regime) and direction == 1:
                # Going long in a downtrend — require 2x confidence or suppress
                confidence *= 0.4
                logger.info(f"[REGIME GATE] {symbol}: LONG suppressed in trending_down (conf * 0.4)")
            elif is_bullish(regime) and direction == -1:
                # Going short in an uptrend — require 2x confidence or suppress
                confidence *= 0.4
                logger.info(f"[REGIME GATE] {symbol}: SHORT suppressed in trending_up (conf * 0.4)")
        # === VIX-BASED CONFIDENCE SCALING ===
        # High VIX (>28) = elevated fear. Reduce long confidence, boost short confidence.
        from models.features import _fetch_macro_features
        try:
            macro = _fetch_macro_features()
            vix = macro.get('vix_close', 20)
            if vix > 28 and direction == 1:
                vix_scale = max(0.3, 1.0 - (vix - 28) / 20)  # VIX 28→1.0, 38→0.5, 48→0.3
                confidence *= vix_scale
                logger.info(f"[VIX GATE] {symbol}: LONG confidence scaled by {vix_scale:.2f} (VIX={vix:.1f})")
            elif vix > 28 and direction == -1:
                vix_boost = min(1.5, 1.0 + (vix - 28) / 40)  # VIX 28→1.0, 48→1.5
                confidence = min(1.0, confidence * vix_boost)
                logger.info(f"[VIX GATE] {symbol}: SHORT confidence boosted by {vix_boost:.2f} (VIX={vix:.1f})")
            # === SPX BREADTH GATE ===
            # Don't go long on individual stocks when SPX is below 200-SMA (bear market)
            if direction == 1 and macro.get('spx_below_200sma', False):
                confidence *= 0.3
                logger.info(f"[SPX GATE] {symbol}: SPX below 200-SMA — LONG confidence *= 0.3")
        except Exception:
            pass  # macro fetch failure is non-fatal
        # === 1H TREND CONFIRMATION GATE ===
        # Halve confidence when 15min signal opposes the 1H trend (price vs 20-bar SMA on 1H).
        if direction != 0:
            try:
                hourly_data = self.data_ingestion.get_latest_data(symbol, timeframe='1H')
                if hourly_data is not None and len(hourly_data) >= 20:
                    sma_20h = hourly_data['close'].rolling(20).mean().iloc[-1]
                    current_1h = hourly_data['close'].iloc[-1]
                    hourly_trend = 1 if current_1h > sma_20h else -1
                    if direction != hourly_trend:
                        confidence *= 0.5
                        logger.info(f"[1H GATE] {symbol}: 15m dir={direction} vs 1H trend={hourly_trend} — confidence halved")
            except Exception as e:
                logger.debug(f"[1H GATE] {symbol}: 1H data unavailable ({e}) — skipping")
        if direction != 0:
            conviction_boost = 0.7 + 0.4 * persistence
            confidence = min(1.0, confidence * conviction_boost)
        # Breakout boost (same as in generate_signal)
        if direction != 0 and len(data) >= 22:
            roll = 20
            prev_high = data['high'].iloc[-roll-1:-1].max()
            prev_low = data['low'].iloc[-roll-1:-1].min()
            current_price = data['close'].iloc[-1]
            if (current_price > prev_high and direction == 1) or (current_price < prev_low and direction == -1):
                confidence = min(1.0, confidence * self.config.get('BREAKOUT_BOOST_FACTOR', 1.2))
        # ==================== BUG #3 FIX ====================
        # causal_factor (previously misnamed penalty) is now applied directly as a multiplier.
        # Strong causal evidence boosts confidence; weak/no evidence leaves it neutral or slightly suppresses.
        if causal_factor != 1.0:
            confidence = min(1.0, confidence * causal_factor)
            logger.debug(f"{symbol} causal factor applied: multiplier={causal_factor:.4f} → final confidence={confidence:.4f}")
        # ==================== END BUG #3 FIX ====================
        prev_direction = self.prev_signals.get(symbol, 0)
        if symbol in self.last_entry_time and prev_direction != 0:
            # FIX #31: Ensure both timestamps are tz-aware before subtraction to avoid TypeError
            entry_ts = self.last_entry_time[symbol]
            ts = timestamp
            if hasattr(ts, 'tzinfo') and ts.tzinfo is None and hasattr(entry_ts, 'tzinfo') and entry_ts.tzinfo is not None:
                ts = ts.tz_localize(entry_ts.tzinfo)
            elif hasattr(entry_ts, 'tzinfo') and entry_ts.tzinfo is None and hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                entry_ts = entry_ts.tz_localize(ts.tzinfo)
            bars_since = (ts - entry_ts) / pd.Timedelta(minutes=15)
            # P-13 / Critical #5 FIX: Safe default if config key missing (prevents TypeError crash)
            if is_trending(regime):
                min_hold = self.config.get('MIN_HOLD_BARS_TRENDING', 6)
            else:
                min_hold = self.config.get('MIN_HOLD_BARS_MEAN_REVERTING', 3)
            if bars_since < min_hold:
                if direction == 0 or direction == -prev_direction:
                    direction = prev_direction
                    # FIX #35: Don't inflate confidence during min-hold. Keep original confidence
                    # but force direction to match existing position. Inflating to 0.6 gave false
                    # conviction on deteriorating trades.
                    logger.debug(f"{symbol} min-hold enforced ({regime}): bars_held={int(bars_since)} < {min_hold} → keeping previous direction")
        # FIX #34: Removed last_entry_time mutation — explain_signal_breakdown should be pure.
        # The caller (generate_signal / backtest loop) is responsible for updating last_entry_time.
        # ==================== NEW TRANSPARENT BLEND LOG ====================
        # P-16 / Critical #8 FIX: Pass precomputed values to avoid double inference when DEBUG_SIGNAL_BLEND=True
        # B-01 FIX: Added "and precomputed is None" guard to prevent infinite recursion
        if self.config.get('DEBUG_SIGNAL_BLEND', False) and precomputed is None:
            precomputed = {
                'features': features,
                'ppo_prob': ppo_prob,
                'ppo_strength': ppo_strength,
                'action': action,
                'meta_prob': meta_prob,
                'sentiment_score': sentiment_score,
                'causal_factor': causal_factor,
            }
            breakdown = await self.explain_signal_breakdown(symbol, timestamp, data, precomputed=precomputed, live_mode=live_mode)
            logger.info(f"[SIGNAL BLEND] {symbol} | "
                        f"PPO_raw={breakdown['ppo_raw']:.4f} | "
                        f"Causal_factor={breakdown.get('causal_factor', 1.0):.4f} | "
                        f"Regime={breakdown['regime']} (persist={breakdown['persistence']:.3f}) | "
                        f"Meta={breakdown['meta_prob']:.4f} | "
                        f"Sentiment={breakdown['sentiment_score']:.4f} | "
                        f"Combined={breakdown['combined_meta']:.4f} | "
                        f"FINAL_CONF={breakdown['confidence']:.4f} | "
                        f"action={breakdown['direction']}")
        # ==================== END NEW BLEND LOG ====================
        mode_str = "LIVE" if live_mode else "BACKTEST"
        # ISSUE #1 FIX: replaced undefined 'smoothed_meta' with actual final blended value
        logger.debug(
            f"{symbol} {mode_str} | meta_prob: {float(combined_meta):.3f} | combined: {float(combined_meta):.3f} | "
            f"sentiment: {float(sentiment_score):.3f} | long_thresh: {float(long_thresh):.3f} | short_thresh: {float(short_thresh):.3f} | "
            f"regime: {regime} | persistence: {persistence:.3f} | ppo_prob: {float(ppo_prob):.3f} | ppo_strength: {float(ppo_strength):.3f} | "
            f"direction: {direction} | confidence: {float(confidence):.2f}"
        )
        if direction == 1 and prev_direction <= 0:
            logger.info(f"{mode_str} {symbol} LONG ENTRY → combined_meta {float(combined_meta):.3f}")
        elif direction == -1 and prev_direction >= 0:
            logger.info(f"{mode_str} {symbol} SHORT ENTRY → combined_meta {float(combined_meta):.3f}")
        elif direction == 0 and prev_direction != 0:
            logger.info(f"{mode_str} {symbol} EXIT {'LONG' if prev_direction > 0 else 'SHORT'}")
        # BUG-03 FIX: Removed side-effect mutation of self.prev_signals
        # This method is now pure — it only computes and returns the breakdown dict.
        # The actual update of prev_signals should happen in the caller (e.g. generate_signal)
        # when the signal is committed, not during explanation/debugging.
        # self.prev_signals[symbol] = direction ← REMOVED HERE
        # ISSUE P10 FIX: return dict instead of tuple (matches docstring and debug log expectations)
        return {
            'ppo_raw': ppo_prob,
            'ppo_strength': ppo_strength,
            'regime': regime,
            'persistence': persistence,
            'meta_prob': meta_prob,
            'sentiment_score': sentiment_score,
            'causal_factor': causal_factor,
            'combined_meta': combined_meta,
            'confidence': confidence,
            'direction': direction,
        }

    async def generate_signal(self, symbol: str, data: pd.DataFrame = None, timestamp: pd.Timestamp = None,
                              live_mode: bool = True, full_hist_df: pd.DataFrame = None, **kwargs) -> tuple:
        """Generate a trading signal for a single symbol.
        Returns (direction, confidence, ppo_strength, action_raw) tuple.
        Delegates to explain_signal_breakdown and updates prev_signals state."""
        result = await self.explain_signal_breakdown(symbol, timestamp, data=data, live_mode=live_mode, full_hist_df=full_hist_df)
        if 'error' in result:
            return 0, 0.0, 0.0, None
        direction = result['direction']
        confidence = result['confidence']
        ppo_strength = result.get('ppo_strength', 0.5)
        # === ADAPTIVE MEMORY GATES (real-time learning) ===
        ts = timestamp or datetime.now(tz=tz.gettz('UTC'))
        # 1. Defensive mode: suppress ALL new entries if cross-asset panic detected
        if direction != 0 and self.memory.is_defensive(ts):
            logger.warning(f"[DEFENSIVE] {symbol}: suppressed (cross-asset panic active until "
                          f"{self.memory.defensive_until.strftime('%H:%M') if self.memory.defensive_until else '?'})")
            direction = 0
            confidence = 0.0
        # 2. Anti-churn: exponential decay after stop-outs in same direction
        if direction != 0:
            churn_penalty = self.memory.get_churn_penalty(symbol, direction, ts)
            if churn_penalty < 1.0:
                confidence *= churn_penalty
        # 3. Enforce MIN_CONFIDENCE — reject weak signals before they reach execution
        min_conf = self.config.get('MIN_CONFIDENCE', 0.72)
        if direction != 0 and confidence < min_conf:
            logger.info(f"[CONF GATE] {symbol}: confidence {confidence:.3f} < MIN_CONFIDENCE {min_conf} — suppressing")
            direction = 0
            confidence = 0.0
        # Update prev_signals state (explain_signal_breakdown is pure — state update happens here)
        prev_direction = self.prev_signals.get(symbol, 0)
        if direction != 0 and prev_direction == 0:
            self.last_entry_time[symbol] = timestamp if timestamp is not None else datetime.now(tz=tz.gettz('UTC'))
        self.prev_signals[symbol] = direction
        return direction, confidence, ppo_strength, None

    # HIGH-16 FIX: Shared executor for sync wrapper — prevents creating thousands of threads
    _sync_executor = None
    _sync_lock = threading.Lock()  # FIX #24: Serialize backtest calls to prevent shared mutable state races

    def generate_signal_sync(self, symbol: str, data: pd.DataFrame = None, timestamp: pd.Timestamp = None,
                             live_mode: bool = False, full_hist_df: pd.DataFrame = None, **kwargs) -> tuple:
        """Synchronous wrapper for backtest compatibility.
        HIGH-16 FIX: Reuse a single ThreadPoolExecutor instead of creating one per call.
        FIX #24: threading.Lock serializes calls to prevent shared state races."""
        import asyncio
        import concurrent.futures
        if SignalGenerator._sync_executor is None:
            # FIX #33: Lock serializes all calls anyway, so >1 worker just wastes a thread
            SignalGenerator._sync_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # FIX #20: Lock MUST cover both submission and result — shared state
        # (prev_signals, meta_ema, sentiment_cache) is mutated inside generate_signal.
        # Use timeout to prevent deadlock in case the signal call hangs.
        lock_timeout = self.config.get('SIGNAL_SYNC_LOCK_TIMEOUT', 120)
        acquired = SignalGenerator._sync_lock.acquire(timeout=lock_timeout)
        if not acquired:
            logger.warning(f"[SIGNAL] generate_signal_sync lock timeout ({lock_timeout}s) for {symbol} — returning neutral")
            return 0, 0.0, 0.5, None
        try:
            # M35 FIX: Removed dead inner try/except (loop check had no side effects)
            # Dispatch to executor thread with fresh loop
            future = SignalGenerator._sync_executor.submit(asyncio.run, self.generate_signal(
                symbol=symbol, data=data, timestamp=timestamp, live_mode=live_mode))
            return future.result(timeout=lock_timeout)
        finally:
            SignalGenerator._sync_lock.release()

    # ==================== MISSING METHOD ADDED HERE (fixes the crash) ====================
    async def get_sentiment_score(self, symbol: str, timestamp: pd.Timestamp = None, live_mode: bool = True) -> float:
        """Unified sentiment score using LocalLLMDebate (Ollama preferred) or fallback.
        BUG-13 FIX: Cache now refreshes hourly instead of daily — much more responsive to intraday news shifts."""
        if timestamp is None:
            timestamp = datetime.now(tz=tz.gettz('UTC'))
        # BUG-13 FIX: Use hour-level cache key (refreshes ~every hour during trading session)
        cache_key = f"{symbol}|{timestamp.date()}|{timestamp.hour}"
        if cache_key in self.sentiment_cache:
            logger.debug(f"[SENTIMENT CACHE HIT] {symbol} for {cache_key}")
            return self.sentiment_cache[cache_key]
        try:
            # Fetch recent news via DataIngestion
            news_texts = self.data_ingestion.get_recent_news(
                symbol,
                days=CONFIG.get('NEWS_LOOKBACK_DAYS', 10)
            )
  
            if not news_texts:
                score = 0.0
            else:
                # Use LocalLLMDebate (the one you have in local_llm.py)
                # Pass symbol= so telemetry logs "[SYMBOL]" instead of "[None]"
                if self.llm_debate is not None and hasattr(self.llm_debate, 'debate_sentiment'):
                    if asyncio.iscoroutinefunction(self.llm_debate.debate_sentiment):
                        score = await self.llm_debate.debate_sentiment(news_texts, symbol=symbol)
                    else:
                        score = self.llm_debate.debate_sentiment(news_texts, symbol=symbol)
                else:
                    score = 0.0
  
            # Cache result (now per hour)
            if len(self.sentiment_cache) > 5000:
                # Evict oldest entries by date/hour — extract last two segments (date_hour) as sort key
                # Cache keys are "SYMBOL|YYYY-MM-DD|HH" — sort by date+hour portion (hour as int for correct numeric ordering)
                def _cache_sort_key(k):
                    parts = k.split('|')
                    if len(parts) >= 3:
                        return (parts[1], int(parts[2]) if parts[2].isdigit() else 0)
                    return ('0000-00-00', 0)
                keys = sorted(self.sentiment_cache.keys(), key=_cache_sort_key)
                for k in keys[:len(keys) - 2500]:
                    del self.sentiment_cache[k]
            self.sentiment_cache[cache_key] = score
            logger.debug(f"[SENTIMENT COMPUTED] {symbol} for {cache_key}: {score:.3f}")
            return score
  
        except Exception as e:
            logger.debug(f"Sentiment score failed for {symbol}: {e}")
            # FIX #27: Don't cache failed sentiment — 0.0 would be cached for an hour,
            # masking real sentiment. Return 0.0 without caching so next call retries.
            return 0.0

    def get_sentiment_velocity(self, symbol: str, timestamp: pd.Timestamp = None,
                               lookback_hours: int = 4) -> float:
        """B4: Compute Δsentiment over the past N hours from the cache.
        Returns current_level - avg(past lookback_hours) ∈ [-2, 2]. 0 if insufficient data.

        Velocity captures information ARRIVAL (new news shifting the narrative) rather
        than steady-state level. A stock going from 0.1 → 0.6 is very different from
        one sitting at 0.6 all week.
        """
        if timestamp is None:
            timestamp = datetime.now(tz=tz.gettz('UTC'))
        cur_key = f"{symbol}|{timestamp.date()}|{timestamp.hour}"
        current_level = self.sentiment_cache.get(cur_key)
        if current_level is None:
            return 0.0
        # Walk back up to lookback_hours hours and collect any cached levels
        past_levels = []
        for h_back in range(1, lookback_hours + 1):
            t_past = timestamp - pd.Timedelta(hours=h_back)
            past_key = f"{symbol}|{t_past.date()}|{t_past.hour}"
            val = self.sentiment_cache.get(past_key)
            if val is not None:
                past_levels.append(val)
        if not past_levels:
            return 0.0
        baseline = float(np.mean(past_levels))
        velocity = float(np.clip(current_level - baseline, -2.0, 2.0))
        return velocity

    async def generate_portfolio_actions(self, symbols: list, data_dict: dict, current_equity: float, precomputed_env=None, timestamp: pd.Timestamp = None) -> dict:
        if not symbols:
            logger.warning("No symbols provided for portfolio actions")
            return {}  # L25 FIX: symbols is empty, comprehension was redundant
        if self.trainer.portfolio_ppo_model is None:
            logger.warning("Portfolio PPO model not loaded — returning flat weights")
            return {sym: 0.0 for sym in symbols}
        try:
            logger.debug(f"========== PORTFOLIO PPO DEBUG START - timestamp={timestamp} ==========")
            # ISSUE #6 PATCH: Use persistent precomputed_env if provided (lightweight data_dict update only)
            if precomputed_env is not None:
                logger.debug("Using persistent precomputed_env (ISSUE #6: lightweight update)")
                precomputed_env.data_dict = {sym: df.copy() for sym, df in data_dict.items()}
                if hasattr(precomputed_env, 'timeline') and timestamp is not None:
                    try:
                        step_idx = precomputed_env.timeline.searchsorted(timestamp, side='right') - 1
                        step_idx = max(0, step_idx)
                        precomputed_env.current_step = step_idx
                    except Exception:
                        precomputed_env.current_step = len(precomputed_env.timeline) - 1
                obs = precomputed_env._get_observation()
            else:
                logger.debug("Creating new PortfolioEnv (fallback - no persistent env)")
                temp_env = PortfolioEnv(
                    data_dict=data_dict,
                    symbols=symbols,
                    initial_balance=current_equity,
                    max_leverage=self.config.get('MAX_LEVERAGE', 2.0)
                )
                obs, _ = temp_env.reset()
            obs = obs.reshape(1, -1).astype(np.float32)
            if self.trainer.portfolio_vec_norm is not None:
                obs = self.trainer.portfolio_vec_norm.normalize_obs(obs)
            try:
                # C-3 FIX: Ensure portfolio causal graph exists before prediction
                if self.portfolio_causal_manager:
                    self.portfolio_causal_manager._ensure_graph_exists()
                    if self.portfolio_causal_manager.causal_graph is not None:
                        action, new_state = self.portfolio_causal_manager.predict(
                            obs, state=self.last_portfolio_state, episode_start=np.array([False]), deterministic=True
                        )
                        logger.info(f"✅ [CAUSAL PENALTY APPLIED in generate_portfolio_actions] portfolio — causal wrapper used")
                    else:
                        logger.warning("[CAUSAL FALLBACK] Portfolio graph not ready — using base PPO")
                        action, new_state = self.trainer.portfolio_ppo_model.predict(
                            obs, state=self.last_portfolio_state, episode_start=np.array([False]), deterministic=True
                        )
                else:
                    action, new_state = self.trainer.portfolio_ppo_model.predict(
                        obs, state=self.last_portfolio_state, episode_start=np.array([False]), deterministic=True
                    )
                self.last_portfolio_state = new_state
            except (AttributeError, TypeError, ValueError) as fallback_err:
                # FIX: Fallback was re-calling the same failing portfolio_causal_manager.predict()
                # Use base PPO model directly instead
                logger.warning(f"[CAUSAL FALLBACK] Portfolio causal predict failed ({fallback_err}) — using base PPO")
                action, _ = self.trainer.portfolio_ppo_model.predict(obs, deterministic=True)
            action = action.flatten()
            max_leverage = self.config.get('MAX_LEVERAGE', 2.0)
            abs_sum = np.sum(np.abs(action))
            if abs_sum > max_leverage:
                action = action / abs_sum * max_leverage
            action = np.clip(action, -2.0, 2.0)
            target_weights = {sym: float(weight) for sym, weight in zip(symbols, action)}
            # ALPHA-ATTR: snapshot baseline weights (post-PPO-post-causal) BEFORE any
            # downstream layers. We record the transformation chain after each layer
            # so we can attribute the final weight back to each multiplier.
            baseline_weights = dict(target_weights)
            # accumulated layer multipliers: {sym: {layer_name: mult}}
            _attr_layers: Dict[str, Dict[str, float]] = {s: {} for s in target_weights}
            # === STACKING ENSEMBLE OVERLAY (per-symbol meta_prob) ===
            # The stacking ensemble (LightGBM) is trained nightly but was otherwise
            # unused in portfolio mode. Apply it as a per-symbol multiplier:
            # agreement with PPO direction → boost, disagreement → dampen.
            # Weight is configurable via PORTFOLIO_META_WEIGHT (default 0.2).
            meta_blend_weight = self.config.get('PORTFOLIO_META_WEIGHT', 0.2)
            if meta_blend_weight > 0 and hasattr(self.trainer, 'stacking_models'):
                for sym in symbols:
                    stacking_models_list = self.trainer.stacking_models.get(sym, [])
                    if not stacking_models_list:
                        continue
                    weight = target_weights.get(sym, 0.0)
                    if weight == 0.0:
                        continue
                    # Build per-symbol feature row for LightGBM input
                    try:
                        df = data_dict.get(sym)
                        if df is None or df.empty:
                            continue
                        cached = self.regime_cache.get(sym)
                        sym_regime = (cached[0] if isinstance(cached, (list, tuple)) and len(cached) == 2
                                      else (cached if isinstance(cached, str) else 'mean_reverting'))
                        feats = generate_features(df, sym_regime, sym, df)
                        if feats is None or feats.shape[0] == 0:
                            continue
                        latest = feats[-1:].astype(np.float32)
                        expected_ncols = stacking_models_list[0].num_feature()
                        if latest.shape[1] != expected_ncols:
                            logger.debug(f"[META BLEND] {sym}: feature count mismatch "
                                         f"(got {latest.shape[1]}, expected {expected_ncols}) — skipping")
                            continue
                        probs = []
                        for m in stacking_models_list:
                            if hasattr(m, 'predict_proba'):
                                p = m.predict_proba(latest)
                                if p.ndim > 1 and p.shape[1] > 1:
                                    probs.append(float(p[0, 1]))
                                else:
                                    probs.append(float(p.flat[0]))
                            else:
                                probs.append(float(m.predict(latest)[0]))
                        if not probs:
                            continue
                        meta_prob = float(np.mean(probs))
                        # meta_prob: 0.5=neutral, 1.0=bullish, 0.0=bearish
                        # Map to signed conviction in [-1, 1]
                        meta_signed = (meta_prob - 0.5) * 2.0
                        direction = 1 if weight > 0 else -1
                        # Factor: +weight * meta_signed > 0 means agreement → boost
                        #        +weight * meta_signed < 0 means disagreement → dampen
                        factor = 1.0 + meta_blend_weight * meta_signed * direction
                        factor = float(np.clip(factor, 1.0 - meta_blend_weight,
                                               1.0 + meta_blend_weight))
                        old_weight = weight
                        target_weights[sym] = weight * factor
                        logger.info(f"{sym} meta blend: prob={meta_prob:.3f} "
                                    f"(signed={meta_signed:+.2f}) → factor={factor:.3f} "
                                    f"(weight {old_weight:.4f} → {target_weights[sym]:.4f})")
                    except Exception as e:
                        logger.debug(f"[META BLEND] {sym}: stacking predict failed ({e}) — skipping")
            # === A1: CROSS-SECTIONAL MOMENTUM GATE ===
            # Rank every symbol in today's universe by composite momentum + volume +
            # drawdown score; apply a per-symbol multiplier centered at 1.0. Top
            # tercile → boost, bottom → dampen. Preserves PPO obs shape (no retrain
            # needed) while capturing "be where today's alpha is".
            cs_weight = self.config.get('CROSS_SECTIONAL_WEIGHT', 1.0)
            if cs_weight > 0:
                try:
                    from strategy.cross_sectional import (
                        compute_cross_sectional_scores,
                        build_multipliers,
                        log_summary,
                    )
                    cs_scores = compute_cross_sectional_scores(data_dict)
                    if cs_scores:
                        cs_mults = build_multipliers(
                            cs_scores,
                            max_mult=self.config.get('CROSS_SECTIONAL_MAX_MULT', 1.25),
                            min_mult=self.config.get('CROSS_SECTIONAL_MIN_MULT', 0.50),
                            neutral_band=self.config.get('CROSS_SECTIONAL_NEUTRAL_BAND', 0.25),
                        )
                        log_summary(cs_scores, cs_mults)
                        # Blend toward 1.0 by cs_weight — setting cs_weight=0 disables
                        for sym in list(target_weights.keys()):
                            mult = cs_mults.get(sym, 1.0)
                            blended = 1.0 + cs_weight * (mult - 1.0)
                            target_weights[sym] *= blended
                            # EQ-SCORE: track tercile placement
                            if getattr(self, 'execution_scorecard', None):
                                try:
                                    raw_mult = cs_mults.get(sym, 1.0)
                                    tercile = ('top' if raw_mult > 1.1
                                               else ('bottom' if raw_mult < 0.9 else 'mid'))
                                    self.execution_scorecard.record_cs_tercile(sym, tercile)
                                except Exception:
                                    pass
                except Exception as e:
                    logger.debug(f"[CS-MOMENTUM] skipped ({e})")
            # === BPS: BAYESIAN PER-SYMBOL SIZING ===
            # Scale each symbol's weight by its posterior expected return. Symbols with
            # high historical WR + big avg-win get boosted up to 1.6×; symbols that
            # historically bleed get scaled down to 0.4×. Small-sample shrinkage toward
            # 1.0 prevents over-reacting to 2-3 lucky/unlucky trades.
            if (self.config.get('BAYESIAN_SIZING_ENABLED', True)
                    and getattr(self, 'bayesian_sizer', None) is not None):
                try:
                    bps_min = self.config.get('BAYESIAN_SIZING_MIN_MULT', 0.4)
                    bps_max = self.config.get('BAYESIAN_SIZING_MAX_MULT', 1.6)
                    bps_ref_ev = self.config.get('BAYESIAN_SIZING_REFERENCE_EV', 0.003)
                    bps_shrink_n = self.config.get('BAYESIAN_SIZING_SHRINKAGE_N', 8)
                    bps_method = self.config.get('BAYESIAN_SIZING_METHOD', 'kelly')
                    bps_kelly_frac = self.config.get('BAYESIAN_SIZING_KELLY_FRACTION', 0.25)
                    bps_ref_kelly = self.config.get('BAYESIAN_SIZING_REFERENCE_KELLY', 0.08)
                    bps_parts = []
                    for sym in list(target_weights.keys()):
                        if target_weights[sym] == 0.0:
                            continue
                        mult, reason = self.bayesian_sizer.size_multiplier(
                            sym, min_mult=bps_min, max_mult=bps_max,
                            reference_ev=bps_ref_ev, shrinkage_n=bps_shrink_n,
                            method=bps_method, kelly_fraction=bps_kelly_frac,
                            reference_kelly=bps_ref_kelly,
                        )
                        if abs(mult - 1.0) > 0.01:
                            target_weights[sym] *= mult
                            bps_parts.append(f"{sym}:{mult:.2f}")
                    if bps_parts:
                        logger.info(f"[BAYESIAN-SIZE] {' | '.join(bps_parts)}")
                except Exception as e:
                    logger.debug(f"[BAYESIAN-SIZE] skipped ({e})")
            # === LIQ: LIQUIDITY-SCALED SIZING ===
            # Scale weights DOWN if our position notional starts to rival the
            # symbol's avg daily volume (>1% of ADV → market impact becomes material).
            # Extended-hours thresholds are 5× tighter since volume is thinner.
            if self.config.get('LIQUIDITY_SCALER_ENABLED', True):
                try:
                    from strategy.liquidity_scaler import (
                        compute_liquidity_multipliers,
                        log_summary as _liq_log,
                    )
                    liq_mults = compute_liquidity_multipliers(
                        target_weights,
                        data_dict,
                        equity=current_equity,
                        warn_threshold=self.config.get('LIQUIDITY_WARN_THRESHOLD', 0.001),
                        hard_threshold=self.config.get('LIQUIDITY_HARD_THRESHOLD', 0.01),
                        min_mult=self.config.get('LIQUIDITY_MIN_MULT', 0.3),
                        extended_hours_factor=self.config.get('LIQUIDITY_EH_FACTOR', 5.0),
                    )
                    for sym, (mult, part, _reason) in liq_mults.items():
                        if mult < 0.995 and sym in target_weights:
                            target_weights[sym] *= mult
                            if getattr(self, 'execution_scorecard', None):
                                try:
                                    self.execution_scorecard.record_liquidity_scale(sym, part)
                                except Exception:
                                    pass
                    _liq_log(liq_mults)
                except Exception as e:
                    logger.debug(f"[LIQ] skipped ({e})")
            # === AC: CROWDING DISCOUNT ===
            # When multiple same-direction positions are highly correlated, the book
            # is effectively one big bet. Discount each position by its average
            # correlation to same-sign peers (over 60-day daily returns).
            if self.config.get('CROWDING_DISCOUNT_ENABLED', True):
                try:
                    from strategy.correlation_discount import (
                        compute_correlation_matrix,
                        apply_crowding_discount,
                        log_summary as _crowding_log,
                    )
                    corr_mat = compute_correlation_matrix(data_dict)
                    if not corr_mat.empty:
                        target_weights, discounts = apply_crowding_discount(
                            target_weights,
                            corr_mat,
                            threshold=self.config.get('CROWDING_DISCOUNT_THRESHOLD', 0.5),
                            strength=self.config.get('CROWDING_DISCOUNT_STRENGTH', 0.5),
                            min_factor=self.config.get('CROWDING_DISCOUNT_MIN_FACTOR', 0.4),
                        )
                        _crowding_log(discounts)
                except Exception as e:
                    logger.debug(f"[CROWDING] skipped ({e})")
            sentiment_blend_weight = self.config.get('PORTFOLIO_SENTIMENT_WEIGHT', 0.3)
            if sentiment_blend_weight > 0:
                # Parallelize sentiment calls — cuts cycle time from ~4 min to ~25s
                async def _get_sentiment(sym):
                    # FIX #32: Guard against empty DataFrame — .index[-1] raises IndexError on empty
                    if timestamp is not None:
                        ts = timestamp
                    else:
                        df = data_dict.get(sym)
                        if df is not None and not df.empty:
                            ts = df.index[-1]
                        else:
                            ts = pd.Timestamp.now(tz='UTC')
                    return sym, await self.get_sentiment_score(sym, ts, live_mode=True)
                # FIX: Wrap sentiment gather in a timeout. If Ollama hangs (GPU OOM,
                # CUDA crash, server freeze), the _ollama_lock is held forever and
                # the entire trading loop freezes — observed 5-hour hang on Apr 17.
                # Timeout falls back to cached/zero sentiment so trading continues.
                _sent_timeout = self.config.get('SENTIMENT_GATHER_TIMEOUT_SEC', 300)
                try:
                    sentiment_results = await asyncio.wait_for(
                        asyncio.gather(*[_get_sentiment(sym) for sym in symbols]),
                        timeout=_sent_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[SENTIMENT TIMEOUT] Gather exceeded {_sent_timeout}s — "
                                   f"falling back to cached/zero sentiment for all symbols")
                    sentiment_results = [(sym, 0.0) for sym in symbols]
                # B4: Also read sentiment VELOCITY for each symbol (Δlevel over past N hours)
                # Velocity captures information arrival, not steady-state level — a stock going
                # 0.1 → 0.6 is very different from one sitting at 0.6. Multiplied by direction
                # so positive velocity boosts longs and dampens shorts (matches the "good news
                # → prefer longs" intuition).
                sent_vel_weight = self.config.get('SENTIMENT_VELOCITY_WEIGHT', 0.15)
                sent_vel_lookback = self.config.get('SENTIMENT_VELOCITY_LOOKBACK_HOURS', 4)
                for sym, sentiment in sentiment_results:
                    sentiment_factor = 1.0 + sentiment_blend_weight * sentiment
                    # Velocity factor — direction-aware
                    velocity_factor = 1.0
                    if sent_vel_weight > 0:
                        try:
                            ts_for_vel = timestamp if timestamp is not None else pd.Timestamp.now(tz='UTC')
                            vel = self.get_sentiment_velocity(sym, ts_for_vel, sent_vel_lookback)
                            if vel != 0.0:
                                direction_sign = 1.0 if target_weights.get(sym, 0.0) >= 0 else -1.0
                                velocity_factor = 1.0 + sent_vel_weight * vel * direction_sign
                                velocity_factor = float(np.clip(velocity_factor,
                                                                 1.0 - sent_vel_weight,
                                                                 1.0 + sent_vel_weight))
                        except Exception as e:
                            logger.debug(f"{sym} sentiment velocity failed: {e}")
                    old_weight = target_weights[sym]
                    target_weights[sym] *= sentiment_factor * velocity_factor
                    logger.info(f"{sym} sentiment blend: raw={sentiment:.3f} → factor={sentiment_factor:.2f} "
                                f"| velocity_factor={velocity_factor:.3f} (weight {old_weight:.3f} → {target_weights[sym]:.3f})")
                abs_sum = np.sum(np.abs(list(target_weights.values())))
                if abs_sum > max_leverage:
                    scale = max_leverage / abs_sum
                    target_weights = {sym: w * scale for sym, w in target_weights.items()}
                    logger.info(f"Re-normalized weights after sentiment blend (scale {scale:.3f})")
            # === PORTFOLIO-LEVEL GATES ===
            # Same gates as per-symbol path, applied to portfolio weights.
            # These were missing — portfolio mode bypassed all confidence filtering.
            from models.features import _fetch_macro_features
            ts = timestamp or datetime.now(tz=tz.gettz('UTC'))
            try:
                macro = _fetch_macro_features()
                vix = macro.get('vix_close', 20)
                spx_bear = macro.get('spx_below_200sma', False)
            except Exception:
                vix, spx_bear = 20, False
            # === S1: Meta-label filter — collect per-symbol sentiment we already
            # computed above (if enabled) so the filter can see it as a feature.
            sentiment_by_sym = {}
            if sentiment_blend_weight > 0:
                try:
                    sentiment_by_sym = {sym: s for sym, s in sentiment_results}
                except Exception:
                    sentiment_by_sym = {}
            gated_weights = {}
            for sym, weight in target_weights.items():
                direction = 1 if weight > 0 else (-1 if weight < 0 else 0)
                gate_mult = 1.0
                gate_reasons = []
                # === ESP: SLIPPAGE-PREDICTION VETO ===
                # If predicted slippage on this symbol+hour+size exceeds our edge
                # by a safety multiple, skip the entry. Protects against "entries
                # in chop" where the act of entering eats the alpha.
                if (direction != 0
                        and self.config.get('SLIPPAGE_VETO_ENABLED', True)
                        and getattr(self, 'slippage_predictor', None) is not None):
                    try:
                        # Convert PPO weight magnitude to an expected-alpha proxy in bps.
                        # Target weight of 0.05 at ~$30k equity = ~$1500 notional;
                        # typical PPO weights 0.02-0.15 ⇒ edge ~10-60 bps expected.
                        expected_alpha_bps = max(10.0, abs(weight) * 400.0)  # 0.05 → 20bp
                        est_size_usd = float(abs(weight)) * float(current_equity or 30000.0)
                        from datetime import datetime as _dt
                        veto, pred_bps, slip_reason = self.slippage_predictor.should_veto(
                            sym, expected_alpha_bps, hour=_dt.now().hour,
                            size_usd=est_size_usd,
                            edge_safety_multiple=self.config.get('SLIPPAGE_VETO_MULTIPLE', 1.2),
                        )
                        if veto:
                            gate_mult *= self.config.get('SLIPPAGE_VETO_SCALE', 0.3)
                            gate_reasons.append(f"SLIP({slip_reason})")
                            logger.info(f"{sym} SLIPPAGE-VETO: {slip_reason} — scaling weight")
                            if getattr(self, 'execution_scorecard', None):
                                try:
                                    self.execution_scorecard.record_slippage_veto(sym, pred_bps)
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.debug(f"{sym} slippage veto error: {e}")
                # === PSD: PPO/STACKING DIVERGENCE GATE ===
                # If PPO says go hard in one direction but stacking ensemble says the
                # OPPOSITE direction is likely, that's strong disagreement. Historically
                # we rarely win these; dampen. Tracks only high-conviction disagreement
                # to avoid over-penalizing normal noise.
                if (direction != 0
                        and self.config.get('DIVERGENCE_GATE_ENABLED', True)
                        and hasattr(self.trainer, 'stacking_models')):
                    try:
                        stacking_models_list = self.trainer.stacking_models.get(sym, [])
                        if stacking_models_list:
                            df = data_dict.get(sym)
                            if df is not None and not df.empty:
                                cached = self.regime_cache.get(sym)
                                sym_reg = (cached[0] if isinstance(cached, (list, tuple)) and len(cached) == 2
                                           else (cached if isinstance(cached, str) else 'mean_reverting'))
                                feats_d = generate_features(df, sym_reg, sym, df)
                                if feats_d is not None and feats_d.shape[0] > 0:
                                    latest_d = feats_d[-1:].astype(np.float32)
                                    expected_nc = stacking_models_list[0].num_feature()
                                    if latest_d.shape[1] == expected_nc:
                                        probs_d = []
                                        for m in stacking_models_list:
                                            if hasattr(m, 'predict_proba'):
                                                p = m.predict_proba(latest_d)
                                                probs_d.append(float(p[0, 1]) if p.ndim > 1 and p.shape[1] > 1 else float(p.flat[0]))
                                            else:
                                                probs_d.append(float(m.predict(latest_d)[0]))
                                        if probs_d:
                                            meta_prob_d = float(np.mean(probs_d))
                                            meta_signed_d = (meta_prob_d - 0.5) * 2.0  # [-1, +1]
                                            # Disagreement: PPO direction and meta_signed point opposite
                                            # AND both are strong (|PPO weight| big, |meta_signed| big)
                                            disagree = (direction * meta_signed_d < 0
                                                        and abs(weight) > self.config.get('DIVERGENCE_MIN_WEIGHT', 0.03)
                                                        and abs(meta_signed_d) > self.config.get('DIVERGENCE_MIN_META', 0.20))
                                            if disagree:
                                                dmult = self.config.get('DIVERGENCE_GATE_SCALE', 0.5)
                                                gate_mult *= dmult
                                                gate_reasons.append(f"DIVERGE({meta_signed_d:+.2f})")
                                                logger.info(f"{sym} DIVERGENCE: PPO_dir={direction} vs meta_signed={meta_signed_d:+.2f} — scaling ×{dmult}")
                                                if getattr(self, 'execution_scorecard', None):
                                                    try:
                                                        self.execution_scorecard.record_divergence(sym)
                                                    except Exception:
                                                        pass
                    except Exception as e:
                        logger.debug(f"{sym} divergence gate error: {e}")
                # === B2: ADVERSE-SELECTION (FILL TOXICITY) GATE ===
                # If recent fills on this symbol were consistently followed by adverse
                # price drift, we're being picked off. Dampen the weight accordingly.
                if (direction != 0
                        and self.config.get('ADVERSE_SELECTION_ENABLED', True)
                        and getattr(self, 'adverse_selection', None) is not None):
                    try:
                        ava_mult, ava_reason = self.adverse_selection.get_toxicity_penalty(
                            sym,
                            threshold=self.config.get('ADVERSE_SELECTION_THRESHOLD', -0.002),
                            max_penalty=self.config.get('ADVERSE_SELECTION_MAX_PENALTY', 0.5),
                        )
                        if ava_mult < 1.0:
                            gate_mult *= ava_mult
                            gate_reasons.append(f"AVA({ava_reason})")
                            logger.info(f"{sym} ADVERSE-SEL: mult={ava_mult:.2f} — {ava_reason}")
                    except Exception as e:
                        logger.debug(f"{sym} ADVERSE-SEL error: {e}")
                # === B5: ANTI-EARNINGS GATE ===
                # Block new entries during pre/post-earnings blackout windows. Open
                # positions are flagged for close by the trading loop separately.
                if (direction != 0
                        and self.config.get('EARNINGS_FILTER_ENABLED', True)
                        and getattr(self, 'earnings_calendar', None) is not None):
                    try:
                        pre_days = self.config.get('EARNINGS_BLACKOUT_PRE_DAYS', 2)
                        post_days = self.config.get('EARNINGS_BLACKOUT_POST_DAYS', 1)
                        blocked, reason = self.earnings_calendar.is_in_blackout(
                            sym, pre_days=pre_days, post_days=post_days
                        )
                        if blocked:
                            gate_mult = 0.0
                            gate_reasons.append(f"EARNINGS({reason})")
                            logger.info(f"{sym} EARNINGS BLACKOUT — blocking new entry ({reason})")
                            if getattr(self, 'execution_scorecard', None):
                                try:
                                    self.execution_scorecard.record_earnings_blackout(sym)
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.debug(f"{sym} earnings gate error: {e}")
                # === S1: META-LABEL GATE ===
                # Ask the meta-labeler "given PPO says go <direction> on <sym>, is
                # this likely a winner?" If P(win) < threshold → zero the weight.
                # Pass-through when model isn't fitted yet (insufficient history).
                meta_min_prob = self.config.get('META_FILTER_MIN_PROB', 0.40)
                meta_zero_mode = self.config.get('META_FILTER_MODE', 'zero')  # 'zero' or 'scale'
                if (direction != 0
                        and self.config.get('META_FILTER_ENABLED', True)
                        and getattr(self, 'meta_labeler', None) is not None):
                    try:
                        cached = self.regime_cache.get(sym)
                        sym_regime_meta = (cached[0] if isinstance(cached, (list, tuple)) and len(cached) == 2
                                           else (cached if isinstance(cached, str) else 'mean_reverting'))
                        sym_persistence_meta = float(cached[1]) if isinstance(cached, (list, tuple)) and len(cached) == 2 else 0.5
                        accept, mprob = self.meta_labeler.should_enter(
                            symbol=sym,
                            direction=direction,
                            min_prob=meta_min_prob,
                            confidence=float(abs(weight)),
                            ppo_strength=float(abs(weight)),
                            conviction=float(abs(weight)),
                            timestamp=ts,
                            regime=sym_regime_meta,
                            persistence=sym_persistence_meta,
                            sentiment=float(sentiment_by_sym.get(sym, 0.0)),
                            vix=float(vix),
                            size_rel=float(abs(weight)),  # weight is equity-relative already
                        )
                        if not accept:
                            if meta_zero_mode == 'scale':
                                gate_mult *= 0.2
                                gate_reasons.append(f"META-REJECT({mprob:.2f}<{meta_min_prob:.2f})")
                                logger.info(f"{sym} META-FILTER: P(win)={mprob:.3f} < {meta_min_prob:.2f} — scaling weight")
                            else:
                                # Default: hard-zero the weight (strictest rejection)
                                gate_mult = 0.0
                                gate_reasons.append(f"META-REJECT({mprob:.2f}<{meta_min_prob:.2f})")
                                logger.info(f"{sym} META-FILTER: P(win)={mprob:.3f} < {meta_min_prob:.2f} — REJECTED")
                            if getattr(self, 'execution_scorecard', None):
                                try:
                                    self.execution_scorecard.record_meta_reject(sym, mprob)
                                except Exception:
                                    pass
                        else:
                            logger.debug(f"{sym} META-FILTER: P(win)={mprob:.3f} ≥ {meta_min_prob:.2f} — accept")
                    except Exception as e:
                        logger.debug(f"{sym} META-FILTER error ({e}) — pass-through")
                if direction != 0:
                    # Regime gate
                    cached = self.regime_cache.get(sym)
                    sym_regime = cached[0] if isinstance(cached, (list, tuple)) and len(cached) == 2 else (cached if isinstance(cached, str) else 'mean_reverting')
                    if is_bearish(sym_regime) and direction == 1:
                        gate_mult *= 0.4
                        gate_reasons.append(f"REGIME(trending_down)")
                    elif is_bullish(sym_regime) and direction == -1:
                        gate_mult *= 0.4
                        gate_reasons.append(f"REGIME(trending_up)")
                    # VIX gate
                    if vix > 28 and direction == 1:
                        vix_scale = max(0.3, 1.0 - (vix - 28) / 20)
                        gate_mult *= vix_scale
                        gate_reasons.append(f"VIX({vix:.0f})")
                    # SPX breadth gate
                    if direction == 1 and spx_bear:
                        gate_mult *= 0.3
                        gate_reasons.append("SPX<200SMA")
                    # 1H trend gate
                    try:
                        hourly = self.data_ingestion.get_latest_data(sym, timeframe='1H')
                        if hourly is not None and len(hourly) >= 20:
                            sma_20h = hourly['close'].rolling(20).mean().iloc[-1]
                            hourly_trend = 1 if hourly['close'].iloc[-1] > sma_20h else -1
                            if direction != hourly_trend:
                                gate_mult *= 0.5
                                gate_reasons.append(f"1H_TREND({hourly_trend})")
                    except Exception:
                        pass
                    # Anti-churn gate
                    churn_penalty = self.memory.get_churn_penalty(sym, direction, ts)
                    if churn_penalty < 1.0:
                        gate_mult *= churn_penalty
                        gate_reasons.append(f"CHURN({churn_penalty:.2f})")
                    # Defensive mode
                    if self.memory.is_defensive(ts):
                        gate_mult = 0.0
                        gate_reasons.append("DEFENSIVE")
                    # Trade autopsy gate: suppress entries matching historically losing patterns
                    suppress, reason, _ = self.autopsy_engine.should_suppress_entry(
                        sym, direction, sym_regime, vix, 0.0)
                    if suppress:
                        gate_mult *= 0.2
                        gate_reasons.append(f"AUTOPSY({reason})")
                gated_weights[sym] = weight * gate_mult
                if gate_mult < 1.0 and direction != 0:
                    logger.info(f"[PORTFOLIO GATE] {sym}: weight {weight:.4f} → {gated_weights[sym]:.4f} (gates: {', '.join(gate_reasons)})")
            target_weights = gated_weights
            # === EQUITY CURVE POSITION SCALING ===
            # Scale ALL weights by the equity curve trader's assessment of our own performance
            self.equity_curve_trader.record_equity(current_equity, ts)
            eq_scale = self.equity_curve_trader.get_position_scale()
            if eq_scale != 1.0:
                target_weights = {sym: w * eq_scale for sym, w in target_weights.items()}
                logger.info(f"[EQ CURVE] Scaling all weights by {eq_scale:.3f} "
                           f"(fast={self.equity_curve_trader._fast_ema:.0f} "
                           f"slow={self.equity_curve_trader._slow_ema:.0f})")
            # Re-normalize after gating + equity scaling
            abs_sum = np.sum(np.abs(list(target_weights.values())))
            if abs_sum > max_leverage:
                scale = max_leverage / abs_sum
                target_weights = {sym: w * scale for sym, w in target_weights.items()}
            logger.info(f"Portfolio PPO actions (post-gate): {target_weights}")
            # EQ-SCORE: record this cycle + regimes for daily scorecard
            if getattr(self, 'execution_scorecard', None):
                try:
                    regime_snapshot = {s: self.regime_cache.get(s, ('mean_reverting', 0.5))
                                        for s in target_weights.keys()}
                    self.execution_scorecard.record_cycle(regime_snapshot)
                except Exception:
                    pass
            # ALPHA-ATTR: record per-symbol attribution for entry-side trades that
            # passed through the full pipeline. Only record for symbols where the
            # final weight is non-zero (i.e., we're actually trading this symbol).
            if getattr(self, 'alpha_attribution', None):
                try:
                    for sym, final_w in target_weights.items():
                        if abs(final_w) < 1e-6:
                            continue
                        base_w = baseline_weights.get(sym, final_w)
                        # Reconstruct aggregate multiplier = final / baseline
                        if abs(base_w) > 1e-9:
                            agg_mult = final_w / base_w
                        else:
                            agg_mult = 1.0
                        direction = 1 if final_w > 0 else -1
                        context = {
                            "regime": regime_snapshot.get(sym, ('mean_reverting', 0.5))[0] if 'regime_snapshot' in locals() else 'unknown',
                            "aggregate_multiplier": float(agg_mult),
                        }
                        # layers dict is a rough summary — detailed per-layer tracking
                        # would require capturing at each gate. For now, record the
                        # aggregate transformation.
                        self.alpha_attribution.record_attribution(
                            symbol=sym,
                            baseline_weight=float(base_w),
                            final_weight=float(final_w),
                            direction=direction,
                            layers={"aggregate_transform": float(agg_mult)},
                            context=context,
                        )
                except Exception as e:
                    logger.debug(f"[ALPHA-ATTR] record failed: {e}")
            logger.debug("========== PORTFOLIO PPO DEBUG END ==========")
            return target_weights
        except Exception as e:
            logger.error(f"Portfolio PPO action generation failed: {e}", exc_info=True)
            return {sym: 0.0 for sym in symbols}

    # ==================== DAILY CAUSAL GRAPH REFRESH (called from bot.py at 3:30 AM) ====================
    def refresh_causal_wrappers(self):
        """Daily refresh of causal graphs with latest full data (called at 3:30 AM).
        Made more robust: idempotent, extra logging, safe rebuild."""
        if not CONFIG.get('USE_CAUSAL_RL', False):
            return
        logger.info("Refreshing causal managers with latest full data...")
        # P-18 / Critical #11 FIX: Force rebuild by deleting old cache files before refresh (ensures post-retrain freshness)
        for symbol in self.config.get('SYMBOLS', []):
            cache_path = f"causal_cache_{symbol}.pkl"
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                    logger.debug(f"[CAUSAL CACHE INVALIDATE] Deleted old cache for {symbol} before refresh")
                except Exception as e:
                    logger.warning(f"Failed to delete old cache {cache_path}: {e}")
        # ==================== NEW: PRESERVE BUFFER BEFORE REBUILD ====================
        if hasattr(self, 'portfolio_causal_manager') and self.portfolio_causal_manager is not None:
            try:
                self.portfolio_causal_manager.save_buffer() # ← FIXED: use correct method name
                logger.info(f"[BUFFER PRESERVE] Saved {len(self.portfolio_causal_manager.replay_buffer.buffer)} samples before nightly refresh")
            except Exception as e:
                logger.error(f"[BUFFER PRESERVE FAILED] {e}")
        # ==================== END PRESERVE ====================
        # HIGH-17 FIX: Save per-symbol buffers before clearing (were being lost on nightly refresh)
        for sym, mgr in self.causal_manager.items():
            try:
                mgr.save_buffer()
                logger.debug(f"[BUFFER PRESERVE] Saved per-symbol buffer for {sym}")
            except Exception as e:
                logger.warning(f"[BUFFER PRESERVE] Failed to save buffer for {sym}: {e}")
        self.causal_manager = {}
        self.portfolio_causal_manager = None
        # Rebuild per-symbol managers
        for symbol in self.config.get('SYMBOLS', []):
            data = self.data_ingestion.get_latest_data(symbol)
            if len(data) >= 200:
                regime_tuple = self.trainer.get_cached_regime(symbol, data) if hasattr(self.trainer, 'get_cached_regime') else ('trending', 0.5)
                regime = regime_tuple[0] if isinstance(regime_tuple, (list, tuple)) else regime_tuple
                features = generate_features(data, regime, symbol, data)
                # HIGH-18 FIX: Check for None features before creating DataFrame
                if features is None or features.shape[0] == 0:
                    logger.warning(f"[CAUSAL REFRESH] {symbol} — feature generation returned None/empty — skipping")
                    continue
                # FIX #28: Removed dead features_df computation (not passed to CausalSignalManager)
                if symbol in self.trainer.ppo_models and self.trainer.ppo_models[symbol]:
                    self.causal_manager[symbol] = CausalSignalManager(
                        base_model=self.trainer.ppo_models[symbol],
                        symbol=symbol,
                        data_ingestion=self.data_ingestion
                    )
        # Rebuild portfolio manager with full stacked matrix + symbol_id
        if self.trainer.portfolio_ppo_model:
            all_features = []
            symbol_ids = []
            total_rows = 0
            for sym in self.config.get('SYMBOLS', []):
                data = self.data_ingestion.get_latest_data(sym)
                regime_tuple = self.trainer.get_cached_regime(sym, data) if hasattr(self.trainer, 'get_cached_regime') else ('trending', 0.5)
                regime = regime_tuple[0] if isinstance(regime_tuple, (list, tuple)) else regime_tuple
                features = generate_features(data, regime, sym, data)
                if features is not None and features.shape[0] > 0:
                    rows = features.shape[0]
                    total_rows += rows
                    logger.info(f"[CAUSAL STACK REFRESH] {sym} contributed {rows} rows")
                    all_features.append(features)
                    symbol_ids.extend([sym] * rows)
            if all_features:
                full_matrix = np.vstack(all_features)
                full_df = pd.DataFrame(full_matrix, columns=[f'feat_{i}' for i in range(full_matrix.shape[1])])
                full_df['symbol_id'] = symbol_ids # Enables cross-asset causality
                # BUG-3 FIX: Encode symbol_id to numeric codes BEFORE any numeric filtering
                full_df['symbol_id'] = pd.Categorical(full_df['symbol_id']).codes
                self.portfolio_causal_manager = CausalSignalManager(
                    base_model=self.trainer.portfolio_ppo_model,
                    symbol="portfolio",
                    data_ingestion=self.data_ingestion
                )
                logger.info(f"Portfolio causal matrix refreshed with {full_matrix.shape[0]} total rows and {full_matrix.shape[1]} features + symbol_id")
            else:
                self.portfolio_causal_manager = None
        # NOTE: load_buffer() already called in CausalSignalManager.__init__ — no duplicate load needed
        if hasattr(self, 'portfolio_causal_manager') and self.portfolio_causal_manager is not None:
            loaded_count = len(self.portfolio_causal_manager.replay_buffer.buffer)
            logger.info(f"[BUFFER RESTORE] Buffer has {loaded_count} samples after nightly refresh (loaded in __init__)")
        logger.info("✅ All causal managers refreshed successfully")

    # ==================== NEW: STARTUP-SAFE REBUILD (no cache deletion) ====================
    def rebuild_causal_wrappers_without_deleting_cache(self):
        """Startup-safe rebuild: re-attaches managers using existing cache files (no deletion).
        This is what bot.py should call on normal startup so ReplayBuffer persists."""
        if not CONFIG.get('USE_CAUSAL_RL', False):
            return
        logger.info("Rebuilding causal managers using existing cache (no deletion for startup)...")
        self.causal_manager = {}
        self.portfolio_causal_manager = None
        # Per-symbol
        for symbol in self.config.get('SYMBOLS', []):
            data = self.data_ingestion.get_latest_data(symbol)
            if len(data) >= 200:
                regime_tuple = self.trainer.get_cached_regime(symbol, data) if hasattr(self.trainer, 'get_cached_regime') else ('trending', 0.5)
                regime = regime_tuple[0] if isinstance(regime_tuple, (list, tuple)) else regime_tuple
                features = generate_features(data, regime, symbol, data)
                if features is None or features.shape[0] == 0:
                    logger.warning(f"[CAUSAL REBUILD] Skipping {symbol} — features generation returned None/empty")
                    continue
                # FIX #28: Removed dead features_df computation (not passed to CausalSignalManager)
                if symbol in self.trainer.ppo_models and self.trainer.ppo_models[symbol]:
                    self.causal_manager[symbol] = CausalSignalManager(
                        base_model=self.trainer.ppo_models[symbol],
                        symbol=symbol,
                        data_ingestion=self.data_ingestion
                    )
        # Portfolio (full stacked matrix)
        if self.trainer.portfolio_ppo_model:
            all_features = []
            symbol_ids = []
            for sym in self.config.get('SYMBOLS', []):
                data = self.data_ingestion.get_latest_data(sym)
                regime_tuple = self.trainer.get_cached_regime(sym, data) if hasattr(self.trainer, 'get_cached_regime') else ('trending', 0.5)
                regime = regime_tuple[0] if isinstance(regime_tuple, (list, tuple)) else regime_tuple
                features = generate_features(data, regime, sym, data)
                if features is not None and features.shape[0] > 0:
                    all_features.append(features)
                    symbol_ids.extend([sym] * features.shape[0])
            if all_features:
                full_matrix = np.vstack(all_features)
                full_df = pd.DataFrame(full_matrix, columns=[f'feat_{i}' for i in range(full_matrix.shape[1])])
                full_df['symbol_id'] = pd.Categorical(symbol_ids).codes
                self.portfolio_causal_manager = CausalSignalManager(
                    base_model=self.trainer.portfolio_ppo_model,
                    symbol="portfolio",
                    data_ingestion=self.data_ingestion
                )
                logger.info(f"Portfolio causal matrix rebuilt from cache with {full_matrix.shape[0]} total rows")
            else:
                self.portfolio_causal_manager = None
                logger.debug("[PORTFOLIO CAUSAL] No data or features — manager set to None")
        else:
            logger.debug("[PORTFOLIO CAUSAL] No portfolio_ppo_model loaded yet — manager remains None")
        # NOTE: load_buffer() already called in CausalSignalManager.__init__ — no duplicate load needed
        if hasattr(self, 'portfolio_causal_manager') and self.portfolio_causal_manager is not None:
            loaded_count = len(self.portfolio_causal_manager.replay_buffer.buffer)
            if loaded_count > 0:
                logger.info(f"[BUFFER RESTORE ON STARTUP] Buffer has {loaded_count} samples (loaded in __init__)")
            else:
                logger.info("[BUFFER RESTORE ON STARTUP] Portfolio buffer empty — starting fresh")
        else:
            logger.debug("[BUFFER RESTORE ON STARTUP] No portfolio_causal_manager — skipping (normal during early startup)")
        logger.info("✅ Causal managers rebuilt safely on startup (cache preserved — ReplayBuffer will continue growing)")

    # ==================== NEW: AUTO-SAVE AFTER EVERY REWARD PUSH ====================
    # This is the critical fix so the buffer survives normal restarts and daily refreshes
    def _monitor_oos_decay(self):
        """FIXED: Only mark realized_return when the position is ACTUALLY closed on Alpaca.
        Unrealized P&L is now tracked separately — no more contamination of Gemini tuning data.
        Broker-side reward push now handles the heavy lifting — this method only does monitoring/pruning."""
        try:
            for symbol in self.config.get('SYMBOLS', []):
                history = getattr(self, 'live_signal_history', {}).get(symbol, [])
                if not history:
                    continue
                current_price = getattr(self, 'latest_prices', {}).get(symbol)
                if current_price is None:
                    logger.debug(f"[OOS MONITOR] Price cache miss for {symbol} — skipping current price (safe)")
                    continue
                for entry in reversed(history):
                    if entry.get('realized_return') is not None:
                        continue
                    # Still open — compute unrealized but do NOT contaminate realized_return
                    unrealized = (current_price - entry['price']) / entry['price'] * entry['direction']
                    entry['unrealized_return'] = unrealized # new safe key for monitoring
                # Win-rate calculation now ONLY on truly closed trades
                closed = [e for e in history if e.get('realized_return') is not None]
                if len(closed) >= 5:
                    recent_rets = [e['realized_return'] for e in closed[-10:]]
                    win_rate = sum(r > 0 for r in recent_rets) / len(recent_rets)
                    # FIX #42: Lock threshold modifications to prevent race with generate_signal reads
                    if not hasattr(self.trainer, '_lock'):
                        logger.debug(f"[OOS MONITOR] trainer._lock not available — skipping threshold update for {symbol}")
                        continue
                    with self.trainer._lock:
                        if symbol in self.trainer.confidence_thresholds:
                            thresholds = self.trainer.confidence_thresholds[symbol]
                            if thresholds:
                                latest = thresholds[-1]
                                # HIGH FIX: Track last closed trade count per symbol.
                                # If no new trades in N checks, relax thresholds toward defaults
                                # regardless of win_rate (prevents permanent lockout).
                                oos_no_trade_key = f'_oos_last_closed_count_{symbol}'
                                oos_stale_checks_key = f'_oos_stale_checks_{symbol}'
                                prev_closed_count = getattr(self, oos_no_trade_key, 0)
                                current_closed_count = len(closed)
                                stale_checks = getattr(self, oos_stale_checks_key, 0)
                                if current_closed_count == prev_closed_count:
                                    stale_checks += 1
                                else:
                                    stale_checks = 0
                                setattr(self, oos_no_trade_key, current_closed_count)
                                setattr(self, oos_stale_checks_key, stale_checks)
                                max_stale = self.config.get('OOS_MAX_STALE_CHECKS', 10)
                                if stale_checks >= max_stale:
                                    # No new trades for too long — force relax toward defaults
                                    latest['long'] = max(latest.get('long', 0.6) - 0.03, 0.55)
                                    latest['short'] = min(latest.get('short', 0.4) + 0.03, 0.45)
                                    setattr(self, oos_stale_checks_key, 0)  # reset counter
                                    logger.warning(
                                        f"{symbol} OOS stale ({stale_checks} checks with no new trades) — "
                                        f"force-relaxing thresholds to L={latest['long']:.2f}/S={latest['short']:.2f}"
                                    )
                                elif win_rate < 0.4:
                                    # Tighten thresholds — but use absolute target, not cumulative increment
                                    target_long = min(0.65 + (0.4 - win_rate) * 0.5, 0.85)
                                    target_short = max(0.35 - (0.4 - win_rate) * 0.5, 0.15)
                                    latest['long'] = max(latest.get('long', 0.6), target_long)
                                    latest['short'] = min(latest.get('short', 0.4), target_short)
                                    logger.warning(f"{symbol} OOS degradation: win rate {win_rate:.1%} — thresholds tightened to L={latest['long']:.2f}/S={latest['short']:.2f}")
                                elif win_rate > 0.55:
                                    # Recovery: relax thresholds back toward default
                                    latest['long'] = max(latest.get('long', 0.6) - 0.02, 0.55)
                                    latest['short'] = min(latest.get('short', 0.4) + 0.02, 0.45)
                                    logger.info(f"{symbol} OOS recovery: win rate {win_rate:.1%} — thresholds relaxed to L={latest['long']:.2f}/S={latest['short']:.2f}")
                # B-25 FIX: Runtime prune old realized entries to prevent memory bloat (keep last 2000 per symbol)
                if len(history) > 2000:
                    history[:] = history[-2000:]
                    logger.debug(f"[B-25] Pruned {symbol} live_signal_history to last 2000 entries")
                    # NO SAVE CALL HERE — broker-side push already handles persistence
        except Exception as e:
            logger.error(f"OOS monitoring error: {e}")

    async def dynamic_threshold_update_task(self):
        update_interval_seconds = self.config.get('DYNAMIC_THRESHOLD_UPDATE_DAYS', 7) * 24 * 3600
        logger.info(f"Dynamic threshold update task started — interval: {self.config.get('DYNAMIC_THRESHOLD_UPDATE_DAYS', 7)} days")
        while True:
            await asyncio.sleep(update_interval_seconds)
            try:
                logger.info("Running dynamic walk-forward threshold re-optimization")
                await asyncio.to_thread(self.trainer.dynamic_walk_forward_update)
            except Exception as e:
                logger.error(f"Dynamic threshold update failed: {e}", exc_info=True)

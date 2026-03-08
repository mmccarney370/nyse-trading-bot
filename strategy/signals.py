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
import os
os.environ["TQDM_DISABLE"] = "1"
os.environ["DISABLE_TQDM"] = "1"
os.environ["TQDM_DISABLE"] = "True"
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
from strategy.regime import detect_regime, get_regime_with_window # ← ADDED: rolling window helper
from models.portfolio_env import PortfolioEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
import torch # ← Required for torch.no_grad()
# NEW: Causal RL dependencies
from dowhy import CausalModel
import pgmpy.estimators.GES as GES
import networkx as nx
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

class SignalGenerator:
    def __init__(self, config, data_ingestion, trainer, regime_cache=None): # BUG #4 FIX: now accepts shared cache from bot.py
        self.config = config
        self.data_ingestion = data_ingestion
        self.trainer = trainer
        self.news_api = NewsApiClient(api_key=config['NEWS_API_KEY'])
        self.prev_signals = {}
        self.last_entry_time = {}
        self.meta_ema = {}
        self.sentiment_cache = {}
        self.lstm_states = {}
        self.last_portfolio_state = None
        self.is_recurrent = config.get('PPO_RECURRENT', True)
        # BUG #4 FIX: Shared regime cache with bot.py (4AM precompute + _get_all_regimes now syncs live to signal generation)
        self.regime_cache = regime_cache if regime_cache is not None else {}
        # NEW: Initialize live_signal_history (required for OOS monitoring after reward push moved to alpaca.py)
        self.live_signal_history = {}
        # NEW: Initialize latest_prices cache (used by OOS monitoring)
        self.latest_prices = {}
        try:
            from transformers import pipeline
            self.finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                truncation=True,
                max_length=512
            )
            logger.info("finBERT sentiment analyzer loaded successfully")
        except Exception as e:
            logger.warning(f"finBERT not available ({e}) — falling back to VADER")
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.finbert_pipeline = None
            self.vader_analyzer = SentimentIntensityAnalyzer()
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
            for symbol in config.get('SYMBOLS', []):
                data = self.data_ingestion.get_latest_data(symbol)
                if len(data) >= 200:
                    features = generate_features(data, 'trending', symbol, data)
                    features_df = pd.DataFrame(features, columns=[f'feat_{i}' for i in range(features.shape[1])])
                    if symbol in self.trainer.ppo_models and self.trainer.ppo_models[symbol]:
                        self.causal_manager[symbol] = CausalSignalManager(
                            base_model=self.trainer.ppo_models[symbol],
                            features_df=features_df,
                            symbol=symbol,
                            data_ingestion=self.data_ingestion
                        )
            if self.trainer.portfolio_ppo_model:
                # FULL STACKED FEATURE MATRIX + symbol_id (this gives thousands of rows → meaningful edges)
                all_features = []
                symbol_ids = []
                total_rows = 0
                for sym in config.get('SYMBOLS', []):
                    data = self.data_ingestion.get_latest_data(sym)
                    features = generate_features(data, 'trending', sym, data)
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
                        features_df=full_df,
                        symbol="portfolio",
                        data_ingestion=self.data_ingestion
                    )
                    logger.info(f"Portfolio causal matrix built with {full_matrix.shape[0]} total rows and {full_matrix.shape[1]} features + symbol_id")
                else:
                    self.portfolio_causal_manager = None
            else:
                self.portfolio_causal_manager = None
        # ROBUSTNESS: If portfolio model is already loaded (edge case / future load order), build immediately
        if CONFIG.get('USE_CAUSAL_RL', False) and self.trainer.portfolio_ppo_model is not None and self.portfolio_causal_manager is None:
            logger.info("Portfolio model already loaded at init time — building causal manager immediately using cache")
            self.rebuild_causal_wrappers_without_deleting_cache()

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
                if obs is None or not isinstance(obs, list) or len(obs) == 0:
                    continue
                try:
                    obs_array = np.array(obs, dtype=np.float32).reshape(1, -1)
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
                if obs is None or not isinstance(obs, list) or len(obs) == 0:
                    continue
                try:
                    obs_array = np.array(obs, dtype=np.float32).reshape(1, -1)
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
    async def explain_signal_breakdown(self, symbol: str, timestamp: pd.Timestamp, data: pd.DataFrame = None, precomputed: dict = None, live_mode: bool = True) -> dict:
        """Returns a clean dict showing exactly how the final confidence score was built.
        Call this anytime (bot, REPL, notebook) to verify the blend.
        P-16 / Critical #8 FIX: Accepts precomputed dict to avoid double inference when DEBUG_SIGNAL_BLEND=True.
        B-01 FIX: Added guard to prevent infinite recursion when DEBUG_SIGNAL_BLEND=True"""
        if data is None:
            data = self.data_ingestion.get_latest_data(symbol)
        if len(data) < 200:
            return {"error": "insufficient data"}
        regime, persistence = get_regime_with_window( # ← CHANGED: use rolling window
            symbol=symbol,
            data_ingestion=self.data_ingestion,
            lookback_short=CONFIG.get('REGIME_SHORT_LOOKBACK', 96),
            weight_short=CONFIG.get('REGIME_SHORT_WEIGHT', 0.6),
            cache=self.regime_cache
        )
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
            features = generate_features(data, regime, symbol, data)
            ppo_prob = 0.5
            ppo_strength = 0.5
            action = None
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
                ppo_prob = (action[0][0] + 1) / 2 if hasattr(action, '__getitem__') else (float(action) + 1) / 2
                ppo_strength = abs(action[0][0]) if hasattr(action, '__getitem__') else abs(float(action))
        # Stacking meta-prob
        if precomputed is None:
            stacking_probs = []
            for model in self.trainer.stacking_models.get(symbol, []):
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(latest_features)[0]
                    stacking_probs.append(float(prob[1] if prob.shape[0] > 1 else prob[0]))
                else:
                    prob = model.predict(latest_features)[0]
                    stacking_probs.append(float(prob))
            meta_prob = np.mean(stacking_probs) if stacking_probs else 0.5
        # Causal factor (renamed for clarity)
        if precomputed is None:
            causal_factor = 1.0
            causal_manager = self.causal_manager.get(symbol)
            if causal_manager and action is not None:
                try:
                    # C-3 FIX: Ensure graph exists + safe fallback
                    causal_manager._ensure_graph_exists()
                    if causal_manager.causal_graph is not None:
                        penalty_factor = causal_manager.compute_penalty_factor(obs, action)
                        penalized_action = action * penalty_factor
                        causal_factor = penalty_factor # ISSUE P10 FIX: direct multiplier (boost if >1.0, suppress if <1.0)
                        logger.debug(f"[CAUSAL] {symbol} penalty factor applied: {penalty_factor:.4f}")
                    else:
                        logger.debug(f"[CAUSAL FALLBACK] {symbol} graph not ready — neutral causal factor=1.0")
                except Exception as e:
                    logger.debug(f"[CAUSAL] Penalty computation failed for {symbol}: {e} — neutral factor")
        # Sentiment (always compute if not precomputed, but can be passed)
        if precomputed is None:
            sentiment_score = await self.get_sentiment_score(symbol, timestamp, live_mode=live_mode)
        # Full blend (exactly the same math you already use)
        combined_meta = (1 - self.config.get('SENTIMENT_WEIGHT', 0.2)) * meta_prob + self.config.get('SENTIMENT_WEIGHT', 0.2) * sentiment_score
        long_thresh, short_thresh = self.trainer.get_current_thresholds(symbol, timestamp)
        logger.debug(f"{symbol}: thresholds (long>{long_thresh:.3f}, short<{short_thresh:.3f})")
        if 0.50 < combined_meta < 0.68:
            confidence = 0.0
            direction = 0
        else:
            direction = 0
            confidence = 0.0
            if combined_meta > long_thresh:
                direction = 1
                confidence = (combined_meta - long_thresh) / (1.0 - long_thresh)
            elif combined_meta < short_thresh:
                direction = -1
                confidence = (short_thresh - combined_meta) / short_thresh
        if direction != 0:
            conviction_boost = 0.7 + 0.3 * persistence
            confidence = min(1.0, confidence * conviction_boost)
        # Breakout boost (same as in generate_signal)
        if direction != 0 and len(data) >= 22:
            roll = 20
            prev_high = data['high'].iloc[-roll-1:-1].max()
            prev_low = data['low'].iloc[-roll-1:-1].min()
            current_price = data['close'].iloc[-1]
            if (current_price > prev_high and direction == 1) or (current_price < prev_low and direction == -1):
                confidence = min(1.0, confidence * 1.4)
        # ==================== BUG #3 FIX ====================
        # causal_factor (previously misnamed penalty) is now applied directly as a multiplier.
        # Strong causal evidence boosts confidence; weak/no evidence leaves it neutral or slightly suppresses.
        if 'causal_factor' in locals() and causal_factor != 1.0:
            confidence = min(1.0, confidence * causal_factor)
            logger.debug(f"{symbol} causal factor applied: multiplier={causal_factor:.4f} → final confidence={confidence:.4f}")
        # ==================== END BUG #3 FIX ====================
        prev_direction = self.prev_signals.get(symbol, 0)
        if symbol in self.last_entry_time and prev_direction != 0:
            bars_since = (timestamp - self.last_entry_time[symbol]) / pd.Timedelta(minutes=15)
            # P-13 / Critical #5 FIX: Safe default if config key missing (prevents TypeError crash)
            if regime == 'trending':
                min_hold = self.config.get('MIN_HOLD_BARS_TRENDING', 48)
            else:
                min_hold = self.config.get('MIN_HOLD_BARS_MEAN_REVERTING', 24)
            if bars_since < min_hold:
                if direction == 0 or direction == -prev_direction:
                    direction = prev_direction
                    confidence = max(confidence, 0.6)
                    logger.debug(f"{symbol} min-hold enforced ({regime}): bars_held={int(bars_since)} < {min_hold} → keeping previous direction")
        if direction != 0 and prev_direction == 0:
            self.last_entry_time[symbol] = timestamp
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

    # ==================== MISSING METHOD ADDED HERE (fixes the crash) ====================
    async def get_sentiment_score(self, symbol: str, timestamp: pd.Timestamp = None, live_mode: bool = True) -> float:
        """Unified sentiment score using LocalLLMDebate (Ollama preferred) or fallback.
        BUG-13 FIX: Cache now refreshes hourly instead of daily — much more responsive to intraday news shifts."""
        if timestamp is None:
            timestamp = datetime.now(tz=tz.gettz('UTC'))
        # BUG-13 FIX: Use hour-level cache key (refreshes ~every hour during trading session)
        cache_key = f"{symbol}_{timestamp.date()}_{timestamp.hour}"
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
                if self.llm_debate is not None and hasattr(self.llm_debate, 'debate_sentiment'):
                    if asyncio.iscoroutinefunction(self.llm_debate.debate_sentiment):
                        score = await self.llm_debate.debate_sentiment(news_texts)
                    else:
                        score = self.llm_debate.debate_sentiment(news_texts)
                else:
                    score = 0.0
  
            # Cache result (now per hour)
            self.sentiment_cache[cache_key] = score
            logger.debug(f"[SENTIMENT COMPUTED] {symbol} for {cache_key}: {score:.3f}")
            return score
  
        except Exception as e:
            logger.debug(f"Sentiment score failed for {symbol}: {e}")
            score = 0.0
            self.sentiment_cache[cache_key] = score
            return score

    async def generate_portfolio_actions(self, symbols: list, data_dict: dict, current_equity: float, precomputed_env=None, timestamp: pd.Timestamp = None) -> dict:
        if not symbols:
            logger.warning("No symbols provided for portfolio actions")
            return {sym: 0.0 for sym in symbols}
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
                        step_idx = precomputed_env.timeline.get_loc(timestamp, method='ffill')
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
                    max_leverage=self.config.get('MAX_LEVERAGE', 3.0)
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
            except (AttributeError, TypeError, ValueError):
                # Fallback if causal wrapper fails
                if self.portfolio_causal_manager:
                    logger.warning("[CAUSAL FALLBACK] Portfolio causal predict failed — using base PPO")
                    action, _ = self.portfolio_causal_manager.predict(obs, deterministic=True)
                else:
                    action, _ = self.trainer.portfolio_ppo_model.predict(obs, deterministic=True)
            action = action.flatten()
            max_leverage = self.config.get('MAX_LEVERAGE', 3.0)
            abs_sum = np.sum(np.abs(action))
            if abs_sum > max_leverage:
                action = action / abs_sum * max_leverage
            action = np.clip(action, -2.0, 2.0)
            target_weights = {sym: float(weight) for sym, weight in zip(symbols, action)}
            sentiment_blend_weight = self.config.get('PORTFOLIO_SENTIMENT_WEIGHT', 0.3)
            if sentiment_blend_weight > 0:
                for sym in symbols:
                    ts = timestamp if timestamp is not None else data_dict[sym].index[-1]
                    sentiment = await self.get_sentiment_score(sym, ts, live_mode=True)
                    # ISSUE P9 / BUG-17 FIX: symmetric scaling for sentiment in [-1, 1] range
                    # +1 → boost (1 + weight), -1 → suppress (1 - weight), 0 → neutral (1.0)
                    sentiment_factor = 1.0 + sentiment_blend_weight * sentiment
                    old_weight = target_weights[sym]
                    target_weights[sym] *= sentiment_factor
                    logger.info(f"{sym} sentiment blend: raw={sentiment:.3f} → factor={sentiment_factor:.2f} (weight {old_weight:.3f} → {target_weights[sym]:.3f})")
                abs_sum = np.sum(np.abs(list(target_weights.values())))
                if abs_sum > max_leverage:
                    scale = max_leverage / abs_sum
                    target_weights = {sym: w * scale for sym, w in target_weights.items()}
                    logger.info(f"Re-normalized weights after sentiment blend (scale {scale:.3f})")
            logger.info(f"Portfolio PPO actions (sentiment blended): {target_weights}")
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
        self.causal_manager = {}
        self.portfolio_causal_manager = None
        # Rebuild per-symbol managers
        for symbol in self.config.get('SYMBOLS', []):
            data = self.data_ingestion.get_latest_data(symbol)
            if len(data) >= 200:
                features = generate_features(data, 'trending', symbol, data)
                features_df = pd.DataFrame(features, columns=[f'feat_{i}' for i in range(features.shape[1])])
                if symbol in self.trainer.ppo_models and self.trainer.ppo_models[symbol]:
                    self.causal_manager[symbol] = CausalSignalManager(
                        base_model=self.trainer.ppo_models[symbol],
                        features_df=features_df,
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
                features = generate_features(data, 'trending', sym, data)
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
                    features_df=full_df,
                    symbol="portfolio",
                    data_ingestion=self.data_ingestion
                )
                logger.info(f"Portfolio causal matrix refreshed with {full_matrix.shape[0]} total rows and {full_matrix.shape[1]} features + symbol_id")
            else:
                self.portfolio_causal_manager = None
        # ==================== NEW: RESTORE BUFFER AFTER REBUILD ====================
        if hasattr(self, 'portfolio_causal_manager') and self.portfolio_causal_manager is not None:
            try:
                self.portfolio_causal_manager.load_buffer() # ← FIXED: use correct method name
                loaded_count = len(self.portfolio_causal_manager.replay_buffer.buffer)
                logger.info(f"[BUFFER RESTORE] Loaded {loaded_count} samples after nightly refresh")
            except Exception as e:
                logger.error(f"[BUFFER RESTORE FAILED] {e}")
        # ==================== END RESTORE ====================
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
                features = generate_features(data, 'trending', symbol, data)
                features_df = pd.DataFrame(features, columns=[f'feat_{i}' for i in range(features.shape[1])])
                if symbol in self.trainer.ppo_models and self.trainer.ppo_models[symbol]:
                    self.causal_manager[symbol] = CausalSignalManager(
                        base_model=self.trainer.ppo_models[symbol],
                        features_df=features_df,
                        symbol=symbol,
                        data_ingestion=self.data_ingestion
                    )
        # Portfolio (full stacked matrix)
        if self.trainer.portfolio_ppo_model:
            all_features = []
            symbol_ids = []
            for sym in self.config.get('SYMBOLS', []):
                data = self.data_ingestion.get_latest_data(sym)
                features = generate_features(data, 'trending', sym, data)
                if features is not None and features.shape[0] > 0:
                    all_features.append(features)
                    symbol_ids.extend([sym] * features.shape[0])
            if all_features:
                full_matrix = np.vstack(all_features)
                full_df = pd.DataFrame(full_matrix, columns=[f'feat_{i}' for i in range(full_matrix.shape[1])])
                full_df['symbol_id'] = pd.Categorical(symbol_ids).codes
                self.portfolio_causal_manager = CausalSignalManager(
                    base_model=self.trainer.portfolio_ppo_model,
                    features_df=full_df,
                    symbol="portfolio",
                    data_ingestion=self.data_ingestion
                )
                logger.info(f"Portfolio causal matrix rebuilt from cache with {full_matrix.shape[0]} total rows")
            else:
                self.portfolio_causal_manager = None
                logger.debug("[PORTFOLIO CAUSAL] No data or features — manager set to None")
        else:
            logger.debug("[PORTFOLIO CAUSAL] No portfolio_ppo_model loaded yet — manager remains None")
        # ==================== LOAD BUFFER ON STARTUP ====================
        if hasattr(self, 'portfolio_causal_manager') and self.portfolio_causal_manager is not None:
            try:
                self.portfolio_causal_manager.load_buffer() # ← FIXED: use correct method name
                loaded_count = len(self.portfolio_causal_manager.replay_buffer.buffer)
                if loaded_count > 0:
                    logger.info(f"[BUFFER RESTORE ON STARTUP] Successfully loaded {loaded_count} samples")
                else:
                    logger.info("[BUFFER RESTORE ON STARTUP] Portfolio buffer file exists but was empty — starting fresh")
            except Exception as e:
                logger.warning(f"[BUFFER RESTORE ON STARTUP] Failed to load: {e} — starting with empty buffer this run")
        else:
            logger.debug("[BUFFER RESTORE ON STARTUP] No portfolio_causal_manager — skipping load (normal during early startup)")
        # ==================== END LOAD ====================
        logger.info("✅ Causal managers rebuilt safely on startup (cache preserved — ReplayBuffer will continue growing)")

    # ==================== NEW: AUTO-SAVE AFTER EVERY REWARD PUSH ====================
    # This is the critical fix so the buffer survives normal restarts and daily refreshes
    def _monitor_oos_decay(self):
        """FIXED: Only mark realized_return when the position is ACTUALLY closed on Alpaca.
        Unrealized P&L is now tracked separately — no more contamination of Gemini tuning data.
        Broker-side reward push now handles the heavy lifting — this method only does monitoring/pruning."""
        try:
            for symbol in self.config['SYMBOLS']:
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
                    if win_rate < 0.4:
                        logger.warning(f"{symbol} OOS degradation detected: recent win rate {win_rate:.1%} — capping conviction")
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
                self.trainer.dynamic_walk_forward_update()
            except Exception as e:
                logger.error(f"Dynamic threshold update failed: {e}", exc_info=True)

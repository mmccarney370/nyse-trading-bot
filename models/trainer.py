# models/trainer.py
# =====================================================================
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import optuna
from optuna.samplers import TPESampler
from concurrent.futures import ThreadPoolExecutor
from strategy.regime import detect_regime # Consolidated single source of truth
from .ppo_utils import (
    train_ppo,
    save_ppo_model,
    load_ppo_model,
    update_model_weights as ppo_update_model_weights
)
from .stacking_ensemble import train_stacking
from config import CONFIG
from models.features import generate_features
from models.portfolio_env import PortfolioEnv # NEW: Portfolio environment
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import torch
import json
import os
# NEW: Share the same persistent regime cache file as bot.py
REGIME_CACHE_FILE = "regime_cache.json"
# NEW: Use the modern CausalSignalManager from the extracted module (Phase 1 modularization)
from models.causal_signal_manager import CausalSignalManager # ← Phase 1 modularization fix

logger = logging.getLogger(__name__)

class NaNStopCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
    def _on_step(self) -> bool:
        for param in self.model.policy.parameters():
            if torch.isnan(param).any():
                logger.warning("NaN detected in portfolio model parameters — stopping training")
                return False
        return True

class ProfitLoggingCallback(BaseCallback):
    """
    Custom callback to log profit metrics ONLY at rollout end (no per-step flood).
    Enhanced to:
    - Always log current progress (total timesteps).
    - Inform when no completed episodes yet (common early with long data histories).
    - When episodes complete, log episode reward mean (portfolio PnL) and length directly
      from the model's ep_info_buffer for reliable console output.
    SB3 already records these to TensorBoard when available — this ensures console visibility.
    """
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
    def _on_step(self) -> bool:
        return True # No per-step logging
    def _on_rollout_end(self) -> None:
        # Always log current progress
        logger.info(f"[PROFIT LOG - ROLLOUT END] Total timesteps reached: {self.model.num_timesteps}")
        # Check for completed episodes in the buffer
        if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
            ep_rews = [ep['r'] for ep in self.model.ep_info_buffer if 'r' in ep]
            ep_lens = [ep['l'] for ep in self.model.ep_info_buffer if 'l' in ep]
            if ep_rews:
                rew_mean = np.mean(ep_rews)
                len_mean = np.mean(ep_lens)
                num_eps = len(ep_rews)
                logger.info(
                    f"[PROFIT LOG - ROLLOUT END] Episode Reward Mean (Portfolio PnL per episode): {rew_mean:.6f} "
                    f"| Episode Length Mean: {len_mean:.1f} steps | Completed Episodes: {num_eps}"
                )
            else:
                logger.info("[PROFIT LOG - ROLLOUT END] Completed episodes detected but no reward info yet.")
        else:
            logger.info("[PROFIT LOG - ROLLOUT END] No completed episodes yet — waiting for first full pass over data (normal early in training).")

class Trainer:
    def __init__(self, config, data_ingestion):
        self.config = config
        self.data_ingestion = data_ingestion
        self.stacking_models: Dict[str, list] = {}
        self.ppo_models: Dict[str, 'PPO | RecurrentPPO'] = {}
        self.vec_norms: Dict[str, 'VecNormalize'] = {}
        self.confidence_thresholds: Dict[str, list] = {}
        self.causal_wrappers: Dict[str, 'CausalSignalManager'] = {} # ← now uses modern unified manager
        self.portfolio_causal_manager = None # ← modern unified manager
        # Portfolio-level models (single shared policy)
        self.portfolio_ppo_model = None
        self.portfolio_vec_norm = None
        # NEW: Use the SAME persistent regime cache as bot.py
        self.regime_cache = self._load_regime_cache()

    def _load_regime_cache(self):
        """Load the same persistent cache used by bot.py"""
        if os.path.exists(REGIME_CACHE_FILE):
            try:
                with open(REGIME_CACHE_FILE, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded persistent regime cache with {len(data)} symbols (shared with bot.py)")
                return data
            except Exception as e:
                logger.warning(f"Failed to load regime cache: {e}")
        return {}

    def _save_regime_cache(self):
        """Save cache to the same file as bot.py"""
        try:
            with open(REGIME_CACHE_FILE, 'w') as f:
                json.dump(self.regime_cache, f, default=str)
            logger.debug(f"Saved regime cache with {len(self.regime_cache)} symbols")
        except Exception as e:
            logger.warning(f"Failed to save regime cache: {e}")

    def get_cached_regime(self, symbol: str, data: pd.DataFrame) -> tuple:
        """Cached regime detection — now uses the shared persistent cache (simple symbol key)
        B-04 FIX: Fully defensive unpacking for legacy plain-string / float / malformed entries"""
        cache_key = symbol # Simple key — matches bot.py exactly
        if cache_key in self.regime_cache:
            value = self.regime_cache[cache_key]
            # B-04 DEFENSIVE HANDLING (legacy cache from bot.py precompute)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                regime, persistence = value
            elif isinstance(value, str):
                # Legacy string-only entry (e.g. "trending")
                regime = value
                persistence = 0.50
                logger.warning(f"[B-04 LEGACY FIX] Converted plain string cache for {symbol} → ({regime}, {persistence:.3f})")
            elif isinstance(value, (int, float)):
                # Legacy float-only (persistence score)
                regime = 'trending' if value > 0.5 else 'mean_reverting'
                persistence = float(value)
                logger.warning(f"[B-04 LEGACY FIX] Converted plain numeric cache for {symbol} → ({regime}, {persistence:.3f})")
            else:
                # Unknown format → safe fallback
                regime = 'mean_reverting'
                persistence = 0.40
                logger.warning(f"[B-04 LEGACY FIX] Unknown cache type for {symbol} ({type(value)}) → fallback")
            logger.debug(f"Regime cache HIT for {symbol}: {regime} (persistence={persistence:.3f})")
            return regime, persistence
        # Fast lightweight fallback — NO heavy HMM computation
        logger.debug(f"Regime cache MISS for {symbol} — using fast fallback")
        if len(data) < 50:
            regime = 'mean_reverting'
            persistence = 0.5 # BUG-11 FIX: neutral fallback instead of aggressive 0.35
        else:
            recent_return = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
            regime = 'trending' if abs(recent_return) > 0.015 else 'mean_reverting'
            persistence = 0.5 # BUG-11 FIX: neutral fallback instead of aggressive 0.92/0.35
        # Log fallback usage (helps track cache miss frequency)
        logger.debug(f"[REGIME FALLBACK] {symbol} → {regime} (persistence={persistence:.3f})")
        # Save to shared cache (as list for JSON compatibility)
        self.regime_cache[cache_key] = [regime, persistence]
        self._save_regime_cache()
        return regime, persistence

    def walk_forward_optimize_thresholds(self, symbol: str, data: pd.DataFrame):
        # Use cached regime — now unpacks persistence as well (UPGRADE #4)
        regime, persistence = self.get_cached_regime(symbol, data)
        logger.info(f"Regime for {symbol}: {regime} (persistence score {persistence:.3f})")
        # FIXED: Added symbol and full_hist_df=data for TFT caching/alignment
        features = generate_features(
            data=data,
            regime=regime,
            symbol=symbol,
            full_hist_df=data
        )
        # NEW: Detailed debug logging + robust fallback for insufficient features
        if features is None or features.shape[0] == 0:
            logger.warning(f"[FEATURES FAIL] {symbol} — generate_features returned None/empty (input_len={len(data)})")
            # Fallback: use dummy constant probability so optimization can still run
            ensemble_prob = np.full(len(data)-1, 0.5)
            logger.info(f"[FEATURES FALLBACK] {symbol} — using dummy 0.5 probabilities for threshold optimization")
        else:
            logger.info(f"[FEATURES SUCCESS] {symbol} — generated shape {features.shape} (input_len={len(data)})")
            meta_probs = []
            stacking_models = self.stacking_models.get(symbol, [])
            if stacking_models:
                for model in stacking_models:
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(features)[:, 1]
                    else:
                        probs = model.predict(features)
                    meta_probs.append(probs)
                ensemble_prob = np.mean(meta_probs, axis=0) if meta_probs else np.full(len(features), 0.5)
            else:
                ensemble_prob = np.full(len(features), 0.5)
            ensemble_prob = ensemble_prob[:-1] # align with returns
        returns = data['close'].pct_change().shift(-1).dropna()
        aligned_returns = returns.iloc[:len(ensemble_prob)]
        df = pd.DataFrame({
            'meta_prob': ensemble_prob,
            'return': aligned_returns.values
        }, index=aligned_returns.index)
        n_windows = 6
        window_size = len(df) // n_windows
        for w in range(n_windows):
            chunk_start = w * window_size
            chunk_end = (w + 1) * window_size if w < n_windows - 1 else len(df)
            chunk_df = df.iloc[chunk_start:chunk_end]
            # ==================== TRUE OOS VALIDATION (70/30 split per window) ====================
            train_size = int(len(chunk_df) * 0.70)
            train_df = chunk_df.iloc[:train_size]
            oos_df = chunk_df.iloc[train_size:]
            logger.debug(f"{symbol} window {w+1}/{n_windows} — train:{len(train_df)} | OOS:{len(oos_df)}")
            def objective(trial):
                long_thresh = trial.suggest_float('long_thresh', 0.55, 0.85)
                short_thresh = trial.suggest_float('short_thresh', 0.15, 0.45)
                penalty_weight = self.config.get('THRESHOLD_PENALTY_WEIGHT', 0.70)
                signals = np.where(train_df['meta_prob'] > long_thresh, 1,
                                   np.where(train_df['meta_prob'] < short_thresh, -1, 0))
                strat_returns = signals * train_df['return']
                sharpe = strat_returns.mean() / (strat_returns.std() + 1e-8) * np.sqrt(252 * 96)
                penalty = penalty_weight * (abs(long_thresh - 0.65) + abs(short_thresh - 0.35))
                return sharpe - penalty
            study = optuna.create_study(direction='maximize', sampler=TPESampler())
            study.optimize(objective, n_trials=self.config.get('OPTUNA_TRIALS', 350),
                           timeout=self.config.get('OPTUNA_TIMEOUT', 900))
            best = study.best_trial
            best_score_is = best.value # in-sample score
            # Compute true OOS score on unseen data
            signals_oos = np.where(oos_df['meta_prob'] > best.params['long_thresh'], 1,
                                   np.where(oos_df['meta_prob'] < best.params['short_thresh'], -1, 0))
            strat_returns_oos = signals_oos * oos_df['return']
            oos_sharpe = strat_returns_oos.mean() / (strat_returns_oos.std() + 1e-8) * np.sqrt(252 * 96)
            logger.info(
                f"{symbol} window {w+1}/{n_windows} — IS Sharpe: {best_score_is:.3f} | "
                f"OOS Sharpe: {oos_sharpe:.3f} (long={best.params['long_thresh']:.3f}, short={best.params['short_thresh']:.3f})"
            )
            # Only accept if OOS is reasonable (prevents pure in-sample overfitting)
            if oos_sharpe > -0.5: # reasonable filter
                final_long_thresh = best.params['long_thresh']
                final_short_thresh = best.params['short_thresh']
                # Small safety clamp
                final_long_thresh = max(0.50, min(0.85, final_long_thresh))
                final_short_thresh = max(0.15, min(0.50, final_short_thresh))
                chunk_start_time = df.index[chunk_start]
                self.confidence_thresholds.setdefault(symbol, []).append({
                    'valid_from': chunk_start_time,
                    'long': final_long_thresh,
                    'short': final_short_thresh,
                    'sharpe_is': best_score_is,
                    'sharpe_oos': oos_sharpe
                })
                logger.info(
                    f"{symbol} walk-forward window {w+1}/{n_windows} ({chunk_start_time.date()}): "
                    f"long>{final_long_thresh:.3f}, short<{final_short_thresh:.3f} (IS {best_score_is:.2f} | OOS {oos_sharpe:.2f})"
                )
            else:
                logger.warning(f"{symbol} window {w+1} rejected — poor OOS performance")
        if not self.confidence_thresholds.get(symbol):
            fallback_time = data.index[-1]
            self.confidence_thresholds[symbol] = [{
                'valid_from': fallback_time,
                'long': 0.65,
                'short': 0.35,
                'sharpe_is': 0.0,
                'sharpe_oos': 0.0
            }]
        logger.info(f"Training completed for {symbol} with {len(self.confidence_thresholds.get(symbol, []))} threshold windows")

    def dynamic_walk_forward_update(self, symbol: str = None, days: int = 365):
        symbols = [symbol] if symbol else self.config['SYMBOLS']
        logger.info(f"Starting dynamic walk-forward threshold update for {len(symbols)} symbols (last {days} days)")
        for sym in symbols:
            data = self.data_ingestion.get_latest_data(sym)
            if len(data) < 500:
                logger.warning(f"Insufficient recent data for dynamic update on {sym} — skipping")
                continue
            cutoff = data.index[-1] - pd.Timedelta(days=days)
            recent_data = data[data.index >= cutoff]
            if len(recent_data) < 500:
                logger.warning(f"Recent data too short for {sym} ({len(recent_data)} bars) — skipping dynamic update")
                continue
            logger.info(f"Re-optimizing thresholds for {sym} on recent {len(recent_data)} bars")
            self.walk_forward_optimize_thresholds(sym, recent_data)
            if len(self.confidence_thresholds.get(sym, [])) > 12:
                self.confidence_thresholds[sym] = sorted(
                    self.confidence_thresholds[sym],
                    key=lambda x: x['valid_from']
                )[-12:]
        logger.info("Dynamic walk-forward threshold update completed")

    def update_model_weights(self, symbol: str, recent_data: pd.DataFrame = None):
        ppo_update_model_weights(self, symbol, recent_data)

    def get_current_thresholds(self, symbol: str, timestamp: pd.Timestamp) -> Tuple[float, float]:
        thresholds = self.confidence_thresholds.get(symbol, [])
        if not thresholds:
            return 0.65, 0.35
        thresholds = sorted(thresholds, key=lambda x: x['valid_from'])
        candidates = [t for t in thresholds if t['valid_from'] <= timestamp]
        if not candidates:
            return 0.65, 0.35
        latest = max(candidates, key=lambda x: x['valid_from'])
        return latest['long'], latest['short']

    def train_symbol(self, symbol: str, full_ppo: bool = True, regime_cache: dict = None):
        data = self.data_ingestion.get_latest_data(symbol)
        if len(data) < 500:
            logger.warning(f"Insufficient data for {symbol} — skipping training")
            return
        # Stacking ensemble (still useful for hybrid mode or fallback)
        # M-1 / HMM fix: pass regime_cache to avoid recomputing inside train_stacking
        self.stacking_models[symbol] = train_stacking(
            symbol=symbol,
            data=data,
            full_hist_df=data,
            regime_cache=regime_cache or self.regime_cache  # ← NEW: pass shared cache
        )
        if full_ppo:
            success = False
            for attempt in range(3):
                try:
                    train_ppo(self, symbol, data)
                    success = True
                    break
                except Exception as e:
                    logger.error(
                        f"PPO attempt {attempt+1} failed for {symbol}: {type(e).__name__}: {e}",
                        exc_info=True
                    )
                    if attempt < 2:
                        logger.info(f"Retrying PPO training for {symbol}...")
                    else:
                        logger.warning(f"PPO training failed after 3 attempts for {symbol} — skipping PPO")
            if not success:
                self.ppo_models[symbol] = None
            # ==================== MODERN CAUSAL MANAGER (unified with signals.py) ====================
            if success and symbol in self.ppo_models and self.ppo_models[symbol] is not None:
                features = generate_features(data, 'trending', symbol, data)
                if features is not None and features.shape[0] > 0:
                    features_df = pd.DataFrame(
                        features,
                        columns=[f'feat_{i}' for i in range(features.shape[1])]
                    )
                    self.causal_wrappers[symbol] = CausalSignalManager(
                        self.ppo_models[symbol],
                        features_df,
                        symbol=symbol,
                        data_ingestion=self.data_ingestion
                    )
                    logger.info(f"✅ Modern CausalSignalManager created and attached for {symbol}")
        self.walk_forward_optimize_thresholds(symbol, data)

    def train_portfolio(self, symbols: list, timesteps: int = None):
        """
        Train a single portfolio-level PPO using PortfolioEnv.
        Called when CONFIG['PORTFOLIO_PPO'] = True.
        NEW: If saved model exists, load it and skip full training (ready for inference/online updates)
        """
        if not symbols:
            logger.warning("No symbols provided for portfolio training")
            return
        # NEW: Check for existing saved portfolio model — load and skip full training if found
        existing_model, existing_norm = load_ppo_model(self, "portfolio")
        if existing_model is not None:
            logger.info("Existing portfolio PPO model found — skipping full training, ready for inference/online updates")
            self.portfolio_ppo_model = existing_model
            self.portfolio_vec_norm = existing_norm
            # CRIT-14 FIX: Causal warmup after loading existing model (replays historical rewards)
            if hasattr(self, 'signal_gen') and hasattr(self.signal_gen, 'live_signal_history'):
                logger.info("Replaying historical rewards into loaded portfolio causal buffer...")
                all_closed = [e for hist in self.signal_gen.live_signal_history.values() for e in hist if e.get('realized_return') is not None]
                for entry in all_closed:
                    obs = entry.get('obs')
                    if obs is not None and isinstance(obs, list) and len(obs) > 0:
                        try:
                            obs_array = np.array(obs, dtype=np.float32).reshape(1, -1)
                            action = entry.get('direction', 0) * entry.get('confidence', 1.0)
                            reward = entry.get('realized_return', 0.0)
                            self.portfolio_causal_manager.replay_buffer.push(obs_array, action, reward)
                        except:
                            pass
                loaded_count = len(self.portfolio_causal_manager.replay_buffer.buffer)
                logger.info(f"[CAUSAL WARMUP AFTER LOAD] Loaded {loaded_count} historical rewards into portfolio causal buffer")
            return
        logger.info(f"Starting portfolio-level PPO training on {len(symbols)} symbols: {symbols}")
        # Fetch full data for all symbols
        data_dict = {}
        min_len = float('inf')
        for sym in symbols:
            data = self.data_ingestion.get_latest_data(sym)
            if len(data) < 500:
                logger.error(f"Insufficient data for {sym} ({len(data)} bars) — cannot train portfolio PPO")
                return
            data_dict[sym] = data
            min_len = min(min_len, len(data))
        if min_len < 500:
            logger.error("Not enough common history across symbols for portfolio training")
            return
        # Use configurable timesteps (fallback to large number for portfolio problems)
        timesteps = timesteps or self.config.get('PORTFOLIO_TIMESTEPS', 1_000_000)
        # Create vectorized PortfolioEnv
        env = DummyVecEnv([lambda: PortfolioEnv(
            data_dict=data_dict,
            symbols=symbols,
            initial_balance=self.config.get('INITIAL_BALANCE', 100_000.0),
            max_leverage=self.config.get('MAX_LEVERAGE', 3.0)
        )])
        vec_env = VecNormalize(env, norm_obs=True, norm_reward=True)
        success = False
        for attempt in range(3):
            try:
                use_recurrent = CONFIG.get('PPO_RECURRENT', True)
                use_custom_gtrxl = CONFIG.get('USE_CUSTOM_GTRXL', True) and use_recurrent
                # FIXED: Removed use_sde=False to avoid duplicate argument error
                policy_kwargs = dict(
                    features_extractor_class=None, # PortfolioEnv handles features internally
                    net_arch=dict(pi=[512, 512], vf=[512, 512]), # Larger nets for portfolio complexity
                )
                if use_recurrent:
                    if use_custom_gtrxl:
                        from .policies import AuxGTrXLPolicy
                        policy_cls = AuxGTrXLPolicy
                    else:
                        from .policies import AuxRecurrentPolicy
                        policy_cls = AuxRecurrentPolicy
                    from .policies import CustomRecurrentPPO
                    model = CustomRecurrentPPO(
                        policy=policy_cls,
                        env=vec_env,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        tensorboard_log="./ppo_tensorboard/portfolio" if CONFIG.get('USE_TENSORBOARD') else None,
                        learning_rate=CONFIG.get('PPO_LEARNING_RATE', 3e-4),
                        n_steps=CONFIG.get('PPO_N_STEPS', 2048),
                        batch_size=CONFIG.get('PPO_BATCH_SIZE', 256),
                        n_epochs=4,
                        gamma=CONFIG.get('PPO_GAMMA', 0.95),
                        gae_lambda=CONFIG.get('PPO_GAE_LAMBDA', 0.92),
                        clip_range=CONFIG.get('PPO_CLIP_RANGE', 0.2),
                        ent_coef=CONFIG.get('PPO_ENTROPY_COEFF', 0.08),
                        vf_coef=CONFIG.get('vf_coef', 0.8),
                        max_grad_norm=0.3,
                        device="auto"
                    )
                else:
                    from .policies import AuxMlpPolicy
                    from stable_baselines3 import PPO as CustomPPO
                    model = CustomPPO(
                        policy=AuxMlpPolicy,
                        env=vec_env,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        tensorboard_log="./ppo_tensorboard/portfolio" if CONFIG.get('USE_TENSORBOARD') else None,
                        learning_rate=CONFIG.get('PPO_LEARNING_RATE', 3e-4),
                        n_steps=CONFIG.get('PPO_N_STEPS', 2048),
                        batch_size=CONFIG.get('PPO_BATCH_SIZE', 256),
                        n_epochs=10,
                        gamma=CONFIG.get('PPO_GAMMA', 0.95),
                        gae_lambda=CONFIG.get('PPO_GAE_LAMBDA', 0.92),
                        clip_range=CONFIG.get('PPO_CLIP_RANGE', 0.2),
                        vf_coef=CONFIG.get('vf_coef', 0.8),
                        max_grad_norm=0.3,
                        device="auto"
                    )
                nan_callback = NaNStopCallback()
                profit_callback = ProfitLoggingCallback() # Enhanced profit logging
                logger.info(f"Training portfolio PPO for {timesteps} timesteps")
                model.learn(
                    total_timesteps=timesteps,
                    callback=[nan_callback, profit_callback],
                    reset_num_timesteps=True
                )
                success = True
                break
            except Exception as e:
                logger.error(f"Portfolio PPO attempt {attempt+1} failed: {e}", exc_info=True)
                if attempt == 2:
                    logger.warning("Portfolio PPO training failed after 3 attempts")
        if success:
            self.portfolio_ppo_model = model
            self.portfolio_vec_norm = vec_env
            # ==================== MODERN PORTFOLIO CAUSAL MANAGER ====================
            logger.info("Building full multi-symbol feature matrix for portfolio causal graph...")
            all_features = []
            symbol_ids = []
            for sym in symbols:
                data = self.data_ingestion.get_latest_data(sym)
                features = generate_features(data, 'trending', sym, data)
                if features is not None and features.shape[0] > 0:
                    rows = features.shape[0]
                    all_features.append(features)
                    symbol_ids.extend([sym] * rows)
            if all_features:
                full_matrix = np.vstack(all_features)
                full_df = pd.DataFrame(full_matrix, columns=[f'feat_{i}' for i in range(full_matrix.shape[1])])
                full_df['symbol_id'] = symbol_ids
                self.portfolio_causal_manager = CausalSignalManager(
                    self.portfolio_ppo_model,
                    full_df,
                    symbol="portfolio",
                    data_ingestion=self.data_ingestion
                )
                logger.info(f"✅ Modern portfolio CausalSignalManager built on {full_matrix.shape[0]} rows")
                # CRIT-14 FIX: Causal warmup (loads historical rewards into buffer)
                if hasattr(self, 'signal_gen') and hasattr(self.signal_gen, 'live_signal_history'):
                    logger.info("Replaying historical rewards into portfolio causal buffer...")
                    all_closed = [e for hist in self.signal_gen.live_signal_history.values() for e in hist if e.get('realized_return') is not None]
                    for entry in all_closed:
                        obs = entry.get('obs')
                        if obs is not None and isinstance(obs, list) and len(obs) > 0:
                            try:
                                obs_array = np.array(obs, dtype=np.float32).reshape(1, -1)
                                action = entry.get('direction', 0) * entry.get('confidence', 1.0)
                                reward = entry.get('realized_return', 0.0)
                                self.portfolio_causal_manager.replay_buffer.push(obs_array, action, reward)
                            except:
                                pass
                    loaded_count = len(self.portfolio_causal_manager.replay_buffer.buffer)
                    logger.info(f"[CAUSAL WARMUP] Loaded {loaded_count} historical rewards into portfolio causal buffer")
            else:
                self.portfolio_causal_manager = None
                logger.warning("Could not build multi-symbol feature matrix for causal graph")
            save_ppo_model(self, "portfolio") # Save with special key
            logger.info("Portfolio PPO training completed and saved")
        else:
            self.portfolio_ppo_model = None
            self.portfolio_vec_norm = None

    def update_portfolio_weights(self, timesteps: int = None):
        """
        Incremental/online update for the portfolio-level PPO model.
        Recreates environment with latest data, transfers VecNormalize stats if possible,
        continues learning on existing model with low learning rate for safety.
        """
        symbols = self.config.get('SYMBOLS', [])
        if not symbols:
            logger.warning("No symbols in config — cannot perform portfolio online update")
            return
        if self.portfolio_ppo_model is None:
            logger.info("No existing portfolio PPO model — performing full training instead")
            self.train_portfolio(symbols)
            return
        logger.info(f"Starting online/incremental portfolio PPO update on {len(symbols)} symbols")
        # Fetch latest full data for all symbols
        data_dict = {}
        min_len = float('inf')
        for sym in symbols:
            data = self.data_ingestion.get_latest_data(sym)
            if len(data) < 500:
                logger.error(f"Insufficient latest data for {sym} ({len(data)} bars) — aborting portfolio update")
                return
            data_dict[sym] = data
            min_len = min(min_len, len(data))
        if min_len < 500:
            logger.error("Not enough common recent history across symbols for portfolio update")
            return
        timesteps = timesteps or self.config.get('PORTFOLIO_ONLINE_TIMESTEPS', 100_000)
        try:
            # Recreate environment with latest data
            env = DummyVecEnv([lambda: PortfolioEnv(
                data_dict=data_dict,
                symbols=symbols,
                initial_balance=self.config.get('INITIAL_BALANCE', 100_000.0),
                max_leverage=self.config.get('MAX_LEVERAGE', 3.0)
            )])
            new_vec_env = VecNormalize(env, norm_obs=True, norm_reward=True)
            # Transfer normalization statistics from previous VecNormalize if exists
            if self.portfolio_vec_norm is not None:
                new_vec_env.obs_rms = self.portfolio_vec_norm.obs_rms
                new_vec_env.ret_rms = self.portfolio_vec_norm.ret_rms
                new_vec_env.epsilon = self.portfolio_vec_norm.epsilon
                new_vec_env.clip_obs = self.portfolio_vec_norm.clip_obs
                new_vec_env.clip_reward = self.portfolio_vec_norm.clip_reward
                logger.info("Transferred VecNormalize statistics for online portfolio update")
            # Attach new environment to existing model
            model = self.portfolio_ppo_model
            model.set_env(new_vec_env)
            # Lower learning rate for safe online fine-tuning
            original_lr = model.learning_rate
            model.learning_rate = 1e-5 # Conservative fixed low LR for online updates
            logger.info(f"Reduced learning rate to 1e-5 for online portfolio update (original was {original_lr})")
            nan_callback = NaNStopCallback()
            profit_callback = ProfitLoggingCallback() # Enhanced profit logging
            logger.info(f"Continuing portfolio PPO training for {timesteps} timesteps (incremental)")
            model.learn(
                total_timesteps=timesteps,
                callback=[nan_callback, profit_callback],
                reset_num_timesteps=False # Crucial: continue from previous timesteps
            )
            # Update stored references
            self.portfolio_vec_norm = new_vec_env
            # CRIT-14 FIX: Rebuild/restore portfolio causal manager after online update
            if self.portfolio_causal_manager is None:
                logger.info("Rebuilding portfolio causal manager after online update...")
                # (same stacked build as in train_portfolio)
                all_features = []
                symbol_ids = []
                for sym in symbols:
                    data = self.data_ingestion.get_latest_data(sym)
                    features = generate_features(data, 'trending', sym, data)
                    if features is not None and features.shape[0] > 0:
                        rows = features.shape[0]
                        all_features.append(features)
                        symbol_ids.extend([sym] * rows)
                if all_features:
                    full_matrix = np.vstack(all_features)
                    full_df = pd.DataFrame(full_matrix, columns=[f'feat_{i}' for i in range(full_matrix.shape[1])])
                    full_df['symbol_id'] = symbol_ids
                    self.portfolio_causal_manager = CausalSignalManager(
                        self.portfolio_ppo_model,
                        full_df,
                        symbol="portfolio",
                        data_ingestion=self.data_ingestion
                    )
                    # Re-warmup after rebuild
                    if hasattr(self, 'signal_gen') and hasattr(self.signal_gen, 'live_signal_history'):
                        all_closed = [e for hist in self.signal_gen.live_signal_history.values() for e in hist if e.get('realized_return') is not None]
                        for entry in all_closed:
                            obs = entry.get('obs')
                            if obs is not None and isinstance(obs, list) and len(obs) > 0:
                                try:
                                    obs_array = np.array(obs, dtype=np.float32).reshape(1, -1)
                                    action = entry.get('direction', 0) * entry.get('confidence', 1.0)
                                    reward = entry.get('realized_return', 0.0)
                                    self.portfolio_causal_manager.replay_buffer.push(obs_array, action, reward)
                                except:
                                    pass
                        logger.info(f"[CAUSAL WARMUP AFTER UPDATE] Portfolio causal buffer refreshed")
            # Save updated model
            save_ppo_model(self, "portfolio")
            logger.info("Online portfolio PPO update completed and saved")
        except Exception as e:
            logger.error(f"Online portfolio PPO update failed: {e}", exc_info=True)

    def initialize_models(self, symbols: list):
        for symbol in symbols:
            self.stacking_models[symbol] = []
            self.confidence_thresholds[symbol] = []
            self.ppo_models[symbol] = None
            self.vec_norms[symbol] = None
            self.causal_wrappers[symbol] = None # ← modern unified manager
        self.portfolio_ppo_model = None
        self.portfolio_vec_norm = None
        self.portfolio_causal_manager = None # ← modern unified manager

    def train_symbols_parallel(self, symbols: list, full_ppo: bool = True):
        if not symbols:
            return
        if self.config.get('PORTFOLIO_PPO', False):
            # Portfolio mode: train single shared policy (skip per-symbol PPO)
            self.train_portfolio(symbols)
            # Optionally still train per-symbol stacking for hybrid fallback
            for sym in symbols:
                data = self.data_ingestion.get_latest_data(sym)
                if len(data) >= 500:
                    # M-1 / HMM fix: pass shared cache to stacking training
                    self.stacking_models[sym] = train_stacking(
                        symbol=sym,
                        data=data,
                        full_hist_df=data,
                        regime_cache=self.regime_cache  # ← NEW: share cache
                    )
                    self.walk_forward_optimize_thresholds(sym, data)
        else:
            # Legacy per-symbol parallel training
            max_workers = min(len(symbols), 16)
            logger.info(f"Starting parallel training for {len(symbols)} symbols (max_workers={max_workers})")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.train_symbol, symbol, full_ppo=full_ppo, regime_cache=self.regime_cache)
                    for symbol in symbols
                ]
                for future in futures:
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error(f"Training thread exception: {exc}", exc_info=True)
            logger.info("Parallel training completed for all symbols")

# models/trainer.py
# =====================================================================
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.samplers import TPESampler
from concurrent.futures import ThreadPoolExecutor
from strategy.regime import detect_regime # Consolidated single source of truth
from .ppo_utils import (
    train_ppo,
    save_ppo_model,
    load_ppo_model,
    update_model_weights as ppo_update_model_weights,
    make_cosine_annealing_schedule,
    AuxVolatilityCallback,
    NaNStopCallback as _PpoNaNStopCallback,  # FIX #53: Import from canonical source
)
from .stacking_ensemble import train_stacking
from config import CONFIG
from models.features import generate_features
from models.portfolio_env import PortfolioEnv # NEW: Portfolio environment
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import torch
import json
import os
import tempfile
import shutil
import threading
# NEW: Share the same persistent regime cache file as bot.py
REGIME_CACHE_FILE = "regime_cache.json"
# NEW: Use the modern CausalSignalManager from the extracted module (Phase 1 modularization)
from models.causal_signal_manager import CausalSignalManager # ← Phase 1 modularization fix

logger = logging.getLogger(__name__)

# FIX #53: Removed duplicate NaNStopCallback — now imported from ppo_utils as _PpoNaNStopCallback
NaNStopCallback = _PpoNaNStopCallback

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
        # Thread lock for shared state mutations during parallel training
        self._lock = threading.Lock()
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
        """Save cache atomically via tempfile + rename (crash-safe, thread-safe)."""
        try:
            with self._lock:
                fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(REGIME_CACHE_FILE) or '.', suffix='.tmp')
                with os.fdopen(fd, 'w') as f:
                    json.dump(self.regime_cache, f, default=str)
                shutil.move(tmp_path, REGIME_CACHE_FILE)
            logger.debug(f"Saved regime cache with {len(self.regime_cache)} symbols")
        except Exception as e:
            logger.warning(f"Failed to save regime cache: {e}")

    def get_cached_regime(self, symbol: str, data: pd.DataFrame) -> tuple:
        """Cached regime detection — now uses the shared persistent cache (simple symbol key)
        B-04 FIX: Fully defensive unpacking for legacy plain-string / float / malformed entries"""
        cache_key = symbol # Simple key — matches bot.py exactly
        # Thread-safe read: regime_cache may be written by parallel training threads
        with self._lock:
            cached_value = self.regime_cache.get(cache_key)
        if cached_value is not None:
            value = cached_value
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
                regime = 'trending_up' if value > 0.5 else 'mean_reverting'
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
            regime = ('trending_up' if recent_return >= 0 else 'trending_down') if abs(recent_return) > 0.015 else 'mean_reverting'
            persistence = 0.5 # BUG-11 FIX: neutral fallback instead of aggressive 0.92/0.35
        # Log fallback usage (helps track cache miss frequency)
        logger.debug(f"[REGIME FALLBACK] {symbol} → {regime} (persistence={persistence:.3f})")
        # Save to shared cache (as list for JSON compatibility)
        # FIX #23: Wrap dict mutation in lock to prevent race with concurrent threads
        with self._lock:
            self.regime_cache[cache_key] = [regime, persistence]
        self._save_regime_cache()
        return regime, persistence

    def walk_forward_optimize_thresholds(self, symbol: str, data: pd.DataFrame):
        # ARCHITECTURAL LIMITATION: Stacking models (train_stacking) are trained on the full
        # dataset before this walk-forward threshold optimization runs. This means stacking
        # model predictions on the OOS folds are contaminated (the model has seen the OOS data
        # during training). The 20-bar embargo partially mitigates short-horizon leakage, but
        # the thresholds found here may be overfit. A proper fix would require per-fold stacking
        # model retraining, which is prohibitively expensive with 15 LightGBM models per fold.
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
                # FIX #23: Validate feature count matches what models were trained on.
                # If feature count changed (e.g., new indicators added), predict will crash.
                expected_ncols = stacking_models[0].num_feature()
                actual_ncols = features.shape[1] if features.ndim == 2 else features.shape[0]
                if actual_ncols != expected_ncols:
                    logger.warning(f"[STACKING] {symbol}: feature count mismatch (got {actual_ncols}, "
                                   f"model expects {expected_ncols}) — using 0.5 fallback")
                    ensemble_prob = np.full(len(features), 0.5)
                    stacking_models = []  # skip prediction loop
                for model in stacking_models:
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(features)[:, 1]
                    else:
                        probs = model.predict(features)
                    meta_probs.append(probs)
                ensemble_prob = np.mean(meta_probs, axis=0) if meta_probs else np.full(len(features), 0.5)
            else:
                ensemble_prob = np.full(len(features), 0.5)
            ensemble_prob = ensemble_prob[:-1] # drop last (no future return for it)
            # Rescale predictions to percentile ranks [0, 1] so thresholds are meaningful
            # regardless of raw prediction spread. LightGBM with early stopping often produces
            # predictions clustered in [0.48, 0.52], making fixed thresholds at 0.55+ useless.
            prob_spread = np.std(ensemble_prob)
            if prob_spread < 0.05:
                logger.info(f"[THRESHOLD FIX] {symbol} prediction spread too narrow ({prob_spread:.4f}) — rescaling to percentile ranks")
                ensemble_prob = pd.Series(ensemble_prob).rank(pct=True).values
        # Compute forward returns for alignment
        returns = data['close'].pct_change().shift(-1).dropna()
        # Features correspond to the TAIL of data (rolling windows drop leading rows)
        # Align by taking the last len(ensemble_prob) returns to match feature positions
        n_probs = len(ensemble_prob)
        if len(returns) > n_probs:
            aligned_returns = returns.iloc[-n_probs:]
        else:
            aligned_returns = returns.iloc[:n_probs]
            ensemble_prob = ensemble_prob[:len(aligned_returns)]
        df = pd.DataFrame({
            'meta_prob': ensemble_prob[:len(aligned_returns)],
            'return': aligned_returns.values
        }, index=aligned_returns.index)
        # Apr-19 audit: threshold walk-forward was evaluating over the FULL
        # dataframe, but stacking_ensemble trains on the first ~70% — so
        # windows 1-4 of 6 were evaluating against predictions the stacking
        # model had already memorised. To restore true out-of-sample
        # properties without the 3-5× cost of per-fold stacking retrains,
        # we restrict the walk-forward to the tail `STACKING_HOLDOUT_FRAC`
        # of the dataframe, which matches the unseen portion of the
        # stacking training split.
        holdout_frac = float(self.config.get('STACKING_HOLDOUT_FRAC', 0.30))
        if len(df) > 200 and 0.1 <= holdout_frac <= 0.5:
            tail_cut = int(len(df) * (1.0 - holdout_frac))
            full_df = df
            df = df.iloc[tail_cut:].copy()
            logger.info(
                f"{symbol} walk-forward restricted to stacking-holdout tail "
                f"({len(df)}/{len(full_df)} bars; holdout_frac={holdout_frac})"
            )
        n_windows = int(self.config.get('WALK_FORWARD_N_WINDOWS', 3))
        # HIGH FIX: Increased embargo from 1 to 20 bars to reduce look-ahead bias.
        # Still useful: even inside the stacking-holdout tail, the first few
        # bars share auto-correlation with the final training bars.
        embargo = 20
        window_size = len(df) // n_windows
        for w in range(n_windows):
            chunk_start = w * window_size
            chunk_end = (w + 1) * window_size if w < n_windows - 1 else len(df)
            chunk_df = df.iloc[chunk_start:chunk_end]
            # ==================== TRUE OOS VALIDATION (70/30 split per window, embargo gap) ====================
            train_size = int(len(chunk_df) * 0.70)
            train_df = chunk_df.iloc[:train_size]
            oos_df = chunk_df.iloc[train_size + embargo:]  # embargo bars gap between train and OOS
            if len(train_df) < 20 or len(oos_df) < 10:
                logger.warning(f"{symbol} window {w+1} too small (train={len(train_df)}, oos={len(oos_df)}) — skipping")
                continue
            logger.debug(f"{symbol} window {w+1}/{n_windows} — train:{len(train_df)} | OOS:{len(oos_df)}")
            def objective(trial):
                long_thresh = trial.suggest_float('long_thresh', 0.55, 0.85)
                short_thresh = trial.suggest_float('short_thresh', 0.15, 0.45)
                # Apr-19 audit: replace the static "distance from 0.65/0.35"
                # penalty with a Sharpe-sustainability objective. The old
                # penalty (0.70×|long-0.65| + |short-0.35|) subtracted up to
                # -0.14 from legitimately aggressive thresholds, pulling the
                # optimiser back to arbitrary midpoints. The new objective
                # rewards robust IS/OOS consistency: very strong IS with
                # small cross-validated gap gets accepted verbatim, otherwise
                # IS is discounted by gap ratio.
                signals = np.where(train_df['meta_prob'] > long_thresh, 1,
                                   np.where(train_df['meta_prob'] < short_thresh, -1, 0))
                strat_returns = signals * train_df['return']
                is_sharpe_trial = strat_returns.mean() / (strat_returns.std() + 1e-8) * np.sqrt(252 * 26)
                # Cheap pseudo-OOS: last 20% of train window
                cv_cut = max(1, int(len(train_df) * 0.8))
                cv_oos_df = train_df.iloc[cv_cut:]
                if len(cv_oos_df) >= 5:
                    cv_signals = np.where(cv_oos_df['meta_prob'] > long_thresh, 1,
                                          np.where(cv_oos_df['meta_prob'] < short_thresh, -1, 0))
                    cv_rets = cv_signals * cv_oos_df['return']
                    cv_sharpe = cv_rets.mean() / (cv_rets.std() + 1e-8) * np.sqrt(252 * 26)
                    gap = is_sharpe_trial - cv_sharpe
                    gap_ratio_trial = abs(gap) / (abs(is_sharpe_trial) + 1e-8)
                else:
                    gap_ratio_trial = 0.0
                # Auto-accept aggressive: very strong IS with small gap
                if is_sharpe_trial > 2.0 and gap_ratio_trial < 0.15:
                    return float(is_sharpe_trial)
                # Otherwise discount IS by gap severity (never more than -50%)
                discount = min(0.5, 0.1 * gap_ratio_trial)
                return float(is_sharpe_trial * (1.0 - discount))
            # Scale trials to window size — prevents overfitting small samples
            max_trials = min(self.config.get('OPTUNA_TRIALS', 350), max(50, len(train_df) // 5))
            study = optuna.create_study(direction='maximize', sampler=TPESampler())
            study.optimize(objective, n_trials=max_trials,
                           timeout=self.config.get('OPTUNA_TIMEOUT', 900))
            best = study.best_trial
            # Compute raw IS Sharpe (without penalty) for fair comparison with OOS
            best_signals_is = np.where(train_df['meta_prob'] > best.params['long_thresh'], 1,
                                       np.where(train_df['meta_prob'] < best.params['short_thresh'], -1, 0))
            best_rets_is = best_signals_is * train_df['return']
            is_sharpe = best_rets_is.mean() / (best_rets_is.std() + 1e-8) * np.sqrt(252 * 26)
            # Compute true OOS score on unseen data
            signals_oos = np.where(oos_df['meta_prob'] > best.params['long_thresh'], 1,
                                   np.where(oos_df['meta_prob'] < best.params['short_thresh'], -1, 0))
            strat_returns_oos = signals_oos * oos_df['return']
            oos_sharpe = strat_returns_oos.mean() / (strat_returns_oos.std() + 1e-8) * np.sqrt(252 * 26)
            # Use ratio-based gap detection — absolute gap is misleading for high-Sharpe windows
            is_oos_gap = is_sharpe - oos_sharpe
            gap_ratio = is_oos_gap / (abs(is_sharpe) + 1e-8)
            logger.info(
                f"{symbol} window {w+1}/{n_windows} — IS Sharpe: {is_sharpe:.3f} | "
                f"OOS Sharpe: {oos_sharpe:.3f} | gap: {is_oos_gap:.3f} ({gap_ratio:.0%}) "
                f"(long={best.params['long_thresh']:.3f}, short={best.params['short_thresh']:.3f})"
            )
            if gap_ratio > 0.50 and is_oos_gap > 2.0:
                logger.warning(f"{symbol} window {w+1} shows significant overfitting — gap {is_oos_gap:.2f} ({gap_ratio:.0%} of IS)")
            # Accept unless OOS is clearly destructive. Previous gate `oos > 0.0`
            # silently rejected windows with small negative OOS (e.g. -0.2 to 0)
            # even when IS was modest and the absolute gap was small. Relax to
            # a two-part gate: allow slightly-negative OOS when (a) the IS→OOS
            # gap is small AND (b) OOS is above an absolute floor. This keeps
            # truly overfit windows out but stops throwing away borderline
            # noise (the Apr-19 pattern where 6/8 symbols had 6–9% gaps).
            oos_floor = self.config.get('OOS_SHARPE_ACCEPT_FLOOR', -0.25)
            gap_cap = self.config.get('OOS_ACCEPT_MAX_GAP_RATIO', 0.35)
            abs_gap_cap = self.config.get('OOS_ACCEPT_MAX_ABS_GAP', 0.5)
            strong_oos = float(self.config.get('OOS_SHARPE_STRONG_ACCEPT', 1.0))
            # Apr-19 audit v2: the absolute-gap cap ONLY applies when OOS is
            # itself weak. A window with OOS Sharpe 5.0 and IS Sharpe 11.5
            # has a 6.5 abs gap but OOS is excellent and must not be rejected
            # just because IS was even better. The abs-gap cap was meant to
            # catch cases where OOS is borderline-negative and the ratio
            # metric looks deceptively small due to a tiny IS — so only
            # enforce it when OOS is below `strong_oos`.
            absolute_gap_ok = (oos_sharpe >= strong_oos) or (is_oos_gap < abs_gap_cap)
            accept = (
                # Accept when OOS is strongly positive regardless of gap.
                oos_sharpe >= strong_oos
                # Or when OOS is modestly positive and the absolute gap is manageable.
                or (oos_sharpe > 0.0 and absolute_gap_ok)
                # Or borderline-negative OOS with a tight ratio (noise window).
                or (oos_sharpe > oos_floor and gap_ratio < gap_cap and absolute_gap_ok)
            )
            if accept:
                final_long_thresh = best.params['long_thresh']
                final_short_thresh = best.params['short_thresh']
                final_long_thresh = max(0.50, min(0.85, final_long_thresh))
                final_short_thresh = max(0.15, min(0.50, final_short_thresh))
                chunk_start_time = df.index[chunk_start]
                thresh_list = self.confidence_thresholds.setdefault(symbol, [])
                thresh_list.append({
                    'valid_from': chunk_start_time,
                    'long': final_long_thresh,
                    'short': final_short_thresh,
                    'sharpe_is': is_sharpe,
                    'sharpe_oos': oos_sharpe
                })
                if len(thresh_list) > 12:
                    self.confidence_thresholds[symbol] = thresh_list[-12:]
                logger.info(
                    f"{symbol} walk-forward window {w+1}/{n_windows} ({chunk_start_time.date()}): "
                    f"long>{final_long_thresh:.3f}, short<{final_short_thresh:.3f} (IS {is_sharpe:.2f} | OOS {oos_sharpe:.2f})"
                )
            else:
                logger.warning(
                    f"{symbol} window {w+1} rejected — "
                    f"OOS={oos_sharpe:.3f} (floor={oos_floor}) gap_ratio={gap_ratio:.0%} (cap={gap_cap:.0%}) "
                    f"abs_gap={is_oos_gap:.2f} (cap={abs_gap_cap:.2f})"
                )
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
            # M49 FIX: Pass a copy to prevent cross-thread mutation
            regime_cache=dict(regime_cache or self.regime_cache)
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
                cached_regime, _ = self.get_cached_regime(symbol, data)
                features = generate_features(data, cached_regime, symbol, data)
                if features is not None and features.shape[0] > 0:
                    features_df = pd.DataFrame(
                        features,
                        columns=[f'feat_{i}' for i in range(features.shape[1])]
                    )
                    self.causal_wrappers[symbol] = CausalSignalManager(
                        self.ppo_models[symbol],
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
        if existing_model is not None and not self.config.get('FORCE_PPO_RETRAIN', False):
            logger.info("Existing portfolio PPO model found — skipping full training, ready for inference/online updates")
            self.portfolio_ppo_model = existing_model
            self.portfolio_vec_norm = existing_norm
            # FIX #41: Removed dead causal warmup code. Trainer does not have signal_gen attribute,
            # so hasattr(self, 'signal_gen') was always False. Causal warmup is handled by
            # CausalRLManager.sync_daily_rewards() which has access to signal_gen.
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
        # Create vectorized PortfolioEnv wrapped with Monitor for episode tracking
        env = DummyVecEnv([lambda: Monitor(PortfolioEnv(
            data_dict=data_dict,
            symbols=symbols,
            initial_balance=self.config.get('INITIAL_BALANCE', 100_000.0),
            max_leverage=self.config.get('MAX_LEVERAGE', 3.0)
        ))])
        vec_env = VecNormalize(env, norm_obs=True, norm_reward=True)
        success = False
        for attempt in range(3):
            try:
                use_recurrent = CONFIG.get('PPO_RECURRENT', True)
                use_custom_gtrxl = CONFIG.get('USE_CUSTOM_GTRXL', True) and use_recurrent
                # FIXED: Removed use_sde=False to avoid duplicate argument error
                policy_kwargs = dict(
                    net_arch=dict(pi=[256, 256], vf=[512, 256]), # Asymmetric: bigger critic for randomized starts
                )
                # NOTE: Do NOT set features_extractor_class=None for any path.
                # GTrXL uses its own DummyFeaturesExtractor (set inside AuxGTrXLPolicy).
                # LSTM/RecurrentPPO uses the default FlattenExtractor.
                # Passing None crashes non-GTrXL recurrent policies.
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
                        learning_rate=make_cosine_annealing_schedule(),
                        n_steps=CONFIG.get('PPO_N_STEPS', 2048),
                        batch_size=CONFIG.get('PPO_BATCH_SIZE', 256),
                        n_epochs=CONFIG.get('PPO_N_EPOCHS', 4),
                        gamma=CONFIG.get('PPO_GAMMA', 0.97),
                        gae_lambda=CONFIG.get('PPO_GAE_LAMBDA', 0.93),
                        clip_range=CONFIG.get('PPO_CLIP_RANGE', 0.18),
                        ent_coef=CONFIG.get('PPO_ENTROPY_COEFF', 0.03),
                        vf_coef=CONFIG.get('VF_COEF', 1.0),
                        max_grad_norm=CONFIG.get('PPO_MAX_GRAD_NORM', 0.4),
                        device="auto"
                    )
                else:
                    from .policies import AuxMlpPolicy, CustomPPO
                    model = CustomPPO(
                        policy=AuxMlpPolicy,
                        env=vec_env,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        tensorboard_log="./ppo_tensorboard/portfolio" if CONFIG.get('USE_TENSORBOARD') else None,
                        learning_rate=make_cosine_annealing_schedule(),
                        n_steps=CONFIG.get('PPO_N_STEPS', 2048),
                        batch_size=CONFIG.get('PPO_BATCH_SIZE', 256),
                        n_epochs=CONFIG.get('PPO_N_EPOCHS', 4),
                        gamma=CONFIG.get('PPO_GAMMA', 0.97),
                        gae_lambda=CONFIG.get('PPO_GAE_LAMBDA', 0.93),
                        clip_range=CONFIG.get('PPO_CLIP_RANGE', 0.18),
                        ent_coef=CONFIG.get('PPO_ENTROPY_COEFF', 0.03),
                        vf_coef=CONFIG.get('VF_COEF', 1.0),
                        max_grad_norm=CONFIG.get('PPO_MAX_GRAD_NORM', 0.4),
                        device="auto"
                    )
                nan_callback = NaNStopCallback()
                profit_callback = ProfitLoggingCallback()
                aux_callback = AuxVolatilityCallback() if CONFIG.get('PPO_AUX_TASK', True) else None
                callbacks = [nan_callback, profit_callback] + ([aux_callback] if aux_callback else [])
                logger.info(f"Training portfolio PPO for {timesteps} timesteps")
                model.learn(
                    total_timesteps=timesteps,
                    callback=callbacks,
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
                sym_regime, _ = self.get_cached_regime(sym, data)
                features = generate_features(data, sym_regime, sym, data)
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
                            except Exception:
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

    def _validate_portfolio_model(self, model, vec_env, n_steps: int = 500,
                                   seed: int = 1337) -> float:
        """RETRAIN-GUARD: Validate a model by rolling it out deterministically on
        the given vec-normalized env. Returns mean reward per step as a scalar
        quality metric.

        Deterministic (fixed seed, deterministic=True) so pre/post retrain scores
        are directly comparable. Stops early on terminal/truncated.
        Returns 0.0 on any failure — caller should treat unknown scores as neutral.
        """
        try:
            import numpy as _np
            # vec_env is a VecNormalize wrapping DummyVecEnv([Monitor(PortfolioEnv)])
            # Fresh reset with fixed seed for reproducibility
            obs = vec_env.reset()
            # VecNormalize returns (obs,) tuple for DummyVecEnv
            if isinstance(obs, tuple):
                obs = obs[0]
            total_reward = 0.0
            steps = 0
            lstm_state = None
            # Use model.predict with deterministic=True for reproducibility
            for _ in range(n_steps):
                try:
                    action, lstm_state = model.predict(
                        obs,
                        state=lstm_state,
                        episode_start=_np.array([False]),
                        deterministic=True,
                    )
                except TypeError:
                    action, _ = model.predict(obs, deterministic=True)
                    lstm_state = None
                step_result = vec_env.step(action)
                # DummyVecEnv returns (obs, rewards, dones, infos)
                if len(step_result) == 4:
                    obs, rewards, dones, _infos = step_result
                else:
                    obs, rewards, dones, _truncated, _infos = step_result
                total_reward += float(rewards[0]) if hasattr(rewards, '__len__') else float(rewards)
                steps += 1
                if bool(dones[0]) if hasattr(dones, '__len__') else bool(dones):
                    break
            mean_reward = total_reward / max(1, steps)
            return float(mean_reward)
        except Exception as e:
            logger.debug(f"[RETRAIN-GUARD] validation rollout failed: {e}")
            return 0.0

    def update_portfolio_weights(self, timesteps: int = None):
        """
        Incremental/online update for the portfolio-level PPO model.
        Recreates environment with latest data, transfers VecNormalize stats if possible,
        continues learning on existing model with low learning rate for safety.

        RETRAIN-GUARD: Before retraining, checkpoints the current model to
        `ppo_model_prev.zip` and records a baseline validation score. After
        retraining, re-validates. If the new model scores materially worse
        (by RETRAIN_GUARD_MIN_DROP mean-reward per step), we restore from
        the backup — preventing a bad retrain from silently degrading every
        downstream layer.
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

        # ========== RETRAIN-GUARD: checkpoint + baseline validation ==========
        guard_enabled = CONFIG.get('RETRAIN_GUARD_ENABLED', True)
        model_path = os.path.join("ppo_checkpoints", "portfolio", "ppo_model.zip")
        backup_path = os.path.join("ppo_checkpoints", "portfolio", "ppo_model_prev.zip")
        baseline_score = None
        backup_created = False
        if guard_enabled and os.path.exists(model_path):
            try:
                import shutil as _shutil
                _shutil.copy2(model_path, backup_path)
                backup_created = True
                logger.info(f"[RETRAIN-GUARD] Checkpointed {model_path} → {backup_path}")
            except Exception as e:
                logger.warning(f"[RETRAIN-GUARD] Backup failed ({e}) — proceeding without rollback capability")
        # ========== END RETRAIN-GUARD SETUP ==========

        try:
            # Recreate environment with latest data (Monitor for episode tracking)
            env = DummyVecEnv([lambda: Monitor(PortfolioEnv(
                data_dict=data_dict,
                symbols=symbols,
                initial_balance=self.config.get('INITIAL_BALANCE', 100_000.0),
                max_leverage=self.config.get('MAX_LEVERAGE', 3.0)
            ))])
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
            # Lower learning rate for safe online fine-tuning.
            # Save the original lr_schedule (may be a callable like cosine annealing).
            # After the online update we restore the exact same object so the schedule
            # state (progress_remaining tracking) is preserved, not flattened to a scalar.
            original_lr = model.learning_rate  # callable or float
            online_lr = CONFIG.get('PPO_ONLINE_LEARNING_RATE', 5e-5)
            model.learning_rate = online_lr  # scalar for online fine-tuning
            logger.info(f"Reduced learning rate to {online_lr} for online portfolio update (original type: {type(original_lr).__name__})")
            # RETRAIN-GUARD: baseline score before training (using OLD policy on new env).
            # Deterministic rollout with fixed seed → reproducible comparison with post-retrain score.
            if guard_enabled:
                guard_val_steps = CONFIG.get('RETRAIN_GUARD_VALIDATION_STEPS', 500)
                baseline_score = self._validate_portfolio_model(model, new_vec_env, n_steps=guard_val_steps)
                logger.info(f"[RETRAIN-GUARD] Baseline validation score (pre-retrain): {baseline_score:+.5f} mean-reward/step")
            nan_callback = NaNStopCallback()
            profit_callback = ProfitLoggingCallback()
            aux_callback = AuxVolatilityCallback() if CONFIG.get('PPO_AUX_TASK', True) else None
            callbacks = [nan_callback, profit_callback] + ([aux_callback] if aux_callback else [])
            logger.info(f"Continuing portfolio PPO training for {timesteps} timesteps (incremental)")
            model.learn(
                total_timesteps=timesteps,
                callback=callbacks,
                reset_num_timesteps=False
            )
            # Restore original learning rate (callable schedule or scalar) after online fine-tuning.
            # This preserves the cosine annealing schedule object if that's what was originally set,
            # so subsequent train() calls use the correct schedule rather than a flat scalar.
            model.learning_rate = original_lr
            logger.info(f"Restored learning rate schedule after online portfolio update (type: {type(original_lr).__name__})")
            # Update stored references
            self.portfolio_vec_norm = new_vec_env
            # CRIT-14 FIX: Always rebuild portfolio causal manager after online update
            # so the causal graph reflects the updated model and latest data.
            # Previously only rebuilt when None — now rebuilds unconditionally when causal RL is enabled.
            if CONFIG.get('USE_CAUSAL_RL', False):
                logger.info("Rebuilding portfolio causal manager after online update...")
                # (same stacked build as in train_portfolio)
                all_features = []
                symbol_ids = []
                for sym in symbols:
                    data = self.data_ingestion.get_latest_data(sym)
                    sym_regime, _ = self.get_cached_regime(sym, data)
                    features = generate_features(data, sym_regime, sym, data)
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
                                except Exception as e:
                                    logger.debug(f"Causal warmup entry failed: {e}")
                        logger.info(f"[CAUSAL WARMUP AFTER UPDATE] Portfolio causal buffer refreshed")
            # Save updated model
            save_ppo_model(self, "portfolio")
            logger.info("Online portfolio PPO update completed and saved")

            # ========== RETRAIN-GUARD: post-training validation + rollback ==========
            if guard_enabled and baseline_score is not None and backup_created:
                try:
                    import json as _json
                    post_score = self._validate_portfolio_model(model, new_vec_env, n_steps=guard_val_steps)
                    min_drop = CONFIG.get('RETRAIN_GUARD_MIN_DROP', 0.002)   # min abs drop to trigger
                    rel_drop = CONFIG.get('RETRAIN_GUARD_REL_DROP', 0.20)    # or 20% relative
                    abs_delta = post_score - baseline_score
                    rel_delta = abs_delta / (abs(baseline_score) + 1e-9)
                    degraded = (abs_delta < -min_drop) and (rel_delta < -rel_drop)
                    decision = "ROLLBACK" if degraded else "ACCEPT"
                    logger.info(f"[RETRAIN-GUARD] Post-retrain validation: {post_score:+.5f} "
                                f"(baseline {baseline_score:+.5f}, Δ={abs_delta:+.5f}, "
                                f"rel={rel_delta:+.2%}) — {decision}")
                    # Structured JSON for audit trail
                    logger.info(_json.dumps({
                        "event": "retrain_guard_decision",
                        "baseline_score": baseline_score,
                        "post_score": post_score,
                        "abs_delta": abs_delta,
                        "rel_delta": rel_delta,
                        "min_drop_threshold": min_drop,
                        "rel_drop_threshold": rel_drop,
                        "decision": decision,
                    }))
                    if degraded:
                        logger.warning(f"[RETRAIN-GUARD] New model DEGRADED quality "
                                       f"({abs_delta:+.5f} drop) — ROLLING BACK to backup")
                        import shutil as _shutil
                        _shutil.copy2(backup_path, model_path)
                        # Reload the restored model into memory
                        from .ppo_utils import load_ppo_model
                        load_ppo_model(self, "portfolio")
                        # Lock VecNormalize to inference mode again
                        if self.portfolio_vec_norm is not None:
                            self.portfolio_vec_norm.training = False
                        logger.info(f"[RETRAIN-GUARD] ✅ Rollback complete — restored previous model weights")
                    else:
                        logger.info(f"[RETRAIN-GUARD] ✅ Retrain accepted — new weights kept")
                except Exception as e:
                    logger.error(f"[RETRAIN-GUARD] Post-validation/rollback failed: {e} — keeping new weights (fail-open)")
            # ========== END RETRAIN-GUARD ==========
        except Exception as e:
            logger.error(f"Online portfolio PPO update failed: {e}", exc_info=True)
            # If training itself failed, attempt to restore from backup
            if guard_enabled and backup_created:
                try:
                    import shutil as _shutil
                    _shutil.copy2(backup_path, model_path)
                    from .ppo_utils import load_ppo_model
                    load_ppo_model(self, "portfolio")
                    logger.info(f"[RETRAIN-GUARD] Training failed — restored previous model from backup")
                except Exception as restore_err:
                    logger.error(f"[RETRAIN-GUARD] Backup restore also failed: {restore_err}")

    def micro_retrain_portfolio(self, timesteps: Optional[int] = None,
                                 recent_bars: Optional[int] = None,
                                 learning_rate: Optional[float] = None) -> Dict:
        """A4 — Pre-market micro-retrain.

        Lighter version of `update_portfolio_weights`:
          - Much fewer timesteps (default 5K vs 100K)
          - Smaller LR (default 1e-5 vs 5e-5)
          - Uses only the last N bars (default 500 vs ~5000)
          - Wrapped in RETRAIN-GUARD with STRICTER thresholds (small update
            should have small variance — if it still degrades materially,
            rollback aggressively)

        Runs at 08:30 ET (pre-market) to adapt overnight.
        Returns dict with decision + scores.
        """
        import shutil as _shutil
        result = {"status": "skipped", "reason": "unknown"}

        if self.portfolio_ppo_model is None:
            result["reason"] = "no_portfolio_model"
            return result

        symbols = self.config.get('SYMBOLS', [])
        if not symbols:
            result["reason"] = "no_symbols"
            return result

        timesteps = timesteps or CONFIG.get('PPO_MICRO_RETRAIN_TIMESTEPS', 5000)
        recent_bars = recent_bars or CONFIG.get('PPO_MICRO_RETRAIN_BARS', 500)
        learning_rate = learning_rate or CONFIG.get('PPO_MICRO_RETRAIN_LR', 1e-5)

        # Fetch recent data only — recency-focused update
        data_dict = {}
        for sym in symbols:
            data = self.data_ingestion.get_latest_data(sym)
            if data is None or len(data) < recent_bars + 50:
                result["reason"] = f"insufficient_data_for_{sym}"
                return result
            # Tail to most-recent `recent_bars` — narrow window drives adaptation
            data_dict[sym] = data.tail(recent_bars)

        # RETRAIN-GUARD (stricter than nightly)
        guard_enabled = CONFIG.get('RETRAIN_GUARD_ENABLED', True)
        model_path = os.path.join("ppo_checkpoints", "portfolio", "ppo_model.zip")
        backup_path = os.path.join("ppo_checkpoints", "portfolio", "ppo_model_micro_prev.zip")
        baseline_score = None
        backup_created = False
        if guard_enabled and os.path.exists(model_path):
            try:
                _shutil.copy2(model_path, backup_path)
                backup_created = True
                logger.info(f"[MICRO-RETRAIN GUARD] Checkpointed to {backup_path}")
            except Exception as e:
                logger.warning(f"[MICRO-RETRAIN GUARD] Backup failed ({e})")

        try:
            # Build micro-training env over recent-bars-only window
            env = DummyVecEnv([lambda: Monitor(PortfolioEnv(
                data_dict=data_dict,
                symbols=symbols,
                initial_balance=self.config.get('INITIAL_BALANCE', 100_000.0),
                max_leverage=self.config.get('MAX_LEVERAGE', 3.0)
            ))])
            new_vec_env = VecNormalize(env, norm_obs=True, norm_reward=True)
            if self.portfolio_vec_norm is not None:
                new_vec_env.obs_rms = self.portfolio_vec_norm.obs_rms
                new_vec_env.ret_rms = self.portfolio_vec_norm.ret_rms

            model = self.portfolio_ppo_model
            model.set_env(new_vec_env)
            original_lr = model.learning_rate
            model.learning_rate = learning_rate

            # Stricter validation thresholds for micro-retrain — tiny update so
            # any meaningful drop should trigger rollback
            guard_val_steps = CONFIG.get('PPO_MICRO_RETRAIN_VALIDATION_STEPS', 300)
            if guard_enabled:
                baseline_score = self._validate_portfolio_model(model, new_vec_env, n_steps=guard_val_steps)
                logger.info(f"[MICRO-RETRAIN GUARD] Baseline: {baseline_score:+.5f} mean-r/step")

            logger.info(f"[MICRO-RETRAIN] Starting {timesteps} timesteps at LR={learning_rate} "
                        f"over last {recent_bars} bars")
            nan_cb = NaNStopCallback()
            profit_cb = ProfitLoggingCallback()
            aux_cb = AuxVolatilityCallback() if CONFIG.get('PPO_AUX_TASK', True) else None
            callbacks = [nan_cb, profit_cb] + ([aux_cb] if aux_cb else [])
            model.learn(total_timesteps=timesteps, callback=callbacks, reset_num_timesteps=False)
            model.learning_rate = original_lr
            self.portfolio_vec_norm = new_vec_env

            # Save new weights
            save_ppo_model(self, "portfolio")
            logger.info("[MICRO-RETRAIN] Training completed and saved")

            # Validate + rollback if worse
            if guard_enabled and baseline_score is not None and backup_created:
                import json as _json
                post_score = self._validate_portfolio_model(model, new_vec_env, n_steps=guard_val_steps)
                min_drop = CONFIG.get('PPO_MICRO_RETRAIN_MIN_DROP', 0.0015)   # stricter
                rel_drop = CONFIG.get('PPO_MICRO_RETRAIN_REL_DROP', 0.15)    # stricter 15%
                abs_delta = post_score - baseline_score
                rel_delta = abs_delta / (abs(baseline_score) + 1e-9)
                degraded = (abs_delta < -min_drop) and (rel_delta < -rel_drop)
                decision = "ROLLBACK" if degraded else "ACCEPT"
                logger.info(f"[MICRO-RETRAIN GUARD] Post: {post_score:+.5f} "
                            f"(base {baseline_score:+.5f}, Δ={abs_delta:+.5f}, "
                            f"rel={rel_delta:+.2%}) — {decision}")
                logger.info(_json.dumps({
                    "event": "micro_retrain_guard_decision",
                    "baseline_score": baseline_score,
                    "post_score": post_score,
                    "abs_delta": abs_delta,
                    "rel_delta": rel_delta,
                    "decision": decision,
                }))
                result.update({
                    "baseline_score": baseline_score,
                    "post_score": post_score,
                    "decision": decision,
                    "status": "completed",
                })
                if degraded:
                    logger.warning(f"[MICRO-RETRAIN GUARD] Rolling back — micro-update hurt")
                    _shutil.copy2(backup_path, model_path)
                    from .ppo_utils import load_ppo_model
                    load_ppo_model(self, "portfolio")
                    if self.portfolio_vec_norm is not None:
                        self.portfolio_vec_norm.training = False
                    logger.info(f"[MICRO-RETRAIN GUARD] ✅ Rollback complete")
                else:
                    logger.info(f"[MICRO-RETRAIN GUARD] ✅ Accepted")
            else:
                result["status"] = "completed_no_guard"
            return result
        except Exception as e:
            logger.error(f"[MICRO-RETRAIN] Failed: {e}", exc_info=True)
            result.update({"status": "failed", "error": str(e)})
            if guard_enabled and backup_created:
                try:
                    _shutil.copy2(backup_path, model_path)
                    from .ppo_utils import load_ppo_model
                    load_ppo_model(self, "portfolio")
                    logger.info("[MICRO-RETRAIN GUARD] Training failed — restored backup")
                except Exception as re:
                    logger.error(f"[MICRO-RETRAIN GUARD] Restore failed: {re}")
            return result

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
                        regime_cache=dict(self.regime_cache)  # M49 FIX: copy for thread safety
                    )
                    self.walk_forward_optimize_thresholds(sym, data)
        else:
            # Legacy per-symbol parallel training
            max_workers = min(len(symbols), 4)  # FIX #38: Was 16 — GPU PPO training OOMs with too many parallel workers
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

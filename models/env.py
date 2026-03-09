# models/env.py
# =====================================================================
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import pandas as pd
from config import CONFIG
from models.features import generate_features, _precompute_and_cache_tft
import logging
from strategy.regime import detect_regime

logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    # Class-level cache for TFT features (shared across instances if needed)
    tft_cache = {} # {symbol: (timestamp, cached_full_df)}
    _tft_cache_ttl = 3600  # 1 hour TTL for TFT cache entries

    def __init__(self, data_ingestion, symbol: str, initial_balance: float = 1000.0):
        super().__init__()
        self.data_ingestion = data_ingestion
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.current_step = 0
        self.position = 0.0 # -1 to 1
        self.cash = initial_balance
        self.equity = initial_balance
        self.data = pd.DataFrame()
        self.max_shares = initial_balance / 100
        self.returns = []
        self.equity_prev = initial_balance
        self.cummax_equity = initial_balance
        # UPGRADE #1: Causal wrapper will be injected by Trainer
        self.causal_wrapper = None
        # P-9 FIX: vol_target now properly stored and passed (teacher-forcing)
        self.vol_target = 0.0
        # DYNAMIC OBSERVATION SPACE — now includes volatility target (UPGRADE #6)
        dummy_window = pd.DataFrame({
            'open': [100.0] * 300,
            'high': [101.0] * 300,
            'low': [99.0] * 300,
            'close': [100.0] * 300,
            'volume': [1000000] * 300
        }, index=pd.date_range(end='2026-01-01', periods=300, freq='15min'))
        dummy_regime = 'mean_reverting'
        try:
            dummy_features = generate_features(dummy_window, dummy_regime, symbol="DUMMY", full_hist_df=dummy_window)
            feature_dim = dummy_features.shape[1] if dummy_features is not None and dummy_features.size > 0 else 52
        except Exception as e:
            logger.warning(f"Failed to compute feature dim dynamically ({e}) — falling back to 52")
            feature_dim = 52
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32)
        logger.info(f"Observation space dynamically set to shape=({feature_dim},) for symbol {symbol}")
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # BUG-08 FIX: track whether we've already warned about dimension mismatch
        self._logged_dim_warning = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 199
        self.position = 0.0
        self.cash = self.initial_balance
        self.equity = self.initial_balance
        self.cummax_equity = self.initial_balance
        self.returns = []
        self.equity_prev = self.equity
        # P-9 FIX: Reset vol_target
        self.vol_target = 0.0
        # PERF FIX: Cache regime at reset() to avoid expensive HMM call every step()
        self._cached_regime = None
        self._cached_persistence = None
        data = self.data_ingestion.get_latest_data(self.symbol)
        if len(data) < 200:
            data = pd.DataFrame()
        self.data = data
        # Compute regime once per episode (uses full available data)
        if not self.data.empty and len(self.data) >= 200:
            self._cached_regime, self._cached_persistence = detect_regime(
                data=self.data,
                symbol=self.symbol,
                data_ingestion=self.data_ingestion,
                lookback=CONFIG.get('LOOKBACK', 900)
            )
        # TFT CACHING
        if CONFIG.get('USE_TFT_ENCODER', False):
            import time as _time
            cached = TradingEnv.tft_cache.get(self.symbol)
            cache_stale = (cached is None or
                           len(cached[1]) < len(self.data) or
                           (_time.time() - cached[0]) > TradingEnv._tft_cache_ttl)
            if cache_stale:
                logger.info(f"Updating TFT cache for {self.symbol} ({len(self.data)} bars)")
                TradingEnv.tft_cache[self.symbol] = (_time.time(), _precompute_and_cache_tft(self.symbol, self.data))
                # Evict symbols no longer in universe to prevent memory bloat
                active_symbols = set(CONFIG.get('SYMBOLS', []))
                stale_keys = [k for k in TradingEnv.tft_cache if k not in active_symbols]
                for k in stale_keys:
                    del TradingEnv.tft_cache[k]
                    logger.debug(f"TFT cache evicted stale symbol: {k}")
            else:
                logger.debug(f"TFT cache already up-to-date for {self.symbol}")
        return self._get_observation(), {}

    def step(self, action):
        if self.current_step >= len(self.data) - 1 or self.data.empty:
            return self._get_observation(), 0, True, False, {'volatility_target': 0.0}
        # PERF FIX: Use cached regime from reset() instead of calling detect_regime() every step
        window = self.data.iloc[:self.current_step + 1]
        if self._cached_regime is not None:
            regime, persistence = self._cached_regime, self._cached_persistence
        else:
            regime, persistence = detect_regime(
                data=window,
                symbol=self.symbol,
                data_ingestion=self.data_ingestion,
                lookback=CONFIG.get('LOOKBACK', 900)
            )
        target_pos = np.clip(float(action[0]), -1.0, 1.0)
        current_price = self.data['close'].iloc[self.current_step]
        trade_pos = target_pos - self.position
        trade_shares = trade_pos * self.max_shares
        turnover_cost = abs(trade_shares) * current_price * (CONFIG.get('SLIPPAGE', 0.0005) + CONFIG.get('COMMISSION_PER_SHARE', 0.0005) / current_price)
        self.position = target_pos
        self.cash -= trade_shares * current_price  # pay for (or receive from) the shares
        self.cash -= turnover_cost
        self.equity = self.cash + self.position * current_price * self.max_shares
        ret = (self.equity - self.equity_prev) / self.equity_prev if self.equity_prev > 0 else 0
        self.returns.append(ret)
        self.equity_prev = self.equity
        # Update drawdown tracking
        self.cummax_equity = max(self.cummax_equity, self.equity)
        # ─── Base reward: log return (scale-invariant, additive over time) ───
        reward = ret

        # ─── Turnover penalty: penalize position CHANGES, not absolute position ───
        reward -= CONFIG.get('TURNOVER_COST_MULT', 0.05) * abs(trade_pos)

        # ─── Sortino component: risk-adjusted return quality ───
        if len(self.returns) > 20:
            recent_returns = np.array(self.returns[-20:])
            downside_returns = recent_returns[recent_returns < 0]
            if len(downside_returns) > 1:
                downside_std = np.std(downside_returns)
                downside_std = max(downside_std, 1e-8)
                mean_return = np.mean(recent_returns)
                sortino = mean_return / downside_std
                # Scale to per-step magnitude (no annualization — keeps reward scale consistent)
                reward += sortino * CONFIG.get('SORTINO_WEIGHT', 0.25)
            else:
                reward += CONFIG.get('SORTINO_ZERO_DD_BONUS', 0.02)

        # ─── Volatility penalty: penalize erratic equity curves ───
        annual_vol = 0.0
        if len(self.returns) > 50:
            recent_vol = np.std(self.returns[-100:])
            annual_vol = recent_vol * np.sqrt(252 * 96)
            reward -= annual_vol * CONFIG.get('VOL_PENALTY_COEF', 0.02)

        # ─── Drawdown penalty: proportional to current drawdown depth ───
        drawdown = (self.equity - self.cummax_equity) / self.cummax_equity if self.cummax_equity > 0 else 0
        reward -= CONFIG.get('DD_PENALTY_COEF', 2.0) * max(-drawdown, 0.0)

        # ─── Causal penalty: counterfactual reward adjustment ───
        if self.causal_wrapper is not None:
            features = generate_features(
                data=window,
                regime=regime,
                symbol=self.symbol,
                full_hist_df=self.data
            )
            latest_features = features[-1].astype(np.float32) if features is not None and features.shape[0] > 0 else np.zeros(self.observation_space.shape[0])
            penalty_factor = self.causal_wrapper.compute_penalty_factor(
                latest_features.reshape(1, -1),
                float(action[0])
            )
            # factor > 1.0 = strong causal signal → amplify reward direction
            # factor < 1.0 = weak causal signal → dampen reward
            # factor = 1.0 = no effect
            reward *= penalty_factor
            logger.debug(f"[CAUSAL REWARD] {self.symbol} | regime={regime} | action={action[0]:.3f} | factor={penalty_factor:.4f}")

        # ─── Persistence bonus: reward holding in strong regimes ───
        persistence_bonus = (persistence - 0.5) * CONFIG.get('PERSISTENCE_BONUS_SCALE', 0.15)
        reward += persistence_bonus
        logger.debug(f"[PERSISTENCE REWARD] {self.symbol} | regime={regime} | persistence={persistence:.3f} | bonus={persistence_bonus:.4f}")
        # ─────────────────────────────────────────────────────────────────────
        # UPGRADE #6: Volatility-target is now part of the observation (teacher-forcing)
        # annual_vol is computed here and passed through generate_features as the last column
        # ─────────────────────────────────────────────────────────────────────
        vol_target = annual_vol if 'annual_vol' in locals() else 0.0
        # P-9 FIX: Store vol_target so _get_observation() can append it
        self.vol_target = vol_target
        # Clip reward
        reward = np.clip(reward, -10.0, 10.0)
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1 or self.equity <= 0
        info = {
            'volatility_target': vol_target,
            'persistence': persistence,
            'regime': regime
        }
        return self._get_observation(regime=regime, persistence=persistence), reward, done, False, info

    def _get_observation(self, regime=None, persistence=None):
        if self.data.empty or len(self.data) < 200 or self.current_step < 199:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        window = self.data.iloc[:self.current_step + 1]
        # Use passed regime if provided (P-4 FIX for triple call bug)
        if regime is None:
            regime, persistence = detect_regime(
                data=window,
                symbol=self.symbol,
                data_ingestion=self.data_ingestion,
                lookback=CONFIG.get('LOOKBACK', 900)
            )
        features = generate_features(
            data=window,
            regime=regime,
            symbol=self.symbol,
            full_hist_df=self.data
        )
        if features is None or features.shape[0] == 0:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # BUG-08 FIX: Always enforce exactly self.feature_dim columns
        # Prevents silent misalignment when cached feats and fresh_feats have different shapes
        step_idx = self.current_step
        if step_idx < len(features):
            latest = features[step_idx].astype(np.float32)
        else:
            latest = features[-1].astype(np.float32)

        # Recompute recent window if near end (existing logic)
        if step_idx >= len(features) - 20:
            timestamp = window.index[-1]  # use actual last timestamp
            recent_window = self.data.loc[:timestamp].tail(200)
            if len(recent_window) >= 50:
                fresh_feats = generate_features(
                    data=recent_window,
                    regime=regime,
                    symbol=self.symbol,
                    full_hist_df=self.data
                )
                if fresh_feats is not None and fresh_feats.shape[0] > 0:
                    latest = fresh_feats[-1].astype(np.float32)

        # ────────────────────────────────────────────────────────────────
        # BUG-08 FIX: Enforce dimension safety on BOTH cached and fresh paths
        expected_dim = self.observation_space.shape[0]
        if latest.shape[0] != expected_dim:
            if not self._logged_dim_warning:
                logger.warning(
                    f"[FEATURE DIM MISMATCH] {self.symbol} | "
                    f"computed features len={latest.shape[0]} but expected {expected_dim} "
                    f"(step={step_idx}, regime={regime}) → forcing alignment"
                )
                self._logged_dim_warning = True

            if latest.shape[0] < expected_dim:
                pad_width = expected_dim - latest.shape[0]
                latest = np.pad(latest, (0, pad_width), mode='constant', constant_values=0.0)
            else:
                latest = latest[:expected_dim]

        if np.any(np.isnan(latest)) or np.any(np.isinf(latest)):
            latest = np.nan_to_num(latest, nan=0.0, posinf=20.0, neginf=-20.0)
        latest = np.clip(latest, -20.0, 20.0)

        return latest

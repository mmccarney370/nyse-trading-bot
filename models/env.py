# models/env.py
# =====================================================================
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import pandas as pd
from config import CONFIG
from models.features import generate_features
import logging
from strategy.regime import detect_regime

logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    # H17 FIX: Cache feature dimension at class level so all instances share the same obs space.
    # Computed once on first instantiation, never changes (generate_features output is deterministic).
    _cached_feature_dim = None

    def __init__(self, data_ingestion, symbol: str, initial_balance: float = 1000.0):
        super().__init__()
        self.data_ingestion = data_ingestion
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.current_step = 0
        self.position = 0.0 # -1 to 1 (fractional target, used for turnover penalty)
        self.cash = initial_balance
        self.equity = initial_balance
        self.data = pd.DataFrame()
        self._shares_held = 0.0  # actual shares held (positive=long, negative=short)
        self.returns = []
        self.equity_prev = initial_balance
        self.cummax_equity = initial_balance
        self.causal_wrapper = None
        self.vol_target = 0.0
        # H17 FIX: Compute feature dim once and cache at class level
        if TradingEnv._cached_feature_dim is None:
            dummy_window = pd.DataFrame({
                'open': [100.0] * 300,
                'high': [101.0] * 300,
                'low': [99.0] * 300,
                'close': [100.0] * 300,
                'volume': [1000000] * 300
            }, index=pd.date_range(end='2026-01-01', periods=300, freq='15min'))
            try:
                dummy_features = generate_features(dummy_window, 'mean_reverting', symbol="DUMMY", full_hist_df=dummy_window)
                TradingEnv._cached_feature_dim = dummy_features.shape[1] if dummy_features is not None and dummy_features.size > 0 else 53
            except Exception as e:
                logger.warning(f"Failed to compute feature dim dynamically ({e}) — falling back to 53")
                TradingEnv._cached_feature_dim = 53
        feature_dim = TradingEnv._cached_feature_dim
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32)
        logger.info(f"Observation space dynamically set to shape=({feature_dim},) for symbol {symbol}")
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # BUG-08 FIX: track whether we've already warned about dimension mismatch
        self._logged_dim_warning = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 199
        self.position = 0.0
        self._shares_held = 0.0
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
        # TFT features are cached on-disk by _precompute_and_cache_tft (called lazily from
        # generate_features when USE_TFT_ENCODER is True). No separate in-memory cache needed.
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
        # C1 FIX: Track shares directly instead of circular max_shares formula.
        # target_pos in [-1, 1] maps to fraction of current equity deployed.
        # desired_value = target_pos * equity, desired_shares = desired_value / price.
        if current_price > 0 and self.equity > 0:
            desired_shares = (target_pos * self.equity) / current_price
        else:
            desired_shares = 0.0
        trade_shares = desired_shares - self._shares_held
        trade_pos = target_pos - self.position
        turnover_cost = abs(trade_shares * current_price) * (CONFIG.get('SLIPPAGE', 0.0005) + CONFIG.get('COMMISSION_PER_SHARE', 0.0005) / max(current_price, 0.01))
        self._shares_held = desired_shares
        self.position = target_pos
        self.cash -= trade_shares * current_price
        self.cash -= turnover_cost
        # Equity = cash + market value of held shares (price moves between steps create P&L)
        self.equity = self.cash + self._shares_held * current_price
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
                # FIX #44: Clip sortino to prevent reward explosion when downside_std is tiny
                sortino = np.clip(sortino, -5.0, 5.0)
                # Scale to per-step magnitude (no annualization — keeps reward scale consistent)
                reward += sortino * CONFIG.get('SORTINO_WEIGHT', 0.25)
            else:
                reward += CONFIG.get('SORTINO_ZERO_DD_BONUS', 0.02)

        # ─── Volatility penalty: penalize erratic equity curves ───
        annual_vol = 0.0
        if len(self.returns) > 50:
            recent_vol = np.std(self.returns[-100:])
            # FIX #41: Use CONFIG key for annualization factor (252 trading days * 26 fifteen-min bars/day)
            annual_vol = recent_vol * np.sqrt(CONFIG.get('ANNUALIZATION_FACTOR', 252 * 26))
            reward -= annual_vol * CONFIG.get('VOL_PENALTY_COEF', 0.02)

        # ─── Drawdown penalty: proportional to current drawdown depth ───
        drawdown = (self.equity - self.cummax_equity) / self.cummax_equity if self.cummax_equity > 0 else 0
        reward -= CONFIG.get('DD_PENALTY_COEF', 2.0) * max(-drawdown, 0.0)

        # ─── Opportunity cost (Apr-19 audit fix) ───
        # Idling at |position|≈0 is strictly rewarded relative to small,
        # turnover-penalised rebalances. Small penalty for being flat breaks
        # the asymmetry so the policy cannot learn to sit out calm regimes
        # without tradeoff.
        opp_cost_coef = CONFIG.get('OPPORTUNITY_COST_COEF', 0.0001)
        if opp_cost_coef > 0:
            idle_fraction = max(0.0, 1.0 - abs(self.position))
            reward -= opp_cost_coef * idle_fraction

        # ─── Causal penalty: counterfactual reward adjustment ───
        # Features are computed for causal penalty but NOT cached for _get_observation(),
        # because current_step will be incremented before _get_observation() runs,
        # making pre-increment features 1 bar stale.
        self._step_cached_features = None
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
            reward *= penalty_factor
            logger.debug(f"[CAUSAL REWARD] {self.symbol} | regime={regime} | action={action[0]:.3f} | factor={penalty_factor:.4f}")

        # ─── Persistence bonus: reward holding (low turnover) in strong regimes ───
        # Scale by (1 - |trade_pos|) so the bonus rewards staying in position, not changing.
        # A constant bonus regardless of action just shifts the baseline without teaching anything.
        hold_factor = 1.0 - min(abs(trade_pos), 1.0)  # 1.0 when holding, 0.0 when fully flipping
        persistence_bonus = (persistence - 0.5) * CONFIG.get('PERSISTENCE_BONUS_SCALE', 0.15) * hold_factor
        reward += persistence_bonus
        logger.debug(f"[PERSISTENCE REWARD] {self.symbol} | regime={regime} | persistence={persistence:.3f} | hold={hold_factor:.2f} | bonus={persistence_bonus:.4f}")
        # ─────────────────────────────────────────────────────────────────────
        # UPGRADE #6: Volatility-target is now part of the observation (teacher-forcing)
        # annual_vol is computed here and passed through generate_features as the last column
        # ─────────────────────────────────────────────────────────────────────
        vol_target = annual_vol  # L28 FIX: always in scope (initialized at line 148)
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
        # FIX #43: Reuse features cached by step() if available (avoids duplicate generate_features call)
        cached_feats = getattr(self, '_step_cached_features', None)
        if cached_feats is not None:
            features = cached_feats
            self._step_cached_features = None  # Consume cache (one-time use)
        else:
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
        # NOTE (FIX #40): When step_idx is near len(features)-20, we switch from using the
        # pre-computed feature matrix to a fresh 200-bar window. This can cause a small
        # observation discontinuity because the fresh window uses different lookback context.
        # Known limitation — accepted because the alternative (recomputing features for all
        # steps) is too expensive, and the PPO policy learns to handle minor input shifts.
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

# portfolio_env.py
"""
Portfolio-level Gymnasium environment for multi-asset PPO training.
Key features:
- Observation: Concatenated latest features from all active symbols + portfolio-level aggregates
- Action: Box(low=-2.0, high=2.0, shape=(n_symbols,)) → signed leverage target per symbol
- Reward: Portfolio PnL with volatility & drawdown penalties
- Auxiliary volatility target (for aux head)
- Fully compatible with existing generate_features, detect_regime, TFT caching
- UPGRADE #4 (Feb 20 2026): Now handles detect_regime returning (regime, persistence) tuple
"""
# ────────────────────────────────────────────────────────────────────────────────────────────────
# =====================================================================
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import pandas as pd
from config import CONFIG
from models.features import generate_features
from strategy.regime import detect_regime
import logging
import json
import os
from collections import deque

logger = logging.getLogger(__name__)
REGIME_CACHE_FILE = "regime_cache.json"

class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data_dict: dict[str, pd.DataFrame],
        symbols: list[str],
        initial_balance: float = 100_000.0,
        max_leverage: float = 3.0,
    ):
        super().__init__()
        self.symbols = symbols
        self.n_symbols = len(symbols)
        self.data_dict = {sym: df.copy() for sym, df in data_dict.items()}
        self.initial_balance = initial_balance
        self.max_leverage = CONFIG.get('MAX_LEVERAGE', max_leverage)
        self.max_episode_steps = CONFIG.get('MAX_EPISODE_STEPS', 2048)
        self.action_space = Box(low=-2.0, high=2.0, shape=(self.n_symbols,), dtype=np.float32)
  
        feature_dims = []
        for sym in self.symbols:
            window = self.data_dict[sym].iloc[:min(200, len(self.data_dict[sym]))]
            if len(window) < 50:
                continue
            dummy_features = generate_features(
                data=window,
                regime='trending',
                symbol=sym,
                full_hist_df=self.data_dict[sym]
            )
            if dummy_features is not None and dummy_features.shape[1] > 0:
                feature_dims.append(dummy_features.shape[1])
        self.feature_dim = max(feature_dims) if feature_dims else 60
        self.portfolio_extra_dim = 4 + self.n_symbols
        total_dim = self.n_symbols * self.feature_dim + self.portfolio_extra_dim
        self.observation_space = Box(low=-20.0, high=20.0, shape=(total_dim,), dtype=np.float32)
        self.timeline = self._build_common_timeline()
        self.regime_cache = self._load_regime_cache()
        logger.info(f"Precomputing features and regimes for {self.n_symbols} symbols using shared cache")
        self.precomputed_features = {}
        self.precomputed_regimes = {}
        min_valid_steps = float('inf')
        for sym in self.symbols:
            full_data = self.data_dict[sym]
            n_bars = len(full_data)
      
            regimes = []
            cached_regime = None
            if sym in self.regime_cache:
                value = self.regime_cache[sym]
                if isinstance(value, list):
                    cached_regime = value[0]
                else:
                    cached_regime = value
            else:
                cached_regime = 'trending'
      
            for i in range(n_bars):
                regimes.append(cached_regime)
      
            self.precomputed_regimes[sym] = np.array(regimes)
      
            last_regime_tuple = regimes[-1] if regimes else cached_regime or 'trending'
            last_regime = last_regime_tuple[0] if isinstance(last_regime_tuple, tuple) else last_regime_tuple
            features = generate_features(
                data=full_data,
                regime=last_regime,
                symbol=sym,
                full_hist_df=full_data
            )
            if features is None or features.shape[0] == 0:
                features = np.zeros((n_bars, self.feature_dim))
            else:
                if features.shape[0] < n_bars:
                    pad_rows = n_bars - features.shape[0]
                    pad = np.zeros((pad_rows, features.shape[1]))
                    features = np.vstack([pad, features])
                if features.shape[1] < self.feature_dim:
                    pad_cols = self.feature_dim - features.shape[1]
                    features = np.pad(features, ((0, 0), (0, pad_cols)))
                elif features.shape[1] > self.feature_dim:
                    features = features[:, :self.feature_dim]
            self.precomputed_features[sym] = features.astype(np.float32)
            min_valid_steps = min(min_valid_steps, n_bars)
  
        self.timeline = self.timeline[-min_valid_steps:]
        self.current_step = 0
        self.episode_start = 0
        logger.info(f"Precomputation complete — valid steps: {len(self.timeline)} | max_episode_steps: {self.max_episode_steps}")
  
        self.last_weights = np.zeros(self.n_symbols, dtype=np.float32)
        self.weight_history = deque(maxlen=100)
        self.reset()

    def _load_regime_cache(self):
        if os.path.exists(REGIME_CACHE_FILE):
            try:
                with open(REGIME_CACHE_FILE, 'r') as f:
                    data = json.load(f)
                logger.info(f"PortfolioEnv loaded shared regime cache with {len(data)} symbols")
                return data
            except Exception as e:
                logger.warning(f"Failed to load regime cache in PortfolioEnv: {e}")
        return {}

    def _build_common_timeline(self):
        aligned = None
        for sym, df in self.data_dict.items():
            df = df[['close']].copy()
            df.columns = [f"{sym}_close"]
            if aligned is None:
                aligned = df
            else:
                aligned = aligned.join(df, how='outer')
        aligned = aligned.ffill().bfill().dropna()
        return aligned.index

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.weights = np.zeros(self.n_symbols, dtype=np.float32)
        self.last_weights = np.zeros(self.n_symbols, dtype=np.float32)
        self.weight_history = deque(maxlen=100)
        self.weight_history.append(self.weights.copy())
        # Randomize start position so the agent sees different data each episode
        max_start = max(0, len(self.timeline) - self.max_episode_steps - 1)
        self.current_step = self.np_random.integers(0, max_start + 1) if max_start > 0 else 0
        self.episode_start = self.current_step
        self.cumulative_pnl = 0.0
        self.peak_equity = self.initial_balance
        return self._get_observation(), {}

    def step(self, action):
        if self.current_step >= len(self.timeline) - 1:
            return self._get_observation(), 0.0, True, False, {'volatility_target': 0.0, 'portfolio_return': 0.0, 'drawdown': 0.0}

        raw_targets = np.clip(action, -2.0, 2.0)
        abs_sum = np.sum(np.abs(raw_targets))
        if abs_sum > self.max_leverage:
            raw_targets = raw_targets / abs_sum * self.max_leverage
        target_weights = raw_targets.astype(np.float32)

        timestamp = self.timeline[self.current_step]
        returns = np.zeros(self.n_symbols)
        for i, sym in enumerate(self.symbols):
            if timestamp in self.data_dict[sym].index:
                close_series = self.data_dict[sym]['close']
                prev_close = close_series.asof(timestamp - pd.Timedelta(minutes=15))
                curr_close = close_series.asof(timestamp)
                # Guard against NaN from asof() on index gaps
                if pd.notna(prev_close) and pd.notna(curr_close) and prev_close > 0:
                    returns[i] = (curr_close - prev_close) / prev_close

        # =====================================================================
        # BUG-01 FIX: Reward must be computed using the weights that were
        # actually held during this bar (self.weights), NOT the new target.
        # We update self.weights ONLY AFTER the return is calculated.
        # =====================================================================
        portfolio_return = np.dot(self.weights, returns)   # ← OLD weights (what we actually held)

        self.cumulative_pnl += portfolio_return
        self.balance *= (1 + portfolio_return)
        self.equity = self.balance

        self.weight_history.append(self.weights.copy())
        self.last_weights = self.weights.copy()

        self.weights = target_weights   # ← Now apply new weights for NEXT step

        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = (self.equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        recent_rets = []
        lookback = min(60, self.current_step + 1, len(self.weight_history))
        for offset in range(1, lookback + 1):
            past_step = self.current_step - offset + 1
            if past_step < 0:
                break
            past_ret = self._compute_portfolio_return_at_step(past_step)
            recent_rets.append(past_ret)

        # FIX: Use 252 trading days * 96 fifteen-min bars per day (not 365*24*4 which assumes 24/7 trading)
        annual_vol = np.std(recent_rets) * np.sqrt(252 * 96) if len(recent_rets) > 10 else 0.0

        reward = portfolio_return
        reward -= CONFIG.get('VOL_PENALTY_COEF', 0.03) * annual_vol
        reward -= CONFIG.get('DD_PENALTY_COEF', 2.0) * max(-drawdown, 0.0)
        reward = np.clip(reward, -10.0, 10.0)

        self.current_step += 1
        steps_in_episode = self.current_step - self.episode_start
        at_data_end = self.current_step >= len(self.timeline) - 1
        truncated = steps_in_episode >= self.max_episode_steps
        done = at_data_end or self.equity <= 0 or truncated

        info = {
            'volatility_target': annual_vol,
            'portfolio_return': portfolio_return,
            'drawdown': drawdown,
        }
        # Gymnasium convention: done=terminal (bankrupt/data end), truncated=time limit
        terminal = at_data_end or self.equity <= 0
        return self._get_observation(), reward, terminal, truncated, info

    def _compute_portfolio_return_at_step(self, step_idx):
        timestamp = self.timeline[step_idx]
        returns = np.zeros(self.n_symbols)
        for i, sym in enumerate(self.symbols):
            if timestamp in self.data_dict[sym].index:
                close_series = self.data_dict[sym]['close']
                prev_close = close_series.asof(timestamp - pd.Timedelta(minutes=15))
                curr_close = close_series.asof(timestamp)
                if pd.notna(prev_close) and pd.notna(curr_close) and prev_close > 0:
                    returns[i] = (curr_close - prev_close) / prev_close

        # Deque-safe index: clamp to valid range within the deque
        history_idx = len(self.weight_history) - 1 - (self.current_step - step_idx)
        if 0 <= history_idx < len(self.weight_history):
            past_weights = self.weight_history[history_idx]
        else:
            past_weights = self.last_weights
        return np.dot(past_weights, returns)

    def _get_observation(self):
        if self.current_step >= len(self.timeline):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        step_idx = self.current_step
        all_features = []
        for sym in self.symbols:
            feats = self.precomputed_features[sym]
            if step_idx < len(feats):
                latest = feats[step_idx]
            else:
                latest = feats[-1]

            if step_idx >= len(feats) - 20:
                timestamp = self.timeline[step_idx]
                window = self.data_dict[sym].loc[:timestamp].tail(200)
                if len(window) >= 50:
                    if sym in self.regime_cache:
                        value = self.regime_cache[sym]
                        regime = value[0] if isinstance(value, list) else value
                    else:
                        recent_return = (window['close'].iloc[-1] - window['close'].iloc[-20]) / window['close'].iloc[-20]
                        regime = 'trending' if abs(recent_return) > 0.015 else 'mean_reverting'
                    fresh_feats = generate_features(
                        data=window,
                        regime=regime,
                        symbol=sym,
                        full_hist_df=self.data_dict[sym]
                    )
                    if fresh_feats is not None and fresh_feats.shape[0] > 0:
                        latest = fresh_feats[-1]

            if len(latest) < self.feature_dim:
                latest = np.pad(latest, (0, self.feature_dim - len(latest)))
            elif len(latest) > self.feature_dim:
                latest = latest[:self.feature_dim]
            all_features.append(latest)

        per_symbol_obs = np.concatenate(all_features)

        equity_norm = np.log(self.equity / self.initial_balance + 1e-8)
        drawdown = (self.equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        gross_exposure = np.sum(np.abs(self.weights))
        concentration = np.std(self.weights)

        portfolio_extra = np.concatenate([
            np.array([equity_norm, drawdown], dtype=np.float32),
            self.weights,
            np.array([gross_exposure, concentration], dtype=np.float32)
        ])

        obs = np.concatenate([per_symbol_obs, portfolio_extra])
        obs = np.nan_to_num(obs, nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)
        obs = np.clip(obs, -20.0, 20.0)
        return obs

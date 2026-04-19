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
        max_leverage: float = CONFIG.get('MAX_LEVERAGE', 2.0),
    ):
        super().__init__()
        self.symbols = symbols
        self.n_symbols = len(symbols)
        self.data_dict = {sym: df.copy() for sym, df in data_dict.items()}
        self.initial_balance = initial_balance
        self.max_leverage = max_leverage
        self.max_episode_steps = CONFIG.get('MAX_EPISODE_STEPS', 2048)
        self.action_space = Box(low=-2.0, high=2.0, shape=(self.n_symbols,), dtype=np.float32)
  
        feature_dims = []
        for sym in self.symbols:
            window = self.data_dict[sym].iloc[:min(500, len(self.data_dict[sym]))]
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
        # FIX #55: Feature dim mismatch is zero-padded below (lines ~165-169). This is acceptable
        # and common in multi-asset envs where symbols have different feature counts due to
        # varying data availability or indicator applicability.
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
            close_vals = full_data['close'].values if 'close' in full_data.columns else None
            # Regime heuristic: simple 50-bar return threshold, recomputed every 100 bars.
            # DESIGN NOTE: This is intentionally a fast heuristic, NOT the full HMM ensemble
            # (detect_regime). Calling the HMM ensemble per symbol during env __init__ would
            # be prohibitively slow (multiple HMM fits per symbol). The heuristic provides a
            # rough regime signal sufficient for training-time feature generation.
            current_regime = 'trending_up'  # default until enough data
            for i in range(n_bars):
                if i >= 50 and i % 100 == 0 and close_vals is not None:
                    lookback_close = close_vals[i - 50]
                    if lookback_close > 0:
                        ret_50 = (close_vals[i] - lookback_close) / lookback_close
                        if abs(ret_50) > 0.015:
                            current_regime = 'trending_up' if ret_50 > 0 else 'trending_down'
                        else:
                            current_regime = 'mean_reverting'
                regimes.append(current_regime)

            self.precomputed_regimes[sym] = np.array(regimes)
      
            # Generate features once on the full data. The regime flag is just one column
            # in the 53-feature vector; chunking by regime created small chunks that failed
            # the 100-bar minimum in generate_features(). Use dominant regime instead.
            regime_arr = self.precomputed_regimes[sym]
            unique, counts = np.unique(regime_arr, return_counts=True)
            dominant_regime = unique[counts.argmax()]
            features = generate_features(
                data=full_data, regime=dominant_regime,
                symbol=sym, full_hist_df=full_data
            )
            if features is None or (hasattr(features, 'shape') and features.shape[0] == 0):
                features = np.zeros((n_bars, self.feature_dim))
            else:
                if features.shape[0] > n_bars:
                    features = features[-n_bars:]
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
        # Slice precomputed features to match truncated timeline
        for sym in self.symbols:
            feats = self.precomputed_features[sym]
            if feats.shape[0] > min_valid_steps:
                self.precomputed_features[sym] = feats[-min_valid_steps:]
        self.current_step = 0
        self.episode_start = 0
        logger.info(f"Precomputation complete — valid steps: {len(self.timeline)} | max_episode_steps: {self.max_episode_steps}")
  
        # Precompute per-symbol returns aligned to the common timeline to avoid
        # O(n_symbols * lookback) get_indexer calls per step.
        # Shape: [len(timeline), n_symbols] — returns[t, i] is the return of symbol i at bar t.
        self._precomputed_returns = np.zeros((len(self.timeline), self.n_symbols), dtype=np.float32)
        for i, sym in enumerate(self.symbols):
            close_series = self.data_dict[sym]['close']
            # Align close prices to the common timeline using forward-fill
            aligned_close = close_series.reindex(self.timeline, method='pad')
            # NOTE #18: Bars before a symbol's first data point get NaN→0 via fillna(0.0).
            # This is correct: no data = no return. Zero returns mean the portfolio simply
            # has no exposure to that symbol for those bars.
            n_leading_nan = aligned_close.isna().sum()
            if n_leading_nan > 0:
                logger.debug(f"[PRECOMPUTE] {sym}: {n_leading_nan}/{len(self.timeline)} leading NaN bars "
                             f"(data starts later than timeline) — treated as zero return")
            sym_returns = aligned_close.pct_change().fillna(0.0).values
            self._precomputed_returns[:, i] = sym_returns.astype(np.float32)

        self.last_weights = np.zeros(self.n_symbols, dtype=np.float32)
        self.weight_history = deque(maxlen=max(200, self.max_episode_steps))
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
        self.weight_history = deque(maxlen=max(200, self.max_episode_steps))
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

        # Use precomputed returns array (O(1) lookup instead of O(n_symbols) get_indexer calls)
        returns = self._precomputed_returns[self.current_step]

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

        # FIX: 6.5h trading day = 26 fifteen-min bars, not 96
        annual_vol = np.std(recent_rets) * np.sqrt(252 * 26) if len(recent_rets) > 10 else 0.0

        # ─── Base reward: portfolio return ───
        reward = portfolio_return

        # ─── Turnover penalty ───
        turnover = np.sum(np.abs(target_weights - self.last_weights))
        reward -= CONFIG.get('TURNOVER_COST_MULT', 0.05) * turnover

        # ─── Sortino component (matches env.py) ───
        if len(recent_rets) > 20:
            recent_arr = np.array(recent_rets[-20:])
            downside = recent_arr[recent_arr < 0]
            if len(downside) > 1:
                ds_std = max(np.std(downside), 1e-8)
                sortino = np.mean(recent_arr) / ds_std
                # FIX #20: Clamp Sortino before adding to reward — tiny downside std can produce millions
                sortino = np.clip(sortino, -5.0, 5.0)
                reward += sortino * CONFIG.get('SORTINO_WEIGHT', 0.25)
            else:
                reward += CONFIG.get('SORTINO_ZERO_DD_BONUS', 0.02)

        # ─── Volatility penalty ───
        reward -= CONFIG.get('VOL_PENALTY_COEF', 0.02) * annual_vol

        # ─── Drawdown penalty ───
        reward -= CONFIG.get('DD_PENALTY_COEF', 2.0) * max(-drawdown, 0.0)

        # ─── Opportunity cost (Apr-19 audit fix) ───
        # Holding a near-zero position is NOT free. Without this term the
        # policy learns to idle in calm regimes because every non-zero
        # rebalance costs turnover while inaction costs nothing. A small
        # per-step penalty proportional to (1 - |gross exposure|) breaks the
        # asymmetry: inaction carries its own cost, so the agent has to
        # genuinely outperform neutrality to justify idling.
        opp_cost_coef = CONFIG.get('OPPORTUNITY_COST_COEF', 0.0001)
        if opp_cost_coef > 0:
            gross_exposure = float(np.sum(np.abs(target_weights)))
            idle_fraction = max(0.0, 1.0 - gross_exposure)
            reward -= opp_cost_coef * idle_fraction

        reward = np.clip(reward, -10.0, 10.0)

        self.current_step += 1
        steps_in_episode = self.current_step - self.episode_start
        at_data_end = self.current_step >= len(self.timeline) - 1
        truncated = steps_in_episode >= self.max_episode_steps
        done = at_data_end or self.equity <= 0 or truncated

        # Normalize annual_vol to [0,1] for aux Sigmoid head. Raw annual_vol can be 0.05–2.0+.
        vol_target = min(1.0, annual_vol / 1.0)  # 100% annual vol maps to 1.0, capped at 1.0
        info = {
            'volatility_target': vol_target,
            'portfolio_return': portfolio_return,
            'drawdown': drawdown,
        }
        # Gymnasium convention: done=terminal (bankrupt/data end), truncated=time limit
        terminal = at_data_end or self.equity <= 0
        return self._get_observation(), reward, terminal, truncated, info

    def _compute_portfolio_return_at_step(self, step_idx):
        # Use precomputed returns array (O(1) lookup instead of O(n_symbols) get_indexer)
        returns = self._precomputed_returns[step_idx]

        # Deque-safe index: clamp to valid range within the deque
        history_idx = len(self.weight_history) - 1 - (self.current_step - step_idx)
        if 0 <= history_idx < len(self.weight_history):
            past_weights = self.weight_history[history_idx]
        else:
            past_weights = np.zeros(self.n_symbols, dtype=np.float32)  # No position data = neutral (not current weights)
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

            # Feature regeneration removed — precomputed features are sufficient for training.
            # Calling generate_features() during PPO training caused yfinance API calls,
            # severe slowdown, and blocking I/O.

            if len(latest) < self.feature_dim:
                latest = np.pad(latest, (0, self.feature_dim - len(latest)))
            elif len(latest) > self.feature_dim:
                latest = latest[:self.feature_dim]
            all_features.append(latest)

        per_symbol_obs = np.concatenate(all_features)

        # FIX #56: Use max(self.equity, 1e-8) to prevent NaN from log of negative values
        equity_norm = np.log(max(self.equity, 1e-8) / self.initial_balance + 1e-8)
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

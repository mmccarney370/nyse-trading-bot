# strategy/universe.py
# UPGRADE #4 (Feb 20 2026) — Continuous regime persistence score
# Now uses HMM self-transition probability (0.0-1.0) as regime_score
# Much more granular and powerful than binary trending/mean-reverting
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from strategy.regime import detect_regime, is_trending # Now returns (regime, persistence)
from data.ingestion import DataIngestion
from config import CONFIG

logger = logging.getLogger(__name__)

class UniverseManager:
    def __init__(self, data_ingestion: DataIngestion, live_signal_history: dict, config: dict):
        # ISSUE #5 FIX: Added config parameter and self.config assignment
        # This prevents AttributeError when calling get_current_universe()
        self.data_ingestion = data_ingestion
        self.live_signal_history = live_signal_history
        self.config = config
        self.candidates = CONFIG['UNIVERSE_CANDIDATES']
        self.max_size = CONFIG['MAX_UNIVERSE_SIZE']

    def evaluate_universe(self) -> list:
        """Score all candidates and return top N symbols"""
        scores = {}
        returns_df = pd.DataFrame()
        for symbol in self.candidates:
            try:
                # P-11 / Critical #3 FIX: Use lowercase '1d' to match data handler check
                data = self.data_ingestion.get_latest_data(symbol, timeframe='1d', lookback_days=CONFIG['UNIVERSE_LOOKBACK_DAYS'])
                if len(data) < 50:
                    logger.debug(f"{symbol} insufficient data")
                    continue
                # 1. Liquidity score (normalized 0-1)
                # FIX #36: Use log scale instead of linear 2x cap. Linear capping gave
                # 0.0 at threshold and 1.0 at 2x, with no differentiation above 2x.
                # Log scale: smooth 0→1 mapping that rewards higher volume without saturation.
                avg_volume = data['volume'].mean()
                min_vol = CONFIG['MIN_AVG_VOLUME']
                if avg_volume < min_vol:
                    liquidity_score = 0.0
                else:
                    liquidity_score = min(np.log1p(avg_volume / min_vol), 3.0) / 3.0
                # 2. Regime fit — prefer 1H for less noise (UPGRADE #4)
                regime_data_1h = self.data_ingestion.get_latest_data(symbol, timeframe='1H', lookback_days=CONFIG['UNIVERSE_LOOKBACK_DAYS'])
                if regime_data_1h is not None and len(regime_data_1h) >= 50:
                    regime_data = regime_data_1h
                else:
                    regime_data = data # fallback to daily
                # UPGRADE #4: Unpack continuous persistence score
                regime, persistence = detect_regime(
                    data=regime_data,
                    symbol=symbol,
                    data_ingestion=self.data_ingestion
                )
                # FIX #30: Give trending regimes a higher score than mean-reverting.
                # REGIME_TRENDING_WEIGHT is meant to favor trending stocks, so trending
                # regimes should contribute more than mean-reverting ones.
                if is_trending(regime):
                    regime_score = persistence  # Full persistence for trending
                else:
                    regime_score = persistence * 0.4  # Discount mean-reverting (less favorable for momentum)
                # 3. Recent performance (from live history or quick backtest proxy)
                history = self.live_signal_history.get(symbol, [])
                if len(history) >= 10:
                    recent_rets = [e['realized_return'] for e in history[-20:] if e['realized_return'] is not None]
                    win_rate = sum(r > 0 for r in recent_rets) / len(recent_rets) if recent_rets else 0.5
                    perf_score = win_rate
                else:
                    perf_score = 0.5 # Neutral if little data
                # Collect returns for correlation (use last 60 days, aligned by index)
                daily_ret = data['close'].pct_change().dropna().tail(60)
                returns_df[symbol] = daily_ret
                total_score = (
                    CONFIG['LIQUIDITY_WEIGHT'] * liquidity_score +
                    CONFIG['REGIME_TRENDING_WEIGHT'] * regime_score + # Now uses continuous persistence
                    CONFIG['PERFORMANCE_WEIGHT'] * perf_score
                )
                scores[symbol] = total_score
                logger.debug(f"{symbol} scores - Liq: {liquidity_score:.2f}, "
                             f"Persistence: {persistence:.3f} (regime={regime}), "
                             f"Perf: {perf_score:.2f} → Total: {total_score:.3f}")
            except Exception as e:
                logger.warning(f"Failed scoring {symbol}: {e}")
        if returns_df.empty:
            logger.warning("No valid returns data for diversification scoring")
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.max_size]
            return [s[0] for s in ranked]
        # 4. Diversification: Penalize high correlation clusters
        # Only keep symbols that were successfully scored (avoid phantom correlations)
        returns_df = returns_df[[c for c in returns_df.columns if c in scores]]
        # Drop columns that are entirely NaN
        returns_df = returns_df.dropna(axis=1, how='all')
        # FIX #32: Use pairwise correlation instead of dropna() which drops rows where ANY symbol has NaN.
        # This preserves data for symbols with different trading histories.
        min_corr_rows = CONFIG.get('MIN_CORRELATION_ROWS', 20)
        corr_matrix = returns_df.corr(min_periods=min_corr_rows)
        final_scores = scores.copy()
        for symbol in scores:
            if symbol not in corr_matrix.columns:
                continue
            # Average correlation to other candidates
            # L26 FIX: Handle NaN from sparse data (dropna before mean)
            avg_corr = corr_matrix[symbol].drop(symbol).dropna().mean()
            if np.isnan(avg_corr):
                avg_corr = 0.0  # neutral penalty for data-sparse symbols
            divers_penalty = max(0.0, avg_corr - 0.5) # Penalize if avg corr > 0.5
            final_scores[symbol] -= CONFIG['DIVERSIFICATION_WEIGHT'] * divers_penalty * scores[symbol]
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        new_universe = [s[0] for s in ranked[:self.max_size]]
        logger.info(f"New universe ({len(new_universe)} symbols): {new_universe}")
        logger.info(f"Top scores: {dict(ranked[:10])}")
        return new_universe

    # ISSUE #5 PATCH: Expose current active universe for comparison in Gemini task
    def get_current_universe(self) -> list:
        """Return the current active symbols (used by Gemini task to detect change)."""
        return list(self.config['SYMBOLS'])

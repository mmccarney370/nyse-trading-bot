# strategy/risk.py
# UPDATED March 3 2026 — Critical CVaR + allocation safety patch
# Fixes the "$347 budget → $43k per symbol" buying-power explosion
# Now respects real equity, adds hard caps, clearer logging
# NEW: safe_close_position to fix "insufficient qty available" when brackets are active
# M-1 FIX (March 2026): Aggressive regime caching to eliminate 5–10 redundant HMM calls per cycle
# M-5 FIX (March 2026): Added real-time buying power cap to prevent over-sizing beyond available margin/cash
import logging
import asyncio
import threading
import numpy as np
import pandas as pd
import cvxpy as cp
from datetime import datetime, timedelta, date as date_type
from dateutil import tz
from config import CONFIG
from sklearn.covariance import LedoitWolf
from strategy.regime import detect_regime, is_trending

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config, data_ingestion, broker=None): # ← broker added for safe close + buying power
        self.config = config
        self.data_ingestion = data_ingestion
        self.broker = broker # ← required for safe_close_position + buying power check
        # L22 FIX: Removed dead self.daily_start_equity (never read or updated)
        self.dd_paused = False
        self.immediate_pause_active = False
        # M-1 FIX: Per-symbol regime cache with 60-second TTL (one trading cycle)
        self._regime_cache = {}  # symbol → {'regime': str, 'persistence': float, 'timestamp': datetime}
        # CRIT-8 FIX: Thread safety for regime cache (accessed from multiple threads)
        self._regime_cache_lock = threading.Lock()

    def _get_cached_regime(self, symbol: str, data: pd.DataFrame) -> tuple[str, float]:
        """M-1 FIX: Return cached regime if valid, else compute and cache (TTL=60s)
        CRIT-8 FIX: Thread-safe access via _regime_cache_lock"""
        now = datetime.now(tz=tz.gettz('UTC'))
        with self._regime_cache_lock:
            cache_entry = self._regime_cache.get(symbol)
            if cache_entry and (now - cache_entry['timestamp']).total_seconds() < 60:
                logger.debug(f"[REGIME CACHE HIT] {symbol} — using cached {cache_entry['regime']} (persistence={cache_entry['persistence']:.3f})")
                return cache_entry['regime'], cache_entry['persistence']

        # Cache miss or expired → compute (outside lock — heavy I/O)
        logger.debug(f"[REGIME CACHE MISS] {symbol} — computing fresh regime")
        regime_tuple = detect_regime(
            data=data,
            symbol=symbol,
            data_ingestion=self.data_ingestion,
            lookback=CONFIG.get('LOOKBACK', 900),
            verbose=False
        )
        if isinstance(regime_tuple, (list, tuple)) and len(regime_tuple) == 2:
            regime, persistence = regime_tuple
        else:
            regime, persistence = str(regime_tuple), 0.5

        # FIX #26: Capture timestamp AFTER compute, not before. Pre-computing timestamp
        # means a slow detect_regime() call could produce a cache entry that appears
        # fresher than it actually is, causing stale data to be served for longer.
        cache_time = datetime.now(tz=tz.gettz('UTC'))
        # Cache the result
        with self._regime_cache_lock:
            self._regime_cache[symbol] = {
                'regime': regime,
                'persistence': persistence,
                'timestamp': cache_time
            }
        logger.debug(f"[REGIME CACHE STORE] {symbol} → {regime} (persistence={persistence:.3f})")
        return regime, persistence

    def _compute_current_atr(self, data_window: pd.DataFrame, lookback: int = 50) -> float:
        """Proper classic True Range + EMA(14) ATR — consistent with backtest.py and alpaca.py."""
        if len(data_window) < 14:
            return 0.01
        recent = data_window.tail(lookback)
        high_low = recent['high'] - recent['low']
        high_close = (recent['high'] - recent['close'].shift(1)).abs()
        low_close = (recent['low'] - recent['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        tr = tr.dropna()
        if len(tr) == 0:
            return 0.01
        atr_series = tr.ewm(span=14, adjust=False).mean()
        atr = atr_series.iloc[-1]
        floor = 0.0005 * recent['close'].iloc[-1]
        return max(atr, floor)

    def calculate_position_size(self, equity: float, price: float, symbol: str, data: pd.DataFrame,
                                risk_amount: float | None = None,
                                conviction: float = 1.0,
                                regime: str = None,  # ← Optional: caller can pass precomputed
                                persistence: float = None) -> int | float: # ← FIX #24: returns float when FRACTIONAL_SHARES=True
        """M-1 FIX: Use cached regime if not provided (avoids redundant HMM calls)
        M-5 FIX: Cap shares by real-time buying power + safety margin (prevents over-sizing)"""
        if equity <= 0 or price <= 0:
            return 0
        
        # M-1: Prefer passed regime/persistence, else use cache
        if regime is None or persistence is None:
            regime, persistence = self._get_cached_regime(symbol, data)
        
        base_risk_pct_key = (
            'RISK_PER_TRADE_TRENDING' if is_trending(regime)
            else 'RISK_PER_TRADE_MEAN_REVERTING'
        )
        base_risk_pct = self.config.get(base_risk_pct_key, self.config.get('RISK_PER_TRADE', 0.02))
        if risk_amount is None:
            risk_amount = equity * base_risk_pct
        if len(data) < 50:
            logger.debug(f"Insufficient data ({len(data)} bars) for sizing {symbol} — returning 0")
            return 0
        atr = self._compute_current_atr(data, lookback=50)
        if atr <= 0:
            logger.debug(f"ATR <= 0 for {symbol} — returning 0")
            return 0
        atr_pct = atr / price if price > 0 else 0.01
        atr_pct = max(atr_pct, 0.0005)
        trailing_mult_key = (
            'TRAILING_STOP_ATR_TRENDING' if is_trending(regime)
            else 'TRAILING_STOP_ATR_MEAN_REVERTING'
        )
        trailing_mult = self.config.get(trailing_mult_key, self.config.get('TRAILING_STOP_ATR', 2.0))
        stop_distance_pct = atr_pct * trailing_mult
        logger.info(f"[ATR DEBUG] {symbol} | regime={regime} | price={price:.2f} | atr={atr:.4f} | "
                    f"atr_pct={atr_pct:.6f} | trailing_mult={trailing_mult:.2f} | "
                    f"stop_distance_pct={stop_distance_pct:.6f} | risk_amount={risk_amount:.2f}")
        base_shares = risk_amount / (price * stop_distance_pct) if stop_distance_pct > 0 else 0.0
        if base_shares <= 0:
            logger.info(f"[SIZE ZERO] {symbol} — base_shares={base_shares:.4f}")
            return 0
        conviction_power = conviction ** 2.0
        scaled_shares = base_shares * conviction_power
        scaled_shares *= self.config.get('KELLY_FRACTION', 0.75)
        # ==================== RECOMMENDED UPGRADE: REGIME CONFIDENCE SCALING ====================
        # Dynamic scaling using config key (REGIME_CONFIDENCE_MIN_SIZE_PCT)
        # Weak regime (persistence ~0.5) → reduced size; strong regime → full size
        regime_confidence = max(0.0, persistence - 0.5) * 2.0
        min_size_pct = self.config.get('REGIME_CONFIDENCE_MIN_SIZE_PCT', 0.3)
        size_multiplier = min_size_pct + (1 - min_size_pct) * regime_confidence
        scaled_shares *= size_multiplier
        logger.debug(f"[REGIME CONFIDENCE] {symbol} | persistence={persistence:.3f} → "
                     f"multiplier={size_multiplier:.3f} (min_floor={min_size_pct}) | final_shares={scaled_shares:.1f}")
        # ==================== END RECOMMENDED UPGRADE ====================
        conv_threshold = self.config.get('CONVICTION_THRESHOLD', 0.28)
        if conviction < conv_threshold:
            logger.debug(f"Low conviction skip for {symbol}: {conviction:.2f} < {conv_threshold}")
            scaled_shares = 0
        if self.config.get('FRACTIONAL_SHARES', True):
            shares = round(scaled_shares, 4)  # Keep fractional precision
        else:
            shares = int(scaled_shares)
        max_value = equity * self.config.get('MAX_POSITION_VALUE_FRACTION', 0.2)
        if shares * price > max_value:
            if self.config.get('FRACTIONAL_SHARES', True):
                shares = round(max_value / price, 4)
            else:
                shares = int(max_value / price)
            logger.debug(f"Position capped by MAX_POSITION_VALUE_FRACTION for {symbol}: {shares} shares")

        # M-5 FIX: Real-time buying power safety cap — prevents sizing beyond available cash/margin
        # CRIT-7 FIX: Use pre-fetched buying_power from _cached_buying_power (set by caller in bot.py)
        # instead of making a blocking sync API call here. Falls back to equity cap if not set.
        buying_power = getattr(self, '_cached_buying_power', None)
        if buying_power is not None and buying_power > 0:
            safety_factor = self.config.get('MAX_ORDER_NOTIONAL_PCT', 0.85)
            if self.config.get('FRACTIONAL_SHARES', True):
                max_affordable = round(buying_power * safety_factor / price, 4) if price > 0 else 0
            else:
                max_affordable = int(buying_power * safety_factor / price) if price > 0 else 0
            if shares > max_affordable:
                logger.warning(f"[M-5 BUYING POWER CAP] {symbol}: requested {shares} shares → reduced to {max_affordable} "
                               f"(buying_power=${buying_power:,.0f}, safety_factor={safety_factor})")
                shares = max_affordable
        else:
            logger.debug(f"[M-5] No cached buying power — using equity cap only for {symbol}")

        logger.info(f"[SIZE CALC] {symbol} — base_shares={base_shares:.2f} | conviction={conviction:.2f} | "
                    f"scaled_shares={scaled_shares:.2f} | final_shares={shares}")
        return max(shares, 0)

    def allocate_portfolio_risk(self, equity: float, symbols: list,
                                confidences: list = None,
                                regimes: dict = None) -> dict: # ← NEW: regimes dict for persistence symmetry
        """
        FIXED March 2 2026 + persistence symmetry upgrade.
        Scales total risk budget by average regime confidence before CVaR.
        M-1 FIX: Use cached regimes instead of recomputing
        """
        n = len(symbols)
        if n == 0:
            return {}
        if n == 1:
            return {symbols[0]: equity * self.config.get('RISK_PER_TRADE', 0.02)}
        
        # M-1 FIX: Build regimes dict from cache (avoids redundant HMM calls)
        if regimes is None:
            regimes = {}
            for sym in symbols:
                data = self.data_ingestion.get_latest_data(sym)
                regime, persistence = self._get_cached_regime(sym, data)
                regimes[sym] = (regime, persistence)
        
        # FIX: Determine dominant regime from actual per-symbol regimes dict (was using stale global CURRENT_REGIME)
        if regimes:
            regime_counts = {}
            for sym in symbols:
                r = regimes.get(sym, ('mean_reverting', 0.5))
                r_name = r[0] if isinstance(r, (list, tuple)) else r
                regime_counts[r_name] = regime_counts.get(r_name, 0) + 1
            regime = max(regime_counts, key=regime_counts.get)
        else:
            regime = self.config.get('CURRENT_REGIME', 'mean_reverting')
        base_risk_key = 'RISK_PER_TRADE_TRENDING' if is_trending(regime) else 'RISK_PER_TRADE_MEAN_REVERTING'
        base_risk_pct = self.config.get(base_risk_key, self.config.get('RISK_PER_TRADE', 0.02))
        # === CRITICAL FIX #1: Correct total portfolio risk budget ===
        total_risk_budget = equity * base_risk_pct
        total_risk_budget *= self.config.get('RISK_BUDGET_MULTIPLIER', 1.8) # mild leverage allowed
        # NOTE: Regime persistence scaling removed from allocate_portfolio_risk to prevent
        # triple penalty (allocate_portfolio_risk + calculate_position_size + portfolio_rebalancer).
        # Persistence scaling is now applied ONLY in portfolio_rebalancer for portfolio mode.
        # === CRITICAL FIX #2: Hard safety caps (never blow buying power again) ===
        max_total_risk = equity * self.config.get('MAX_TOTAL_RISK_PCT', 0.12) # max 12% of equity at risk total
        total_risk_budget = min(total_risk_budget, max_total_risk)
        logger.info(f"[CVaR REGIME] Using Gemini-tuned risk budget: ${total_risk_budget:,.0f} | "
                    f"regime={regime} | base_risk_pct={base_risk_pct:.4f} | equity=${equity:,.0f} | "
                    f"multiplier={self.config.get('RISK_BUDGET_MULTIPLIER', 1.8):.2f}")
        # Prepare returns for CVaR
        returns_list = []
        min_len = float('inf')
        for sym in symbols:
            data = self.data_ingestion.get_latest_data(sym)
            if len(data) < 100:
                logger.debug(f"Insufficient data for CVaR on {sym} — using parity fallback")
                min_len = 0
                break
            ret = data['close'].pct_change().dropna().tail(500)
            returns_list.append(ret.values)
            min_len = min(min_len, len(ret))
        if min_len < 50:
            logger.debug("Insufficient overlapping returns for CVaR — using conviction-weighted fallback")
            if confidences is not None and len(confidences) == n and np.sum(np.abs(confidences)) > 0:
                weights = np.array(confidences) / np.sum(np.abs(confidences))
            else:
                weights = np.ones(n) / n
            alloc = {sym: total_risk_budget * weights[i] for i, sym in enumerate(symbols)}
            return alloc
        R = np.column_stack([r[-min_len:] for r in returns_list])
        if np.any(np.isnan(R)) or np.any(np.isinf(R)):
            # HIGH-8 FIX: Explicitly replace inf with 0.0 instead of default 1.7e308
            # which blows up the CVaR optimizer
            R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        # Expected returns (conviction tilt)
        # FIX #29: Use mean historical returns as base, tilted by confidence direction/magnitude.
        # This keeps mu in the same units as the return matrix R, so the optimizer's
        # risk-return tradeoff is meaningful instead of using arbitrary synthetic scale.
        mean_returns = np.mean(R, axis=0)  # historical mean daily returns per symbol
        if confidences is not None and len(confidences) == n:
            conf_arr = np.array(confidences)
            # HIGH-9 FIX: Use raw confidence values clamped to [-1, 1] instead of
            # normalizing by max. The old approach (conf_arr / max_abs) lost magnitude:
            # a portfolio where all signals are 0.01 got the same tilt as one where all
            # signals are 0.9. Clamping preserves absolute signal strength.
            conf_norm = np.clip(conf_arr, -1.0, 1.0)
            # Tilt: base historical return + confidence-scaled adjustment (1 std of returns)
            return_std = np.std(R, axis=0)
            mu = mean_returns + conf_norm * return_std
        else:
            mu = mean_returns
        # Covariance with Ledoit-Wolf + ridge
        try:
            lw = LedoitWolf()
            cov = lw.fit(R).covariance_
        except Exception:
            cov = np.cov(R, rowvar=False)
        trace_norm = np.trace(cov) / n if n > 0 else 0
        cov += np.eye(n) * (1e-4 * trace_norm)
        # CVaR optimization
        w = cp.Variable(n)
        alpha = cp.Variable()
        u = cp.Variable(min_len)
        portfolio_return = mu @ w
        cvar = alpha + (1 / ((1 - 0.95) * min_len)) * cp.sum(u)
        max_weight = max(self.config.get('MAX_SECTOR_CONCENTRATION', 0.30), 1.0 / n)
        # FIX #27: Short weight limit now configurable (was hardcoded -0.15 vs +0.30 long)
        min_short_weight = self.config.get('CVAR_MIN_SHORT_WEIGHT', -0.15)
        constraints = [
            cp.sum(w) == 1,
            w >= min_short_weight,
            w <= max_weight,
        ]
        constraints.extend([
            u >= 0,
            u >= -R @ w - alpha,
            # FIX #19: Tighten CVaR constraint — raw budget/equity (~6.7%) is too loose
            # for 95% CVaR. Use configurable multiplier (default 0.3) to get ~2% limit.
            cvar <= total_risk_budget / equity * self.config.get('CVAR_CONSTRAINT_TIGHTENER', 0.3)
        ])
        prob = cp.Problem(cp.Maximize(portfolio_return), constraints)
        try:
            prob.solve(solver=cp.SCS, max_iters=8000, eps=1e-5)
            if w.value is not None and prob.status in ["optimal", "optimal_inaccurate"]:
                # FIX #23: Use CVaR-optimized weights directly — the solver already handles
                # long/short allocation. Overriding sign with PPO confidence was forcing
                # CVaR-optimized shorts to become longs when PPO confidence was positive.
                weights = np.array(w.value).flatten()
                # H13 FIX: Normalize by gross exposure only when it EXCEEDS 1.0.
                # Dividing by sum(abs) always destroyed signed sum-to-one when shorts existed.
                # Now: cap gross exposure to 1.0 but preserve net direction if within budget.
                max_per_symbol = self.config.get('MAX_SECTOR_CONCENTRATION', 0.30)
                weights = np.clip(weights, -max_per_symbol, max_per_symbol)
                total_abs = np.sum(np.abs(weights))
                if total_abs > 1.0:
                    weights = weights / total_abs
                alloc = {sym: total_risk_budget * weights[i] for i, sym in enumerate(symbols)}
                # === FINAL HARD SAFETY CLAMP (this stops insufficient buying power) ===
                # NOTE (FIX #25): alloc[sym] is risk-dollar allocation, max_per_symbol_dollar is
                # notional cap. This is intentionally conservative: risk dollars should never
                # exceed the notional position cap. The portfolio_rebalancer converts risk→notional
                # downstream using stop_distance_pct, which amplifies risk dollars ~50-100x.
                max_per_symbol_dollar = equity * self.config.get('MAX_POSITION_VALUE_FRACTION', 0.20) # BUG #8 PATCH: consistent with config default + other method
                for sym in alloc:
                    alloc[sym] = np.clip(alloc[sym], -max_per_symbol_dollar, max_per_symbol_dollar)
                # FIX: Only redistribute budget to UNCAPPED symbols (renormalizing all defeated the clamp)
                total_clipped = sum(abs(v) for v in alloc.values())
                if total_clipped > 0 and total_clipped < total_risk_budget:
                    # Some budget was freed by capping — redistribute to uncapped symbols proportionally
                    freed = total_risk_budget - total_clipped
                    # M33 FIX: Exclude zero-weight symbols (np.sign(0)=0 loses freed budget)
                    uncapped = {sym: abs(alloc[sym]) for sym in alloc
                                if 0 < abs(alloc[sym]) < max_per_symbol_dollar * 0.99}
                    uncapped_total = sum(uncapped.values())
                    if uncapped_total > 0:
                        for sym in uncapped:
                            alloc[sym] += np.sign(alloc[sym]) * freed * (uncapped[sym] / uncapped_total)
                    # Re-apply per-symbol cap after redistribution (redistribution may have pushed symbols over)
                    for sym in alloc:
                        alloc[sym] = np.clip(alloc[sym], -max_per_symbol_dollar, max_per_symbol_dollar)
                logger.info(f"[CVaR LIVE SUCCESS] Allocations (signs preserved): {alloc}")
                logger.info(f"[CVaR TOTAL] Total risk allocated: ${sum(abs(v) for v in alloc.values()):,.0f}")
                return alloc
        except Exception as e:
            logger.debug(f"CVaR optimization exception ({e}) — using fallback")
        # Robust fallback
        if confidences is not None and len(confidences) == n and np.sum(np.abs(confidences)) > 0:
            weights = np.array(confidences) / np.sum(np.abs(confidences))
        else:
            weights = np.ones(n) / n
        alloc = {sym: total_risk_budget * weights[i] for i, sym in enumerate(symbols)}
        # Apply safety caps even in fallback mode
        max_per_symbol_frac = self.config.get('MAX_POSITION_VALUE_FRACTION', 0.20)
        max_total_risk = self.config.get('MAX_TOTAL_RISK_PCT', 0.12) * equity
        for sym in alloc:
            max_dollar = max_per_symbol_frac * equity
            if abs(alloc[sym]) > max_dollar:
                alloc[sym] = np.sign(alloc[sym]) * max_dollar
        # Clamp total fallback allocation by max_total_risk (was computed but unused)
        total_abs_alloc = sum(abs(v) for v in alloc.values())
        if total_abs_alloc > max_total_risk:
            scale = max_total_risk / total_abs_alloc
            alloc = {sym: v * scale for sym, v in alloc.items()}
            logger.warning(f"[CVaR FALLBACK] Total allocation ${total_abs_alloc:,.0f} exceeded max_total_risk ${max_total_risk:,.0f} — scaled down by {scale:.3f}")
        logger.info(f"[CVaR FALLBACK] Using conviction-weighted allocations: {alloc}")
        return alloc

    def check_pause_conditions(self, equity: float, daily_equity: dict, equity_history: dict, is_backtest: bool = False) -> bool:
        """Check daily loss threshold and 30-day drawdown circuit breaker.
        C3 FIX: Backtest now checks daily loss + drawdown (was returning False immediately).
        C4 FIX: Uses ET timezone consistently (NYSE operates on ET, matches bot.py key format)."""
        # C4 FIX: Use ET for daily key — bot.py stores daily_equity with ET dates
        today = datetime.now(tz=tz.gettz('America/New_York')).date()
        if today not in daily_equity:
            daily_equity[today] = equity
        daily_loss = (equity - daily_equity[today]) / daily_equity[today] if daily_equity[today] > 0 else 0.0
        # C3 FIX: Backtests check daily loss + drawdown, but skip API failure checks
        # and mutable state (immediate_pause_active/dd_paused) to avoid cross-bar state leakage
        if is_backtest:
            # Daily loss check
            if daily_loss < self.config.get('DAILY_LOSS_THRESHOLD', -0.03):
                return True
            # Drawdown check
            if len(equity_history) >= 2:
                # M34 FIX: Ensure keys are date objects before comparison
                try:
                    date_keys = [d if isinstance(d, date_type) else d for d in equity_history.keys()
                                 if isinstance(d, date_type)]
                    recent_dates = sorted([d for d in date_keys if d >= today - timedelta(days=30)])
                except TypeError:
                    recent_dates = []
                if len(recent_dates) >= 2:
                    equities = np.array([equity_history[d] for d in recent_dates])
                    peak = np.maximum.accumulate(equities)
                    drawdowns = (equities - peak) / peak
                    if drawdowns.min() < -0.15:
                        return True
            return False
        # Live trading: full checks including API failures and state tracking
        api_failures = getattr(self.data_ingestion.data_handler, 'api_failures', 0)
        immediate_trigger = daily_loss < self.config.get('DAILY_LOSS_THRESHOLD', -0.03) or api_failures > self.config.get('API_FAILURE_THRESHOLD', 5)
        if immediate_trigger and not self.immediate_pause_active:
            logger.warning("Trading paused due to daily loss threshold or excessive API failures")
            self.immediate_pause_active = True
        elif not immediate_trigger and self.immediate_pause_active:
            logger.info("Resuming trading: daily loss threshold or API failures cleared")
            self.immediate_pause_active = False
        paused_due_to_dd = False
        if len(equity_history) >= 2:
            recent_dates = sorted([d for d in equity_history.keys() if d >= today - timedelta(days=30)])
            if len(recent_dates) < 2:
                logger.warning(f"[DRAWDOWN CHECK] Skipped — only {len(recent_dates)} dates in last 30 days (need >=2)")
            if len(recent_dates) >= 2:
                equities = np.array([equity_history[d] for d in recent_dates])
                peak = np.maximum.accumulate(equities)
                drawdowns = (equities - peak) / peak
                max_dd = drawdowns.min()
                if max_dd < -0.15 and not self.dd_paused:
                    self.dd_paused = True
                    logger.warning(f"Drawdown circuit breaker triggered: trailing 30-day max DD = {max_dd:.1%}")
                elif max_dd > -0.10 and self.dd_paused:
                    self.dd_paused = False
                    logger.info(f"Drawdown recovered: trailing 30-day max DD = {max_dd:.1%} — resuming")
                paused_due_to_dd = self.dd_paused
        return immediate_trigger or paused_due_to_dd

    # ====================== SAFE POSITION CLOSE (FIXED for C-2) ======================
    async def safe_close_position(self, symbol: str) -> bool:
        """Safely close a position by first cancelling any active bracket orders.
        FIX: Made async — run_until_complete() crashes when called from an already-running event loop."""
        if self.broker is None:
            logger.error(f"Cannot safely close {symbol} — broker not passed to RiskManager")
            return False
        return await self.broker.close_position_safely(symbol)

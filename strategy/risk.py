# strategy/risk.py
# UPDATED March 3 2026 — Critical CVaR + allocation safety patch
# Fixes the "$347 budget → $43k per symbol" buying-power explosion
# Now respects real equity, adds hard caps, clearer logging
# NEW: safe_close_position to fix "insufficient qty available" when brackets are active
# M-1 FIX (March 2026): Aggressive regime caching to eliminate 5–10 redundant HMM calls per cycle
# M-5 FIX (March 2026): Added real-time buying power cap to prevent over-sizing beyond available margin/cash
import logging
import numpy as np
import pandas as pd
import cvxpy as cp
from datetime import datetime, timedelta
from dateutil import tz
from config import CONFIG
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config, data_ingestion, broker=None): # ← broker added for safe close + buying power
        self.config = config
        self.data_ingestion = data_ingestion
        self.broker = broker # ← required for safe_close_position + buying power check
        self.daily_start_equity = {}
        self.dd_paused = False
        self.immediate_pause_active = False
        # M-1 FIX: Per-symbol regime cache with 60-second TTL (one trading cycle)
        self._regime_cache = {}  # symbol → {'regime': str, 'persistence': float, 'timestamp': datetime}

    def _get_cached_regime(self, symbol: str, data: pd.DataFrame) -> tuple[str, float]:
        """M-1 FIX: Return cached regime if valid, else compute and cache (TTL=60s)"""
        now = datetime.now(tz=tz.gettz('UTC'))
        cache_entry = self._regime_cache.get(symbol)
        if cache_entry and (now - cache_entry['timestamp']).total_seconds() < 60:
            logger.debug(f"[REGIME CACHE HIT] {symbol} — using cached {cache_entry['regime']} (persistence={cache_entry['persistence']:.3f})")
            return cache_entry['regime'], cache_entry['persistence']
        
        # Cache miss or expired → compute
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
        
        # Cache the result
        self._regime_cache[symbol] = {
            'regime': regime,
            'persistence': persistence,
            'timestamp': now
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
                                persistence: float = None) -> int: # ← Optional: caller can pass precomputed
        """M-1 FIX: Use cached regime if not provided (avoids redundant HMM calls)
        M-5 FIX: Cap shares by real-time buying power + safety margin (prevents over-sizing)"""
        if equity <= 0 or price <= 0:
            return 0
        
        # M-1: Prefer passed regime/persistence, else use cache
        if regime is None or persistence is None:
            regime, persistence = self._get_cached_regime(symbol, data)
        
        base_risk_pct_key = (
            'RISK_PER_TRADE_TRENDING' if regime == 'trending'
            else 'RISK_PER_TRADE_MEAN_REVERTING' if regime == 'mean_reverting'
            else 'RISK_PER_TRADE'
        )
        base_risk_pct = self.config.get(base_risk_pct_key, self.config['RISK_PER_TRADE'])
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
            'TRAILING_STOP_ATR_TRENDING' if regime == 'trending'
            else 'TRAILING_STOP_ATR_MEAN_REVERTING' if regime == 'mean_reverting'
            else 'TRAILING_STOP_ATR'
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
        if conviction < 0.35:
            logger.debug(f"Low conviction skip for {symbol}: {conviction:.2f} < 0.35")
            scaled_shares = 0
        shares = int(scaled_shares)
        max_value = equity * self.config.get('MAX_POSITION_VALUE_FRACTION', 0.2)
        if shares * price > max_value:
            shares = int(max_value / price)
            logger.debug(f"Position capped by MAX_POSITION_VALUE_FRACTION for {symbol}: {shares} shares")

        # M-5 FIX: Real-time buying power safety cap — prevents sizing beyond available cash/margin
        if self.broker is not None:
            try:
                buying_power = self.broker.get_buying_power()
                safety_factor = self.config.get('MAX_ORDER_NOTIONAL_PCT', 0.85)  # e.g. 85% of BP to leave buffer
                max_affordable = int(buying_power * safety_factor / price) if price > 0 else 0
                if shares > max_affordable:
                    logger.warning(f"[M-5 BUYING POWER CAP] {symbol}: requested {shares} shares → reduced to {max_affordable} "
                                   f"(buying_power=${buying_power:,.0f}, safety_factor={safety_factor})")
                    shares = max_affordable
            except Exception as bp_e:
                logger.warning(f"Buying power fetch failed for {symbol}: {bp_e} — using equity cap only")
        else:
            logger.debug(f"[M-5] Broker not available — skipping buying power cap for {symbol}")

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
        
        # Regime-aware base risk (Gemini-tuned from bot.py)
        regime = self.config.get('CURRENT_REGIME', 'mean_reverting')
        base_risk_key = 'RISK_PER_TRADE_TRENDING' if regime == 'trending' else 'RISK_PER_TRADE_MEAN_REVERTING'
        base_risk_pct = self.config.get(base_risk_key, self.config.get('RISK_PER_TRADE', 0.02))
        # === CRITICAL FIX #1: Correct total portfolio risk budget ===
        total_risk_budget = equity * base_risk_pct
        total_risk_budget *= self.config.get('RISK_BUDGET_MULTIPLIER', 1.8) # mild leverage allowed
        # ==================== NEW: REGIME CONFIDENCE SYMMETRY SCALING ====================
        if regimes is not None:
            # Average persistence across symbols (symmetric with per-symbol scaling)
            persistences = []
            for sym in symbols:
                regime_tuple = regimes.get(sym, ('mean_reverting', 0.5))
                persistence = regime_tuple[1] if isinstance(regime_tuple, (list, tuple)) else 0.5
                persistences.append(persistence)
        
            if persistences:
                avg_persistence = np.mean(persistences)
                regime_confidence = max(0.0, avg_persistence - 0.5) * 2.0
                min_size_pct = self.config.get('REGIME_CONFIDENCE_MIN_SIZE_PCT', 0.3)
                confidence_multiplier = min_size_pct + (1 - min_size_pct) * regime_confidence
            
                total_risk_budget *= confidence_multiplier
                logger.debug(f"[PORTFOLIO REGIME CONFIDENCE] avg_persistence={avg_persistence:.3f} → "
                             f"multiplier={confidence_multiplier:.3f} | adjusted_budget=${total_risk_budget:,.0f}")
        # ==================== END NEW SYMMETRY SCALING ====================
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
            R = np.nan_to_num(R)
        # Expected returns (conviction tilt)
        if confidences is not None and len(confidences) == n:
            mu_raw = np.array(confidences)
            mu = 0.00005 + mu_raw * 0.00025
            mu = mu / np.sum(np.abs(mu)) if np.sum(np.abs(mu)) > 0 else np.ones(n) / n
        else:
            mu = np.ones(n) / n * 0.0001
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
        constraints = [
            cp.sum(w) == 1,
            w >= -self.config.get('MAX_LEVERAGE', 2.5),
            w <= self.config.get('MAX_SECTOR_CONCENTRATION', 0.35),
        ]
        if n > 3:
            constraints.append(w >= -0.15)
        constraints.extend([
            u >= 0,
            u >= -R @ w - alpha,
            cvar <= total_risk_budget / equity
        ])
        prob = cp.Problem(cp.Maximize(portfolio_return), constraints)
        try:
            prob.solve(solver=cp.SCS, max_iters=8000, eps=1e-5)
            if w.value is not None and prob.status in ["optimal", "optimal_inaccurate"]:
                # Preserve PPO sign, use CVaR magnitude
                raw_signed = np.array(confidences) if confidences is not None else np.ones(n)
                raw_signed = raw_signed / (np.sum(np.abs(raw_signed)) + 1e-8)
                cvar_magnitude = np.abs(w.value)
                cvar_magnitude /= np.sum(cvar_magnitude) + 1e-8
                weights = np.sign(raw_signed) * cvar_magnitude
                # Strong per-symbol cap
                max_per_symbol = self.config.get('MAX_SECTOR_CONCENTRATION', 0.25)
                weights = np.clip(weights, -max_per_symbol, max_per_symbol)
                total_abs = np.sum(np.abs(weights))
                if total_abs > 0:
                    weights = weights / total_abs
                alloc = {sym: total_risk_budget * weights[i] for i, sym in enumerate(symbols)}
                # === FINAL HARD SAFETY CLAMP (this stops insufficient buying power) ===
                max_per_symbol_dollar = equity * self.config.get('MAX_POSITION_VALUE_FRACTION', 0.20) # BUG #8 PATCH: consistent with config default + other method
                for sym in alloc:
                    alloc[sym] = np.clip(alloc[sym], -max_per_symbol_dollar, max_per_symbol_dollar)
                # BUG #17 PATCH: renormalize after per-symbol clamp so total risk budget remains exactly respected (prevents lost leverage when any position is capped)
                total_clipped = sum(abs(v) for v in alloc.values())
                if total_clipped > 0:
                    scale = total_risk_budget / total_clipped
                    for sym in alloc:
                        alloc[sym] *= scale
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
        logger.info(f"[CVaR FALLBACK] Using conviction-weighted allocations: {alloc}")
        return alloc

    def check_pause_conditions(self, equity: float, daily_equity: dict, equity_history: dict, is_backtest: bool = False) -> bool:
        """Unchanged — exactly as you had it."""
        if is_backtest:
            return False
        today = datetime.now(tz=tz.gettz('UTC')).date()
        if today not in daily_equity:
            daily_equity[today] = equity
        daily_loss = (equity - daily_equity[today]) / daily_equity[today] if daily_equity[today] > 0 else 0.0
        api_failures = getattr(self.data_ingestion.data_handler, 'api_failures', 0)
        immediate_trigger = daily_loss < self.config['DAILY_LOSS_THRESHOLD'] or api_failures > self.config['API_FAILURE_THRESHOLD']
        if immediate_trigger and not self.immediate_pause_active:
            logger.warning("Trading paused due to daily loss threshold or excessive API failures")
            self.immediate_pause_active = True
        elif not immediate_trigger and self.immediate_pause_active:
            logger.info("Resuming trading: daily loss threshold or API failures cleared")
            self.immediate_pause_active = False
        paused_due_to_dd = False
        if len(equity_history) >= 2:
            recent_dates = sorted([d for d in equity_history.keys() if d >= today - timedelta(days=30)])
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
    def safe_close_position(self, symbol: str) -> bool:
        """Safely close a position by first cancelling any active bracket orders.
        This fixes the 'insufficient qty available' error when brackets are holding shares."""
        if self.broker is None:
            logger.error(f"Cannot safely close {symbol} — broker not passed to RiskManager")
            return False
        # AWAIT the async broker method — returns the real bool result
        return asyncio.get_event_loop().run_until_complete(
            self.broker.close_position_safely(symbol)
        )

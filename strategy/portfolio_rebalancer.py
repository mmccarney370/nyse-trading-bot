# strategy/portfolio_rebalancer.py
"""
Extracted portfolio rebalance logic from bot.py trading_loop()
Handles: PPO inference → CVaR allocation → notional cap → causal scaling → final renormalization
Returns normalized target_weights_dict for order execution
"""
import logging
import numpy as np
from typing import Dict, Any
from datetime import datetime, timedelta       # ← BUG-05 FIX: added for datetime.now()
from dateutil import tz                        # ← BUG-05 FIX: added for tz.gettz('UTC')
from models.portfolio_env import PortfolioEnv # ISSUE #6: import for persistent env
from strategy.regime import is_trending

logger = logging.getLogger(__name__)

class PortfolioRebalancer:
    def __init__(self, config, signal_gen, risk_manager):
        self.config = config
        self.signal_gen = signal_gen
        self.risk_manager = risk_manager

    async def rebalance_portfolio(
        self,
        current_equity: float,
        data_dict: Dict[str, Any],
        prices: Dict[str, float],
        regimes: Dict[str, Any], # ← NEW: accept regimes dict for persistence symmetry
        positions: Dict[str, int],
        precomputed_env: PortfolioEnv = None, # ISSUE #6: Accept persistent env from bot.py
        daily_equity: Dict = None,  # NEW: intraday risk pacing
        live_signal_history: Dict = None,  # NEW: consecutive-loss tracking
    ) -> Dict[str, float]:
        """
        Full rebalance pipeline for portfolio PPO mode.
        Returns: target_weights_dict (sign-preserved, normalized to ~1.0 total abs weight)
        """
        symbols = self.config['SYMBOLS']
        logger.debug("Generating portfolio-level actions via multi-asset PPO")
        # ISSUE #6 PATCH: Use persistent PortfolioEnv if provided (lightweight update only)
        if precomputed_env is not None:
            logger.debug("Using persistent precomputed_env (ISSUE #6: lightweight update)")
            # FIX #37: Removed redundant data_dict copy here — generate_portfolio_actions()
            # (called below via signal_gen) copies data_dict again, making this one wasteful.
            obs = precomputed_env._get_observation() # Reuse precomputed timeline/regimes/features
        else:
            logger.debug("Creating new PortfolioEnv (fallback - no persistent env)")
            precomputed_env = PortfolioEnv(
                data_dict=data_dict,
                symbols=symbols,
                initial_balance=current_equity,
                max_leverage=self.config.get('MAX_LEVERAGE', 2.0)
            )
            obs, _ = precomputed_env.reset()
        target_weights_dict = await self.signal_gen.generate_portfolio_actions(
            symbols=symbols,
            data_dict=data_dict,
            current_equity=current_equity,
            precomputed_env=precomputed_env # Pass persistent env (or new one)
        )
        # Use PPO target weights as conviction proxy for CVaR tilt
        confidences = [target_weights_dict.get(sym, 0.0) for sym in symbols]
        # ==================== NEW: PASS REGIMES FOR PERSISTENCE WEIGHTING SYMMETRY ====================
        logger.debug(f"[PORTFOLIO REBALANCER] Passing regimes dict with {len(regimes)} entries to allocate_portfolio_risk")
        dollar_risk_alloc = self.risk_manager.allocate_portfolio_risk(
            equity=current_equity,
            symbols=symbols,
            confidences=confidences,
            regimes=regimes, # ← NEW: forward regimes dict so persistence can be averaged
            daily_equity=daily_equity,  # NEW: intraday risk pacing
            live_signal_history=live_signal_history,  # NEW: consecutive-loss tracking
        )
        # ==================== END NEW SYMMETRY ====================
        # Apply CVaR dollar risk to final weights
        # FIX: Convert risk dollars → notional position via ATR stop distance
        # risk_dollars is how much we're willing to LOSE, not the position size itself
        for sym in symbols:
            if sym in dollar_risk_alloc and dollar_risk_alloc[sym] != 0:
                price = prices.get(sym)
                if not price or price <= 0 or current_equity <= 0:
                    # FIX #34: Skip symbol entirely when price is missing — leaving PPO-scale
                    # weight while others are in risk-scale causes dangerous scale mismatch
                    target_weights_dict[sym] = 0.0
                    logger.warning(f"[CVaR] {sym} price missing or invalid — setting weight to 0")
                    continue
                risk_dollars = dollar_risk_alloc[sym]
                # Compute stop distance to convert risk → notional
                data = data_dict.get(sym)
                if data is not None and len(data) >= 14:
                    atr = self.risk_manager._compute_current_atr(data, lookback=50)
                    atr_pct = max(atr / price, 0.0005)
                    regime_tuple = regimes.get(sym, ('mean_reverting', 0.5))
                    regime_str = regime_tuple[0] if isinstance(regime_tuple, (list, tuple)) else regime_tuple
                    trail_key = 'TRAILING_STOP_ATR_TRENDING' if is_trending(regime_str) else 'TRAILING_STOP_ATR_MEAN_REVERTING'
                    trailing_mult = self.config.get(trail_key, self.config.get('TRAILING_STOP_ATR', 3.0))
                    stop_distance_pct = atr_pct * trailing_mult
                else:
                    stop_distance_pct = 0.02  # 2% fallback
                # notional = risk / stop_distance (e.g. $35 risk / 1.5% stop = $2,333 position)
                notional = abs(risk_dollars) / stop_distance_pct if stop_distance_pct > 0 else 0
                cvar_weight = np.sign(risk_dollars) * notional / current_equity
                target_weights_dict[sym] = cvar_weight
                logger.debug(f"[CVaR LIVE] {sym} risk=${risk_dollars:,.0f} → notional=${notional:,.0f} "
                             f"(stop_dist={stop_distance_pct:.4f}) → weight={cvar_weight:.4f}")
        # FIX: Normalize CVaR weights to sum(abs) = 1.0 BEFORE notional cap
        # Without this, CVaR risk→notional conversion produces weights >> 1.0 per symbol,
        # and the notional cap clamps ALL of them to the same flat value, erasing CVaR differentiation
        pre_cap_abs = sum(abs(v) for v in target_weights_dict.values())
        if pre_cap_abs > 1e-8:
            for sym in target_weights_dict:
                target_weights_dict[sym] /= pre_cap_abs
            logger.debug(f"[CVaR RENORM] Pre-cap normalization: divided all weights by {pre_cap_abs:.4f}")

        # HARD NOTIONAL CAP (prevents cheap-stock explosion) — now only clips true outliers
        max_notional_per_symbol = current_equity * self.config.get('MAX_POSITION_VALUE_FRACTION', 0.20)
        for sym in target_weights_dict:
            proposed_notional = abs(target_weights_dict[sym]) * current_equity
            if proposed_notional > max_notional_per_symbol and proposed_notional > 0:
                scale = max_notional_per_symbol / proposed_notional
                old_weight = target_weights_dict[sym]
                target_weights_dict[sym] *= scale
                logger.warning(f"[NOTIONAL CAP] {sym} capped from {old_weight:.4f} ({proposed_notional:,.0f}$ notional) → {target_weights_dict[sym]:.4f}")
        # ==================== MULTIPLICATIVE ADJUSTMENTS (all applied before single renorm) ====================
        # 0. Regime-direction override: block shorts in strong trending, block longs in strong mean-reverting
        regime_override_threshold = self.config.get('REGIME_OVERRIDE_PERSISTENCE', 0.80)
        for sym in symbols:
            regime_tuple = regimes.get(sym, ('mean_reverting', 0.5))
            regime_str = regime_tuple[0] if isinstance(regime_tuple, (list, tuple)) else regime_tuple
            persistence = regime_tuple[1] if isinstance(regime_tuple, (list, tuple)) else 0.5
            weight = target_weights_dict.get(sym, 0.0)
            if persistence >= regime_override_threshold and weight != 0:
                # FIX #18: In mean-reverting regimes, shorts SHOULD be allowed — selling
                # overbought bounces is a core mean-reversion strategy. Only block positions
                # that are counter to regime logic:
                # - In trending regime with high persistence: block shorts (trends tend to continue)
                if is_trending(regime_str) and weight < 0:
                    logger.warning(f"[REGIME OVERRIDE] {sym} SHORT weight {weight:.4f} zeroed — "
                                   f"strong trending regime (persistence={persistence:.3f})")
                    target_weights_dict[sym] = 0.0

        # 1. Regime persistence scaling
        # NOTE: This boost (up to ~0.18x at persistence=1.0) is directionally correct but
        # will be attenuated by the subsequent cap + normalization step. This is a known
        # design limitation — the boost still biases allocation toward high-persistence regimes,
        # but the absolute magnitude is reduced after normalization.
        for sym in symbols:
            regime_tuple = regimes.get(sym, ('mean_reverting', 0.5))
            persistence = regime_tuple[1] if isinstance(regime_tuple, (list, tuple)) else 0.5
            # M39 FIX: Guard against KeyError if sym not in target_weights_dict
            if persistence > 0.7 and sym in target_weights_dict:
                target_weights_dict[sym] *= (1.0 + (persistence - 0.7) * 0.6)
                logger.debug(f"[REGIME PERSISTENCE] {sym} boosted by persistence {persistence:.3f}")

        # 2. Causal penalty — Apr-19 audit fix: DEFERRED to the very end of the
        # rebalance pipeline (applied AFTER min-hold, just before final renorm)
        # so that any upstream gate reading |weight| as confidence does not see
        # a causal-damped value and mis-calibrate itself. We compute the
        # penalties here (needs per-symbol features) but apply them later.
        causal_penalty_by_sym: Dict[str, float] = {}
        defer_causal = bool(self.config.get('CAUSAL_PENALTY_AFTER_GATES', True))
        if hasattr(self.signal_gen, 'portfolio_causal_manager') and self.signal_gen.portfolio_causal_manager is not None:
            try:
                from models.features import generate_features as _gen_feat
                for i, sym in enumerate(symbols):
                    sym_data = data_dict.get(sym)
                    if sym_data is not None and len(sym_data) >= 50:
                        regime_tuple = regimes.get(sym, ('mean_reverting', 0.5))
                        regime_str = regime_tuple[0] if isinstance(regime_tuple, (list, tuple)) else regime_tuple
                        full_hist = None
                        if hasattr(self.risk_manager, 'data_ingestion') and self.risk_manager.data_ingestion is not None:
                            try:
                                full_hist = self.risk_manager.data_ingestion.get_latest_data(sym)
                            except Exception:
                                pass
                        sym_features = _gen_feat(sym_data, regime_str, sym, full_hist if full_hist is not None else sym_data)
                        if sym_features is not None and sym_features.shape[0] > 0:
                            sym_obs = sym_features[-1:].reshape(1, -1)
                        else:
                            logger.debug(f"[CAUSAL DIM GUARD] {sym}: per-symbol features empty, skipping causal penalty (factor=1.0)")
                            continue
                    else:
                        logger.debug(f"[CAUSAL DIM GUARD] {sym}: insufficient data ({len(sym_data) if sym_data is not None else 0} bars), skipping causal penalty (factor=1.0)")
                        continue
                    if hasattr(sym_obs, 'ndim'):
                        if sym_obs.ndim == 1:
                            sym_obs = sym_obs.reshape(1, -1)
                        elif sym_obs.ndim > 2:
                            sym_obs = sym_obs.reshape(1, -1)
                    penalty_factor = self.signal_gen.portfolio_causal_manager.compute_penalty_factor(
                        sym_obs,
                        target_weights_dict.get(sym, 0.0)
                    )
                    causal_penalty_by_sym[sym] = float(penalty_factor)
                    if not defer_causal:
                        # Legacy path — apply inline (kept for safety toggle).
                        old_weight = target_weights_dict[sym]
                        target_weights_dict[sym] *= penalty_factor
                        if penalty_factor != 1.0:
                            logger.debug(f"[CAUSAL ADJUST] {sym}: {old_weight:.4f} * {penalty_factor:.4f} = {target_weights_dict[sym]:.4f}")
                if defer_causal:
                    logger.debug("[CAUSAL DEFER] per-symbol penalties computed; will apply after min-hold")
                else:
                    logger.info("[CAUSAL FINAL SCALING] Applied as multiplicative adjustment to CVaR weights")
            except Exception as e:
                logger.debug(f"Causal scaling compute skipped: {e}")

        # 3. Min-hold enforcement: preserve current position weight for symbols within min-hold
        # FIX #24/#25: Actually override target weight with current position weight to prevent rebalance
        if hasattr(self.signal_gen, 'broker') and hasattr(self.signal_gen.broker, 'last_entry_times'):
            for sym in symbols:
                last_entry = self.signal_gen.broker.last_entry_times.get(sym)
                if last_entry:
                    # Count only market-hours bars (9:30-16:00 ET = 6.5h = 26 fifteen-min bars/day)
                    # to avoid overnight gaps inflating bars_since
                    elapsed = datetime.now(tz=tz.gettz('UTC')) - last_entry
                    elapsed_days = elapsed.total_seconds() / 86400
                    # Each full calendar day contributes at most 26 trading bars (6.5h of 15m bars)
                    if elapsed_days <= 1:
                        bars_since = elapsed / timedelta(minutes=15)
                    else:
                        # Approximate: full trading days * 26 bars + partial day fraction
                        full_days = int(elapsed_days)
                        partial_frac = elapsed_days - full_days
                        bars_since = full_days * 26 + partial_frac * 26
                    regime = regimes.get(sym, 'mean_reverting')
                    regime_name = regime[0] if isinstance(regime, (list, tuple)) else regime
                    min_hold = self.config.get(
                        'MIN_HOLD_BARS_TRENDING' if is_trending(regime_name) else 'MIN_HOLD_BARS_MEAN_REVERTING',
                        6 if is_trending(regime_name) else 3
                    )
                    if bars_since < min_hold:
                        # FIX #25: Use positions dict to compute current weight and override target
                        pos_qty = positions.get(sym, 0)
                        price = prices.get(sym, 0)
                        if pos_qty != 0 and price > 0 and current_equity > 0:
                            current_weight = (pos_qty * price) / current_equity
                        else:
                            current_weight = target_weights_dict.get(sym, 0.0)
                        old_weight = target_weights_dict.get(sym, 0.0)
                        target_weights_dict[sym] = current_weight
                        logger.debug(f"MIN-HOLD ACTIVE {sym} (portfolio) → overriding target weight "
                                     f"{old_weight:.4f} with current position weight {current_weight:.4f} "
                                     f"({bars_since:.1f}/{min_hold} bars)")

        # ==================== DEFERRED CAUSAL PENALTY (Apr-19 audit) ===========
        # Apply the per-symbol causal penalty as the LAST multiplier before the
        # final renormalization. This ensures every upstream gate's use of
        # |weight| as confidence sees a non-causal-damped value, keeping gate
        # calibration honest. Min-hold was already applied above, so causal can
        # still override a min-held-held weight toward zero if the causal signal
        # deteriorates mid-hold (desirable — causal failure should unwind even
        # inside min-hold protection).
        if defer_causal and causal_penalty_by_sym:
            for sym, factor in causal_penalty_by_sym.items():
                if sym in target_weights_dict and abs(factor - 1.0) > 1e-6:
                    old_w = target_weights_dict[sym]
                    target_weights_dict[sym] *= factor
                    if factor != 1.0:
                        logger.debug(f"[CAUSAL POST-GATE] {sym}: {old_w:.4f} × {factor:.4f} = {target_weights_dict[sym]:.4f}")
            logger.info("[CAUSAL POST-GATE] Deferred causal penalty applied as final multiplier")

        # ==================== SINGLE FINAL RENORMALIZATION ====================
        # Re-apply notional cap after all adjustments
        for sym in target_weights_dict:
            proposed_notional = abs(target_weights_dict[sym]) * current_equity
            if proposed_notional > max_notional_per_symbol and proposed_notional > 0:
                target_weights_dict[sym] *= max_notional_per_symbol / proposed_notional

        # H15+H16 FIX: Only scale DOWN when gross exposure exceeds MAX_LEVERAGE.
        # Apr-19 audit: allow the leverage cap to flex up when the portfolio's
        # average regime persistence is high (the previous uniform scale-down
        # erased the persistence boost we just applied). Flex grows linearly
        # with avg_persistence above 0.7 up to +20% at persistence=1.0.
        total_abs = sum(abs(v) for v in target_weights_dict.values())
        base_leverage = self.config.get('MAX_LEVERAGE', 2.0)
        pers_values = []
        for _sym in symbols:
            _rt = regimes.get(_sym, ('mean_reverting', 0.5))
            pers_values.append(float(_rt[1]) if isinstance(_rt, (list, tuple)) and len(_rt) == 2 else 0.5)
        avg_persistence = float(sum(pers_values) / len(pers_values)) if pers_values else 0.5
        flex_max = float(self.config.get('LEVERAGE_PERSISTENCE_FLEX_MAX', 0.20))
        flex_start = float(self.config.get('LEVERAGE_PERSISTENCE_FLEX_START', 0.70))
        if avg_persistence > flex_start:
            flex = min(flex_max, (avg_persistence - flex_start) / (1.0 - flex_start) * flex_max)
            target_leverage = base_leverage * (1.0 + flex)
            if flex > 0:
                logger.debug(f"[LEVERAGE FLEX] avg_persistence={avg_persistence:.2f} → "
                             f"cap flexed {base_leverage:.2f} → {target_leverage:.2f}")
        else:
            target_leverage = base_leverage
        if total_abs > target_leverage and total_abs > 1e-8:
            scale = target_leverage / total_abs
            for sym in target_weights_dict:
                target_weights_dict[sym] *= scale
        # M40 FIX: Re-enforce per-symbol cap after leverage normalization
        max_weight = self.config.get('MAX_POSITION_VALUE_FRACTION', 0.20)
        for sym in target_weights_dict:
            target_weights_dict[sym] = max(-max_weight, min(max_weight, target_weights_dict[sym]))
        return target_weights_dict

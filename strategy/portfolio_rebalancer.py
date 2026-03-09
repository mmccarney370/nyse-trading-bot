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
        precomputed_env: PortfolioEnv = None # ISSUE #6: Accept persistent env from bot.py
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
            precomputed_env.data_dict = {sym: df.copy() for sym, df in data_dict.items()}
            obs = precomputed_env._get_observation() # Reuse precomputed timeline/regimes/features
        else:
            logger.debug("Creating new PortfolioEnv (fallback - no persistent env)")
            precomputed_env = PortfolioEnv(
                data_dict=data_dict,
                symbols=symbols,
                initial_balance=current_equity,
                max_leverage=self.config.get('MAX_LEVERAGE', 3.0)
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
            regimes=regimes # ← NEW: forward regimes dict so persistence can be averaged
        )
        # ==================== END NEW SYMMETRY ====================
        # Apply CVaR dollar risk to final weights
        for sym in symbols:
            if sym in dollar_risk_alloc and dollar_risk_alloc[sym] != 0:
                price = prices.get(sym)
                if price and price > 0:
                    cvar_weight = dollar_risk_alloc[sym] / current_equity
                    target_weights_dict[sym] = cvar_weight
                    logger.debug(f"[CVaR LIVE] {sym} adjusted to {cvar_weight:.4f} (dollar risk ${dollar_risk_alloc[sym]:,.0f})")
        # HARD NOTIONAL CAP (prevents cheap-stock explosion)
        max_notional_per_symbol = current_equity * self.config.get('MAX_POSITION_VALUE_FRACTION', 0.20)
        for sym in target_weights_dict:
            proposed_notional = abs(target_weights_dict[sym]) * current_equity
            if proposed_notional > max_notional_per_symbol and proposed_notional > 0:
                scale = max_notional_per_symbol / proposed_notional
                old_weight = target_weights_dict[sym]
                target_weights_dict[sym] *= scale
                logger.warning(f"[NOTIONAL CAP] {sym} capped from {old_weight:.4f} ({proposed_notional:,.0f}$ notional) → {target_weights_dict[sym]:.4f}")
        # ==================== CRIT-11 FIX: REGIME PERSISTENCE SCALING ====================
        for sym in symbols:
            regime_tuple = regimes.get(sym, ('mean_reverting', 0.5))
            persistence = regime_tuple[1] if isinstance(regime_tuple, (list, tuple)) else 0.5
            if persistence > 0.7: # trending → higher conviction weight
                target_weights_dict[sym] *= (1.0 + (persistence - 0.7) * 0.6)
                logger.debug(f"[REGIME PERSISTENCE] {sym} boosted by persistence {persistence:.3f}")
        # renormalize target_weights_dict after notional cap so total portfolio weight sums to 1.0
        total_abs_weight = sum(abs(v) for v in target_weights_dict.values())
        if total_abs_weight > 0:
            scale = 1.0 / total_abs_weight
            for sym in target_weights_dict:
                target_weights_dict[sym] *= scale
        # ==================== CRIT-11 FIX: CAUSAL PENALTY (now BEFORE final renorm) ====================
        if hasattr(self.signal_gen, 'portfolio_causal_manager') and self.signal_gen.portfolio_causal_manager is not None:
            try:
                # FIX: Use causal as a MULTIPLICATIVE adjustment, not a full replacement
                # This preserves the CVaR-optimized weights while applying causal penalty/boost
                for i, sym in enumerate(symbols):
                    penalty_factor = self.signal_gen.portfolio_causal_manager.compute_penalty_factor(
                        obs.reshape(1, -1) if hasattr(obs, 'reshape') else obs,
                        target_weights_dict.get(sym, 0.0)
                    )
                    old_weight = target_weights_dict[sym]
                    target_weights_dict[sym] *= penalty_factor
                    if penalty_factor != 1.0:
                        logger.debug(f"[CAUSAL ADJUST] {sym}: {old_weight:.4f} * {penalty_factor:.4f} = {target_weights_dict[sym]:.4f}")
                logger.info(f"[CAUSAL FINAL SCALING] Applied as multiplicative adjustment to CVaR weights")
                # Renormalize after causal adjustment
                total_abs_weight_causal = sum(abs(v) for v in target_weights_dict.values())
                if total_abs_weight_causal > 0:
                    scale_causal = 1.0 / total_abs_weight_causal
                    for sym in target_weights_dict:
                        target_weights_dict[sym] *= scale_causal
                logger.debug("[CAUSAL RENORM] Final weights renormalized after causal scaling")
            except Exception as e:
                logger.debug(f"Final causal scaling skipped: {e}")
        # ==================== CRIT-11 FIX: MIN-HOLD ENFORCEMENT (Gemini-tuned, regime-aware) ====================
        if hasattr(self.signal_gen, 'broker') and hasattr(self.signal_gen.broker, 'last_entry_times'):
            for sym in symbols:
                last_entry = self.signal_gen.broker.last_entry_times.get(sym)
                if last_entry:
                    bars_since = (datetime.now(tz=tz.gettz('UTC')) - last_entry) / timedelta(minutes=15)
                    regime = regimes.get(sym, 'mean_reverting')
                    regime_name = regime[0] if isinstance(regime, (list, tuple)) else regime
                    if regime_name == 'trending':
                        min_hold = self.config.get('MIN_HOLD_BARS_TRENDING', 44)
                    else:
                        min_hold = self.config.get('MIN_HOLD_BARS_MEAN_REVERTING', 24)
                    if bars_since < min_hold:
                        current_weight = target_weights_dict.get(sym, 0.0)
                        if current_weight != 0:  # only log if we tried to change
                            logger.debug(f"MIN-HOLD ACTIVE {sym} (portfolio) → skipping rebalance (Gemini-tuned {min_hold} bars)")
                        sym_price = prices.get(sym)
                        if sym_price and sym_price > 0:
                            target_weights_dict[sym] = positions.get(sym, 0) * sym_price / current_equity
                        # else: keep current target weight (no price to compute position weight)
        # Final renormalization after min-hold adjustments
        total_abs = sum(abs(v) for v in target_weights_dict.values())
        if total_abs > 0:
            for sym in target_weights_dict:
                target_weights_dict[sym] /= total_abs
        # Return final normalized weights (ready for min-hold / order placement in bot.py)
        return target_weights_dict

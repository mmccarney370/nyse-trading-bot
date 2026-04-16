# strategy/liquidity_scaler.py
"""
LIQ — Liquidity-Scaled Position Sizing.

Every symbol has an average daily dollar volume (ADV). When our position notional
starts to rival ADV, market-impact slippage becomes non-negligible and fills
degrade. Classic institutional rule of thumb: stay under 1% of ADV to keep
impact below a few basis points.

For this bot at $30K equity, participation is tiny (<0.01% of ADV). But:
- Equity grows over time (the point of trading)
- Extended-hours liquidity is 10-20× thinner than RTH
- Some small-cap universe members (SMCI, SOFI) have thinner books than we think

This module computes a `participation_rate = position_notional / ADV` per symbol
and scales weights DOWN when participation exceeds tolerance. It's pure defense —
strictly additive (multiplier ≤ 1.0) and dormant for safe positions.

Formula:
    adv_s            = mean(daily_volume_s) × current_price_s   (daily dollar volume)
    participation_s  = |w_s| · equity / adv_s
    if participation_s ≤ warn_threshold:    mult = 1.0
    if participation_s ≥ hard_threshold:    mult = min_mult
    else:                                    linear interpolation
    extended_hours:  thresholds / eh_factor (tighter in thin liquidity)
"""
from __future__ import annotations

import logging
from datetime import datetime, time
from dateutil import tz
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ET = tz.gettz("America/New_York")


def _is_extended_hours(now: Optional[datetime] = None) -> bool:
    """True if we're outside NYSE RTH (9:30 AM - 4:00 PM ET).
    Extended hours = pre-market (4:00-9:30) and after-hours (16:00-20:00)."""
    if now is None:
        now = datetime.now(tz=_ET)
    else:
        now = now.astimezone(_ET) if now.tzinfo else now.replace(tzinfo=_ET)
    t = now.time()
    return t < time(9, 30) or t >= time(16, 0)


def compute_adv(df: pd.DataFrame, lookback_days: int = 20,
                bars_per_day: int = 26) -> float:
    """Compute average daily dollar volume (ADV in dollars) from 15-min bars.
    Uses last `lookback_days` worth of data."""
    if df is None or len(df) < bars_per_day:
        return 0.0
    total_bars = lookback_days * bars_per_day
    tail = df.tail(total_bars) if len(df) >= total_bars else df
    if tail.empty or 'volume' not in tail.columns:
        return 0.0
    # Dollar volume per bar
    dollar_vol_per_bar = (tail['close'].astype(float) * tail['volume'].astype(float))
    # Average per day = total bar-dollars / days
    days_captured = max(1, len(tail) / bars_per_day)
    adv = float(dollar_vol_per_bar.sum() / days_captured)
    return adv


def compute_liquidity_multipliers(
    target_weights: Mapping[str, float],
    data_dict: Mapping[str, pd.DataFrame],
    equity: float,
    warn_threshold: float = 0.001,   # 0.1% of ADV → no action
    hard_threshold: float = 0.01,    # 1% of ADV → min_mult
    min_mult: float = 0.3,
    extended_hours_factor: float = 5.0,  # thresholds tightened by this factor off-hours
    now: Optional[datetime] = None,
) -> Dict[str, tuple]:
    """Return {symbol: (multiplier, participation_rate, reason)}.

    Participation rates well below `warn_threshold` return multiplier = 1.0.
    Between warn and hard, linear interpolation from 1.0 → min_mult.
    Extended-hours trading tightens both thresholds by `extended_hours_factor`
    (default 5× — means hard_threshold becomes 0.2% of ADV pre-market).
    """
    if equity <= 0:
        return {s: (1.0, 0.0, "no-equity") for s in target_weights}

    eh = _is_extended_hours(now)
    if eh:
        warn_threshold = warn_threshold / extended_hours_factor
        hard_threshold = hard_threshold / extended_hours_factor

    out: Dict[str, tuple] = {}
    for sym, w in target_weights.items():
        if w == 0.0:
            out[sym] = (1.0, 0.0, "flat")
            continue
        df = data_dict.get(sym)
        if df is None or df.empty:
            out[sym] = (1.0, 0.0, "no-data")
            continue
        try:
            adv = compute_adv(df)
            if adv <= 0:
                out[sym] = (1.0, 0.0, "adv-zero")
                continue
            notional = abs(w) * equity
            participation = notional / adv
            if participation <= warn_threshold:
                mult = 1.0
                reason = f"safe({participation*10000:.1f}bp)"
            elif participation >= hard_threshold:
                mult = min_mult
                reason = f"HARD({participation*10000:.0f}bp≥{hard_threshold*10000:.0f}bp)"
                if eh:
                    reason = "EH-" + reason
            else:
                # Linear interpolation
                span = hard_threshold - warn_threshold
                excess = participation - warn_threshold
                mult = 1.0 - (1.0 - min_mult) * (excess / span)
                reason = f"scale({participation*10000:.1f}bp)"
                if eh:
                    reason = "EH-" + reason
            out[sym] = (float(mult), float(participation), reason)
        except Exception as e:
            logger.debug(f"[LIQ] {sym} computation failed: {e}")
            out[sym] = (1.0, 0.0, f"err({e})")
    return out


def log_summary(multipliers: Dict[str, tuple]) -> None:
    """Emit a one-liner if any symbol is being scaled down."""
    active = [(s, m, p, r) for s, (m, p, r) in multipliers.items() if m < 0.995]
    if not active:
        return
    active.sort(key=lambda x: x[1])  # smallest mult first
    parts = [f"{s}:{m:.2f}×({p*10000:.0f}bp)" for s, m, p, _r in active[:5]]
    logger.info(f"[LIQ-SCALE] {' | '.join(parts)}")

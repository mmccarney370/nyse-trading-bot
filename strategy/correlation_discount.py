# strategy/correlation_discount.py
"""
AC — Correlation-aware sizing ("crowding discount").

The 8-symbol universe is tech/fintech-heavy: AMD, NVDA, SMCI, TSLA often move
together. When 4 of them are all long, effective gross exposure >> sum(abs(w))
because one bad day in semis hits all four positions simultaneously.

Fix: for each symbol's target weight, discount it by how correlated it is to
OTHER symbols that are taking the SAME directional bet. Symbols that act like
unique bets keep their full weight; crowded positions get scaled down.

Formula per symbol s with target weight w_s:
    peers      = [other symbols with same-sign non-zero weight]
    avg_corr   = mean(corr(s, p) for p in peers) on 60-day daily returns
    crowding   = max(0, avg_corr - threshold)   # threshold default 0.5
    discount   = 1 - strength * crowding         # strength default 0.5
    w_s_new    = w_s * max(discount, min_factor) # min_factor default 0.4

Examples at threshold=0.5, strength=0.5:
    avg_corr=0.3   → no discount (1.0×)
    avg_corr=0.7   → 1 - 0.5*0.2 = 0.9×  (gentle)
    avg_corr=0.9   → 1 - 0.5*0.4 = 0.8×  (strong reduction)
    avg_corr=1.0   → 1 - 0.5*0.5 = 0.75× (cap)

Hard floor prevents over-suppression — keeps at least min_factor of the signal.
"""
from __future__ import annotations

import logging
from typing import Dict, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_correlation_matrix(
    symbol_dataframes: Mapping[str, pd.DataFrame],
    lookback_bars: int = 60 * 26,  # ~60 days of 15-min bars
    min_periods: int = 20,
) -> pd.DataFrame:
    """Returns a correlation matrix over recent returns. Empty frame if data is
    sparse or fewer than 2 symbols have valid data."""
    series = {}
    for sym, df in symbol_dataframes.items():
        if df is None or len(df) < min_periods + 2:
            continue
        tail = df.tail(lookback_bars)
        rets = tail['close'].pct_change().dropna()
        if len(rets) >= min_periods:
            series[sym] = rets
    if len(series) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(series).dropna(how='all')
    if df.empty or df.shape[1] < 2:
        return pd.DataFrame()
    return df.corr(min_periods=min_periods)


def apply_crowding_discount(
    target_weights: Dict[str, float],
    corr_matrix: pd.DataFrame,
    threshold: float = 0.5,
    strength: float = 0.5,
    min_factor: float = 0.4,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Apply correlation-based discount to each symbol's weight.

    Returns (new_weights, discount_factors) where discount_factors[s] is the
    multiplier applied to symbol s's weight. 1.0 means no change.
    """
    if corr_matrix.empty:
        return dict(target_weights), {s: 1.0 for s in target_weights}

    discounts: Dict[str, float] = {s: 1.0 for s in target_weights}
    new_weights: Dict[str, float] = dict(target_weights)

    for sym, w in target_weights.items():
        if w == 0.0 or sym not in corr_matrix.index:
            continue
        direction = 1 if w > 0 else -1
        # Find peers with same-sign non-zero weight
        peers = [
            p for p, pw in target_weights.items()
            if p != sym and pw != 0.0 and ((1 if pw > 0 else -1) == direction)
            and p in corr_matrix.columns
        ]
        if not peers:
            continue
        try:
            peer_corrs = corr_matrix.loc[sym, peers].dropna().values
        except Exception as e:
            logger.debug(f"[CROWDING] {sym} correlation lookup failed: {e}")
            continue
        if len(peer_corrs) == 0:
            continue
        avg_corr = float(np.mean(peer_corrs))
        crowding = max(0.0, avg_corr - threshold)
        discount = max(min_factor, 1.0 - strength * crowding)
        if discount < 1.0:
            new_weights[sym] = w * discount
            discounts[sym] = discount

    return new_weights, discounts


def log_summary(discounts: Dict[str, float]) -> None:
    """Emit a one-liner for operator visibility if any discount actually fired."""
    active = {s: d for s, d in discounts.items() if d < 0.995}
    if not active:
        return
    parts = [f"{s}:{d:.2f}×" for s, d in sorted(active.items(), key=lambda x: x[1])]
    logger.info(f"[CROWDING-DISCOUNT] {' | '.join(parts)}")

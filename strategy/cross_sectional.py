# strategy/cross_sectional.py
"""
A1 — Cross-sectional momentum + volume gate.

Instead of rotating the universe daily (which requires a PPO retrain to match
the new symbol distribution), we rank the existing universe cross-sectionally
every trading day and apply a multiplicative gate to each symbol's target weight.
Top-ranked symbols (today's best momentum + breadth) get a small boost; bottom-
ranked get substantial dampening.

Why this over daily rotation:
- No retrain required (PPO observation shape stays constant)
- Preserves what the PPO learned about the universe's feature distribution
- Still captures "be where today's alpha is" without blowing up model fidelity
- Cheap to compute (1D bars × universe size, seconds at most)

Scoring components (all z-scored across the current universe):
1. 5-day return — classic short-horizon momentum
2. 20-day return vs 60-day return (acceleration)
3. Volume momentum — recent volume / long-term avg volume
4. Negative of 10-day drawdown — penalize recent pullback severity

Result: per-symbol multiplier in [min_mult, max_mult], centered at 1.0 on median.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_z(values: pd.Series) -> pd.Series:
    """Z-score with safe fallback on zero/nan std."""
    v = values.astype(float)
    mu = float(v.mean())
    sigma = float(v.std())
    if not math.isfinite(sigma) or sigma < 1e-9:
        return pd.Series(0.0, index=values.index)
    return (v - mu) / sigma


def compute_cross_sectional_scores(
    symbol_dataframes: Dict[str, pd.DataFrame],
    momentum_short_bars: int = 5 * 26,  # 5 trading days of 15-min bars
    momentum_long_bars: int = 60 * 26,
    volume_window: int = 20 * 26,
) -> Dict[str, float]:
    """Return per-symbol ranking score (higher = better today).
    Input: dict of symbol → OHLCV DataFrame (15-min bars, recent tail)
    Output: dict of symbol → float score (roughly ~-2 to +2 range after z-scoring).
    Symbols with insufficient data silently default to 0.0 (neutral)."""
    rows = {}
    for sym, df in symbol_dataframes.items():
        if df is None or len(df) < max(momentum_long_bars, volume_window) // 2:
            rows[sym] = {"ret_short": 0.0, "accel": 0.0, "vol_mom": 0.0, "dd": 0.0}
            continue
        close = df['close'].astype(float)
        volume = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(1.0, index=df.index)
        try:
            # 5-day momentum
            if len(close) >= momentum_short_bars + 1:
                ret_short = float(close.iloc[-1] / close.iloc[-momentum_short_bars] - 1.0)
            else:
                ret_short = float(close.iloc[-1] / close.iloc[0] - 1.0)
            # 20d vs 60d acceleration
            if len(close) >= momentum_long_bars:
                ret_20d = float(close.iloc[-1] / close.iloc[-20 * 26] - 1.0)
                ret_60d = float(close.iloc[-1] / close.iloc[-momentum_long_bars] - 1.0)
                accel = ret_20d - ret_60d
            else:
                accel = 0.0
            # Volume momentum: short-window avg / long-window avg
            short_vw = max(26 * 2, len(volume) // 20)
            if len(volume) >= volume_window:
                vol_mom = float(volume.tail(short_vw).mean() / volume.tail(volume_window).mean())
            else:
                vol_mom = 1.0
            # Drawdown: most-recent close vs 10-day peak
            lookback = min(10 * 26, len(close))
            recent_peak = float(close.tail(lookback).max())
            dd = float(close.iloc[-1] / recent_peak - 1.0) if recent_peak > 0 else 0.0
        except Exception as e:
            logger.debug(f"[CS-SCORE] {sym} component failed: {e}")
            ret_short = accel = vol_mom = dd = 0.0
        rows[sym] = {
            "ret_short": ret_short,
            "accel": accel,
            "vol_mom": vol_mom,
            "dd": dd,
        }

    df = pd.DataFrame(rows).T.fillna(0.0)
    if df.empty:
        return {}
    # Z-score each component cross-sectionally
    z_ret = _safe_z(df['ret_short'])
    z_accel = _safe_z(df['accel'])
    z_vol = _safe_z(df['vol_mom'])
    z_dd = _safe_z(df['dd'])  # dd is negative for bigger drawdowns → we'll ADD (not subtract) so bigger DD → lower score
    score = (
        1.0 * z_ret +
        0.6 * z_accel +
        0.4 * z_vol +
        0.8 * z_dd
    )
    return {sym: float(score.loc[sym]) for sym in score.index}


def build_multipliers(
    scores: Dict[str, float],
    max_mult: float = 1.25,
    min_mult: float = 0.50,
    neutral_band: float = 0.25,
) -> Dict[str, float]:
    """Convert raw scores into per-symbol multipliers centered at 1.0.

    - Top-tercile score → max_mult (boost)
    - Bottom-tercile score → min_mult (dampen)
    - Middle tercile → linearly interpolated, with a `neutral_band` flat zone
      around 0 to avoid whipsawing on noise.
    """
    if not scores:
        return {}
    values = np.asarray(list(scores.values()))
    if len(values) < 2 or values.std() < 1e-9:
        return {s: 1.0 for s in scores}
    hi = float(np.quantile(values, 0.67))
    lo = float(np.quantile(values, 0.33))
    mults = {}
    for sym, s in scores.items():
        if -neutral_band <= s <= neutral_band:
            mults[sym] = 1.0
            continue
        if s >= hi:
            mults[sym] = max_mult
        elif s <= lo:
            mults[sym] = min_mult
        else:
            # Linear interpolation between lo/1 and hi/1 to 1.0 at 0
            if s > 0:
                frac = (s - neutral_band) / max(hi - neutral_band, 1e-9)
                mults[sym] = 1.0 + (max_mult - 1.0) * float(np.clip(frac, 0.0, 1.0))
            else:
                frac = (abs(s) - neutral_band) / max(abs(lo) - neutral_band, 1e-9)
                mults[sym] = 1.0 - (1.0 - min_mult) * float(np.clip(frac, 0.0, 1.0))
    return mults


def log_summary(scores: Dict[str, float], mults: Dict[str, float]) -> None:
    """Emit a one-line summary of today's ranking for operator visibility."""
    try:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        parts = [f"{s}:{sc:+.2f}×{mults.get(s, 1.0):.2f}" for s, sc in ranked]
        logger.info(f"[CS-MOMENTUM] {' | '.join(parts)}")
    except Exception:
        pass

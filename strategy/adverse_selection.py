# strategy/adverse_selection.py
"""
B2 — Adverse Selection Detector.

Adverse selection happens when your fill price moves AGAINST you systematically
shortly after the fill — you were "picked off" by better-informed counterparties.
In HFT/MM literature it's the central execution-quality metric.

Approach:
1. On every fill, log (symbol, side, fill_price, fill_time) via `record_fill`.
2. For each recorded fill, periodically sample the current mid (close) at
   T+1min, T+5min, T+15min, T+30min; compute signed drift relative to side:
       drift = (price_at_T+N - fill_price) / fill_price × sign(side)
   A NEGATIVE signed drift means price moved against us immediately after the
   fill → toxic.
3. Roll by symbol: the past 20 fills' mean signed drift at 5min → AVA score.
   If AVA_score < threshold (e.g. -0.002 = -20bp consistent drift against),
   flag the symbol as toxic → external consumers (signal generator) can use
   `get_toxicity_penalty(symbol)` to dampen that symbol's weight.

Why roll by symbol, not globally: execution quality is symbol- and time-of-day
specific. SMCI fills at 9:30 may be toxic while SOFI fills mid-day may be fine.

Persists to adverse_selection.pkl. Non-intrusive: failures always return neutral.
"""
from __future__ import annotations

import logging
import os
import pickle
import threading
from collections import deque
from datetime import datetime, timedelta
from dateutil import tz
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_UTC = tz.gettz("UTC")


class AdverseSelectionDetector:
    """Track post-fill drift per symbol to detect execution toxicity."""

    SAMPLE_OFFSETS_MIN = [1, 5, 15, 30]   # minutes to sample after fill

    def __init__(
        self,
        persist_path: str = "adverse_selection.pkl",
        rolling_window: int = 20,
        primary_offset_min: int = 5,
    ):
        self.persist_path = persist_path
        self.rolling_window = rolling_window
        self.primary_offset_min = primary_offset_min
        # pending fills waiting for samples: list of dicts
        self._pending: List[dict] = []
        # completed drift samples by symbol: {sym: deque([(t_offset_min → signed_drift)])}
        # We only store the primary offset result in the rolling window for simplicity.
        self._rolled: Dict[str, deque] = {}
        self._lock = threading.RLock()
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record_fill(self, symbol: str, side: int, fill_price: float,
                    fill_time: Optional[datetime] = None) -> None:
        """Call on every entry fill. side=+1 for long, -1 for short."""
        if fill_time is None:
            fill_time = datetime.now(tz=_UTC)
        with self._lock:
            self._pending.append({
                "symbol": symbol,
                "side": int(side),
                "fill_price": float(fill_price),
                "fill_time": fill_time,
                "samples": {},   # offset_min → price
            })
        self._save()

    def update_prices(self, latest_prices: Dict[str, float],
                      now: Optional[datetime] = None) -> int:
        """Fill in sample prices for any pending fills whose time offsets have
        elapsed. Returns count of newly completed samples.
        Called from the monitor/trading loop every cycle."""
        if now is None:
            now = datetime.now(tz=_UTC)
        completed = 0
        with self._lock:
            still_pending = []
            for p in self._pending:
                sym = p["symbol"]
                price = latest_prices.get(sym)
                if price is None or price <= 0:
                    still_pending.append(p)
                    continue
                elapsed = (now - p["fill_time"]).total_seconds() / 60.0
                done = True
                for off in self.SAMPLE_OFFSETS_MIN:
                    if off in p["samples"]:
                        continue
                    if elapsed >= off:
                        # Compute signed drift
                        drift = (price - p["fill_price"]) / p["fill_price"] * p["side"]
                        p["samples"][off] = {"price": price, "drift": float(drift)}
                        if off == self.primary_offset_min:
                            self._rolled.setdefault(sym, deque(maxlen=self.rolling_window)).append(float(drift))
                            completed += 1
                    else:
                        done = False
                if not done and elapsed < max(self.SAMPLE_OFFSETS_MIN) + 2:
                    still_pending.append(p)
                # else: fully done OR too stale → drop
            self._pending = still_pending
        if completed:
            self._save()
            self._log_any_toxic()
        return completed

    def get_toxicity_score(self, symbol: str) -> float:
        """Return the mean signed drift at primary offset across the last N fills.
        NEGATIVE = toxic (consistent adverse move after fills).
        Returns 0.0 if insufficient data."""
        with self._lock:
            q = self._rolled.get(symbol)
        if not q or len(q) < 3:
            return 0.0
        return float(sum(q) / len(q))

    def get_toxicity_penalty(self, symbol: str,
                             threshold: float = -0.002,
                             max_penalty: float = 0.5) -> Tuple[float, str]:
        """Return (multiplier, reason). Multiplier=1.0 when safe,
        decreasing linearly as toxicity worsens past threshold.
        At (2×threshold), multiplier = 1 - max_penalty."""
        score = self.get_toxicity_score(symbol)
        if score >= threshold:
            return 1.0, f"clean({score:+.4f})"
        # Score is negative and below threshold; linearly scale to max_penalty
        excess = abs(score - threshold)
        scale = min(1.0, excess / abs(threshold))
        mult = 1.0 - max_penalty * scale
        mult = max(1.0 - max_penalty, mult)
        return mult, f"toxic({score:+.4f})"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_any_toxic(self) -> None:
        """Emit one-line summary if any symbol crossed into toxic territory."""
        toxic = []
        with self._lock:
            for sym, q in self._rolled.items():
                if len(q) >= 3:
                    score = sum(q) / len(q)
                    if score < -0.001:
                        toxic.append((sym, score, len(q)))
        if toxic:
            toxic.sort(key=lambda x: x[1])  # worst first
            parts = [f"{s}:{sc:+.4f}(n={n})" for s, sc, n in toxic[:5]]
            logger.info(f"[ADVERSE-SEL] Toxic fills detected — {' | '.join(parts)}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(self):
        try:
            import tempfile, shutil
            dir_name = os.path.dirname(self.persist_path) or "."
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, dir=dir_name, suffix=".tmp") as tmp:
                with self._lock:
                    pickle.dump(
                        {
                            "pending": self._pending,
                            "rolled": {k: list(v) for k, v in self._rolled.items()},
                            "rolling_window": self.rolling_window,
                            "primary_offset_min": self.primary_offset_min,
                        },
                        tmp,
                    )
                tmp.flush()
                os.fsync(tmp.fileno())
            shutil.move(tmp.name, self.persist_path)
        except Exception as e:
            logger.debug(f"[ADVERSE-SEL] save failed: {e}")

    def _load(self):
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
            with self._lock:
                self._pending = data.get("pending", []) or []
                rolled = data.get("rolled", {}) or {}
                self._rolled = {
                    k: deque(v, maxlen=self.rolling_window)
                    for k, v in rolled.items()
                }
            logger.info(
                f"[ADVERSE-SEL] Loaded state: {len(self._pending)} pending, "
                f"{sum(len(q) for q in self._rolled.values())} completed drift samples"
            )
        except Exception as e:
            logger.debug(f"[ADVERSE-SEL] load failed: {e}")

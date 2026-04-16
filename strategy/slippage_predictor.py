# strategy/slippage_predictor.py
"""
ESP — Execution Slippage Predictor.

Every entry has a realized slippage = |fill_price - limit_price| / limit_price.
Bot has been logging it in group.slippage but never using it to gate future
entries. This module:

1. Collects slippage samples (symbol, hour-of-day, direction, size_usd_log, slip_bps).
2. Fits simple grouped averages (cheap, robust on small N).
3. At entry time, predicts expected slippage bps. If predicted > threshold and
   alpha is thin, the gate can veto or dampen.

Why grouped averages vs linear regression: our N is low (<50 fills/symbol),
feature space is small, and slippage distributions are fat-tailed. Group means
with fallbacks produce well-calibrated estimates faster than gradient models.

Fallback cascade when data is sparse:
    (symbol, hour-bucket, size-bucket) → most specific
    (symbol, hour-bucket)              →
    (symbol,)                          →
    (global)                           → coarse fallback
    median across dataset              → last resort
"""
from __future__ import annotations

import logging
import os
import pickle
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _hour_bucket(hour: int) -> str:
    """Bucket hour-of-day to reduce sparsity. Opening/close/mid buckets."""
    if 9 <= hour < 11:
        return "open"
    if 11 <= hour < 14:
        return "mid"
    if 14 <= hour < 16:
        return "close"
    return "offhr"


def _size_bucket(size_usd: float) -> str:
    """Bucket order notional. Large orders often have more slippage."""
    if size_usd < 1000:
        return "xs"
    if size_usd < 5000:
        return "s"
    if size_usd < 15000:
        return "m"
    return "l"


class SlippagePredictor:
    """Learn expected per-fill slippage conditioned on (symbol, hour, size)."""

    def __init__(
        self,
        persist_path: str = "slippage_predictor.pkl",
        max_samples: int = 2000,
    ):
        self.persist_path = persist_path
        self.max_samples = max_samples
        # samples: list of dicts {symbol, hour_bucket, size_bucket, slip_bps, t}
        self._samples: List[Dict] = []
        # precomputed group means {(sym, hour, size): (mean_bps, n)}
        self._groups: Dict[Tuple, Tuple[float, int]] = {}
        self._global_median: float = 5.0  # conservative prior: 5bp
        self._lock = threading.RLock()
        self._load()
        self._recompute_groups()

    # ------------------------------------------------------------------
    # Record samples
    # ------------------------------------------------------------------
    def record(
        self,
        symbol: str,
        slip_bps: float,
        hour: Optional[int] = None,
        size_usd: Optional[float] = None,
        direction: int = 1,
    ) -> None:
        """Called from the fill handler. slip_bps is in basis points
        (|fill_price - limit| / limit * 10000)."""
        if slip_bps is None or not np.isfinite(slip_bps) or slip_bps < 0:
            return
        # Outlier cap: 500bp = 5% — anything higher is probably a data issue
        slip_bps = float(min(slip_bps, 500.0))
        h = hour if hour is not None else datetime.now().hour
        sz = float(size_usd) if size_usd else 0.0
        with self._lock:
            self._samples.append({
                "symbol": symbol,
                "hour_bucket": _hour_bucket(h),
                "size_bucket": _size_bucket(sz),
                "slip_bps": slip_bps,
                "direction": int(direction),
            })
            # Trim oldest when exceeding cap
            if len(self._samples) > self.max_samples:
                self._samples = self._samples[-self.max_samples :]
        # Lazy recompute: every 10 new samples
        if len(self._samples) % 10 == 0:
            self._recompute_groups()
        self._save()

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict_bps(
        self,
        symbol: str,
        hour: Optional[int] = None,
        size_usd: Optional[float] = None,
    ) -> Tuple[float, str]:
        """Predict expected slippage in basis points. Returns (bps, source)."""
        h = hour if hour is not None else datetime.now().hour
        sz = size_usd if size_usd is not None else 0.0
        hb = _hour_bucket(h)
        sb = _size_bucket(sz)
        with self._lock:
            # Fallback cascade, most-specific to least
            for key, source in (
                ((symbol, hb, sb), "sym+hour+size"),
                ((symbol, hb, None), "sym+hour"),
                ((symbol, None, None), "sym"),
                ((None, hb, sb), "hour+size"),
                ((None, hb, None), "hour"),
                ((None, None, None), "global"),
            ):
                hit = self._groups.get(key)
                if hit is not None and hit[1] >= 3:
                    return float(hit[0]), source
            return float(self._global_median), "prior"

    def should_veto(
        self,
        symbol: str,
        expected_alpha_bps: float,
        hour: Optional[int] = None,
        size_usd: Optional[float] = None,
        edge_safety_multiple: float = 1.2,
    ) -> Tuple[bool, float, str]:
        """Return (veto, predicted_bps, reason). Veto when predicted slippage
        exceeds expected alpha by `edge_safety_multiple`. Only vetos when we
        actually have enough samples — insufficient data → no veto."""
        pred, source = self.predict_bps(symbol, hour, size_usd)
        if source == "prior":
            return False, pred, f"insufficient-data({pred:.1f}bp)"
        threshold = expected_alpha_bps * edge_safety_multiple
        if pred > threshold:
            return True, pred, f"pred={pred:.1f}bp > {threshold:.1f}bp ({source})"
        return False, pred, f"pred={pred:.1f}bp ≤ {threshold:.1f}bp ({source})"

    # ------------------------------------------------------------------
    # Internal — group mean computation
    # ------------------------------------------------------------------
    def _recompute_groups(self) -> None:
        with self._lock:
            groups: Dict[Tuple, List[float]] = defaultdict(list)
            for s in self._samples:
                sym = s["symbol"]
                hb = s["hour_bucket"]
                sb = s["size_bucket"]
                slip = s["slip_bps"]
                groups[(sym, hb, sb)].append(slip)
                groups[(sym, hb, None)].append(slip)
                groups[(sym, None, None)].append(slip)
                groups[(None, hb, sb)].append(slip)
                groups[(None, hb, None)].append(slip)
                groups[(None, None, None)].append(slip)
            self._groups = {
                k: (float(np.mean(vs)), len(vs)) for k, vs in groups.items()
            }
            if self._samples:
                self._global_median = float(np.median([s["slip_bps"] for s in self._samples]))

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
                            "samples": self._samples,
                            "global_median": self._global_median,
                        },
                        tmp,
                    )
                tmp.flush()
                os.fsync(tmp.fileno())
            shutil.move(tmp.name, self.persist_path)
        except Exception as e:
            logger.debug(f"[SLIPPAGE] save failed: {e}")

    def _load(self):
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
            with self._lock:
                self._samples = data.get("samples", []) or []
                self._global_median = float(data.get("global_median", 5.0))
            logger.info(f"[SLIPPAGE] Loaded {len(self._samples)} slippage samples from disk")
        except Exception as e:
            logger.debug(f"[SLIPPAGE] load failed: {e}")

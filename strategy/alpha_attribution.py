# strategy/alpha_attribution.py
"""
ALPHA-ATTR — Per-Trade Alpha Attribution Logger.

Every closed trade gets a structured JSON log line recording:
  - The FINAL weight the bot placed
  - The BASELINE weight (what raw PPO would have traded)
  - Every LAYER's multiplier that transformed baseline → final
  - The REALIZED return
  - MFE/MAE fingerprint
  - Exit reason (stop, TP, safe-close)
  - Trade duration + regime at open/close

Why this matters: with 18+ multiplicative layers, when a trade loses $50 we need
to know which layer pushed it out of neutral. Historical log lets us compute:
  - "What is my WR conditional on BPS multiplier > 1.2?"
  - "Does meta-filter actually reject losers?"
  - "Do trending-up aligned REX positions outperform counter-trend?"

Without this logger, those questions are unanswerable.

Design:
  - `record_attribution(symbol, layers_dict)` is called by signals.py at the end
    of generate_portfolio_actions. Stored in memory keyed by (symbol, timestamp)
    until the trade closes.
  - `emit_on_close(symbol, realized_return, mfe, mae, exit_reason, ...)` is
    called by stream.py on trade close. Finds the matching entry attribution,
    merges with realized data, and emits one structured JSON line to disk.
  - Persists to `alpha_attribution_log.jsonl` — jq/grep-friendly.

Strictly additive, failure-safe (any error in attribution never blocks trading).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from datetime import datetime
from dateutil import tz
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_UTC = tz.gettz("UTC")


class AlphaAttribution:
    """Records per-trade attribution from entry signal → realized outcome."""

    def __init__(
        self,
        persist_path: str = "alpha_attribution_log.jsonl",
        max_pending: int = 500,
        pending_path: str = "alpha_attribution_pending.json",
    ):
        self.persist_path = persist_path
        self.pending_path = pending_path
        # pending attributions: list of dicts, appended at entry, popped at close.
        # Persisted to disk so a restart during an open position doesn't orphan
        # the entry-side attribution (exit would otherwise log as exit_only).
        self._pending: List[Dict[str, Any]] = []
        self._max_pending = max_pending
        self._lock = threading.RLock()
        self._load_pending()

    # ------------------------------------------------------------------
    # Entry-side recording
    # ------------------------------------------------------------------
    def record_attribution(
        self,
        symbol: str,
        baseline_weight: float,
        final_weight: float,
        direction: int,
        layers: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Call once per signal-gen cycle per symbol. `layers` maps layer
        name → multiplier applied (1.0 = no change).

        Example layers dict:
            {
                "causal_penalty": 0.95,
                "stacking_meta_blend": 1.03,
                "bayesian_sizing": 1.26,
                "cross_sectional": 1.25,
                "crowding_discount": 1.0,
                "sentiment_level": 1.08,
                "sentiment_velocity": 1.02,
                "regime_gate": 1.0,
                "vix_gate": 1.0,
                "meta_filter_rejected": 0.0,  # if it zeroed the weight
                "adverse_selection": 1.0,
                "slippage_veto": 1.0,
                "divergence_gate": 1.0,
                "earnings_blackout": 1.0,
                "eq_curve_scale": 1.001,
            }
        """
        with self._lock:
            self._pending.append({
                "symbol": symbol,
                "entry_ts": datetime.now(tz=_UTC).isoformat(),
                "baseline_weight": float(baseline_weight),
                "final_weight": float(final_weight),
                "direction": int(direction),
                "layers": {k: float(v) for k, v in (layers or {}).items()},
                "context": dict(context or {}),
            })
            # Cap in-memory pending list to avoid unbounded growth
            if len(self._pending) > self._max_pending:
                self._pending = self._pending[-self._max_pending :]
            self._save_pending()

    # ------------------------------------------------------------------
    # Exit-side emission
    # ------------------------------------------------------------------
    def emit_on_close(
        self,
        symbol: str,
        realized_return: float,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        mfe: Optional[float] = None,
        mae: Optional[float] = None,
        exit_reason: Optional[str] = None,
        regime_at_open: Optional[str] = None,
        regime_at_close: Optional[str] = None,
        held_bars: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Called by stream.py on trade close. Finds the most recent matching
        entry attribution (by symbol), merges with realized outcome, and emits
        one JSON line to the log. Returns the merged dict (None if no match)."""
        # Pop the most recent matching entry
        entry_attr = None
        with self._lock:
            for i in range(len(self._pending) - 1, -1, -1):
                if self._pending[i]["symbol"] == symbol:
                    entry_attr = self._pending.pop(i)
                    break
            if entry_attr is not None:
                self._save_pending()

        if entry_attr is None:
            # No matching entry (trade may have been opened before logger started
            # or across a restart). Emit a partial record with just exit data.
            record = {
                "event": "alpha_attribution_exit_only",
                "symbol": symbol,
                "exit_ts": datetime.now(tz=_UTC).isoformat(),
                "realized_return": float(realized_return),
                "entry_price": float(entry_price) if entry_price is not None else None,
                "exit_price": float(exit_price) if exit_price is not None else None,
                "mfe_pct": float(mfe) if mfe is not None else None,
                "mae_pct": float(mae) if mae is not None else None,
                "exit_reason": exit_reason,
                "regime_at_close": regime_at_close,
                "held_bars": held_bars,
                "note": "no_matching_entry_attribution",
            }
        else:
            record = {
                "event": "alpha_attribution_closed_trade",
                "symbol": symbol,
                "direction": entry_attr["direction"],
                "entry_ts": entry_attr["entry_ts"],
                "exit_ts": datetime.now(tz=_UTC).isoformat(),
                "baseline_weight": entry_attr["baseline_weight"],
                "final_weight": entry_attr["final_weight"],
                "layers": entry_attr["layers"],
                "context": entry_attr["context"],
                "realized_return": float(realized_return),
                "entry_price": float(entry_price) if entry_price is not None else None,
                "exit_price": float(exit_price) if exit_price is not None else None,
                "mfe_pct": float(mfe) if mfe is not None else None,
                "mae_pct": float(mae) if mae is not None else None,
                "exit_reason": exit_reason,
                "regime_at_open": regime_at_open,
                "regime_at_close": regime_at_close,
                "held_bars": held_bars,
            }

        self._persist(record)
        return record

    def _persist(self, record: Dict[str, Any]) -> None:
        try:
            with open(self.persist_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            logger.info(f"[ALPHA-ATTR] {record['symbol']} closed — "
                        f"realized={record['realized_return']:+.4f} "
                        f"final_w={record.get('final_weight', 'n/a')} "
                        f"exit={record.get('exit_reason', 'unknown')}")
        except Exception as e:
            logger.debug(f"[ALPHA-ATTR] persist failed: {e}")

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def clear(self) -> None:
        """For testing — drop all pending attributions."""
        with self._lock:
            self._pending.clear()
            self._save_pending()

    # ------------------------------------------------------------------
    # Pending-list persistence (restart-safe)
    # ------------------------------------------------------------------
    def _save_pending(self) -> None:
        try:
            import tempfile, shutil
            dir_name = os.path.dirname(self.pending_path) or "."
            with tempfile.NamedTemporaryFile(mode="w", delete=False,
                                             dir=dir_name, suffix=".tmp") as tmp:
                json.dump(self._pending, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
            shutil.move(tmp.name, self.pending_path)
        except Exception as e:
            logger.debug(f"[ALPHA-ATTR] pending save failed: {e}")

    def _load_pending(self) -> None:
        if not os.path.exists(self.pending_path):
            return
        try:
            with open(self.pending_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                with self._lock:
                    self._pending = data[-self._max_pending:]
                logger.info(f"[ALPHA-ATTR] Restored {len(self._pending)} pending attributions from disk")
        except Exception as e:
            logger.debug(f"[ALPHA-ATTR] pending load failed: {e}")

# strategy/execution_scorecard.py
"""
EQ-SCORE — Execution-Quality Scorecard.

Aggregates daily telemetry from every execution-quality subsystem into a single
structured log line. Answers "what did our 18-layer gate cascade actually do
today?" — essential for operator visibility and for Gemini to tune intelligently.

Tracked metrics:
    - Slippage: per-symbol mean bps from today's fills
    - Adverse selection: toxicity score per symbol (5-min post-fill drift)
    - Meta-filter: reject count per symbol + rejection rate
    - Divergence gate: trigger count per symbol
    - Liquidity scaler: active scaling events count
    - Cross-sectional: how often each symbol was in top/bottom tercile today
    - Bayesian sizing: current multiplier per symbol (live posterior snapshot)
    - Regime distribution: how many symbols in each regime today

Emits a single JSON line at market close (16:30 ET) so grep + jq can parse it.
Also persists to `execution_scorecard_log.jsonl` for historical analysis.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import Counter, defaultdict
from datetime import date, datetime
from dateutil import tz
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ET = tz.gettz("America/New_York")


class ExecutionScorecard:
    """Lightweight in-memory tally per trading day. Persisted at market close."""

    def __init__(
        self,
        persist_path: str = "execution_scorecard_log.jsonl",
    ):
        self.persist_path = persist_path
        self._lock = threading.RLock()
        # Per-day tallies — resets each ET day
        self._current_date: Optional[date] = None
        self._counters: Dict[str, Counter] = defaultdict(Counter)
        self._symbol_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "meta_rejects": 0,
            "divergence_hits": 0,
            "slippage_veto_hits": 0,
            "liquidity_scale_hits": 0,
            "cs_top_tercile": 0,
            "cs_bottom_tercile": 0,
            "avs_toxic_flags": 0,
            "earnings_blackouts": 0,
            "entries_submitted": 0,
            "entries_filled": 0,
        })
        self._regime_distribution: Counter = Counter()
        self._cycle_count = 0

    def _ensure_current_day(self):
        """Reset tallies if day has rolled over."""
        today = datetime.now(tz=_ET).date()
        if self._current_date is None:
            self._current_date = today
        elif today != self._current_date:
            # Day rolled → flush yesterday's scorecard before reset
            self._flush_and_reset(today)

    def _flush_and_reset(self, new_date: date):
        """Write scorecard for the completed day; reset tallies for new day."""
        try:
            self.emit(snapshot=None, final_for_date=self._current_date)
        except Exception as e:
            logger.debug(f"[EQ-SCORE] flush-on-rollover failed: {e}")
        with self._lock:
            self._counters.clear()
            self._symbol_stats = defaultdict(lambda: {
                "meta_rejects": 0, "divergence_hits": 0, "slippage_veto_hits": 0,
                "liquidity_scale_hits": 0, "cs_top_tercile": 0, "cs_bottom_tercile": 0,
                "avs_toxic_flags": 0, "earnings_blackouts": 0,
                "entries_submitted": 0, "entries_filled": 0,
            })
            self._regime_distribution.clear()
            self._cycle_count = 0
            self._current_date = new_date

    # ------------------------------------------------------------------
    # Event recording — called by subsystems
    # ------------------------------------------------------------------
    def record_cycle(self, regimes: Dict[str, Any]) -> None:
        """Called once per signal-gen cycle. Updates regime distribution."""
        with self._lock:
            self._ensure_current_day()
            self._cycle_count += 1
            for sym, reg in regimes.items():
                reg_name = reg[0] if isinstance(reg, (list, tuple)) else str(reg)
                self._regime_distribution[reg_name] += 1

    def record_meta_reject(self, symbol: str, prob: float) -> None:
        with self._lock:
            self._ensure_current_day()
            self._symbol_stats[symbol]["meta_rejects"] += 1

    def record_divergence(self, symbol: str) -> None:
        with self._lock:
            self._ensure_current_day()
            self._symbol_stats[symbol]["divergence_hits"] += 1

    def record_slippage_veto(self, symbol: str, pred_bps: float) -> None:
        with self._lock:
            self._ensure_current_day()
            self._symbol_stats[symbol]["slippage_veto_hits"] += 1

    def record_liquidity_scale(self, symbol: str, participation: float) -> None:
        with self._lock:
            self._ensure_current_day()
            self._symbol_stats[symbol]["liquidity_scale_hits"] += 1

    def record_cs_tercile(self, symbol: str, tercile: str) -> None:
        """tercile ∈ {'top', 'mid', 'bottom'}."""
        with self._lock:
            self._ensure_current_day()
            if tercile == "top":
                self._symbol_stats[symbol]["cs_top_tercile"] += 1
            elif tercile == "bottom":
                self._symbol_stats[symbol]["cs_bottom_tercile"] += 1

    def record_ava_toxic(self, symbol: str, score: float) -> None:
        with self._lock:
            self._ensure_current_day()
            self._symbol_stats[symbol]["avs_toxic_flags"] += 1

    def record_earnings_blackout(self, symbol: str) -> None:
        with self._lock:
            self._ensure_current_day()
            self._symbol_stats[symbol]["earnings_blackouts"] += 1

    def record_entry_submitted(self, symbol: str) -> None:
        with self._lock:
            self._ensure_current_day()
            self._symbol_stats[symbol]["entries_submitted"] += 1

    def record_entry_filled(self, symbol: str) -> None:
        with self._lock:
            self._ensure_current_day()
            self._symbol_stats[symbol]["entries_filled"] += 1

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------
    def emit(
        self,
        snapshot: Optional[Dict[str, Any]] = None,
        final_for_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Build the scorecard dict, log it, persist it. Returns the dict."""
        with self._lock:
            report_date = final_for_date or self._current_date or datetime.now(tz=_ET).date()
            total_entries = sum(v["entries_submitted"] for v in self._symbol_stats.values())
            total_filled = sum(v["entries_filled"] for v in self._symbol_stats.values())
            total_rejects = sum(v["meta_rejects"] for v in self._symbol_stats.values())
            total_divergences = sum(v["divergence_hits"] for v in self._symbol_stats.values())
            total_slippage_vetos = sum(v["slippage_veto_hits"] for v in self._symbol_stats.values())
            total_liq_scales = sum(v["liquidity_scale_hits"] for v in self._symbol_stats.values())

            # Rejection rate: rejects / (rejects + entries). Approximate — a cycle can
            # reject multiple symbols, but this gives a sense of filter activity.
            gate_activity_denom = max(1, total_rejects + total_divergences + total_slippage_vetos
                                          + total_liq_scales + total_entries)
            rejection_share = total_rejects / gate_activity_denom

            report = {
                "event": "execution_scorecard_daily",
                "date": report_date.isoformat() if report_date else None,
                "cycle_count": self._cycle_count,
                "entries_submitted": total_entries,
                "entries_filled": total_filled,
                "meta_rejects_total": total_rejects,
                "divergence_hits_total": total_divergences,
                "slippage_vetos_total": total_slippage_vetos,
                "liquidity_scales_total": total_liq_scales,
                "earnings_blackouts_total": sum(v["earnings_blackouts"] for v in self._symbol_stats.values()),
                "rejection_share_of_gate_activity": round(rejection_share, 4),
                "regime_distribution_bar_counts": dict(self._regime_distribution),
                "per_symbol": {sym: dict(v) for sym, v in self._symbol_stats.items()},
                "snapshot": snapshot or {},
                "generated_at": datetime.now(tz=_ET).isoformat(),
            }

        # Log one human-readable summary line + full JSON
        def _one_line():
            symbols_with_activity = [
                s for s, v in report["per_symbol"].items()
                if (v["meta_rejects"] + v["divergence_hits"] + v["slippage_veto_hits"]
                    + v["liquidity_scale_hits"] + v["earnings_blackouts"] > 0)
            ]
            return (f"[EQ-SCORE] {report['date']} | cycles={report['cycle_count']} "
                    f"| entries_sub={total_entries} filled={total_filled} "
                    f"| meta_rej={total_rejects} divergence={total_divergences} "
                    f"| slip_veto={total_slippage_vetos} liq_scale={total_liq_scales} "
                    f"| active_syms={len(symbols_with_activity)}")

        logger.info(_one_line())
        logger.info(json.dumps(report))

        # Persist as JSONL for historical analysis
        try:
            with open(self.persist_path, "a") as f:
                f.write(json.dumps(report) + "\n")
        except Exception as e:
            logger.debug(f"[EQ-SCORE] persist failed: {e}")

        return report

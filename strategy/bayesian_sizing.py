# strategy/bayesian_sizing.py
"""
BPS — Bayesian Per-Symbol Sizing.

Today's P&L story: SMCI 100% WR carried the book while JPM (28%) / TSLA (25%) /
AAPL (0%) bled. Flat CVaR sizing treats all symbols identically. Bayesian sizing
maintains a Beta(α, β) posterior on P(win) per symbol, updated every closed
trade, and scales the target weight by an expected-return multiplier.

Conjugate update:
    win → α += 1    loss → β += 1
    E[P(win)] = α / (α + β)
    Var[P(win)] = αβ / ((α+β)^2 (α+β+1))   — uncertainty shrinks with trade count

Multiplier:
    ev_per_trade = p_win * avg_win + (1 - p_win) * avg_loss          (avg_loss is negative)
    multiplier   = 1.0 + gain * ev_per_trade / reference_ev
    clamped to [MIN_MULT, MAX_MULT]

Uncertainty shrinkage: when n < min_trades, blend toward a neutral prior so we
don't size up massively on 2 trades. This is the "prior informs small-sample
estimates" Bayesian discipline.

Persistence: posterior state saved to bayesian_sizing.pkl across restarts.
"""
from __future__ import annotations

import logging
import math
import os
import pickle
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Neutral Bayesian prior: α=2, β=3 → prior mean = 0.4, equivalent to "5 trades worth of
# mild pessimism" — close to the observed base rate (~30% WR) without being cripplingly
# pessimistic. Updated by real trades within a few days of live activity.
DEFAULT_PRIOR_ALPHA = 2.0
DEFAULT_PRIOR_BETA = 3.0


class BayesianSymbolSizer:
    """Per-symbol Beta posterior + expected-return sizing multiplier."""

    def __init__(
        self,
        symbols: List[str],
        persist_path: str = "bayesian_sizing.pkl",
        prior_alpha: float = DEFAULT_PRIOR_ALPHA,
        prior_beta: float = DEFAULT_PRIOR_BETA,
    ):
        self.symbols = list(symbols)
        self.persist_path = persist_path
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        # symbol → {"alpha", "beta", "avg_win", "avg_loss", "n"}
        self._state: Dict[str, Dict[str, float]] = {}
        self._lock = threading.RLock()
        self._load()
        # Ensure every symbol has an entry
        for s in self.symbols:
            if s not in self._state:
                self._state[s] = self._fresh_entry()

    @staticmethod
    def _fresh_entry() -> Dict[str, float]:
        return {
            "alpha": DEFAULT_PRIOR_ALPHA,
            "beta": DEFAULT_PRIOR_BETA,
            "avg_win": 0.01,     # prior: 1% typical win
            "avg_loss": -0.008,  # prior: -0.8% typical loss
            "n": 0,
        }

    # ------------------------------------------------------------------
    # Update from closed trades
    # ------------------------------------------------------------------
    def update(self, symbol: str, realized_return: float) -> None:
        """Apply a single closed-trade observation."""
        if realized_return is None or (isinstance(realized_return, float) and math.isnan(realized_return)):
            return
        with self._lock:
            st = self._state.setdefault(symbol, self._fresh_entry())
            st["n"] = int(st.get("n", 0)) + 1
            if realized_return > 0:
                st["alpha"] = float(st.get("alpha", DEFAULT_PRIOR_ALPHA)) + 1.0
                # Running mean of wins (simple incremental)
                n_wins = max(1, int(st["alpha"] - DEFAULT_PRIOR_ALPHA))
                prev_avg = float(st.get("avg_win", 0.01))
                st["avg_win"] = prev_avg + (realized_return - prev_avg) / n_wins
            else:
                st["beta"] = float(st.get("beta", DEFAULT_PRIOR_BETA)) + 1.0
                n_losses = max(1, int(st["beta"] - DEFAULT_PRIOR_BETA))
                prev_avg = float(st.get("avg_loss", -0.008))
                st["avg_loss"] = prev_avg + (realized_return - prev_avg) / n_losses
        self._save()

    def fit_from_history(self, history: Dict[str, List[dict]]) -> Dict[str, Dict[str, float]]:
        """Bulk-initialize from live_signal_history. Replays every closed trade."""
        # Reset to prior, then replay
        with self._lock:
            self._state = {s: self._fresh_entry() for s in self.symbols}
        for sym, entries in (history or {}).items():
            for e in entries:
                rr = e.get("realized_return")
                if rr is None:
                    continue
                self.update(sym, float(rr))
        summary = {}
        with self._lock:
            for s, st in self._state.items():
                summary[s] = {
                    "p_win": self.p_win(s),
                    "n": int(st.get("n", 0)),
                    "avg_win": float(st.get("avg_win", 0.0)),
                    "avg_loss": float(st.get("avg_loss", 0.0)),
                }
        logger.info(f"[BAYESIAN-SIZE] Fit from history — {len(summary)} symbols initialized")
        return summary

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def p_win(self, symbol: str) -> float:
        """Posterior mean P(win) = α / (α + β)."""
        with self._lock:
            st = self._state.get(symbol, self._fresh_entry())
            a = float(st.get("alpha", DEFAULT_PRIOR_ALPHA))
            b = float(st.get("beta", DEFAULT_PRIOR_BETA))
        if a + b < 1e-9:
            return 0.5
        return a / (a + b)

    def expected_return(self, symbol: str) -> float:
        """E[R per trade] = p_win·avg_win + (1-p_win)·avg_loss."""
        p = self.p_win(symbol)
        with self._lock:
            st = self._state.get(symbol, self._fresh_entry())
            aw = float(st.get("avg_win", 0.01))
            al = float(st.get("avg_loss", -0.008))
        return p * aw + (1.0 - p) * al

    def size_multiplier(
        self,
        symbol: str,
        min_mult: float = 0.4,
        max_mult: float = 1.6,
        reference_ev: float = 0.003,  # +0.3% ev_per_trade → +1 on the scale
        shrinkage_n: int = 8,
    ) -> Tuple[float, str]:
        """Convert per-symbol expected return into a weight multiplier in
        [min_mult, max_mult]. Below `shrinkage_n` total trades we blend toward
        1.0 (neutral) so small samples don't dominate."""
        ev = self.expected_return(symbol)
        with self._lock:
            st = self._state.get(symbol, self._fresh_entry())
            n = int(st.get("n", 0))
        # Core scale factor from EV
        raw = 1.0 + (ev / reference_ev)
        raw = max(min_mult, min(max_mult, raw))
        # Shrinkage toward 1.0 for small n
        shrink = min(1.0, n / max(1, shrinkage_n))
        mult = 1.0 + shrink * (raw - 1.0)
        mult = max(min_mult, min(max_mult, mult))
        reason = f"p_win={self.p_win(symbol):.2f} ev={ev:+.4f} n={n} shrink={shrink:.2f}"
        return float(mult), reason

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Diagnostic snapshot of current posteriors (for logging/debug)."""
        out = {}
        with self._lock:
            for s, st in self._state.items():
                out[s] = {
                    "p_win": self.p_win(s),
                    "n": int(st.get("n", 0)),
                    "ev": self.expected_return(s),
                }
        return out

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
                            "state": self._state,
                            "prior_alpha": self.prior_alpha,
                            "prior_beta": self.prior_beta,
                        },
                        tmp,
                    )
                tmp.flush()
                os.fsync(tmp.fileno())
            shutil.move(tmp.name, self.persist_path)
        except Exception as e:
            logger.debug(f"[BAYESIAN-SIZE] save failed: {e}")

    def _load(self):
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
            with self._lock:
                self._state = data.get("state", {}) or {}
            logger.info(f"[BAYESIAN-SIZE] Loaded {len(self._state)} symbol posteriors from disk")
        except Exception as e:
            logger.debug(f"[BAYESIAN-SIZE] load failed: {e}")

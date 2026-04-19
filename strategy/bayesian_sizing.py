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

    def kelly_fraction(self, symbol: str) -> float:
        """Full Kelly fraction: f* = (p·b − q) / b
        where p=P(win), q=1−p, b=|avg_win|/|avg_loss|.
        Returns the raw Kelly fraction (can be negative if symbol has negative edge).
        Caller should scale by fractional-Kelly coefficient (default 0.25) for safety."""
        p = self.p_win(symbol)
        q = 1.0 - p
        with self._lock:
            st = self._state.get(symbol, self._fresh_entry())
            aw = abs(float(st.get("avg_win", 0.01)))
            al = abs(float(st.get("avg_loss", 0.008)))
        if al < 1e-9 or aw < 1e-9:
            return 0.0
        b = aw / al  # win:loss ratio
        # Standard Kelly: f* = (bp - q) / b = p - q/b
        f_star = (b * p - q) / b
        return float(f_star)

    def size_multiplier(
        self,
        symbol: str,
        min_mult: float = 0.4,
        max_mult: float = 1.6,
        reference_ev: float = 0.003,  # legacy (EV-based) path — kept for rollback
        shrinkage_n: int = 8,
        method: str = "kelly",          # NEW: "kelly" (default) or "ev" (legacy)
        kelly_fraction: float = 0.25,    # ¼ Kelly is institutional-standard safety
        reference_kelly: float = 0.08,   # raw f* of 0.08 → full boost at max_mult
        persistence: float = 0.5,        # Apr-19: high-persistence regimes unlock wider cap
        proven_n: int = 20,              # Apr-19: n threshold for "proven" upsize
        proven_p_win: float = 0.60,      # Apr-19: p_win threshold for proven upsize
        proven_persistence: float = 0.85,  # Apr-19: regime persistence threshold
        proven_max_mult: float = 2.0,    # Apr-19: cap when proven criteria all met
    ) -> Tuple[float, str]:
        """Convert per-symbol edge into a weight multiplier ∈ [min_mult, max_mult].

        KELLY METHOD (default):
            Raw Kelly f* = (p·b − q) / b, scaled by fractional coefficient (0.25)
            then mapped to multiplier. Positive edge → boost up to max_mult;
            negative edge → dampen down to min_mult. Mathematically optimal for
            long-run geometric growth.

        EV METHOD (legacy, backward-compat):
            mult = 1 + EV / reference_ev, clipped.

        Shrinkage: when total closed trades n < shrinkage_n, blend toward 1.0
        (neutral) so 2-3 lucky trades don't dominate sizing."""
        with self._lock:
            st = self._state.get(symbol, self._fresh_entry())
            n = int(st.get("n", 0))

        if method == "kelly":
            # Fractional Kelly as signed edge, then remap around 1.0
            f_raw = self.kelly_fraction(symbol)             # ∈ roughly [-0.5, +0.5]
            f_scaled = f_raw * kelly_fraction                # ¼-Kelly typically
            p = self.p_win(symbol)
            # Apr-19: "Proven winner" upsize — when we have enough closed trades
            # to trust the posterior (n ≥ proven_n), the win rate is genuinely
            # positive (p_win ≥ proven_p_win), and the regime is persistent
            # enough to not whipsaw (persistence ≥ proven_persistence), expand
            # the upside cap from max_mult (1.6) to proven_max_mult (2.0).
            dynamic_max = float(max_mult)
            proven = (n >= proven_n and p >= proven_p_win and persistence >= proven_persistence)
            if proven:
                dynamic_max = float(max(dynamic_max, proven_max_mult))
            # Map scaled fraction → multiplier.
            if f_scaled >= 0:
                raw = 1.0 + (f_scaled / max(reference_kelly, 1e-9)) * (dynamic_max - 1.0)
            else:
                raw = 1.0 + (f_scaled / max(reference_kelly, 1e-9)) * (1.0 - min_mult)
            raw = max(min_mult, min(dynamic_max, raw))
            # Shrinkage toward 1.0 for small n (avoid overreacting to lucky streaks)
            shrink = min(1.0, n / max(1, shrinkage_n))
            mult = 1.0 + shrink * (raw - 1.0)
            mult = max(min_mult, min(dynamic_max, mult))
            reason = (f"kelly_f*={f_raw:+.3f} scaled(¼)={f_scaled:+.3f} "
                      f"p={p:.2f} n={n} shrink={shrink:.2f}"
                      f"{' proven-upsize' if proven else ''}")
            return float(mult), reason
        # --- legacy EV method ---
        ev = self.expected_return(symbol)
        raw = 1.0 + (ev / reference_ev)
        raw = max(min_mult, min(max_mult, raw))
        shrink = min(1.0, n / max(1, shrinkage_n))
        mult = 1.0 + shrink * (raw - 1.0)
        mult = max(min_mult, min(max_mult, mult))
        reason = f"ev-legacy p_win={self.p_win(symbol):.2f} ev={ev:+.4f} n={n} shrink={shrink:.2f}"
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

# strategy/meta_filter.py
"""
S1 — Meta-labeling filter (Lopez de Prado triple-barrier style).

After the portfolio PPO emits a candidate action, the meta-labeler asks:
"given that PPO says go <direction> on <symbol>, is this a *good* trade right
now?" A lightweight binary classifier is trained nightly on closed trades from
live_signal_history — features are the metadata already stored at entry time
(direction, confidence, ppo_strength, conviction, hour/day, regime, sentiment),
labels are sign(realized_return). At inference we zero out candidate weights
whose P(win) falls below a conservative threshold.

Defensive by design:
- Returns neutral (P=0.5, pass-through) until at least MIN_TRAIN_SAMPLES trades
  exist. With 30-50 trades a small LightGBM can give a modest edge; more data
  monotonically improves it.
- Strictly additive: on model absence, import failure, or prediction error, the
  filter falls through to allowing the PPO signal unchanged.
- Threshold is intentionally conservative (0.40 default). We only reject trades
  the model thinks are *clearly* likely to lose — tighten later as data grows.
"""
from __future__ import annotations

import logging
import os
import pickle
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

logger = logging.getLogger(__name__)

# Fixed feature order — MUST match between fit() and predict(). Adding new
# features requires retraining. Symbol is encoded via a learned mapping rather
# than one-hot to keep the vector small.
_FEATURE_NAMES = [
    "direction",
    "confidence_abs",
    "ppo_strength",
    "conviction",
    "hour_of_day",
    "day_of_week",
    "symbol_id",  # integer encoding (stable across restarts via symbol_map)
    "regime_trending",  # 1 if trending_up, -1 if trending_down, 0 else
    "regime_persistence",  # 0-1
    "sentiment",  # -1 to 1
    "vix_level",  # normalized (vix_close / 20)
    "size_rel",   # size * price / equity — captures sizing at entry
]


class MetaLabeler:
    """Global (pooled across symbols) meta-label classifier.

    Usage:
        mlabeler = MetaLabeler(symbols=config['SYMBOLS'])
        mlabeler.fit_from_history(live_signal_history)  # nightly
        accept, prob = mlabeler.should_enter(symbol, direction, ppo_strength,
                                              confidence, conviction, ...)
    """

    def __init__(self, symbols: List[str], persist_path: str = "meta_filter.pkl"):
        self.symbols = list(symbols)
        self.symbol_map = {s: i for i, s in enumerate(self.symbols)}
        self.persist_path = persist_path
        self.model = None
        self.train_stats = {}
        self._lock = threading.RLock()
        self._load()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _featurize_entry(
        self,
        symbol: str,
        direction: int,
        confidence: float,
        ppo_strength: float,
        conviction: float,
        timestamp: datetime,
        regime: str = "mean_reverting",
        persistence: float = 0.5,
        sentiment: float = 0.0,
        vix: float = 20.0,
        size_rel: float = 0.0,
    ) -> np.ndarray:
        """Build one feature row. All args are scalars already present at
        signal-generation time. vix and sentiment default to neutral if
        unknown."""
        hour = getattr(timestamp, "hour", 12)
        dow = getattr(timestamp, "weekday", lambda: 2)()
        if regime == "trending_up":
            reg_flag = 1.0
        elif regime == "trending_down":
            reg_flag = -1.0
        else:
            reg_flag = 0.0
        sym_id = float(self.symbol_map.get(symbol, -1))
        return np.array(
            [
                float(direction),
                float(abs(confidence)),
                float(ppo_strength),
                float(conviction),
                float(hour),
                float(dow),
                sym_id,
                reg_flag,
                float(persistence),
                float(np.clip(sentiment, -1.0, 1.0)),
                float(vix / 20.0),
                float(size_rel),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit_from_history(
        self,
        history: Dict[str, List[dict]],
        min_train_samples: int = 30,
        equity_estimate: float = 30000.0,
    ) -> Dict:
        """Fit the classifier from live_signal_history. Returns stats dict.
        If <min_train_samples closed trades exist, skips fit (keeps prior model
        if any, else leaves self.model=None)."""
        if not _LGB_AVAILABLE:
            logger.warning("[META-FILTER] lightgbm unavailable — meta filter disabled")
            return {"status": "lgb_unavailable"}

        X_rows: List[np.ndarray] = []
        y_rows: List[int] = []
        for sym, entries in (history or {}).items():
            for e in entries:
                rr = e.get("realized_return")
                if rr is None:
                    continue
                ts = e.get("timestamp")
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except Exception:
                        ts = datetime.now()
                # size_rel: use stored size × price ≈ notional, divided by a rough equity
                notional = float(e.get("size", 0) or 0) * float(e.get("price", 0) or 0)
                size_rel = notional / equity_estimate if equity_estimate > 0 else 0.0
                row = self._featurize_entry(
                    symbol=sym,
                    direction=int(e.get("direction", 0)),
                    confidence=float(e.get("confidence", 0.0)),
                    ppo_strength=float(e.get("ppo_strength", 0.0)),
                    conviction=float(e.get("conviction", 0.0)),
                    timestamp=ts,
                    regime=str(e.get("regime", "mean_reverting")),
                    persistence=float(e.get("persistence", 0.5)),
                    sentiment=float(e.get("sentiment", 0.0)),
                    vix=float(e.get("vix", 20.0)),
                    size_rel=size_rel,
                )
                X_rows.append(row)
                y_rows.append(1 if rr > 0 else 0)

        n = len(y_rows)
        if n < min_train_samples:
            logger.info(
                f"[META-FILTER] Only {n} closed trades (need ≥{min_train_samples}) — "
                f"skipping fit, filter stays in pass-through mode"
            )
            return {"status": "insufficient_data", "n_samples": n}

        X = np.asarray(X_rows, dtype=np.float32)
        y = np.asarray(y_rows, dtype=np.int32)
        # Small model to avoid memorization on tiny dataset
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 8,
            "max_depth": 3,
            "min_data_in_leaf": max(3, n // 15),
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "lambda_l2": 0.1,
        }
        try:
            dset = lgb.Dataset(X, label=y, feature_name=_FEATURE_NAMES)
            booster = lgb.train(params, dset, num_boost_round=60)
            # Compute calibration diagnostics
            preds = booster.predict(X)
            wr = float(y.mean())
            mean_pred = float(preds.mean())
            # Brier score (lower = better calibration, baseline = var(y))
            brier = float(((preds - y) ** 2).mean())
            with self._lock:
                self.model = booster
                self.train_stats = {
                    "n_samples": n,
                    "base_wr": wr,
                    "mean_pred": mean_pred,
                    "brier": brier,
                    "fitted_at": datetime.utcnow().isoformat(),
                }
            self._save()
            logger.info(
                f"[META-FILTER] Fit complete — n={n} base_wr={wr:.3f} "
                f"mean_pred={mean_pred:.3f} brier={brier:.4f}"
            )
            return {"status": "ok", **self.train_stats}
        except Exception as e:
            logger.error(f"[META-FILTER] Fit failed: {e}", exc_info=True)
            return {"status": "fit_failed", "error": str(e)}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_prob(
        self,
        symbol: str,
        direction: int,
        confidence: float,
        ppo_strength: float,
        conviction: float,
        timestamp: Optional[datetime] = None,
        regime: str = "mean_reverting",
        persistence: float = 0.5,
        sentiment: float = 0.0,
        vix: float = 20.0,
        size_rel: float = 0.0,
    ) -> float:
        """Return P(trade wins). Returns 0.5 (neutral) if model not fitted or
        any error occurs — caller should treat 0.5 as 'no information'."""
        with self._lock:
            model = self.model
        if model is None:
            return 0.5
        if timestamp is None:
            timestamp = datetime.now()
        try:
            row = self._featurize_entry(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                ppo_strength=ppo_strength,
                conviction=conviction,
                timestamp=timestamp,
                regime=regime,
                persistence=persistence,
                sentiment=sentiment,
                vix=vix,
                size_rel=size_rel,
            ).reshape(1, -1)
            prob = float(model.predict(row, num_iteration=model.best_iteration)[0])
            return prob
        except Exception as e:
            logger.debug(f"[META-FILTER] predict failed for {symbol}: {e}")
            return 0.5

    def should_enter(
        self,
        symbol: str,
        direction: int,
        min_prob: float,
        **kwargs,
    ) -> Tuple[bool, float]:
        """Return (accept, prob). accept=True if prob >= (Brier-adjusted min_prob)
        OR model not yet fitted (pass-through)."""
        prob = self.predict_prob(symbol=symbol, direction=direction, **kwargs)
        with self._lock:
            fitted = self.model is not None
            brier = float(self.train_stats.get("brier", 0.30))
        if not fitted:
            return True, 0.5
        # Brier-calibrated threshold: tighten when the model is well-calibrated
        # (brier < 0.30), loosen when poorly calibrated. Clipped to [0.35, 0.55]
        # so the threshold can never collapse below a sane floor.
        adjusted_min = min_prob * (1.0 + (0.30 - brier) * 2.0)
        adjusted_min = float(np.clip(adjusted_min, 0.35, 0.55))
        return prob >= adjusted_min, prob

    def is_fitted(self) -> bool:
        """Used by callers that want to apply a pre-fit dampener rather than
        treating `should_enter` pass-through as genuine acceptance."""
        with self._lock:
            return self.model is not None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(self):
        try:
            # Atomic write via tempfile + rename
            import tempfile, shutil
            dir_name = os.path.dirname(self.persist_path) or "."
            with tempfile.NamedTemporaryFile(
                mode="wb", delete=False, dir=dir_name, suffix=".tmp"
            ) as tmp:
                pickle.dump(
                    {
                        "model": self.model,
                        "symbol_map": self.symbol_map,
                        "train_stats": self.train_stats,
                        "feature_names": _FEATURE_NAMES,
                    },
                    tmp,
                )
                tmp.flush()
                os.fsync(tmp.fileno())
            shutil.move(tmp.name, self.persist_path)
        except Exception as e:
            logger.warning(f"[META-FILTER] Save failed: {e}")

    def _load(self):
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
            # Schema check: feature names must match (guard against silent drift)
            if data.get("feature_names") != _FEATURE_NAMES:
                logger.warning(
                    "[META-FILTER] Persisted feature schema differs from current — "
                    "discarding cached model"
                )
                return
            self.model = data.get("model")
            self.symbol_map = data.get("symbol_map", self.symbol_map)
            self.train_stats = data.get("train_stats", {})
            if self.model is not None:
                logger.info(
                    f"[META-FILTER] Loaded cached model — "
                    f"n_samples={self.train_stats.get('n_samples', '?')} "
                    f"base_wr={self.train_stats.get('base_wr', '?')}"
                )
        except Exception as e:
            logger.warning(f"[META-FILTER] Load failed: {e}")

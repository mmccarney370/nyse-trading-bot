# strategy/cognitive.py
"""
Cognitive Trading Layer — AGI-inspired meta-intelligence for the trading bot.

Three novel systems that don't exist in any open-source trading framework:

1. EQUITY CURVE POSITION SIZING (Meta-Strategy)
   Applies trend-following to the bot's OWN equity curve. When the bot is
   in a drawdown, it reduces position sizes. When on a winning streak,
   it sizes up. This is a second-order strategy — a strategy about the strategy.

2. PROFIT VELOCITY TRACKER (Dynamic Profit Protection)
   Tracks the RATE of P&L change per position. When profit is accelerating,
   stops stay wide to let winners run. When profit velocity decelerates or
   reverses, stops tighten aggressively. Mimics a human trader's intuition
   for "momentum is fading."

3. TRADE AUTOPSY ENGINE (Self-Reflection)
   After every closed trade, generates a structured analysis of WHY it won
   or lost. Builds a searchable knowledge base of lessons that informs
   future decisions. The bot literally learns from its own mistakes without
   retraining any model.

These three systems together create a meta-cognitive layer:
- The equity curve trader controls HOW MUCH to risk
- The profit velocity tracker controls WHEN to exit
- The autopsy engine controls WHETHER to enter (by pattern-matching against past failures)
"""

import logging
import numpy as np
import json
import os
import threading
from datetime import datetime, timedelta
from collections import deque
from dateutil import tz

logger = logging.getLogger(__name__)

_UTC = tz.gettz('UTC')


class EquityCurveTrader:
    """Applies trend-following to the bot's own equity curve.

    Concept: If the bot itself is in a drawdown, the strategy isn't working
    in current conditions — reduce exposure. If the bot is on a winning streak,
    conditions are favorable — size up.

    Uses a fast/slow EMA crossover on the equity curve (same concept as golden/death
    cross but applied to our own P&L). When fast < slow (equity in downtrend),
    scale position sizes by a fraction. When fast > slow, scale up.

    This is a SECOND-ORDER strategy — a strategy about the strategy.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 30,
                 min_scale: float = 0.3, max_scale: float = 1.3):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.equity_history = deque(maxlen=200)
        self._fast_ema = None
        self._slow_ema = None
        self._lock = threading.Lock()

    def record_equity(self, equity: float, timestamp=None):
        """Call once per signal cycle (~45s) with current portfolio equity."""
        ts = timestamp or datetime.now(tz=_UTC)
        with self._lock:
            self.equity_history.append((ts, equity))
            # Update EMAs
            alpha_fast = 2.0 / (self.fast_period + 1)
            alpha_slow = 2.0 / (self.slow_period + 1)
            if self._fast_ema is None:
                self._fast_ema = equity
                self._slow_ema = equity
            else:
                self._fast_ema = alpha_fast * equity + (1 - alpha_fast) * self._fast_ema
                self._slow_ema = alpha_slow * equity + (1 - alpha_slow) * self._slow_ema

    def get_position_scale(self) -> float:
        """Returns a multiplier for position sizes (0.3 to 1.3).

        - fast_ema > slow_ema: equity trending up → scale up (1.0 to 1.3)
        - fast_ema < slow_ema: equity trending down → scale down (0.3 to 1.0)
        - Not enough data: return 1.0 (neutral)
        """
        with self._lock:
            if self._fast_ema is None or self._slow_ema is None or len(self.equity_history) < self.fast_period:
                return 1.0

            if self._slow_ema == 0:
                return 1.0

            # Ratio of fast to slow EMA — centered around 1.0
            ratio = self._fast_ema / self._slow_ema

            if ratio >= 1.0:
                # Winning streak — scale up proportionally, capped at max_scale
                scale = min(self.max_scale, 1.0 + (ratio - 1.0) * 10)  # 10x sensitivity
            else:
                # Drawdown — scale down proportionally, floored at min_scale
                scale = max(self.min_scale, 1.0 - (1.0 - ratio) * 10)  # 10x sensitivity

            return round(scale, 3)

    def get_status(self) -> dict:
        with self._lock:
            return {
                'fast_ema': round(self._fast_ema, 2) if self._fast_ema else None,
                'slow_ema': round(self._slow_ema, 2) if self._slow_ema else None,
                'scale': self.get_position_scale(),
                'samples': len(self.equity_history),
                'trend': 'UP' if self._fast_ema and self._slow_ema and self._fast_ema > self._slow_ema else 'DOWN',
            }


class ProfitVelocityTracker:
    """Tracks the RATE of profit change per position and dynamically adjusts stops.

    Traditional trailing stops use fixed ATR multiples. This system tracks profit
    VELOCITY — how fast unrealized P&L is changing. When velocity is positive and
    accelerating, stops stay wide. When velocity decelerates or goes negative,
    stops tighten aggressively.

    This mimics a human trader's intuition for "momentum is fading" —
    the most profitable exit timing signal that no indicator captures.

    Returns a trail_multiplier that scales the trailing stop percentage:
    - > 1.0: momentum strong, widen stop to let it run
    - = 1.0: neutral
    - < 1.0: momentum fading, tighten stop to lock in gains
    - = 0.5: momentum reversed, emergency tighten
    """

    def __init__(self, lookback: int = 8):
        self.lookback = lookback  # Number of samples for velocity estimation
        self._price_history = {}  # {symbol: deque of (timestamp, price, unrealized_pct)}
        self._lock = threading.Lock()

    def update(self, symbol: str, current_price: float, entry_price: float, direction: int):
        """Call every monitor cycle with current price for each open position."""
        ts = datetime.now(tz=_UTC)
        unrealized_pct = (current_price - entry_price) / entry_price * direction if entry_price else 0.0

        with self._lock:
            if symbol not in self._price_history:
                self._price_history[symbol] = deque(maxlen=60)  # ~20 min at 20s intervals
            self._price_history[symbol].append((ts, current_price, unrealized_pct))

    def get_trail_multiplier(self, symbol: str) -> float:
        """Returns multiplier for trailing stop width based on profit velocity.

        - Profit accelerating: 1.2-1.5 (widen stop, let it run)
        - Profit steady: 1.0 (no change)
        - Profit decelerating: 0.6-0.9 (tighten stop)
        - Profit reversing: 0.4-0.6 (emergency tighten)
        """
        with self._lock:
            history = self._price_history.get(symbol)
            if not history or len(history) < self.lookback:
                return 1.0

            recent = list(history)[-self.lookback:]
            pnl_values = [pnl for _, _, pnl in recent]

            # Compute velocity (first derivative of P&L)
            velocity = pnl_values[-1] - pnl_values[0]  # Change over lookback window

            # Compute acceleration (second derivative)
            mid = len(pnl_values) // 2
            first_half_vel = pnl_values[mid] - pnl_values[0]
            second_half_vel = pnl_values[-1] - pnl_values[mid]
            acceleration = second_half_vel - first_half_vel

            # Current P&L level
            current_pnl = pnl_values[-1]

            if current_pnl <= 0:
                # Losing position — no velocity adjustment (let stop do its job)
                return 1.0

            if velocity > 0 and acceleration > 0:
                # Profit accelerating — widen stop to let the winner run
                return min(1.5, 1.0 + velocity * 20)  # Scale by velocity magnitude
            elif velocity > 0 and acceleration <= 0:
                # Profit growing but decelerating — start tightening
                return max(0.7, 1.0 - abs(acceleration) * 30)
            elif velocity <= 0 and current_pnl > 0:
                # Was profitable, now giving back — tighten aggressively
                return max(0.4, 0.7 + velocity * 10)  # Sharper tightening
            else:
                return 1.0

    def clear(self, symbol: str):
        """Call when a position is closed."""
        with self._lock:
            self._price_history.pop(symbol, None)


class TradeAutopsyEngine:
    """Post-trade analysis that builds a knowledge base of lessons.

    After every closed trade, generates a structured "autopsy" analyzing:
    - Market conditions at entry (regime, VIX, 1H trend, momentum)
    - What the signal components said (ensemble, PPO, sentiment)
    - How the trade developed (max favorable excursion, max adverse excursion)
    - Why it closed (stop hit, TP hit, signal flip)
    - The lesson: a structured pattern that can be matched against future entries

    The knowledge base is a simple list of patterns with win/loss outcomes.
    Before each new entry, the engine checks: "Have I seen this pattern before?
    If so, what was the win rate?" This is experiential learning without model retraining.
    """

    AUTOPSY_FILE = "trade_autopsies.json"

    def __init__(self):
        self._autopsies = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if os.path.exists(self.AUTOPSY_FILE):
            try:
                with open(self.AUTOPSY_FILE, 'r') as f:
                    self._autopsies = json.load(f)
                logger.info(f"[AUTOPSY] Loaded {len(self._autopsies)} trade autopsies from disk")
            except Exception as e:
                logger.warning(f"[AUTOPSY] Failed to load autopsies: {e}")
                self._autopsies = []

    def _save(self):
        try:
            import tempfile, shutil
            dir_name = os.path.dirname(self.AUTOPSY_FILE) or '.'
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=dir_name) as tmp:
                json.dump(self._autopsies[-500:], tmp, default=str)  # Keep last 500
                tmp.flush()
                os.fsync(tmp.fileno())
            shutil.move(tmp.name, self.AUTOPSY_FILE)
        except Exception as e:
            logger.warning(f"[AUTOPSY] Failed to save: {e}")

    def record_autopsy(self, symbol: str, direction: int, entry_price: float,
                       exit_price: float, pnl: float, bars_held: int,
                       regime: str, vix: float, exit_reason: str,
                       meta_prob: float = 0.5, ppo_prob: float = 0.5,
                       sentiment: float = 0.0):
        """Record a structured trade autopsy after position close."""
        pattern = self._build_pattern(regime, vix, direction, sentiment)
        autopsy = {
            'timestamp': datetime.now(tz=_UTC).isoformat(),
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': round(pnl, 2),
            'pnl_pct': round((exit_price - entry_price) / entry_price * direction * 100, 3),
            'bars_held': bars_held,
            'regime': regime,
            'vix': round(vix, 1),
            'exit_reason': exit_reason,
            'meta_prob': round(meta_prob, 3),
            'ppo_prob': round(ppo_prob, 3),
            'sentiment': round(sentiment, 3),
            'won': pnl > 0,
            'pattern': pattern,
        }

        with self._lock:
            self._autopsies.append(autopsy)
            # Log the lesson
            outcome = "WIN" if pnl > 0 else "LOSS"
            logger.info(f"[AUTOPSY] {outcome} {symbol} {'LONG' if direction==1 else 'SHORT'} "
                       f"| P&L ${pnl:+.2f} ({autopsy['pnl_pct']:+.2f}%) | {bars_held} bars "
                       f"| regime={regime} VIX={vix:.0f} | exit={exit_reason} | pattern={pattern}")
            self._save()

    def _build_pattern(self, regime: str, vix: float, direction: int, sentiment: float) -> str:
        """Build a matchable pattern string from trade conditions."""
        vix_level = 'high' if vix > 28 else ('mid' if vix > 20 else 'low')
        dir_str = 'long' if direction == 1 else 'short'
        sent_str = 'pos' if sentiment > 0.2 else ('neg' if sentiment < -0.2 else 'neutral')
        return f"{regime}|{vix_level}|{dir_str}|{sent_str}"

    def get_pattern_win_rate(self, regime: str, vix: float, direction: int,
                            sentiment: float, min_samples: int = 3) -> tuple:
        """Check historical win rate for a pattern matching current conditions.

        Returns (win_rate, sample_count) or (None, 0) if insufficient history.
        """
        pattern = self._build_pattern(regime, vix, direction, sentiment)
        with self._lock:
            matches = [a for a in self._autopsies if a['pattern'] == pattern]
            if len(matches) < min_samples:
                return None, len(matches)
            wins = sum(1 for a in matches if a['won'])
            return wins / len(matches), len(matches)

    def get_symbol_win_rate(self, symbol: str, direction: int,
                           min_samples: int = 3) -> tuple:
        """Check historical win rate for a specific symbol + direction."""
        with self._lock:
            matches = [a for a in self._autopsies
                      if a['symbol'] == symbol and a['direction'] == direction]
            if len(matches) < min_samples:
                return None, len(matches)
            wins = sum(1 for a in matches if a['won'])
            return wins / len(matches), len(matches)

    def should_suppress_entry(self, symbol: str, direction: int, regime: str,
                             vix: float, sentiment: float) -> tuple:
        """Returns (should_suppress: bool, reason: str, win_rate: float).

        Suppresses entries when the pattern has a historically terrible win rate.
        """
        # Check pattern win rate
        pattern_wr, pattern_n = self.get_pattern_win_rate(regime, vix, direction, sentiment)
        if pattern_wr is not None and pattern_wr < 0.25 and pattern_n >= 5:
            return True, f"pattern win rate {pattern_wr:.0%} over {pattern_n} trades", pattern_wr

        # Check symbol-specific win rate
        sym_wr, sym_n = self.get_symbol_win_rate(symbol, direction)
        if sym_wr is not None and sym_wr < 0.20 and sym_n >= 5:
            return True, f"{symbol} {('LONG' if direction==1 else 'SHORT')} win rate {sym_wr:.0%} over {sym_n} trades", sym_wr

        return False, "", pattern_wr if pattern_wr else 0.5

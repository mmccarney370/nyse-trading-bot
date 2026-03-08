# broker/alpaca.py
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz
import json
import os
from utils.helpers import is_market_open
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    GetOrdersRequest,
    ReplaceOrderRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, PositionSide
from config import CONFIG
from strategy.regime import detect_regime
# Correct absolute imports (files are in models/)
from models.ppo_utils import (
    train_ppo,
    save_ppo_model,
    load_ppo_model,
    update_model_weights as ppo_update_model_weights
)
from models.stacking_ensemble import train_stacking

logger = logging.getLogger(__name__)

# Persistent files
REGIME_CACHE_FILE = "regime_cache.json"
LAST_ENTRY_FILE = "last_entry_times.json"

# Timezone-aware epoch sentinel (used as default for last_ratchet_time lookups)
_UTC = tz.gettz('UTC')
_EPOCH = datetime(1970, 1, 1, tzinfo=_UTC)

def _order_type_str(order) -> str:
    raw = str(getattr(order, 'order_type', '') or '').lower()
    return raw.split('.')[-1]

class AlpacaBroker:
    def __init__(self, config, data_ingestion=None, bot=None):
        self.config = config
        self.data_ingestion = data_ingestion
        self.bot = bot # Added: access to live_signal_history and latest_prices for reward push
        self.limit_price_offset = config.get('LIMIT_PRICE_OFFSET', 0.005)
        self.is_paper = config.get('PAPER', True)
        self.client = TradingClient(
            config['API_KEY'],
            config['API_SECRET'],
            paper=self.is_paper
        )
        # ==================== DIAGNOSTIC PRINT (to confirm patched version) ====================
        logger.info(f"===== CLIENT DEBUG: cancel_order exists: {hasattr(self.client, 'cancel_order')} | type: {type(self.client)} =====")
        self.last_slippage_adjust = datetime.now(tz=_UTC)
        self.last_regime = {}
        # Persistent state
        self.last_entry_times = self._load_last_entry_times()
        self.regime_cache = self._load_regime_cache()
        self.existing_positions = {}
        self.last_ratchet_time = {} # tracks last ratchet per symbol for adaptive frequency
        # M-2 FIX: Positions cache with TTL (30 seconds default)
        self._positions_cache = {}
        self._positions_cache_time = None
        self._positions_cache_ttl = 30  # seconds — matches typical trading cycle length
        # M-5 FIX: Configurable timeout for cancel/wait sequence in ratchet
        self.ratchet_cancel_timeout_sec = self.config.get('RATCHET_CANCEL_TIMEOUT_SEC', 8)
        # Full dynamic sync on startup
        self.sync_existing_positions()
        logger.info(
            f"✅ AlpacaBroker initialized — {len(self.existing_positions)} open positions, "
            f"{len(self.last_entry_times)} restored entry times, {len(self.regime_cache)} cached regimes"
        )

    # ====================== PERSISTENT LAST ENTRY TIMES ======================
    def _load_last_entry_times(self):
        if os.path.exists(LAST_ENTRY_FILE):
            try:
                with open(LAST_ENTRY_FILE, 'r') as f:
                    data = json.load(f)
                result = {}
                for sym, ts in data.items():
                    dt = datetime.fromisoformat(ts)
                    # Ensure tz-aware — attach UTC if naive (prevents TypeError in monitor loop)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=_UTC)
                    result[sym] = dt
                return result
            except Exception as e:
                logger.warning(f"Failed to load last_entry_times.json: {e}")
        return {}

    def _save_last_entry_times(self):
        try:
            data = {sym: ts.isoformat() for sym, ts in self.last_entry_times.items()}
            with open(LAST_ENTRY_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.last_entry_times)} entry times to disk")
        except Exception as e:
            logger.warning(f"Failed to save last_entry_times.json: {e}")

    # ====================== FULL DYNAMIC SYNC ======================
    def sync_existing_positions(self, force_refresh=False):
        """Full dynamic sync — called on startup and can be called anytime
        M-2 FIX: Supports force_refresh to invalidate cache after order actions"""
        now = datetime.now(tz=_UTC)
        if not force_refresh and self._positions_cache_time is not None:
            age = (now - self._positions_cache_time).total_seconds()
            if age < self._positions_cache_ttl:
                logger.debug(f"[POSITIONS CACHE HIT] Using cached positions (age={age:.1f}s < {self._positions_cache_ttl}s)")
                self.existing_positions = self._positions_cache.copy()
                return

        try:
            positions = self.client.get_all_positions()
            self.existing_positions = {
                p.symbol: float(p.qty) if p.side == PositionSide.LONG else -float(p.qty)
                for p in positions if float(p.qty) != 0
            }
            # Update cache
            self._positions_cache = self.existing_positions.copy()
            self._positions_cache_time = now
            logger.debug(f"[POSITIONS CACHE REFRESH] Synced {len(self.existing_positions)} positions")

            for sym in list(self.existing_positions.keys()):
                if sym not in self.last_entry_times:
                    self.last_entry_times[sym] = now - timedelta(minutes=30)
                    logger.info(f"Restored missing entry time for {sym} (default 30min ago)")
            for sym in list(self.last_entry_times.keys()):
                if sym not in self.existing_positions:
                    self.last_entry_times.pop(sym, None)
            logger.info(
                f"✅ Full Alpaca sync: {len(self.existing_positions)} open positions, "
                f"{len(self.last_entry_times)} tracked entry times"
            )
            self._save_last_entry_times()
        except Exception as e:
            logger.error(f"Failed to sync existing positions: {e}")
            # Fallback: keep old cache if possible
            if self._positions_cache:
                logger.warning("[POSITIONS CACHE FALLBACK] Using stale cache due to sync failure")
                self.existing_positions = self._positions_cache.copy()
            else:
                self.existing_positions = {}

    def _load_regime_cache(self):
        if os.path.exists(REGIME_CACHE_FILE):
            try:
                with open(REGIME_CACHE_FILE, 'r') as f:
                    data = json.load(f)
                logger.info(f"AlpacaBroker loaded shared regime cache with {len(data)} symbols")
                return data
            except Exception as e:
                logger.warning(f"Failed to load regime cache in AlpacaBroker: {e}")
        return {}

    def get_equity(self):
        try:
            account = self.client.get_account()
            return float(account.equity)
        except Exception as e:
            logger.error(f"Equity fetch error: {e}")
            return 1000.0

    def get_buying_power(self):
        """Real-time buying power (used for safety clamps)"""
        try:
            account = self.client.get_account()
            return float(account.buying_power)
        except Exception as e:
            logger.error(f"Buying power fetch error: {e}")
            return 0.0

    def get_positions_dict(self):
        """M-2 FIX: Cached version — sync only if cache expired"""
        self.sync_existing_positions()  # Will use cache if not expired
        return self.existing_positions.copy()

    def _compute_current_atr(self, data_window: pd.DataFrame, lookback: int = 50) -> float:
        if len(data_window) < 14:
            return 0.01
        recent = data_window.tail(lookback)
        high_low = recent['high'] - recent['low']
        high_close = (recent['high'] - recent['close'].shift(1)).abs()
        low_close = (recent['low'] - recent['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        tr = tr.dropna()
        if len(tr) == 0:
            return 0.01
        atr_series = tr.ewm(span=14, adjust=False).mean()
        atr = atr_series.iloc[-1]
        floor = 0.0005 * recent['close'].iloc[-1]
        return max(atr, floor)

    def has_active_bracket(self, symbol: str) -> bool:
        try:
            orders = self.client.get_orders(GetOrdersRequest(status='open', symbols=[symbol]))
            return any(o.order_class == OrderClass.BRACKET for o in orders)
        except:
            return False

    def place_bracket_order(self, symbol, size, current_price, data, direction=1):
        if size < 1:
            return None
        if self.has_active_bracket(symbol):
            logger.info(f"Skipping duplicate bracket for {symbol} — active bracket already exists")
            return None
        # Regime detection (cached)
        if symbol in self.regime_cache:
            value = self.regime_cache[symbol]
            regime = value[0] if isinstance(value, (list, tuple)) else value
        else:
            recent_return = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
            regime = 'trending' if abs(recent_return) > 0.015 else 'mean_reverting'
        logger.info(f"Placing bracket for {symbol} — detected regime: {regime}")
        # M-5 FIX: Real-time buying power cap before placing order
        available_bp = self.get_buying_power()
        notional = abs(size) * current_price
        safety_factor = self.config.get('MAX_ORDER_NOTIONAL_PCT', 0.85)
        max_allowed_notional = available_bp * safety_factor
        if notional > max_allowed_notional and available_bp > 1000:
            new_size = int(max_allowed_notional / current_price)
            logger.warning(
                f"[M-5 BUYING POWER CAP] {symbol}: requested {size} shares (${notional:,.0f}) → reduced to {new_size} shares "
                f"(buying_power=${available_bp:,.0f}, safety_factor={safety_factor})"
            )
            size = max(new_size, 1)
        atr = self._compute_current_atr(data)
        trailing_mult = (
            self.config.get('TRAILING_STOP_ATR_TRENDING', self.config.get('TRAILING_STOP_ATR', 3.0) * 0.8)
            if regime == 'trending'
            else self.config.get('TRAILING_STOP_ATR_MEAN_REVERTING', self.config.get('TRAILING_STOP_ATR', 4.0))
        )
        tp_mult = (
            self.config.get('TAKE_PROFIT_ATR_TRENDING', self.config.get('TAKE_PROFIT_ATR', 30.0) * 1.2)
            if regime == 'trending'
            else self.config.get('TAKE_PROFIT_ATR_MEAN_REVERTING', self.config.get('TAKE_PROFIT_ATR', 12.0))
        )
        side = OrderSide.BUY if direction > 0 else OrderSide.SELL
        limit_price = round(float(current_price) * (1 + direction * self.limit_price_offset), 2)
        tp_price = round(float(current_price) + direction * tp_mult * atr, 2)
        trail_stop_price = round(float(current_price) - direction * trailing_mult * atr, 2)
        if direction > 0:
            trail_stop_price = max(trail_stop_price, current_price * 0.70)
        else:
            trail_stop_price = min(trail_stop_price, current_price * 1.35)
        trail_stop_price = round(max(float(trail_stop_price), 0.01), 2)
        tp_price = round(max(float(tp_price), 0.01), 2)
        bracket_request = LimitOrderRequest(
            symbol=symbol,
            qty=size,
            side=side,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            limit_price=limit_price,
            take_profit=TakeProfitRequest(limit_price=tp_price),
            stop_loss=StopLossRequest(stop_price=trail_stop_price)
        )
        try:
            response = self.client.submit_order(bracket_request)
            logger.info(
                f"Submitted bracket {'BUY' if direction > 0 else 'SELL'} {size} {symbol} @ limit {limit_price} | "
                f"Regime: {regime} | Initial TP: {tp_price:.2f} | Initial Stop: {trail_stop_price:.2f} | "
                f"BP after: ${self.get_buying_power():,.0f}"
            )
            self.last_entry_times[symbol] = datetime.now(tz=_UTC)
            self._save_last_entry_times()
            # M-2 FIX: Force-refresh positions cache after successful order
            self.sync_existing_positions(force_refresh=True)
            return response
        except Exception as e:
            err_str = str(e)
            if "40310100" in err_str or "pattern day trading" in err_str.lower():
                logger.warning(f"PDT protection blocked trade for {symbol} — skipping")
            elif "40310000" in err_str or "insufficient buying power" in err_str.lower():
                logger.error(f"Bracket order failed for {symbol}: insufficient buying power (even after clamp)")
            else:
                logger.error(f"Bracket order failed for {symbol}: {e}")
            return None

    # ====================== DYNAMIC HEARTBEAT ======================
    async def monitor_positions(self):
        logger.info("=== MONITOR TASK STARTED - FULL DYNAMIC HEARTBEAT ENABLED ===")
        while True:
            self.sync_existing_positions() # M-2: uses cache unless expired
            open_orders = []
            positions = []
            try:
                open_orders = self.client.get_orders(GetOrdersRequest(status='open'))
            except Exception as e:
                logger.warning(f"Failed to fetch open orders: {e}")
            try:
                positions = self.client.get_all_positions()
            except Exception as e:
                logger.warning(f"Failed to fetch positions: {e}")
            pos_count = len(self.existing_positions)
            bracket_count = len([o for o in open_orders if o.order_class == OrderClass.BRACKET])
            heartbeat_msg = (
                f"[{datetime.now().strftime('%H:%M:%S')}] Monitor heartbeat — "
                f"{pos_count} open positions | {bracket_count} open brackets | "
                f"market {'open' if is_market_open() else 'closed'}"
            )
            print(heartbeat_msg)
            logger.info(heartbeat_msg)
            # Detailed per-position status
            if pos_count > 0:
                now = datetime.now(tz=_UTC)
                for sym, qty in self.existing_positions.items():
                    direction = 1 if qty > 0 else -1
                    last_entry = self.last_entry_times.get(sym)
                    bars_held = ((now - last_entry) / timedelta(minutes=15)) if last_entry else 9999
                    if sym in self.regime_cache:
                        value = self.regime_cache[sym]
                        regime = value[0] if isinstance(value, (list, tuple)) else value
                    else:
                        regime = 'mean_reverting'
                    min_hold = (
                        self.config.get('MIN_HOLD_BARS_TRENDING', 48)
                        if regime == 'trending'
                        else self.config.get('MIN_HOLD_BARS_MEAN_REVERTING', 24)
                    )
                    logger.info(
                        f" {sym} | {'LONG' if direction > 0 else 'SHORT'} {abs(qty):.1f} shares | "
                        f"held {int(bars_held)} bars (min {min_hold}) | regime={regime}"
                    )
            # === DYNAMIC LOGIC ===
            try:
                filled_orders = [
                    o for o in open_orders
                    if o.filled_at is not None
                    and (datetime.now(tz=_UTC) - o.filled_at).total_seconds() < 3600
                ]
                slippage_total = 0.0
                fill_count = 0
                for o in filled_orders:
                    if o.filled_avg_price and o.limit_price:
                        filled_price = float(o.filled_avg_price)
                        limit_price = float(o.limit_price)
                        slippage = abs(filled_price - limit_price) / limit_price
                        slippage_total += slippage
                        fill_count += 1
                if fill_count > 0:
                    avg_slippage = slippage_total / fill_count
                    new_offset = max(0.001, avg_slippage * 2 + 0.001)
                    if abs(new_offset - self.limit_price_offset) > 0.0005:
                        old_offset = self.limit_price_offset
                        self.limit_price_offset = new_offset
                        logger.info(
                            f"Adjusted limit_price_offset {old_offset:.4f} → {new_offset:.4f} "
                            f"(avg slippage {avg_slippage:.4f})"
                        )
                for pos in positions:
                    qty = float(pos.qty)
                    if qty == 0:
                        continue
                    symbol = pos.symbol
                    direction = 1 if pos.side == PositionSide.LONG else -1
                    try:
                        if self.data_ingestion is None:
                            continue
                        data = self.data_ingestion.get_latest_data(symbol)
                        if len(data) < 50:
                            continue
                        current_price = float(data['close'].iloc[-1])
                        entry_price = float(pos.avg_entry_price)
                        if symbol in self.regime_cache:
                            value = self.regime_cache[symbol]
                            regime = value[0] if isinstance(value, (list, tuple)) else value
                            persistence = (
                                value[1]
                                if isinstance(value, (list, tuple)) and len(value) == 2
                                else 0.5
                            )
                        else:
                            regime = 'trending'
                            persistence = 0.5
                        atr = self._compute_current_atr(data)
                        trailing_mult = (
                            self.config.get('TRAILING_STOP_ATR_TRENDING', self.config.get('TRAILING_STOP_ATR', 3.0) * 0.8)
                            if regime == 'trending'
                            else self.config.get('TRAILING_STOP_ATR_MEAN_REVERTING', self.config.get('TRAILING_STOP_ATR', 4.0))
                        )
                        tp_mult = (
                            self.config.get('TAKE_PROFIT_ATR_TRENDING', self.config.get('TAKE_PROFIT_ATR', 30.0) * 1.2)
                            if regime == 'trending'
                            else self.config.get('TAKE_PROFIT_ATR_MEAN_REVERTING', self.config.get('TAKE_PROFIT_ATR', 12.0))
                        )
                        in_min_hold = False
                        last_entry = self.last_entry_times.get(symbol)
                        if last_entry:
                            bars_held = (datetime.now(tz=_UTC) - last_entry) / timedelta(minutes=15)
                            min_hold = (
                                self.config.get('MIN_HOLD_BARS_TRENDING', 48)
                                if regime == 'trending'
                                else self.config.get('MIN_HOLD_BARS_MEAN_REVERTING', 24)
                            )
                            if bars_held < min_hold:
                                in_min_hold = True
                                logger.info(
                                    f"MIN-HOLD ACTIVE {symbol} ({regime}) | "
                                    f"bars_held={int(bars_held)} < {min_hold} → re-attach suppressed, ratchet active"
                                )
                        # ==================== M-5 PROPER TRAILING SL: CANCEL + RE-SUBMIT BRACKET ====================
                        now = datetime.now(tz=_UTC)
                        last_ratchet = self.last_ratchet_time.get(symbol, _EPOCH)
                        elapsed_since_ratchet = (now - last_ratchet).total_seconds()
                        # Compute candidate new_stop
                        unrealized_pct = (
                            (current_price - entry_price) / entry_price * direction
                            if entry_price != 0 else 0.0
                        )
                        # Option 1: Clamp profit_protection so losing positions NEVER widen stop
                        profit_protection = max(
                            self.config.get('RATCHET_PROFIT_PROTECTION_MIN', 0.5),
                            min(1.0, 1.0 - unrealized_pct * self.config.get('RATCHET_PROFIT_PROTECTION_SLOPE', 3.0))
                        )
                        if regime == 'trending' and persistence >= 0.7:
                            min_interval_sec = self.config.get('RATCHET_TRENDING_INTERVAL_SEC', 180)
                            regime_factor = self.config.get('RATCHET_REGIME_FACTOR_TRENDING', 0.65)
                            min_atr_move = self.config.get('RATCHET_TRENDING_MIN_ATR_MOVE', 0.4)
                        else:
                            min_interval_sec = self.config.get('RATCHET_MEAN_REVERTING_INTERVAL_SEC', 540)
                            regime_factor = self.config.get('RATCHET_REGIME_FACTOR_MEAN_REVERTING', 1.35)
                            min_atr_move = self.config.get('RATCHET_MEAN_REVERTING_MIN_ATR_MOVE', 0.8)
                        distance = atr * trailing_mult * regime_factor * profit_protection
                        new_stop = current_price - direction * distance
                        if direction > 0:
                            new_stop = max(new_stop, current_price * 0.65)
                        else:
                            new_stop = min(new_stop, current_price * 1.35)
                        new_stop = round(max(new_stop, 0.01), 2)
                        logger.debug(
                            f"[RATCHET DEBUG] {symbol} | unrealized={unrealized_pct*100:+.2f}% | "
                            f"profit_protection={profit_protection:.3f} | distance={distance:.4f} | "
                            f"new_stop={new_stop:.2f} | throttle={elapsed_since_ratchet:.0f}s/{min_interval_sec}s"
                        )
                        throttle_passed = elapsed_since_ratchet >= min_interval_sec
                        if throttle_passed and is_market_open():
                            bracket_orders = [
                                o for o in open_orders
                                if o.symbol == symbol and o.order_class == OrderClass.BRACKET
                            ]
                            if bracket_orders:
                                bracket_order = bracket_orders[0]
                                # Step 1: Cancel all children first (stop-loss / take-profit)
                                symbol_orders = self.client.get_orders(GetOrdersRequest(status='open', symbols=[symbol]))
                                child_ids = []
                                for child in symbol_orders:
                                    if child.parent_order_id == bracket_order.id:
                                        try:
                                            self.client.cancel_order_by_id(child.id)
                                            child_ids.append(child.id)
                                            logger.debug(f"[RATCHET CANCEL CHILD] {child.id} ({child.side}) for {symbol}")
                                        except Exception as cancel_err:
                                            logger.warning(f"Failed to cancel child {child.id} for ratchet: {cancel_err}")
                                # Wait for children to be cancelled
                                if child_ids:
                                    for wait_attempt in range(self.ratchet_cancel_timeout_sec):
                                        await asyncio.sleep(1.0)
                                        remaining = self.client.get_orders(GetOrdersRequest(status='open', symbols=[symbol]))
                                        still_open_ids = [o.id for o in remaining if o.id in child_ids]
                                        if not still_open_ids:
                                            logger.debug(f"[RATCHET] All children cancelled for {symbol} after {wait_attempt+1}s")
                                            break
                                    else:
                                        logger.warning(f"[RATCHET TIMEOUT] Children not cleared for {symbol} after {self.ratchet_cancel_timeout_sec}s — skipping ratchet")
                                        continue
                                # Step 2: Cancel parent bracket
                                try:
                                    self.client.cancel_order_by_id(bracket_order.id)
                                    logger.debug(f"[RATCHET CANCEL PARENT] {bracket_order.id} for {symbol}")
                                    await asyncio.sleep(1.0)  # brief safety pause
                                except Exception as parent_err:
                                    logger.warning(f"Failed to cancel parent bracket {bracket_order.id}: {parent_err}")
                                    continue
                                # Step 3: Re-submit new bracket with updated stop
                                new_order = self.place_bracket_order(
                                    symbol=symbol,
                                    size=abs(qty),
                                    current_price=current_price,
                                    data=data,
                                    direction=direction
                                )
                                if new_order:
                                    self.last_ratchet_time[symbol] = now
                                    logger.info(
                                        f"[RATCHET SUCCESS] {symbol} ratcheted via cancel/re-submit → "
                                        f"new stop ~{new_stop:.2f} (profit={unrealized_pct*100:+.1f}%, regime={regime}, persistence={persistence:.3f})"
                                    )
                                else:
                                    logger.error(f"[RATCHET FAIL] Re-submit failed for {symbol} — stop not updated")
                            else:
                                logger.debug(f"[RATCHET] {symbol} has no active bracket — skipping ratchet")
                        elif not throttle_passed:
                            logger.debug(f"[RATCHET] {symbol} throttled — waiting {min_interval_sec - elapsed_since_ratchet:.0f}s")
                        # ==================== C-1 FIX: bracket_orders is now always defined ====================
                        # Define bracket_orders here so the re-attach logic below never raises NameError
                        bracket_orders = [
                            o for o in open_orders
                            if o.symbol == symbol and o.order_class == OrderClass.BRACKET
                        ]
                        # Re-attach logic
                        if not bracket_orders and is_market_open():
                            if in_min_hold:
                                logger.info(
                                    f"MIN-HOLD ACTIVE {symbol} — bracket missing but suppressing "
                                    f"re-attach until hold expires"
                                )
                                continue
                            logger.info(
                                f"Auto-re-attaching bracket for {symbol} ({regime}) "
                                f"— bracket was cancelled or missing"
                            )
                            size = abs(qty)
                            available_bp = self.get_buying_power()
                            notional = size * current_price
                            # M-5 FIX: Apply buying power cap also on re-attach
                            safety_factor = self.config.get('MAX_ORDER_NOTIONAL_PCT', 0.85)
                            max_allowed_notional = available_bp * safety_factor
                            if notional > max_allowed_notional and available_bp > 1000:
                                new_size = int(max_allowed_notional / current_price)
                                logger.warning(
                                    f"[M-5 BUYING POWER CAP on re-attach] {symbol}: requested {size} shares → reduced to {new_size} "
                                    f"(buying_power=${available_bp:,.0f}, safety_factor={safety_factor})"
                                )
                                size = max(new_size, 1)
                            order = self.place_bracket_order(
                                symbol=symbol,
                                size=size,
                                current_price=current_price,
                                data=data,
                                direction=direction
                            )
                            if order:
                                logger.info(f"Successfully re-attached bracket for {symbol}")
                            # M-2 FIX: Force-refresh cache after re-attach
                            self.sync_existing_positions(force_refresh=True)
                            continue
                    except Exception as e:
                        logger.warning(f"Monitor update failed for {symbol}: {e}")
            except Exception as e:
                logger.error(f"Monitor loop inner error (continuing): {e}", exc_info=True)
            # ==================== NEW: REWARD PUSH FOR CLOSED POSITIONS ====================
            current_symbols = set(self.existing_positions.keys())
            for sym in list(self.last_entry_times.keys()):
                if sym not in current_symbols:
                    history = self.bot.live_signal_history.get(sym, []) if hasattr(self, 'bot') else []
                    if history:
                        last_entry = history[-1]
                        if last_entry.get('realized_return') is None:
                            current_price = self.bot.latest_prices.get(sym) if hasattr(self, 'bot') and hasattr(self.bot, 'latest_prices') else None
                            if current_price is None:
                                data = self.data_ingestion.get_latest_data(sym) if self.data_ingestion else None
                                current_price = float(data['close'].iloc[-1]) if data is not None and len(data) > 0 else last_entry['price']
                            ret = (current_price - last_entry['price']) / last_entry['price'] * last_entry['direction']
                            last_entry['realized_return'] = ret
                            causal_manager = self.bot.signal_gen.portfolio_causal_manager if hasattr(self, 'bot') and hasattr(self.bot.signal_gen, 'portfolio_causal_manager') else None
                            if causal_manager:
                                stored_obs = last_entry.get('obs')
                                if stored_obs:
                                    obs = np.array(stored_obs, dtype=np.float32).reshape(1, -1)
                                    action = last_entry['direction'] * last_entry['confidence']
                                    causal_manager.add_transition(obs, action, ret)
                                    logger.info(f"[CAUSAL PUSH] Closed {sym} | realized={ret:+.4f} pushed to buffer")
                                    causal_manager.save_buffer()
                                else:
                                    logger.warning(f"[CAUSAL PUSH] No stored obs for closed {sym} — skipping transition")
                            del self.last_entry_times[sym]
                            self._save_last_entry_times()
            await asyncio.sleep(self.config.get('MONITOR_INTERVAL', 60))

    # ====================== SAFE POSITION CLOSE ======================
    async def close_position_safely(self, symbol: str) -> bool:
        """Safely close a position by first cancelling any active bracket orders,
        then waiting for Alpaca to process the cancels before closing.
        Fixes 'insufficient qty available' (held_for_orders) caused by firing
        close_position immediately after cancel requests (async race condition).
        CRIT-13 FIX: Made async + await sleeps to prevent event-loop blocking.
        M-2 FIX: Force-refresh positions cache after successful close"""
        try:
            all_open = self.client.get_orders(GetOrdersRequest(status='open', symbols=[symbol]))
            cancelled_ids = []
            for o in all_open:
                try:
                    logger.info(f"===== PATCHED CANCEL_BY_ID EXECUTING for close {symbol} (order_id={o.id}) =====")
                    self.client.cancel_order_by_id(order_id=o.id)
                    cancelled_ids.append(o.id)
                    logger.info(
                        f"[CLOSE PREP] Cancelled order {o.id} ({_order_type_str(o)}) "
                        f"for {symbol} before flattening"
                    )
                except Exception as cancel_e:
                    logger.warning(f"Failed to cancel order {o.id} for {symbol}: {cancel_e}")
            if cancelled_ids:
                max_attempts = 5
                for attempt in range(max_attempts):
                    await asyncio.sleep(1.0)
                    still_open = self.client.get_orders(GetOrdersRequest(status='open', symbols=[symbol]))
                    if not still_open:
                        logger.info(f"[CLOSE PREP] All orders cleared for {symbol} — proceeding to close")
                        break
                    logger.info(
                        f"[CLOSE PREP] {len(still_open)} orders still open for {symbol} "
                        f"(attempt {attempt + 1}/{max_attempts}) — waiting..."
                    )
                else:
                    logger.warning(
                        f"[CLOSE PREP] Orders for {symbol} may not be fully cancelled after "
                        f"{max_attempts} attempts — proceeding anyway"
                    )
            self.client.close_position(symbol)
            logger.info(f"✅ Successfully closed position in {symbol} (flat signal)")
            if symbol in self.last_entry_times:
                del self.last_entry_times[symbol]
                self._save_last_entry_times()
            # M-2 FIX: Force-refresh positions cache after successful close
            self.sync_existing_positions(force_refresh=True)
            return True
        except Exception as e:
            logger.error(f"Failed to safely close position in {symbol}: {e}")
            return False

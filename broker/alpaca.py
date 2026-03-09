# broker/alpaca.py
# Event-driven broker: trailing stops, ReplaceOrderRequest PATCH, fractional shares, extended hours
import logging
import asyncio
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz
import json
import os
import tempfile
import shutil

from utils.helpers import is_market_open
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest,
    TrailingStopOrderRequest,
    ReplaceOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderType, PositionSide, PositionIntent
from config import CONFIG
from broker.order_tracker import OrderTracker, GroupState

logger = logging.getLogger(__name__)

# Persistent files
REGIME_CACHE_FILE = "regime_cache.json"
LAST_ENTRY_FILE = "last_entry_times.json"

_UTC = tz.gettz('UTC')
_EPOCH = datetime(1970, 1, 1, tzinfo=_UTC)


def _order_type_str(order) -> str:
    raw = str(getattr(order, 'order_type', '') or '').lower()
    return raw.split('.')[-1]


class AlpacaBroker:
    def __init__(self, config, data_ingestion=None, bot=None):
        self.config = config
        self.data_ingestion = data_ingestion
        self.bot = bot
        self.limit_price_offset = config.get('LIMIT_PRICE_OFFSET', 0.005)
        self.is_paper = config.get('PAPER', True)
        self.use_extended_hours = config.get('EXTENDED_HOURS', True)
        self.use_fractional = config.get('FRACTIONAL_SHARES', True)
        self.client = TradingClient(
            config['API_KEY'],
            config['API_SECRET'],
            paper=self.is_paper
        )
        # Order tracker (persistent state machine)
        self.tracker = OrderTracker()
        # Legacy state (kept for bot.py compatibility)
        self.last_entry_times = self._load_last_entry_times()
        self.regime_cache = self._load_regime_cache()
        self.existing_positions = {}
        self.last_ratchet_time = {}
        self._positions_lock = threading.Lock()  # Thread safety for sync_existing_positions
        # Positions cache with TTL
        self._positions_cache = {}
        self._positions_cache_time = None
        self._positions_cache_ttl = 30
        self._last_known_equity = 100000.0
        # Full dynamic sync on startup
        self.sync_existing_positions()
        self._reconcile_tracker_on_startup()
        logger.info(
            f"AlpacaBroker initialized — {len(self.existing_positions)} open positions, "
            f"{len(self.tracker.get_open_groups())} tracked groups, "
            f"extended_hours={self.use_extended_hours}, fractional={self.use_fractional}"
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
            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(LAST_ENTRY_FILE) or '.', suffix='.tmp')
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
            shutil.move(tmp_path, LAST_ENTRY_FILE)
        except Exception as e:
            logger.warning(f"Failed to save last_entry_times.json: {e}")

    # ====================== FULL DYNAMIC SYNC ======================
    def sync_existing_positions(self, force_refresh=False):
        now = datetime.now(tz=_UTC)
        with self._positions_lock:
            if not force_refresh and self._positions_cache_time is not None:
                age = (now - self._positions_cache_time).total_seconds()
                if age < self._positions_cache_ttl:
                    self.existing_positions = self._positions_cache.copy()
                    return
        try:
            positions = self.client.get_all_positions()
            new_positions = {
                p.symbol: float(p.qty) if p.side == PositionSide.LONG else -float(p.qty)
                for p in positions if float(p.qty) != 0
            }
            with self._positions_lock:
                self.existing_positions = new_positions
                self._positions_cache = new_positions.copy()
                self._positions_cache_time = now
            for sym in list(new_positions.keys()):
                if sym not in self.last_entry_times:
                    self.last_entry_times[sym] = now - timedelta(minutes=30)
            for sym in list(self.last_entry_times.keys()):
                if sym not in new_positions:
                    self.last_entry_times.pop(sym, None)
            self._save_last_entry_times()
        except Exception as e:
            logger.error(f"Failed to sync existing positions: {e}")
            with self._positions_lock:
                if self._positions_cache:
                    self.existing_positions = self._positions_cache.copy()
                else:
                    self.existing_positions = {}

    def _reconcile_tracker_on_startup(self):
        """Full reconciliation of OrderTracker vs Alpaca positions + open orders on restart.

        Handles:
        1. Stale tracker groups (no Alpaca position) → remove
        2. Orphaned positions (no tracker group) → create group + attach trailing stop
        3. Tracked positions with missing trailing stop → resubmit
        4. Tracker direction mismatch with actual position → fix
        5. Pending entries that never filled → clean up
        """
        # --- Fetch all open orders from Alpaca ---
        open_orders = {}  # symbol → list of orders
        try:
            all_orders = self.client.get_orders(GetOrdersRequest(status='open'))
            for order in all_orders:
                sym = order.symbol
                if sym not in open_orders:
                    open_orders[sym] = []
                open_orders[sym].append(order)
            logger.info(f"[RECONCILE] Fetched {len(all_orders)} open orders across {len(open_orders)} symbols")
        except Exception as e:
            logger.error(f"[RECONCILE] Failed to fetch open orders: {e}")

        # --- 1. Remove stale tracker groups (no position on Alpaca) ---
        for sym in list(self.tracker.groups.keys()):
            group = self.tracker.groups[sym]
            if sym not in self.existing_positions:
                if group.state == GroupState.PENDING_ENTRY:
                    # Entry never filled — cancel any lingering entry order
                    if group.entry_order_id:
                        try:
                            self.client.cancel_order_by_id(group.entry_order_id)
                            logger.info(f"[RECONCILE] Canceled stale entry order for {sym}")
                        except Exception:
                            pass
                elif group.state == GroupState.OPEN:
                    # Position closed while bot was down (stop filled)
                    logger.info(f"[RECONCILE] {sym} was OPEN in tracker but no position — stop likely filled while offline")
                logger.info(f"[RECONCILE] Removing stale tracker group for {sym}")
                self.tracker.remove_group(sym)

        # --- 2. Process each actual Alpaca position ---
        for sym, qty in self.existing_positions.items():
            direction = 1 if qty > 0 else -1
            abs_qty = abs(qty)
            group = self.tracker.groups.get(sym)
            sym_orders = open_orders.get(sym, [])

            # Identify existing order types for this symbol
            has_trailing_stop = False
            trailing_stop_order_id = None
            has_limit_close = False
            for order in sym_orders:
                order_type = str(getattr(order, 'order_type', '')).lower().split('.')[-1]
                order_side = str(getattr(order, 'side', '')).lower().split('.')[-1]
                is_close_side = (direction > 0 and order_side == 'sell') or (direction < 0 and order_side == 'buy')
                if order_type == 'trailing_stop' and is_close_side:
                    has_trailing_stop = True
                    trailing_stop_order_id = str(order.id)
                elif order_type == 'limit' and is_close_side:
                    has_limit_close = True

            if group and group.state == GroupState.OPEN:
                # --- 3. Tracked position: verify direction + trailing stop ---
                if group.direction != direction:
                    logger.warning(
                        f"[RECONCILE] {sym} direction mismatch: tracker={group.direction}, "
                        f"actual={direction} — updating tracker"
                    )
                    group.direction = direction
                    self.tracker._save()

                if has_trailing_stop:
                    # Update tracker with the actual order ID from Alpaca
                    if trailing_stop_order_id and group.trailing_stop_id != trailing_stop_order_id:
                        old_id = group.trailing_stop_id
                        group.trailing_stop_id = trailing_stop_order_id
                        self.tracker._order_id_index.pop(old_id, None)
                        self.tracker._order_id_index[trailing_stop_order_id] = sym
                        self.tracker._save()
                        logger.info(f"[RECONCILE] {sym} updated trailing stop ID: {old_id} → {trailing_stop_order_id}")
                    logger.info(f"[RECONCILE] {sym} OK — tracked + trailing stop active")
                else:
                    # Trailing stop is missing — need to resubmit
                    logger.warning(f"[RECONCILE] {sym} tracked but trailing stop MISSING — will resubmit")
                    group.trailing_stop_id = None
                    self.tracker._save()
                    self._reconcile_submit_trailing_stop(sym, abs_qty, direction, group)

            elif group and group.state == GroupState.PENDING_ENTRY:
                # Entry was pending — check if it actually filled
                logger.info(f"[RECONCILE] {sym} has position but tracker is PENDING_ENTRY — marking as filled")
                current_price = self._get_current_price(sym)
                if not has_trailing_stop:
                    self._reconcile_submit_trailing_stop(sym, abs_qty, direction, group)
                else:
                    group.trailing_stop_id = trailing_stop_order_id
                    if trailing_stop_order_id:
                        self.tracker._order_id_index[trailing_stop_order_id] = sym
                group.state = GroupState.OPEN
                group.entry_price = current_price
                group.filled_qty = abs_qty
                group.filled_at = datetime.now(tz=_UTC).isoformat()
                self.tracker._save()

            else:
                # --- 4. Orphaned position: no tracker group at all ---
                logger.info(f"[RECONCILE] {sym} orphaned position ({abs_qty:.2f} {'LONG' if direction > 0 else 'SHORT'}) — creating tracker + protective orders")
                current_price = self._get_current_price(sym)
                regime = self.regime_cache.get(sym, {})
                if isinstance(regime, tuple):
                    regime_str, persistence = regime
                elif isinstance(regime, dict):
                    regime_str = regime.get('regime', 'mean_reverting')
                    persistence = regime.get('persistence', 0.5)
                else:
                    regime_str = str(regime) if regime else 'mean_reverting'
                    persistence = 0.5

                entry_id = f"reconcile_{sym}_{datetime.now(tz=_UTC).strftime('%Y%m%d%H%M%S')}"
                self.tracker.create_group(sym, direction, entry_id, regime=regime_str, persistence=persistence)

                if has_trailing_stop and trailing_stop_order_id:
                    # Already has a trailing stop on Alpaca — just record it
                    atr = self._reconcile_get_atr(sym)
                    tp_price = self._get_tp_price(current_price, atr, regime_str, direction)
                    trail_pct = 2.0  # estimate; will be corrected by ratchet
                    stop_est = round(current_price * (1 - direction * trail_pct / 100), 2)
                    self.tracker.mark_entry_filled(
                        symbol=sym, fill_price=current_price, filled_qty=abs_qty,
                        trailing_stop_id=trailing_stop_order_id, take_profit_id='',
                        trail_percent=trail_pct, tp_price=tp_price, stop_price=stop_est,
                    )
                    logger.info(f"[RECONCILE] {sym} linked existing trailing stop {trailing_stop_order_id}")
                else:
                    # No trailing stop — submit one now
                    group = self.tracker.groups[sym]
                    self._reconcile_submit_trailing_stop(sym, abs_qty, direction, group)

        logger.info(
            f"[RECONCILE] Complete — {len(self.existing_positions)} positions, "
            f"{len(self.tracker.get_open_groups())} tracked groups"
        )

    def _reconcile_submit_trailing_stop(self, symbol: str, qty: float, direction: int, group):
        """Submit a trailing stop during startup reconciliation (synchronous).
        NOTE: Alpaca does NOT support trailing stops for fractional shares — round to whole shares."""
        current_price = self._get_current_price(symbol)
        atr = self._reconcile_get_atr(symbol)
        regime = group.regime or 'mean_reverting'
        trail_pct = self._get_trail_percent(current_price, atr, regime)
        tp_price = self._get_tp_price(current_price, atr, regime, direction)
        close_side = OrderSide.SELL if direction > 0 else OrderSide.BUY
        stop_est = round(current_price * (1 - direction * trail_pct / 100), 2)

        # Alpaca rejects trailing stop orders for fractional shares — round down to whole shares
        # Remaining fractional qty will be protected by the software TP in monitor loop
        if qty != int(qty):
            whole_qty = int(qty)
            if whole_qty < 1:
                logger.warning(f"[RECONCILE] {symbol} qty={qty} is fractional with <1 whole share — "
                               f"trailing stop not possible, relying on software TP in monitor loop")
                # Still mark entry filled with no trailing stop so monitor loop can manage
                self.tracker.mark_entry_filled(
                    symbol=symbol, fill_price=current_price, filled_qty=qty,
                    trailing_stop_id='', take_profit_id='',
                    trail_percent=trail_pct, tp_price=tp_price, stop_price=stop_est,
                )
                return
            logger.info(f"[RECONCILE] {symbol} rounding qty {qty} → {whole_qty} whole shares for trailing stop")
            qty = whole_qty

        try:
            trail_req = TrailingStopOrderRequest(
                symbol=symbol, qty=qty, side=close_side,
                time_in_force=self._tif_for_qty(qty), trail_percent=trail_pct,
            )
            resp = self.client.submit_order(trail_req)
            trailing_stop_id = str(resp.id)
            logger.info(f"[RECONCILE] {symbol} trailing stop submitted @ {trail_pct}% | id={trailing_stop_id}")

            self.tracker.mark_entry_filled(
                symbol=symbol, fill_price=current_price, filled_qty=qty,
                trailing_stop_id=trailing_stop_id, take_profit_id='',
                trail_percent=trail_pct, tp_price=tp_price, stop_price=stop_est,
            )
        except Exception as e:
            logger.error(f"[RECONCILE] Failed to submit trailing stop for {symbol}: {e}")

    def _reconcile_get_atr(self, symbol: str) -> float:
        """Get ATR for reconciliation — best effort."""
        if self.data_ingestion:
            data = self.data_ingestion.get_latest_data(symbol)
            if data is not None and len(data) >= 14:
                return self._compute_current_atr(data)
        return 0.01

    def _get_current_price(self, symbol: str) -> float:
        """Get latest price for a symbol — best effort."""
        if self.data_ingestion:
            data = self.data_ingestion.get_latest_data(symbol)
            if data is not None and len(data) > 0:
                return float(data['close'].iloc[-1])
        # Fallback: use Alpaca position cost basis
        try:
            positions = self.client.get_all_positions()
            for p in positions:
                if p.symbol == symbol:
                    return float(p.current_price)
        except Exception:
            pass
        return 0.0

    def _load_regime_cache(self):
        if os.path.exists(REGIME_CACHE_FILE):
            try:
                with open(REGIME_CACHE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load regime cache: {e}")
        return {}

    def get_equity(self):
        try:
            account = self.client.get_account()
            equity = float(account.equity)
            self._last_known_equity = equity
            return equity
        except Exception as e:
            logger.error(f"Equity fetch error: {e} — returning last known equity {self._last_known_equity}")
            return self._last_known_equity

    def get_buying_power(self):
        try:
            account = self.client.get_account()
            return float(account.buying_power)
        except Exception as e:
            logger.error(f"Buying power fetch error: {e}")
            return 0.0

    def get_positions_dict(self):
        self.sync_existing_positions()
        return self.existing_positions.copy()

    # ====================== ATR COMPUTATION ======================
    def _compute_current_atr(self, data_window: pd.DataFrame, lookback: int = 50) -> float:
        if len(data_window) < 14:
            return 0.01
        recent = data_window.tail(lookback)
        high_low = recent['high'] - recent['low']
        high_close = (recent['high'] - recent['close'].shift(1)).abs()
        low_close = (recent['low'] - recent['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).dropna()
        if len(tr) == 0:
            return 0.01
        atr = tr.ewm(span=14, adjust=False).mean().iloc[-1]
        floor = 0.0005 * recent['close'].iloc[-1]
        return max(atr, floor)

    def _get_regime(self, symbol: str, data=None):
        """Get regime + persistence for a symbol from cache or fallback."""
        if symbol in self.regime_cache:
            value = self.regime_cache[symbol]
            if isinstance(value, (list, tuple)):
                regime = value[0]
                persistence = value[1] if len(value) >= 2 else 0.5
            else:
                regime = value
                persistence = 0.5
            return regime, persistence
        if data is not None and len(data) >= 20:
            recent_return = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
            regime = 'trending' if abs(recent_return) > 0.015 else 'mean_reverting'
            return regime, 0.5
        return 'mean_reverting', 0.5

    def _get_trail_percent(self, current_price: float, atr: float, regime: str) -> float:
        """Compute trailing stop as a percentage of price (for TrailingStopOrderRequest)."""
        trailing_mult = (
            self.config.get('TRAILING_STOP_ATR_TRENDING', self.config.get('TRAILING_STOP_ATR', 3.0) * 0.8)
            if regime == 'trending'
            else self.config.get('TRAILING_STOP_ATR_MEAN_REVERTING', self.config.get('TRAILING_STOP_ATR', 4.0))
        )
        trail_distance = atr * trailing_mult
        trail_pct = (trail_distance / current_price) * 100  # Alpaca expects percentage
        # Clamp: at least 0.5%, at most 35%
        return round(max(0.5, min(35.0, trail_pct)), 2)

    def _tif_for_qty(self, qty: float) -> TimeInForce:
        """Fractional orders MUST use DAY; whole-share orders use GTC."""
        if qty != int(qty):
            return TimeInForce.DAY
        return TimeInForce.GTC

    def _get_tp_price(self, current_price: float, atr: float, regime: str, direction: int) -> float:
        """Compute take-profit limit price."""
        tp_mult = (
            self.config.get('TAKE_PROFIT_ATR_TRENDING', self.config.get('TAKE_PROFIT_ATR', 30.0) * 1.2)
            if regime == 'trending'
            else self.config.get('TAKE_PROFIT_ATR_MEAN_REVERTING', self.config.get('TAKE_PROFIT_ATR', 12.0))
        )
        tp_price = current_price + direction * tp_mult * atr
        return round(max(tp_price, 0.01), 2)

    # ====================== DUPLICATE CHECK ======================
    def has_active_orders(self, symbol: str) -> bool:
        """Check if symbol already has open orders (entry or exit)."""
        with self.tracker._lock:
            group = self.tracker.groups.get(symbol)
            if group and group.state in (GroupState.PENDING_ENTRY, GroupState.OPEN):
                return True
        try:
            orders = self.client.get_orders(GetOrdersRequest(status='open', symbols=[symbol]))
            return len(orders) > 0
        except Exception as e:
            logger.warning(f"Failed to check active orders for {symbol}: {e}")
            return False

    # ====================== ENTRY: SIMPLE LIMIT ORDER ======================
    def place_bracket_order(self, symbol, size, current_price, data, direction=1):
        """Submit a limit entry order. Exit orders (trailing stop + TP) are placed
        on fill via the websocket handler in stream.py."""
        if self.has_active_orders(symbol):
            logger.info(f"Skipping entry for {symbol} — active orders or tracker group already exists")
            return None

        regime, persistence = self._get_regime(symbol, data)

        # Fractional shares: allow float qty, but enforce minimum notional
        # NOTE: Alpaca does NOT allow fractional shares for short sells
        if self.use_fractional and direction > 0:
            if size * current_price < 1.0:
                logger.info(f"Skipping {symbol} — notional ${size * current_price:.2f} below $1 minimum")
                return None
            qty = round(size, 4)  # Alpaca supports up to 9 decimal places for fractional
        else:
            size = int(size)
            if size < 1:
                logger.debug(f"Skipping {symbol} — qty rounds to 0 (size={size:.4f})")
                return None
            qty = size

        # Buying power safety cap
        available_bp = self.get_buying_power()
        notional = abs(qty) * current_price
        safety_factor = self.config.get('MAX_ORDER_NOTIONAL_PCT', 0.85)
        max_allowed = available_bp * safety_factor
        if notional > max_allowed and available_bp > 1000:
            if self.use_fractional:
                qty = round(max_allowed / current_price, 4)
            else:
                qty = int(max_allowed / current_price)
            if qty * current_price < 1.0:
                logger.warning(f"[BP CAP] {symbol}: even after cap, qty too small — skipping")
                return None
            logger.warning(
                f"[BP CAP] {symbol}: reduced to {qty} (${qty * current_price:,.0f} / "
                f"${available_bp:,.0f} BP, safety={safety_factor})"
            )

        side = OrderSide.BUY if direction > 0 else OrderSide.SELL
        limit_price = round(float(current_price) * (1 + direction * self.limit_price_offset), 2)
        position_intent = PositionIntent.BUY_TO_OPEN if direction > 0 else PositionIntent.SELL_TO_OPEN

        tif = self._tif_for_qty(qty)
        entry_request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=tif,
            limit_price=limit_price,
            extended_hours=self.use_extended_hours,
            position_intent=position_intent,
        )

        try:
            response = self.client.submit_order(entry_request)
            order_id = str(response.id)

            # Register in tracker — exit orders will be submitted on fill via websocket
            self.tracker.create_group(
                symbol=symbol,
                direction=direction,
                entry_order_id=order_id,
                regime=regime,
                persistence=persistence,
                extended_hours=self.use_extended_hours,
            )
            # Temporarily store limit_price as entry_price for slippage measurement
            with self.tracker._lock:
                group = self.tracker.groups.get(symbol)
                if group:
                    group.entry_price = limit_price
                    self.tracker._save()

            self.last_entry_times[symbol] = datetime.now(tz=_UTC)
            self._save_last_entry_times()
            self.sync_existing_positions(force_refresh=True)

            logger.info(
                f"Entry submitted: {'BUY' if direction > 0 else 'SELL'} {qty} {symbol} @ limit {limit_price} | "
                f"regime={regime} persist={persistence:.2f} | extended={self.use_extended_hours} | "
                f"exits on fill via websocket"
            )
            return response
        except Exception as e:
            err_str = str(e)
            if "40310100" in err_str or "pattern day trading" in err_str.lower():
                logger.warning(f"PDT protection blocked trade for {symbol}")
            elif "40310000" in err_str or "insufficient buying power" in err_str.lower():
                logger.error(f"Entry failed for {symbol}: insufficient buying power")
            else:
                logger.error(f"Entry failed for {symbol}: {e}")
            return None

    # ====================== EXIT ORDERS (called by stream.py on entry fill) ======================
    async def submit_exit_orders(self, symbol: str, group, fill_price: float, filled_qty: float):
        """Submit trailing stop + limit TP as two independent orders (manual OCO via websocket)."""
        direction = group.direction
        regime = group.regime

        # Fetch fresh data for ATR
        atr = 0.01
        if self.data_ingestion:
            data = self.data_ingestion.get_latest_data(symbol)
            if data is not None and len(data) >= 14:
                atr = self._compute_current_atr(data)

        trail_pct = self._get_trail_percent(fill_price, atr, regime)
        tp_price = self._get_tp_price(fill_price, atr, regime, direction)

        close_side = OrderSide.SELL if direction > 0 else OrderSide.BUY
        position_intent = PositionIntent.SELL_TO_CLOSE if direction > 0 else PositionIntent.BUY_TO_CLOSE

        # === 1. Trailing Stop (native Alpaca trailing) ===
        # NOTE: Alpaca does NOT support extended_hours or fractional shares on trailing stop orders
        trailing_stop_id = None
        trail_qty = filled_qty
        if trail_qty != int(trail_qty):
            trail_qty = int(trail_qty)
            if trail_qty < 1:
                logger.warning(f"[EXIT] {symbol} fractional qty {filled_qty} — no trailing stop possible, "
                               f"relying on software TP in monitor loop")
                trail_qty = 0
            else:
                logger.info(f"[EXIT] {symbol} rounding qty {filled_qty} → {trail_qty} for trailing stop")
        if trail_qty >= 1:
            try:
                trail_req = TrailingStopOrderRequest(
                    symbol=symbol,
                    qty=trail_qty,
                    side=close_side,
                    time_in_force=self._tif_for_qty(trail_qty),
                    trail_percent=trail_pct,
                    position_intent=position_intent,
                )
                trail_resp = await asyncio.to_thread(self.client.submit_order, trail_req)
                trailing_stop_id = str(trail_resp.id)
                logger.info(f"[EXIT] {symbol} trailing stop: {trail_pct}% trail | id={trailing_stop_id}")
            except Exception as e:
                logger.error(f"[EXIT] Failed to submit trailing stop for {symbol}: {e}")

        # === 2. Take-Profit ===
        # NOTE: Alpaca reserves all qty for the trailing stop, so a separate TP limit
        # order would fail with "insufficient qty". Instead, store the TP price in the
        # tracker and enforce it in the monitor loop by canceling the trailing stop
        # and closing the position when TP is hit.
        tp_order_id = ''
        logger.info(f"[EXIT] {symbol} take-profit target: ${tp_price:.2f} (enforced via monitor loop)")

        # Compute initial stop_price estimate for logging
        stop_price_est = round(fill_price * (1 - direction * trail_pct / 100), 2)

        # Update tracker
        if trailing_stop_id:
            self.tracker.mark_entry_filled(
                symbol=symbol,
                fill_price=fill_price,
                filled_qty=filled_qty,
                trailing_stop_id=trailing_stop_id,
                take_profit_id=tp_order_id,
                trail_percent=trail_pct,
                tp_price=tp_price,
                stop_price=stop_price_est,
            )

    # ====================== RATCHET VIA PATCH (ReplaceOrderRequest) ======================
    def ratchet_trailing_stop(self, symbol: str, current_price: float, atr: float):
        """Tighten trailing stop using ReplaceOrderRequest PATCH — no cancel/resubmit needed."""
        group = self.tracker.groups.get(symbol)
        if not group or group.state != GroupState.OPEN or not group.trailing_stop_id:
            return

        now = datetime.now(tz=_UTC)
        last_ratchet = self.last_ratchet_time.get(symbol, _EPOCH)
        elapsed = (now - last_ratchet).total_seconds()

        regime = group.regime
        persistence = group.persistence
        direction = group.direction
        entry_price = group.entry_price or current_price

        # Skip losing positions
        unrealized_pct = (current_price - entry_price) / entry_price * direction if entry_price else 0.0
        if unrealized_pct <= 0:
            return

        # Regime-adaptive throttle
        if regime == 'trending' and persistence >= 0.7:
            min_interval = self.config.get('RATCHET_TRENDING_INTERVAL_SEC', 180)
            regime_factor = self.config.get('RATCHET_REGIME_FACTOR_TRENDING', 0.65)
        else:
            min_interval = self.config.get('RATCHET_MEAN_REVERTING_INTERVAL_SEC', 540)
            regime_factor = self.config.get('RATCHET_REGIME_FACTOR_MEAN_REVERTING', 1.35)

        if elapsed < min_interval:
            return

        # Compute tighter trail percentage
        profit_protection = max(
            self.config.get('RATCHET_PROFIT_PROTECTION_MIN', 0.30),
            min(1.0, 1.0 - unrealized_pct * self.config.get('RATCHET_PROFIT_PROTECTION_SLOPE', 1.5))
        )
        trailing_mult = (
            self.config.get('TRAILING_STOP_ATR_TRENDING', self.config.get('TRAILING_STOP_ATR', 3.0) * 0.8)
            if regime == 'trending'
            else self.config.get('TRAILING_STOP_ATR_MEAN_REVERTING', self.config.get('TRAILING_STOP_ATR', 4.0))
        )
        distance = atr * trailing_mult * regime_factor * profit_protection
        new_trail_pct = round(max(0.5, min(35.0, (distance / current_price) * 100)), 2)

        # Only tighten (lower trail %), never widen
        old_trail = group.trail_percent or 99.0
        if new_trail_pct >= old_trail:
            return

        try:
            replace_req = ReplaceOrderRequest(trail=new_trail_pct)
            self.client.replace_order_by_id(group.trailing_stop_id, replace_req)
            self.last_ratchet_time[symbol] = now
            self.tracker.update_trail(symbol, new_trail_pct)
            logger.info(
                f"[RATCHET] {symbol} trail tightened {old_trail:.2f}% -> {new_trail_pct:.2f}% "
                f"(profit={unrealized_pct*100:+.1f}%, regime={regime})"
            )
        except Exception as e:
            logger.warning(f"[RATCHET] Failed PATCH for {symbol}: {e} — will retry next cycle")

    # ====================== MONITOR LOOP ======================
    async def monitor_positions(self):
        """Heartbeat loop: ratchet trailing stops, re-attach missing exits, slippage adaptation."""
        logger.info("=== MONITOR TASK STARTED — EVENT-DRIVEN ARCHITECTURE ===")
        while True:
            await asyncio.to_thread(self.sync_existing_positions)

            pos_count = len(self.existing_positions)
            tracked = len(self.tracker.get_open_groups())
            heartbeat_msg = (
                f"[{datetime.now().strftime('%H:%M:%S')}] Monitor — "
                f"{pos_count} positions | {tracked} tracked groups | "
                f"market {'open' if is_market_open() else 'closed'}"
            )
            print(heartbeat_msg)
            logger.info(heartbeat_msg)

            # === Per-position logic ===
            try:
                for sym, qty in list(self.existing_positions.items()):
                    if qty == 0 or self.data_ingestion is None:
                        continue
                    data = self.data_ingestion.get_latest_data(sym)
                    if data is None or len(data) < 50:
                        continue

                    direction = 1 if qty > 0 else -1
                    current_price = float(data['close'].iloc[-1])
                    atr = self._compute_current_atr(data)

                    # Log position status
                    last_entry = self.last_entry_times.get(sym)
                    bars_held = ((datetime.now(tz=_UTC) - last_entry) / timedelta(minutes=15)) if last_entry else 9999
                    regime, persistence = self._get_regime(sym, data)
                    min_hold = (
                        self.config.get('MIN_HOLD_BARS_TRENDING', 6)
                        if regime == 'trending'
                        else self.config.get('MIN_HOLD_BARS_MEAN_REVERTING', 3)
                    )
                    logger.info(
                        f"  {sym} | {'LONG' if direction > 0 else 'SHORT'} {abs(qty):.2f} | "
                        f"held {int(bars_held)} bars (min {min_hold}) | regime={regime}"
                    )

                    # === Ratchet trailing stop via PATCH ===
                    if is_market_open():
                        await asyncio.to_thread(self.ratchet_trailing_stop, sym, current_price, atr)

                    # === Software TP check (trailing stop holds all qty, TP enforced here) ===
                    group = self.tracker.groups.get(sym)
                    if group and group.state == GroupState.OPEN and group.tp_price:
                        tp_hit = (direction > 0 and current_price >= group.tp_price) or \
                                 (direction < 0 and current_price <= group.tp_price)
                        if tp_hit:
                            logger.info(f"[TP HIT] {sym} @ {current_price:.2f} >= TP {group.tp_price:.2f} — closing position")
                            # Cancel trailing stop first
                            if group.trailing_stop_id:
                                try:
                                    await asyncio.to_thread(self.client.cancel_order_by_id, group.trailing_stop_id)
                                except Exception as e:
                                    logger.warning(f"[TP HIT] Failed to cancel trailing stop for {sym}: {e}")
                            # Close position
                            try:
                                await asyncio.to_thread(self.client.close_position, sym)
                                logger.info(f"[TP HIT] {sym} position closed")
                                self.tracker.mark_closed(sym)
                                self.tracker.remove_group(sym)
                                self.last_entry_times.pop(sym, None)
                                self._save_last_entry_times()
                            except Exception as e:
                                logger.error(f"[TP HIT] Failed to close {sym}: {e}")
                                # Race condition: position may have been closed by trailing stop fill
                                err_str = str(e).lower()
                                if 'position does not exist' in err_str or '40410000' in str(e):
                                    logger.info(f"[TP HIT] {sym} position already closed — cleaning up tracker")
                                    self.tracker.mark_closed(sym)
                                    self.tracker.remove_group(sym)
                                    self.last_entry_times.pop(sym, None)
                                    self._save_last_entry_times()
                            continue

                    # === Re-attach missing exits for untracked positions ===
                    group = self.tracker.groups.get(sym)
                    if not group or group.state == GroupState.CLOSED:
                        # Position exists but no tracker group — re-attach protective orders
                        if is_market_open() and bars_held >= min_hold:
                            await self._reattach_exits(sym, qty, direction, current_price, atr, regime)

            except Exception as e:
                logger.error(f"Monitor per-position error: {e}", exc_info=True)

            # === Slippage adaptation from recent fills ===
            await asyncio.to_thread(self._adapt_slippage)

            await asyncio.sleep(self.config.get('MONITOR_INTERVAL', 60))

    async def _reattach_exits(self, symbol: str, qty: float, direction: int,
                              current_price: float, atr: float, regime: str):
        """Re-attach trailing stop + TP for orphaned positions (no tracker group)."""
        logger.info(f"[REATTACH] Creating protective orders for orphaned position {symbol}")

        trail_pct = self._get_trail_percent(current_price, atr, regime)
        tp_price = self._get_tp_price(current_price, atr, regime, direction)
        close_side = OrderSide.SELL if direction > 0 else OrderSide.BUY
        position_intent = PositionIntent.SELL_TO_CLOSE if direction > 0 else PositionIntent.BUY_TO_CLOSE
        abs_qty = abs(qty)

        trailing_stop_id = None
        tp_order_id = None

        # NOTE: Alpaca does NOT support extended_hours or fractional shares on trailing stop orders
        # Omit position_intent on reattach — let Alpaca infer from existing position
        trail_qty = abs_qty
        if trail_qty != int(trail_qty):
            trail_qty = int(trail_qty)
            if trail_qty < 1:
                logger.warning(f"[REATTACH] {symbol} fractional qty {abs_qty} — no trailing stop, "
                               f"relying on software TP in monitor loop")
            else:
                logger.info(f"[REATTACH] {symbol} rounding qty {abs_qty} → {trail_qty} for trailing stop")
        if trail_qty >= 1:
            try:
                trail_req = TrailingStopOrderRequest(
                    symbol=symbol, qty=trail_qty, side=close_side,
                    time_in_force=self._tif_for_qty(trail_qty), trail_percent=trail_pct,
                )
                resp = await asyncio.to_thread(self.client.submit_order, trail_req)
                trailing_stop_id = str(resp.id)
                logger.info(f"[REATTACH] {symbol} trailing stop @ {trail_pct}%")
            except Exception as e:
                logger.error(f"[REATTACH] Failed trailing stop for {symbol}: {e}")

        # TP enforced via monitor loop (Alpaca can't hold two closing orders)
        logger.info(f"[REATTACH] {symbol} take-profit target: ${tp_price:.2f} (enforced via monitor)")

        # Create a recovery tracker group
        if trailing_stop_id:
            entry_id = f"reattach_{symbol}_{datetime.now(tz=_UTC).strftime('%Y%m%d%H%M%S')}"
            self.tracker.create_group(symbol, direction, entry_id, regime=regime)
            stop_est = round(current_price * (1 - direction * trail_pct / 100), 2)
            self.tracker.mark_entry_filled(
                symbol=symbol, fill_price=current_price, filled_qty=abs_qty,
                trailing_stop_id=trailing_stop_id, take_profit_id='',
                trail_percent=trail_pct, tp_price=tp_price, stop_price=stop_est,
            )

    def _adapt_slippage(self):
        """Adapt limit_price_offset based on recent fill slippage."""
        try:
            recent_closed = self.client.get_orders(GetOrdersRequest(
                status='closed',
                after=datetime.now(tz=_UTC) - timedelta(hours=1)
            ))
        except Exception as e:
            logger.warning(f"Failed to fetch recent orders for slippage adaptation: {e}")
            return
        slippage_total = 0.0
        fill_count = 0
        for o in recent_closed:
            if o.filled_at and o.filled_avg_price and o.limit_price:
                filled_at_utc = o.filled_at.astimezone(_UTC) if o.filled_at.tzinfo else o.filled_at.replace(tzinfo=_UTC)
                if (datetime.now(tz=_UTC) - filled_at_utc).total_seconds() < 3600:
                    slippage = abs(float(o.filled_avg_price) - float(o.limit_price)) / float(o.limit_price)
                    slippage_total += slippage
                    fill_count += 1
        if fill_count > 0:
            avg_slippage = slippage_total / fill_count
            new_offset = max(0.001, avg_slippage * 2 + 0.001)
            if abs(new_offset - self.limit_price_offset) > 0.0005:
                old = self.limit_price_offset
                self.limit_price_offset = new_offset
                logger.info(f"Slippage adaptation: offset {old:.4f} -> {new_offset:.4f} (avg={avg_slippage:.4f})")

    # ====================== SAFE POSITION CLOSE ======================
    async def close_position_safely(self, symbol: str) -> bool:
        """Cancel all open orders for symbol, then close position."""
        try:
            all_open = await asyncio.to_thread(
                self.client.get_orders, GetOrdersRequest(status='open', symbols=[symbol]))
            cancelled_ids = []
            for o in all_open:
                try:
                    await asyncio.to_thread(self.client.cancel_order_by_id, o.id)
                    cancelled_ids.append(o.id)
                    logger.info(f"[CLOSE PREP] Cancelled {o.id} ({_order_type_str(o)}) for {symbol}")
                except Exception as cancel_e:
                    logger.warning(f"Failed to cancel order {o.id} for {symbol}: {cancel_e}")

            if cancelled_ids:
                for attempt in range(5):
                    await asyncio.sleep(1.0)
                    still_open = await asyncio.to_thread(
                        self.client.get_orders, GetOrdersRequest(status='open', symbols=[symbol]))
                    if not still_open:
                        break
                else:
                    logger.warning(f"[CLOSE PREP] Orders may not be fully cancelled for {symbol}")

            await asyncio.to_thread(self.client.close_position, symbol)
            logger.info(f"Closed position in {symbol}")

            # Clean up tracker and entry times
            self.tracker.remove_group(symbol)
            self.last_entry_times.pop(symbol, None)
            self._save_last_entry_times()
            self.sync_existing_positions(force_refresh=True)
            return True
        except Exception as e:
            logger.error(f"Failed to close position in {symbol}: {e}")
            return False

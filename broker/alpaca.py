# broker/alpaca.py
# Event-driven broker: trailing stops, ReplaceOrderRequest PATCH, fractional shares, extended hours
import logging
import asyncio
import threading
from typing import Optional, Dict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz
import json
import os
import tempfile
import shutil

from utils.helpers import is_market_open
from strategy.regime import is_trending
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
        self._ratchet_pending = set()  # Symbols with in-flight replace — skip until websocket confirms
        self._tp_in_progress = set()  # Symbols with TP enforcement in progress — skip ratchet
        self._sets_lock = threading.Lock()  # Thread safety for _ratchet_pending / _tp_in_progress
        self._positions_lock = threading.Lock()  # Thread safety for sync_existing_positions
        self._entry_times_lock = threading.Lock()  # FIX #11: Thread safety for last_entry_times access
        self._symbol_locks = {}  # Per-symbol threading locks to prevent TOCTOU races between monitor and stream
        self._symbol_locks_lock = threading.Lock()  # Protects lazy creation of per-symbol locks
        # HTB (hard-to-borrow) symbols: GTC orders are rejected by Alpaca on these;
        # we discover them dynamically by inspecting rejection reasons and then
        # fall back to DAY TIF on future submissions for the same symbol.
        self._htb_symbols = set(config.get('HTB_SYMBOLS', ['PLTR']))
        self._htb_lock = threading.Lock()
        # Apr-19 FIX: symbols with a fractional remainder after a partial exit
        # fill need to be actively swept on the next monitor cycle — native
        # trailing stops don't cover fractional qty.
        self._pending_fractional_close: Dict[str, float] = {}
        self._fractional_close_lock = threading.Lock()
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

    def _record_close(self, symbol: str, direction: int, entry_price: float,
                      exit_price: float, filled_qty: float, regime: str,
                      exit_reason: str, filled_at: str = None):
        """Record a trade closure in the autopsy engine and clear velocity tracker.
        Called from ALL close paths (stream fill, software SL, reattach block, frac cleanup)."""
        if not hasattr(self, 'signal_gen') or not self.signal_gen:
            return
        pnl = (exit_price - entry_price) * filled_qty * direction
        try:
            from models.features import _fetch_macro_features
            vix = _fetch_macro_features().get('vix_close', 20)
        except Exception:
            vix = 20
        bars_held = 0
        if filled_at:
            try:
                entry_time = datetime.fromisoformat(filled_at)
                bars_held = int((datetime.now(tz=_UTC) - entry_time).total_seconds() / 900)
            except Exception:
                pass
        self.signal_gen.autopsy_engine.record_autopsy(
            symbol=symbol, direction=direction,
            entry_price=entry_price, exit_price=exit_price,
            pnl=pnl, bars_held=bars_held,
            regime=regime, vix=vix, exit_reason=exit_reason,
        )
        self.signal_gen.profit_velocity.clear(symbol)

    def _get_symbol_lock(self, symbol: str) -> threading.Lock:
        """Lazily create and return a per-symbol threading.Lock to prevent TOCTOU races.
        FIX #18: Changed from asyncio.Lock to threading.Lock so stream handler (different thread)
        actually contends with the monitor loop. Critical sections are short so blocking is fine."""
        with self._symbol_locks_lock:
            if symbol not in self._symbol_locks:
                self._symbol_locks[symbol] = threading.Lock()
            return self._symbol_locks[symbol]

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
        """HIGH-15 FIX: Snapshot under _entry_times_lock to prevent RuntimeError
        from concurrent dict modification by another thread."""
        try:
            with self._entry_times_lock:
                data = {sym: ts.isoformat() for sym, ts in self.last_entry_times.items()}
            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(LAST_ENTRY_FILE) or '.', suffix='.tmp')
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
            shutil.move(tmp_path, LAST_ENTRY_FILE)
        except Exception as e:
            logger.warning(f"Failed to save last_entry_times.json: {e}")

    def _save_last_entry_times_unlocked(self):
        """Save variant for callers that already hold _entry_times_lock (avoids deadlock)."""
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
            # FIX #10: Don't set cache time before API call — if the call fails,
            # stale cache_time prevents retries. Set it AFTER success only.
        try:
            positions = self.client.get_all_positions()
            new_positions = {
                p.symbol: float(p.qty) if p.side == PositionSide.LONG else -float(p.qty)
                for p in positions if float(p.qty) != 0
            }
            with self._positions_lock:
                self.existing_positions = new_positions
                self._positions_cache = new_positions.copy()
                self._positions_cache_time = datetime.now(tz=_UTC)  # Set AFTER successful fetch
            with self._entry_times_lock:
                for sym in list(new_positions.keys()):
                    if sym not in self.last_entry_times:
                        self.last_entry_times[sym] = now - timedelta(minutes=30)
                for sym in list(self.last_entry_times.keys()):
                    if sym not in new_positions:
                        self.last_entry_times.pop(sym, None)
                # HIGH-15: Use unlocked variant since we already hold _entry_times_lock
                self._save_last_entry_times_unlocked()
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
        # CRIT-4 FIX: Acquire tracker lock for each group access to prevent race with concurrent stream handler
        with self.tracker._lock:
            groups_snapshot = {sym: self.tracker.groups[sym] for sym in self.tracker.groups}
        # M24 FIX: Batch stale group removal — single _save() instead of N
        stale_syms = []
        for sym, group in groups_snapshot.items():
            if sym not in self.existing_positions:
                if group.state == GroupState.PENDING_ENTRY:
                    if group.entry_order_id:
                        try:
                            self.client.cancel_order_by_id(group.entry_order_id)
                            logger.info(f"[RECONCILE] Canceled stale entry order for {sym}")
                        except Exception:
                            pass
                elif group.state == GroupState.OPEN:
                    logger.info(f"[RECONCILE] {sym} was OPEN in tracker but no position — stop likely filled while offline")
                elif group.state == GroupState.PENDING_EXIT:
                    logger.info(f"[RECONCILE] {sym} was PENDING_EXIT — exit completed while offline")
                stale_syms.append(sym)
        if stale_syms:
            with self.tracker._lock:
                for sym in stale_syms:
                    group = self.tracker.groups.pop(sym, None)
                    if group:
                        for oid in (group.entry_order_id, group.trailing_stop_id, group.take_profit_id):
                            if oid:
                                self.tracker._order_id_index.pop(oid, None)
                        logger.info(f"[RECONCILE] Removed stale tracker group for {sym}")
                self.tracker._save()  # Single write for all stale groups

        # --- 1b. Cancel orphaned closing orders for symbols with no position ---
        # Only cancel protective/closing orders (trailing stop, stop, stop_limit).
        # Preserve pending entry limit orders — they may fill and create a new position.
        for sym, orders in open_orders.items():
            if sym not in self.existing_positions:
                for order in orders:
                    order_type = str(getattr(order, 'order_type', '')).lower().split('.')[-1]
                    order_id_str = str(order.id)
                    # Always cancel stop-type orders (these are closing orders)
                    is_closing_order = order_type in ('trailing_stop', 'stop', 'stop_limit')
                    # For limit orders, only cancel if NOT a tracked pending entry
                    if order_type == 'limit':
                        tracked_group = self.tracker.lookup_by_order_id(order_id_str)
                        if tracked_group and order_id_str == tracked_group.entry_order_id:
                            logger.debug(f"[RECONCILE] Preserving pending entry limit order for {sym}: {order_id_str}")
                            continue  # Skip — this is a pending entry, not a closing order
                        is_closing_order = True  # Untracked limit order with no position — orphaned
                    if is_closing_order:
                        try:
                            self.client.cancel_order_by_id(order_id_str)
                            logger.info(f"[RECONCILE] Cancelled orphaned {order_type} order for {sym} "
                                        f"(no position exists) — id={order.id}")
                        except Exception as e:
                            logger.warning(f"[RECONCILE] Failed to cancel orphaned order for {sym}: {e}")

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
                with self.tracker._lock:
                    if group.direction != direction:
                        logger.warning(
                            f"[RECONCILE] {sym} direction mismatch: tracker={group.direction}, "
                            f"actual={direction} — updating tracker"
                        )
                        group.direction = direction
                        self.tracker._save()

                    if has_trailing_stop:
                        if trailing_stop_order_id and group.trailing_stop_id != trailing_stop_order_id:
                            old_id = group.trailing_stop_id
                            group.trailing_stop_id = trailing_stop_order_id
                            self.tracker._order_id_index.pop(old_id, None)
                            self.tracker._order_id_index[trailing_stop_order_id] = sym
                            self.tracker._save()
                            logger.info(f"[RECONCILE] {sym} updated trailing stop ID: {old_id} → {trailing_stop_order_id}")
                        logger.info(f"[RECONCILE] {sym} OK — tracked + trailing stop active")
                    else:
                        logger.warning(f"[RECONCILE] {sym} tracked but trailing stop MISSING — will resubmit")
                        group.trailing_stop_id = None
                        self.tracker._save()
                if not has_trailing_stop:
                    self._reconcile_submit_trailing_stop(sym, abs_qty, direction, group)

            elif group and group.state == GroupState.PENDING_ENTRY:
                # Entry was pending — check if it actually filled
                logger.info(f"[RECONCILE] {sym} has position but tracker is PENDING_ENTRY — marking as filled")
                current_price = self._get_current_price(sym)
                # FIX #16: Submit trailing stop OUTSIDE lock to avoid blocking tracker
                # during the Alpaca API call. Update tracker state after.
                if not has_trailing_stop:
                    self._reconcile_submit_trailing_stop(sym, abs_qty, direction, group)
                with self.tracker._lock:
                    if has_trailing_stop:
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
        trail_pct = self._get_trail_percent(current_price, atr, regime, direction=direction)
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
                # FIX #19: Update fields directly if group is already OPEN (mark_entry_filled rejects OPEN groups)
                from broker.order_tracker import GroupState
                with self.tracker._lock:
                    grp = self.tracker.groups.get(symbol)
                    if grp and grp.state == GroupState.OPEN:
                        grp.trail_percent = trail_pct
                        grp.tp_price = tp_price
                        grp.stop_price = stop_est
                        grp.entry_price = grp.entry_price or current_price
                        grp.filled_qty = grp.filled_qty or qty
                        self.tracker._save()
                    else:
                        self.tracker.mark_entry_filled(
                            symbol=symbol, fill_price=current_price, filled_qty=qty,
                            trailing_stop_id='', take_profit_id='',
                            trail_percent=trail_pct, tp_price=tp_price, stop_price=stop_est,
                        )
                return
            logger.info(f"[RECONCILE] {symbol} rounding qty {qty} → {whole_qty} whole shares for trailing stop")
            qty = whole_qty

        try:
            resp = self._submit_trailing_stop_with_htb_fallback(
                symbol=symbol, qty=qty, side=close_side, trail_percent=trail_pct,
            )
            trailing_stop_id = str(resp.id)
            logger.info(f"[RECONCILE] {symbol} trailing stop submitted @ {trail_pct}% | id={trailing_stop_id}")

            # FIX #19: mark_entry_filled rejects OPEN groups (state guard).
            # During reconciliation the group may already be OPEN, so update fields directly.
            from broker.order_tracker import GroupState
            with self.tracker._lock:
                grp = self.tracker.groups.get(symbol)
                if grp and grp.state == GroupState.OPEN:
                    grp.trailing_stop_id = trailing_stop_id
                    grp.trail_percent = trail_pct
                    grp.tp_price = tp_price
                    grp.stop_price = stop_est
                    grp.entry_price = grp.entry_price or current_price
                    grp.filled_qty = grp.filled_qty or qty
                    self.tracker._order_id_index[trailing_stop_id] = symbol
                    self.tracker._save()
                else:
                    # Group is still PENDING_ENTRY — normal path
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
        """Get latest price for a symbol — best effort.
        HIGH-9 FIX: Never return 0.0 — raises ValueError instead to prevent division by zero."""
        if self.data_ingestion:
            data = self.data_ingestion.get_latest_data(symbol)
            if data is not None and len(data) > 0:
                price = float(data['close'].iloc[-1])
                if price > 0:
                    return price
        # Fallback: use Alpaca position cost basis (single symbol fetch, not all positions)
        try:
            pos = self.client.get_open_position(symbol)
            if pos:
                price = float(pos.current_price)
                if price > 0:
                    return price
        except Exception:
            pass
        # M23 FIX: Return 0.0 instead of raising — callers in reconciliation don't catch ValueError
        logger.error(f"[PRICE] Cannot determine current price for {symbol} — no valid data source")
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
        # FIX: Return copy inside lock to prevent torn reads from concurrent writes
        with self._positions_lock:
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
            regime = ('trending_up' if recent_return >= 0 else 'trending_down') if abs(recent_return) > 0.015 else 'mean_reverting'
            return regime, 0.5
        return 'mean_reverting', 0.5

    def _rex_alignment_mults(self, regime: str, direction: int) -> tuple[float, float]:
        """REX (Regime-Conditional Exits): compute (tp_mult, trail_mult) applied
        on top of the regime-base ATR multipliers, based on whether the trade
        ALIGNS with or OPPOSES the current regime direction.

        Align with trend (long in uptrend, short in downtrend):
            → wider TP, wider trail (let winners run in momentum regime)
        Oppose the trend (counter-trend bet):
            → tighter TP, tighter trail (take quick profits, accept quick exit)
        Mean-reverting regime:
            → tighter TP (expect reversal), slightly tighter trail

        Fully configurable via REX_* config keys. Direction=0 means caller
        doesn't know direction yet → returns (1.0, 1.0) neutral."""
        if direction == 0 or not self.config.get('REX_ENABLED', True):
            return 1.0, 1.0
        if regime == 'trending_up':
            if direction > 0:
                return (self.config.get('REX_ALIGN_TP_MULT', 1.4),
                        self.config.get('REX_ALIGN_TRAIL_MULT', 1.25))
            else:
                return (self.config.get('REX_OPPOSE_TP_MULT', 0.7),
                        self.config.get('REX_OPPOSE_TRAIL_MULT', 0.75))
        if regime == 'trending_down':
            if direction < 0:
                return (self.config.get('REX_ALIGN_TP_MULT', 1.4),
                        self.config.get('REX_ALIGN_TRAIL_MULT', 1.25))
            else:
                return (self.config.get('REX_OPPOSE_TP_MULT', 0.7),
                        self.config.get('REX_OPPOSE_TRAIL_MULT', 0.75))
        # mean_reverting: take profits fast, modest trail tightening
        return (self.config.get('REX_MR_TP_MULT', 0.85),
                self.config.get('REX_MR_TRAIL_MULT', 0.92))

    def _get_trail_percent(self, current_price: float, atr: float, regime: str,
                           direction: int = 0) -> float:
        """Compute trailing stop as a percentage of price (for TrailingStopOrderRequest).
        REX: when direction is provided, applies regime-alignment multiplier so trades
        aligned with the trend get wider trails (let winners run), counter-trend or
        mean-reverting trades get tighter trails."""
        trailing_mult = (
            self.config.get('TRAILING_STOP_ATR_TRENDING', self.config.get('TRAILING_STOP_ATR', 3.0) * 0.8)
            if is_trending(regime)
            else self.config.get('TRAILING_STOP_ATR_MEAN_REVERTING', self.config.get('TRAILING_STOP_ATR', 4.0))
        )
        _tp_align, trail_align = self._rex_alignment_mults(regime, direction)
        trail_distance = atr * trailing_mult * trail_align
        trail_pct = (trail_distance / current_price) * 100  # Alpaca expects percentage
        # Clamp: at least 0.5%, at most 35%
        return round(max(0.5, min(35.0, trail_pct)), 2)

    def _tif_for_qty(self, qty: float, symbol: Optional[str] = None) -> TimeInForce:
        """Fractional qty → DAY. Hard-to-borrow symbols → DAY (Alpaca rejects
        GTC on HTB). Whole shares on normal symbols → GTC."""
        if qty != int(qty):
            return TimeInForce.DAY
        if symbol is not None:
            with self._htb_lock:
                if symbol in self._htb_symbols:
                    return TimeInForce.DAY
        return TimeInForce.GTC

    def _is_htb_rejection(self, err: Exception) -> bool:
        msg = str(err).lower()
        return ("hard-to-borrow" in msg or "hard to borrow" in msg
                or "only day orders are allowed" in msg)

    def _mark_htb(self, symbol: str) -> None:
        with self._htb_lock:
            newly_added = symbol not in self._htb_symbols
            self._htb_symbols.add(symbol)
        if newly_added:
            logger.warning(f"[HTB] {symbol} flagged hard-to-borrow — future trailing stops will use DAY TIF")

    def _submit_trailing_stop_with_htb_fallback(self, symbol: str, qty: float,
                                                 side: OrderSide, trail_percent: float,
                                                 position_intent=None):
        """Submit a trailing stop; on hard-to-borrow rejection, retry with DAY TIF
        and remember the symbol so subsequent submissions skip the GTC attempt."""
        tif = self._tif_for_qty(qty, symbol)
        kwargs = dict(symbol=symbol, qty=qty, side=side,
                      time_in_force=tif, trail_percent=trail_percent)
        if position_intent is not None:
            kwargs['position_intent'] = position_intent
        try:
            return self.client.submit_order(TrailingStopOrderRequest(**kwargs))
        except Exception as e:
            if tif == TimeInForce.GTC and self._is_htb_rejection(e):
                self._mark_htb(symbol)
                kwargs['time_in_force'] = TimeInForce.DAY
                return self.client.submit_order(TrailingStopOrderRequest(**kwargs))
            raise

    def _get_tp_price(self, current_price: float, atr: float, regime: str, direction: int) -> float:
        """Compute take-profit limit price.
        HIGH-7 FIX: Clamp TP distance to at most 50% of current price to prevent near-zero
        short TPs when ATR multiplier is high (e.g. trending regime with 30x ATR).
        REX: applies regime-alignment multiplier (aligned trades get farther TP for
        more upside; counter-trend / mean-reverting get tighter TP for faster profit-taking)."""
        tp_mult_regime = (
            self.config.get('TAKE_PROFIT_ATR_TRENDING', self.config.get('TAKE_PROFIT_ATR', 30.0) * 1.2)
            if is_trending(regime)
            else self.config.get('TAKE_PROFIT_ATR_MEAN_REVERTING', self.config.get('TAKE_PROFIT_ATR', 12.0))
        )
        tp_align, _trail_align = self._rex_alignment_mults(regime, direction)
        tp_distance = tp_mult_regime * atr * tp_align
        # HIGH-7 FIX: Cap TP distance at 50% of price to avoid near-zero short TPs
        max_tp_distance = current_price * 0.50
        tp_distance = min(tp_distance, max_tp_distance)
        tp_price = current_price + direction * tp_distance
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
        # Apr-19: Alpaca also rejects trailing stops on fractional qty. When the
        # requested size is ≥ 1 share, round DOWN to whole shares so the native
        # trailing stop covers the full position. Only keep fractional when size
        # is strictly < 1 (those positions fall back to the software TP monitor).
        if self.use_fractional and direction > 0:
            if size * current_price < 1.0:
                logger.info(f"Skipping {symbol} — notional ${size * current_price:.2f} below $1 minimum")
                return None
            if self.config.get('PREFER_WHOLE_SHARES_FOR_TS', True) and size >= 1.0:
                qty = float(int(size))
            else:
                qty = round(size, 4)
        else:
            size = int(size)
            if size < 1:
                logger.debug(f"Skipping {symbol} — qty rounds to 0 (size={size})")  # L17 FIX: size is int
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

            with self._entry_times_lock:
                self.last_entry_times[symbol] = datetime.now(tz=_UTC)
                self._save_last_entry_times_unlocked()  # C2 FIX: was _save_last_entry_times() → deadlock (re-acquires same Lock)
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
        # Per-symbol lock prevents TOCTOU races with monitor loop
        # FIX #18: threading.Lock — acquire in thread to avoid blocking event loop
        lock = self._get_symbol_lock(symbol)
        await asyncio.to_thread(lock.acquire)
        try:
            await self._submit_exit_orders_inner(symbol, group, fill_price, filled_qty)
        finally:
            lock.release()

    async def _submit_exit_orders_inner(self, symbol: str, group, fill_price: float, filled_qty: float):
        """Inner implementation of submit_exit_orders (called under per-symbol lock)."""
        direction = group.direction
        regime = group.regime

        # Fetch fresh data for ATR
        atr = 0.01
        if self.data_ingestion:
            data = self.data_ingestion.get_latest_data(symbol)
            if data is not None and len(data) >= 14:
                atr = self._compute_current_atr(data)

        trail_pct = self._get_trail_percent(fill_price, atr, regime, direction=direction)
        tp_price = self._get_tp_price(fill_price, atr, regime, direction)

        close_side = OrderSide.SELL if direction > 0 else OrderSide.BUY
        position_intent = PositionIntent.SELL_TO_CLOSE if direction > 0 else PositionIntent.BUY_TO_CLOSE

        # === 1. Trailing Stop (native Alpaca trailing) ===
        # NOTE: Alpaca does NOT support fractional shares on trailing stop orders
        trailing_stop_id = None
        trail_qty = filled_qty
        if trail_qty != int(trail_qty):
            whole_qty = int(trail_qty)
            fractional_remainder = round(trail_qty - whole_qty, 4)
            if whole_qty < 1:
                logger.warning(f"[EXIT] {symbol} fractional qty {filled_qty} — no trailing stop possible, "
                               f"relying on software TP/SL in monitor loop for ALL {filled_qty} shares")
                trail_qty = 0
            else:
                logger.info(f"[EXIT] {symbol} trailing stop covers {whole_qty}/{filled_qty} shares "
                            f"({fractional_remainder} fractional remainder protected by software monitor)")
                trail_qty = whole_qty
        if trail_qty >= 1:
            try:
                trail_resp = await asyncio.to_thread(
                    self._submit_trailing_stop_with_htb_fallback,
                    symbol, trail_qty, close_side, trail_pct, position_intent,
                )
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

        # Update tracker — CRIT-3 FIX: Always mark entry filled even if trailing stop submission
        # failed, so the group transitions to OPEN and the monitor loop can re-attach exits.
        # Without this, the group stays PENDING_ENTRY forever and the position is unprotected.
        self.tracker.mark_entry_filled(
            symbol=symbol,
            fill_price=fill_price,
            filled_qty=filled_qty,
            trailing_stop_id=trailing_stop_id or '',
            take_profit_id=tp_order_id,
            trail_percent=trail_pct,
            tp_price=tp_price,
            stop_price=stop_price_est,
        )
        if not trailing_stop_id:
            logger.warning(f"[EXIT] {symbol} entry marked filled but trailing stop MISSING — "
                           f"monitor loop must re-attach protection")

    # ====================== RATCHET VIA PATCH (ReplaceOrderRequest) ======================
    def ratchet_trailing_stop(self, symbol: str, current_price: float, atr: float):
        """Tighten trailing stop using ReplaceOrderRequest PATCH — no cancel/resubmit needed.
        NOTE: The check-then-act on group state + PATCH is not fully atomic. A brief window
        exists where the group could change between the read and the PATCH. This is acceptable
        because: (1) _ratchet_pending prevents concurrent PATCHes, (2) Alpaca rejects PATCHes
        on orders that no longer exist, and (3) the stream handler updates IDs atomically."""
        # Skip ratchet if TP enforcement is canceling this trailing stop or replace already in-flight
        with self._sets_lock:
            if symbol in self._tp_in_progress:
                return
            if symbol in self._ratchet_pending:
                logger.debug(f"[RATCHET] {symbol} skipped — replace PATCH already in-flight")
                return
        # Read all group fields under lock to prevent stale reads from concurrent stream handler.
        # Apr-19: MFE/MAE updates moved to the top of _monitor_one_position so that
        # TP-block / cooldown paths also record peak excursion. Keep original_trail
        # initialization here (it's cheap and needs to run on the first ratchet).
        with self.tracker._lock:
            group = self.tracker.groups.get(symbol)
            if not group or group.state != GroupState.OPEN or not group.trailing_stop_id:
                return
            regime = group.regime
            persistence = group.persistence
            direction = group.direction
            entry_price = group.entry_price or current_price
            trailing_stop_id = group.trailing_stop_id
            old_trail = group.trail_percent or 99.0
            live_unrealized = (current_price - entry_price) / entry_price * direction if entry_price else 0.0
            if group.original_trail_percent is None and group.trail_percent:
                group.original_trail_percent = float(group.trail_percent)
            mfe = group.max_favorable_pct
            mae = group.max_adverse_pct
            original_trail = group.original_trail_percent or old_trail
        now = datetime.now(tz=_UTC)
        last_ratchet = self.last_ratchet_time.get(symbol, _EPOCH)
        elapsed = (now - last_ratchet).total_seconds()

        # Regime-adaptive throttle (reused below for both profit and loss branches)
        if is_trending(regime) and persistence >= 0.7:
            min_interval = self.config.get('RATCHET_TRENDING_INTERVAL_SEC', 180)
            regime_factor = self.config.get('RATCHET_REGIME_FACTOR_TRENDING', 0.65)
        else:
            min_interval = self.config.get('RATCHET_MEAN_REVERTING_INTERVAL_SEC', 540)
            regime_factor = self.config.get('RATCHET_REGIME_FACTOR_MEAN_REVERTING', 1.35)

        unrealized_pct = live_unrealized

        # === S3: ASYMMETRIC LOSS-SIDE TIGHTENING ===
        # When the position is underwater and hasn't gone meaningfully positive,
        # tighten the trail to cut losses faster. This is the "don't let winners
        # turn into losers" idea applied as "don't let losers become disasters".
        # Only fires when MFE is small (i.e., trade never went green in a real way).
        loss_tighten_enabled = self.config.get('RATCHET_LOSS_TIGHTEN_ENABLED', True)
        loss_tighten_thresh = self.config.get('RATCHET_LOSS_TIGHTEN_THRESHOLD', -0.007)  # -0.7%
        loss_tighten_factor = self.config.get('RATCHET_LOSS_TIGHTEN_FACTOR', 0.55)
        mfe_disqualify = self.config.get('RATCHET_LOSS_TIGHTEN_MFE_MAX', 0.004)  # 0.4% MFE
        # Apr-19 FIX: loss-side tightening uses its own short throttle so a
        # trade turning bad at t+30s can have its stop cut immediately instead
        # of waiting 180-540s for the profit-ratchet cooldown.
        loss_tighten_interval = self.config.get('RATCHET_LOSS_TIGHTEN_MIN_INTERVAL_SEC', 45)
        if unrealized_pct <= 0:
            if (loss_tighten_enabled
                    and unrealized_pct <= loss_tighten_thresh
                    and mfe <= mfe_disqualify
                    and elapsed >= loss_tighten_interval):
                tight_trail = round(max(0.5, original_trail * loss_tighten_factor), 2)
                if tight_trail < old_trail:
                    logger.warning(f"[RATCHET LOSS-TIGHTEN] {symbol} unrealized "
                                   f"{unrealized_pct*100:+.2f}% (MFE {mfe*100:+.2f}%) — "
                                   f"tightening trail {old_trail:.2f}% → {tight_trail:.2f}%")
                    try:
                        with self._sets_lock:
                            self._ratchet_pending.add(symbol)
                        replace_req = ReplaceOrderRequest(trail=tight_trail)
                        new_order = self.client.replace_order_by_id(trailing_stop_id, replace_req)
                        new_order_id = str(new_order.id) if new_order and hasattr(new_order, 'id') else None
                        if new_order_id and new_order_id != trailing_stop_id:
                            with self.tracker._lock:
                                fresh_group = self.tracker.groups.get(symbol)
                                if fresh_group and fresh_group.state == GroupState.OPEN:
                                    fresh_group.trailing_stop_id = new_order_id
                                    fresh_group.trail_percent = tight_trail
                                    self.tracker._save()
                        self.last_ratchet_time[symbol] = now
                    except Exception as e:
                        logger.warning(f"[RATCHET LOSS-TIGHTEN] {symbol} replace failed: {e}")
                    finally:
                        with self._sets_lock:
                            self._ratchet_pending.discard(symbol)
            return

        if elapsed < min_interval:
            return

        # === TIERED RATCHET: profit level controls aggressiveness ===
        # Tier 1: Small profit (< 1%) — no ratchet, let the position breathe
        # Tier 2: Moderate profit (1-3%) — gentle tightening, floor at 70% of original trail
        # Tier 3: Strong profit (> 3%) — aggressive tightening, lock in gains
        ratchet_tier1 = self.config.get('RATCHET_TIER1_PCT', 0.01)  # 1% profit
        ratchet_tier2 = self.config.get('RATCHET_TIER2_PCT', 0.03)  # 3% profit

        if unrealized_pct < ratchet_tier1:
            # Tier 1: Too early to ratchet — let the trade develop
            return

        trailing_mult = (
            self.config.get('TRAILING_STOP_ATR_TRENDING', self.config.get('TRAILING_STOP_ATR', 3.0) * 0.8)
            if is_trending(regime)
            else self.config.get('TRAILING_STOP_ATR_MEAN_REVERTING', self.config.get('TRAILING_STOP_ATR', 4.0))
        )

        if unrealized_pct < ratchet_tier2:
            # Tier 2: Moderate profit — gentle tightening (floor at 70% of original trail)
            progress = (unrealized_pct - ratchet_tier1) / (ratchet_tier2 - ratchet_tier1)
            profit_protection = 1.0 - progress * 0.30  # scales from 1.0 → 0.70
            trail_floor_pct = self.config.get('RATCHET_TIER2_FLOOR_PCT', 1.0)
        else:
            # Tier 3: Strong profit — aggressive tightening, lock in gains
            profit_protection = max(
                self.config.get('RATCHET_PROFIT_PROTECTION_MIN', 0.30),
                0.70 - (unrealized_pct - ratchet_tier2) * 1.5
            )
            trail_floor_pct = self.config.get('RATCHET_TIER3_FLOOR_PCT', 0.5)

        # === PROFIT VELOCITY ADJUSTMENT ===
        # Dynamic trail width based on P&L momentum — tighten when momentum fades
        velocity_mult = 1.0
        if hasattr(self, 'signal_gen') and self.signal_gen:
            pv = self.signal_gen.profit_velocity
            pv.update(symbol, current_price, entry_price, direction)
            velocity_mult = pv.get_trail_multiplier(symbol)
            if velocity_mult != 1.0:
                logger.info(f"[VELOCITY] {symbol}: trail mult={velocity_mult:.2f} "
                           f"({'widening' if velocity_mult > 1 else 'tightening'})")

        distance = atr * trailing_mult * regime_factor * profit_protection * velocity_mult
        new_trail_pct = round(max(trail_floor_pct, min(35.0, (distance / current_price) * 100)), 2)

        # Only tighten (lower trail %), never widen (old_trail read under lock above)
        if new_trail_pct >= old_trail:
            return

        try:
            with self._sets_lock:
                self._ratchet_pending.add(symbol)
            replace_req = ReplaceOrderRequest(trail=new_trail_pct)
            old_id = trailing_stop_id  # Use local copy read under lock
            new_order = self.client.replace_order_by_id(old_id, replace_req)
            # Update tracker immediately with new order ID (don't wait for websocket)
            new_order_id = str(new_order.id) if new_order and hasattr(new_order, 'id') else None
            if new_order_id and new_order_id != old_id:
                with self.tracker._lock:
                    fresh_group = self.tracker.groups.get(symbol)
                    if fresh_group and fresh_group.state == GroupState.OPEN:
                        fresh_group.trailing_stop_id = new_order_id
                        # H7 FIX: Persist trail_percent atomically with new order ID
                        fresh_group.trail_percent = new_trail_pct
                        self.tracker._order_id_index.pop(old_id, None)
                        self.tracker._order_id_index[new_order_id] = symbol
                        self.tracker._save()
                    else:
                        logger.warning(f"[RATCHET] {symbol} group changed during PATCH — skipping tracker update")
                logger.debug(f"[RATCHET] {symbol} trailing stop ID updated: {old_id} → {new_order_id}")
            self.last_ratchet_time[symbol] = now
            # H7 FIX: update_trail is now redundant (persisted above), but keep for non-PATCH paths
            self.tracker.update_trail(symbol, new_trail_pct)
            tier = 'T3-aggressive' if unrealized_pct >= ratchet_tier2 else 'T2-gentle'
            logger.info(
                f"[RATCHET] {symbol} trail tightened {old_trail:.2f}% -> {new_trail_pct:.2f}% "
                f"(profit={unrealized_pct*100:+.1f}%, {tier}, regime={regime})"
            )
        except Exception as e:
            logger.warning(f"[RATCHET] Failed PATCH for {symbol}: {e} — will retry next cycle")
        finally:
            # Apr-19 FIX: discard the in-flight flag in finally so a malformed
            # replace_order response (e.g. object without .id) or any exception
            # between add() and discard() cannot leave the symbol permanently
            # stuck in _ratchet_pending — which would silently disable all
            # future ratcheting for that symbol.
            with self._sets_lock:
                self._ratchet_pending.discard(symbol)

    # ====================== PER-SYMBOL MONITOR (called under per-symbol lock) ======================
    async def _monitor_one_position(self, sym: str, qty: float):
        """Process a single position under its per-symbol asyncio.Lock (prevents TOCTOU with stream)."""
        data = self.data_ingestion.get_latest_data(sym)
        if data is None or len(data) < 50:
            return

        # FIX #8: Read tracker group ONCE under lock to avoid TOCTOU with stream handler.
        # Use this local snapshot throughout the method instead of repeated .groups.get(sym).
        with self.tracker._lock:
            tracked_group = self.tracker.groups.get(sym)
        if tracked_group is None:
            # No tracked group — fall back to position-derived direction for untracked cleanup below
            pass
        direction = tracked_group.direction if tracked_group else (1 if qty > 0 else -1)
        current_price = float(data['close'].iloc[-1])
        atr = self._compute_current_atr(data)

        # Apr-19 FIX: hoist MFE/MAE update to the START of monitor so that even
        # when the ratchet is skipped (TP in progress, cooldown, etc.) the peak
        # excursion is still recorded. Previously TIME-STOP saw MFE≈0 on trades
        # that had spiked +2% and settled flat, incorrectly flagging them as
        # dead. Updates happen under tracker lock against the LIVE group pointer.
        if tracked_group is not None:
            with self.tracker._lock:
                live_group = self.tracker.groups.get(sym)
                if live_group and live_group.state == GroupState.OPEN and live_group.entry_price:
                    live_unrealized = (current_price - live_group.entry_price) / live_group.entry_price * live_group.direction
                    updated = False
                    if live_unrealized > live_group.max_favorable_pct:
                        live_group.max_favorable_pct = float(live_unrealized)
                        updated = True
                    if live_unrealized < live_group.max_adverse_pct:
                        live_group.max_adverse_pct = float(live_unrealized)
                        updated = True
                    if updated:
                        self.tracker._save()
                    # Rebind snapshot so downstream checks see the fresh MFE/MAE
                    tracked_group = live_group

        # Log position status
        last_entry = self.last_entry_times.get(sym)
        # FIX: Use configured trading interval instead of hardcoded 15 min
        bar_interval_sec = self.config.get('TRADING_INTERVAL', 900)  # default 15 min = 900s
        bars_held = ((datetime.now(tz=_UTC) - last_entry) / timedelta(seconds=bar_interval_sec)) if last_entry else 9999
        regime, persistence = self._get_regime(sym, data)
        min_hold = (
            self.config.get('MIN_HOLD_BARS_TRENDING', 6)
            if is_trending(regime)
            else self.config.get('MIN_HOLD_BARS_MEAN_REVERTING', 3)
        )
        logger.info(
            f"  {sym} | {'LONG' if direction > 0 else 'SHORT'} {abs(qty):.2f} | "
            f"held {int(bars_held)} bars (min {min_hold}) | regime={regime}"
        )

        # === TIME-STOP: Liquidate dead trades ===
        # A position held long past its minimum hold AND with zero meaningful
        # excursion (MFE tiny, MAE tiny) is a "dead trade" — thesis isn't playing
        # out either way. Free the capital for a better opportunity.
        # Apr-19 FIX: re-read tracker.groups here — stream may have transitioned
        # this group to CLOSED/PENDING_EXIT since our initial snapshot.
        with self.tracker._lock:
            tracked_group = self.tracker.groups.get(sym)
        if (self.config.get('TIME_STOP_ENABLED', True)
                and tracked_group is not None
                and tracked_group.state == GroupState.OPEN
                and bars_held >= self.config.get('TIME_STOP_THRESHOLD_BARS', 96)):
            mfe = float(getattr(tracked_group, 'max_favorable_pct', 0.0))
            mae = float(getattr(tracked_group, 'max_adverse_pct', 0.0))
            mfe_ceiling = self.config.get('TIME_STOP_MFE_CEILING', 0.005)   # 0.5%
            mae_floor = self.config.get('TIME_STOP_MAE_FLOOR', -0.005)       # -0.5%
            # Dead trade: never went meaningfully green AND never went meaningfully red
            if abs(mfe) < mfe_ceiling and abs(mae) < abs(mae_floor):
                logger.warning(
                    f"[TIME-STOP] {sym} DEAD TRADE — held {int(bars_held)} bars, "
                    f"MFE={mfe*100:+.2f}%, MAE={mae*100:+.2f}% — liquidating to free capital"
                )
                try:
                    # FIX: Must cancel trailing stop FIRST — client.close_position()
                    # fails when qty is reserved by an active exit order.
                    # close_position_safely() handles the cancel-then-close dance.
                    ok = await self.close_position_safely(sym)
                    if ok:
                        logger.info(f"[TIME-STOP] {sym} position closed (dead trade liquidation)")
                    else:
                        logger.warning(f"[TIME-STOP] {sym} safe-close returned False — position may remain")
                    return  # Skip rest of monitoring for this position — it's closed
                except Exception as e:
                    logger.error(f"[TIME-STOP] {sym} safe-close failed: {e}")

        # === Recover PENDING_EXIT groups stuck longer than 60s ===
        # Apr-19 FIX: re-read snapshot — state may have transitioned.
        with self.tracker._lock:
            group = self.tracker.groups.get(sym)
        if group and group.state == GroupState.PENDING_EXIT:
            try:
                # HIGH-8 FIX: Use exit_initiated_at (when exit was submitted) not filled_at (entry fill)
                exit_ts = getattr(group, 'exit_initiated_at', None) or getattr(group, 'filled_at', None)
                if exit_ts:
                    filled_dt = datetime.fromisoformat(str(exit_ts))
                    if filled_dt.tzinfo is None:
                        filled_dt = filled_dt.replace(tzinfo=_UTC)
                    age = (datetime.now(tz=_UTC) - filled_dt).total_seconds()
                else:
                    age = 9999
                if age > 60:
                    logger.warning(f"[PENDING_EXIT RECOVERY] {sym} stuck in PENDING_EXIT for {age:.0f}s — force-closing")
                    try:
                        await asyncio.to_thread(self.client.close_position, sym)
                        logger.info(f"[PENDING_EXIT RECOVERY] {sym} position closed")
                    except Exception as e:
                        err_str = str(e).lower()
                        if 'position does not exist' in err_str or '40410000' in str(e):
                            logger.info(f"[PENDING_EXIT RECOVERY] {sym} already closed")
                        else:
                            logger.error(f"[PENDING_EXIT RECOVERY] Failed to close {sym}: {e}")
                    self.tracker.mark_closed(sym)
                    self.tracker.remove_group(sym)
                    with self._entry_times_lock:
                        self.last_entry_times.pop(sym, None)
                        self._save_last_entry_times_unlocked()
                    return
            except Exception as e:
                logger.error(f"[PENDING_EXIT RECOVERY] Error recovering {sym}: {e}")

        # === Software TP check FIRST (before ratchet to avoid mid-replace conflicts) ===
        group = tracked_group  # FIX #8: use local snapshot
        if group and group.state == GroupState.OPEN and group.tp_price:
            # BUG FIX: Use group.direction (from entry) not position-derived direction.
            # Alpaca paper may report short sells as LONG positions, causing
            # direction=1 with a below-entry TP → immediate false TP hit.
            tp_dir = group.direction
            tp_hit = (tp_dir > 0 and current_price >= group.tp_price) or \
                     (tp_dir < 0 and current_price <= group.tp_price)
            if tp_hit:
                logger.info(f"[TP HIT] {sym} @ {current_price:.2f} {'≥' if tp_dir > 0 else '≤'} TP {group.tp_price:.2f} — closing position")
                # Block ratchet from issuing PATCH while we cancel trailing stop
                with self._sets_lock:
                    self._tp_in_progress.add(sym)
                # FIX #21: Wrap in try/finally so _tp_in_progress is always cleared
                try:
                    # Cancel trailing stop first — retry with fresh order ID if mid-replace
                    trail_cancelled = False
                    if group.trailing_stop_id:
                        for attempt in range(3):
                            try:
                                # Re-read order ID in case _handle_replaced updated it
                                fresh_group = self.tracker.groups.get(sym)
                                cancel_id = fresh_group.trailing_stop_id if fresh_group else group.trailing_stop_id
                                await asyncio.to_thread(self.client.cancel_order_by_id, cancel_id)
                                trail_cancelled = True
                                logger.info(f"[TP HIT] Cancelled trailing stop for {sym}: {cancel_id}")
                                break
                            except Exception as e:
                                err_str = str(e)
                                if '42210000' in err_str or 'replaced' in err_str.lower():
                                    # Order mid-replace — wait for new ID from websocket
                                    logger.debug(f"[TP HIT] {sym} trailing stop mid-replace, waiting... (attempt {attempt+1})")
                                    await asyncio.sleep(1.5)
                                elif '40410000' in err_str or 'not found' in err_str.lower():
                                    trail_cancelled = True  # Already gone
                                    break
                                else:
                                    logger.warning(f"[TP HIT] Failed to cancel trailing stop for {sym}: {e}")
                                    break
                    else:
                        trail_cancelled = True
                    # Close position (only if trailing stop is out of the way)
                    if trail_cancelled:
                        try:
                            await asyncio.to_thread(self.client.close_position, sym)
                            logger.info(f"[TP HIT] {sym} position closed")
                            self.tracker.mark_closed(sym)
                            self.tracker.remove_group(sym)
                            with self._entry_times_lock:
                                self.last_entry_times.pop(sym, None)
                                self._save_last_entry_times_unlocked()
                        except Exception as e:
                            logger.error(f"[TP HIT] Failed to close {sym}: {e}")
                            err_str = str(e).lower()
                            if 'position does not exist' in err_str or '40410000' in str(e):
                                logger.info(f"[TP HIT] {sym} position already closed — cleaning up tracker")
                                self.tracker.mark_closed(sym)
                                self.tracker.remove_group(sym)
                                with self._entry_times_lock:
                                    self.last_entry_times.pop(sym, None)
                                    self._save_last_entry_times_unlocked()
                    else:
                        logger.warning(f"[TP HIT] {sym} — could not cancel trailing stop, will retry next cycle")
                finally:
                    with self._sets_lock:
                        self._tp_in_progress.discard(sym)
                return

        # === Software trailing stop for positions without native trailing stop ===
        # H5 FIX: Re-read live group from tracker under lock for writes (snapshot may be orphaned)
        group = tracked_group  # read-only checks use snapshot
        if group and group.state == GroupState.OPEN and not group.trailing_stop_id:
            if group.stop_price and group.entry_price and group.trail_percent:
                sl_dir = group.direction
                trail_dist = current_price * group.trail_percent / 100
                new_stop = round(current_price - sl_dir * trail_dist, 2)
                if sl_dir > 0 and new_stop > group.stop_price:
                    with self.tracker._lock:
                        live_group = self.tracker.groups.get(sym)
                        if live_group and live_group.state == GroupState.OPEN:
                            live_group.stop_price = new_stop
                            self.tracker._save()
                    logger.debug(f"[SOFTWARE TRAIL] {sym} stop trailed up to {new_stop:.2f}")
                elif sl_dir < 0 and new_stop < group.stop_price:
                    with self.tracker._lock:
                        live_group = self.tracker.groups.get(sym)
                        if live_group and live_group.state == GroupState.OPEN:
                            live_group.stop_price = new_stop
                            self.tracker._save()
                    logger.debug(f"[SOFTWARE TRAIL] {sym} stop trailed down to {new_stop:.2f}")

                sl_hit = (sl_dir > 0 and current_price <= group.stop_price) or \
                         (sl_dir < 0 and current_price >= group.stop_price)
                if sl_hit:
                    logger.info(f"[SOFTWARE SL] {sym} @ {current_price:.2f} hit stop {group.stop_price:.2f} — closing")
                    try:
                        # Cancel existing trailing stop first — it holds shares reserved
                        # which prevents close_position from working
                        if group.trailing_stop_id:
                            try:
                                await asyncio.to_thread(self.client.cancel_order_by_id, group.trailing_stop_id)
                                logger.info(f"[SOFTWARE SL] {sym} cancelled trailing stop {group.trailing_stop_id} before close")
                            except Exception as cancel_e:
                                logger.debug(f"[SOFTWARE SL] {sym} trailing stop cancel failed (may already be filled): {cancel_e}")
                        await asyncio.to_thread(self.client.close_position, sym)
                        logger.info(f"[SOFTWARE SL] {sym} position closed")
                        self._record_close(sym, group.direction, group.entry_price or current_price,
                                          current_price, group.filled_qty, group.regime, 'software_stop',
                                          getattr(group, 'filled_at', None))
                        self.tracker.mark_closed(sym)
                        self.tracker.remove_group(sym)
                        with self._entry_times_lock:
                            self.last_entry_times.pop(sym, None)
                            self._save_last_entry_times_unlocked()
                    except Exception as e:
                        logger.error(f"[SOFTWARE SL] Failed to close {sym}: {e}")
                    return

        # === Ratchet trailing stop via PATCH (after TP check to avoid mid-replace conflicts) ===
        if is_market_open():
            await asyncio.to_thread(self.ratchet_trailing_stop, sym, current_price, atr)

        # === Auto-close tracked fractional orphans (< 1 share, no trailing stop) ===
        group = tracked_group  # FIX #8: use local snapshot
        if group and group.state == GroupState.OPEN and not group.trailing_stop_id and abs(qty) < 1.0:
            notional = abs(qty) * current_price
            logger.info(f"[FRAC CLEANUP] {sym} tracked OPEN but only {abs(qty):.4f} shares "
                        f"(${notional:.2f}) with no trailing stop — closing orphan")
            try:
                await asyncio.to_thread(self.client.close_position, sym)
                logger.info(f"[FRAC CLEANUP] {sym} fractional orphan closed")
                self.tracker.mark_closed(sym)
                self.tracker.remove_group(sym)
                with self._entry_times_lock:
                    self.last_entry_times.pop(sym, None)
                    self._save_last_entry_times_unlocked()
            except Exception as e:
                err_str = str(e).lower()
                if 'position does not exist' in err_str or '40410000' in str(e):
                    logger.info(f"[FRAC CLEANUP] {sym} already gone — cleaning up tracker")
                    self.tracker.mark_closed(sym)
                    self.tracker.remove_group(sym)
                    with self._entry_times_lock:
                        self.last_entry_times.pop(sym, None)
                        self._save_last_entry_times_unlocked()
                else:
                    logger.error(f"[FRAC CLEANUP] Failed to close {sym}: {e}")
            return

        # === Re-attach trailing stop for tracked groups that lost it ===
        group = tracked_group  # FIX #8: use local snapshot
        if group and group.state == GroupState.OPEN and not group.trailing_stop_id and abs(qty) >= 1:
            if is_market_open():
                # Anti-churn check: if too many recent stops, close instead of reattaching
                if hasattr(self, 'signal_gen') and self.signal_gen and hasattr(self.signal_gen, 'memory'):
                    churn = self.signal_gen.memory.get_churn_penalty(sym, group.direction)
                    if churn < 0.3:
                        logger.warning(f"[REATTACH-STOP BLOCKED] {sym}: churn {churn:.3f} — closing instead")
                        try:
                            await asyncio.to_thread(self.client.close_position, sym)
                            self.tracker.mark_closed(sym)
                            self.tracker.remove_group(sym)
                        except Exception as e:
                            logger.error(f"[REATTACH-STOP BLOCKED] Failed to close {sym}: {e}")
                        return
                logger.info(f"[REATTACH-STOP] {sym} tracked but no trailing stop — resubmitting")
                trail_pct = group.trail_percent or self._get_trail_percent(current_price, atr, regime, direction=group.direction)
                close_side = OrderSide.SELL if group.direction > 0 else OrderSide.BUY
                trail_qty = int(abs(qty))
                try:
                    resp = await asyncio.to_thread(
                        self._submit_trailing_stop_with_htb_fallback,
                        sym, trail_qty, close_side, trail_pct,
                    )
                    with self.tracker._lock:
                        group.trailing_stop_id = str(resp.id)
                        group.stop_price = round(current_price * (1 - group.direction * trail_pct / 100), 2)
                        self.tracker._order_id_index[str(resp.id)] = sym
                        self.tracker._save()
                    logger.info(f"[REATTACH-STOP] {sym} trailing stop restored @ {trail_pct}% | id={resp.id}")
                except Exception as e:
                    logger.error(f"[REATTACH-STOP] Failed to resubmit trailing stop for {sym}: {e}")

        # === Re-attach missing exits for untracked positions ===
        if not group or group.state == GroupState.CLOSED:
            # Position exists but no tracker group — check if it's a tiny fractional remainder
            abs_qty = abs(qty)
            if abs_qty < 1.0:
                # Fractional remainder worth < 1 share — close it (negligible value, can't have trailing stop)
                notional = abs_qty * current_price
                logger.info(f"[CLEANUP] {sym} fractional remainder {abs_qty:.4f} shares (${notional:.2f}) — closing")
                try:
                    await asyncio.to_thread(self.client.close_position, sym)
                    logger.info(f"[CLEANUP] {sym} fractional remainder closed")
                except Exception as e:
                    logger.warning(f"[CLEANUP] Failed to close {sym} fractional remainder: {e}")
            elif is_market_open() and bars_held >= min_hold:
                # Re-attach protective orders for full positions
                await self._reattach_exits(sym, qty, direction, current_price, atr, regime)

    # ====================== MONITOR LOOP ======================
    async def monitor_positions(self):
        """Heartbeat loop: ratchet trailing stops, re-attach missing exits, slippage adaptation."""
        logger.info("=== MONITOR TASK STARTED — EVENT-DRIVEN ARCHITECTURE ===")
        while True:
            await asyncio.to_thread(self.sync_existing_positions)

            pos_count = len(self.existing_positions)
            tracked = len(self.tracker.get_open_groups())
            heartbeat_msg = (
                f"[{datetime.now(tz=_UTC).strftime('%H:%M:%S')}] Monitor — "
                f"{pos_count} positions | {tracked} tracked groups | "
                f"market {'open' if is_market_open() else 'closed'}"
            )
            print(heartbeat_msg)
            logger.info(heartbeat_msg)

            # === Per-position logic ===
            # FIX #23: Snapshot to avoid mutation during iteration (already using list())
            try:
                for sym, qty in list(self.existing_positions.items()):
                    if qty == 0 or self.data_ingestion is None:
                        continue
                    # Per-symbol lock prevents TOCTOU races between monitor and stream handler
                    # FIX #18: threading.Lock — acquire in thread to avoid blocking event loop
                    lock = self._get_symbol_lock(sym)
                    await asyncio.to_thread(lock.acquire)
                    try:
                        await self._monitor_one_position(sym, qty)
                    finally:
                        lock.release()

            except Exception as e:
                logger.error(f"Monitor per-position error: {e}", exc_info=True)

            # === Apr-19 FIX: fractional-remainder sweep ===
            # Exit fills that left a fractional remainder (< 1 share) are
            # flagged by stream.py into _pending_fractional_close. Native
            # trailing stops can't cover them, so we close them here on the
            # next monitor tick. Skip when the market is closed — close orders
            # won't route.
            if is_market_open():
                with self._fractional_close_lock:
                    pending_syms = list(self._pending_fractional_close.keys())
                for _sym in pending_syms:
                    try:
                        ok = await self.close_position_safely(_sym)
                        if ok:
                            logger.info(f"[FRAC-SWEEP] {_sym} fractional remainder closed")
                            with self._fractional_close_lock:
                                self._pending_fractional_close.pop(_sym, None)
                        else:
                            logger.debug(f"[FRAC-SWEEP] {_sym} sweep returned False — retry next cycle")
                    except Exception as _frac_e:
                        logger.debug(f"[FRAC-SWEEP] {_sym} sweep failed: {_frac_e} — retry next cycle")

            # === Slippage adaptation from recent fills ===
            await asyncio.to_thread(self._adapt_slippage)

            await asyncio.sleep(self.config.get('MONITOR_INTERVAL', 60))

    async def _reattach_exits(self, symbol: str, qty: float, direction: int,
                              current_price: float, atr: float, regime: str):
        """Re-attach trailing stop + TP for orphaned positions (no tracker group).
        Now checks anti-churn gate — if the signal generator's memory shows recent
        stops in this direction, close the position instead of reattaching."""
        # === ANTI-CHURN GATE: Check if we should even keep this position ===
        if hasattr(self, 'signal_gen') and self.signal_gen and hasattr(self.signal_gen, 'memory'):
            mem = self.signal_gen.memory
            churn_penalty = mem.get_churn_penalty(symbol, direction)
            if churn_penalty < 0.3:
                # Too many recent stops in this direction — close instead of reattach
                logger.warning(f"[REATTACH BLOCKED] {symbol}: churn penalty {churn_penalty:.3f} — "
                             f"closing orphan instead of reattaching (too many recent stops)")
                try:
                    await asyncio.to_thread(self.client.close_position, symbol)
                    logger.info(f"[REATTACH BLOCKED] {symbol} position closed")
                except Exception as e:
                    logger.error(f"[REATTACH BLOCKED] Failed to close {symbol}: {e}")
                return
            if mem.is_defensive():
                logger.warning(f"[REATTACH BLOCKED] {symbol}: defensive mode active — closing orphan")
                try:
                    await asyncio.to_thread(self.client.close_position, symbol)
                    logger.info(f"[REATTACH BLOCKED] {symbol} position closed (defensive)")
                except Exception as e:
                    logger.error(f"[REATTACH BLOCKED] Failed to close {symbol}: {e}")
                return
        logger.info(f"[REATTACH] Creating protective orders for orphaned position {symbol}")

        # Cancel any stale orders holding qty before submitting new ones
        try:
            open_orders = await asyncio.to_thread(
                self.client.get_orders, GetOrdersRequest(status='open', symbols=[symbol])
            )
            for order in open_orders:
                try:
                    await asyncio.to_thread(self.client.cancel_order_by_id, str(order.id))
                    logger.info(f"[REATTACH] Cancelled stale order for {symbol}: {order.id}")
                except Exception:
                    pass
            if open_orders:
                await asyncio.sleep(0.5)  # Brief pause for cancels to settle
        except Exception as e:
            logger.debug(f"[REATTACH] Could not check open orders for {symbol}: {e}")

        trail_pct = self._get_trail_percent(current_price, atr, regime, direction=direction)
        tp_price = self._get_tp_price(current_price, atr, regime, direction)
        close_side = OrderSide.SELL if direction > 0 else OrderSide.BUY
        position_intent = PositionIntent.SELL_TO_CLOSE if direction > 0 else PositionIntent.BUY_TO_CLOSE
        # FIX (Apr 16 gap #2): The caller's qty can be stale (tracker/sync drift — saw
        # AMD tracker=10.4257 but Alpaca actual=0.4257, causing submit-order rejection).
        # Always re-query actual available qty from Alpaca right before submitting the
        # protective stop so we don't request size we don't have.
        abs_qty = abs(qty)
        try:
            live_pos = await asyncio.to_thread(self.client.get_open_position, symbol)
            if live_pos and hasattr(live_pos, 'qty'):
                live_qty = abs(float(live_pos.qty))
                if live_qty > 0 and abs(live_qty - abs_qty) / max(abs_qty, 1e-6) > 0.05:
                    logger.warning(f"[REATTACH] {symbol} qty-drift: tracker={abs_qty:.4f} "
                                   f"actual={live_qty:.4f} — using actual")
                    abs_qty = live_qty
                elif live_qty == 0:
                    logger.warning(f"[REATTACH] {symbol} actual position zero — skipping reattach")
                    return
        except Exception as e:
            logger.debug(f"[REATTACH] {symbol} live-qty check failed: {e} — using caller qty {abs_qty}")

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
                resp = await asyncio.to_thread(
                    self._submit_trailing_stop_with_htb_fallback,
                    symbol, trail_qty, close_side, trail_pct,
                )
                trailing_stop_id = str(resp.id)
                logger.info(f"[REATTACH] {symbol} trailing stop @ {trail_pct}%")
            except Exception as e:
                logger.error(f"[REATTACH] Failed trailing stop for {symbol}: {e}")

        # TP enforced via monitor loop (Alpaca can't hold two closing orders)
        logger.info(f"[REATTACH] {symbol} take-profit target: ${tp_price:.2f} (enforced via monitor)")

        # Create a recovery tracker group (even without trailing stop, for software TP protection)
        entry_id = f"reattach_{symbol}_{datetime.now(tz=_UTC).strftime('%Y%m%d%H%M%S')}"
        self.tracker.create_group(symbol, direction, entry_id, regime=regime)
        stop_est = round(current_price * (1 - direction * trail_pct / 100), 2)
        self.tracker.mark_entry_filled(
            symbol=symbol, fill_price=current_price, filled_qty=abs_qty,
            trailing_stop_id=trailing_stop_id or '', take_profit_id='',
            trail_percent=trail_pct, tp_price=tp_price, stop_price=stop_est,
        )

    def _adapt_slippage(self):
        """Adapt limit_price_offset based on recent ENTRY fill slippage only.
        BUG FIX: Was including exit orders (trailing stops, TP sells) which have completely
        different price levels, causing a positive feedback loop that spiraled the offset
        from 0.4% to 1000%+."""
        try:
            recent_closed = self.client.get_orders(GetOrdersRequest(
                status='closed',
                after=datetime.now(tz=_UTC) - timedelta(hours=1)
            ))
        except Exception as e:
            logger.warning(f"Failed to fetch recent orders for slippage adaptation: {e}")
            return
        # FIX #41: Build set of known entry order IDs from tracker for definitive identification
        known_entry_ids = set()
        with self.tracker._lock:
            for group in self.tracker.groups.values():
                if group.entry_order_id:
                    known_entry_ids.add(group.entry_order_id)
        slippage_total = 0.0
        fill_count = 0
        for o in recent_closed:
            if not (o.filled_at and o.filled_avg_price and o.limit_price):
                continue
            order_id = str(o.id) if o.id else ''
            # Definitively identify entry orders via tracker
            if order_id and order_id in known_entry_ids:
                is_entry = True
            else:
                # Fallback heuristic for orders not in tracker (e.g., already closed groups)
                order_type = str(getattr(o, 'order_type', '')).lower().split('.')[-1]
                if order_type != 'limit':
                    continue
                # Only count if slippage is small (< 2%) — larger gaps indicate exit orders
                slippage_check = abs(float(o.filled_avg_price) - float(o.limit_price)) / float(o.limit_price)
                if slippage_check > 0.02:
                    continue
                is_entry = True
            if not is_entry:
                continue
            fill_price = float(o.filled_avg_price)
            limit_price = float(o.limit_price)
            slippage = abs(fill_price - limit_price) / limit_price
            filled_at_utc = o.filled_at.astimezone(_UTC) if o.filled_at.tzinfo else o.filled_at.replace(tzinfo=_UTC)
            if (datetime.now(tz=_UTC) - filled_at_utc).total_seconds() < 3600:
                slippage_total += slippage
                fill_count += 1
        if fill_count > 0:
            avg_slippage = slippage_total / fill_count
            new_offset = max(0.001, min(0.02, avg_slippage * 2 + 0.001))  # Hard cap at 2%
            if abs(new_offset - self.limit_price_offset) > 0.0005:
                old = self.limit_price_offset
                self.limit_price_offset = new_offset
                logger.info(f"Slippage adaptation: offset {old:.4f} -> {new_offset:.4f} (avg={avg_slippage:.4f})")

    # ====================== SAFE POSITION CLOSE ======================
    async def close_position_safely(self, symbol: str) -> bool:
        """Cancel all open orders for symbol, then close position.
        H8 FIX: Acquires per-symbol lock to prevent race with monitor and stream."""
        # H8 FIX: Acquire per-symbol lock to serialize with monitor and stream
        lock = self._get_symbol_lock(symbol)
        await asyncio.to_thread(lock.acquire)
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

            # H6 FIX: mark_closed before remove_group (sets state=CLOSED + closed_at)
            self.tracker.mark_closed(symbol)
            self.tracker.remove_group(symbol)
            with self._entry_times_lock:
                self.last_entry_times.pop(symbol, None)
                self._save_last_entry_times_unlocked()
            await asyncio.to_thread(self.sync_existing_positions, force_refresh=True)
            return True
        except Exception as e:
            logger.error(f"Failed to close position in {symbol}: {e}")
            return False
        finally:
            lock.release()

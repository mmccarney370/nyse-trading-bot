# broker/stream.py
# WebSocket trade-update handler: drives fill-based OCO, slippage tracking, causal reward push
import logging
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np  # HIGH-11 FIX: import at module level, not per-call in _push_reward
from datetime import datetime
from dateutil import tz
from typing import TYPE_CHECKING

from alpaca.trading.stream import TradingStream
from broker.order_tracker import GroupState  # M30 FIX: Module-level import for enum comparisons

if TYPE_CHECKING:
    from broker.alpaca import AlpacaBroker

logger = logging.getLogger(__name__)
_UTC = tz.gettz('UTC')


class TradeStreamHandler:
    """Wraps Alpaca TradingStream — converts websocket events into OrderTracker state transitions.

    ARCHITECTURAL NOTE (event loop boundary):
    Alpaca's TradingStream runs its own internal event loop (via asyncio.to_thread in run()).
    Our async handlers (_on_trade_update, _handle_fill, etc.) execute on that internal loop,
    NOT the main bot event loop. This means:
      - asyncio.Lock cannot protect shared state (different loops).
      - All shared state mutations use threading.Lock (OrderTracker._lock, _sets_lock, etc.).
      - Background I/O (_push_reward) uses threads, not asyncio tasks on the main loop.
    This is intentional and correct. Do not replace threading locks with asyncio locks here.
    """

    def __init__(self, broker: 'AlpacaBroker'):
        self.broker = broker
        self.config = broker.config
        self._stream: TradingStream | None = None
        self._reconnect_delay = self.config.get('STREAM_RECONNECT_DELAY_SEC', 5)
        # FIX #11: Shared thread pool for disk I/O in _push_reward (replaces unbounded Thread spawns)
        self._io_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='stream_io')
        # FIX #11 (stream): Track recently closed symbols to filter from post-exit sync
        # (position may still appear in API before settlement clears it)
        self._recently_closed: dict[str, float] = {}  # {symbol: timestamp}
        self._recently_closed_ttl = 10.0  # seconds to suppress re-sync of closed position

    def _create_stream(self) -> TradingStream:
        stream = TradingStream(
            self.config['API_KEY'],
            self.config['API_SECRET'],
            paper=self.config.get('PAPER', True),
        )
        stream.subscribe_trade_updates(self._on_trade_update)
        return stream

    async def run(self):
        """Run forever with auto-reconnect."""
        while True:
            try:
                old_stream = self._stream
                if old_stream is not None:
                    try:
                        old_stream.stop()
                    except Exception:
                        pass
                self._stream = self._create_stream()
                logger.info("[STREAM] Connected to Alpaca trade updates websocket")
                await asyncio.to_thread(self._stream.run)
            except Exception as e:
                logger.error(f"[STREAM] Disconnected: {e} — reconnecting in {self._reconnect_delay}s")
                await asyncio.sleep(self._reconnect_delay)

    async def _on_trade_update(self, data):
        """Dispatch trade update events to appropriate handlers.
        HIGH-12 NOTE: Alpaca TradingStream may dispatch this from a background thread.
        We ensure all async work runs on the main event loop via asyncio.run_coroutine_threadsafe
        if called from a non-async context."""
        try:
            event = data.event
            order = data.order
            # L20+L21 FIX: Simplified attribute access; guard against empty order_id
            order_id = str(getattr(order, 'id', None) or getattr(order, 'order_id', '') or '')
            symbol = str(getattr(order, 'symbol', '') or '')
            if not order_id:
                logger.warning(f"[STREAM] Received trade update with no order_id — skipping")
                return

            logger.debug(f"[STREAM] event={event} symbol={symbol} order_id={order_id}")

            if event == 'fill':
                await self._handle_fill(order, order_id, symbol)
            elif event == 'partial_fill':
                await self._handle_partial_fill(order, order_id, symbol)
            elif event in ('canceled', 'expired', 'rejected'):
                await self._handle_cancel(order, order_id, symbol, event)
            elif event == 'replaced':
                await self._handle_replaced(order, order_id, symbol)
            elif event in ('pending_replace', 'pending_cancel'):
                logger.debug(f"[STREAM] Order {event}: {symbol} {order_id} — awaiting confirmation")
            elif event == 'new':
                logger.debug(f"[STREAM] New order accepted: {symbol} {order_id}")
        except Exception as e:
            logger.error(f"[STREAM] Error handling trade update: {e}", exc_info=True)

    async def _handle_fill(self, order, order_id: str, symbol: str):
        """Handle a full fill event.
        Guard: If group is already PENDING_EXIT or CLOSED, skip to prevent double processing
        (e.g. partial_fill completing + real fill event arriving for the same order)."""
        tracker = self.broker.tracker
        fill_price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
        filled_qty = float(order.filled_qty) if order.filled_qty else 0.0

        group = tracker.lookup_by_order_id(order_id)
        if not group:
            logger.info(f"[STREAM] Fill for untracked order {order_id} ({symbol}) — ignoring")
            return

        # Guard against double-call: if already processed (e.g. partial_fill→_handle_fill + real fill),
        # skip exit processing to prevent double reward push / double close
        if order_id in (group.trailing_stop_id, group.take_profit_id):
            if group.state in (GroupState.PENDING_EXIT, GroupState.CLOSED):
                logger.info(f"[STREAM] Fill for {symbol} order {order_id} skipped — group already {group.state.value}")
                return

        if order_id == group.entry_order_id:
            # === ENTRY FILL: Submit trailing stop + take-profit ===
            # HIGH-14 FIX: Guard against duplicate processing — partial_fill completing
            # plus real fill event can both reach here. Only process if group is still
            # in PENDING_ENTRY state (mark_entry_filled transitions to OPEN).
            if group.state != GroupState.PENDING_ENTRY:
                logger.info(f"[STREAM] Entry fill for {symbol} skipped — group already {group.state.value} (duplicate event)")
                return

            logger.info(f"[STREAM] ENTRY FILLED: {symbol} @ {fill_price:.2f} x {filled_qty}")

            # Measure slippage
            if group.entry_price:  # limit_price stored temporarily
                group.slippage = abs(fill_price - group.entry_price) / group.entry_price

            # B2: Adverse-selection — record this fill for post-fill drift tracking
            if hasattr(self.broker, 'adverse_selection') and self.broker.adverse_selection:
                try:
                    self.broker.adverse_selection.record_fill(
                        symbol=symbol,
                        side=int(group.direction),
                        fill_price=float(fill_price),
                    )
                except Exception as e:
                    logger.debug(f"[ADVERSE-SEL] record_fill failed for {symbol}: {e}")
            # ESP: record realized slippage for future prediction
            # NOTE: NO local `from datetime import datetime` here — Python sees any
            # assignment in a function and marks the name as local for the entire
            # function, which UnboundLocalErrors lines 234/432 that use the module-
            # level datetime. The module-level import at line 8 covers us.
            if (hasattr(self.broker, 'slippage_predictor') and self.broker.slippage_predictor
                    and group.slippage is not None):
                try:
                    size_usd = float(fill_price) * float(filled_qty)
                    self.broker.slippage_predictor.record(
                        symbol=symbol,
                        slip_bps=float(group.slippage) * 10000.0,
                        hour=datetime.now().hour,
                        size_usd=size_usd,
                        direction=int(group.direction),
                    )
                except Exception as e:
                    logger.debug(f"[SLIPPAGE] record failed for {symbol}: {e}")

            await self.broker.submit_exit_orders(symbol, group, fill_price, filled_qty)

        elif order_id and order_id in (group.trailing_stop_id, group.take_profit_id):
            # === EXIT FILL: Cancel the other leg (manual OCO) ===
            # FIX #36: Acquire per-symbol lock to prevent race with monitor loop
            lock = self.broker._get_symbol_lock(symbol)
            await asyncio.to_thread(lock.acquire)
            try:
                # HIGH-13 FIX: Re-read group after acquiring lock — the reference from
                # line 106 may be stale if another thread modified tracker between then
                # and now (e.g. monitor loop replaced the group).
                group = tracker.lookup_by_order_id(order_id)
                if not group:
                    logger.info(f"[STREAM] Exit fill for {symbol} order {order_id} — group gone after lock acquisition, skipping")
                    return
                cancel_id = tracker.mark_exit_fill(symbol, order_id, fill_price)
                if cancel_id:
                    try:
                        await asyncio.to_thread(self.broker.client.cancel_order_by_id, cancel_id)
                        logger.info(f"[STREAM] OCO cancel sent for {symbol}: {cancel_id}")
                    except Exception as e:
                        logger.warning(f"[STREAM] Failed to cancel opposing order {cancel_id}: {e}")

                # Push realized return to causal buffer
                self._push_reward(symbol, group, fill_price)

                # Feed adaptive memory: record stop-out or TP hit
                is_stop = (order_id == group.trailing_stop_id)
                direction = group.direction if hasattr(group, 'direction') else 1
                pnl = (fill_price - group.entry_price) * group.filled_qty
                if direction == -1:  # short position: profit when price drops
                    pnl = -pnl
                # BPS: update Bayesian posterior with realized return
                if hasattr(self.broker, 'signal_gen') and self.broker.signal_gen \
                        and hasattr(self.broker.signal_gen, 'bayesian_sizer') \
                        and self.broker.signal_gen.bayesian_sizer \
                        and group.entry_price:
                    try:
                        realized_return = pnl / (abs(group.entry_price) * abs(group.filled_qty) + 1e-9)
                        self.broker.signal_gen.bayesian_sizer.update(symbol, float(realized_return))
                    except Exception as e:
                        logger.debug(f"[BAYESIAN-SIZE] update failed for {symbol}: {e}")
                if hasattr(self.broker, 'signal_gen') and self.broker.signal_gen:
                    mem = self.broker.signal_gen.memory
                    if is_stop:
                        mem.record_stop_out(symbol, direction, fill_price)
                    mem.record_trade_outcome(
                        symbol=symbol, direction=direction, pnl=pnl,
                        meta_prob=getattr(group, '_meta_prob', 0.5),
                        ppo_prob=getattr(group, '_ppo_prob', 0.5),
                        sentiment=getattr(group, '_sentiment', 0.0),
                    )
                # Record autopsy + clear velocity via unified helper
                exit_reason = 'stop' if is_stop else 'take_profit'
                self.broker._record_close(
                    symbol, direction, group.entry_price or fill_price,
                    fill_price, group.filled_qty,
                    getattr(group, 'regime', 'mean_reverting'),
                    exit_reason, getattr(group, 'filled_at', None)
                )

                # HIGH-13 FIX: Perform all close operations under single lock acquisition
                # to prevent monitor seeing intermediate states.
                # Combined mark_closed + remove_group into one lock + single _save() call.
                with tracker._lock:
                    group_ref = tracker.groups.get(symbol)
                    if group_ref:
                        group_ref.state = GroupState.CLOSED
                        group_ref.closed_at = datetime.now(tz=_UTC).isoformat()
                    group_removed = tracker.groups.pop(symbol, None)
                    if group_removed:
                        for oid in (group_removed.entry_order_id, group_removed.trailing_stop_id, group_removed.take_profit_id):
                            if oid:
                                tracker._order_id_index.pop(oid, None)
                    tracker._save()
                    logger.info(f"[TRACKER] {symbol} CLOSED and removed (single save)")

                # FIX #10: Log fractional remainder warning — monitor will clean up in ~20s
                filled_qty = float(order.filled_qty) if order.filled_qty else 0.0
                orig_qty = float(order.qty) if hasattr(order, 'qty') and order.qty else filled_qty
                if filled_qty > 0 and orig_qty > filled_qty:
                    remainder = orig_qty - filled_qty
                    logger.warning(
                        f"[STREAM] {symbol} exit filled {filled_qty} of {orig_qty} — "
                        f"fractional remainder {remainder:.4f} shares unprotected until next monitor cycle"
                    )

                # Mark symbol as recently closed before sync
                import time as _time
                self._recently_closed[symbol] = _time.time()

                # Clean up entry time under lock (fast, no I/O)
                self.broker.last_entry_times.pop(symbol, None)
            finally:
                # H10+H11 FIX: Release per-symbol lock BEFORE sleep and sync.
                # The critical tracker state is already committed. Holding the lock during
                # the 2s sleep + sync blocked ALL trade updates for this symbol.
                lock.release()

            # Post-lock: settlement delay + sync (non-blocking for other trade updates)
            await asyncio.sleep(2.0)
            await asyncio.to_thread(self.broker.sync_existing_positions, force_refresh=True)

            # Filter out recently closed symbols that may have reappeared
            now_ts = _time.time()
            # M31 FIX: Protect positions mutation with _positions_lock
            with self.broker._positions_lock:
                for rc_sym, rc_time in list(self._recently_closed.items()):
                    if now_ts - rc_time < self._recently_closed_ttl:
                        self.broker.existing_positions.pop(rc_sym, None)
                    else:
                        del self._recently_closed[rc_sym]

            # Persist entry times (I/O outside lock)
            await asyncio.to_thread(self.broker._save_last_entry_times)

    async def _handle_partial_fill(self, order, order_id: str, symbol: str):
        filled_qty = float(order.filled_qty) if order.filled_qty else 0.0
        total_qty = float(order.qty) if order.qty else 0.0
        logger.info(f"[STREAM] Partial fill: {symbol} {order_id} filled_qty={filled_qty}/{total_qty}")
        # Track partial exit fills — if this is an exit order and fully filled, process as exit
        tracker = self.broker.tracker
        group = tracker.lookup_by_order_id(order_id)
        if group and order_id in (group.trailing_stop_id, group.take_profit_id):
            remaining = total_qty - filled_qty if total_qty > 0 else 0
            if remaining <= 0.001:
                # Fully filled via partial fills — process as full exit
                logger.info(f"[STREAM] Partial fills complete for {symbol} — processing as full exit")
                # M28 FIX: Removed dead fill_price assignment (value was never used)
                await self._handle_fill(order, order_id, symbol)

    async def _handle_cancel(self, order, order_id: str, symbol: str, event: str):
        logger.info(f"[STREAM] Order {event}: {symbol} {order_id}")
        tracker = self.broker.tracker
        group = tracker.lookup_by_order_id(order_id)
        if not group:
            return
        # If entry was cancelled/rejected, clean up the group — but check for partial fills first
        # M30 FIX: Use enum comparison instead of string
        if order_id == group.entry_order_id and group.state == GroupState.PENDING_ENTRY:
            partial_qty = float(order.filled_qty) if hasattr(order, 'filled_qty') and order.filled_qty else 0.0
            if partial_qty > 0:
                # Partial fill occurred before cancel — position exists on Alpaca.
                # Transition to OPEN so the monitor loop can reattach protective orders.
                fill_price = float(order.filled_avg_price) if hasattr(order, 'filled_avg_price') and order.filled_avg_price else 0.0
                logger.warning(
                    f"[STREAM] Entry order {event} for {symbol} with partial fill "
                    f"qty={partial_qty} @ {fill_price:.2f} — transitioning to OPEN for monitor reattach"
                )
                with tracker._lock:
                    group.state = GroupState.OPEN
                    group.filled_qty = partial_qty
                    if fill_price:
                        group.entry_price = fill_price
                    direction = group.direction if hasattr(group, 'direction') and group.direction else 1
                    if fill_price and not getattr(group, 'stop_price', None):
                        group.stop_price = fill_price * (0.95 if direction > 0 else 1.05)
                    if fill_price and not getattr(group, 'tp_price', None):
                        group.tp_price = fill_price * (1.10 if direction > 0 else 0.90)
                    tracker._save()
                # H12 FIX: Immediately submit exit orders to protect the partial position
                # instead of waiting up to 60s for the next monitor cycle.
                if fill_price and partial_qty > 0:
                    try:
                        await self.broker.submit_exit_orders(symbol, group, fill_price, partial_qty)
                        logger.info(f"[STREAM] Protective exit orders submitted for partial fill {symbol} qty={partial_qty}")
                    except Exception as exit_e:
                        logger.warning(f"[STREAM] Failed to submit exit orders for partial {symbol}: {exit_e} — monitor will reattach")
            else:
                logger.warning(f"[STREAM] Entry order {event} for {symbol} — removing group")
                tracker.remove_group(symbol)
        elif order_id in (group.trailing_stop_id, group.take_profit_id) and group.state == GroupState.OPEN:
            # Clear the cancelled order ID so monitor loop can reattach
            with tracker._lock:
                if order_id == group.trailing_stop_id:
                    tracker._order_id_index.pop(order_id, None)
                    group.trailing_stop_id = None  # M29 FIX: Use None not empty string
                    tracker._save()
                    logger.warning(
                        f"[STREAM] Trailing stop {event} for {symbol} (order={order_id}) — "
                        f"cleared from tracker. Monitor will re-attach."
                    )
                elif order_id == group.take_profit_id:
                    tracker._order_id_index.pop(order_id, None)
                    group.take_profit_id = None  # M29 FIX: Use None not empty string
                    tracker._save()
                    logger.warning(
                        f"[STREAM] Take-profit {event} for {symbol} (order={order_id}) — "
                        f"cleared from tracker. Monitor loop will re-attach exits."
                    )
                else:
                    logger.warning(
                        f"[STREAM] Exit order {event} for {symbol} (order={order_id}) — "
                        f"position may be UNPROTECTED. Monitor loop will re-attach exits."
                    )

    async def _handle_replaced(self, order, order_id: str, symbol: str):
        """Handle order replacement (e.g. ratchet tightening trailing stop via PATCH).
        CRITICAL: Alpaca creates a NEW order ID on replace — must update tracker."""
        # Alpaca 'replaced' event: data.order is the ORIGINAL order.
        # The new order ID is in order.replaced_by (not order.id).
        if hasattr(order, 'replaced_by') and order.replaced_by:
            new_order_id = str(order.replaced_by)
        else:
            # Fallback: use order.id (may be wrong but better than nothing)
            new_order_id = str(order.id) if hasattr(order, 'id') else order_id
            logger.warning(f"[STREAM] 'replaced' event for {symbol} missing replaced_by — falling back to order.id={new_order_id}")
        logger.info(f"[STREAM] Order replaced: {symbol} old={order_id} new={new_order_id}")
        tracker = self.broker.tracker
        # Use a single lock acquisition for lookup + update to prevent stale reference
        with tracker._lock:
            sym = tracker._order_id_index.get(order_id)
            group = tracker.groups.get(sym) if sym else None
            if group and order_id == group.trailing_stop_id:
                old_id = group.trailing_stop_id
                group.trailing_stop_id = new_order_id
                tracker._order_id_index.pop(old_id, None)
                tracker._order_id_index[new_order_id] = sym
                # FIX #21: Don't overwrite trail_percent from replaced event — data.order is the
                # ORIGINAL order, so trail_percent/stop_price are stale. The ratchet already set
                # the correct new values on the group before submitting the replace request.
                # Only update the order ID (done above), not the trail parameters.
                tracker._save()
                logger.info(f"[STREAM] Trailing stop ID updated: {old_id} → {new_order_id}")
            elif group and order_id == group.take_profit_id:
                # FIX: Handle TP order replacement — track the new order ID
                old_id = group.take_profit_id
                group.take_profit_id = new_order_id
                tracker._order_id_index.pop(old_id, None)
                tracker._order_id_index[new_order_id] = sym
                # Don't overwrite tp_price from replaced event — data.order is the ORIGINAL order,
                # so limit_price is the old TP value. The correct new TP was already set before
                # the replace request was submitted.
                tracker._save()
                logger.info(f"[STREAM] Take-profit ID updated: {old_id} → {new_order_id}")
        # Best-effort cleanup of ratchet in-flight flag.
        # The ratchet already clears this flag immediately after PATCH success,
        # so this is just a redundant safety net. Use discard() directly without
        # _sets_lock to avoid lock ordering inversion (tracker._lock → _sets_lock
        # here vs _sets_lock → tracker._lock in ratchet_trailing_stop).
        # Python set.discard() is atomic under GIL for a single operation.
        target_sym = sym or symbol
        if target_sym:
            self.broker._ratchet_pending.discard(target_sym)

    def _push_reward(self, symbol: str, group, fill_price: float):
        """Push realized return to causal buffer if bot context is available.
        HIGH-11 FIX: This is called from async context — schedule disk I/O via to_thread."""
        bot = self.broker.bot
        if bot is None:
            return
        if not hasattr(bot, 'live_signal_history'):
            return
        entry_price = group.entry_price
        if not entry_price:
            return
        ret = (fill_price - entry_price) / entry_price * group.direction
        # Push to live_signal_history
        # NOTE: Thread safety — Python's GIL protects dict/list mutations (get, append,
        # __setitem__) at the bytecode level, so these operations are atomic.
        # The disk I/O is already scheduled on the main event loop via call_soon_threadsafe below.
        history = bot.live_signal_history.get(symbol, [])
        if history:
            last_entry = history[-1]
            if last_entry.get('realized_return') is None:
                last_entry['realized_return'] = ret
                last_entry['closed_at'] = datetime.now(tz=_UTC).isoformat()
        # Push to causal manager
        causal_manager = getattr(getattr(bot, 'signal_gen', None), 'portfolio_causal_manager', None)
        if causal_manager and history:
            last_entry = history[-1]
            stored_obs = last_entry.get('obs')
            if stored_obs:
                obs = np.array(stored_obs, dtype=np.float32).reshape(1, -1)
                action = last_entry.get('direction', 1) * last_entry.get('confidence', 0.5)
                causal_manager.add_transition(obs, action, ret)
                # FIX #18: Save buffer in background thread to avoid blocking stream
                def _save_causal():
                    try:
                        causal_manager.save_buffer()
                    except Exception as e:
                        logger.warning(f"[STREAM] Causal buffer save failed: {e}")
                self._io_pool.submit(_save_causal)  # FIX #11: use shared pool instead of unbounded threads
                logger.info(f"[STREAM] Causal push: {symbol} realized={ret:+.4f}")
        # FIX #18: Persist live_signal_history in background thread to avoid blocking stream
        if hasattr(bot, '_save_live_signal_history'):
            def _save_history():
                try:
                    bot._save_live_signal_history()
                except Exception as e:
                    logger.warning(f"[STREAM] Signal history save failed: {e}")
            self._io_pool.submit(_save_history)  # FIX #11: use shared pool instead of unbounded threads

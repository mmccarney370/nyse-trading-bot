# broker/stream.py
# WebSocket trade-update handler: drives fill-based OCO, slippage tracking, causal reward push
import logging
import asyncio
from datetime import datetime
from dateutil import tz
from typing import TYPE_CHECKING

from alpaca.trading.stream import TradingStream

if TYPE_CHECKING:
    from broker.alpaca import AlpacaBroker

logger = logging.getLogger(__name__)
_UTC = tz.gettz('UTC')


class TradeStreamHandler:
    """Wraps Alpaca TradingStream — converts websocket events into OrderTracker state transitions."""

    def __init__(self, broker: 'AlpacaBroker'):
        self.broker = broker
        self.config = broker.config
        self._stream: TradingStream | None = None
        self._reconnect_delay = self.config.get('STREAM_RECONNECT_DELAY_SEC', 5)

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
        """Dispatch trade update events to appropriate handlers."""
        try:
            event = data.event
            order = data.order
            order_id = str(order.id) if hasattr(order, 'id') else str(getattr(order, 'order_id', ''))
            symbol = order.symbol if hasattr(order, 'symbol') else ''

            logger.debug(f"[STREAM] event={event} symbol={symbol} order_id={order_id}")

            if event == 'fill':
                await self._handle_fill(order, order_id, symbol)
            elif event == 'partial_fill':
                await self._handle_partial_fill(order, order_id, symbol)
            elif event in ('canceled', 'expired', 'rejected'):
                await self._handle_cancel(order, order_id, symbol, event)
            elif event == 'replaced':
                await self._handle_replaced(order, order_id, symbol)
            elif event == 'new':
                logger.debug(f"[STREAM] New order accepted: {symbol} {order_id}")
        except Exception as e:
            logger.error(f"[STREAM] Error handling trade update: {e}", exc_info=True)

    async def _handle_fill(self, order, order_id: str, symbol: str):
        """Handle a full fill event."""
        tracker = self.broker.tracker
        fill_price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
        filled_qty = float(order.filled_qty) if order.filled_qty else 0.0

        group = tracker.lookup_by_order_id(order_id)
        if not group:
            logger.info(f"[STREAM] Fill for untracked order {order_id} ({symbol}) — ignoring")
            return

        if order_id == group.entry_order_id:
            # === ENTRY FILL: Submit trailing stop + take-profit ===
            logger.info(f"[STREAM] ENTRY FILLED: {symbol} @ {fill_price:.2f} x {filled_qty}")

            # Measure slippage
            if group.entry_price:  # limit_price stored temporarily
                group.slippage = abs(fill_price - group.entry_price) / group.entry_price

            await self.broker.submit_exit_orders(symbol, group, fill_price, filled_qty)

        elif order_id and order_id in (group.trailing_stop_id, group.take_profit_id):
            # === EXIT FILL: Cancel the other leg (manual OCO) ===
            cancel_id = tracker.mark_exit_fill(symbol, order_id, fill_price)
            if cancel_id:
                try:
                    await asyncio.to_thread(self.broker.client.cancel_order_by_id, cancel_id)
                    logger.info(f"[STREAM] OCO cancel sent for {symbol}: {cancel_id}")
                except Exception as e:
                    logger.warning(f"[STREAM] Failed to cancel opposing order {cancel_id}: {e}")

            # Push realized return to causal buffer
            self._push_reward(symbol, group, fill_price)

            tracker.mark_closed(symbol)
            tracker.remove_group(symbol)
            self.broker.sync_existing_positions(force_refresh=True)

            # Clean up entry time
            self.broker.last_entry_times.pop(symbol, None)
            self.broker._save_last_entry_times()

    async def _handle_partial_fill(self, order, order_id: str, symbol: str):
        filled_qty = float(order.filled_qty) if order.filled_qty else 0.0
        logger.info(f"[STREAM] Partial fill: {symbol} {order_id} filled_qty={filled_qty}")

    async def _handle_cancel(self, order, order_id: str, symbol: str, event: str):
        logger.info(f"[STREAM] Order {event}: {symbol} {order_id}")
        tracker = self.broker.tracker
        group = tracker.lookup_by_order_id(order_id)
        if not group:
            return
        # If entry was cancelled/rejected, clean up the group
        if order_id == group.entry_order_id and group.state.value == 'pending_entry':
            logger.warning(f"[STREAM] Entry order {event} for {symbol} — removing group")
            tracker.remove_group(symbol)
        elif order_id in (group.trailing_stop_id, group.take_profit_id) and group.state.value == 'open':
            # Clear the cancelled order ID so monitor loop can reattach
            with tracker._lock:
                if order_id == group.trailing_stop_id:
                    tracker._order_id_index.pop(order_id, None)
                    group.trailing_stop_id = ''
                    tracker._save()
                    logger.warning(
                        f"[STREAM] Trailing stop {event} for {symbol} (order={order_id}) — "
                        f"cleared from tracker. Monitor will re-attach."
                    )
                else:
                    logger.warning(
                        f"[STREAM] Exit order {event} for {symbol} (order={order_id}) — "
                        f"position may be UNPROTECTED. Monitor loop will re-attach exits."
                    )

    async def _handle_replaced(self, order, order_id: str, symbol: str):
        """Handle order replacement (e.g. ratchet tightening trailing stop via PATCH).
        CRITICAL: Alpaca creates a NEW order ID on replace — must update tracker."""
        new_order_id = str(order.id) if hasattr(order, 'id') else order_id
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
                new_trail = float(order.trail_percent) if hasattr(order, 'trail_percent') and order.trail_percent else None
                new_stop = float(order.stop_price) if hasattr(order, 'stop_price') and order.stop_price else None
                if new_trail:
                    group.trail_percent = new_trail
                    if new_stop is not None:
                        group.stop_price = new_stop
                tracker._save()
                logger.info(f"[STREAM] Trailing stop ID updated: {old_id} → {new_order_id}")
        # Clear ratchet in-flight flag so next ratchet can proceed
        self.broker._ratchet_pending.discard(sym or symbol)

    def _push_reward(self, symbol: str, group, fill_price: float):
        """Push realized return to causal buffer if bot context is available."""
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
        history = bot.live_signal_history.get(symbol, [])
        if history:
            last_entry = history[-1]
            if last_entry.get('realized_return') is None:
                last_entry['realized_return'] = ret
        # Push to causal manager
        causal_manager = getattr(getattr(bot, 'signal_gen', None), 'portfolio_causal_manager', None)
        if causal_manager and history:
            last_entry = history[-1]
            stored_obs = last_entry.get('obs')
            if stored_obs:
                import numpy as np
                obs = np.array(stored_obs, dtype=np.float32).reshape(1, -1)
                action = last_entry.get('direction', 1) * last_entry.get('confidence', 0.5)
                causal_manager.add_transition(obs, action, ret)
                causal_manager.save_buffer()
                logger.info(f"[STREAM] Causal push: {symbol} realized={ret:+.4f}")
        # FIX: Persist live_signal_history after recording realized return
        if hasattr(bot, '_save_live_signal_history'):
            try:
                bot._save_live_signal_history()
            except Exception as e:
                logger.warning(f"[STREAM] Failed to save live_signal_history after reward push: {e}")

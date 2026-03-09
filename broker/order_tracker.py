# broker/order_tracker.py
# Persistent order-group state machine: tracks entry → exit lifecycle per symbol
import json
import os
import logging
import tempfile
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from dateutil import tz
from enum import Enum
from typing import Optional, Dict

logger = logging.getLogger(__name__)

_UTC = tz.gettz('UTC')
ORDER_TRACKER_FILE = "order_tracker.json"


class GroupState(str, Enum):
    PENDING_ENTRY = "pending_entry"       # Limit entry submitted, awaiting fill
    OPEN = "open"                         # Entry filled, exit orders active
    PENDING_EXIT = "pending_exit"         # Exit triggered, awaiting full close
    CLOSED = "closed"                     # Position fully closed


@dataclass
class OrderGroup:
    symbol: str
    direction: int                        # +1 long, -1 short
    state: GroupState = GroupState.PENDING_ENTRY
    entry_order_id: Optional[str] = None
    entry_price: Optional[float] = None
    filled_qty: float = 0.0
    trailing_stop_id: Optional[str] = None
    take_profit_id: Optional[str] = None
    trail_percent: Optional[float] = None  # Current trail % on the trailing stop
    tp_price: Optional[float] = None
    stop_price: Optional[float] = None     # Last known stop price (for logging)
    regime: str = "mean_reverting"
    persistence: float = 0.5
    created_at: Optional[str] = None
    filled_at: Optional[str] = None
    closed_at: Optional[str] = None
    slippage: Optional[float] = None       # Entry slippage measured on fill
    extended_hours: bool = False

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(tz=_UTC).isoformat()
        if isinstance(self.state, str):
            self.state = GroupState(self.state)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['state'] = self.state.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'OrderGroup':
        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


class OrderTracker:
    """Manages all active OrderGroups with atomic disk persistence."""

    def __init__(self, filepath: str = ORDER_TRACKER_FILE):
        self.filepath = filepath
        self.groups: Dict[str, OrderGroup] = {}  # keyed by symbol
        self._order_id_index: Dict[str, str] = {}  # order_id → symbol (reverse lookup)
        self._load()

    def _load(self):
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            for sym, gd in data.items():
                group = OrderGroup.from_dict(gd)
                self.groups[sym] = group
                self._index_group(group)
            logger.info(f"OrderTracker loaded {len(self.groups)} groups from {self.filepath}")
        except Exception as e:
            logger.warning(f"Failed to load order tracker: {e}")

    def _save(self):
        try:
            data = {sym: g.to_dict() for sym, g in self.groups.items()}
            # Atomic write
            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(self.filepath) or '.', suffix='.tmp')
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
            shutil.move(tmp_path, self.filepath)
        except Exception as e:
            logger.error(f"Failed to save order tracker: {e}")

    def _index_group(self, group: OrderGroup):
        for oid in (group.entry_order_id, group.trailing_stop_id, group.take_profit_id):
            if oid:
                self._order_id_index[oid] = group.symbol

    def lookup_by_order_id(self, order_id: str) -> Optional[OrderGroup]:
        sym = self._order_id_index.get(order_id)
        return self.groups.get(sym) if sym else None

    def create_group(self, symbol: str, direction: int, entry_order_id: str,
                     regime: str = "mean_reverting", persistence: float = 0.5,
                     extended_hours: bool = False) -> OrderGroup:
        group = OrderGroup(
            symbol=symbol,
            direction=direction,
            entry_order_id=entry_order_id,
            regime=regime,
            persistence=persistence,
            extended_hours=extended_hours,
        )
        self.groups[symbol] = group
        self._order_id_index[entry_order_id] = symbol
        self._save()
        logger.info(f"[TRACKER] Created group for {symbol} dir={direction} entry_id={entry_order_id}")
        return group

    def mark_entry_filled(self, symbol: str, fill_price: float, filled_qty: float,
                          trailing_stop_id: str, take_profit_id: str,
                          trail_percent: float, tp_price: float, stop_price: float):
        group = self.groups.get(symbol)
        if not group:
            logger.warning(f"[TRACKER] mark_entry_filled: no group for {symbol}")
            return
        group.state = GroupState.OPEN
        group.entry_price = fill_price
        group.filled_qty = filled_qty
        group.trailing_stop_id = trailing_stop_id
        group.take_profit_id = take_profit_id
        group.trail_percent = trail_percent
        group.tp_price = tp_price
        group.stop_price = stop_price
        group.filled_at = datetime.now(tz=_UTC).isoformat()
        # Measure entry slippage (limit vs fill)
        if trailing_stop_id:
            self._order_id_index[trailing_stop_id] = symbol
        if take_profit_id:
            self._order_id_index[take_profit_id] = symbol
        self._save()
        logger.info(
            f"[TRACKER] {symbol} FILLED @ {fill_price:.2f} qty={filled_qty} | "
            f"trail={trail_percent:.2f}% | TP={tp_price:.2f}"
        )

    def mark_exit_fill(self, symbol: str, exit_order_id: str, fill_price: float) -> Optional[str]:
        """Mark one exit leg as filled. Returns the OTHER order_id to cancel (manual OCO)."""
        group = self.groups.get(symbol)
        if not group:
            return None
        group.state = GroupState.PENDING_EXIT
        cancel_id = None
        if exit_order_id == group.trailing_stop_id:
            cancel_id = group.take_profit_id
            logger.info(f"[TRACKER] {symbol} stop filled @ {fill_price:.2f} → cancel TP {cancel_id}")
        elif exit_order_id == group.take_profit_id:
            cancel_id = group.trailing_stop_id
            logger.info(f"[TRACKER] {symbol} TP filled @ {fill_price:.2f} → cancel trail {cancel_id}")
        else:
            logger.warning(f"[TRACKER] {symbol} unknown exit order {exit_order_id}")
        self._save()
        return cancel_id

    def mark_closed(self, symbol: str):
        group = self.groups.get(symbol)
        if not group:
            return
        group.state = GroupState.CLOSED
        group.closed_at = datetime.now(tz=_UTC).isoformat()
        self._save()
        logger.info(f"[TRACKER] {symbol} CLOSED")

    def remove_group(self, symbol: str):
        group = self.groups.pop(symbol, None)
        if group:
            for oid in (group.entry_order_id, group.trailing_stop_id, group.take_profit_id):
                self._order_id_index.pop(oid, None)
            self._save()

    def get_open_groups(self) -> Dict[str, OrderGroup]:
        return {s: g for s, g in self.groups.items() if g.state in (GroupState.PENDING_ENTRY, GroupState.OPEN)}

    def update_trail(self, symbol: str, new_trail_percent: float, new_stop_price: float = None):
        group = self.groups.get(symbol)
        if group:
            group.trail_percent = new_trail_percent
            if new_stop_price is not None:
                group.stop_price = new_stop_price
            self._save()

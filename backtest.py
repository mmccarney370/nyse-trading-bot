# backtest.py
# FINAL UPDATED VERSION (Feb 2026) — COMPLETE, NON-TRUNCATED, FULLY PATCHED + REALISM ENHANCEMENTS + HMM REGIME UPGRADE
# FIXED: Precomputation flooding (PortfolioEnv created once at the very start and passed to generate_portfolio_actions)
# FIXED: Regime detection stuck on final bar (forced index sort + dedup + explicit last_bar logging)
# FIXED: Regime look-ahead bias in backtest — added is_backtest=True flag to detect_regime calls
# FIXED: TypeError 'int' object is not subscriptable in portfolio mode
# FIXED: Pause check spam (logs only on state change or new day)
# FIXED: Simulated position updates in portfolio backtest
# NEW: is_backtest=True flag passed to check_pause_conditions()
# NEW: Proper daily equity reset in backtest loop
# CRITICAL FIX (Feb 17 2026): Pass timestamp to generate_portfolio_actions + robust slicing for short-history symbols (AAPL/SMCI)
# CRITICAL ACCOUNTING FIX (Feb 18 2026): Removed duplicate cash -= exit_cost on exits (was causing massive negative equity on short closes)
# NEW FIXES (Feb 18 2026): Global leverage cap, per-symbol notional cap, cash floor, finite checks, extra debug logging
# NEW ACCOUNTING FIXES (Feb 18 2026): Fixed short position cash flows (use abs_qty for buys/sells, correct PNL signs)
# - Entry short: cash += abs_qty * entry_price (sell to open)
# - Exit short: cash -= abs_qty * exit_price (buy to close)
# - PNL: (entry_price - exit_price) * abs_qty for shorts (positive if entry > exit)
# - Global leverage enforced EVERY bar: normalize target_weights if sum(abs) > MAX_LEVERAGE
# - Skip entries if insufficient cash (with log)
# - More logging on huge positions/PNL
# All original per-symbol logic preserved unchanged as fallback
import logging
import numpy as np
import pandas as pd
import asyncio
import random
from datetime import datetime, timedelta
from dateutil import tz
from config import CONFIG
import warnings
from strategy.regime import detect_regime
from models.portfolio_env import PortfolioEnv # Required for portfolio mode precomputation
warnings.filterwarnings("ignore", category=UserWarning, module="vectorbt")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
logger = logging.getLogger(__name__)
class Backtester:
    def __init__(self, config, data_ingestion, trainer, signal_gen, risk_manager):
        self.config = config
        self.data_ingestion = data_ingestion
        self.trainer = trainer
        self.signal_gen = signal_gen
        self.risk_manager = risk_manager
        self.assumed_spread = config.get('ASSUMED_SPREAD', 0.0001)
        self.slippage = config.get('SLIPPAGE', 0.0003)
        self.commission_per_share = config.get('COMMISSION_PER_SHARE', 0.0005)
        self.debug_mode = config.get('BACKTEST_DEBUG', False)
    def _compute_current_atr(self, data_window: pd.DataFrame, lookback: int = 50) -> float:
        """Proper classic True Range + EMA(14) ATR — pandas only, no annualization."""
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
    def run_backtest(self, start_date: str = None, end_date: str = None) -> dict:
        print("*** DEBUG MARKER: RUNNING VERSION WITH UNCONDITIONAL MIN-HOLD LOGS + REALISM FRICTION + CONSOLIDATED REGIME v2026-02-17 ***")
        logger.info("*** DEBUG MARKER: RUNNING VERSION WITH UNCONDITIONAL MIN-HOLD LOGS + REALISM FRICTION + CONSOLIDATED REGIME v2026-02-17 ***")
        logger.info("=== USING UPDATED BACKTEST.PY WITH FULL MIN-HOLD DEBUG + SLIPPAGE/COMMISSION/SPREAD + CONSOLIDATED REGIME DETECTION + is_backtest FLAG ===")
        logger.info(f"CONFIG CHECK: BACKTEST_DEBUG={self.debug_mode}, "
                    f"MIN_HOLD_BARS={self.config.get('MIN_HOLD_BARS')}, "
                    f"TRAILING_STOP_ATR_TRENDING={self.config.get('TRAILING_STOP_ATR_TRENDING')}, "
                    f"COMMISSION_PER_SHARE={self.commission_per_share}, "
                    f"SLIPPAGE_RANGE=0.0002-0.0005, SPREAD={self.assumed_spread}")
        if start_date is None or end_date is None:
            end_dt = datetime.now(tz=tz.gettz('UTC'))
            start_dt = end_dt - timedelta(days=180)
        else:
            start_dt = pd.to_datetime(start_date).tz_localize('UTC')
            end_dt = pd.to_datetime(end_date).tz_localize('UTC')
        symbols_config = self.config['SYMBOLS']
        if not symbols_config:
            logger.warning("No symbols configured for backtest")
            return {'stats': {}, 'per_symbol': {}, 'equity_curve': pd.DataFrame()}
        initial_equity = 100000.0
        cash = initial_equity
        trade_records = []
        equity_curve = [(start_dt, initial_equity)]
        total_signals = 0
        # === FETCH ALL DATA ONCE ===
        full_dfs_15m = {}
        full_dfs_1h = {}
        price_series = []
        for symbol in symbols_config:
            buffered_start = start_dt - timedelta(days=60)
            data_15m = self.data_ingestion.data_handler.fetch_data(
                symbol, '15Min', buffered_start, end_dt
            )
            if data_15m.empty or len(data_15m) < 300:
                logger.warning(f"Insufficient 15Min data for {symbol} — skipping in backtest")
                continue
            data_15m = data_15m[(data_15m.index >= start_dt) & (data_15m.index <= end_dt)]
            full_dfs_15m[symbol] = data_15m
            data_1h = self.data_ingestion.data_handler.fetch_data(
                symbol, '1H', buffered_start, end_dt
            )
            if data_1h.empty:
                logger.debug(f"No 1H data fetched for {symbol} — will fallback to 15Min for regime")
                data_1h = pd.DataFrame()
            else:
                data_1h = data_1h[(data_1h.index >= start_dt) & (data_1h.index <= end_dt)]
            full_dfs_1h[symbol] = data_1h
            price_series.append(data_15m['close'].rename(symbol))
        if not price_series:
            logger.error("No symbols had sufficient data for backtest")
            return {'stats': {}, 'per_symbol': {}, 'equity_curve': pd.DataFrame()}
        close_prices = pd.concat(price_series, axis=1, join='outer')
        close_prices = close_prices.ffill().bfill().dropna(how='all')
        common_index = close_prices.index
        if len(common_index) == 0:
            logger.error("No common data found across symbols for backtest")
            return {'stats': {}, 'per_symbol': {}, 'equity_curve': pd.DataFrame()}
        logger.info(f"Common index length after fill: {len(common_index)} bars from {common_index[0]} to {common_index[-1]}")
        # Reset signal generator state
        self.signal_gen.prev_signals = {}
        self.signal_gen.last_entry_time = {}
        self.signal_gen.last_exit_time = {}
        self.signal_gen.meta_ema = {}
        self.signal_gen.lstm_states = {}
        daily_equity = {}
        equity_history = {}
        current_day = None
        last_logged_day = None
        last_pause_state = False
        # === PORTFOLIO PPO MODE ===
        if self.config.get('PORTFOLIO_PPO', False):
            logger.info("Running backtest in PORTFOLIO PPO mode")
            positions = {sym: {'signed_qty': 0, 'entry_time': None, 'stop_price': None, 'tp_price': None, 'regime': None} for sym in symbols_config}
            entry_prices = {sym: 0.0 for sym in symbols_config}
            entry_costs = {sym: 0.0 for sym in symbols_config}
            # Precompute PortfolioEnv ONCE at the start (fixes flooding)
            precomputed_env = PortfolioEnv(
                data_dict=full_dfs_15m,
                symbols=symbols_config,
                initial_balance=initial_equity,
                max_leverage=self.config.get('MAX_LEVERAGE', 3.0)
            )
            logger.info("PortfolioEnv precomputed once at start — flooding fixed")
            for i in range(200, len(common_index)):
                timestamp = common_index[i]
                current_day = timestamp.date()
                # Build current data dict (sliced to this exact timestamp) — ROBUST SLICING FOR SHORT-HISTORY SYMBOLS
                data_dict = {}
                current_prices = {}
                for sym in symbols_config:
                    if sym not in full_dfs_15m:
                        continue
                    df = full_dfs_15m[sym]
                    window = df[df.index <= timestamp].copy()
                    if window.empty:
                        window = df.iloc[-1:].copy() if not df.empty else pd.DataFrame()
                        logger.debug(f"STALE DATA FALLBACK for {sym} at {timestamp} — using last available bar {window.index[-1] if not window.empty else 'N/A'}")
                    if len(window) >= 200:
                        data_dict[sym] = window
                        current_prices[sym] = window['close'].iloc[-1]
                    elif len(window) >= 50:
                        data_dict[sym] = window
                        current_prices[sym] = window['close'].iloc[-1]
                        logger.debug(f"Short window for {sym} at {timestamp} (len={len(window)})")
                    else:
                        logger.debug(f"Skipping {sym} at {timestamp} — window too short (<50 bars)")
                        continue
                if len(data_dict) < len(symbols_config):
                    logger.debug(f"Skipping bar {timestamp} — insufficient data for all symbols")
                    continue
                pre_trade_equity = cash + sum(
                    positions.get(sym, {}).get('signed_qty', 0) * current_prices.get(sym, 0)
                    for sym in symbols_config
                )
                # Prevent NaN/inf explosion
                if not np.isfinite(pre_trade_equity):
                    logger.error(f"Non-finite pre_trade_equity at {timestamp}: {pre_trade_equity} — clamping and resetting cash")
                    pre_trade_equity = 0.0
                    cash = 0.0
                equity_history[timestamp.date()] = pre_trade_equity
                if current_day not in daily_equity:
                    daily_equity[current_day] = pre_trade_equity
                pause = self.risk_manager.check_pause_conditions(
                    equity=pre_trade_equity,
                    daily_equity=daily_equity,
                    equity_history=equity_history,
                    is_backtest=True
                )
                if self.debug_mode and (pause != last_pause_state or current_day != last_logged_day):
                    logger.info(f"[PAUSE CHECK] pause={pause} at {timestamp.date()}")
                    last_pause_state = pause
                    last_logged_day = current_day
                if pause:
                    equity_curve.append((timestamp, pre_trade_equity))
                    continue
                # Generate portfolio actions
                try:
                    loop = asyncio.get_running_loop()
                    # Already in async context — use nest_asyncio or thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        target_weights = pool.submit(
                            asyncio.run,
                            self.signal_gen.generate_portfolio_actions(
                                symbols=symbols_config,
                                data_dict=data_dict,
                                current_equity=pre_trade_equity,
                                precomputed_env=precomputed_env,
                                timestamp=timestamp
                            )
                        ).result()
                except RuntimeError:
                    # No running loop — safe to use run_until_complete
                    target_weights = asyncio.get_event_loop().run_until_complete(
                        self.signal_gen.generate_portfolio_actions(
                            symbols=symbols_config,
                            data_dict=data_dict,
                            current_equity=pre_trade_equity,
                            precomputed_env=precomputed_env,
                            timestamp=timestamp
                        )
                    )
                # Regime detection
                regimes = {}
                for sym in symbols_config:
                    regime_data = full_dfs_1h.get(sym, pd.DataFrame())
                    if len(regime_data) > 0:
                        regime_data = regime_data.sort_index()
                        regime_data = regime_data[~regime_data.index.duplicated(keep='last')]
                        regime_window = regime_data[regime_data.index <= timestamp]
                        if len(regime_window) == 0:
                            regime_window = regime_data.iloc[-1:]
                    else:
                        regime_window = data_dict[sym]
                    if len(regime_window) >= 50:
                        regime_result = detect_regime(
                            data=regime_window,
                            symbol=sym,
                            data_ingestion=self.data_ingestion,
                            lookback=CONFIG.get('LOOKBACK', 900),
                            is_backtest=True
                        )
                        regimes[sym] = regime_result[0] if isinstance(regime_result, (list, tuple)) else regime_result
                        actual_last_bar = regime_window.index[-1] if not regime_window.empty else "No data"
                        logger.info(f"Regime for {sym} at {timestamp} | using last_bar={actual_last_bar}")
                    else:
                        regimes[sym] = 'mean_reverting'
                # Enforce GLOBAL leverage cap
                abs_sum_weights = sum(abs(w) for w in target_weights.values())
                max_leverage = self.config.get('MAX_LEVERAGE', 3.0)
                if abs_sum_weights > max_leverage:
                    scale = max_leverage / abs_sum_weights
                    target_weights = {s: w * scale for s, w in target_weights.items()}
                    logger.debug(f"Global leverage capped: abs_sum={abs_sum_weights:.2f} > {max_leverage} → scaled by {scale:.3f}")
                # Rebalance logic
                for sym in symbols_config:
                    target_weight = target_weights.get(sym, 0.0)
                    price = current_prices.get(sym)
                    if price is None or price <= 0:
                        continue
                    target_qty_raw = (target_weight * pre_trade_equity) / price if price > 0 else 0.0
                    if not np.isfinite(target_qty_raw):
                        logger.error(f"Non-finite target_qty_raw for {sym} at {timestamp}: {target_qty_raw} — skipping")
                        continue
                    max_abs_qty = 10000
                    target_qty_raw = np.clip(target_qty_raw, -max_abs_qty, max_abs_qty)
                    max_notional = pre_trade_equity * 0.25
                    max_shares = int(max_notional / price) if price > 0 else 0
                    target_qty_raw = np.clip(target_qty_raw, -max_shares, max_shares)
                    target_qty = round(target_qty_raw)
                    current_qty = positions.get(sym, {}).get('signed_qty', 0)
                    direction = 1 if target_qty > 0 else (-1 if target_qty < 0 else 0)
                    target_qty = abs(target_qty) * direction
                    if (current_qty * direction <= 0 and current_qty != 0) or target_qty == 0:
                        if current_qty != 0:
                            slippage_pct = random.uniform(0.0002, 0.0005)
                            spread_half = self.assumed_spread / 2
                            qty_abs = abs(current_qty)
                            effective_exit_price = price * (1 - np.sign(current_qty) * (slippage_pct + spread_half))
                            commission = qty_abs * self.commission_per_share
                            if current_qty > 0:  # Long close: sell
                                gross_pnl = current_qty * (effective_exit_price - entry_prices[sym])
                                cash += qty_abs * effective_exit_price - commission
                            else:  # Short close: buy back
                                gross_pnl = -current_qty * (entry_prices[sym] - effective_exit_price)  # Positive if entry > exit
                                cash -= qty_abs * effective_exit_price + commission
                            net_pnl = gross_pnl - entry_costs.get(sym, 0.0) - commission
                            trade_records.append({
                                'symbol': sym,
                                'exit_time': timestamp,
                                'exit_price': effective_exit_price,
                                'gross_pnl': gross_pnl,
                                'pnl': net_pnl,
                                'exit_reason': 'REBALANCE'
                            })
                            if self.debug_mode:
                                logger.info(f"PORTFOLIO REBALANCE CLOSE {sym} qty={current_qty} pnl={net_pnl:+.1f}")
                            positions[sym]['signed_qty'] = 0
                            entry_prices[sym] = 0.0
                            entry_costs[sym] = 0.0
                    if target_qty != 0 and current_qty * direction <= 0 and abs(target_qty) >= 1:
                        size = abs(target_qty)
                        regime = regimes.get(sym, 'mean_reverting')
                        slippage_pct = random.uniform(0.0002, 0.0005)
                        spread_half = self.assumed_spread / 2
                        effective_entry_price = price * (1 + direction * (slippage_pct + spread_half))
                        commission = size * self.commission_per_share
                        entry_cost = size * effective_entry_price * (slippage_pct + spread_half) + commission
                        required_cash = size * effective_entry_price + commission if direction > 0 else 0  # Shorts don't deduct cash on entry
                        if cash < required_cash and direction > 0:
                            logger.warning(f"Insufficient cash for {sym} LONG entry: needed {required_cash:.2f}, have {cash:.2f} — skipping")
                            continue
                        if direction > 0:  # Long entry: buy
                            cash -= size * effective_entry_price + commission
                        else:  # Short entry: sell
                            cash += size * effective_entry_price - commission
                        if cash < 0:
                            logger.warning(f"Cash negative after entry {sym}: {cash:.2f} — clamping to 0")
                            cash = 0.0
                        atr = self._compute_current_atr(data_dict.get(sym))
                        trailing_mult = self.config.get(
                            f'TRAILING_STOP_ATR_{regime.upper()}',
                            self.config['TRAILING_STOP_ATR_TRENDING'] if regime == 'trending' else self.config.get('TRAILING_STOP_ATR_MEAN_REVERTING', self.config.get('TRAILING_STOP_ATR', 4.0))
                        )
                        tp_mult = self.config.get(
                            f'TAKE_PROFIT_ATR_{regime.upper()}',
                            self.config['TAKE_PROFIT_ATR'] * (1.2 if regime == 'trending' else 1.0)
                        )
                        stop_price = effective_entry_price - direction * trailing_mult * atr
                        if direction > 0:
                            stop_price = max(stop_price, effective_entry_price * 0.65)
                        else:
                            stop_price = min(stop_price, effective_entry_price * 1.35)
                        tp_distance = tp_mult * atr
                        tp_price = effective_entry_price + direction * tp_distance
                        positions[sym] = {
                            'signed_qty': target_qty,
                            'entry_time': timestamp,
                            'entry_price': effective_entry_price,
                            'stop_price': stop_price,
                            'tp_price': tp_price,
                            'entry_cost': entry_cost,
                            'regime': regime
                        }
                        entry_prices[sym] = effective_entry_price
                        entry_costs[sym] = entry_cost
                        trade_records.append({
                            'symbol': sym,
                            'entry_time': timestamp,
                            'entry_price': effective_entry_price,
                            'signed_qty': target_qty,
                            'direction': direction,
                            'regime': regime
                        })
                        if self.debug_mode:
                            logger.info(f"PORTFOLIO REBALANCE ENTRY {sym} {'LONG' if direction>0 else 'SHORT'} size={size} @ {effective_entry_price:.2f} regime={regime}")
                # === SIMULATED POSITION UPDATES / MONITORING (ADDED FOR CONSISTENCY WITH LIVE) ===
                for sym in list(positions.keys()):
                    pos = positions[sym]
                    signed_qty = pos['signed_qty']
                    if signed_qty == 0:
                        continue
                    quoted_price = current_prices.get(sym)
                    if quoted_price is None or quoted_price <= 0:
                        continue
                    direction = np.sign(signed_qty)
                    regime = pos['regime']
                    data_window = data_dict.get(sym)
                    if data_window is None or len(data_window) < 14:
                        continue
                    atr = self._compute_current_atr(data_window)
                    trailing_mult = self.config.get(
                        f'TRAILING_STOP_ATR_{regime.upper()}',
                        self.config['TRAILING_STOP_ATR_TRENDING'] if regime == 'trending' else self.config.get('TRAILING_STOP_ATR_MEAN_REVERTING', self.config.get('TRAILING_STOP_ATR', 4.0))
                    )
                    new_stop = quoted_price - direction * trailing_mult * atr
                    if direction > 0:
                        new_stop = max(new_stop, quoted_price * 0.65)
                    else:
                        new_stop = min(new_stop, quoted_price * 1.35)
                    if self.debug_mode:
                        distance_pct = (pos['stop_price'] - quoted_price) / quoted_price * 100 if direction > 0 else (quoted_price - pos['stop_price']) / quoted_price * 100
                        logger.info(f"TRAILING STOP {sym} | current_price={quoted_price:.2f} | stop={pos['stop_price']:.2f} | "
                                    f"distance={distance_pct:.2f}% | atr={atr:.4f} | mult={trailing_mult}")
                    if ((direction > 0 and new_stop > pos['stop_price']) or
                        (direction < 0 and new_stop < pos['stop_price'])):
                        old_stop = pos['stop_price']
                        pos['stop_price'] = new_stop
                        if self.debug_mode:
                            logger.info(f"Stop ratchet {sym} → {new_stop:.2f} (improved from {old_stop:.2f})")
                    if regime != 'trending':
                        tp_mult = self.config.get(f'TAKE_PROFIT_ATR_{regime.upper()}', self.config['TAKE_PROFIT_ATR'])
                        new_tp = quoted_price + direction * tp_mult * atr
                        if ((direction > 0 and new_tp > pos['tp_price'] * 1.003) or
                            (direction < 0 and new_tp < pos['tp_price'] * 0.997)):
                            pos['tp_price'] = new_tp
                    tp_hit = (direction > 0 and quoted_price >= pos['tp_price']) or (direction < 0 and quoted_price <= pos['tp_price'])
                    sl_hit = (direction > 0 and quoted_price <= pos['stop_price']) or (direction < 0 and quoted_price >= pos['stop_price'])
                    reversal = False
                    flat = False
                    last_entry = pos['entry_time']
                    bars_held = ((timestamp - last_entry) / pd.Timedelta(minutes=15)) if last_entry else 9999
                    min_hold_bars = self.config.get('MIN_HOLD_BARS_TRENDING' if regime == 'trending' else 'MIN_HOLD_BARS_MEAN_REVERTING', self.config.get('MIN_HOLD_BARS', 6))
                    if self.debug_mode:
                        delta_min = (timestamp - last_entry).total_seconds() / 60 if last_entry else None
                        logger.info(f"MIN-HOLD CHECK {sym} | last_entry={last_entry} | current={timestamp} | "
                                    f"delta_min={delta_min} | bars_held={int(bars_held)} | min_hold={min_hold_bars}")
                    if bars_held < min_hold_bars:
                        logger.info(f"MIN-HOLD ACTIVE {sym} | bars_held={int(bars_held)} < {min_hold_bars} → skipping ALL exits")
                        continue
                    exit_reason = None
                    if tp_hit:
                        exit_reason = 'TP'
                    elif sl_hit:
                        exit_reason = 'SL'
                    if exit_reason:
                        slippage_pct = random.uniform(0.0002, 0.0005)
                        spread_half = self.assumed_spread / 2
                        qty_abs = abs(signed_qty)
                        effective_exit_price = quoted_price * (1 - direction * (slippage_pct + spread_half))
                        commission = qty_abs * self.commission_per_share
                        if direction > 0:  # Long close: sell
                            gross_pnl = signed_qty * (effective_exit_price - pos['entry_price'])
                            cash += qty_abs * effective_exit_price - commission
                        else:  # Short close: buy back
                            gross_pnl = qty_abs * (pos['entry_price'] - effective_exit_price)  # Positive if entry > exit
                            cash -= qty_abs * effective_exit_price + commission
                        net_pnl = gross_pnl - pos.get('entry_cost', 0.0) - commission
                        if cash < 0:
                            logger.warning(f"Cash negative after exit {sym}: {cash:.2f} — clamping to 0")
                            cash = 0.0
                        if self.debug_mode:
                            logger.info(f"EXIT FRICTION {sym} | slippage={slippage_pct:.4f}, spread_half={spread_half:.4f}, "
                                        f"commission={commission:.2f}, effective_price={effective_exit_price:.4f}, net_pnl={net_pnl:+.2f}")
                        for rec in reversed(trade_records):
                            if rec['symbol'] == sym and 'exit_time' not in rec:
                                rec['exit_time'] = timestamp
                                rec['exit_price'] = effective_exit_price
                                rec['gross_pnl'] = gross_pnl
                                rec['pnl'] = net_pnl
                                rec['exit_reason'] = exit_reason
                                break
                        self.signal_gen.last_exit_time[sym] = timestamp
                        positions[sym]['signed_qty'] = 0
                        if self.debug_mode:
                            logger.info(f"EXIT {sym} {exit_reason} bars≈{int(bars_held)} pnl={net_pnl:+.1f} gross={gross_pnl:+.1f}")
                portfolio_value = cash + sum(
                    positions.get(sym, {}).get('signed_qty', 0) * current_prices.get(sym, 0)
                    for sym in symbols_config
                )
                logger.debug(f"Bar {timestamp} | cash={cash:.2f} | portfolio_value={portfolio_value:.2f} | "
                             f"positions sum={sum(p.get('signed_qty', 0) for p in positions.values())}")
                equity_curve.append((timestamp, portfolio_value))
            # Force close remaining positions at end — use final index timestamp
            timestamp = common_index[-1]
            current_prices = close_prices.loc[timestamp] if timestamp in close_prices.index else close_prices.iloc[-1]
            for sym in list(positions.keys()):
                if positions.get(sym, {}).get('signed_qty', 0) != 0 and sym in current_prices:
                    price = current_prices[sym]
                    pos = positions[sym]
                    signed_qty = pos.get('signed_qty', 0)
                    qty_abs = abs(signed_qty)
                    slippage_pct = random.uniform(0.0002, 0.0005)
                    spread_half = self.assumed_spread / 2
                    direction = np.sign(signed_qty)
                    effective_exit_price = price * (1 - direction * (slippage_pct + spread_half))
                    commission = qty_abs * self.commission_per_share
                    if direction > 0:  # Long close: sell
                        gross_pnl = signed_qty * (effective_exit_price - pos['entry_price'])
                        cash += qty_abs * effective_exit_price - commission
                    else:  # Short close: buy back
                        gross_pnl = qty_abs * (pos['entry_price'] - effective_exit_price)
                        cash -= qty_abs * effective_exit_price + commission
                    net_pnl = gross_pnl - pos.get('entry_cost', 0.0) - commission
                    if cash < 0:
                        logger.warning(f"Cash negative after END_OF_DATA close {sym}: {cash:.2f} — clamping to 0")
                        cash = 0.0
                    trade_records.append({
                        'symbol': sym,
                        'exit_time': timestamp,
                        'exit_price': effective_exit_price,
                        'gross_pnl': gross_pnl,
                        'pnl': net_pnl,
                        'exit_reason': 'END_OF_DATA'
                    })
                    logger.info(f"END_OF_DATA close {sym} pnl={net_pnl:+.1f}")
                    logger.info(f"END_OF_DATA {sym} | entry_price={pos['entry_price']:.2f} | exit_price={effective_exit_price:.2f} | gross_pnl={gross_pnl:+.1f} | costs={commission + pos.get('entry_cost', 0.0):.1f}")
                    positions[sym]['signed_qty'] = 0
        # === PER-SYMBOL MODE (original logic preserved unchanged) ===
        else:
            positions = {}
            signals = {}
            confidences = {}
            for i, timestamp in enumerate(common_index):
                current_prices = close_prices.loc[timestamp]
                day = timestamp.date()
                if day != current_day:
                    current_day = day
                    daily_equity[day] = cash + sum(
                        pos.get('signed_qty', 0) * current_prices.get(sym, 0)
                        for sym, pos in positions.items()
                    )
                pre_trade_equity = cash + sum(
                    pos.get('signed_qty', 0) * current_prices.get(sym, 0)
                    for sym, pos in positions.items()
                )
                equity_history[timestamp.date()] = pre_trade_equity
                # Clear stale signals from prior bar — prevents re-trading old signals
                signals = {}
                confidences = {}
                if i > 0:
                    prev_timestamp = common_index[i - 1]
                    for sym in full_dfs_15m:
                        signal_data_window = full_dfs_15m[sym].loc[:prev_timestamp]
                        if len(signal_data_window) < 200:
                            continue
                        regime_data_window = full_dfs_1h.get(sym, pd.DataFrame()).loc[:prev_timestamp]
                        if len(regime_data_window) < 50:
                            regime_data_window = signal_data_window
                        regime_result = detect_regime(
                            data=regime_data_window,
                            symbol=sym,
                            data_ingestion=self.data_ingestion,
                            lookback=CONFIG.get('LOOKBACK', 900),
                            is_backtest=True
                        )
                        regime = regime_result[0] if isinstance(regime_result, (list, tuple)) else regime_result
                        try:
                            signal_output = self.signal_gen.generate_signal_sync(
                                symbol=sym,
                                timestamp=prev_timestamp,
                                data=signal_data_window,
                                live_mode=False,
                                full_hist_df=full_dfs_15m.get(sym)
                            )
                            direction, confidence, ppo_strength = signal_output[:3]
                            if direction != 0:
                                total_signals += 1
                                signals[sym] = {
                                    'direction': direction,
                                    'confidence': confidence,
                                    'ppo_strength': ppo_strength,
                                    'regime': regime
                                }
                                confidences[sym] = confidence
                        except Exception as e:
                            logger.debug(f"Signal error {sym} {prev_timestamp}: {e}")
                pause = self.risk_manager.check_pause_conditions(
                    equity=pre_trade_equity,
                    daily_equity=daily_equity,
                    equity_history=equity_history,
                    is_backtest=True
                )
                if self.debug_mode and (pause != last_pause_state or day != last_logged_day):
                    logger.info(f"[PAUSE CHECK] pause={pause} at {day}")
                    last_pause_state = pause
                    last_logged_day = day
                alloc_dollars = {}
                if signals and not pause:
                    signal_symbols = list(signals.keys())
                    signal_confidences = [confidences[s] for s in signal_symbols]
                    try:
                        alloc_dollars = self.risk_manager.allocate_portfolio_risk(
                            pre_trade_equity, signal_symbols, signal_confidences
                        )
                    except Exception as e:
                        logger.warning(f"CVaR failed → equal risk parity ({e})")
                        risk_per = (pre_trade_equity * self.config['RISK_PER_TRADE']) / len(signal_symbols)
                        alloc_dollars = {s: risk_per for s in signal_symbols}
                    for sym, sig in signals.items():
                        direction = sig['direction']
                        confidence = sig['confidence']
                        regime = sig['regime']
                        dollar_risk = alloc_dollars.get(sym, 0)
                        if self.debug_mode:
                            logger.info(f"[ALLOC] {sym} | dollar_risk={dollar_risk:.2f}")
                        if dollar_risk <= 0:
                            continue
                        dollar_risk *= self.config.get(f'RISK_MULT_{regime.upper()}', 1.0)
                        price = current_prices[sym]
                        data_window = full_dfs_15m[sym].loc[:timestamp]
                        size = self.risk_manager.calculate_position_size(
                            equity=pre_trade_equity,
                            price=price,
                            symbol=sym,
                            data=data_window,
                            risk_amount=dollar_risk,
                            conviction=confidence,
                            regime=regime
                        )
                        signed_size = int(size * direction)
                        if signed_size == 0 and size > 0.01:
                            signed_size = direction if direction > 0 else -1
                            logger.info(f"[FORCE MIN SIZE] {sym} - forced to 1 share (raw size {size:.4f}, price {price:.2f})")
                        if self.debug_mode:
                            risk_distance = self._compute_current_atr(data_window) * self.config.get(
                                f'TRAILING_STOP_ATR_{regime.upper()}',
                                self.config['TRAILING_STOP_ATR_TRENDING'] if regime == 'trending' else self.config.get('TRAILING_STOP_ATR_MEAN_REVERTING', self.config.get('TRAILING_STOP_ATR', 4.0))
                            )
                            logger.info(f"[DEBUG ENTRY] {sym} | dollar_risk={dollar_risk:.2f} | atr={self._compute_current_atr(data_window):.4f} | mult={self.config.get(f'TRAILING_STOP_ATR_{regime.upper()}', self.config['TRAILING_STOP_ATR_TRENDING'] if regime == 'trending' else self.config.get('TRAILING_STOP_ATR_MEAN_REVERTING', self.config.get('TRAILING_STOP_ATR', 4.0)))} | risk_distance={risk_distance:.4f} | size_raw={size} | signed_size={signed_size}")
                        if signed_size == 0:
                            if self.debug_mode:
                                logger.info(f"[SKIP ENTRY] {sym} — size=0 (likely ATR too large or risk_amount too small)")
                            continue
                        if sym in positions:
                            continue
                        last_exit = self.signal_gen.last_exit_time.get(sym)
                        if last_exit == timestamp:
                            if self.debug_mode:
                                logger.info(f"Re-entry blocked for {sym} — exited this bar")
                            continue
                        min_hold_bars = self.config.get('MIN_HOLD_BARS_TRENDING' if regime == 'trending' else 'MIN_HOLD_BARS_MEAN_REVERTING', self.config.get('MIN_HOLD_BARS', 6))
                        if last_exit and (timestamp - last_exit) < timedelta(minutes=15 * min_hold_bars):
                            continue
                        atr = self._compute_current_atr(data_window)
                        trailing_mult = self.config.get(
                            f'TRAILING_STOP_ATR_{regime.upper()}',
                            self.config['TRAILING_STOP_ATR_TRENDING'] if regime == 'trending' else self.config.get('TRAILING_STOP_ATR_MEAN_REVERTING', self.config.get('TRAILING_STOP_ATR', 4.0))
                        )
                        tp_mult = self.config.get(
                            f'TAKE_PROFIT_ATR_{regime.upper()}',
                            self.config['TAKE_PROFIT_ATR'] * (1.2 if regime == 'trending' else 1.0)
                        )
                        stop_price = price - direction * trailing_mult * atr
                        if direction > 0:
                            stop_price = max(stop_price, price * 0.65)
                        else:
                            stop_price = min(stop_price, price * 1.35)
                        tp_distance = tp_mult * atr
                        tp_price = price + direction * tp_distance
                        positions[sym] = {
                            'signed_qty': signed_size,
                            'entry_price': price,
                            'stop_price': stop_price,
                            'tp_price': tp_price,
                            'entry_cost': 0.0,
                            'regime': regime
                        }
                        slippage_pct = random.uniform(0.0002, 0.0005)
                        spread_half = self.assumed_spread / 2
                        effective_entry_price = price * (1 + direction * (slippage_pct + spread_half))
                        commission = abs(signed_size) * self.commission_per_share
                        entry_cost = abs(signed_size) * effective_entry_price * (slippage_pct + spread_half) + commission
                        if direction > 0:
                            cash -= abs(signed_size) * effective_entry_price + commission
                        else:
                            cash += abs(signed_size) * effective_entry_price - commission
                        positions[sym]['entry_cost'] = entry_cost
                        positions[sym]['entry_price'] = effective_entry_price
                        if self.debug_mode:
                            logger.info(f"ENTRY FRICTION {sym} | slippage={slippage_pct:.4f}, spread_half={spread_half:.4f}, "
                                        f"commission={commission:.2f}, entry_cost={entry_cost:.2f}, "
                                        f"effective_price={effective_entry_price:.4f}")
                        self.signal_gen.last_entry_time[sym] = timestamp
                        if self.debug_mode:
                            logger.info(f"ENTRY {sym} {'LONG' if direction>0 else 'SHORT'} size={signed_size} "
                                        f"@ effective {effective_entry_price:.4f} (quoted {price:.2f}) atr={atr:.4f} regime={regime} timestamp={timestamp}")
                        trade_records.append({
                            'symbol': sym,
                            'entry_time': timestamp,
                            'entry_price': effective_entry_price,
                            'signed_qty': signed_size,
                            'direction': direction
                        })
                for sym in list(positions.keys()):
                    if sym not in current_prices:
                        continue
                    pos = positions[sym]
                    quoted_price = current_prices[sym]
                    direction = np.sign(pos['signed_qty'])
                    regime = pos['regime']
                    atr = self._compute_current_atr(full_dfs_15m[sym].loc[:timestamp])
                    trailing_mult = self.config.get(
                        f'TRAILING_STOP_ATR_{regime.upper()}',
                        self.config['TRAILING_STOP_ATR_TRENDING'] if regime == 'trending' else self.config.get('TRAILING_STOP_ATR_MEAN_REVERTING', self.config.get('TRAILING_STOP_ATR', 4.0))
                    )
                    new_stop = quoted_price - direction * trailing_mult * atr
                    if direction > 0:
                        new_stop = max(new_stop, quoted_price * 0.65)
                    else:
                        new_stop = min(new_stop, quoted_price * 1.35)
                    if self.debug_mode:
                        distance_pct = (pos['stop_price'] - quoted_price) / quoted_price * 100 if direction > 0 else (quoted_price - pos['stop_price']) / quoted_price * 100
                        logger.info(f"TRAILING STOP {sym} | current_price={quoted_price:.2f} | stop={pos['stop_price']:.2f} | "
                                    f"distance={distance_pct:.2f}% | atr={atr:.4f} | mult={trailing_mult}")
                    if ((direction > 0 and new_stop > pos['stop_price']) or
                        (direction < 0 and new_stop < pos['stop_price'])):
                        old_stop = pos['stop_price']
                        pos['stop_price'] = new_stop
                        if self.debug_mode:
                            logger.info(f"Stop ratchet {sym} → {new_stop:.2f} (improved from {old_stop:.2f})")
                    if regime != 'trending':
                        tp_mult = self.config.get(f'TAKE_PROFIT_ATR_{regime.upper()}', self.config['TAKE_PROFIT_ATR'])
                        new_tp = quoted_price + direction * tp_mult * atr
                        if ((direction > 0 and new_tp > pos['tp_price'] * 1.003) or
                            (direction < 0 and new_tp < pos['tp_price'] * 0.997)):
                            pos['tp_price'] = new_tp
                    tp_hit = (direction > 0 and quoted_price >= pos['tp_price']) or (direction < 0 and quoted_price <= pos['tp_price'])
                    sl_hit = (direction > 0 and quoted_price <= pos['stop_price']) or (direction < 0 and quoted_price >= pos['stop_price'])
                    reversal = sym in signals and signals[sym]['direction'] == -direction and signals[sym]['confidence'] > 0.85
                    flat = sym in signals and signals[sym]['direction'] == 0 and signals[sym]['confidence'] > 0.75
                    last_entry = self.signal_gen.last_entry_time.get(sym)
                    bars_held = ((timestamp - last_entry) / pd.Timedelta(minutes=15)) if last_entry else 9999
                    min_hold_bars = self.config.get('MIN_HOLD_BARS_TRENDING' if regime == 'trending' else 'MIN_HOLD_BARS_MEAN_REVERTING', self.config.get('MIN_HOLD_BARS', 6))
                    if self.debug_mode:
                        delta_min = (timestamp - last_entry).total_seconds() / 60 if last_entry else None
                        logger.info(f"MIN-HOLD CHECK {sym} | last_entry={last_entry} | current={timestamp} | "
                                    f"delta_min={delta_min} | bars_held={int(bars_held)} | min_hold={min_hold_bars}")
                    if bars_held < min_hold_bars:
                        logger.info(f"MIN-HOLD ACTIVE {sym} | bars_held={int(bars_held)} < {min_hold_bars} → skipping ALL exits")
                        continue
                    exit_reason = None
                    if tp_hit:
                        exit_reason = 'TP'
                    elif sl_hit:
                        exit_reason = 'SL'
                    elif reversal:
                        exit_reason = 'REVERSAL'
                    elif flat:
                        exit_reason = 'FLAT'
                    if exit_reason:
                        signed_qty = pos['signed_qty']
                        slippage_pct = random.uniform(0.0002, 0.0005)
                        spread_half = self.assumed_spread / 2
                        qty_abs = abs(signed_qty)
                        effective_exit_price = quoted_price * (1 - direction * (slippage_pct + spread_half))
                        commission = qty_abs * self.commission_per_share
                        if direction > 0:  # Long close: sell
                            gross_pnl = signed_qty * (effective_exit_price - pos['entry_price'])
                            cash += qty_abs * effective_exit_price - commission
                        else:  # Short close: buy back
                            gross_pnl = qty_abs * (pos['entry_price'] - effective_exit_price)
                            cash -= qty_abs * effective_exit_price + commission
                        net_pnl = gross_pnl - pos.get('entry_cost', 0.0) - commission
                        if cash < 0:
                            logger.warning(f"Cash negative after exit {sym}: {cash:.2f} — clamping to 0")
                            cash = 0.0
                        if self.debug_mode:
                            logger.info(f"EXIT FRICTION {sym} | slippage={slippage_pct:.4f}, spread_half={spread_half:.4f}, "
                                        f"commission={commission:.2f}, effective_price={effective_exit_price:.4f}, net_pnl={net_pnl:+.2f}")
                        for rec in reversed(trade_records):
                            if rec['symbol'] == sym and 'exit_time' not in rec:
                                rec['exit_time'] = timestamp
                                rec['exit_price'] = effective_exit_price
                                rec['gross_pnl'] = gross_pnl
                                rec['pnl'] = net_pnl
                                rec['exit_reason'] = exit_reason
                                break
                        self.signal_gen.last_exit_time[sym] = timestamp
                        del positions[sym]
                        if self.debug_mode:
                            logger.info(f"EXIT {sym} {exit_reason} bars≈{int(bars_held)} pnl={net_pnl:+.1f} gross={gross_pnl:+.1f}")
                portfolio_value = cash + sum(
                    pos.get('signed_qty', 0) * current_prices.get(sym, 0)
                    for sym, pos in positions.items()
                )
                equity_curve.append((timestamp, portfolio_value))
            # Force close remaining positions at end — use final index timestamp
            timestamp = common_index[-1]
            # Rebuild current_prices from final bar (loop variable may be stale from continue)
            current_prices = {}
            for sym in symbols_config:
                if sym in full_dfs_15m and not full_dfs_15m[sym].empty:
                    current_prices[sym] = full_dfs_15m[sym]['close'].iloc[-1]
            for sym in list(positions.keys()):
                if sym in current_prices:
                    quoted_price = current_prices[sym]
                    pos = positions[sym]
                    signed_qty = pos.get('signed_qty', 0)
                    qty_abs = abs(signed_qty)
                    slippage_pct = random.uniform(0.0002, 0.0005)
                    spread_half = self.assumed_spread / 2
                    direction = np.sign(signed_qty)
                    effective_exit_price = quoted_price * (1 - direction * (slippage_pct + spread_half))
                    commission = qty_abs * self.commission_per_share
                    if direction > 0:  # Long close: sell
                        gross_pnl = signed_qty * (effective_exit_price - pos['entry_price'])
                        cash += qty_abs * effective_exit_price - commission
                    else:  # Short close: buy back
                        gross_pnl = qty_abs * (pos['entry_price'] - effective_exit_price)
                        cash -= qty_abs * effective_exit_price + commission
                    net_pnl = gross_pnl - pos.get('entry_cost', 0.0) - commission
                    if cash < 0:
                        logger.warning(f"Cash negative after END_OF_DATA close {sym}: {cash:.2f} — clamping to 0")
                        cash = 0.0
                    trade_records.append({
                        'symbol': sym,
                        'exit_time': timestamp,
                        'exit_price': effective_exit_price,
                        'gross_pnl': gross_pnl,
                        'pnl': net_pnl,
                        'exit_reason': 'END_OF_DATA'
                    })
                    logger.info(f"END_OF_DATA close {sym} pnl={net_pnl:+.1f}")
                    logger.info(f"END_OF_DATA {sym} | entry_price={pos['entry_price']:.2f} | exit_price={effective_exit_price:.2f} | gross_pnl={gross_pnl:+.1f} | costs={commission + pos.get('entry_cost', 0.0):.1f}")
                    del positions[sym]
        final_equity = cash
        if final_equity < 0:
            logger.warning(f"Negative final equity detected: {final_equity:.2f} — clamping to 0")
            final_equity = 0
        equity_curve.append((timestamp, final_equity))
        logger.info(f"Final equity after END_OF_DATA closes: {final_equity:.2f}")
        equity_df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity']).set_index('timestamp')
        total_return = (equity_df['equity'].iloc[-1] / initial_equity - 1) * 100 if len(equity_df) > 0 else 0.0
        returns = equity_df['equity'].pct_change().dropna()
        if len(returns) < 2 or returns.std() == 0 or np.isnan(returns.std()):
            sharpe = 0.0
        else:
            annualization_factor = np.sqrt(252 * 96)
            sharpe = returns.mean() / returns.std() * annualization_factor
        cummax = equity_df['equity'].cummax()
        max_dd = ((cummax - equity_df['equity']) / cummax).max() * 100 if len(cummax) > 0 else 0.0
        trades_df = pd.DataFrame(trade_records)
        total_trades_count = len(trades_df)
        win_rate = (trades_df['pnl'] > 0).mean() * 100 if total_trades_count > 0 else 0.0
        total_net_pnl = trades_df['pnl'].sum() if 'pnl' in trades_df and not trades_df.empty else 0.0
        total_gross_pnl = trades_df['gross_pnl'].sum() if 'gross_pnl' in trades_df and not trades_df.empty else 0.0
        total_costs = trades_df['pnl'].sum() - trades_df['gross_pnl'].sum() if 'pnl' in trades_df and 'gross_pnl' in trades_df else 0.0
        logger.info(f"DEBUG TOTAL PNL: gross={total_gross_pnl:.1f} | net={total_net_pnl:.1f} | costs={total_costs:.1f} | trades={len(trade_records)}")
        logger.info(f"DEBUG Final equity: {equity_df['equity'].iloc[-1]:.2f} | Initial: {initial_equity:.2f} | Return calc: {(equity_df['equity'].iloc[-1] / initial_equity - 1) * 100:.2f}%")
        stats = {
            'Total Return [%]': round(total_return, 2),
            'Sharpe Ratio': round(sharpe, 2),
            'Max Drawdown [%]': round(max_dd, 2),
            'Win Rate [%]': round(win_rate, 2),
            'Total Trades': int(total_trades_count),
            'Signals Generated': total_signals,
            'Final Equity': round(equity_df['equity'].iloc[-1], 2) if len(equity_df) > 0 else initial_equity
        }
        per_symbol_details = {}
        if total_trades_count > 0:
            for symbol in symbols_config:
                sym_trades = trades_df[trades_df['symbol'] == symbol]
                sym_pnl = sym_trades['pnl'].sum() if len(sym_trades) > 0 else 0.0
                sym_return = (sym_pnl / initial_equity) * 100
                sym_count = len(sym_trades)
                sym_win = (sym_trades['pnl'] > 0).mean() * 100 if sym_count > 0 else 0.0
                per_symbol_details[symbol] = {
                    'return_%': round(sym_return, 2),
                    'trades': int(sym_count),
                    'win_rate_%': round(sym_win, 2)
                }
        else:
            for symbol in symbols_config:
                per_symbol_details[symbol] = {'return_%': 0.0, 'trades': 0, 'win_rate_%': 0.0}
        logger.info(f"Backtest complete | Return: {total_return:+.2f}% | Sharpe: {sharpe:.2f} | "
                    f"Max DD: {max_dd:.2f}% | Win Rate: {win_rate:.1f}% | Trades: {total_trades_count}")
        return {'stats': stats, 'per_symbol': per_symbol_details, 'equity_curve': equity_df}

# strategy/regime.py
import numpy as np
import pandas as pd
import logging
import threading
from config import CONFIG
# Suppress hmmlearn warnings
import warnings
warnings.filterwarnings("ignore", message="Model is not converging")
logging.getLogger('hmmlearn.base').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None
    logger.warning("hmmlearn not installed — regime detection will fall back to Hurst exponent permanently")

# FIX #21: Track consecutive divergence signals per symbol to reduce false triggers
_divergence_streak = {}  # {symbol: int} — count of consecutive bars with divergence
_divergence_streak_lock = threading.Lock()  # FIX #15: Protects _divergence_streak from concurrent thread access


# === Regime direction helpers ===
# Regime strings are now: 'trending_up', 'trending_down', 'mean_reverting'
# These helpers let all comparison sites work without knowing the exact strings.
def is_trending(regime: str) -> bool:
    """True for both trending_up and trending_down."""
    return regime is not None and regime.startswith('trending')

def is_bullish(regime: str) -> bool:
    return regime == 'trending_up'

def is_bearish(regime: str) -> bool:
    return regime == 'trending_down'

def reset_divergence_streaks():
    """FIX #14: Clear global divergence streak state between backtest runs to prevent cross-contamination."""
    with _divergence_streak_lock:
        _divergence_streak.clear()

def detect_regime(
    data: pd.DataFrame,
    symbol: str = None,
    data_ingestion=None,
    lookback: int = None,  # HIGH-24 FIX: Don't evaluate CONFIG at import time (was stale)
    is_backtest: bool = False,
    verbose: bool = False  # NEW: Control verbosity
) -> tuple[str, float]:
    """
    Returns (regime: str, persistence_score: float 0.0-1.0)
    persistence_score = average self-transition probability from HMM ensemble
    (higher = more persistent/trending behavior)
    """
    # HIGH-24 FIX: Resolve lookback at call time, not import time
    if lookback is None:
        lookback = CONFIG.get('LOOKBACK', 900)
    # Multi-TF support: prefer 1H when available
    used_tf = '15Min'
    full_data = data
    if symbol is not None and data_ingestion is not None and not is_backtest:
        tf_list = [tf.lower() for tf in CONFIG.get('TIMEFRAMES', [])]
        if any(tf in ['1h', '1hour', '60min'] for tf in tf_list):
            higher_data = data_ingestion.get_latest_data(symbol, timeframe='1H')
            if higher_data is not None and len(higher_data) >= 50:
                full_data = higher_data
                used_tf = '1H'

    prices = full_data['close'].tail(lookback + 100)
    volume = full_data['volume'].tail(lookback + 100)

    if len(prices) < 50:
        if verbose:
            logger.debug(f"Very short series (len={len(prices)}) — using neutral default (mean_reverting, 0.5)")
        return 'mean_reverting', 0.5

    # ISSUE #4 FIX: Removed quick overrides that bypassed HMM ~50% of the time
    # These forced fake persistence scores (0.92 / 0.35) too aggressively
    # Now HMM ensemble runs ALWAYS (unless data truly insufficient)
    # Old overrides now just logged at debug level so we can see how often they would have fired
    recent_window = 50
    recent_return = 0.0
    if len(prices) > recent_window + 1:
        recent_return = prices.pct_change(periods=recent_window).iloc[-1]
        if abs(recent_return) > 0.025:
            logger.debug(f"[REGIME OVERRIDE SKIPPED] {symbol} strong recent return {recent_return:.4f} — would have forced trending 0.92, but using HMM instead")
        if recent_return < -0.005:
            logger.debug(f"[REGIME OVERRIDE SKIPPED] {symbol} weak recent return {recent_return:.4f} — would have forced mean-reverting 0.35, but using HMM instead")

    # Primary: HMM Ensemble — now much more robust
    if GaussianHMM is not None:
        try:
            returns = prices.pct_change().dropna()
            # NEW: Early skip if data is too flat — prevents degenerate fits
            if len(returns) < 100 or returns.std() < 5e-4:
                raise ValueError(f"Data too low variance (std={returns.std():.6f}) for HMM — skipping to Hurst")

            rolling_vol = returns.rolling(window=14).std().fillna(0.01)
            vol_change = volume.replace(0, np.nan).ffill()
            log_vol_change = np.log(vol_change / vol_change.shift(1)).fillna(0)
            aligned_idx = returns.index.intersection(rolling_vol.index).intersection(log_vol_change.index)
            if len(aligned_idx) < 100:
                raise ValueError(f"Insufficient aligned data for HMM ({len(aligned_idx)} bars)")

            observations = np.column_stack([
                np.clip(returns.loc[aligned_idx].values, -0.2, 0.2),
                np.clip(rolling_vol.loc[aligned_idx].values, 0, 0.2),
                np.clip(log_vol_change.loc[aligned_idx].values, -5, 5)
            ])

            # Configurable params
            ensemble_size = CONFIG.get('HMM_ENSEMBLE_SIZE', 4)  # reduced default — faster with higher n_init
            n_components = CONFIG.get('HMM_N_COMPONENTS', 2)
            seeds = [42, 123, 456, 789, 1011, 2024, 314, 271][:ensemble_size]

            trending_votes = 0
            votes_cast = 0  # Track actual votes (excludes skipped indistinguishable states)
            self_probs = []
            direction_votes = []  # Track signed mean return of current state per model

            # FIX: Compute dynamic_min_covar ONCE outside loop (observations don't change per seed)
            data_std = np.std(observations, axis=0).mean()
            dynamic_min_covar = max(1e-3, min(5e-2, data_std * 2.0))

            for i, seed in enumerate(seeds):
                try:

                    model = GaussianHMM(
                        n_components=n_components,
                        covariance_type="diag",
                        n_iter=2000,
                        random_state=seed,
                        min_covar=dynamic_min_covar,
                        tol=1e-4,
                    )
                    model.fit(observations)
                    hidden_states = model.predict(observations)
                    current_state = hidden_states[-1]
                    self_prob = model.transmat_[current_state, current_state]
                    self_probs.append(self_prob)
                    state_means = model.means_  # [n_components, n_features]
                    abs_mean_returns = np.abs(state_means[:, 0])
                    max_abs = abs_mean_returns.max()
                    min_abs = abs_mean_returns.min()
                    # Record the signed mean return of the current state for direction voting
                    direction_votes.append(float(state_means[current_state, 0]))
                    if max_abs > 1e-6 and (max_abs - min_abs) / (max_abs + 1e-8) > 0.3:
                        votes_cast += 1
                        trending_state = np.argmax(abs_mean_returns)
                        if current_state == trending_state:
                            trending_votes += 1
                    # else: skip vote — states are indistinguishable (no vote cast)
                except Exception as fit_e:
                    logger.warning(f"HMM fit failed for seed {seed} on {symbol}: {type(fit_e).__name__}: {fit_e}")
                    continue  # skip bad fit — ensemble continues

            if not self_probs:
                raise ValueError("All HMM fits failed — no valid transition probabilities")

            avg_self_prob = np.mean(self_probs)
            actual_fits = len(self_probs)
            # Use votes_cast (not actual_fits) as denominator so skipped votes
            # don't create an impossible majority threshold biasing toward mean_reverting
            if votes_cast > 0:
                is_trend = trending_votes >= (votes_cast + 1) // 2
            else:
                is_trend = False

            if is_trend:
                # Determine trend direction from majority of signed mean returns
                avg_direction = np.mean(direction_votes) if direction_votes else 0.0
                regime = 'trending_up' if avg_direction >= 0 else 'trending_down'
            else:
                regime = 'mean_reverting'

            if verbose:
                logger.debug(
                    f"HMM Ensemble | {symbol} | TF={used_tf} | votes={trending_votes}/{votes_cast} (fits={actual_fits}) | "
                    f"persistence={avg_self_prob:.3f} | regime={regime} | "
                    f"avg_direction={np.mean(direction_votes):.6f} | last_bar={full_data.index[-1]}"
                )
            else:
                logger.info(
                    f"HMM Ensemble | {symbol} | regime={regime} | persistence={avg_self_prob:.3f} | votes={trending_votes}/{votes_cast}"
                )

            persistence = float(avg_self_prob)

        except Exception as e:
            logger.warning(f"HMM ensemble failed ({e}) — falling back to Hurst")
            regime, persistence = None, None  # signal Hurst fallback needed

    else:
        regime, persistence = None, None  # GaussianHMM not available

    # Hurst fallback (only if HMM didn't produce a result)
    if regime is None:
        log_returns = np.log(prices / prices.shift(1)).dropna()
        if len(log_returns) < 50:
            return 'trending', 0.70

        max_lag = min(100, len(log_returns) // 2)
        lags = range(2, max_lag + 1)
        tau = []
        valid_lags = []
        for lag in lags:
            diff = log_returns.diff(lag).dropna()
            if len(diff) < 10:
                continue
            std_val = np.std(diff)
            if std_val > 1e-8:
                tau.append(std_val)
                valid_lags.append(lag)

        if len(tau) < 10:
            return 'mean_reverting', 0.45

        log_lags = np.log(valid_lags)
        log_tau = np.log(tau)
        try:
            poly = np.polyfit(log_lags, log_tau, 1)
            hurst = poly[0]
        except Exception:
            return 'mean_reverting', 0.45

        if hurst > CONFIG.get('HURST_TREND_THRESHOLD', 0.45):
            # Determine direction from recent price action
            recent_ret = (prices.iloc[-1] / prices.iloc[-min(20, len(prices))] - 1) if len(prices) >= 2 else 0
            regime = 'trending_up' if recent_ret >= 0 else 'trending_down'
        else:
            regime = 'mean_reverting'
        # FIX #34: Rescale Hurst persistence to [0.5, 0.95] to match HMM range
        persistence = 0.5 + (min(max(hurst, 0.3), 0.7) - 0.3) * (0.95 - 0.5) / (0.7 - 0.3)
        logger.info(f"Hurst fallback | {symbol} | Hurst={hurst:.3f} | persistence={persistence:.3f} | regime={regime}")

    # ==================== DIVERGENCE DETECTION (applies to BOTH HMM and Hurst results) ====================
    if len(prices) >= 50 and CONFIG.get('USE_DIVERGENCE', True):
        try:
            # FIX #34: Vectorized RSI using ewm instead of slow rolling.apply()
            _price_series = pd.Series(prices)
            _delta = _price_series.diff()
            _up = _delta.clip(lower=0).ewm(span=14, adjust=False).mean()
            _down = (-_delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
            _rs = _up / (_down + 1e-10)
            rsi = 100 - (100 / (1 + _rs))
            price_curr = prices.iloc[-1]
            price_prev_high = prices.iloc[-20:-5].max() if len(prices) >= 25 else prices.iloc[-20:].max()
            price_prev_low = prices.iloc[-20:-5].min() if len(prices) >= 25 else prices.iloc[-20:].min()
            rsi_curr = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
            rsi_prev_high = rsi.iloc[-20:-5].max() if len(rsi) >= 25 else 50.0
            rsi_prev_low = rsi.iloc[-20:-5].min() if len(rsi) >= 25 else 50.0
            # FIX: Guard against NaN from too few bars in rolling RSI
            if pd.isna(rsi_prev_high):
                rsi_prev_high = 50.0
            if pd.isna(rsi_prev_low):
                rsi_prev_low = 50.0
            if pd.isna(price_prev_high) or pd.isna(price_prev_low):
                # Not enough price data for divergence check — reset streak and skip
                sym_key = symbol or '_default_'
                with _divergence_streak_lock:
                    _divergence_streak[sym_key] = 0
                return regime, persistence
            bull_div = 1.0 if (price_curr < price_prev_low and rsi_curr > rsi_prev_low) else 0.0
            bear_div = 1.0 if (price_curr > price_prev_high and rsi_curr < rsi_prev_high) else 0.0
            # FIX #21: Require N consecutive divergent bars before triggering to reduce
            # false signals in trending markets. Single-point comparison is too noisy.
            div_threshold = CONFIG.get('DIVERGENCE_STREAK_THRESHOLD', 3)
            sym_key = symbol or '_default_'
            with _divergence_streak_lock:
                if bull_div == 1.0 or bear_div == 1.0:
                    _divergence_streak[sym_key] = _divergence_streak.get(sym_key, 0) + 1
                    streak_val = _divergence_streak[sym_key]
                else:
                    # No divergence this bar — reset streak
                    _divergence_streak[sym_key] = 0
                    streak_val = 0
            if bull_div == 1.0 or bear_div == 1.0:
                if streak_val >= div_threshold:
                    div_type = 'BULLISH' if bull_div == 1.0 else 'BEARISH'
                    regime = 'mean_reverting'
                    persistence = max(0.0, persistence - 0.15)
                    logger.debug(f"{div_type} DIVERGENCE confirmed for {symbol} "
                                 f"(streak={streak_val}) — setting mean_reverting")
                else:
                    logger.debug(f"Divergence signal for {symbol} "
                                 f"(streak={streak_val}/{div_threshold}) — not yet confirmed")
        except Exception as e:
            logger.warning(f"Divergence check failed for {symbol}: {e}")

    return regime, persistence


# =============================================================================
# NEW: Rolling Window Regime Detection (modular helper)
# =============================================================================
def get_regime_with_window(
    symbol: str,
    data_ingestion,
    lookback_full: int = 900,
    lookback_short: int = 96, # ~1 trading day of 15min bars
    weight_short: float = 0.6, # 0.0 = ignore short-term, 1.0 = ignore long-term
    cache: dict = None # reuse shared regime_cache from bot.py / signals.py
) -> tuple[str, float]:
    """
    Modular rolling window regime detection.
    Blends short-term (recent) regime with long-term (cached/full) regime.
    Returns (final_regime: str, final_persistence: float)
    """
    # Fetch data once and reuse for both short-term and long-term regime detection
    full_data = data_ingestion.get_latest_data(symbol, timeframe='15Min')

    # Short-term regime (recent window)
    recent_data = full_data.tail(lookback_short)
    if len(recent_data) >= 50:
        # FIX #26: Pass is_backtest=True to prevent detect_regime from re-fetching 1H data
        # via data_ingestion — forces it to use the passed-in 15Min data
        short_regime, short_persist = detect_regime(
            data=recent_data,
            symbol=symbol,
            data_ingestion=data_ingestion,
            lookback=lookback_short,
            is_backtest=True,
            verbose=False
        )
    else:
        short_regime, short_persist = 'mean_reverting', 0.5

    # Long-term regime (use cache first, then full data — no re-fetch needed)
    if cache and symbol in cache:
        cached = cache[symbol]
        long_regime = cached[0] if isinstance(cached, (list, tuple)) else str(cached)
        long_persist = cached[1] if isinstance(cached, (list, tuple)) else 0.5
    else:
        long_regime, long_persist = detect_regime(
            data=full_data,
            symbol=symbol,
            data_ingestion=data_ingestion,
            lookback=lookback_full,
            verbose=False
        )
        if cache is not None:
            cache[symbol] = (long_regime, long_persist)

    # Weighted blend — compare regime type (trending vs MR), not exact string (direction may differ)
    short_is_trend = is_trending(short_regime)
    long_is_trend = is_trending(long_regime)
    if short_is_trend == long_is_trend:
        # Same regime type — trust short-term direction
        final_regime = short_regime
        final_persist = weight_short * short_persist + (1 - weight_short) * long_persist
    else:
        # Conflict: trust short-term more, but dilute persistence
        final_regime = short_regime
        final_persist = weight_short * short_persist + (1 - weight_short) * 0.5

    logger.debug(f"{symbol} regime blend: short={short_regime} ({short_persist:.3f}), "
                 f"long={long_regime} ({long_persist:.3f}) → final={final_regime} ({final_persist:.3f})")
    return final_regime, final_persist

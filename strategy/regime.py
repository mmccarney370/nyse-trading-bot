# strategy/regime.py
import numpy as np
import pandas as pd
import logging
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

def detect_regime(
    data: pd.DataFrame,
    symbol: str = None,
    data_ingestion=None,
    lookback: int = CONFIG.get('LOOKBACK', 900),
    is_backtest: bool = False,
    verbose: bool = False  # NEW: Control verbosity
) -> tuple[str, float]:
    """
    Returns (regime: str, persistence_score: float 0.0-1.0)
    persistence_score = average self-transition probability from HMM ensemble
    (higher = more persistent/trending behavior)
    """
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
            logger.debug(f"Very short series (len={len(prices)}) — forcing trending + high persistence")
        return 'trending', 0.85

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
            self_probs = []

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
                        n_init=20  # ← increased: more restarts per seed
                    )
                    model.fit(observations)
                    hidden_states = model.predict(observations)
                    current_state = hidden_states[-1]
                    self_prob = model.transmat_[current_state, current_state]
                    self_probs.append(self_prob)
                    if self_prob > 0.80:
                        trending_votes += 1
                except Exception as fit_e:
                    logger.debug(f"HMM fit failed for seed {seed} on {symbol}: {fit_e} — skipping this fit")
                    continue  # skip bad fit — ensemble continues

            if not self_probs:
                raise ValueError("All HMM fits failed — no valid transition probabilities")

            avg_self_prob = np.mean(self_probs)
            actual_fits = len(self_probs)
            regime = 'trending' if trending_votes >= (actual_fits + 1) // 2 else 'mean_reverting'

            if verbose:
                logger.debug(
                    f"HMM Ensemble | {symbol} | TF={used_tf} | votes={trending_votes}/{actual_fits} | "
                    f"persistence={avg_self_prob:.3f} | regime={regime} | last_bar={full_data.index[-1]}"
                )
            else:
                logger.info(
                    f"HMM Ensemble | {symbol} | regime={regime} | persistence={avg_self_prob:.3f}"
                )

            regime, persistence = regime, float(avg_self_prob)

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

        regime = 'trending' if hurst > CONFIG.get('HURST_TREND_THRESHOLD', 0.45) else 'mean_reverting'
        persistence = min(max((hurst - 0.3) / 0.4, 0.0), 1.0)
        logger.info(f"Hurst fallback | {symbol} | Hurst={hurst:.3f} | persistence={persistence:.3f} | regime={regime}")

    # ==================== DIVERGENCE DETECTION (applies to BOTH HMM and Hurst results) ====================
    if len(prices) >= 50 and CONFIG.get('USE_DIVERGENCE', True):
        try:
            def _rsi_func(x):
                d = x.diff()
                up_sum = d.clip(lower=0).sum()
                down_sum = abs(d.clip(upper=0).sum())
                if down_sum < 1e-10:
                    return 100.0
                return 100 - 100 / (1 + up_sum / down_sum)
            rsi = pd.Series(prices).rolling(14).apply(_rsi_func, raw=False)
            price_curr = prices.iloc[-1]
            price_prev_high = prices.iloc[-20:-5].max() if len(prices) >= 25 else prices.iloc[-20:].max()
            price_prev_low = prices.iloc[-20:-5].min() if len(prices) >= 25 else prices.iloc[-20:].min()
            rsi_curr = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
            rsi_prev_high = rsi.iloc[-20:-5].max() if len(rsi) >= 25 else 50.0
            rsi_prev_low = rsi.iloc[-20:-5].min() if len(rsi) >= 25 else 50.0
            bull_div = 1.0 if (price_curr < price_prev_low and rsi_curr > rsi_prev_low) else 0.0
            bear_div = 1.0 if (price_curr > price_prev_high and rsi_curr < rsi_prev_high) else 0.0
            if bull_div == 1.0:
                regime = 'mean_reverting'
                persistence = min(1.0, persistence + 0.15)
                logger.debug(f"BULLISH DIVERGENCE detected for {symbol} — reversion signal, setting mean_reverting")
            elif bear_div == 1.0:
                regime = 'mean_reverting'
                persistence = min(1.0, persistence + 0.15)
                logger.debug(f"BEARISH DIVERGENCE detected for {symbol} — reversion signal, setting mean_reverting")
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
    # Short-term regime (recent window)
    recent_data = data_ingestion.get_latest_data(symbol, timeframe='15Min').tail(lookback_short)
    if len(recent_data) >= 50:
        short_regime, short_persist = detect_regime(
            data=recent_data,
            symbol=symbol,
            data_ingestion=data_ingestion,
            lookback=lookback_short,
            verbose=False
        )
    else:
        short_regime, short_persist = 'mean_reverting', 0.5

    # Long-term regime (use cache first, then full data)
    if cache and symbol in cache:
        cached = cache[symbol]
        long_regime = cached[0] if isinstance(cached, (list, tuple)) else str(cached)
        long_persist = cached[1] if isinstance(cached, (list, tuple)) else 0.5
    else:
        long_data = data_ingestion.get_latest_data(symbol, timeframe='15Min')
        long_regime, long_persist = detect_regime(
            data=long_data,
            symbol=symbol,
            data_ingestion=data_ingestion,
            lookback=lookback_full,
            verbose=False
        )
        if cache is not None:
            cache[symbol] = (long_regime, long_persist)

    # Weighted blend
    if short_regime == long_regime:
        final_regime = short_regime
        final_persist = weight_short * short_persist + (1 - weight_short) * long_persist
    else:
        # Conflict: trust short-term more, but dilute persistence
        final_regime = short_regime
        final_persist = weight_short * short_persist + (1 - weight_short) * 0.5

    logger.debug(f"{symbol} regime blend: short={short_regime} ({short_persist:.3f}), "
                 f"long={long_regime} ({long_persist:.3f}) → final={final_regime} ({final_persist:.3f})")
    return final_regime, final_persist

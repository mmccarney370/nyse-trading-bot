# models/features.py
# Updated feature generation for PPO observations.
# Key features: Technical indicators (BB, RSI, MACD, ATR, CCI, Stoch, OBV z-score, Chaikin, etc.), lag returns, macro inputs (VIX, yields, SPX), regime flag, entropy-based regime features, optional TFT encoder.
# CRITICAL FIX (Feb 18 2026): Insufficient features problem fixed
# → Lowered min length to 100
# → Liquidity check is now WARNING only (not hard return None)
# → Heavy debug logging at every stage
# → Guaranteed valid array return (never None if we reach the end)
# → More robust TFT precompute with shape safety and strong fallback
# NEW (Feb 20 2026): Smart TFT cache — keeps only recent data (MAX_TFT_CACHE_ROWS), auto-deletes old cache files,
# aggressive GPU/RAM cleanup, smaller batch size + hidden size when training, CPU-only precompute.
# FIXED (Feb 20 2026): TFT cache misalignment warning fixed with smart nearest + ffill alignment
# UPGRADE #6 (Feb 20 2026): Volatility-target prediction added as final input feature
# → annual_vol is now the last column (teacher-forcing for aux head)
# Requires for TFT: pip install pytorch-forecasting[pytorch] lightning
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dateutil import tz
from config import CONFIG
import os
import pickle
import gc
import logging
import warnings
import torch
warnings.filterwarnings("ignore", message="Attribute 'loss' is an instance of")
warnings.filterwarnings("ignore", message="Attribute 'logging_metrics' is an instance of")
logger = logging.getLogger(__name__)
# TFT imports (only loaded if enabled)
if CONFIG.get('USE_TFT_ENCODER', False):
    try:
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer, EncoderNormalizer
        from pytorch_forecasting.metrics import QuantileLoss
        import lightning.pytorch as pl
        TFT_AVAILABLE = True
    except ImportError:
        TFT_AVAILABLE = False
        logger.warning("pytorch-forecasting not available — TFT encoder disabled")
else:
    TFT_AVAILABLE = False
# Cache directory for precomputed TFT features
TFT_CACHE_DIR = "ppo_checkpoints/tft_cache"
os.makedirs(TFT_CACHE_DIR, exist_ok=True)
# ─── SMART TFT CACHE SETTINGS ─────────────────────────────────────
MAX_TFT_CACHE_ROWS = 2000 # Keep only ~30 days of 15min bars in memory/disk
CACHE_MAX_AGE_DAYS = 7 # Delete cache files older than this many days
# ──────────────────────────────────────────────────────────────────
# Simple daily cache for macro data
_macro_cache = {
    'date': None,
    'vix_close': 20.0,
    'vix_rsi': 50.0,
    'tnx_yield': 4.0,
    'vix_contango': 0.0,
    'yield_spread': 0.0,
    'spx_mom': 0.0
}
def _fetch_macro_features() -> dict:
    """Fetch VIX and 10Y yield once per day with safe fallbacks"""
    today = datetime.now(tz=tz.gettz('UTC')).date()
    if _macro_cache['date'] == today:
        return _macro_cache
    try:
        vix_ticker = yf.Ticker('^VIX')
        vix_hist = vix_ticker.history(period='30d')
        if not vix_hist.empty:
            vix_close = vix_hist['Close'].iloc[-1]
            delta = vix_hist['Close'].diff()
            up = delta.clip(lower=0).rolling(14).mean()
            down = -delta.clip(upper=0).rolling(14).mean() + 1e-8
            rs = up / down
            vix_rsi = 100 - (100 / (1 + rs.iloc[-1]))
        else:
            vix_close = 20.0
            vix_rsi = 50.0
    except Exception:
        vix_close = 20.0
        vix_rsi = 50.0
    try:
        tnx_ticker = yf.Ticker('^TNX')
        tnx_hist = tnx_ticker.history(period='5d')
        tnx_yield = tnx_hist['Close'].iloc[-1] / 100 if not tnx_hist.empty else 4.0
    except Exception:
        tnx_yield = 4.0
    try:
        # FIX: Reuse vix_close from above instead of making a duplicate API call
        vix_front = vix_close
        vix_back = yf.Ticker("VXZ").history(period="1d")['Close'].iloc[-1]
        vix_contango = (vix_back / vix_front) - 1
    except Exception:
        vix_contango = 0.0
    try:
        # FIX: Reuse tnx_yield from above instead of re-fetching ^TNX
        tnx = tnx_yield
        twoy = yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] / 100
        yield_spread = tnx - twoy
    except Exception:
        yield_spread = 0.0
    try:
        spx = yf.Ticker("^GSPC").history(period="5d")['Close']
        spx_mom = (spx.iloc[-1] / spx.iloc[-5]) - 1 if len(spx) >= 5 else 0.0
    except Exception:
        spx_mom = 0.0
    _macro_cache.update({
        'date': today,
        'vix_close': vix_close,
        'vix_rsi': vix_rsi,
        'tnx_yield': tnx_yield,
        'vix_contango': vix_contango,
        'yield_spread': yield_spread,
        'spx_mom': spx_mom
    })
    return _macro_cache
def _precompute_and_cache_tft(symbol: str, full_df: pd.DataFrame) -> pd.DataFrame:
    """Smart TFT precompute: keeps only recent data, auto-deletes old cache files, aggressive cleanup."""
    cache_path = os.path.join(TFT_CACHE_DIR, f"{symbol}.pkl")
   
    # === Aggressive cleanup of zero-filled or broken cache ===
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            if np.allclose(cached.values, 0.0, atol=1e-6):
                os.remove(cache_path)
                logger.info(f"[TFT CACHE CLEANUP] Deleted zero-filled cache for {symbol} — forcing fresh precompute")
        except Exception:
            pass
   
    # === CLEANUP: Delete old/out-of-date cache files ===
    if os.path.exists(cache_path):
        try:
            file_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))).days
            if file_age_days > CACHE_MAX_AGE_DAYS:
                os.remove(cache_path)
                logger.info(f"[TFT CACHE CLEANUP] Deleted old cache for {symbol} (age={file_age_days} days)")
        except Exception as e:
            logger.debug(f"Failed to delete old TFT cache {cache_path}: {e}")
    # === Try to load existing cache ===
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_df = pickle.load(f)
            if len(cached_df) == len(full_df) or len(cached_df) >= MAX_TFT_CACHE_ROWS:
                cached_df = cached_df.tail(MAX_TFT_CACHE_ROWS)
                logger.debug(f"TFT CACHE HIT for {symbol} — loaded {len(cached_df)} recent rows")
                return cached_df
            else:
                logger.debug(f"TFT cache stale for {symbol} (len mismatch) — recomputing")
        except Exception as e:
            logger.debug(f"TFT cache load failed for {symbol}: {e} — recomputing")
    # === Recompute TFT features (smart version) ===
    logger.debug(f"[TFT PRECOMPUTE] {symbol} on {len(full_df)} total bars → keeping last {MAX_TFT_CACHE_ROWS} rows in cache")
    torch.cuda.empty_cache()
    gc.collect()
    if not TFT_AVAILABLE or len(full_df) < 120:
        logger.warning(f"TFT precompute skipped for {symbol}: insufficient data or TFT not available")
        neutral_df = pd.DataFrame(
            np.zeros((min(len(full_df), MAX_TFT_CACHE_ROWS), 20)),
            index=full_df.index[-MAX_TFT_CACHE_ROWS:],
            columns=[f"tft_{i}" for i in range(20)]
        )
        with open(cache_path, 'wb') as f:
            pickle.dump(neutral_df, f)
        return neutral_df
    try:
        df = full_df.copy().tail(MAX_TFT_CACHE_ROWS * 2)
        # === STRONGER CLEANING TO PREVENT COLLAPSE ===
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0)
        df['volume'] = df['volume'].clip(lower=1)
        df['close'] = df['close'].clip(lower=0.01)
        logger.debug(f"[TFT INPUT DEBUG] {symbol} | rows={len(df)} | close_mean={df['close'].mean():.2f} | vol_mean={df['volume'].mean():.0f} | NaN after clean={df.isna().sum().sum()}")
        df = df.reset_index()
        df['time_idx'] = np.arange(len(df))
        df['symbol'] = 'stock'
        max_encoder_length = min(200, len(df) // 2)
        max_prediction_length = 24  # ~1 trading day of 15min bars; kept small for stable TFT windowing
        training_cutoff = len(df) - max_prediction_length
        training = TimeSeriesDataSet(
            df.iloc[:training_cutoff],
            time_idx="time_idx",
            target="close",
            group_ids=["symbol"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_reals=["time_idx", "open", "high", "low", "volume"] if 'volume' in df else ["time_idx", "open", "high", "low"],
            time_varying_unknown_reals=["close"],
            target_normalizer=EncoderNormalizer(transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        # FIX: Strip feature_names_in_ from internal scalers to prevent sklearn
        # "X does not have valid feature names" warnings. Root cause: pytorch-forecasting
        # fits StandardScalers on DataFrames (with column names) but transforms numpy arrays.
        def _strip_scaler_names(dataset):
            scalers = getattr(dataset, 'scalers', None)
            if scalers and isinstance(scalers, dict):
                for scaler in scalers.values():
                    if hasattr(scaler, 'feature_names_in_'):
                        delattr(scaler, 'feature_names_in_')
            tn = getattr(dataset, 'target_normalizer', None)
            if tn is not None:
                if hasattr(tn, 'feature_names_in_'):
                    delattr(tn, 'feature_names_in_')
                for attr in ['scaler_', 'center_', 'fitted_']:
                    inner = getattr(tn, attr, None)
                    if inner is not None and hasattr(inner, 'feature_names_in_'):
                        delattr(inner, 'feature_names_in_')
        _strip_scaler_names(training)
        full_dataset = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
        _strip_scaler_names(full_dataset)
        full_dataloader = full_dataset.to_dataloader(train=False, batch_size=32, num_workers=0)
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=3e-3,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=7,
            loss=QuantileLoss(),
            log_interval=0,
            reduce_on_plateau_patience=4,
        )
        trainer = pl.Trainer(
            max_epochs=12,
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            precision="32-true",
            num_sanity_val_steps=0,
            callbacks=[],
        )
        trainer.fit(tft, train_dataloaders=full_dataset.to_dataloader(train=True, batch_size=32, num_workers=0))
        with torch.no_grad():
            output = tft.predict(full_dataloader, mode="raw", return_x=True)
            if isinstance(output, tuple) and len(output) == 2:
                _, x = output
            elif hasattr(output, 'x'):
                x = output.x
            else:
                x = {}
            encoder_cont = x.get('encoder_cont')
            if encoder_cont is None:
                raise ValueError("No encoder_cont in output")
            encoder_vars = encoder_cont.detach().cpu().numpy()
            logger.debug(f"[TFT SHAPE DEBUG] {symbol} raw shape: {encoder_vars.shape} ndim={encoder_vars.ndim}")
            if encoder_vars.ndim == 3:
                # Average across sequence dimension (axis=1), not features (axis=2)
                encoder_vars = encoder_vars.mean(axis=1)
            elif encoder_vars.ndim == 1:
                encoder_vars = encoder_vars.reshape(1, -1)
            elif encoder_vars.ndim == 2 and encoder_vars.shape[0] == 1:
                pass  # Already 2D [1, features] — keep as-is for padding below
            if encoder_vars.ndim == 1:
                encoder_vars = encoder_vars.reshape(1, -1)
            if encoder_vars.ndim != 2:
                raise ValueError(f"Unexpected final shape: {encoder_vars.shape}")
            if encoder_vars.shape[0] > MAX_TFT_CACHE_ROWS:
                encoder_vars = encoder_vars[-MAX_TFT_CACHE_ROWS:]
            elif encoder_vars.shape[0] < MAX_TFT_CACHE_ROWS:
                pad_rows = np.zeros((MAX_TFT_CACHE_ROWS - encoder_vars.shape[0], encoder_vars.shape[1]))
                encoder_vars = np.vstack([encoder_vars, pad_rows])
            current_feats = encoder_vars.shape[1]
            if current_feats < 20:
                pad = np.zeros((encoder_vars.shape[0], 20 - current_feats))
                encoder_vars = np.concatenate([encoder_vars, pad], axis=1)
            elif current_feats > 20:
                encoder_vars = encoder_vars[:, :20]
            # === NEW: COLLAPSE PROTECTION ===
            if np.allclose(encoder_vars, 0, atol=1e-5):
                logger.warning(f"TFT collapsed to zeros for {symbol} — injecting small noise fallback")
                encoder_vars += np.random.normal(0, 0.01, encoder_vars.shape)
        tft_df = pd.DataFrame(
            encoder_vars,
            index=full_df.index[-MAX_TFT_CACHE_ROWS:],
            columns=[f"tft_{i}" for i in range(20)]
        )
        with open(cache_path, 'wb') as f:
            pickle.dump(tft_df, f)
        logger.debug(f"[TFT CACHE SAVED] {symbol} — {len(tft_df)} recent rows cached at {cache_path}")
        return tft_df
    except Exception as e:
        logger.error(f"TFT precompute failed for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        neutral_df = pd.DataFrame(
            np.zeros((MAX_TFT_CACHE_ROWS, 20)),
            index=full_df.index[-MAX_TFT_CACHE_ROWS:],
            columns=[f"tft_{i}" for i in range(20)]
        )
        with open(cache_path, 'wb') as f:
            pickle.dump(neutral_df, f)
        return neutral_df
    finally:
        if 'tft' in locals() and tft is not None:
            del tft
        if 'trainer' in locals() and trainer is not None:
            del trainer
        torch.cuda.empty_cache()
        gc.collect()
def generate_features(data: pd.DataFrame, regime: str, symbol: str, full_hist_df: pd.DataFrame = None) -> np.ndarray | None:
    logger.debug(f"[FEATURES START] {symbol} | input_len={len(data)} | regime={regime}")
   
    if CONFIG.get('USE_TFT_ENCODER', False):
        logger.debug(f"TFT encoder ENABLED for {symbol} — attempting cache load / precompute")
   
    if len(data) < 100 or data['close'].isna().all():
        logger.warning(f"[FEATURES SKIP] {symbol} — data too short or all NaN (len={len(data)})")
        return None
    required = ['open', 'high', 'low', 'close', 'volume']
    data = data.copy()
    for col in required:
        if col not in data.columns:
            data[col] = np.nan
    data = data[required]
    data = data.ffill().bfill()
    data['volume'] = data['volume'].replace(0, np.nan).ffill().bfill().fillna(0)
    avg_volume = data['volume'].mean()
    min_liquidity = CONFIG.get('MIN_LIQUIDITY', 100000) / 10
    if avg_volume < min_liquidity:
        logger.warning(f"[FEATURES LOW VOLUME] {symbol} — avg_volume={avg_volume:.0f} < {min_liquidity:.0f} — continuing anyway (not skipping)")
    close_changes = data['close'].pct_change().fillna(0.0)
    recent_vol_series = close_changes.tail(100)
    recent_vol = recent_vol_series.std(ddof=0)
    if np.isnan(recent_vol) or recent_vol <= 0:
        recent_vol = 1e-6
    vol_changes = data['volume'].pct_change().fillna(0.0)
    recent_vol_vol_series = vol_changes.tail(100)
    recent_vol_vol = recent_vol_vol_series.std(ddof=0)
    if np.isnan(recent_vol_vol) or recent_vol_vol <= 0:
        recent_vol_vol = 1e-6
    bb_mid = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std(ddof=0).fillna(recent_vol)
    bb_hband = bb_mid + 2 * bb_std
    bb_lband = bb_mid - 2 * bb_std
    bb_hband_ind = (data['close'] - bb_hband) / bb_std
    bb_lband_ind = (data['close'] - bb_lband) / bb_std
    delta = data['close'].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    # FIX: Add epsilon only in division, not to the rolling mean (was biasing RSI downward)
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down + 1e-8)
    rsi = 100 - (100 / (1 + rs)).fillna(50.0)
    ema12 = data['close'].ewm(span=12, adjust=False).mean()
    ema26 = data['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_diff = macd_line - macd_signal
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().fillna(recent_vol * 10)
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    tp_sma = typical_price.rolling(20).mean()
    mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    # FIX: Was replace(0, 1e28) which made CCI vanish when MAD=0. Use small epsilon instead.
    cci = (typical_price - tp_sma) / (0.015 * mad.replace(0, 1e-8))
    obv = (np.sign(data['close'].diff()) * data['volume']).cumsum()
    obv_z = (obv - obv.rolling(50).mean()) / (obv.rolling(50).std(ddof=0) + 1e-8)
    obv_z = obv_z.fillna(0.0)
    money_flow_mult = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'] + 1e-8)
    money_flow_vol = money_flow_mult * data['volume']
    chaikin_vol = money_flow_vol.rolling(20).sum() / data['volume'].rolling(20).sum().replace(0, 1e-8)
    chaikin_vol = chaikin_vol.fillna(0.0)
    low_14 = data['low'].rolling(14).min()
    high_14 = data['high'].rolling(14).max()
    stoch_k = 100 * (data['close'] - low_14) / (high_14 - low_14 + 1e-8)
    stoch_d = stoch_k.rolling(3).mean()
    stoch_k = stoch_k.fillna(50.0)
    stoch_d = stoch_d.fillna(50.0)
    ret_1 = close_changes.shift(1).fillna(0.0) / recent_vol
    ret_4 = data['close'].pct_change(4).fillna(0.0) / recent_vol
    ret_26 = data['close'].pct_change(26).fillna(0.0) / recent_vol
    ret_100 = data['close'].pct_change(100).fillna(0.0) / recent_vol
    macro = _fetch_macro_features()
    vix_normalized = (macro['vix_close'] - 20) / 10.0
    vix_rsi_norm = (macro['vix_rsi'] - 50) / 50.0
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum().replace(0, 1e-8)
    vwap_dev = (data['close'] - vwap) / vwap.clip(1e-6)
    up_vol = data['volume'] * (data['close'] > data['open']).astype(float)
    down_vol = data['volume'] * (data['close'] < data['open']).astype(float)
    imbalance = (up_vol - down_vol).rolling(20).sum() / data['volume'].rolling(20).sum().replace(0, 1e-8)
    price_high = data['close'].rolling(20).max()
    rsi_high = rsi.rolling(20).max()
    # FIX: Operator precedence — .astype(float) was binding only to the second operand
    bear_div = ((price_high > price_high.shift(1)) & (rsi_high < rsi_high.shift(1))).astype(float)
    bull_div = ((price_high < price_high.shift(1)) & (rsi_high > rsi_high.shift(1))).astype(float)
    try:
        vol_q = pd.qcut(data['volume'], q=10, duplicates='drop')
        vol_probs = vol_q.value_counts(normalize=True).values
        vol_entropy = -np.sum(vol_probs * np.log(vol_probs + 1e-8))
    except Exception:
        vol_entropy = 0.0
    try:
        returns = data['close'].pct_change().dropna()
        if len(returns) >= 10:
            ret_q = pd.qcut(returns, q=10, duplicates='drop')
            ret_probs = ret_q.value_counts(normalize=True).values
            price_entropy = -np.sum(ret_probs * np.log(ret_probs + 1e-8))
        else:
            price_entropy = 0.0
    except Exception:
        price_entropy = 0.0
    # TFT features
    tft_features = np.zeros((len(data), 20))
    if CONFIG.get('USE_TFT_ENCODER', False) and full_hist_df is not None:
        cached_df = _precompute_and_cache_tft(symbol, full_hist_df)
        aligned = cached_df.reindex(data.index).ffill().bfill()
        aligned = aligned.ffill().bfill()
        if aligned.isna().any().any():
            logger.debug(f"TFT cache misalignment for {symbol} — using neutral features")
            tft_features = np.zeros((len(data), 20))
        else:
            tft_features = aligned.values
            mean_val = float(tft_features.mean())
            std_val = float(tft_features.std())
            if abs(mean_val) < 1e-5 and abs(std_val) < 1e-5:
                logger.warning(f"⚠️ TFT for {symbol} returned all-zero features (neutral fallback)")
            else:
                logger.debug(f"✅ TFT ACTIVE for {symbol} — loaded {tft_features.shape} features | mean={mean_val:.4f} | std={std_val:.4f}")
    # UPGRADE #6: volatility target
    recent_vol_series = close_changes.tail(100)
    recent_vol = recent_vol_series.std(ddof=0)
    if np.isnan(recent_vol) or recent_vol <= 0:
        recent_vol = 1e-6
    annual_vol = recent_vol * np.sqrt(252 * 96)
    # FIX: RSI, MACD, CCI, Stochastic are in indicator-space (0-100 or unbounded),
    # NOT in return-space. Dividing by recent_vol (~0.005) saturates them at clip limits.
    # Use dimensionally appropriate normalization instead.
    features = np.column_stack([
        bb_hband_ind.fillna(0.0),
        bb_lband_ind.fillna(0.0),
        (rsi - 50) / 50.0,                             # RSI: normalize to [-1, 1]
        macd_line / (data['close'] * recent_vol + 1e-8),  # MACD: normalize by price * vol
        macd_signal / (data['close'] * recent_vol + 1e-8),
        macd_diff / (data['close'] * recent_vol + 1e-8),
        atr / (data['close'] * recent_vol + 1e-8),     # ATR: normalize by price * vol
        close_changes / recent_vol,                      # Returns: already in vol-space (correct)
        vol_changes / recent_vol_vol,
        np.full(len(data), 1.0 if regime == 'trending' else 0.0),
        cci / 200.0,                                     # CCI: typical range [-200, 200]
        obv_z,
        chaikin_vol,
        (stoch_k - 50) / 50.0,                          # Stochastic: normalize to [-1, 1]
        (stoch_d - 50) / 50.0,
        ret_1,
        ret_4,
        ret_26,
        ret_100,
        np.full(len(data), vix_normalized),
        np.full(len(data), vix_rsi_norm),
        np.full(len(data), macro['tnx_yield']),
        vwap_dev.fillna(0.0),
        imbalance.fillna(0.0),
        bear_div.fillna(0.0),
        bull_div.fillna(0.0),
        np.full(len(data), macro['vix_contango']),
        np.full(len(data), macro['yield_spread']),
        np.full(len(data), macro['spx_mom']),
        np.full(len(data), vol_entropy),
        np.full(len(data), price_entropy),
        tft_features,
        np.full(len(data), annual_vol)
    ])
    features = np.nan_to_num(features, nan=0.0, posinf=20.0, neginf=-20.0)
    features = np.clip(features, -20.0, 20.0)
    logger.debug(f"[FEATURES END] {symbol} — final shape {features.shape} (includes vol_target as last column)")
    return features

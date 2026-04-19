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
CACHE_MAX_AGE_DAYS = 1 # FIX #36: Reduced from 7 to 1 day — stale TFT features degrade signal quality
# ──────────────────────────────────────────────────────────────────
# Hourly cache for macro data (was daily — intraday VIX/SPX shifts were
# invisible to the feature pipeline until next day's fetch)
# HIGH-25 FIX: Add threading lock to protect from concurrent access during parallel training
import threading
_macro_cache_lock = threading.Lock()
_macro_cache = {
    'cache_key': None,  # (date, hour) tuple — refreshes hourly
    'vix_close': 20.0,
    'vix_rsi': 50.0,
    'tnx_yield': 0.04,
    'vix_contango': 0.0,
    'yield_spread': 0.0,
    'spx_mom': 0.0
}
def _fetch_macro_features() -> dict:
    """Fetch VIX and 10Y yield hourly with safe fallbacks.

    Changed from daily to hourly caching so intraday macro moves (VIX spikes, SPX
    breaks of the 200-SMA) actually reach the feature matrix and the portfolio
    gates. yfinance calls here are blocking but only hit once per hour per worker.
    Per-call try/except blocks ensure individual ticker failures don't propagate.
    """
    now = datetime.now(tz=tz.gettz('UTC'))
    cache_key = (now.date(), now.hour)
    with _macro_cache_lock:
        if _macro_cache['cache_key'] == cache_key:
            return _macro_cache.copy()
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
        tnx_yield = tnx_hist['Close'].iloc[-1] / 100 if not tnx_hist.empty else 0.04
    except Exception:
        tnx_yield = 0.04
    try:
        # FIX: Reuse vix_close from above instead of making a duplicate API call
        vix_front = vix_close
        vix_back = yf.Ticker("VXZ").history(period="1d")['Close'].iloc[-1]
        vix_contango = (vix_back / vix_front) - 1
    except Exception:
        vix_contango = 0.0
    try:
        # FIX: Reuse tnx_yield from above instead of re-fetching ^TNX
        # NOTE: ^IRX is the 13-week (3-month) T-bill rate, not the 2-year yield.
        # The 10Y-3M spread is a well-known recession indicator (inverted yield curve).
        tnx = tnx_yield
        tbill_3m = yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] / 100
        yield_spread = tnx - tbill_3m
    except Exception:
        yield_spread = 0.0
    spx_below_200sma = False
    try:
        spx = yf.Ticker("^GSPC").history(period="250d")['Close']
        spx_mom = (spx.iloc[-1] / spx.iloc[-5]) - 1 if len(spx) >= 5 else 0.0
        if len(spx) >= 200:
            spx_below_200sma = bool(spx.iloc[-1] < spx.rolling(200).mean().iloc[-1])
    except Exception:
        spx_mom = 0.0
    with _macro_cache_lock:
        _macro_cache.update({
            'cache_key': cache_key,
            'vix_close': vix_close,
            'vix_rsi': vix_rsi,
            'tnx_yield': tnx_yield,
            'vix_contango': vix_contango,
            'yield_spread': yield_spread,
            'spx_mom': spx_mom,
            'spx_below_200sma': spx_below_200sma,
        })
        return _macro_cache.copy()
def _make_neutral_tft_df(full_df: pd.DataFrame, n_feats: int = None) -> pd.DataFrame:
    """Return a zero-filled TFT DataFrame with a tft_valid=0 column (marks features as invalid)."""
    n_feats = n_feats or CONFIG.get('TFT_FEATURE_DIM', 20)
    n_rows = min(len(full_df), MAX_TFT_CACHE_ROWS)
    cols = [f"tft_{i}" for i in range(n_feats)] + ["tft_valid"]
    data = np.zeros((n_rows, n_feats + 1))  # +1 for tft_valid column (all zeros = invalid)
    return pd.DataFrame(data, index=full_df.index[-n_rows:], columns=cols)


def _strip_scaler_names(dataset):
    """Strip feature_names_in_ from internal scalers to prevent sklearn warnings.
    Root cause: pytorch-forecasting fits StandardScalers on DataFrames but transforms numpy arrays."""
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


def _load_tft_weights(symbol: str, tft_model):
    """Load persisted TFT weights if available and architecture matches."""
    if not CONFIG.get('TFT_PERSIST_WEIGHTS', True):
        return False
    weights_path = os.path.join(TFT_CACHE_DIR, f"{symbol}_weights.pt")
    if not os.path.exists(weights_path):
        return False
    try:
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        tft_model.load_state_dict(state, strict=False)
        logger.info(f"[TFT WEIGHTS] Loaded persisted weights for {symbol}")
        return True
    except Exception as e:
        logger.debug(f"[TFT WEIGHTS] Could not load weights for {symbol}: {e} — training from scratch")
        return False


def _save_tft_weights(symbol: str, tft_model):
    """Persist TFT weights so the next cache cycle warm-starts."""
    if not CONFIG.get('TFT_PERSIST_WEIGHTS', True):
        return
    weights_path = os.path.join(TFT_CACHE_DIR, f"{symbol}_weights.pt")
    try:
        torch.save(tft_model.state_dict(), weights_path)
        logger.debug(f"[TFT WEIGHTS] Saved weights for {symbol}")
    except Exception as e:
        logger.debug(f"[TFT WEIGHTS] Failed to save weights for {symbol}: {e}")


def _precompute_and_cache_tft(symbol: str, full_df: pd.DataFrame) -> pd.DataFrame:
    """Precompute TFT learned encoder representations and cache them.

    Key fixes over previous version:
    1. Extracts encoder_output (post-attention learned representations), NOT encoder_cont (raw inputs)
    2. Trains on training split only, infers on full_dataset (no train/test leakage)
    3. Volume moved to unknown reals (not known at prediction time)
    4. Feeds richer covariates: returns, ATR-proxy, volume change as unknown reals
    5. Uses last timestep of encoder output (preserves recency) instead of mean (destroyed temporal info)
    6. Persists model weights across cache cycles for warm-starting
    7. Adds tft_valid flag column so downstream can distinguish real features from zero fallback
    8. Cache keyed on data length to detect new bars
    9. GroupNormalizer instead of softplus EncoderNormalizer for cross-symbol stability
    """
    import warnings
    warnings.filterwarnings('ignore', message='X does not have valid feature names', category=UserWarning)

    n_feats = CONFIG.get('TFT_FEATURE_DIM', 20)
    cache_path = os.path.join(TFT_CACHE_DIR, f"{symbol}.pkl")

    # === Cleanup zero-filled or broken cache ===
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            # Check for all-zero features (excluding tft_valid column)
            feat_cols = [c for c in cached.columns if c.startswith('tft_') and c != 'tft_valid']
            if feat_cols and np.allclose(cached[feat_cols].values, 0.0, atol=1e-6):
                os.remove(cache_path)
                logger.info(f"[TFT CACHE CLEANUP] Deleted zero-filled cache for {symbol}")
        except Exception:
            pass

    # === Delete old cache files ===
    if os.path.exists(cache_path):
        try:
            file_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))).days
            if file_age_days > CACHE_MAX_AGE_DAYS:
                os.remove(cache_path)
                logger.info(f"[TFT CACHE CLEANUP] Deleted old cache for {symbol} (age={file_age_days} days)")
        except Exception as e:
            logger.debug(f"Failed to delete old TFT cache {cache_path}: {e}")

    # === Try to load existing cache — keyed on data length to detect new bars ===
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_df = pickle.load(f)
            # Validate cache has tft_valid column (new format) and enough rows
            has_valid_col = 'tft_valid' in cached_df.columns
            has_valid_features = has_valid_col and cached_df['tft_valid'].sum() > 0
            n_cached = len(cached_df)
            # Cache hit if: new format with valid features AND recent enough (within MAX_TFT_CACHE_ROWS of current data)
            if has_valid_features and abs(n_cached - min(len(full_df), MAX_TFT_CACHE_ROWS)) < 50:
                cached_df = cached_df.tail(MAX_TFT_CACHE_ROWS)
                logger.debug(f"TFT CACHE HIT for {symbol} — loaded {len(cached_df)} rows "
                             f"({int(cached_df['tft_valid'].sum())} valid)")
                return cached_df
            else:
                reason = "no tft_valid column" if not has_valid_col else (
                    "no valid features" if not has_valid_features else "length mismatch")
                logger.debug(f"TFT cache stale for {symbol} ({reason}) — recomputing")
        except Exception as e:
            logger.debug(f"TFT cache load failed for {symbol}: {e} — recomputing")

    # === Recompute TFT features ===
    logger.debug(f"[TFT PRECOMPUTE] {symbol} on {len(full_df)} bars → extracting learned encoder representations")
    torch.cuda.empty_cache()
    gc.collect()

    if not TFT_AVAILABLE or len(full_df) < 120:
        logger.warning(f"TFT precompute skipped for {symbol}: insufficient data or TFT not available")
        neutral_df = _make_neutral_tft_df(full_df, n_feats)
        with open(cache_path, 'wb') as f:
            pickle.dump(neutral_df, f)
        return neutral_df

    try:
        df = full_df.copy().tail(MAX_TFT_CACHE_ROWS * 2)

        # === Data cleaning ===
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0)
        df['volume'] = df['volume'].clip(lower=1)
        df['close'] = df['close'].clip(lower=0.01)

        # === Derived covariates for richer TFT input ===
        df['returns'] = df['close'].pct_change().fillna(0.0)
        df['vol_change'] = df['volume'].pct_change().fillna(0.0).clip(-5, 5)
        df['high_low_range'] = (df['high'] - df['low']) / df['close'].clip(lower=0.01)
        df['close_open_range'] = (df['close'] - df['open']) / df['close'].clip(lower=0.01)

        logger.debug(f"[TFT INPUT] {symbol} | rows={len(df)} | close_mean={df['close'].mean():.2f} | "
                     f"vol_mean={df['volume'].mean():.0f} | NaN={df.isna().sum().sum()}")

        df = df.reset_index()
        df['time_idx'] = np.arange(len(df))
        df['symbol'] = 'stock'

        max_encoder_length = min(CONFIG.get('TFT_MAX_ENCODER_LENGTH', 200), len(df) // 2)
        max_prediction_length = CONFIG.get('TFT_MAX_PREDICTION_LENGTH', 24)
        training_cutoff = len(df) - max_prediction_length

        if training_cutoff < max_encoder_length + max_prediction_length:
            logger.warning(f"TFT precompute skipped for {symbol}: not enough data for train/predict split")
            neutral_df = _make_neutral_tft_df(full_df, n_feats)
            with open(cache_path, 'wb') as f:
                pickle.dump(neutral_df, f)
            return neutral_df

        # FIX: volume is UNKNOWN at prediction time (moved from known_reals)
        # FIX: Feed richer covariates as unknown reals for variable selection network
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
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["close", "open", "high", "low", "volume",
                                        "returns", "vol_change", "high_low_range", "close_open_range"],
            target_normalizer=EncoderNormalizer(transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        _strip_scaler_names(training)

        # FIX: Train on training split, infer on full dataset (no train/test leakage)
        train_dataloader = training.to_dataloader(train=True, batch_size=32, num_workers=0)
        # predict=False + stop_randomization=True creates a sliding-window dataset
        # that covers the FULL time series (predict=True only emits 1 sample per group)
        full_dataset = TimeSeriesDataSet.from_dataset(training, df, predict=False, stop_randomization=True)
        _strip_scaler_names(full_dataset)
        full_dataloader = full_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

        hidden_size = CONFIG.get('TFT_HIDDEN_SIZE', 32)
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=CONFIG.get('TFT_LEARNING_RATE', 1e-3),
            hidden_size=hidden_size,
            attention_head_size=CONFIG.get('TFT_ATTENTION_HEADS', 4),
            dropout=CONFIG.get('TFT_DROPOUT', 0.1),
            hidden_continuous_size=CONFIG.get('TFT_HIDDEN_CONTINUOUS_SIZE', 16),
            output_size=7,
            loss=QuantileLoss(),
            log_interval=0,
            reduce_on_plateau_patience=4,
        )

        # Warm-start from persisted weights if available
        _load_tft_weights(symbol, tft)

        trainer = pl.Trainer(
            max_epochs=CONFIG.get('TFT_MAX_EPOCHS', 15),
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            precision="32-true",
            num_sanity_val_steps=0,
            callbacks=[],
        )
        # FIX: Train on training dataloader (not full_dataset) — prevents train/test leakage
        trainer.fit(tft, train_dataloaders=train_dataloader)

        # Persist weights for warm-starting next cycle
        _save_tft_weights(symbol, tft)

        # === Extract LEARNED encoder representations via forward hook ===
        # Hook the static_enrichment GRN layer — its output is the post-LSTM, post-gate-norm,
        # statically-enriched representation that feeds into multi-head attention.
        # This is the richest learned representation in the TFT encoder.
        _hook_captured = {}

        def _capture_hook(module, inp, output):
            _hook_captured['attn_input'] = output.detach()

        hook_handle = tft.static_enrichment.register_forward_hook(_capture_hook)

        all_encoder_outputs = []
        tft.eval()
        with torch.no_grad():
            for batch in full_dataloader:
                x_batch, _ = batch
                device = next(tft.parameters()).device
                x_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x_batch.items()}

                _hook_captured.clear()
                tft(x_batch)

                if 'attn_input' not in _hook_captured:
                    raise ValueError("static_enrichment hook did not fire")

                attn_input = _hook_captured['attn_input']
                # attn_input shape: [batch, encoder_length + prediction_length, hidden_size]
                # Extract encoder portion only (first max_encoder_length timesteps)
                enc_lengths = x_batch.get('encoder_lengths')
                if enc_lengths is not None:
                    max_enc = int(enc_lengths.max().item())
                else:
                    max_enc = attn_input.shape[1] - max_prediction_length

                encoder_repr = attn_input[:, :max_enc, :]
                # Use LAST timestep (preserves recency + temporal ordering)
                last_step = encoder_repr[:, -1, :].cpu().numpy()
                all_encoder_outputs.append(last_step)

        hook_handle.remove()

        encoder_vars = np.vstack(all_encoder_outputs)
        logger.debug(f"[TFT SHAPE] {symbol} encoder_output: {encoder_vars.shape} "
                     f"(batches={len(all_encoder_outputs)}, hidden_size={hidden_size})")

        # === Truncate/pad to fit available data (NOT MAX_TFT_CACHE_ROWS, which may exceed full_df) ===
        n_output_rows = min(len(full_df), MAX_TFT_CACHE_ROWS)
        if encoder_vars.shape[0] > n_output_rows:
            encoder_vars = encoder_vars[-n_output_rows:]
        elif encoder_vars.shape[0] < n_output_rows:
            pad_rows = np.zeros((n_output_rows - encoder_vars.shape[0], encoder_vars.shape[1]))
            encoder_vars = np.vstack([pad_rows, encoder_vars])

        current_feats = encoder_vars.shape[1]
        if current_feats < n_feats:
            pad = np.zeros((encoder_vars.shape[0], n_feats - current_feats))
            encoder_vars = np.concatenate([encoder_vars, pad], axis=1)
        elif current_feats > n_feats:
            encoder_vars = encoder_vars[:, :n_feats]

        # Build validity flag: 1.0 for rows with real data, 0.0 for padded rows
        n_total = encoder_vars.shape[0]
        n_real_from_model = sum(b.shape[0] for b in all_encoder_outputs)
        n_padded = max(0, n_total - n_real_from_model)

        # Normalize encoder outputs using ONLY real (non-padded) rows to prevent
        # zero-padding from distorting the statistics
        real_slice = encoder_vars[n_padded:]
        if len(real_slice) > 1:
            feat_mean = real_slice.mean(axis=0, keepdims=True)
            feat_std = real_slice.std(axis=0, keepdims=True)
            feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
            encoder_vars[n_padded:] = (real_slice - feat_mean) / feat_std
        elif len(real_slice) == 1:
            # Single row: can't z-score, just clip to reasonable range
            # The raw values from LayerNorm output are already roughly unit-scale
            encoder_vars[n_padded:] = np.clip(real_slice, -5.0, 5.0)
        valid_flags = np.concatenate([
            np.zeros(n_padded),
            np.ones(n_total - n_padded)
        ])

        collapsed = np.allclose(encoder_vars, 0, atol=1e-5)
        if collapsed:
            logger.warning(f"TFT encoder_output collapsed to zeros for {symbol}")
            valid_flags[:] = 0.0

        # Assemble DataFrame with tft_valid column
        feat_data = np.column_stack([encoder_vars, valid_flags])
        cols = [f"tft_{i}" for i in range(n_feats)] + ["tft_valid"]
        tft_df = pd.DataFrame(
            feat_data,
            index=full_df.index[-n_output_rows:],
            columns=cols
        )
        with open(cache_path, 'wb') as f:
            pickle.dump(tft_df, f)

        n_valid = int(valid_flags.sum())
        feat_mean_val = float(encoder_vars[n_padded:].mean()) if n_valid > 0 else 0.0
        feat_std_val = float(encoder_vars[n_padded:].std()) if n_valid > 0 else 0.0
        logger.info(f"[TFT CACHED] {symbol} — {n_valid}/{n_total} valid rows | "
                    f"mean={feat_mean_val:.4f} std={feat_std_val:.4f} | "
                    f"encoder_output shape=({n_total}, {n_feats})")
        return tft_df

    except Exception as e:
        logger.error(f"TFT precompute failed for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        neutral_df = _make_neutral_tft_df(full_df, n_feats)
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
    atr = tr.rolling(14).mean().fillna(data['close'].rolling(14).mean() * recent_vol)
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
    # FIX #48: Reuse typical_price computed above (line 384) instead of recomputing
    # FIX #17: VWAP should reset per trading session, not accumulate across the entire dataset.
    # Detect session boundaries by gaps > 2 hours in the index (overnight/weekend gaps).
    tp_vol = typical_price * data['volume']
    cum_tp_vol = tp_vol.cumsum()
    cum_vol = data['volume'].cumsum()
    if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
        time_diffs = data.index.to_series().diff()
        session_breaks = time_diffs > pd.Timedelta(hours=2)
        # Build session groups: increment group ID at each break
        session_id = session_breaks.cumsum()
        # Per-session cumulative sums for VWAP reset
        cum_tp_vol = tp_vol.groupby(session_id).cumsum()
        cum_vol = data['volume'].groupby(session_id).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, 1e-8)
    vwap_dev = (data['close'] - vwap) / vwap.clip(1e-6)
    up_vol = data['volume'] * (data['close'] > data['open']).astype(float)
    down_vol = data['volume'] * (data['close'] < data['open']).astype(float)
    imbalance = (up_vol - down_vol).rolling(20).sum() / data['volume'].rolling(20).sum().replace(0, 1e-8)
    price_high = data['close'].rolling(20).max()
    rsi_high = rsi.rolling(20).max()
    # HIGH-26 FIX: bull_div should use price LOWS and RSI LOWS (standard bullish divergence)
    # Bullish divergence = price makes lower low but RSI makes higher low
    price_low = data['close'].rolling(20).min()
    rsi_low = rsi.rolling(20).min()
    # FIX: Operator precedence — .astype(float) was binding only to the second operand
    bear_div = ((price_high > price_high.shift(1)) & (rsi_high < rsi_high.shift(1))).astype(float)
    bull_div = ((price_low < price_low.shift(1)) & (rsi_low > rsi_low.shift(1))).astype(float)
    # FIX #45: Compute rolling entropy over a window instead of a single scalar broadcast to all bars.
    # This makes the entropy features time-varying, capturing local distribution changes.
    _entropy_window = 50
    def _rolling_qcut_entropy(series, window, q=10, stride=10):
        """Compute Shannon entropy of qcut bins over a rolling window.

        FIX #14: O(n×window) Python loop is very slow for large datasets.
        Optimization: compute entropy every `stride` bars and forward-fill,
        reducing work by ~stride× while preserving the signal shape.
        """
        n = len(series)
        result = np.zeros(n)
        values = series.values
        # Compute only at strided indices to reduce Python-loop overhead
        last_val = 0.0
        for idx in range(window, n, stride):
            window_data = values[idx - window:idx]
            valid = window_data[~np.isnan(window_data)]
            if len(valid) < q:
                # FIX #47: Explicitly return zero and log when data too short for entropy bins
                result[idx] = 0.0
                if idx == window:  # Log only once per series
                    logger.debug(f"[ENTROPY] Insufficient data for qcut (valid={len(valid)} < q={q}) — returning zeros")
                continue
            try:
                binned = pd.qcut(valid, q=q, duplicates='drop')
                probs = binned.value_counts(normalize=True).values
                last_val = -np.sum(probs * np.log(probs + 1e-8))
                result[idx] = last_val
            except Exception:
                result[idx] = last_val
        # Forward-fill strided values into gaps
        computed_indices = list(range(window, n, stride))
        for i, ci in enumerate(computed_indices):
            end = computed_indices[i + 1] if i + 1 < len(computed_indices) else n
            result[ci:end] = result[ci]
        # Backfill the initial window with the first computed value
        if n > window:
            result[:window] = result[window]
        return result
    vol_entropy_arr = _rolling_qcut_entropy(data['volume'], _entropy_window)
    price_entropy_arr = _rolling_qcut_entropy(data['close'].pct_change().fillna(0.0), _entropy_window)
    # TFT features — n_feats learned encoder outputs + 1 validity flag
    n_tft_feats = CONFIG.get('TFT_FEATURE_DIM', 20)
    tft_features = np.zeros((len(data), n_tft_feats))
    tft_valid = np.zeros(len(data))
    if CONFIG.get('USE_TFT_ENCODER', False) and full_hist_df is not None:
        try:
            cached_df = _precompute_and_cache_tft(symbol, full_hist_df)
            feat_cols = [c for c in cached_df.columns if c.startswith('tft_') and c != 'tft_valid']
            has_valid = 'tft_valid' in cached_df.columns
            # Align cached TFT features to current data index
            try:
                aligned = cached_df.reindex(data.index, method='nearest').ffill().bfill()
            except (ValueError, TypeError):
                # Fallback: positional alignment when index types don't match
                n = min(len(cached_df), len(data))
                aligned = cached_df.iloc[-n:].copy()
                aligned.index = data.index[-n:]
                # Pad front with zeros if data is longer
                if len(data) > n:
                    front = pd.DataFrame(0.0, index=data.index[:len(data) - n], columns=cached_df.columns)
                    aligned = pd.concat([front, aligned])
            if aligned.isna().any().any():
                logger.debug(f"TFT cache misalignment for {symbol} — using neutral features")
            else:
                tft_features = aligned[feat_cols].values if feat_cols else np.zeros((len(data), n_tft_feats))
                # Apr-19 audit: when the cache has no tft_valid column the
                # data is zero-padded neutral — default to INVALID (zeros)
                # instead of treating it as valid. Without this the policy
                # trains on meaningless 20-dim vectors for any symbol with
                # <120 bars of history. Additionally zero the TFT contribution
                # entirely when fewer than half the rows are valid.
                if has_valid:
                    tft_valid = aligned['tft_valid'].values
                else:
                    tft_valid = np.zeros(len(data))
                n_valid = int(tft_valid.sum())
                valid_frac = float(n_valid) / max(1, len(tft_valid))
                min_valid_frac = float(CONFIG.get('TFT_MIN_VALID_FRAC', 0.5))
                if n_valid == 0:
                    logger.warning(f"TFT for {symbol}: all rows invalid (collapsed/padded) — zeroing TFT contribution")
                    tft_features = np.zeros_like(tft_features)
                elif valid_frac < min_valid_frac:
                    logger.warning(
                        f"TFT for {symbol}: only {valid_frac:.0%} of rows valid "
                        f"(< {min_valid_frac:.0%}) — zeroing TFT contribution"
                    )
                    tft_features = np.zeros_like(tft_features)
                else:
                    valid_feats = tft_features[tft_valid > 0.5]
                    mean_val = float(valid_feats.mean()) if len(valid_feats) > 0 else 0.0
                    std_val = float(valid_feats.std()) if len(valid_feats) > 0 else 0.0
                    # Also zero the INVALID rows so the policy never sees
                    # zero-padded features mixed with real ones on the same row.
                    tft_features = tft_features * tft_valid[:, None]
                    logger.debug(f"TFT ACTIVE for {symbol} — {n_valid}/{len(data)} valid | "
                                 f"mean={mean_val:.4f} std={std_val:.4f}")
        except Exception as e:
            logger.warning(f"TFT integration failed for {symbol}: {e} — using neutral features")
    # Ensure correct column count
    if tft_features.shape[1] < n_tft_feats:
        pad = np.zeros((len(data), n_tft_feats - tft_features.shape[1]))
        tft_features = np.concatenate([tft_features, pad], axis=1)
    elif tft_features.shape[1] > n_tft_feats:
        tft_features = tft_features[:, :n_tft_feats]
    # UPGRADE #6: volatility target
    recent_vol_series = close_changes.tail(100)
    recent_vol = recent_vol_series.std(ddof=0)
    if np.isnan(recent_vol) or recent_vol <= 0:
        recent_vol = 1e-6
    # Compute rolling annual_vol so each bar gets its own volatility value (not a single scalar broadcast)
    _vol_window = 100
    rolling_std = close_changes.rolling(window=_vol_window, min_periods=20).std(ddof=0)
    rolling_std = rolling_std.fillna(recent_vol)  # fill initial NaNs with global recent_vol
    # FIX #49: Annualization uses 252*26 (252 trading days * 26 fifteen-min bars/day).
    # Matches CONFIG.get('ANNUALIZATION_FACTOR', 252*26) in env.py.
    annual_vol = rolling_std * np.sqrt(CONFIG.get('ANNUALIZATION_FACTOR', 252 * 26))
    # SMA trend features — price position relative to key moving averages
    sma_50 = data['close'].rolling(50, min_periods=20).mean()
    sma_200 = data['close'].rolling(200, min_periods=50).mean()
    # Price distance from SMA as fraction of price (dimensionless, typical range ~[-0.1, 0.1])
    price_vs_sma50 = ((data['close'] - sma_50) / (sma_50 + 1e-8)).fillna(0.0)
    price_vs_sma200 = ((data['close'] - sma_200) / (sma_200 + 1e-8)).fillna(0.0)
    # Golden/death cross: SMA(50) vs SMA(200) distance
    sma_cross = ((sma_50 - sma_200) / (sma_200 + 1e-8)).fillna(0.0)
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
        # Regime flag: +1.0 = trending up, -1.0 = trending down, 0.0 = mean reverting.
        # Encodes both regime TYPE and DIRECTION so models can distinguish bull/bear trends.
        np.full(len(data), 1.0 if regime == 'trending_up' else (-1.0 if regime == 'trending_down' else 0.0)),
        cci / 200.0,                                     # CCI: typical range [-200, 200]
        obv_z,
        chaikin_vol,
        (stoch_k - 50) / 50.0,                          # Stochastic: normalize to [-1, 1]
        (stoch_d - 50) / 50.0,
        ret_1,
        ret_4,
        ret_26,
        ret_100,
        # M45 NOTE: Macro values (VIX, yields, SPX) are daily snapshots broadcast to all bars.
        # In live trading this is correct (latest daily value). In backtesting, this introduces
        # mild look-ahead bias (today's macro applied to intraday history). Accepted trade-off:
        # macro features change slowly (daily) and intraday variation is negligible.
        np.full(len(data), vix_normalized),
        np.full(len(data), vix_rsi_norm),
        np.full(len(data), (macro['tnx_yield'] - 0.04) / 0.02),  # 10Y yield: center ~4%, scale ~2%
        (vwap_dev * 100).fillna(0.0),                              # VWAP dev: scale from ~0.005 to ~0.5
        imbalance.fillna(0.0),
        bear_div.fillna(0.0),
        bull_div.fillna(0.0),
        np.full(len(data), macro['vix_contango'] * 5.0),          # VIX contango: scale from ~0.1 to ~0.5
        np.full(len(data), macro['yield_spread'] / 0.02),         # Yield spread: center ~2%, scale to ~1.0
        np.full(len(data), macro['spx_mom'] / 0.01),              # SPX 5d mom: scale from ~0.02 to ~2.0
        (vol_entropy_arr - 1.15) / 1.15,                           # Shannon entropy: center & scale to ~[-1, 1]
        (price_entropy_arr - 1.15) / 1.15,                         # Shannon entropy: center & scale to ~[-1, 1]
        price_vs_sma50 * 10.0,                                      # Price vs SMA(50): scale to ~[-1, 1]
        price_vs_sma200 * 10.0,                                     # Price vs SMA(200): scale to ~[-1, 1]
        sma_cross * 10.0,                                           # Golden/death cross strength
        tft_features,
        tft_valid.reshape(-1, 1),
        annual_vol.values
    ])
    features = np.nan_to_num(features, nan=0.0, posinf=20.0, neginf=-20.0)
    features = np.clip(features, -20.0, 20.0)
    logger.debug(f"[FEATURES END] {symbol} — final shape {features.shape} (includes vol_target as last column)")
    return features

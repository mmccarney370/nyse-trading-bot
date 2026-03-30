# models/stacking_ensemble.py
"""
LightGBM stacking ensemble training for meta-probability generation.
This module contains the logic previously in trainer.py's _train_stacking method.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
from config import CONFIG
from models.features import generate_features
from strategy.regime import detect_regime

logger = logging.getLogger(__name__)

def train_stacking(
    symbol: str,
    data: pd.DataFrame,
    full_hist_df: pd.DataFrame = None,
    regime_cache: dict = None  # ← NEW: optional shared regime cache to avoid redundant HMM calls
) -> list:
    """
    Train an ensemble of LightGBM models for stacking/meta-probability generation.
   
    Args:
        symbol: The stock ticker symbol (used for logging and TFT caching)
        data: DataFrame with OHLCV columns and sufficient history for feature generation
        full_hist_df: Optional full historical DataFrame for TFT caching/lookup.
                      If provided, enables precomputed TFT features. If None, falls back
                      to neutral TFT features.
        regime_cache: Optional dict of {symbol: (regime_str, persistence)} from bot.py/trainer.py
                      If provided and symbol is present, reuses cached regime instead of recomputing.
   
    Returns:
        List of trained LightGBM Booster models (or empty list on failure)
    """
    logger.info(f"Training {CONFIG.get('NUM_BASE_MODELS', 15)} stacking models for {symbol}")
   
    # M-1 / HMM fix: prefer cached regime if available (avoids redundant HMM calls during training)
    if regime_cache and symbol in regime_cache:
        cached = regime_cache[symbol]
        regime = cached[0] if isinstance(cached, (list, tuple)) else str(cached)
        persistence = cached[1] if isinstance(cached, (list, tuple)) and len(cached) == 2 else 0.5
        logger.debug(f"[STACKING] Using cached regime for {symbol}: {regime} (persistence={persistence:.3f})")
    else:
        # Fallback: compute fresh with symbol passed (allows 1H upgrade + future caching)
        regime_tuple = detect_regime(data, symbol=symbol)
        regime, persistence = regime_tuple if isinstance(regime_tuple, tuple) else (regime_tuple, 0.5)
        logger.debug(f"[STACKING] Computed fresh regime for {symbol}: {regime} (persistence={persistence:.3f})")
   
    # Generate features using the current regime (now guaranteed consistent with rest of bot)
    # FIXED: Pass symbol and full_hist_df for TFT caching support
    features = generate_features(
        data=data,
        regime=regime,  # now a string, not a tuple
        symbol=symbol,
        full_hist_df=full_hist_df
    )
   
    if features is None or len(features) < 300:
        logger.warning(f"Insufficient features for stacking on {symbol} — returning empty models")
        return []

    # FIX #57: Sanitize inf values before LightGBM — inf causes silent NaN propagation in trees
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
   
    # Multi-bar forward labels: predict direction over the actual holding horizon.
    # 1-bar labels have SNR ~0.005 (coin flip). 8-bar labels have ~2.8x higher SNR
    # and align with MIN_HOLD_BARS_TRENDING (8), matching actual trading behavior.
    horizon = CONFIG.get('LABEL_HORIZON_BARS', 8)
    forward_returns = data['close'].pct_change(horizon).shift(-horizon)
    labels_full = (forward_returns > 0).astype(int).values  # last `horizon` elements are NaN → 0

    # Align: features are tail-aligned to data. Drop last `horizon` feature rows (no complete forward return).
    n_feat = len(features) - horizon
    if n_feat < 200:
        logger.warning(f"[STACKING] {symbol}: only {n_feat} usable rows after {horizon}-bar horizon — too few for training")
        return []
    features = features[:n_feat]

    # Labels for the same data rows: tail slice of length n_feat, excluding last `horizon` data rows
    labels = labels_full[-(n_feat + horizon):-horizon]

    # Final safety: truncate to shorter of the two
    min_len = min(len(features), len(labels))
    features = features[:min_len]
    labels = labels[:min_len]
   
    # Time-based train/validation split to prevent overfitting
    # Use first 70% for training, last 30% for validation (no shuffle — respects temporal order)
    split_idx = int(len(features) * 0.70)
    train_features = features[:split_idx]
    train_labels = labels[:split_idx]
    val_features = features[split_idx:]
    val_labels = labels[split_idx:]

    if len(train_features) < 200 or len(val_features) < 50:
        logger.warning(f"Train/val split too small for {symbol} (train={len(train_features)}, val={len(val_features)}) — using full data")
        train_features = features
        train_labels = labels
        val_features = None
        val_labels = None

    models = []
    n_models = CONFIG.get('NUM_BASE_MODELS', 15)
    # M43 FIX: Use .get() with default to prevent KeyError
    params = CONFIG.get('LIGHTGBM_PARAMS', {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'learning_rate': 0.025, 'num_leaves': 40
    }).copy()

    params['num_iterations'] = CONFIG.get('LGB_NUM_ITERATIONS', 250)
    params['verbose'] = -1

    # Let LightGBM handle its own internal bagging via bagging_fraction param.
    # Removed outer bootstrap loop to avoid double bagging (outer bootstrap + LightGBM's
    # internal subsampling), which reduces effective sample diversity.
    # FIX #11: bagging_fraction requires bagging_freq > 0 to actually take effect.
    # Also set per-model random seeds for ensemble diversity.
    params['bagging_freq'] = params.get('bagging_freq', 1)
    for i in range(n_models):
        params['seed'] = i * 42 + 7
        train_data = lgb.Dataset(train_features, label=train_labels)
        if val_features is not None:
            val_data = lgb.Dataset(val_features, label=val_labels, reference=train_data)
            callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
            model = lgb.train(params, train_data, valid_sets=[val_data], callbacks=callbacks)
        else:
            # FIX #58: Cap iterations when no validation set to prevent overfitting
            no_val_params = params.copy()
            no_val_params['num_iterations'] = min(no_val_params.get('num_iterations', 250), 100)
            model = lgb.train(no_val_params, train_data)
        models.append(model)

        if (i + 1) % 5 == 0:
            logger.debug(f"Trained model {i+1}/{n_models} for {symbol}")

    logger.info(f"Trained {len(models)} stacking models for {symbol} (train={len(train_features)}, val={len(val_features) if val_features is not None else 0})")
    return models

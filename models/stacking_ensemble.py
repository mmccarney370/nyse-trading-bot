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
   
    # Prepare labels: next bar direction (1 = up, 0 = down)
    # pct_change() produces NaN at index 0; shift(-1) produces NaN at last index
    # dropna() removes both → labels has len(data)-2 rows starting from index 1
    returns = data['close'].pct_change().shift(-1).dropna()
    labels = (returns > 0).astype(int).values

    # Align features with labels:
    # labels[i] = future return from bar i → i+1 (indices 0..N-2)
    # features[i] = features at bar i (indices 0..N-1)
    # Drop first feature row (may have NaN from rolling windows) AND first label
    # so features[1] pairs with labels[1] (= future return from bar 1 → 2)
    features = features[1:-1]   # bars 1..N-2
    labels = labels[1:]         # future returns from bar 1..N-2
    # Final safety: truncate to shorter of the two
    min_len = min(len(features), len(labels))
    features = features[:min_len]
    labels = labels[:min_len]
   
    models = []
    n_models = CONFIG.get('NUM_BASE_MODELS', 15)
    params = CONFIG['LIGHTGBM_PARAMS'].copy()
   
    # Fixed training parameters
    params['num_iterations'] = CONFIG.get('LGB_NUM_ITERATIONS', 250)
    params['verbose'] = -1
   
    for i in range(n_models):
        # Bootstrap sampling if bagging_fraction < 1
        if params.get('bagging_fraction', 1.0) < 1.0:
            sample_size = int(len(features) * params['bagging_fraction'])
            indices = np.random.choice(len(features), size=sample_size, replace=True)
            X_boot = features[indices]
            y_boot = labels[indices]
        else:
            X_boot = features
            y_boot = labels
       
        # Create dataset and train
        train_data = lgb.Dataset(X_boot, label=y_boot)
        model = lgb.train(params, train_data)
        models.append(model)
       
        # Optional: log progress every 5 models
        if (i + 1) % 5 == 0:
            logger.debug(f"Trained model {i+1}/{n_models} for {symbol}")
   
    logger.info(f"Trained {len(models)} stacking models for {symbol}")
    return models

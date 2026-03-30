# config.py
import os
from datetime import timedelta
from dotenv import load_dotenv
load_dotenv(override=True)
# Runtime/dynamic values only — defaults & types now live in Pydantic class below
CONFIG = {
    # ==================== API & Broker Settings ====================
    'API_KEY': os.getenv('ALPACA_API_KEY'),
    'API_SECRET': os.getenv('ALPACA_API_SECRET'),
    'BASE_URL': 'https://paper-api.alpaca.markets/v2',
    'POLYGON_API_KEY': os.getenv('POLYGON_API_KEY'),
    'TIINGO_API_KEY': os.getenv('TIINGO_API_KEY'),
    'FINNHUB_API_KEY': os.getenv('FINNHUB_API_KEY'),
    'NEWS_API_KEY': os.getenv('NEWS_API_KEY'),
    'GROQ_API_KEY': os.getenv('GROQ_API_KEY'),
    'REDIS_HOST': os.getenv('REDIS_HOST', 'localhost'),
    'REDIS_PORT': int(os.getenv('REDIS_PORT', '6379')),
    'REDIS_DB': int(os.getenv('REDIS_DB', '0')),
    'PAPER': True,
    # ==================== Trading Universe ====================
    'SYMBOLS': ['SOFI', 'PLTR', 'AMD', 'NVDA', 'JPM', 'TSLA', 'AAPL', 'SMCI'],
    'UNIVERSE_CANDIDATES': [
        'SOFI', 'PLTR', 'AMD', 'NVDA', 'JPM', 'TSLA', 'AAPL', 'SMCI',
        'MSFT', 'GOOGL', 'META', 'AMZN', 'NFLX', 'COIN', 'HOOD', 'MARA', 'RIOT',
        'UPST', 'SQ', 'SHOP', 'UBER', 'CRWD', 'PANW', 'ZS', 'DDOG', 'SNOW',
        'ARM', 'PATH', 'RBLX', 'UNH', 'VST', 'CEG', 'NET', 'MDB'
    ],
    # ==================== Operational & Misc ====================
    'CURRENT_REGIME': 'mean_reverting',
}
# ==================== Pydantic Validation Layer — SINGLE SOURCE OF TRUTH ====================
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any
class TradingBotConfig(BaseModel):
    # ==================== API & Broker Settings ====================
    API_KEY: str = Field(min_length=1)
    API_SECRET: str = Field(min_length=1)
    BASE_URL: str
    POLYGON_API_KEY: str | None = None
    TIINGO_API_KEY: str | None = None
    FINNHUB_API_KEY: str | None = None
    NEWS_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    REDIS_HOST: str = 'localhost'
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    PAPER: bool = True
    # ==================== Trading Universe ====================
    SYMBOLS: List[str]
    UNIVERSE_CANDIDATES: List[str]
    MAX_UNIVERSE_SIZE: int = 8
    UNIVERSE_UPDATE_INTERVAL_HOURS: int = 168
    MIN_AVG_VOLUME: int = 5_000_000
    UNIVERSE_LOOKBACK_DAYS: int = 90
    # L34 NOTE: LIQUIDITY + REGIME + PERFORMANCE = 0.90 (scoring weights, sum to ~1.0).
    # DIVERSIFICATION_WEIGHT is a penalty multiplier, not a scoring weight — applied separately.
    LIQUIDITY_WEIGHT: float = 0.15
    REGIME_TRENDING_WEIGHT: float = 0.45
    PERFORMANCE_WEIGHT: float = 0.30
    DIVERSIFICATION_WEIGHT: float = 0.10  # Penalty multiplier, not additive weight
    TIMEFRAMES: List[str] = Field(default_factory=lambda: ['15Min', '1H'])
    # ==================== Risk & Position Sizing ====================
    RISK_PER_TRADE: float = 0.015                      # Was 0.012 — deploy more per trade with improved signals
    TRAILING_STOP_ATR: float = 4.5                     # Was 4.0 — wider stops to avoid premature exits
    TAKE_PROFIT_ATR: float = 20.0                      # Was 18.0 — larger targets
    RISK_PER_TRADE_MEAN_REVERTING: float = 0.008       # Was 0.006 — MR is lower risk, can allocate more
    RISK_PER_TRADE_TRENDING: float = 0.03              # Was 0.025 — trends are the main profit driver
    # M54 FIX: Swapped — trends need wider stops (ride momentum), MR needs tighter (quick exits)
    TRAILING_STOP_ATR_MEAN_REVERTING: float = 3.5      # Tight for quick MR trades
    TRAILING_STOP_ATR_TRENDING: float = 4.5            # Wide to ride trend momentum
    TAKE_PROFIT_ATR_MEAN_REVERTING: float = 8.0        # Was 10.0 — MR trades should book profit faster
    TAKE_PROFIT_ATR_TRENDING: float = 35.0             # Was 30.0 — let trend winners run further
    MAX_POSITIONS: int = 5                             # Was 6 — concentrate capital in higher-conviction trades
    MAX_ATR_VOLATILITY: float = 0.06
    KELLY_FRACTION: float = 0.55                       # Was 0.50 — slightly more aggressive
    COMMISSION_PER_SHARE: float = 0.0005
    SLIPPAGE: float = 0.0001
    ASSUMED_SPREAD: float = 0.0
    LIMIT_PRICE_OFFSET: float = 0.003                  # Was 0.004 — tighter limit orders for better fills
    DAILY_LOSS_THRESHOLD: float = -0.03                # HIGH-19 FIX: Was -0.20 (20% daily loss!) — 3% is standard
    API_FAILURE_THRESHOLD: int = 5                    # HIGH-20 FIX: Was 5000 — circuit breaker never fired
    MIN_LIQUIDITY: int = 100000
    MAX_SECTOR_CONCENTRATION: float = 0.30
    MAX_TOTAL_RISK_PCT: float = 0.18                   # Was 0.15 — allow more total risk with better signals
    MAX_POSITION_VALUE_FRACTION: float = 0.30          # Was 0.25 — bigger conviction bets
    RISK_BUDGET_MULTIPLIER: float = 2.2                # Was 2.0 — more capital deployed
    # ==================== Signal Generation ====================
    MIN_CONFIDENCE: float = 0.72                       # Was 0.78 — take more trades with improved signal pipeline
    PORTFOLIO_SENTIMENT_WEIGHT: float = 0.25           # Was 0.35 — sentiment is noisy, reduce influence
    SENTIMENT_WEIGHT: float = 0.18                     # Was 0.25 — let PPO/stacking dominate more
    PPO_SIGNAL_WEIGHT: float = 0.20                    # PPO's direct contribution to combined signal (rest is LightGBM ensemble)
    USE_LLM_DEBATE: bool = True
    LLM_DEBATE_WEIGHT: float = 0.6
    MIN_VOLATILITY: float = 0.005                      # Was 0.006 — trade in calmer conditions
    MIN_HOLD_BARS: int = 6                             # Was 8 — allow faster exits when signal flips
    MIN_HOLD_BARS_TRENDING: int = 8                    # Was 10 — still give trends time but not too long
    MIN_HOLD_BARS_MEAN_REVERTING: int = 3              # Was 4 — MR trades are quick by nature
    EMA_ALPHA: float = 0.01                            # Was 0.008 — faster signal adaptation
    CONVICTION_THRESHOLD: float = 0.25                 # Was 0.28 — lower bar to enter with better signals
    DEAD_ZONE_LOW: float = 0.46                        # Was 0.48 — wider signal acceptance zone
    DEAD_ZONE_HIGH: float = 0.62                       # Was 0.64 — wider signal acceptance zone
    # ==================== Regime Detection ====================
    REGIME_METHOD: str = 'ensemble_hmm'
    HMM_N_COMPONENTS: int = 3
    HMM_ENSEMBLE_SIZE: int = 5
    HMM_ENSEMBLE: bool = True
    ENTROPY_HIGH_THRESHOLD: float = 2.5
    HURST_TREND_THRESHOLD: float = 0.45
    USE_ENTROPY_FEATURES: bool = True
    USE_DIVERGENCE: bool = True
    # NEW: Rolling window support for more responsive regime detection
    REGIME_SHORT_LOOKBACK: int = 96
    REGIME_SHORT_WEIGHT: float = 0.55                  # Was 0.6 — slightly more long-term bias
    REGIME_CONFIDENCE_MIN_SIZE_PCT: float = 0.45       # Was 0.3 — more engaged on marginal regimes
    REGIME_OVERRIDE_PERSISTENCE: float = 0.80          # Block counter-trend entries above this persistence
    # ==================== Multi-Regime Filters ====================
    MOM_THRESHOLD_TRENDING: float = 0.015
    MOM_THRESHOLD_MEAN_REVERTING: float = 0.03
    BREAKOUT_BOOST_FACTOR: float = 1.2
    # ==================== Model Training (Stacking + PPO) ====================
    PPO_TIMESTEPS: int = 200_000                       # Per-symbol PPO training budget
    PPO_RECURRENT: bool = True
    USE_CUSTOM_GTRXL: bool = True
    GTRXL_HIDDEN_SIZE: int = 256
    GTRXL_NUM_LAYERS: int = 2                           # 2 layers — stable for 53-dim obs
    GTRXL_NUM_HEADS: int = 8                            # 256/8=32 dims per head — good attention resolution
    GTRXL_MEMORY_LENGTH: int = 64                       # XL memory length per layer
    GTRXL_EVAL_CHUNK_SIZE: int = 64                     # Chunk size for batched evaluate_actions
    PPO_LSTM_HIDDEN_SIZE: int = 512
    PPO_N_LSTM_LAYERS: int = 4
    # === Learning rate: linear warmup then constant — cosine decayed too fast, killing learning in back half ===
    PPO_LEARNING_RATE: float = 2.5e-5                  # Was 3e-5 — slightly lower for stability; held constant after warmup
    PPO_LEARNING_RATE_MIN: float = 1.5e-5              # Was 5e-6 — higher floor keeps policy learning throughout training
    PPO_LR_WARMUP_FRAC: float = 0.05                   # Warmup first 5% of training (ramp 0 → PPO_LEARNING_RATE)
    PPO_ONLINE_LEARNING_RATE: float = 1.5e-5           # Online must be gentler than initial training
    # === Entropy: slightly more exploration to avoid premature convergence ===
    PPO_ENTROPY_COEFF: float = 0.015                   # Was 0.01 — policy converged too fast then couldn't escape suboptimal plateau
    # === Discount + GAE: tuned for 15min bars, ~6.5h trading day ===
    PPO_GAMMA: float = 0.995                           # Was 0.97 — higher gamma for longer horizon (2048-step episodes need longer credit assignment)
    PPO_GAE_LAMBDA: float = 0.95                       # Was 0.93 — smoother advantage estimation reduces variance
    # === Clipping: tighter to prevent the runaway updates we saw ===
    PPO_CLIP_RANGE: float = 0.12                       # Was 0.15 — tighter clip keeps policy updates conservative
    PPO_OVERRIDE_CONF: float = 0.85
    VF_COEF: float = 0.75                              # Was 0.5 — critic needs more gradient to fit 8-asset reward surface (explained_var was ~0.35)
    RISK_PENALTY_COEF: float = 0.10
    # === Reward shaping: lighter penalties so profitable actions dominate the signal ===
    VOL_PENALTY_COEF: float = 0.01                     # Was 0.02 — too heavy kills profitable vol strategies
    DD_PENALTY_COEF: float = 1.0                       # Was 2.0 — DD penalty was 2x the base return, drowning profit signal
    PPO_AUX_TASK: bool = True
    PPO_AUX_LOSS_WEIGHT: float = 0.10                  # Was 0.25 — aux loss is a known no-op (SB3 buffer lacks infos); minimize interference
    LABEL_HORIZON_BARS: int = 8                        # Forward return horizon for LightGBM labels (matches MIN_HOLD_BARS_TRENDING)
    NUM_BASE_MODELS: int = 20                          # LightGBM ensemble size
    PPO_ONLINE_UPDATE_TIMESTEPS: int = 75_000
    PPO_MAX_GRAD_NORM: float = 0.5                     # Was 0.4 — slightly relaxed (LR reduction already tames gradients)
    PPO_N_STEPS: int = 2048                            # 1 full episode per rollout
    PPO_BATCH_SIZE: int = 512                          # Was 256 — larger minibatches = lower gradient variance for transformers
    PPO_N_EPOCHS: int = 3                              # Was 4 — fewer passes reduces overfitting to single rollout (was causing high clip_fraction)
    MAX_EPISODE_STEPS: int = 2048                      # Full episode length
    # ==================== PORTFOLIO-LEVEL PPO ====================
    PORTFOLIO_PPO: bool = True
    MAX_LEVERAGE: float = 2.0
    PORTFOLIO_TIMESTEPS: int = 500_000                  # Full training budget — ~244 episodes
    PORTFOLIO_ONLINE_TIMESTEPS: int = 75_000
    ONLINE_PPO_UPDATE_HOURS: int = 4
    LIGHTGBM_PARAMS: Dict[str, Any] = Field(
        default_factory=lambda: {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1,
            'learning_rate': 0.025,                    # Was 0.03 — slightly slower for better generalization
            'num_leaves': 40,                          # Was 31 — more expressive
            'max_depth': 6,                            # NEW — prevent overfitting deep trees
            'min_data_in_leaf': 20,                    # NEW — prevent leaf overfitting
            'bagging_fraction': 0.75,                  # Was 0.7 — slightly more data per tree
            'bagging_freq': 3,                         # Was 5 — bag more often
            'feature_fraction': 0.85,                  # NEW — feature randomness for diversity
            'lambda_l2': 0.1,                          # NEW — L2 regularization
        }
    )
    # ==================== Advanced Feature Encoding (TFT) ====================
    USE_TFT_ENCODER: bool = True
    TFT_HIDDEN_SIZE: int = 32
    TFT_ATTENTION_HEADS: int = 4
    TFT_DROPOUT: float = 0.1
    TFT_HIDDEN_CONTINUOUS_SIZE: int = 16
    TFT_MAX_EPOCHS: int = 15
    TFT_FEATURE_DIM: int = 20                              # Encoder output features extracted per bar
    TFT_MAX_ENCODER_LENGTH: int = 200                      # Max lookback for TFT encoder window
    TFT_MAX_PREDICTION_LENGTH: int = 24                    # ~1 trading day of 15min bars
    TFT_LEARNING_RATE: float = 1e-3
    TFT_PERSIST_WEIGHTS: bool = True                       # Save/reload model weights across cache cycles
    # ==================== Walk-Forward Threshold Optimization ====================
    OPTUNA_TRIALS: int = 500
    OPTUNA_TIMEOUT: int = 900
    THRESHOLD_PENALTY_WEIGHT: float = 0.50
    # ==================== Reward Shaping (env.py / portfolio_env.py) ====================
    TURNOVER_COST_MULT: float = 0.03                   # Was 0.05 — lighter turnover penalty lets policy rebalance freely
    SORTINO_WEIGHT: float = 0.30                       # Was 0.25 — stronger Sortino drives Sharpe-optimal behavior
    SORTINO_ZERO_DD_BONUS: float = 0.03                # Was 0.02 — reward no-drawdown periods more
    PERSISTENCE_BONUS_SCALE: float = 0.10              # Was 0.15 — reduced since it's now action-dependent (earned, not free)
    CURRENT_REGIME: str = 'mean_reverting'
    # ==================== Operational & Misc ====================
    TRADING_INTERVAL: int = 45                         # Was 60 — faster reaction to market opportunities
    MONITOR_INTERVAL: int = 20                         # Was 30 — faster TP/SL checks for tighter execution
    LOOKBACK: int = 1200
    CACHE_TTL_DAYS: int = 30
    VIX_THRESHOLD: int = 28                            # Was 30 — slightly more cautious in high-VIX
    REQUEST_INTERVAL: float = 1.5                      # Was 2.0 — faster API polling
    DYNAMIC_THRESHOLD_UPDATE_DAYS: int = 5             # Was 7 — more responsive threshold adaptation
    MAX_ORDER_NOTIONAL_PCT: float = 0.85               # Was 0.80 — deploy more capital per order
    LGB_NUM_ITERATIONS: int = 250                      # NEW — was hardcoded 200 in stacking_ensemble.py
    # ==================== Debugging & Features ====================
    USE_LOCAL_TICKDB: bool = True
    TICKDB_ENGINE: str = 'arcticdb'
    USE_TENSORBOARD: bool = True
    BACKTEST_DEBUG: bool = False
    LOG_LEVEL: str = 'INFO'
    RUN_BACKTEST_ON_STARTUP: bool = False
    FORCE_PPO_RETRAIN: bool = False                    # Set True to force full retrain (PPO already trained with new params)
    DEBUG_SIGNAL_BLEND: bool = True
    # ==================== Local LLM Settings ====================
    USE_LOCAL_LLM: bool = True
    LOCAL_LLM_MODEL: str = 'sentiment-70b'
    LOCAL_LLM_FALLBACK: str = 'llama3.1:8b'
    OLLAMA_HOST: str = 'http://localhost:11434'
    NEWS_LOOKBACK_DAYS: int = 10
    # ==================== Causal RL Settings ====================
    USE_CAUSAL_RL: bool = True
    CAUSAL_DISCOVERY_METHOD: str = 'ges'  # FIX #43: Changed from 'pc' to match actual GES implementation
    CAUSAL_LLM_REFINEMENT: bool = True
    CAUSAL_PENALTY_WEIGHT: float = 0.40                # Was 0.34 — stronger causal influence
    CAUSAL_REWARD_FACTOR: float = 0.7                  # NEW — was hardcoded 0.5 in env.py
    # L33: CAUSAL_EDGE_THRESHOLD and COUNTERFACTUAL_SAMPLES removed (dead code, never read by GES fast-path)
    # ==================== Multi-Agent RL Settings ====================
    USE_MULTI_AGENT: bool = False  # M55 FIX: Feature is dead code (not imported anywhere) — disabled
    AGENT_HIERARCHY: str = 'regime-signal-execution'
    SIGNAL_AGENTS_MODE: str = 'per_symbol'
    USE_AGENT_DEBATE: bool = True
    MAPPO_TIMESTEPS: int = 500_000
    AGENT_REWARD_SHARE: float = 0.7
    # ==================== Trailing Stop Ratcheting ====================
    # Regime-adaptive frequency and sensitivity (used in alpaca.py monitor_positions)
    RATCHET_TRENDING_INTERVAL_SEC: int = 240           # Was 300 — slightly faster ratcheting for profit protection
    RATCHET_MEAN_REVERTING_INTERVAL_SEC: int = 480     # Was 600 — 8 min between MR ratchets
    RATCHET_TRENDING_MIN_ATR_MOVE: float = 0.4         # Was 0.5 — ratchet on smaller moves to lock in more profit
    RATCHET_MEAN_REVERTING_MIN_ATR_MOVE: float = 0.6   # Was 0.8 — tighter for MR too
    RATCHET_REGIME_FACTOR_TRENDING: float = 0.65       # Was 0.70 — slightly tighter trail in trends
    RATCHET_REGIME_FACTOR_MEAN_REVERTING: float = 1.20 # Was 1.35 — tighter MR trail for quicker profit capture
    RATCHET_PROFIT_PROTECTION_SLOPE: float = 0.8       # Was 1.0 — slower tightening lets trends breathe
    RATCHET_PROFIT_PROTECTION_MIN: float = 0.35        # Was 0.40 — tighter floor locks in more profit
    RATCHET_TIER1_PCT: float = 0.008                   # Was 0.01 — start ratcheting earlier (0.8% profit)
    RATCHET_TIER2_PCT: float = 0.025                   # Was 0.03 — transition to aggressive ratchet sooner
    RATCHET_TIER2_FLOOR_PCT: float = 0.8               # Was 1.0 — tighter floor locks in more
    RATCHET_TIER3_FLOOR_PCT: float = 0.4               # Was 0.5 — tighter aggressive floor
    # ==================== Broker Architecture ====================
    EXTENDED_HOURS: bool = True                          # Trade pre/post market
    FRACTIONAL_SHARES: bool = True                       # Allow fractional qty
    STREAM_RECONNECT_DELAY_SEC: int = 5                  # Websocket reconnect delay

    @model_validator(mode='after')
    def _symbols_subset_of_universe(self):
        missing = set(self.SYMBOLS) - set(self.UNIVERSE_CANDIDATES)
        if missing:
            raise ValueError(f"SYMBOLS contains tickers not in UNIVERSE_CANDIDATES: {missing}")
        return self
# ==================== Apply Validation ====================
# NOTE: CONFIG is a plain dict after this point. Direct mutations (e.g. from Gemini tuner)
# bypass Pydantic validation. The Gemini tuner validates via staged_changes + ABSOLUTE_BOUNDS
# after applying changes. Any other CONFIG mutators should re-validate similarly.
settings = TradingBotConfig.model_validate(CONFIG)
CONFIG.update(settings.model_dump())
print("✅ Pydantic validation passed — CONFIG is now fully type-safe")

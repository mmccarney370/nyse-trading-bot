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
from pydantic import BaseModel, Field
from typing import List, Dict, Any
class TradingBotConfig(BaseModel):
    # ==================== API & Broker Settings ====================
    API_KEY: str
    API_SECRET: str
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
    LIQUIDITY_WEIGHT: float = 0.15
    REGIME_TRENDING_WEIGHT: float = 0.45
    PERFORMANCE_WEIGHT: float = 0.30
    DIVERSIFICATION_WEIGHT: float = 0.10
    TIMEFRAMES: List[str] = Field(default_factory=lambda: ['15Min', '1H'])
    # ==================== Risk & Position Sizing ====================
    RISK_PER_TRADE: float = 0.012
    TRAILING_STOP_ATR: float = 4.0
    TAKE_PROFIT_ATR: float = 18.0
    RISK_PER_TRADE_MEAN_REVERTING: float = 0.006      # Was 0.002 — 3x more capital on mean-reversion
    RISK_PER_TRADE_TRENDING: float = 0.025             # Slightly tighter from 0.02975
    TRAILING_STOP_ATR_MEAN_REVERTING: float = 3.0      # Tighter from 3.5 — capture more MR profit
    TRAILING_STOP_ATR_TRENDING: float = 2.0            # Tighter from 2.5 — lock in trend gains
    TAKE_PROFIT_ATR_MEAN_REVERTING: float = 8.0        # Tighter from 12 — MR trades don't run far
    TAKE_PROFIT_ATR_TRENDING: float = 25.0             # Tighter from 32.5 — realize more trend profits
    MAX_POSITIONS: int = 6                             # Was 4 — more diversification with fractional shares
    MAX_ATR_VOLATILITY: float = 0.06
    KELLY_FRACTION: float = 0.50                       # Was 0.38 — more aggressive sizing
    COMMISSION_PER_SHARE: float = 0.0005
    SLIPPAGE: float = 0.0001
    ASSUMED_SPREAD: float = 0.0
    LIMIT_PRICE_OFFSET: float = 0.004                  # Tighter from 0.005 — dynamic adaptation handles rest
    DAILY_LOSS_THRESHOLD: float = -0.25                # Tighter from -0.3 — cut losses sooner
    API_FAILURE_THRESHOLD: int = 5000
    MIN_LIQUIDITY: int = 100000
    MAX_SECTOR_CONCENTRATION: float = 0.30             # Was 0.4 — more diversified
    MAX_TOTAL_RISK_PCT: float = 0.15                   # Was 0.12 — allow more total risk with better signals
    MAX_POSITION_VALUE_FRACTION: float = 0.25          # Was 0.20 — bigger conviction bets
    RISK_BUDGET_MULTIPLIER: float = 2.0                # Was 1.8 — more capital deployed
    # ==================== Signal Generation ====================
    MIN_CONFIDENCE: float = 0.82                       # Was 0.88 — take more trades with decent confidence
    PORTFOLIO_SENTIMENT_WEIGHT: float = 0.35           # Was 0.40 — slightly less sentiment dependency
    SENTIMENT_WEIGHT: float = 0.25                     # Was 0.20 — sentiment more integrated
    USE_LLM_DEBATE: bool = True
    LLM_DEBATE_WEIGHT: float = 0.6
    MIN_VOLATILITY: float = 0.006                      # Was 0.008 — trade in calmer conditions too
    MIN_HOLD_BARS: int = 6                             # Was 8
    MIN_HOLD_BARS_TRENDING: int = 6                    # Was 8 — exit faster if signal flips
    MIN_HOLD_BARS_MEAN_REVERTING: int = 3              # Was 4 — MR trades should be quick
    EMA_ALPHA: float = 0.008                           # Was 0.005 — faster signal adaptation
    CONVICTION_THRESHOLD: float = 0.28                 # NEW — was hardcoded 0.35 in risk.py
    DEAD_ZONE_LOW: float = 0.48                        # NEW — was hardcoded 0.50 in signals.py
    DEAD_ZONE_HIGH: float = 0.64                       # NEW — was hardcoded 0.68 in signals.py
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
    # ==================== Multi-Regime Filters ====================
    MOM_THRESHOLD_TRENDING: float = 0.015
    MOM_THRESHOLD_MEAN_REVERTING: float = 0.03
    BREAKOUT_BOOST_FACTOR: float = 1.2
    # ==================== Model Training (Stacking + PPO) ====================
    PPO_TIMESTEPS: int = 150_000                       # Was 100k — more training with fresh features
    PPO_RECURRENT: bool = True
    USE_CUSTOM_GTRXL: bool = True
    GTRXL_HIDDEN_SIZE: int = 256
    GTRXL_NUM_LAYERS: int = 4
    GTRXL_NUM_HEADS: int = 16
    PPO_LSTM_HIDDEN_SIZE: int = 512
    PPO_N_LSTM_LAYERS: int = 4
    PPO_LEARNING_RATE: float = 3e-4                    # Was 7.8e-5 — align with cosine schedule initial_lr
    PPO_LEARNING_RATE_MIN: float = 1e-6                # NEW — cosine schedule floor
    PPO_ONLINE_LEARNING_RATE: float = 5e-5             # NEW — was hardcoded 1e-5 (too conservative)
    PPO_ENTROPY_COEFF: float = 0.04                    # Was 0.023 — more exploration, especially early
    PPO_GAMMA: float = 0.96                            # Was 0.95 — slightly longer horizon
    PPO_GAE_LAMBDA: float = 0.93                       # Was 0.92 — slightly more bootstrapping
    PPO_CLIP_RANGE: float = 0.15                       # Was 0.13 — slightly wider clipping
    PPO_OVERRIDE_CONF: float = 0.93                    # Was 0.95 — let PPO override more often
    vf_coef: float = 0.5                               # Was 0.6 — reduce value function dominance
    RISK_PENALTY_COEF: float = 0.10
    VOL_PENALTY_COEF: float = 0.015                    # Was 0.0005 — unified for env.py and portfolio_env.py
    DD_PENALTY_COEF: float = 1.5                       # Was 0.065 — unified, moderate (env.py had 3.0!)
    PPO_AUX_TASK: bool = True
    PPO_AUX_LOSS_WEIGHT: float = 0.25                  # Was 0.2 — slightly stronger aux signal
    NUM_BASE_MODELS: int = 20                          # Was 15 — larger ensemble for better stacking
    PPO_ONLINE_UPDATE_TIMESTEPS: int = 75_000          # Was 100k — faster online adaptation
    PPO_MAX_GRAD_NORM: float = 0.5                     # NEW — was hardcoded 0.3 (too tight for recurrent)
    PPO_N_STEPS: int = 128                             # NEW — rollout buffer size per symbol
    PPO_BATCH_SIZE: int = 64                           # NEW — minibatch size
    PPO_N_EPOCHS: int = 5                              # NEW — was hardcoded 4 (slightly more training per rollout)
    # ==================== PORTFOLIO-LEVEL PPO ====================
    PORTFOLIO_PPO: bool = True
    MAX_LEVERAGE: float = 2.0                          # Was 2.15 — slightly more conservative
    PORTFOLIO_TIMESTEPS: int = 1_200_000               # Was 1M — more training for 8-symbol portfolio
    PORTFOLIO_ONLINE_TIMESTEPS: int = 75_000           # Was 100k — faster online updates
    ONLINE_PPO_UPDATE_HOURS: int = 4                   # Was 6 — more frequent adaptation
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
    # ==================== Walk-Forward Threshold Optimization ====================
    OPTUNA_TRIALS: int = 500
    OPTUNA_TIMEOUT: int = 900
    THRESHOLD_PENALTY_WEIGHT: float = 0.50
    # ==================== Reward Shaping (env.py / portfolio_env.py) ====================
    TURNOVER_COST_MULT: float = 0.3                    # NEW — was hardcoded 0.0005*1000=0.5 in env.py
    SORTINO_WEIGHT: float = 0.20                       # NEW — was hardcoded 0.25 in env.py
    SORTINO_ZERO_DD_BONUS: float = 0.03                # NEW — was hardcoded 0.05 in env.py
    PERSISTENCE_BONUS_SCALE: float = 0.4               # NEW — was hardcoded 1.0 in env.py (way too high)
    CURRENT_REGIME: str = 'mean_reverting'
    # ==================== Operational & Misc ====================
    TRADING_INTERVAL: int = 45                         # Was 60 — faster trading cycles
    MONITOR_INTERVAL: int = 45                         # Was 60 — faster monitor heartbeat
    LOOKBACK: int = 1200
    CACHE_TTL_DAYS: int = 30
    VIX_THRESHOLD: int = 28                            # Was 30 — slightly more cautious in high-VIX
    REQUEST_INTERVAL: float = 1.5                      # Was 2.0 — faster API polling
    DYNAMIC_THRESHOLD_UPDATE_DAYS: int = 5             # Was 7 — more responsive threshold adaptation
    MAX_ORDER_NOTIONAL_PCT: float = 0.80               # Was 0.75 — slightly more capital per order
    LGB_NUM_ITERATIONS: int = 250                      # NEW — was hardcoded 200 in stacking_ensemble.py
    # ==================== Debugging & Features ====================
    USE_LOCAL_TICKDB: bool = True
    TICKDB_ENGINE: str = 'arcticdb'
    USE_TENSORBOARD: bool = True
    BACKTEST_DEBUG: bool = True
    LOG_LEVEL: str = 'INFO'
    RUN_BACKTEST_ON_STARTUP: bool = False
    DEBUG_SIGNAL_BLEND: bool = True
    # ==================== Local LLM Settings ====================
    USE_LOCAL_LLM: bool = True
    LOCAL_LLM_MODEL: str = 'sentiment-70b'
    LOCAL_LLM_FALLBACK: str = 'llama3.1:8b'
    OLLAMA_HOST: str = 'http://localhost:11434'
    NEWS_LOOKBACK_DAYS: int = 10
    # ==================== Causal RL Settings ====================
    USE_CAUSAL_RL: bool = True
    CAUSAL_DISCOVERY_METHOD: str = 'pc'
    CAUSAL_LLM_REFINEMENT: bool = True
    CAUSAL_PENALTY_WEIGHT: float = 0.40                # Was 0.34 — stronger causal influence
    CAUSAL_REWARD_FACTOR: float = 0.7                  # NEW — was hardcoded 0.5 in env.py
    CAUSAL_EDGE_THRESHOLD: float = 0.30                # Was 0.35 — discover more edges
    COUNTERFACTUAL_SAMPLES: int = 8                    # Was 5 — more robust estimates
    # ==================== Multi-Agent RL Settings ====================
    USE_MULTI_AGENT: bool = True
    AGENT_HIERARCHY: str = 'regime-signal-execution'
    SIGNAL_AGENTS_MODE: str = 'per_symbol'
    USE_AGENT_DEBATE: bool = True
    MAPPO_TIMESTEPS: int = 500_000
    AGENT_REWARD_SHARE: float = 0.7
    # ==================== Trailing Stop Ratcheting ====================
    # Regime-adaptive frequency and sensitivity (used in alpaca.py monitor_positions)
    RATCHET_TRENDING_INTERVAL_SEC: int = 120           # Was 180 — faster ratchet in trends
    RATCHET_MEAN_REVERTING_INTERVAL_SEC: int = 360     # Was 540 — faster in MR too
    RATCHET_TRENDING_MIN_ATR_MOVE: float = 0.3         # Was 0.4 — more aggressive
    RATCHET_MEAN_REVERTING_MIN_ATR_MOVE: float = 0.6   # Was 0.8 — tighter
    RATCHET_REGIME_FACTOR_TRENDING: float = 0.45        # Was 0.525 — tighter trail in trends
    RATCHET_REGIME_FACTOR_MEAN_REVERTING: float = 1.15 # Was 1.35 — tighter in MR
    RATCHET_PROFIT_PROTECTION_SLOPE: float = 1.5       # Was 1.25 — more aggressive tightening on gains
    RATCHET_PROFIT_PROTECTION_MIN: float = 0.30         # Was 0.35 — lower floor = tighter on big gains
    # ==================== Broker Architecture ====================
    EXTENDED_HOURS: bool = True                          # Trade pre/post market
    FRACTIONAL_SHARES: bool = True                       # Allow fractional qty
    STREAM_RECONNECT_DELAY_SEC: int = 5                  # Websocket reconnect delay
# ==================== Apply Validation ====================
settings = TradingBotConfig.model_validate(CONFIG)
CONFIG.update(settings.model_dump())
print("✅ Pydantic validation passed — CONFIG is now fully type-safe")

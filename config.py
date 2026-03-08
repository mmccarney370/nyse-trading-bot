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
    RISK_PER_TRADE: float = 0.008
    TRAILING_STOP_ATR: float = 5.0
    TAKE_PROFIT_ATR: float = 15.0
    RISK_PER_TRADE_MEAN_REVERTING: float = 0.002
    RISK_PER_TRADE_TRENDING: float = 0.02975
    TRAILING_STOP_ATR_MEAN_REVERTING: float = 3.5
    TRAILING_STOP_ATR_TRENDING: float = 2.5
    TAKE_PROFIT_ATR_MEAN_REVERTING: float = 12.0
    TAKE_PROFIT_ATR_TRENDING: float = 32.5125
    MAX_POSITIONS: int = 4
    MAX_ATR_VOLATILITY: float = 0.05
    KELLY_FRACTION: float = 0.38
    COMMISSION_PER_SHARE: float = 0.0005
    SLIPPAGE: float = 0.0001
    ASSUMED_SPREAD: float = 0.0
    LIMIT_PRICE_OFFSET: float = 0.005
    DAILY_LOSS_THRESHOLD: float = -0.3
    API_FAILURE_THRESHOLD: int = 5000
    MIN_LIQUIDITY: int = 100000
    MAX_SECTOR_CONCENTRATION: float = 0.4
    MAX_TOTAL_RISK_PCT: float = 0.12
    MAX_POSITION_VALUE_FRACTION: float = 0.20
    RISK_BUDGET_MULTIPLIER: float = 1.8
    # ==================== Signal Generation ====================
    MIN_CONFIDENCE: float = 0.88
    PORTFOLIO_SENTIMENT_WEIGHT: float = 0.40
    SENTIMENT_WEIGHT: float = 0.20
    USE_LLM_DEBATE: bool = True
    LLM_DEBATE_WEIGHT: float = 0.6
    MIN_VOLATILITY: float = 0.008
    MIN_HOLD_BARS: int = 8
    MIN_HOLD_BARS_TRENDING: int = 8
    MIN_HOLD_BARS_MEAN_REVERTING: int = 4
    EMA_ALPHA: float = 0.005
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
    REGIME_SHORT_LOOKBACK: int = 96 # ~1 trading day (15min bars)
    REGIME_SHORT_WEIGHT: float = 0.6 # 0.0 = ignore short-term, 1.0 = ignore long-term
    # NEW: Minimum position size % when persistence is marginal (0.5)
    # Prevents zero-size positions on weak regime consensus
    REGIME_CONFIDENCE_MIN_SIZE_PCT: float = 0.3 # e.g. 30% min size at persistence=0.5
    # ==================== Multi-Regime Filters ====================
    MOM_THRESHOLD_TRENDING: float = 0.015
    MOM_THRESHOLD_MEAN_REVERTING: float = 0.03
    BREAKOUT_BOOST_FACTOR: float = 1.2
    # ==================== Model Training (Stacking + PPO) ====================
    PPO_TIMESTEPS: int = 100000
    PPO_RECURRENT: bool = True
    USE_CUSTOM_GTRXL: bool = True
    GTRXL_HIDDEN_SIZE: int = 256
    GTRXL_NUM_LAYERS: int = 4
    GTRXL_NUM_HEADS: int = 16
    PPO_LSTM_HIDDEN_SIZE: int = 512
    PPO_N_LSTM_LAYERS: int = 4
    PPO_LEARNING_RATE: float = 7.8e-5
    PPO_ENTROPY_COEFF: float = 0.023
    PPO_GAMMA: float = 0.95
    PPO_GAE_LAMBDA: float = 0.92
    PPO_CLIP_RANGE: float = 0.13
    PPO_OVERRIDE_CONF: float = 0.95
    vf_coef: float = 0.6
    RISK_PENALTY_COEF: float = 0.115
    VOL_PENALTY_COEF: float = 0.0005
    DD_PENALTY_COEF: float = 0.065
    PPO_AUX_TASK: bool = True
    PPO_AUX_LOSS_WEIGHT: float = 0.2
    NUM_BASE_MODELS: int = 15
    PPO_ONLINE_UPDATE_TIMESTEPS: int = 100_000
    # ==================== PORTFOLIO-LEVEL PPO ====================
    PORTFOLIO_PPO: bool = True
    MAX_LEVERAGE: float = 2.15
    PORTFOLIO_TIMESTEPS: int = 1_000_000
    PORTFOLIO_ONLINE_TIMESTEPS: int = 100_000
    ONLINE_PPO_UPDATE_HOURS: int = 6
    LIGHTGBM_PARAMS: Dict[str, Any] = Field(
        default_factory=lambda: {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1,
            'learning_rate': 0.03,
            'num_leaves': 31,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
        }
    )
    # ==================== Advanced Feature Encoding (TFT) ====================
    USE_TFT_ENCODER: bool = True
    # ==================== Walk-Forward Threshold Optimization ====================
    OPTUNA_TRIALS: int = 500
    OPTUNA_TIMEOUT: int = 900
    THRESHOLD_PENALTY_WEIGHT: float = 0.50
    # ==================== Operational & Misc ====================
    TRADING_INTERVAL: int = 60
    MONITOR_INTERVAL: int = 60
    LOOKBACK: int = 1200
    CACHE_TTL_DAYS: int = 30
    VIX_THRESHOLD: int = 30
    REQUEST_INTERVAL: float = 2.0
    DYNAMIC_THRESHOLD_UPDATE_DAYS: int = 7
    MAX_ORDER_NOTIONAL_PCT: float = 0.75
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
    CAUSAL_PENALTY_WEIGHT: float = 0.34
    CAUSAL_EDGE_THRESHOLD: float = 0.35
    COUNTERFACTUAL_SAMPLES: int = 5
    # ==================== Multi-Agent RL Settings ====================
    USE_MULTI_AGENT: bool = True
    AGENT_HIERARCHY: str = 'regime-signal-execution'
    SIGNAL_AGENTS_MODE: str = 'per_symbol'
    USE_AGENT_DEBATE: bool = True
    MAPPO_TIMESTEPS: int = 500_000
    AGENT_REWARD_SHARE: float = 0.7
    # ==================== Trailing Stop Ratcheting ====================
    # Regime-adaptive frequency and sensitivity (used in alpaca.py monitor_positions)
    RATCHET_TRENDING_INTERVAL_SEC: int = 180           # ~3 min (heartbeat)
    RATCHET_MEAN_REVERTING_INTERVAL_SEC: int = 540     # ~9 min
    RATCHET_TRENDING_MIN_ATR_MOVE: float = 0.4         # aggressive in trends
    RATCHET_MEAN_REVERTING_MIN_ATR_MOVE: float = 0.8   # conservative in chop
    RATCHET_REGIME_FACTOR_TRENDING: float = 0.525       # tighter trail in strong trends
    RATCHET_REGIME_FACTOR_MEAN_REVERTING: float = 1.35 # looser in chop
    RATCHET_PROFIT_PROTECTION_SLOPE: float = 1.25      # how aggressively to tighten on gains
    RATCHET_PROFIT_PROTECTION_MIN: float = 0.35         # floor for protection factor
# ==================== Apply Validation ====================
settings = TradingBotConfig.model_validate(CONFIG)
CONFIG.update(settings.model_dump())
print("✅ Pydantic validation passed — CONFIG is now fully type-safe")

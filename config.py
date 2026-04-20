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
    # Portfolio-mode weight for the nightly-trained LightGBM stacking ensemble.
    # Applied as a multiplicative per-symbol factor (agreement with PPO → boost,
    # disagreement → dampen) inside generate_portfolio_actions. Set 0.0 to disable.
    PORTFOLIO_META_WEIGHT: float = 0.2
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
    # Apr-19 audit: the aux volatility head trains on misaligned targets —
    # AuxVolatilityCallback captures vol_target per env step, but SB3's
    # RolloutBuffer subsamples/reorders observations before train(), so the
    # i-th target does not match the i-th buffer row. This made the aux loss
    # effectively random noise. Disabled by default. The env still emits
    # `volatility_target` in info so a future observation-feature-style
    # re-introduction is easy (add vol_target to the obs vector instead).
    PPO_AUX_TASK: bool = False
    PPO_AUX_LOSS_WEIGHT: float = 0.10                  # Irrelevant while PPO_AUX_TASK=False; kept for rollback
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
    # Swapped Apr 17: 70B was causing GPU OOM hangs (5-hour trading halt + repeated
    # TimeoutErrors). 8B is 10× faster, almost never OOMs, and still produces usable
    # sentiment. 70B kept as fallback for the rare case 8B fails.
    LOCAL_LLM_MODEL: str = 'llama3.1:8b'
    LOCAL_LLM_FALLBACK: str = 'sentiment-70b'
    OLLAMA_HOST: str = 'http://localhost:11434'
    NEWS_LOOKBACK_DAYS: int = 10
    # ==================== Causal RL Settings ====================
    USE_CAUSAL_RL: bool = True
    CAUSAL_DISCOVERY_METHOD: str = 'ges'  # FIX #43: Changed from 'pc' to match actual GES implementation
    CAUSAL_LLM_REFINEMENT: bool = True
    CAUSAL_PENALTY_WEIGHT: float = 0.40                # Was 0.34 — stronger causal influence
    CAUSAL_REWARD_FACTOR: float = 0.7                  # NEW — was hardcoded 0.5 in env.py
    # Seed the portfolio replay buffer by replaying recent historical bars through the
    # persistent PortfolioEnv at startup. Without this, the portfolio causal graph stays
    # in fallback mode until ≥100 live trades accumulate (takes weeks at typical cadence).
    # Set to 0 to disable bootstrap entirely.
    CAUSAL_BOOTSTRAP_STEPS: int = 1500
    # Gaussian std injected into bootstrap actions for causal discovery. Converged PPO
    # is near-deterministic (std ~0.08) so GES finds 0 edges; this provides the
    # treatment variance GES needs to identify action→reward structure.
    CAUSAL_BOOTSTRAP_NOISE_SIGMA: float = 0.4
    # Cap GES input width. Portfolio observations are 460-dim so raw buffer samples hit
    # 462 cols → GES runs for hours (O(p^3·n)). Cap selects the top-variance features +
    # always keeps action/reward. Per-symbol buffers are already below this so unaffected.
    CAUSAL_MAX_FEATURES: int = 58
    # ==================== Intraday Risk Pacing ====================
    # Graduated multiplier on the CVaR risk budget based on today's state. Tier1 fires
    # at small drawdowns (scale 0.5×), Tier2 at larger ones (0.2×). Consecutive-loss
    # floor kicks in after N losing closes today. All cumulative with the existing
    # hard daily-loss halt (DAILY_LOSS_THRESHOLD).
    RISK_PACING_ENABLED: bool = True
    RISK_PACING_TIER1_LOSS: float = -0.005   # -0.5% intraday → scale 0.5×
    RISK_PACING_TIER1_SCALE: float = 0.5
    RISK_PACING_TIER2_LOSS: float = -0.015   # -1.5% intraday → scale 0.2×
    RISK_PACING_TIER2_SCALE: float = 0.2
    RISK_PACING_CONSECUTIVE_LOSSES: int = 3  # after N in a row today
    RISK_PACING_CONSECUTIVE_LOSS_SCALE: float = 0.3
    # ==================== S1: Meta-Labeling Filter ====================
    # Lopez-de-Prado-style secondary gate. Nightly-trained LightGBM classifier
    # answers "given PPO says go <dir> on <sym>, is this likely a winner?" Any
    # candidate weight whose P(win) < MIN_PROB is zeroed (default) or scaled
    # (set MODE='scale' for softer behavior). Pass-through when model hasn't yet
    # reached MIN_TRAIN samples — strictly additive, never blocks when unsure.
    META_FILTER_ENABLED: bool = True
    META_FILTER_MIN_TRAIN: int = 30      # need ≥N closed trades to fit
    META_FILTER_MIN_PROB: float = 0.33   # reject if P(win) below this (lowered from 0.40 Apr 16 —
                                         # base_wr=0.308 meant 0.40 rejected ~69% of candidates;
                                         # 0.33 is "clearly below baseline" only. Gemini can tune further)
    META_FILTER_MODE: str = "zero"       # "zero" (hard reject) or "scale" (0.2× dampen)
    # ==================== S2: Regime-Detector Ensemble ====================
    # Legacy HMM alone labeled every bar mean_reverting even in clear trends. The
    # ensemble gives 4 classical signals a vote — slope, ADX, lag-1 autocorrelation,
    # range expansion — combined with HMM as a 5th voter when available.
    REGIME_DETECTOR_MODE: str = "ensemble"  # "ensemble" (default) or "hmm" (legacy)
    # Voter thresholds — raise → more mean-reverting; lower → more trending
    REGIME_SLOPE_THRESHOLD: float = 0.025                # normalized slope over window
    REGIME_ADX_TREND_THRESHOLD: float = 25.0             # ADX ≥ this → trend
    REGIME_ADX_NO_TREND_THRESHOLD: float = 20.0          # ADX ≤ this → mean-revert
    REGIME_AUTOCORR_TREND_THRESHOLD: float = 0.06        # |lag-1 rho| ≥ this → signal
    REGIME_RANGE_EXPANSION_THRESHOLD: float = 1.30       # short/long ATR ratio
    REGIME_RANGE_CONTRACTION_THRESHOLD: float = 0.85
    # Ensemble aggregation: how much of the vote mass must be trending to declare one
    REGIME_ENSEMBLE_TREND_DOMINANCE: float = 0.50
    # Force-refresh persisted regime cache on startup. Normally the version-marker
    # in regime_cache.json handles this automatically, but setting this True for
    # one run forces a clean recompute via detect_regime (useful after code changes).
    REGIME_FORCE_REFRESH_ON_STARTUP: bool = False
    # ==================== A1: Cross-Sectional Momentum Gate ====================
    # Daily re-rank of the current universe by momentum + volume + drawdown. Applies
    # a multiplicative gate to target weights so today's winners get a boost and
    # today's laggards get dampened. Preserves PPO obs shape (no retrain) while
    # capturing cross-sectional alpha. Set WEIGHT=0 to disable.
    CROSS_SECTIONAL_WEIGHT: float = 1.0        # 0.0 disables; 1.0 full effect
    CROSS_SECTIONAL_MAX_MULT: float = 1.25     # top-tercile boost
    CROSS_SECTIONAL_MIN_MULT: float = 0.50     # bottom-tercile dampen
    CROSS_SECTIONAL_NEUTRAL_BAND: float = 0.25  # z-score within [-0.25, 0.25] = no change
    # ==================== B5: Anti-Earnings Filter ====================
    # Auto-flat positions N days before earnings and block new entries during a
    # pre/post blackout window. Cheapest WR boost: don't trade through earnings.
    EARNINGS_FILTER_ENABLED: bool = True
    EARNINGS_BLACKOUT_PRE_DAYS: int = 2     # no NEW entries in last 2 days pre-earnings
    EARNINGS_BLACKOUT_POST_DAYS: int = 1    # no NEW entries for 1 day after
    EARNINGS_CLOSE_PRE_DAYS: int = 1        # close OPEN positions 1 day before earnings
    # ==================== B4: Sentiment Velocity ====================
    # Level alone saturates (0.2-0.6 range mostly). Velocity = Δ level over past N hours
    # captures information ARRIVAL. Applied direction-aware so positive Δ boosts longs
    # and dampens shorts. Weight=0.15 means +0.5 velocity gives ~1.075x boost.
    SENTIMENT_VELOCITY_WEIGHT: float = 0.15
    SENTIMENT_VELOCITY_LOOKBACK_HOURS: int = 4
    # Max seconds for ALL sentiment calls in one cycle. If Ollama hangs (GPU crash,
    # OOM, server freeze), this prevents the entire trading loop from freezing.
    # Observed: 5-hour hang on Apr 17 from a single stuck Ollama call.
    SENTIMENT_GATHER_TIMEOUT_SEC: int = 300  # 5 min max for all 8 symbols
    # ==================== AC: Correlation-Aware Sizing ====================
    # Discount each position by its average correlation to same-sign peers. Prevents
    # 4 correlated tech longs from becoming one effective mega-position.
    CROWDING_DISCOUNT_ENABLED: bool = True
    CROWDING_DISCOUNT_THRESHOLD: float = 0.5   # only kicks in above this avg correlation
    CROWDING_DISCOUNT_STRENGTH: float = 0.5    # slope of discount (1.0 = aggressive)
    CROWDING_DISCOUNT_MIN_FACTOR: float = 0.4  # hard floor, never below this
    # ==================== B2: Adverse Selection Detector ====================
    # Tracks post-fill price drift per symbol. If recent fills were consistently
    # followed by adverse moves, we're being picked off → dampen that symbol's
    # weight until drift recovers. Rolling 20-fill window at 5-min offset.
    ADVERSE_SELECTION_ENABLED: bool = True
    ADVERSE_SELECTION_THRESHOLD: float = -0.002  # -20bp consistent drift = toxic
    ADVERSE_SELECTION_MAX_PENALTY: float = 0.5   # up to 50% weight reduction
    # ==================== BPS: Bayesian Per-Symbol Sizing ====================
    # Beta(α, β) posterior per symbol updated after every closed trade. Scales
    # each symbol's target weight by expected return. SMCI at 100% WR → up to 1.6×;
    # TSLA at 25% WR → down to 0.4×. Small-sample shrinkage prevents over-reacting.
    BAYESIAN_SIZING_ENABLED: bool = True
    BAYESIAN_SIZING_MIN_MULT: float = 0.4
    BAYESIAN_SIZING_MAX_MULT: float = 1.6
    BAYESIAN_SIZING_REFERENCE_EV: float = 0.003   # +0.3% ev_per_trade → full boost
    BAYESIAN_SIZING_SHRINKAGE_N: int = 8          # need this many trades before full weight
    # KELLY: mathematically-optimal sizing via f* = (pb - q)/b with fractional safety.
    # Default is ¼ Kelly — industry-standard conservative multiplier.
    # reference_kelly=0.08 means a ¼-Kelly f* of 0.08 triggers full max_mult boost.
    BAYESIAN_SIZING_METHOD: str = "kelly"         # "kelly" or "ev" (legacy)
    BAYESIAN_SIZING_KELLY_FRACTION: float = 0.25
    BAYESIAN_SIZING_REFERENCE_KELLY: float = 0.08
    # ==================== ESP: Slippage-Prediction Veto ====================
    # Learn realized slippage per (symbol, hour, size). At entry, predict expected
    # slippage; if > edge × safety multiple → skip or dampen.
    SLIPPAGE_VETO_ENABLED: bool = True
    SLIPPAGE_VETO_MULTIPLE: float = 2.0    # pred_bps > edge × this → veto (raised from 1.2 Apr-19 after 136 vetoes/3 fills)
    SLIPPAGE_VETO_SCALE: float = 0.5       # dampen to 50% when vetoed (was 0.3 — too punitive given calibration noise)
    SLIPPAGE_VETO_MIN_SAMPLES: int = 5     # require ≥N fills in the specific bucket before veto fires
    # ==================== Walk-forward OOS acceptance ====================
    # Pre-Apr-19 the accept-gate was `oos > 0.0` which silently rejected windows
    # with small negative OOS even when IS was modest and the gap was tight.
    OOS_SHARPE_ACCEPT_FLOOR: float = -0.25     # allow slightly-negative OOS …
    OOS_ACCEPT_MAX_GAP_RATIO: float = 0.35     # …only when IS→OOS gap ratio is below this
    # ==================== Broker trailing-stop compatibility ====================
    # Alpaca rejects trailing stops on fractional qty. Round DOWN to whole shares
    # when the requested size ≥ 1 so the native trailing stop covers the full
    # position. Only genuine sub-1-share positions fall back to software TP.
    PREFER_WHOLE_SHARES_FOR_TS: bool = True
    # Hard-to-borrow symbols where Alpaca only accepts DAY TIF on trailing stops.
    # The broker dynamically adds to this set when it sees an HTB rejection.
    HTB_SYMBOLS: List[str] = Field(default_factory=lambda: ["PLTR"])
    # ==================== Signal-layer audit fixes (Apr-19) ====================
    # Gate cascade short-circuit: below this gate_mult, skip remaining gates
    # and log only the earliest veto reason (stops silent 0.3 × 0.5 × 0.3 cascade).
    GATE_SHORT_CIRCUIT_THRESHOLD: float = 0.01
    # Meta-filter is pre-fit until ~2 weeks of closed trades exist. Apply this
    # dampener during the pre-fit window rather than a silent free pass.
    META_FILTER_PREFIT_DAMPENER: float = 0.8
    # Equity-curve drawdown scale: skip downscale for positions aligned with
    # the dominant regime (shorts in trending_down / longs in trending_up).
    EQ_SCALE_DIRECTION_AWARE: bool = True
    # Apply the portfolio_rebalancer causal penalty AFTER min-hold / as the last
    # multiplier before final renorm (Apr-19 audit).
    CAUSAL_PENALTY_AFTER_GATES: bool = True
    # ==================== Risk/Exit audit fixes (Apr-19) ====================
    # Loss-tighten has its own short throttle so a failing thesis is cut
    # quickly, not held 180-540s for the profit-ratchet cooldown.
    RATCHET_LOSS_TIGHTEN_MIN_INTERVAL_SEC: int = 45
    # Bayesian sizer "proven winner" unlock: when all three thresholds are met
    # the max multiplier is raised from 1.6 to 2.0.
    BAYESIAN_PROVEN_N: int = 20
    BAYESIAN_PROVEN_P_WIN: float = 0.60
    BAYESIAN_PROVEN_PERSISTENCE: float = 0.85
    BAYESIAN_PROVEN_MAX_MULT: float = 2.0
    # CVaR: partition insufficient-data symbols out of the optimization instead
    # of falling back to uniform for everyone.
    CVAR_MIN_QUALIFIED_SYMBOLS: int = 3
    CVAR_INSUFFICIENT_BUDGET_SHARE: float = 0.15
    # Leverage cap flex: allow up to +20% gross exposure when average regime
    # persistence is high, so the persistence boost isn't erased by rescale.
    LEVERAGE_PERSISTENCE_FLEX_MAX: float = 0.20
    LEVERAGE_PERSISTENCE_FLEX_START: float = 0.70
    # ==================== Training audit fixes (Apr-19) ====================
    # Walk-forward OOS absolute-gap cap. Only applied when OOS is itself
    # below `OOS_SHARPE_STRONG_ACCEPT` — a window with OOS=5.0 and IS=11.5
    # has a 6.5 abs gap but is excellent; we never reject strong-OOS
    # windows because IS was even better.
    OOS_ACCEPT_MAX_ABS_GAP: float = 0.5
    OOS_SHARPE_STRONG_ACCEPT: float = 1.0
    # Opportunity-cost reward term — breaks the "idle is strictly rewarded"
    # bias under calm regimes by penalising low |position| lightly.
    OPPORTUNITY_COST_COEF: float = 0.0001
    # Stacking ensemble label lag (bars shifted beyond horizon) to avoid the
    # same-bar lookback leak where indicators computed on bar t are scored
    # against bar t's own forward return.
    LABEL_HORIZON_LAG_BARS: int = 1
    # Threshold walk-forward runs over the stacking-holdout tail only, since
    # stacking ensemble is trained on the earlier portion of the dataframe.
    STACKING_HOLDOUT_FRAC: float = 0.30
    # Walk-forward window count when operating over the holdout tail.
    WALK_FORWARD_N_WINDOWS: int = 3
    # GTrXL inference rolling-window size — tokens of self-attention during
    # single-step inference (matches training chunk length semantics).
    GTRXL_INFERENCE_WINDOW: int = 32
    # ==================== Data / execution audit fixes (Apr-19) ====================
    # Reject extended-hours bars from the 15Min store so HMM regime
    # detection and feature stats stay calibrated on regular hours.
    STREAM_REJECT_EXTENDED_HOURS: bool = True
    # TFT features are zeroed out when the fraction of valid rows falls
    # below this threshold — prevents zero-padded neutral vectors from
    # leaking into the PPO observation as if they were real signal.
    TFT_MIN_VALID_FRAC: float = 0.5
    # Fractional-remainder cleanup throttle (prevents log spam + API hammer
    # when shares are stuck as held_for_orders by a stale pending close).
    FRAC_CLEANUP_MIN_INTERVAL_SEC: int = 300
    # ==================== Tuner provider selection (Apr-19) ====================
    # Switch between the Gemini 2.5 Flash tuner and the Claude Opus 4.7 tuner.
    # Claude is the quality default when ANTHROPIC_API_KEY is configured;
    # the dispatch layer falls back to Gemini on missing key or API error.
    # Values: "claude" | "gemini".
    TUNER_PROVIDER: str = "claude"
    # Claude-specific model selection. Opus gives the deepest reasoning;
    # Sonnet 4.6 is ~6× cheaper for ~90% of the strategic quality.
    CLAUDE_TUNER_MODEL: str = "claude-opus-4-7"
    # Extended-thinking budget in tokens. More budget = longer reasoning
    # but higher latency and a small cost bump. 16K is a good balance for
    # a nightly tuning call that must weigh ~15 parameters across 20+
    # groups. The SDK enforces thinking_budget < max_tokens.
    CLAUDE_TUNER_THINKING_BUDGET: int = 16000
    # Budget for the visible (non-thinking) output.
    CLAUDE_TUNER_MAX_OUTPUT: int = 4000
    # ==================== PSD: PPO–Stacking Divergence Gate ====================
    # If PPO says go hard in one direction but stacking ensemble predicts the
    # OPPOSITE direction, that's high-conviction disagreement — historically
    # unprofitable. Dampen rather than blindly follow PPO.
    DIVERGENCE_GATE_ENABLED: bool = True
    DIVERGENCE_GATE_SCALE: float = 0.5     # dampen to 50% on strong disagreement
    DIVERGENCE_MIN_WEIGHT: float = 0.03    # PPO weight must exceed this to trigger
    DIVERGENCE_MIN_META: float = 0.20      # |meta_signed| must exceed this to trigger
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
    # ==================== S3: Asymmetric Loss-Side Tightening ====================
    # When a position is underwater and hasn't gone meaningfully green, tighten the
    # trailing stop so we cut losses faster instead of letting a losing trade drift
    # all the way to the original wide stop. Gated by MFE so we don't penalize
    # winning trades that pulled back.
    RATCHET_LOSS_TIGHTEN_ENABLED: bool = True
    RATCHET_LOSS_TIGHTEN_THRESHOLD: float = -0.007     # fires at -0.7% unrealized
    RATCHET_LOSS_TIGHTEN_FACTOR: float = 0.55          # trail → 55% of original width
    RATCHET_LOSS_TIGHTEN_MFE_MAX: float = 0.004        # skip if MFE ever exceeded +0.4%
    # ==================== REX: Regime-Conditional Exits ====================
    # Apply a multiplier on top of the regime-base ATR multipliers, based on whether
    # the trade ALIGNS with or OPPOSES the current regime direction. Aligned trades
    # (long in uptrend, short in downtrend) get wider TP + trail — let winners run.
    # Counter-trend trades get tighter TP + trail — take profits fast, accept quick
    # exits. Mean-reverting regime → tighter TP (expect reversal). Strictly additive:
    # REX_ENABLED=False returns multipliers to 1.0 neutral.
    REX_ENABLED: bool = True
    REX_ALIGN_TP_MULT: float = 1.4       # Long in uptrend: TP ×1.4 of regime-base
    REX_ALIGN_TRAIL_MULT: float = 1.25   # Long in uptrend: trail ×1.25 of regime-base
    REX_OPPOSE_TP_MULT: float = 0.7      # Short in uptrend: TP ×0.7 (quick profit)
    REX_OPPOSE_TRAIL_MULT: float = 0.75  # Short in uptrend: trail ×0.75 (tight stop)
    REX_MR_TP_MULT: float = 0.85         # Mean-reverting: TP ×0.85 (fast profit)
    REX_MR_TRAIL_MULT: float = 0.92      # Mean-reverting: trail ×0.92 (modest tighten)
    # ==================== LIQ: Liquidity-Scaled Sizing ====================
    # Scale down weights when position notional > small % of symbol's ADV.
    # Prevents market-impact slippage as equity grows or in thin-liquidity tickers.
    # Extended hours (pre/post market) tightens thresholds by EH_FACTOR.
    LIQUIDITY_SCALER_ENABLED: bool = True
    LIQUIDITY_WARN_THRESHOLD: float = 0.001   # 0.1% of ADV → no scaling
    LIQUIDITY_HARD_THRESHOLD: float = 0.01    # 1.0% of ADV → floor min_mult
    LIQUIDITY_MIN_MULT: float = 0.3           # floor when hard-threshold breached
    LIQUIDITY_EH_FACTOR: float = 5.0          # extended-hours thresholds / 5
    # ==================== RETRAIN-GUARD: PPO Retrain Safety ====================
    # Before every nightly PPO retrain, checkpoint the current model + compute a
    # baseline validation score (deterministic rollout on live env). After retrain,
    # recompute the score. If the new model is materially worse (abs AND rel drop),
    # ROLLBACK to the checkpoint. Prevents bad retrains from silently degrading
    # every downstream layer. Runs only on update_portfolio_weights (nightly retrain path).
    RETRAIN_GUARD_ENABLED: bool = True
    RETRAIN_GUARD_VALIDATION_STEPS: int = 500   # env steps for deterministic rollout
    RETRAIN_GUARD_MIN_DROP: float = 0.002       # need >0.002 abs mean-reward/step drop
    RETRAIN_GUARD_REL_DROP: float = 0.20        # AND >20% relative drop
    # ==================== A4: Pre-Market PPO Micro-Retrain ====================
    # A lighter retrain that fires at 08:30 ET (1 hr before market open). Adapts
    # the model to overnight news/futures moves using just the most recent bars.
    # Strictly gated by RETRAIN-GUARD with tighter thresholds than the 18:00 run.
    # Much smaller than the nightly 100K retrain — 5K timesteps, LR=1e-5, recent-bars only.
    PPO_MICRO_RETRAIN_ENABLED: bool = True
    PPO_MICRO_RETRAIN_HOUR: int = 8             # ET hour for pre-market fire
    PPO_MICRO_RETRAIN_MINUTE: int = 30
    PPO_MICRO_RETRAIN_TIMESTEPS: int = 5000     # 20× smaller than nightly
    PPO_MICRO_RETRAIN_LR: float = 1e-5           # 5× smaller than nightly online LR
    PPO_MICRO_RETRAIN_BARS: int = 500            # last-N bars only (recency-focused)
    PPO_MICRO_RETRAIN_VALIDATION_STEPS: int = 300  # fewer than nightly — faster
    PPO_MICRO_RETRAIN_MIN_DROP: float = 0.0015  # stricter than nightly
    PPO_MICRO_RETRAIN_REL_DROP: float = 0.15    # stricter 15% relative
    # ==================== TIME-STOP: Dead Trade Liquidation ====================
    # Close positions that have been held past THRESHOLD_BARS without meaningful
    # excursion in EITHER direction (both MFE and MAE stayed small). Such trades
    # are "dead" — thesis not playing out, capital tied up for nothing.
    # Frees slot for a better opportunity.
    TIME_STOP_ENABLED: bool = True
    TIME_STOP_THRESHOLD_BARS: int = 96           # held ≥96 bars (≈24 RTH hours)
    TIME_STOP_MFE_CEILING: float = 0.005         # never went +0.5% up
    TIME_STOP_MAE_FLOOR: float = -0.005          # never went -0.5% down
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

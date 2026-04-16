# gemini_tuner.py
# EXPANDED + UNIVERSE MANAGEMENT VERSION (Feb 26 2026)
# • Suggests full "proposed_universe" of exactly MAX_UNIVERSE_SIZE
# • Safe clamping preserved
# • Persistence via dynamic_config.json
# ISSUE #2 PATCH: Removed duplicate TUNABLE_PARAMS — now reads defaults & bounds from CONFIG (Pydantic-validated)
# ISSUE #5 PATCH: Added intra-week rotation trigger if Gemini proposes new universe AND performance is poor
import os
import json
import logging
import glob
from datetime import datetime, timezone
from dotenv import load_dotenv
from google.genai import Client
from tensorboard.backend.event_processing import event_accumulator
from config import CONFIG
from typing import Any, Dict, List
import tempfile   # ← Priority 1: for atomic writes
import shutil     # ← Priority 1: for atomic rename

load_dotenv()

# ────────────────────────────────────────────────────────────────────────────────
# MOVED: Gemini API key is now checked only when actually needed (Issue #3 fix)
# No longer crashes entire bot on import if key is missing
# ────────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Still load it globally, but don't raise here

logger = logging.getLogger(__name__)

# FIX #24: Use absolute paths based on project root to avoid CWD-relative fragility
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TB_LOG_DIR = os.path.join(_PROJECT_ROOT, "ppo_tensorboard", "portfolio")
DYNAMIC_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "dynamic_config.json")
TUNING_HISTORY_PATH = os.path.join(_PROJECT_ROOT, "tuning_history.json")  # stores last few tuning changes for temporal context

def load_dynamic_config():
    if os.path.exists(DYNAMIC_CONFIG_PATH):
        try:
            with open(DYNAMIC_CONFIG_PATH, 'r') as f:
                dynamic = json.load(f)
            # FIX #33: Validate BEFORE applying to CONFIG. On validation failure,
            # revert to pre-update state so invalid values don't persist.
            from config import TradingBotConfig
            pre_update_snapshot = dict(CONFIG)
            CONFIG.update(dynamic)
            if 'SYMBOLS' in dynamic:
                CONFIG['SYMBOLS'] = dynamic['SYMBOLS']
                logger.info(f"[GEMINI TUNER] Restored persisted universe: {CONFIG['SYMBOLS']}")
            try:
                validated = TradingBotConfig.model_validate(CONFIG)
                CONFIG.update(validated.model_dump())
                logger.debug("[GEMINI TUNER] Dynamic config passed Pydantic validation")
            except Exception as val_err:
                # M10 FIX: Atomic revert — update keys individually instead of clear()+update()
                # which created a brief window where CONFIG was empty.
                logger.warning(f"[GEMINI TUNER] Dynamic config failed Pydantic validation: {val_err} — reverting to previous config")
                CONFIG.update(pre_update_snapshot)
                return
            logger.info(f"[GEMINI TUNER] Loaded {len(dynamic)} persisted settings")
        except Exception as e:
            logger.warning(f"Failed to load dynamic config: {e}")

def save_dynamic_config(changes: dict):
    """Save dynamic config and tuning history atomically (Priority 1 fix)"""
    try:
        current = {}
        if os.path.exists(DYNAMIC_CONFIG_PATH):
            with open(DYNAMIC_CONFIG_PATH, 'r') as f:
                current = json.load(f)
        current.update(changes)

        # Atomic write for dynamic_config.json
        path_config = DYNAMIC_CONFIG_PATH
        dir_name = os.path.dirname(path_config) or '.'
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=dir_name) as tmp:
            json.dump(current, tmp, indent=2, default=str)
            tmp.flush()
            os.fsync(tmp.fileno())
        shutil.move(tmp.name, path_config)
        logger.info(f"[ATOMIC SAVE] Saved {len(changes)} changes to dynamic_config.json")

        # Atomic write for tuning_history.json (append + keep last 5)
        history_entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "changes": changes,
            "pnl_summary": "N/A"  # can be populated later if needed
        }
        history = []
        if os.path.exists(TUNING_HISTORY_PATH):
            with open(TUNING_HISTORY_PATH, 'r') as f:
                history = json.load(f)
        history.append(history_entry)
        # Keep only last 5 entries to avoid bloat
        history = history[-5:]

        path_history = TUNING_HISTORY_PATH
        dir_name = os.path.dirname(path_history) or '.'
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=dir_name) as tmp:
            json.dump(history, tmp, indent=2, default=str)
            tmp.flush()
            os.fsync(tmp.fileno())
        shutil.move(tmp.name, path_history)
        logger.debug(f"[ATOMIC TUNING HISTORY] Appended change — history now has {len(history)} entries")

    except Exception as e:
        logger.error(f"Failed to save dynamic config or tuning history: {e}")

def get_recent_ppo_scalars(log_dir: str = TB_LOG_DIR, max_points: int = 50) -> dict:
    if not os.path.exists(log_dir):
        return {}
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        return {}
    event_files.sort(key=os.path.getmtime, reverse=True)
    latest_file = event_files[0]
    try:
        # H4 FIX: EventAccumulator needs the directory containing the event file, not the file path.
        # Passing a file path may silently return empty tags in some tensorboard versions.
        ea = event_accumulator.EventAccumulator(os.path.dirname(latest_file))
        ea.Reload()
    except Exception:
        return {}
    tags = ['rollout/ep_rew_mean', 'rollout/ep_len_mean', 'train/entropy_loss']
    scalars = {}
    for tag in tags:
        if tag in ea.Tags().get('scalars', []) and ea.Scalars(tag):
            values = ea.Scalars(tag)[-max_points:]
            if values:
                cleaned = tag.replace('train/', '').replace('rollout/', '').replace('ep_rew_mean', 'episode_reward_mean')
                scalars[cleaned] = round(values[-1].value, 6)
    return scalars

# ────────────────────────────────────────────────────────────────────────────────
# Structured logging function — duplicate removed (second copy near bottom was identical)
# ────────────────────────────────────────────────────────────────────────────────
def log_structured_gemini_change(param: str, old_value: Any, new_value: Any, pnl_context: dict):
    """
    Structured JSON logging for each Gemini-applied change.
    Makes future analysis easy (grep, dashboards, correlation with PnL).
    """
    entry = {
        "event": "gemini_change",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "param": param,
        "old": old_value,
        "new": new_value,
        "pnl_context": pnl_context
    }
    logger.info(json.dumps(entry, default=str))  # default=str handles datetime/numpy/non-serializable

def query_gemini_for_tuning(context_data: dict, current_config: dict, symbol_performance: dict = None) -> dict:
    logger.info("[GEMINI TUNER] Querying Gemini 2.5 Flash...")
    # ────────────────────────────────────────────────────────────────────────────────
    # ISSUE #3 FIX: Check API key only here (when we actually need Gemini)
    # If missing → graceful fallback instead of crashing entire bot on import
    # ────────────────────────────────────────────────────────────────────────────────
    global GEMINI_API_KEY
    if not GEMINI_API_KEY:
        logger.error("[GEMINI TUNER] GEMINI_API_KEY missing or empty in .env — skipping Gemini query")
        return {}
    ppo_scalars = get_recent_ppo_scalars()
    context_data['ppo_scalars'] = ppo_scalars or {"status": "no_data"}
    performance_str = json.dumps(symbol_performance or {}, indent=2) if symbol_performance else "No performance data yet"
    # NEW: Load previous 1-2 tuning changes for temporal context
    previous_changes = []
    if os.path.exists(TUNING_HISTORY_PATH):
        try:
            with open(TUNING_HISTORY_PATH, 'r') as f:
                history = json.load(f)
            previous_changes = history[-2:]  # last 2 entries
            logger.debug(f"[TEMPORAL CONTEXT] Loaded {len(previous_changes)} previous tuning changes")
        except Exception as e:
            logger.warning(f"Failed to load tuning history: {e}")
    # NEW: Entropy-gated tuning focus
    entropy = ppo_scalars.get('entropy_loss', -1.0)
    if entropy > -0.05:  # entropy_loss near 0 = LOW entropy = policy is peaked/over-exploiting
        tuning_focus = "EXPLOITATION: Focus on Risk, Sizing, ATR bounds, ratcheting thresholds, min-hold bars to capitalize on the learned policy."
    else:  # entropy_loss very negative = HIGH entropy = policy is flat/exploring
        tuning_focus = "EXPLORATION: Focus on PPO_ENTROPY_COEFF, PPO_LEARNING_RATE, PPO_GAE_LAMBDA, PPO_CLIP_RANGE to stabilize learning."
    context_data['tuning_focus'] = tuning_focus
    client = Client(api_key=GEMINI_API_KEY)
    # ISSUE #2 PATCH: No more TUNABLE_PARAMS — use CONFIG values directly
    prompt = f"""You are a quantitative parameter optimization agent embedded in a live algorithmic trading system. You receive full system telemetry and output precise parameter adjustments. Your responses directly modify a production trading bot — accuracy and restraint are critical.

OBJECTIVE: Maximize risk-adjusted returns. Primary target: Sharpe ratio >2.0. Hard constraint: max drawdown <12%. Secondary: profit factor >1.5, win rate >55%. When all targets are met, make minimal or no changes.

=== SYSTEM TELEMETRY ===

Portfolio State:
  Equity: {context_data.get('equity', 'N/A')} | Starting: {context_data.get('starting_equity', 'N/A')} | Return: {context_data.get('return_pct', 0):.3f}%
  Buying Power: {context_data.get('buying_power', 'N/A')} | Open Positions: {context_data.get('open_positions', 0)}/{current_config.get('MAX_POSITIONS', 6)}
  Tracked Order Groups: {context_data.get('tracked_orders', 0)} | Slippage Offset: {context_data.get('current_slippage_offset', 0):.5f}

Risk Metrics:
  Sharpe Ratio: {context_data.get('sharpe_ratio', 0)} | Sortino Ratio: {context_data.get('sortino_ratio', 0)}
  Max Drawdown: {context_data.get('max_drawdown_pct', 0)}% | Profit Factor: {context_data.get('profit_factor', 0)}

Trade Statistics:
  Total Trades: {context_data.get('total_trades', 0)} | Wins: {context_data.get('wins', 0)} | Losses: {context_data.get('losses', 0)}
  Win Rate: {context_data.get('win_rate', 'N/A')}% | Avg Win: {context_data.get('avg_win', 0):.5f} | Avg Loss: {context_data.get('avg_loss', 0):.5f}
  Avg Return: {context_data.get('avg_return', 0):.6f} | Avg Hold Time: {context_data.get('avg_hold_bars', 0)} bars (15min)

Regime State:
  Dominant Regime: {context_data.get('regime', 'mixed')} | Avg Persistence: {context_data.get('avg_persistence', 0.5)}
  Regime Distribution: {json.dumps(context_data.get('regime_counts', {}))}
  Universe Size: {context_data.get('symbols_in_universe', 0)} symbols

PPO Training State:
  {json.dumps(context_data.get('ppo_scalars', {}))}
  Tuning Focus (entropy-gated): {context_data.get('tuning_focus', 'balanced')}

Per-Symbol Breakdown (regime, persistence, sharpe, win_rate, pnl_dollars, recent_10_win_rate, current_price):
{performance_str}

=== PREVIOUS TUNING ACTIONS (anti-oscillation context) ===
{json.dumps(previous_changes, indent=2) if previous_changes else "First tuning cycle — no history."}

=== DIAGNOSTIC PROTOCOL ===

Execute these steps IN ORDER. Do not skip steps.

Step 1 — TRIAGE: Classify system state into exactly one category:
  CRITICAL:  max_drawdown >12% OR profit_factor <0.8 OR Sharpe <0 → aggressive risk reduction
  DEGRADED:  win_rate <50% OR Sharpe <1.0 OR avg_loss > 2*avg_win → targeted fix for weakest metric
  SUBOPTIMAL: Sharpe 1.0-2.0 OR profit_factor 1.0-1.5 OR win_rate 50-58% → incremental optimization
  HEALTHY:   Sharpe >2.0, DD <8%, win_rate >58%, profit_factor >1.5 → HOLD. Return mostly nulls.

Step 2 — ROOT CAUSE: Given the triage category, identify the single most impactful bottleneck:
  a) Drawdown dominance → stops too wide, risk-per-trade too high, or leverage excessive
  b) Low win rate → signal quality (dead zone, confidence thresholds), or premature exits (hold bars)
  c) Low Sharpe despite positive P&L → return variance too high, tighten trailing stops and TP
  d) Low trade count (<2/day) → filters too aggressive (MIN_CONFIDENCE, CONVICTION_THRESHOLD, dead zone)
  e) Win/loss asymmetry (avg_loss >> avg_win) → trailing stop too loose, TP too tight
  f) Regime mismatch → regime detection params mistuned for current market conditions
  g) PPO collapse (entropy → 0 or → -∞) → RL hyperparameters destabilizing policy
  h) Per-symbol drag → specific symbols consistently negative → universe rotation needed

Step 3 — PRESCRIBE: Propose parameter changes that directly address the root cause. Every change must have a causal rationale. Do NOT scatter-shot unrelated params.

Step 4 — UNIVERSE: Select exactly {current_config.get('MAX_UNIVERSE_SIZE', 8)} symbols from UNIVERSE_CANDIDATES.
  Selection criteria (weighted): liquidity (30%), regime diversity (25%), recent momentum (25%), low inter-correlation (20%).
  Flag any symbol with sharpe <-0.5 or >20 consecutive losing trades for replacement.

=== PARAMETER GROUPS & HARD BOUNDS ===

GROUP 1: Risk & Sizing (max ±20% change per cycle)
  RISK_PER_TRADE_TRENDING      [{0.010:.4f} – {0.040:.4f}]  current: {current_config.get('RISK_PER_TRADE_TRENDING')}
  RISK_PER_TRADE_MEAN_REVERTING [{0.002:.4f} – {0.015:.4f}]  current: {current_config.get('RISK_PER_TRADE_MEAN_REVERTING')}
  KELLY_FRACTION                [0.25 – 0.65]                current: {current_config.get('KELLY_FRACTION')}
  RISK_BUDGET_MULTIPLIER        [1.2 – 2.5]                  current: {current_config.get('RISK_BUDGET_MULTIPLIER')}
  MAX_LEVERAGE                  [1.5 – 2.5]                  current: {current_config.get('MAX_LEVERAGE')}
  MAX_POSITIONS                 [4 – 8]                       current: {current_config.get('MAX_POSITIONS')}
  MAX_POSITION_VALUE_FRACTION   [0.15 – 0.30]                current: {current_config.get('MAX_POSITION_VALUE_FRACTION')}
  CONVICTION_THRESHOLD          [0.20 – 0.40]                current: {current_config.get('CONVICTION_THRESHOLD')}

GROUP 2: Trailing Stops & Take-Profit (max ±25%)
  TRAILING_STOP_ATR_TRENDING    [1.5 – 3.5]   current: {current_config.get('TRAILING_STOP_ATR_TRENDING')}
  TRAILING_STOP_ATR_MEAN_REVERTING [2.0 – 5.0] current: {current_config.get('TRAILING_STOP_ATR_MEAN_REVERTING')}
  TAKE_PROFIT_ATR_TRENDING      [15.0 – 40.0]  current: {current_config.get('TAKE_PROFIT_ATR_TRENDING')}
  TAKE_PROFIT_ATR_MEAN_REVERTING [5.0 – 15.0]  current: {current_config.get('TAKE_PROFIT_ATR_MEAN_REVERTING')}

GROUP 3: Ratcheting / Trailing Stop Tightening (max ±30%)
  RATCHET_TRENDING_INTERVAL_SEC       [60 – 360]    current: {current_config.get('RATCHET_TRENDING_INTERVAL_SEC')}
  RATCHET_MEAN_REVERTING_INTERVAL_SEC [180 – 900]   current: {current_config.get('RATCHET_MEAN_REVERTING_INTERVAL_SEC')}
  RATCHET_REGIME_FACTOR_TRENDING      [0.3 – 0.8]   current: {current_config.get('RATCHET_REGIME_FACTOR_TRENDING')}
  RATCHET_REGIME_FACTOR_MEAN_REVERTING [0.8 – 1.8]  current: {current_config.get('RATCHET_REGIME_FACTOR_MEAN_REVERTING')}
  RATCHET_PROFIT_PROTECTION_SLOPE     [0.8 – 2.0]   current: {current_config.get('RATCHET_PROFIT_PROTECTION_SLOPE')}
  RATCHET_PROFIT_PROTECTION_MIN       [0.20 – 0.50] current: {current_config.get('RATCHET_PROFIT_PROTECTION_MIN')}

GROUP 4: Signal Generation & Confidence
  MIN_CONFIDENCE     [0.75 – 0.92]  current: {current_config.get('MIN_CONFIDENCE')}
  DEAD_ZONE_LOW      [0.42 – 0.52]  current: {current_config.get('DEAD_ZONE_LOW')}
  DEAD_ZONE_HIGH     [0.58 – 0.70]  current: {current_config.get('DEAD_ZONE_HIGH')}
  SENTIMENT_WEIGHT   [0.10 – 0.40]  current: {current_config.get('SENTIMENT_WEIGHT')}
  PORTFOLIO_SENTIMENT_WEIGHT [0.15 – 0.50] current: {current_config.get('PORTFOLIO_SENTIMENT_WEIGHT')}
  EMA_ALPHA          [0.003 – 0.015] current: {current_config.get('EMA_ALPHA')}
  MIN_HOLD_BARS_TRENDING      [3 – 12]  current: {current_config.get('MIN_HOLD_BARS_TRENDING')}
  MIN_HOLD_BARS_MEAN_REVERTING [2 – 8]  current: {current_config.get('MIN_HOLD_BARS_MEAN_REVERTING')}
  REGIME_OVERRIDE_PERSISTENCE  [0.65 – 0.95] current: {current_config.get('REGIME_OVERRIDE_PERSISTENCE')}

GROUP 5: PPO / RL Hyperparameters (max ±15% — RL is sensitive)
  PPO_LEARNING_RATE    [1e-4 – 5e-4]  current: {current_config.get('PPO_LEARNING_RATE')}
  PPO_ENTROPY_COEFF    [0.01 – 0.08]  current: {current_config.get('PPO_ENTROPY_COEFF')}
  PPO_GAMMA            [0.93 – 0.98]  current: {current_config.get('PPO_GAMMA')}
  PPO_GAE_LAMBDA       [0.90 – 0.97]  current: {current_config.get('PPO_GAE_LAMBDA')}
  PPO_CLIP_RANGE       [0.10 – 0.25]  current: {current_config.get('PPO_CLIP_RANGE')}
  vf_coef              [0.3 – 1.5]    current: {current_config.get('VF_COEF')}
  PPO_AUX_LOSS_WEIGHT  [0.10 – 0.40]  current: {current_config.get('PPO_AUX_LOSS_WEIGHT')}

GROUP 6: Reward Shaping (max ±25%)
  DD_PENALTY_COEF          [0.5 – 3.0]   current: {current_config.get('DD_PENALTY_COEF')}
  VOL_PENALTY_COEF         [0.005 – 0.03] current: {current_config.get('VOL_PENALTY_COEF')}
  TURNOVER_COST_MULT       [0.1 – 0.5]   current: {current_config.get('TURNOVER_COST_MULT')}
  SORTINO_WEIGHT           [0.10 – 0.35]  current: {current_config.get('SORTINO_WEIGHT')}
  PERSISTENCE_BONUS_SCALE  [0.1 – 0.8]   current: {current_config.get('PERSISTENCE_BONUS_SCALE')}
  CAUSAL_PENALTY_WEIGHT    [0.20 – 0.50]  current: {current_config.get('CAUSAL_PENALTY_WEIGHT')}
  CAUSAL_REWARD_FACTOR     [0.3 – 1.0]   current: {current_config.get('CAUSAL_REWARD_FACTOR')}

GROUP 7: Regime Detection
  REGIME_SHORT_LOOKBACK        [48 – 192]   current: {current_config.get('REGIME_SHORT_LOOKBACK')}
  REGIME_SHORT_WEIGHT          [0.3 – 0.8]  current: {current_config.get('REGIME_SHORT_WEIGHT')}
  REGIME_CONFIDENCE_MIN_SIZE_PCT [0.25 – 0.70] current: {current_config.get('REGIME_CONFIDENCE_MIN_SIZE_PCT')}
  HURST_TREND_THRESHOLD        [0.35 – 0.55] current: {current_config.get('HURST_TREND_THRESHOLD')}
  VIX_THRESHOLD                [22 – 35]     current: {current_config.get('VIX_THRESHOLD')}
  BREAKOUT_BOOST_FACTOR        [1.0 – 1.5]   current: {current_config.get('BREAKOUT_BOOST_FACTOR')}

GROUP 8: Intraday Risk Pacing (S4) — throttle as today's losses accumulate
  RISK_PACING_TIER1_LOSS       [-0.010 – -0.002]  current: {current_config.get('RISK_PACING_TIER1_LOSS')}
  RISK_PACING_TIER1_SCALE      [0.3 – 0.8]        current: {current_config.get('RISK_PACING_TIER1_SCALE')}
  RISK_PACING_TIER2_LOSS       [-0.030 – -0.010]  current: {current_config.get('RISK_PACING_TIER2_LOSS')}
  RISK_PACING_TIER2_SCALE      [0.1 – 0.4]        current: {current_config.get('RISK_PACING_TIER2_SCALE')}
  RISK_PACING_CONSECUTIVE_LOSSES       [2 – 5]    current: {current_config.get('RISK_PACING_CONSECUTIVE_LOSSES')}
  RISK_PACING_CONSECUTIVE_LOSS_SCALE   [0.15 – 0.6] current: {current_config.get('RISK_PACING_CONSECUTIVE_LOSS_SCALE')}

GROUP 9: Asymmetric Trailing & Loss Tightening (S3) — cut losers, let winners breathe
  RATCHET_LOSS_TIGHTEN_THRESHOLD   [-0.015 – -0.003]  current: {current_config.get('RATCHET_LOSS_TIGHTEN_THRESHOLD')}
  RATCHET_LOSS_TIGHTEN_FACTOR      [0.35 – 0.85]      current: {current_config.get('RATCHET_LOSS_TIGHTEN_FACTOR')}
  RATCHET_LOSS_TIGHTEN_MFE_MAX     [0.001 – 0.010]    current: {current_config.get('RATCHET_LOSS_TIGHTEN_MFE_MAX')}

GROUP 10: Meta-Label Filter (S1) — reject low-probability trades before entry
  META_FILTER_MIN_PROB   [0.30 – 0.55]  current: {current_config.get('META_FILTER_MIN_PROB')}

GROUP 11: Cross-Sectional Momentum Gate (A1) — tilt toward today's leaders
  CROSS_SECTIONAL_WEIGHT     [0.5 – 1.2]  current: {current_config.get('CROSS_SECTIONAL_WEIGHT')}
  CROSS_SECTIONAL_MAX_MULT   [1.10 – 1.50]  current: {current_config.get('CROSS_SECTIONAL_MAX_MULT')}
  CROSS_SECTIONAL_MIN_MULT   [0.30 – 0.70]  current: {current_config.get('CROSS_SECTIONAL_MIN_MULT')}
  CROSS_SECTIONAL_NEUTRAL_BAND [0.10 – 0.50]  current: {current_config.get('CROSS_SECTIONAL_NEUTRAL_BAND')}

GROUP 12: Anti-Earnings Filter (B5) — blackout around earnings events
  EARNINGS_BLACKOUT_PRE_DAYS   [1 – 4]  current: {current_config.get('EARNINGS_BLACKOUT_PRE_DAYS')}
  EARNINGS_BLACKOUT_POST_DAYS  [0 – 3]  current: {current_config.get('EARNINGS_BLACKOUT_POST_DAYS')}
  EARNINGS_CLOSE_PRE_DAYS      [0 – 3]  current: {current_config.get('EARNINGS_CLOSE_PRE_DAYS')}

GROUP 13: Sentiment Velocity (B4) — Δsentiment as additional multiplier
  SENTIMENT_VELOCITY_WEIGHT          [0.0 – 0.35]  current: {current_config.get('SENTIMENT_VELOCITY_WEIGHT')}
  SENTIMENT_VELOCITY_LOOKBACK_HOURS  [2 – 8]       current: {current_config.get('SENTIMENT_VELOCITY_LOOKBACK_HOURS')}

GROUP 14: Correlation-Aware Sizing (AC) — discount crowded directional bets
  CROWDING_DISCOUNT_THRESHOLD  [0.35 – 0.75]  current: {current_config.get('CROWDING_DISCOUNT_THRESHOLD')}
  CROWDING_DISCOUNT_STRENGTH   [0.2 – 1.0]    current: {current_config.get('CROWDING_DISCOUNT_STRENGTH')}
  CROWDING_DISCOUNT_MIN_FACTOR [0.25 – 0.70]  current: {current_config.get('CROWDING_DISCOUNT_MIN_FACTOR')}

GROUP 15: Portfolio Meta-Blend (stacking ensemble overlay)
  PORTFOLIO_META_WEIGHT  [0.0 – 0.40]  current: {current_config.get('PORTFOLIO_META_WEIGHT')}

GROUP 16: Adverse-Selection Detector (B2) — penalize symbols with toxic fills
  ADVERSE_SELECTION_THRESHOLD   [-0.005 – -0.0005]  current: {current_config.get('ADVERSE_SELECTION_THRESHOLD')}
  ADVERSE_SELECTION_MAX_PENALTY [0.2 – 0.7]          current: {current_config.get('ADVERSE_SELECTION_MAX_PENALTY')}

GROUP 17: Bayesian Per-Symbol Sizing (BPS) — size by posterior P(win)
  BAYESIAN_SIZING_MIN_MULT       [0.2 – 0.7]    current: {current_config.get('BAYESIAN_SIZING_MIN_MULT')}
  BAYESIAN_SIZING_MAX_MULT       [1.2 – 2.0]    current: {current_config.get('BAYESIAN_SIZING_MAX_MULT')}
  BAYESIAN_SIZING_REFERENCE_EV   [0.001 – 0.006] current: {current_config.get('BAYESIAN_SIZING_REFERENCE_EV')}
  BAYESIAN_SIZING_SHRINKAGE_N    [4 – 20]       current: {current_config.get('BAYESIAN_SIZING_SHRINKAGE_N')}

GROUP 18: Slippage Veto (ESP) — skip entries where predicted slip > alpha
  SLIPPAGE_VETO_MULTIPLE  [0.8 – 2.0]   current: {current_config.get('SLIPPAGE_VETO_MULTIPLE')}
  SLIPPAGE_VETO_SCALE     [0.1 – 0.6]   current: {current_config.get('SLIPPAGE_VETO_SCALE')}

GROUP 19: PPO-Stacking Divergence (PSD) — dampen when models strongly disagree
  DIVERGENCE_GATE_SCALE   [0.2 – 0.8]   current: {current_config.get('DIVERGENCE_GATE_SCALE')}
  DIVERGENCE_MIN_WEIGHT   [0.01 – 0.08] current: {current_config.get('DIVERGENCE_MIN_WEIGHT')}
  DIVERGENCE_MIN_META     [0.10 – 0.40] current: {current_config.get('DIVERGENCE_MIN_META')}

GROUP 20: Kelly Sizing (KELLY) — mathematically-optimal per-symbol sizing
  BAYESIAN_SIZING_KELLY_FRACTION   [0.10 – 0.50]  current: {current_config.get('BAYESIAN_SIZING_KELLY_FRACTION')}
  BAYESIAN_SIZING_REFERENCE_KELLY  [0.04 – 0.15]  current: {current_config.get('BAYESIAN_SIZING_REFERENCE_KELLY')}

GROUP 21: Regime-Conditional Exits (REX) — differentiated TP/trail by alignment
  REX_ALIGN_TP_MULT    [1.15 – 1.80]  current: {current_config.get('REX_ALIGN_TP_MULT')}
  REX_ALIGN_TRAIL_MULT [1.05 – 1.50]  current: {current_config.get('REX_ALIGN_TRAIL_MULT')}
  REX_OPPOSE_TP_MULT   [0.50 – 0.90]  current: {current_config.get('REX_OPPOSE_TP_MULT')}
  REX_OPPOSE_TRAIL_MULT[0.55 – 0.95]  current: {current_config.get('REX_OPPOSE_TRAIL_MULT')}
  REX_MR_TP_MULT       [0.65 – 1.00]  current: {current_config.get('REX_MR_TP_MULT')}
  REX_MR_TRAIL_MULT    [0.75 – 1.05]  current: {current_config.get('REX_MR_TRAIL_MULT')}

=== CONSTRAINTS ===
1. HEALTHY systems get NO changes or at most 2 minor tweaks. Do NOT fix what is working.
2. Maximum 6 parameter changes per cycle. Focused, high-conviction changes only.
3. Anti-oscillation: if PREVIOUS TUNING ACTIONS show param X moved in one direction, do NOT reverse it unless telemetry CLEARLY shows it degraded performance (compare before/after metrics).
4. Every proposed value must fall within the HARD BOUNDS listed above. Out-of-bounds values will be clamped.
5. Change magnitude limits per group are enforced server-side. Propose your ideal value; the system will clamp to max allowed step size.
6. Null = no change. Only include params you are actively changing.
7. For PPO params: prefer small moves (5-10%). RL is sensitive to large jumps.
8. Cross-validate: if you tighten stops, consider whether risk-per-trade should compensate. If you widen dead zone, check that MIN_CONFIDENCE still allows sufficient trade flow.

=== OUTPUT FORMAT (strict JSON, no markdown fences, no commentary outside JSON) ===
{{
  "triage": "CRITICAL | DEGRADED | SUBOPTIMAL | HEALTHY",
  "diagnosis": "One sentence identifying the #1 bottleneck from the diagnostic protocol.",
  "root_cause": "One sentence explaining what market/system condition is causing the bottleneck.",
  "prescription": "2-3 sentences explaining the causal chain from diagnosis → parameter changes. Reference specific telemetry values.",
  "proposed_universe": ["SYM1", "SYM2", ...],
  "parameters": {{
    "PARAM_NAME": new_numeric_value_or_null
  }}
}}"""
    try:
        # M11 FIX: Add timeout to prevent indefinite blocking if Gemini API hangs
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt,
            config={"httpOptions": {"timeout": 120_000}}  # timeout in milliseconds
        )
        content = response.text.strip()
        # FIX: Use regex for robust fence stripping (was fragile with trailing whitespace)
        import re as _re
        fence_match = _re.search(r'```(?:json)?\s*\n?(.*?)```', content, _re.DOTALL)
        if fence_match:
            content = fence_match.group(1).strip()
        tweaks = json.loads(content)
        applied = {}
        # Prepare pnl_context for structured logging (same for all changes)
        pnl_context = {
            "equity": context_data.get("equity", "N/A"),
            "return_pct": context_data.get("return_pct", "N/A"),
            "win_rate": context_data.get("win_rate", "N/A"),
            "sharpe": context_data.get("sharpe_ratio", 0),
            "sortino": context_data.get("sortino_ratio", 0),
            "max_drawdown_pct": context_data.get("max_drawdown_pct", 0),
            "profit_factor": context_data.get("profit_factor", 0),
            "regime": context_data.get("regime", "mixed"),
            "total_trades": context_data.get("total_trades", 0),
            "ppo_scalars": context_data.get("ppo_scalars", {})
        }
        # ISSUE #5 PATCH: Check for intra-week rotation
        current_symbols = set(CONFIG.get('SYMBOLS', []))
        proposed_universe = tweaks.get("proposed_universe", [])
        if isinstance(proposed_universe, list):
            candidates = set(CONFIG.get('UNIVERSE_CANDIDATES', []))
            proposed_set = set([s.upper() for s in proposed_universe if s.upper() in candidates])
            max_size = current_config.get('MAX_UNIVERSE_SIZE', 8)
            if len(proposed_set) >= max(max_size - 1, 4):
                # Performance check: rotate only if poor (win rate < 55% or drawdown > 5%)
                try:
                    # FIX #41: Normalize win_rate — handle both decimal (0.55) and percentage (55) inputs
                    raw_wr = float(context_data.get('win_rate', '50'))
                    # M9 FIX: Explicit percentage detection — values > 1.0 are always percentages.
                    # A win rate ratio above 1.0 is impossible (max=1.0=100%), so any value > 1.0
                    # must be a percentage that needs /100 conversion.
                    recent_win_rate = raw_wr if raw_wr <= 1.0 else raw_wr / 100
                except (ValueError, TypeError):
                    recent_win_rate = 0.5
                # Use max_drawdown_pct directly (the pnl_summary parsing was broken)
                try:
                    drawdown = abs(float(context_data.get('max_drawdown_pct', 0))) / 100
                except (ValueError, TypeError):
                    drawdown = 0.0
                should_rotate = (recent_win_rate < 0.55) or (drawdown > 0.05)
                if proposed_set != current_symbols and should_rotate:
                    applied["rotate_now"] = True
                    applied["proposed_universe"] = {"old": list(current_symbols), "new": list(proposed_set)}
                    applied["SYMBOLS"] = list(proposed_set)
                    # Structured log for universe rotation
                    log_structured_gemini_change(
                        param="SYMBOLS",
                        old_value=list(current_symbols),
                        new_value=list(proposed_set),
                        pnl_context=pnl_context
                    )
                    logger.info(f"[GEMINI TUNER] Intra-week rotation triggered — win rate {recent_win_rate:.1%}, drawdown {drawdown:.1%}")
                elif proposed_set != current_symbols:
                    logger.debug(f"[GEMINI TUNER] Proposed universe different but performance ok — no rotation")
                else:
                    logger.debug(f"[GEMINI TUNER] Proposed universe matches current — no rotation needed")
        # FIX #29: Collect all changes into staging dict first, apply atomically at end.
        # This prevents CONFIG from being half-modified if parsing fails midway.
        staged_changes = {}  # key -> clamped_value
        for key, value in tweaks.get("parameters", {}).items():
            if value is None or key not in current_config:
                continue
            current_val = current_config.get(key)
            if isinstance(current_val, (int, float)):
                # FIX: Category-aware clamp bounds (was uniform 15% for all params)
                risk_params = {'RISK_PER_TRADE', 'RISK_PER_TRADE_TRENDING', 'RISK_PER_TRADE_MEAN_REVERTING',
                               'RISK_BUDGET_MULTIPLIER', 'MAX_TOTAL_RISK_PCT', 'MAX_POSITION_VALUE_FRACTION',
                               'KELLY_FRACTION', 'MAX_LEVERAGE', 'CONVICTION_THRESHOLD'}
                hold_params = {'MIN_HOLD_BARS_TRENDING', 'MIN_HOLD_BARS_MEAN_REVERTING'}
                atr_params = {'TRAILING_STOP_ATR_TRENDING', 'TRAILING_STOP_ATR_MEAN_REVERTING',
                              'TAKE_PROFIT_ATR_TRENDING', 'TAKE_PROFIT_ATR_MEAN_REVERTING'}
                ratchet_params = {'RATCHET_TRENDING_INTERVAL_SEC', 'RATCHET_MEAN_REVERTING_INTERVAL_SEC',
                                  'RATCHET_REGIME_FACTOR_TRENDING', 'RATCHET_REGIME_FACTOR_MEAN_REVERTING',
                                  'RATCHET_PROFIT_PROTECTION_SLOPE', 'RATCHET_PROFIT_PROTECTION_MIN'}
                reward_params = {'DD_PENALTY_COEF', 'VOL_PENALTY_COEF', 'TURNOVER_COST_MULT',
                                 'SORTINO_WEIGHT', 'PERSISTENCE_BONUS_SCALE', 'CAUSAL_PENALTY_WEIGHT',
                                 'CAUSAL_REWARD_FACTOR'}
                ppo_params = {'PPO_LEARNING_RATE', 'PPO_ENTROPY_COEFF', 'PPO_GAMMA', 'PPO_GAE_LAMBDA',
                              'PPO_CLIP_RANGE', 'VF_COEF', 'PPO_AUX_LOSS_WEIGHT'}
                # NEW — newer safeguard params; treat as mid-risk (25% step)
                new_safeguard_params = {
                    'RISK_PACING_TIER1_LOSS', 'RISK_PACING_TIER1_SCALE',
                    'RISK_PACING_TIER2_LOSS', 'RISK_PACING_TIER2_SCALE',
                    'RISK_PACING_CONSECUTIVE_LOSSES', 'RISK_PACING_CONSECUTIVE_LOSS_SCALE',
                    'RATCHET_LOSS_TIGHTEN_THRESHOLD', 'RATCHET_LOSS_TIGHTEN_FACTOR',
                    'RATCHET_LOSS_TIGHTEN_MFE_MAX',
                    'META_FILTER_MIN_PROB',
                    'CROSS_SECTIONAL_WEIGHT', 'CROSS_SECTIONAL_MAX_MULT',
                    'CROSS_SECTIONAL_MIN_MULT', 'CROSS_SECTIONAL_NEUTRAL_BAND',
                    'EARNINGS_BLACKOUT_PRE_DAYS', 'EARNINGS_BLACKOUT_POST_DAYS',
                    'EARNINGS_CLOSE_PRE_DAYS',
                    'SENTIMENT_VELOCITY_WEIGHT', 'SENTIMENT_VELOCITY_LOOKBACK_HOURS',
                    'CROWDING_DISCOUNT_THRESHOLD', 'CROWDING_DISCOUNT_STRENGTH',
                    'CROWDING_DISCOUNT_MIN_FACTOR',
                    'PORTFOLIO_META_WEIGHT',
                    'ADVERSE_SELECTION_THRESHOLD', 'ADVERSE_SELECTION_MAX_PENALTY',
                    'BAYESIAN_SIZING_MIN_MULT', 'BAYESIAN_SIZING_MAX_MULT',
                    'BAYESIAN_SIZING_REFERENCE_EV', 'BAYESIAN_SIZING_SHRINKAGE_N',
                    'SLIPPAGE_VETO_MULTIPLE', 'SLIPPAGE_VETO_SCALE',
                    'DIVERGENCE_GATE_SCALE', 'DIVERGENCE_MIN_WEIGHT', 'DIVERGENCE_MIN_META',
                    'BAYESIAN_SIZING_KELLY_FRACTION', 'BAYESIAN_SIZING_REFERENCE_KELLY',
                    'REX_ALIGN_TP_MULT', 'REX_ALIGN_TRAIL_MULT',
                    'REX_OPPOSE_TP_MULT', 'REX_OPPOSE_TRAIL_MULT',
                    'REX_MR_TP_MULT', 'REX_MR_TRAIL_MULT',
                }
                if key in risk_params:
                    pct_bound = 0.20
                elif key in hold_params:
                    pct_bound = 0.30
                elif key in atr_params:
                    pct_bound = 0.25
                elif key in ratchet_params or key in reward_params:
                    pct_bound = 0.25
                elif key in ppo_params:
                    pct_bound = 0.15  # RL params are sensitive
                elif key in new_safeguard_params:
                    pct_bound = 0.25  # graduated tuning on new safeguards
                else:
                    pct_bound = 0.15
                # FIX: Handle zero and negative values safely
                if current_val == 0:
                    # Allow small absolute change for zero-valued params
                    max_allowed = 0.1
                    min_allowed = -0.1
                elif current_val < 0:
                    max_allowed = current_val * (1 - pct_bound)  # less negative
                    min_allowed = current_val * (1 + pct_bound)  # more negative
                else:
                    max_allowed = current_val * (1 + pct_bound)
                    min_allowed = current_val * (1 - pct_bound)
                clamped = max(min_allowed, min(max_allowed, value))
                # FIX #34: Absolute parameter bounds to prevent multi-cycle drift.
                # These match the HARD BOUNDS in the Gemini prompt.
                ABSOLUTE_BOUNDS = {
                    'RISK_PER_TRADE_TRENDING': (0.010, 0.040),
                    'RISK_PER_TRADE_MEAN_REVERTING': (0.002, 0.015),
                    'KELLY_FRACTION': (0.25, 0.65),
                    'RISK_BUDGET_MULTIPLIER': (1.2, 2.5),
                    'MAX_LEVERAGE': (1.5, 2.5),
                    'MAX_POSITIONS': (4, 8),
                    'MAX_POSITION_VALUE_FRACTION': (0.15, 0.30),
                    'CONVICTION_THRESHOLD': (0.20, 0.40),
                    'TRAILING_STOP_ATR_TRENDING': (1.5, 3.5),
                    'TRAILING_STOP_ATR_MEAN_REVERTING': (2.0, 5.0),
                    'TAKE_PROFIT_ATR_TRENDING': (15.0, 40.0),
                    'TAKE_PROFIT_ATR_MEAN_REVERTING': (5.0, 15.0),
                    'RATCHET_TRENDING_INTERVAL_SEC': (60, 360),
                    'RATCHET_MEAN_REVERTING_INTERVAL_SEC': (180, 900),
                    'RATCHET_REGIME_FACTOR_TRENDING': (0.3, 0.8),
                    'RATCHET_REGIME_FACTOR_MEAN_REVERTING': (0.8, 1.8),
                    'RATCHET_PROFIT_PROTECTION_SLOPE': (0.8, 2.0),
                    'RATCHET_PROFIT_PROTECTION_MIN': (0.20, 0.50),
                    'MIN_CONFIDENCE': (0.75, 0.92),
                    'DEAD_ZONE_LOW': (0.42, 0.52),
                    'DEAD_ZONE_HIGH': (0.58, 0.70),
                    'SENTIMENT_WEIGHT': (0.10, 0.40),
                    'PORTFOLIO_SENTIMENT_WEIGHT': (0.15, 0.50),
                    'EMA_ALPHA': (0.003, 0.015),
                    'MIN_HOLD_BARS_TRENDING': (3, 12),
                    'MIN_HOLD_BARS_MEAN_REVERTING': (2, 8),
                    'REGIME_OVERRIDE_PERSISTENCE': (0.65, 0.95),
                    'PPO_LEARNING_RATE': (1e-4, 5e-4),
                    'PPO_ENTROPY_COEFF': (0.01, 0.08),
                    'PPO_GAMMA': (0.93, 0.98),
                    'PPO_GAE_LAMBDA': (0.90, 0.97),
                    'PPO_CLIP_RANGE': (0.10, 0.25),
                    'VF_COEF': (0.3, 1.5),
                    'PPO_AUX_LOSS_WEIGHT': (0.10, 0.40),
                    'DD_PENALTY_COEF': (0.5, 3.0),
                    'VOL_PENALTY_COEF': (0.005, 0.03),
                    'TURNOVER_COST_MULT': (0.1, 0.5),
                    'SORTINO_WEIGHT': (0.10, 0.35),
                    'PERSISTENCE_BONUS_SCALE': (0.1, 0.8),
                    'CAUSAL_PENALTY_WEIGHT': (0.20, 0.50),
                    'CAUSAL_REWARD_FACTOR': (0.3, 1.0),
                    'REGIME_SHORT_LOOKBACK': (48, 192),
                    'REGIME_SHORT_WEIGHT': (0.3, 0.8),
                    'REGIME_CONFIDENCE_MIN_SIZE_PCT': (0.25, 0.70),
                    'HURST_TREND_THRESHOLD': (0.35, 0.55),
                    'VIX_THRESHOLD': (22, 35),
                    'BREAKOUT_BOOST_FACTOR': (1.0, 1.5),
                    # NEW — S4 Intraday risk pacing
                    'RISK_PACING_TIER1_LOSS': (-0.010, -0.002),
                    'RISK_PACING_TIER1_SCALE': (0.3, 0.8),
                    'RISK_PACING_TIER2_LOSS': (-0.030, -0.010),
                    'RISK_PACING_TIER2_SCALE': (0.1, 0.4),
                    'RISK_PACING_CONSECUTIVE_LOSSES': (2, 5),
                    'RISK_PACING_CONSECUTIVE_LOSS_SCALE': (0.15, 0.6),
                    # NEW — S3 Asymmetric trailing
                    'RATCHET_LOSS_TIGHTEN_THRESHOLD': (-0.015, -0.003),
                    'RATCHET_LOSS_TIGHTEN_FACTOR': (0.35, 0.85),
                    'RATCHET_LOSS_TIGHTEN_MFE_MAX': (0.001, 0.010),
                    # NEW — S1 Meta-filter
                    'META_FILTER_MIN_PROB': (0.30, 0.55),
                    # NEW — A1 Cross-sectional momentum
                    'CROSS_SECTIONAL_WEIGHT': (0.5, 1.2),
                    'CROSS_SECTIONAL_MAX_MULT': (1.10, 1.50),
                    'CROSS_SECTIONAL_MIN_MULT': (0.30, 0.70),
                    'CROSS_SECTIONAL_NEUTRAL_BAND': (0.10, 0.50),
                    # NEW — B5 Anti-earnings
                    'EARNINGS_BLACKOUT_PRE_DAYS': (1, 4),
                    'EARNINGS_BLACKOUT_POST_DAYS': (0, 3),
                    'EARNINGS_CLOSE_PRE_DAYS': (0, 3),
                    # NEW — B4 Sentiment velocity
                    'SENTIMENT_VELOCITY_WEIGHT': (0.0, 0.35),
                    'SENTIMENT_VELOCITY_LOOKBACK_HOURS': (2, 8),
                    # NEW — AC Crowding discount
                    'CROWDING_DISCOUNT_THRESHOLD': (0.35, 0.75),
                    'CROWDING_DISCOUNT_STRENGTH': (0.2, 1.0),
                    'CROWDING_DISCOUNT_MIN_FACTOR': (0.25, 0.70),
                    # NEW — Portfolio meta-blend
                    'PORTFOLIO_META_WEIGHT': (0.0, 0.40),
                    # NEW — B2 Adverse selection
                    'ADVERSE_SELECTION_THRESHOLD': (-0.005, -0.0005),
                    'ADVERSE_SELECTION_MAX_PENALTY': (0.2, 0.7),
                    # NEW — BPS Bayesian sizing
                    'BAYESIAN_SIZING_MIN_MULT': (0.2, 0.7),
                    'BAYESIAN_SIZING_MAX_MULT': (1.2, 2.0),
                    'BAYESIAN_SIZING_REFERENCE_EV': (0.001, 0.006),
                    'BAYESIAN_SIZING_SHRINKAGE_N': (4, 20),
                    # NEW — ESP Slippage veto
                    'SLIPPAGE_VETO_MULTIPLE': (0.8, 2.0),
                    'SLIPPAGE_VETO_SCALE': (0.1, 0.6),
                    # NEW — PSD Divergence gate
                    'DIVERGENCE_GATE_SCALE': (0.2, 0.8),
                    'DIVERGENCE_MIN_WEIGHT': (0.01, 0.08),
                    'DIVERGENCE_MIN_META': (0.10, 0.40),
                    # NEW — KELLY sizing
                    'BAYESIAN_SIZING_KELLY_FRACTION': (0.10, 0.50),
                    'BAYESIAN_SIZING_REFERENCE_KELLY': (0.04, 0.15),
                    # NEW — REX Regime-Conditional Exits
                    'REX_ALIGN_TP_MULT': (1.15, 1.80),
                    'REX_ALIGN_TRAIL_MULT': (1.05, 1.50),
                    'REX_OPPOSE_TP_MULT': (0.50, 0.90),
                    'REX_OPPOSE_TRAIL_MULT': (0.55, 0.95),
                    'REX_MR_TP_MULT': (0.65, 1.00),
                    'REX_MR_TRAIL_MULT': (0.75, 1.05),
                }
                if key in ABSOLUTE_BOUNDS:
                    abs_min, abs_max = ABSOLUTE_BOUNDS[key]
                    clamped = max(abs_min, min(abs_max, clamped))
                # FIX: Preserve integer type for integer params
                if isinstance(current_val, int):
                    clamped = int(round(clamped))
                if clamped != current_val:
                    staged_changes[key] = clamped
        # FIX #29: Apply all changes atomically — CONFIG is only modified if ALL parsing succeeded
        for key, clamped in staged_changes.items():
            old = current_config[key]
            current_config[key] = clamped
            applied[key] = {'old': old, 'new': clamped}
            log_structured_gemini_change(param=key, old_value=old, new_value=clamped, pnl_context=pnl_context)
        # Also apply SYMBOLS change atomically (collected in applied dict above)
        if 'SYMBOLS' in applied and isinstance(applied['SYMBOLS'], list):
            current_config['SYMBOLS'] = applied['SYMBOLS']
        if applied:
            # FIX #75: Filter out non-config metadata keys (rotate_now, proposed_universe)
            # before saving — only persist actual config parameters
            metadata_keys = {'rotate_now', 'proposed_universe'}
            save_dynamic_config({k: current_config[k] for k in applied if k in current_config and k not in metadata_keys})
            logger.info(f"[GEMINI TUNER] Applied {len(applied)} changes")
        return applied
    except Exception as e:
        logger.error(f"[GEMINI TUNER] Query failed: {e}")
        return {}

# Self-test
if __name__ == "__main__":
    print("=== GEMINI TUNER TEST (with regime-aware min-hold) ===")
    test_context = {'pnl_summary': '-1.8%', 'win_rate': '51.2'}
    test_perf = {'NVDA': {'win_rate': 0.68, 'pnl': 1240}, 'SOFI': {'win_rate': 0.41, 'pnl': -320}}
    result = query_gemini_for_tuning(test_context, CONFIG, test_perf)
    print("Test completed. Applied:", result)

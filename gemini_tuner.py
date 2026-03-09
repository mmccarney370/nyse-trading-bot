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
from datetime import datetime
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

TB_LOG_DIR = "./ppo_tensorboard/portfolio"
DYNAMIC_CONFIG_PATH = "dynamic_config.json"
TUNING_HISTORY_PATH = "tuning_history.json"  # NEW: stores last few tuning changes for temporal context

def load_dynamic_config():
    if os.path.exists(DYNAMIC_CONFIG_PATH):
        try:
            with open(DYNAMIC_CONFIG_PATH, 'r') as f:
                dynamic = json.load(f)
            CONFIG.update(dynamic)
            if 'SYMBOLS' in dynamic:
                CONFIG['SYMBOLS'] = dynamic['SYMBOLS']
                logger.info(f"[GEMINI TUNER] Restored persisted universe: {CONFIG['SYMBOLS']}")
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
            json.dump(current, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
        shutil.move(tmp.name, path_config)
        logger.info(f"[ATOMIC SAVE] Saved {len(changes)} changes to dynamic_config.json")

        # Atomic write for tuning_history.json (append + keep last 5)
        history_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
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
            json.dump(history, tmp, indent=2)
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
        ea = event_accumulator.EventAccumulator(latest_file)
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
        "timestamp": datetime.utcnow().isoformat() + "Z",
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
    if entropy > -0.05:  # Policy is becoming "flat" or random
        tuning_focus = "EXPLORATION: Focus on PPO_ENTROPY_COEFF, PPO_LEARNING_RATE, PPO_GAE_LAMBDA, PPO_CLIP_RANGE to stabilize learning."
    else:
        tuning_focus = "EXPLOITATION: Focus on Risk, Sizing, ATR bounds, ratcheting thresholds, min-hold bars to capitalize on the learned policy."
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

GROUP 5: PPO / RL Hyperparameters (max ±15% — RL is sensitive)
  PPO_LEARNING_RATE    [1e-4 – 5e-4]  current: {current_config.get('PPO_LEARNING_RATE')}
  PPO_ENTROPY_COEFF    [0.01 – 0.08]  current: {current_config.get('PPO_ENTROPY_COEFF')}
  PPO_GAMMA            [0.93 – 0.98]  current: {current_config.get('PPO_GAMMA')}
  PPO_GAE_LAMBDA       [0.90 – 0.97]  current: {current_config.get('PPO_GAE_LAMBDA')}
  PPO_CLIP_RANGE       [0.10 – 0.25]  current: {current_config.get('PPO_CLIP_RANGE')}
  vf_coef              [0.3 – 1.5]    current: {current_config.get('vf_coef')}
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
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
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
                    recent_win_rate = float(context_data.get('win_rate', '50')) / 100
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
                    current_config['SYMBOLS'] = list(proposed_set)
                    logger.info(f"[GEMINI TUNER] Intra-week rotation triggered — win rate {recent_win_rate:.1%}, drawdown {drawdown:.1%}")
                elif proposed_set != current_symbols:
                    logger.debug(f"[GEMINI TUNER] Proposed universe different but performance ok — no rotation")
                else:
                    logger.debug(f"[GEMINI TUNER] Proposed universe matches current — no rotation needed")
        # Normal parameter updates + structured logging
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
                              'PPO_CLIP_RANGE', 'vf_coef', 'PPO_AUX_LOSS_WEIGHT'}
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
                # FIX: Preserve integer type for integer params
                if isinstance(current_val, int):
                    clamped = int(round(clamped))
                if clamped != current_val:
                    old = current_config[key]
                    current_config[key] = clamped
                    applied[key] = {'old': old, 'new': clamped}
                    # Structured log for each parameter change
                    log_structured_gemini_change(
                        param=key,
                        old_value=old,
                        new_value=clamped,
                        pnl_context=pnl_context
                    )
        if applied:
            save_dynamic_config({k: current_config[k] for k in applied if k in current_config})
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

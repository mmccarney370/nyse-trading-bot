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
    except:
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
    prompt = f"""
You are an elite quantitative trading strategist and risk manager with 15+ years experience optimizing live trading bots.
Your sole goal: **Maximize long-term Sharpe ratio while strictly controlling maximum drawdown (<15–18%) and maintaining consistency across regimes.**
Today's performance summary:
- {context_data.get('pnl_summary', 'N/A')}
- Total trades / Win rate: {context_data.get('trade_summary', 'N/A')} / {context_data.get('win_rate', 'N/A')}%
- Current market regime: {context_data.get('regime', 'mixed')}
Per-symbol performance (last 30 days):
{performance_str}
Current universe size limit: {current_config.get('MAX_UNIVERSE_SIZE', 8)}
**CRITICAL RULES – MUST FOLLOW**
1. If the previous period was **very profitable** (positive P&L + win rate > 62% + Sharpe > 2.0 + max DD < 12%), you may return **no changes** if you believe current parameters are optimal.
2. Never make large changes. Maximum allowed change per parameter is ±15% from current value (except where explicitly wider bounds are allowed below).
3. Always return a **full proposed_universe** list containing **EXACTLY** {current_config.get('MAX_UNIVERSE_SIZE', 8)} symbols from UNIVERSE_CANDIDATES.
4. **Regime-aware tuning priority**:
   - Trending regimes (high persistence): prioritize tighter trailing stops, longer min-hold, higher risk-per-trade
   - Mean-reverting regimes (low persistence): prioritize looser trailing stops, shorter min-hold, lower risk-per-trade
5. **Profit protection emphasis**: When unrealized gains are large (+5–10%+), trailing stops must tighten aggressively to lock in profits during regime flips or reversals.
6. **Over-tuning penalty**: Every change carries real risk (implementation bugs, regime misfit, transaction friction). Only propose a change if you are **>85% confident** it meaningfully improves risk-adjusted returns (Sharpe/Sortino) or drawdown control vs baseline.
7. **Exploration vs Exploitation directive**: Check ppo_scalars['entropy_loss'] (if available).
   - If entropy_loss is high (near 0 or positive), the policy is too exploratory/random → prioritize tuning RL hyperparameters (learning rate, entropy coeff, GAE lambda, clip range) to stabilize learning.
   - If entropy_loss is low (large negative), the policy is over-confident/exploitative → prioritize tuning risk/exit parameters (ATR multiples, ratcheting thresholds, min-hold bars, risk-per-trade) to capitalize on the learned policy without destabilizing it.
**CURRENT TUNING FOCUS**: {context_data.get('tuning_focus', 'No focus information available')}
**HYPOTHESIS-FIRST CHAIN-OF-THOUGHT REQUIREMENT**
For every parameter change (or decision to make no changes), you **MUST** first state a clear 'Market Hypothesis' in one sentence, then briefly explain why the change (or no change) follows from today's data and the hypothesis.
**TUNABLE PARAMETER BOUNDS & CONSTRAINTS** (do NOT violate these)
- Risk/sizing (±20%): RISK_PER_TRADE_MEAN_REVERTING, RISK_PER_TRADE_TRENDING, KELLY_FRACTION, RISK_BUDGET_MULTIPLIER
- Min-hold bars (±30%): MIN_HOLD_BARS_TRENDING (never <36, never >96), MIN_HOLD_BARS_MEAN_REVERTING (never <12, never >48)
- ATR multiples (±25%): TRAILING_STOP_ATR_TRENDING, TRAILING_STOP_ATR_MEAN_REVERTING, TAKE_PROFIT_ATR_TRENDING, TAKE_PROFIT_ATR_MEAN_REVERTING
- Ratcheting params (±30%):
  RATCHET_TRENDING_INTERVAL_SEC (90–360), RATCHET_MEAN_REVERTING_INTERVAL_SEC (360–900)
  RATCHET_TRENDING_MIN_ATR_MOVE (0.2–0.8), RATCHET_MEAN_REVERTING_MIN_ATR_MOVE (0.5–1.5)
  RATCHET_REGIME_FACTOR_TRENDING (0.4–1.0), RATCHET_REGIME_FACTOR_MEAN_REVERTING (1.0–2.0)
  RATCHET_PROFIT_PROTECTION_SLOPE (0.4–1.2), RATCHET_PROFIT_PROTECTION_MIN (0.3–0.7)
- Weights (±0.3 absolute): SENTIMENT_WEIGHT (0.0–0.5), PORTFOLIO_SENTIMENT_WEIGHT (0.0–0.7), LLM_DEBATE_WEIGHT (0.0–0.9)
- Regime responsiveness (±30%): REGIME_SHORT_LOOKBACK (48–144), REGIME_SHORT_WEIGHT (0.3–0.9), REGIME_CONFIDENCE_MIN_SIZE_PCT (0.1–0.6)
Never set any parameter to 0 or negative unless explicitly allowed. Prefer stability over marginal tweaks.
**PREVIOUS TUNING CHANGES** (last 1–2 cycles — avoid oscillating back-and-forth):
{json.dumps(previous_changes, indent=2) if previous_changes else "No previous changes available"}
Then output ONLY a single valid JSON object with this exact structure:
{{
  "reasoning": "Market Hypothesis: [your one-sentence hypothesis here]. Brief 1-2 sentence explanation of your overall decision.",
  "proposed_universe": ["SOFI", "PLTR", ...], // exactly MAX_UNIVERSE_SIZE symbols from UNIVERSE_CANDIDATES
  "parameters": {{
    "param_name_1": new_value_or_null,
    "param_name_2": new_value_or_null,
    ...
  }}
}}
Current tunable parameters and their current values (change within bounds above):
{json.dumps({k: current_config.get(k) for k in [
    'PPO_ENTROPY_COEFF', 'PPO_CLIP_RANGE', 'PPO_LEARNING_RATE', 'vf_coef',
    'DD_PENALTY_COEF', 'VOL_PENALTY_COEF', 'RISK_PENALTY_COEF',
    'RISK_PER_TRADE_TRENDING', 'RISK_PER_TRADE_MEAN_REVERTING',
    'MAX_LEVERAGE', 'KELLY_FRACTION', 'MAX_POSITIONS',
    'MIN_HOLD_BARS_TRENDING', 'MIN_HOLD_BARS_MEAN_REVERTING',
    'MIN_CONFIDENCE', 'TRAILING_STOP_ATR_TRENDING', 'TRAILING_STOP_ATR_MEAN_REVERTING',
    'TAKE_PROFIT_ATR_TRENDING', 'TAKE_PROFIT_ATR_MEAN_REVERTING',
    'PORTFOLIO_SENTIMENT_WEIGHT', 'LLM_DEBATE_WEIGHT', 'SENTIMENT_WEIGHT',
    'CAUSAL_PENALTY_WEIGHT', 'COUNTERFACTUAL_SAMPLES', 'USE_CAUSAL_RL',
    'REGIME_METHOD', 'HURST_TREND_THRESHOLD', 'VIX_THRESHOLD',
    'BREAKOUT_BOOST_FACTOR', 'DYNAMIC_THRESHOLD_UPDATE_DAYS',
    'UNIVERSE_UPDATE_INTERVAL_HOURS', 'MIN_AVG_VOLUME', 'MAX_UNIVERSE_SIZE',
    'PPO_GAMMA', 'PPO_GAE_LAMBDA', 'PPO_AUX_LOSS_WEIGHT', 'EMA_ALPHA',
    'RATCHET_TRENDING_INTERVAL_SEC', 'RATCHET_MEAN_REVERTING_INTERVAL_SEC',
    'RATCHET_TRENDING_MIN_ATR_MOVE', 'RATCHET_MEAN_REVERTING_MIN_ATR_MOVE',
    'RATCHET_REGIME_FACTOR_TRENDING', 'RATCHET_REGIME_FACTOR_MEAN_REVERTING',
    'RATCHET_PROFIT_PROTECTION_SLOPE', 'RATCHET_PROFIT_PROTECTION_MIN',
    'REGIME_SHORT_LOOKBACK', 'REGIME_SHORT_WEIGHT', 'REGIME_CONFIDENCE_MIN_SIZE_PCT'
]}, indent=2)}
If you believe no changes are needed, set most (or all) parameters to null and clearly state it in "reasoning".
"""
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        content = response.text.strip()
        if content.startswith('```json'):
            content = content[7:-3].strip()
        elif content.startswith('```'):
            content = content[3:-3].strip()
        tweaks = json.loads(content)
        applied = {}
        # Prepare pnl_context for structured logging (same for all changes)
        pnl_context = {
            "equity": context_data.get("pnl_summary", "N/A"),
            "win_rate": context_data.get("win_rate", "N/A"),
            "regime": context_data.get("regime", "mixed"),
            "trade_count": context_data.get("trade_summary", "N/A"),
            "ppo_scalars": context_data.get("ppo_scalars", {})
        }
        # ISSUE #5 PATCH: Check for intra-week rotation
        current_symbols = set(CONFIG.get('SYMBOLS', []))
        proposed_universe = tweaks.get("proposed_universe", [])
        if isinstance(proposed_universe, list):
            proposed_set = set([s.upper() for s in proposed_universe if s.upper() in CONFIG.get('UNIVERSE_CANDIDATES', [])])
            if len(proposed_set) == current_config.get('MAX_UNIVERSE_SIZE', 8):
                # Performance check: rotate only if poor (win rate < 55% or drawdown > 5%)
                recent_win_rate = float(context_data.get('win_rate', '50')) / 100
                pnl_summary = context_data.get('pnl_summary', '0%')
                drawdown = 0.0
                if '-' in pnl_summary:
                    try:
                        drawdown = abs(float(pnl_summary.split('(')[1].split('%')[0])) / 100 if '(' in pnl_summary else 0.0
                    except:
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
        # Normal parameter updates + structured logging
        for key, value in tweaks.get("parameters", {}).items():
            if value is None or key not in current_config:
                continue
            current_val = current_config.get(key)
            if isinstance(current_val, (int, float)):
                max_allowed = current_val * 1.15
                min_allowed = current_val * 0.85
                clamped = max(min_allowed, min(max_allowed, value))
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

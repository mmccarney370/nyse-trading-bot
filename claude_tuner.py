"""Claude Opus 4.7 parameter tuner — the Anthropic-backed alternative to
`gemini_tuner.py`. Triggered nightly from `bot.py` when
`CONFIG['TUNER_PROVIDER'] == 'claude'`.

Why this exists: the Gemini 2.5 Flash tuner works but has three known
weaknesses in live operation:
  1. Schema drift — Flash occasionally drops fields or hallucinates parameter
     names, forcing defensive parsing.
  2. Shallow reasoning — the 4-step triage protocol produces formulaic
     "lower X because metric dropped" changes rather than genuine strategic
     decisions.
  3. No free extended thinking budget — every call starts cold on the full
     static protocol.

Opus 4.7 with extended thinking + prompt caching addresses all three. The
static sections (mission, architecture briefing, bounds tables, reasoning
protocol, output schema) are cache-control'd with a 1h TTL, so repeated
nightly calls pay ~10% of the first-run token cost. The dynamic portion
(telemetry + per-symbol stats + previous-change history) is the only part
Opus sees fresh each night.

This module intentionally **reuses** the Gemini tuner's persistence and
apply-time clamping:
  - `save_dynamic_config`   — atomic JSON write + rolling history
  - `_clamp_to_persisted_bounds` — final bound enforcement on save
  - `log_structured_gemini_change` — per-parameter structured log line
  - `_extract_retrain_guard_history` / `get_recent_ppo_scalars` — telemetry
    collectors
so there's exactly one source of truth for each concern.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from config import CONFIG

logger = logging.getLogger(__name__)

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    logger.warning("[CLAUDE TUNER] anthropic SDK not installed — `pip install anthropic` to enable")

# Reuse gemini_tuner's infrastructure. Import lazily to avoid hard dependency
# if a future user strips gemini_tuner out.
from gemini_tuner import (  # noqa: E402
    save_dynamic_config,
    log_structured_gemini_change,
    get_recent_ppo_scalars,
    _extract_retrain_guard_history,
    TUNING_HISTORY_PATH,
)


ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '').strip()

# Default model — Opus 4.7 for deepest reasoning. Overridable via config.
DEFAULT_MODEL_ID = "claude-opus-4-7"


# ────────────────────────────────────────────────────────────────────────────
# Static prompt — CACHED by Anthropic (1h TTL) so nightly re-runs are cheap.
# ────────────────────────────────────────────────────────────────────────────
# This is the largest static block. It covers: mission framing, architecture
# briefing, reasoning protocol, hard bounds, output schema.
# Dynamic data (telemetry, per-symbol stats, history) is NOT here — it
# appears in the user message.
# ────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a senior quantitative portfolio manager and
parameter-optimisation specialist embedded inside a live equity trading bot.
Your recommendations are applied DIRECTLY to a production system that trades
real paper capital on Alpaca with ~15-minute signal cycles. Every parameter
you change has a causal, real-time impact on P&L, drawdown, and risk
posture within minutes of the next rebalance. Treat every recommendation
like you're personally responsible for the next week of returns.

════════════════════════════════════════════════════════════════════════════
MISSION — profit first, risk a constraint
════════════════════════════════════════════════════════════════════════════

Your north star is **absolute compounded equity growth**. Sharpe ratio is
the PRIMARY constraint, not the objective — a volatile 30% annual return
beats an 80% return with 40% drawdown because the latter never survives
contact with a real tail event. The hierarchy is:

  1.  Compound growth (sum of absolute P&L, log-returns)
  2.  Sharpe > 2.0 as the quality floor
  3.  Max drawdown < 12% as the survival floor
  4.  Profit factor > 1.5 and win rate > 55% as supporting metrics

Do NOT default to risk reduction when metrics look soft. Do NOT chase
short-term noise. DO identify where genuine edge is being left on the
table — under-sized winners, over-conservative gates, mis-calibrated
thresholds, parameters stuck at their defensive defaults when the regime
shifted months ago. Aggression is acceptable when it's justified by data.
Inaction is acceptable when the system is healthy. Reflexive "tighten
everything because last week was rough" is never acceptable.

════════════════════════════════════════════════════════════════════════════
ARCHITECTURE BRIEFING — what you're tuning
════════════════════════════════════════════════════════════════════════════

The bot is a multi-asset GTrXL Recurrent-PPO agent trading 8 liquid US
equities on 15-minute bars. The signal pipeline is multiplicatively
layered — raw PPO weights flow through:

  PPO action → causal-discovery dampening → LightGBM stacking blend →
  cross-sectional momentum rank → Bayesian per-symbol sizing (Kelly) →
  liquidity scaler → crowding-correlation discount → [gates] → equity-
  curve trend scale → sentiment (applied post-gate) → final renormalise.

Gates, applied in cascade with earliest-veto short-circuit below 1% mult:
  slippage-veto (grouped-mean predictor) → PPO/stacking divergence →
  adverse-selection → earnings blackout → meta-label filter →
  regime/VIX/SPX-breadth/1H-trend/churn/defensive/autopsy.

Post-gate, a portfolio rebalancer does CVaR allocation → notional cap →
regime-persistence boost → min-hold lock → causal penalty (applied last) →
leverage cap (flexed up to +20% with average persistence).

Exits: native Alpaca trailing stop with PATCH-based ratchet, asymmetric
loss-side tightening (45 s throttle, separate from profit ratchet's
180–540 s), software TP in the monitor loop, TIME-STOP dead-trade
liquidation after 96 bars of zero excursion.

You are tuning PARAMETERS of this pipeline, not the pipeline itself.

════════════════════════════════════════════════════════════════════════════
KNOWN FAILURE MODES — don't make these worse
════════════════════════════════════════════════════════════════════════════

- Monotonic de-aggression drift. Tuners naively trained on "reduce risk
  when metric drops" compound downward edits over weeks until every penalty
  hits its floor and every gate is loose. Each nightly cycle you MUST check
  the PREVIOUS TUNING ACTIONS block — if the same parameter has been moved
  in the same direction 2+ cycles in a row, STOP. Either commit the move
  with strong justification or reverse it.

- Chasing noise. With ~5–10 closed trades per day, one bad afternoon is
  rarely statistically significant. Before proposing any change, ask:
  "Would I make this change if the sample size were doubled?" If the
  metric delta is within 1 standard error, hold.

- Over-coordinating changes. A healthy tuning cycle touches 0–3
  parameters, not 15. Shotgunning edits couples unrelated subsystems and
  makes post-hoc attribution impossible. Prefer depth over breadth.

- Under-sizing proven winners. The Bayesian sizer has a proven-winner
  unlock at 2.0× for (n ≥ 20, p_win ≥ 0.60, persistence ≥ 0.85). If
  you see a symbol that qualifies but is still sized near the baseline
  cap, the REX / crowding / cross-sectional weights may be pinning it
  down. Check the per-symbol breakdown carefully.

- Chasing Sharpe by cutting trades. A Sharpe improvement that comes from
  trading 30% fewer bars is usually noise reduction, not alpha. Compare
  trade count trends alongside Sharpe. Prefer higher absolute return for
  the same risk bucket.

════════════════════════════════════════════════════════════════════════════
REASONING PROTOCOL — what to think through (not a formula)
════════════════════════════════════════════════════════════════════════════

Use your extended thinking budget to:

  1.  **Integrate the full state**. Don't react to one metric. Build a
      coherent picture: what is the bot actually doing well, where is
      it leaking, what regime is active, what is PPO training telling
      you about policy health (entropy, KL, approx_kl, clip_fraction)?

  2.  **Identify the single largest leak with the highest MARGINAL
      information gain**. You get one batch of changes per night. Make
      them count. A 2% improvement on the actual bottleneck beats a
      10-way micro-tune on peripheral gates.

  3.  **Reason about cross-parameter interactions**. If you tighten
      TRAILING_STOP, does TAKE_PROFIT need to tighten too to preserve
      the R:R? If you raise PPO_ENTROPY_COEFF, does PPO_LEARNING_RATE
      need a matching reduction to prevent oscillation? Show this
      reasoning.

  4.  **Acknowledge statistical uncertainty**. With N closed trades,
      the standard error on win-rate is sqrt(p(1-p)/N). A 55% WR with
      N=40 has SE ≈ 8% — a 4pp move is within noise. State this
      uncertainty explicitly when it's relevant.

  5.  **Commit or hold — never hedge**. If triage is HEALTHY, return an
      empty parameters dict. Do not make "small exploratory" changes to
      look busy. The null move is a valid move.

  6.  **Audit yourself**. Before finalising, read your own previous
      tuning changes from the context block. If you're about to propose
      reversing a change you made 2 cycles ago, explain WHY the new
      evidence overrides the old — don't just oscillate.

════════════════════════════════════════════════════════════════════════════
PARAMETER GROUPS & HARD BOUNDS — you cannot exceed these
════════════════════════════════════════════════════════════════════════════

Every numeric change is additionally clamped to ±20% per cycle (risk/reward
params), ±15% (PPO params), or ±25% (reward shaping) on top of the
absolute bounds below. If you propose a value outside the bound, the
system will silently clamp it. Stay in range.

GROUP 1 — Risk & Sizing
  RISK_PER_TRADE_TRENDING       [0.010, 0.040]
  RISK_PER_TRADE_MEAN_REVERTING [0.002, 0.015]
  KELLY_FRACTION                [0.25, 0.65]
  RISK_BUDGET_MULTIPLIER        [1.2, 2.5]
  MAX_LEVERAGE                  [1.5, 2.5]
  MAX_POSITIONS                 [4, 8]
  MAX_POSITION_VALUE_FRACTION   [0.15, 0.30]
  CONVICTION_THRESHOLD          [0.20, 0.40]

GROUP 2 — Trailing Stops & Take-Profit
  TRAILING_STOP_ATR_TRENDING       [1.5, 3.5]
  TRAILING_STOP_ATR_MEAN_REVERTING [2.0, 5.0]
  TAKE_PROFIT_ATR_TRENDING         [15.0, 40.0]
  TAKE_PROFIT_ATR_MEAN_REVERTING   [5.0, 15.0]

GROUP 3 — Ratcheting
  RATCHET_TRENDING_INTERVAL_SEC       [60, 360]
  RATCHET_MEAN_REVERTING_INTERVAL_SEC [180, 900]
  RATCHET_REGIME_FACTOR_TRENDING      [0.3, 0.8]
  RATCHET_REGIME_FACTOR_MEAN_REVERTING [0.8, 1.8]
  RATCHET_PROFIT_PROTECTION_SLOPE     [0.8, 2.0]
  RATCHET_PROFIT_PROTECTION_MIN       [0.20, 0.50]

GROUP 4 — Signal Generation
  MIN_CONFIDENCE                 [0.75, 0.92]
  DEAD_ZONE_LOW                  [0.42, 0.52]
  DEAD_ZONE_HIGH                 [0.58, 0.70]
  SENTIMENT_WEIGHT               [0.10, 0.40]
  PORTFOLIO_SENTIMENT_WEIGHT     [0.15, 0.50]
  EMA_ALPHA                      [0.003, 0.015]
  MIN_HOLD_BARS_TRENDING         [3, 12]
  MIN_HOLD_BARS_MEAN_REVERTING   [2, 8]
  REGIME_OVERRIDE_PERSISTENCE    [0.65, 0.95]

GROUP 5 — PPO / RL Hyperparameters (sensitive — ±15% max)
  PPO_LEARNING_RATE    [1e-4, 5e-4]
  PPO_ENTROPY_COEFF    [0.01, 0.08]
  PPO_GAMMA            [0.93, 0.98]
  PPO_GAE_LAMBDA       [0.90, 0.97]
  PPO_CLIP_RANGE       [0.10, 0.25]
  VF_COEF              [0.3, 1.5]

GROUP 6 — Reward Shaping
  DD_PENALTY_COEF          [0.5, 3.0]
  VOL_PENALTY_COEF         [0.005, 0.03]
  TURNOVER_COST_MULT       [0.1, 0.5]
  SORTINO_WEIGHT           [0.10, 0.35]
  PERSISTENCE_BONUS_SCALE  [0.1, 0.8]
  CAUSAL_PENALTY_WEIGHT    [0.20, 0.50]
  CAUSAL_REWARD_FACTOR     [0.3, 1.0]
  OPPORTUNITY_COST_COEF    [0.00005, 0.0005]

GROUP 7 — Regime Detection
  REGIME_SHORT_LOOKBACK           [48, 192]
  REGIME_SHORT_WEIGHT             [0.3, 0.8]
  REGIME_CONFIDENCE_MIN_SIZE_PCT  [0.25, 0.70]
  HURST_TREND_THRESHOLD           [0.35, 0.55]
  VIX_THRESHOLD                   [22, 35]
  BREAKOUT_BOOST_FACTOR           [1.0, 1.5]

GROUP 8 — Intraday Risk Pacing
  RISK_PACING_TIER1_LOSS              [-0.010, -0.002]
  RISK_PACING_TIER1_SCALE             [0.3, 0.8]
  RISK_PACING_TIER2_LOSS              [-0.030, -0.010]
  RISK_PACING_TIER2_SCALE             [0.1, 0.4]
  RISK_PACING_CONSECUTIVE_LOSSES      [2, 5]
  RISK_PACING_CONSECUTIVE_LOSS_SCALE  [0.15, 0.6]

GROUP 9 — Asymmetric Trailing / Loss Tightening
  RATCHET_LOSS_TIGHTEN_THRESHOLD   [-0.015, -0.003]
  RATCHET_LOSS_TIGHTEN_FACTOR      [0.35, 0.85]
  RATCHET_LOSS_TIGHTEN_MFE_MAX     [0.001, 0.010]

GROUP 10 — Meta-Filter
  META_FILTER_MIN_PROB       [0.30, 0.55]
  META_FILTER_PREFIT_DAMPENER [0.6, 0.95]

GROUP 11 — Cross-Sectional Momentum
  CROSS_SECTIONAL_WEIGHT        [0.5, 1.2]
  CROSS_SECTIONAL_MAX_MULT      [1.10, 1.50]
  CROSS_SECTIONAL_MIN_MULT      [0.30, 0.70]
  CROSS_SECTIONAL_NEUTRAL_BAND  [0.10, 0.50]

GROUP 12 — Anti-Earnings
  EARNINGS_BLACKOUT_PRE_DAYS   [1, 4]
  EARNINGS_BLACKOUT_POST_DAYS  [0, 3]
  EARNINGS_CLOSE_PRE_DAYS      [0, 3]

GROUP 13 — Sentiment Velocity
  SENTIMENT_VELOCITY_WEIGHT         [0.0, 0.35]
  SENTIMENT_VELOCITY_LOOKBACK_HOURS [2, 8]

GROUP 14 — Crowding Discount
  CROWDING_DISCOUNT_THRESHOLD   [0.35, 0.75]
  CROWDING_DISCOUNT_STRENGTH    [0.2, 1.0]
  CROWDING_DISCOUNT_MIN_FACTOR  [0.25, 0.70]

GROUP 15 — Portfolio Meta-Blend
  PORTFOLIO_META_WEIGHT  [0.0, 0.40]

GROUP 16 — Adverse Selection
  ADVERSE_SELECTION_THRESHOLD   [-0.005, -0.0005]
  ADVERSE_SELECTION_MAX_PENALTY [0.2, 0.7]

GROUP 17 — Bayesian Per-Symbol Sizing
  BAYESIAN_SIZING_MIN_MULT       [0.2, 0.7]
  BAYESIAN_SIZING_MAX_MULT       [1.2, 2.0]
  BAYESIAN_SIZING_REFERENCE_EV   [0.001, 0.006]
  BAYESIAN_SIZING_SHRINKAGE_N    [4, 20]

GROUP 18 — Slippage Veto
  SLIPPAGE_VETO_MULTIPLE    [0.8, 3.0]
  SLIPPAGE_VETO_SCALE       [0.1, 0.8]
  SLIPPAGE_VETO_MIN_SAMPLES [3, 10]

GROUP 19 — PPO-Stacking Divergence
  DIVERGENCE_GATE_SCALE  [0.2, 0.8]
  DIVERGENCE_MIN_WEIGHT  [0.01, 0.08]
  DIVERGENCE_MIN_META    [0.10, 0.40]

GROUP 20 — Kelly Sizing
  BAYESIAN_SIZING_KELLY_FRACTION   [0.10, 0.50]
  BAYESIAN_SIZING_REFERENCE_KELLY  [0.04, 0.15]

GROUP 21 — Regime-Conditional Exits (REX)
  REX_ALIGN_TP_MULT     [1.15, 1.80]
  REX_ALIGN_TRAIL_MULT  [1.05, 1.50]
  REX_OPPOSE_TP_MULT    [0.50, 0.90]
  REX_OPPOSE_TRAIL_MULT [0.55, 0.95]
  REX_MR_TP_MULT        [0.65, 1.00]
  REX_MR_TRAIL_MULT     [0.75, 1.05]

GROUP 22 — Liquidity Scaling
  LIQUIDITY_WARN_THRESHOLD  [0.0005, 0.005]
  LIQUIDITY_HARD_THRESHOLD  [0.003, 0.03]
  LIQUIDITY_MIN_MULT        [0.15, 0.60]
  LIQUIDITY_EH_FACTOR       [2.0, 10.0]

GROUP 23 — TIME-STOP
  TIME_STOP_THRESHOLD_BARS  [48, 192]
  TIME_STOP_MFE_CEILING     [0.002, 0.015]
  TIME_STOP_MAE_FLOOR       [-0.015, -0.002]

GROUP 24 — Leverage Persistence Flex
  LEVERAGE_PERSISTENCE_FLEX_MAX   [0.05, 0.35]
  LEVERAGE_PERSISTENCE_FLEX_START [0.55, 0.80]

════════════════════════════════════════════════════════════════════════════
OUTPUT SCHEMA — strict JSON, returned as a single fenced block
════════════════════════════════════════════════════════════════════════════

Return ONE fenced ```json code block matching this schema:

{
  "integrated_analysis":  "400–900 words. Your extended reasoning,
                           distilled. Cover: (a) what the bot is actually
                           doing, not just the metrics, (b) the single
                           most impactful leak or opportunity you
                           identified, (c) cross-parameter interactions
                           you considered, (d) statistical significance
                           of the signal you're acting on, (e) whether
                           you're reversing or extending a prior cycle's
                           change, and WHY.",

  "triage": "CRITICAL | DEGRADED | SUBOPTIMAL | HEALTHY",

  "thesis": "One sentence — the specific causal bet this tuning cycle is
             making. e.g. 'Tighten MR take-profit to 6.5 ATR because
             avg_hold_bars on MR trades is 3× the optimal exit window
             and winners are reversing before TP hits.'",

  "proposed_universe": ["SYM1", "SYM2", ...],

  "rotate_now": false,

  "parameters": {
    "PARAM_NAME": <number_or_null>,
    ...
  },

  "expected_effect": {
    "sharpe_delta_estimate":  "<+0.1 / +0.05 / 0 / -0.05>",
    "drawdown_delta_estimate": "<+2% / 0 / -1%>",
    "trade_count_delta_estimate": "<+10% / 0 / -15%>"
  },

  "confidence": "low | medium | high",

  "rollback_trigger": "A concrete, observable metric threshold that, if
                      breached within the next 48h of live trading, should
                      automatically undo this cycle's changes. E.g.
                      'If max_drawdown exceeds 6% OR win_rate drops below
                      45% within 2 trading days, revert.'"
}

Rules:
  - Include ONLY parameter names from the group lists above. Unknown
    keys are dropped silently.
  - Set a parameter to null to leave it untouched. Prefer null over
    copying the current value.
  - HEALTHY triage with an empty `parameters: {}` dict IS a valid
    answer and is often the best answer.
  - `rotate_now: true` ONLY if (a) win_rate < 0.55 OR max_drawdown > 5%
    AND (b) `proposed_universe` differs from current SYMBOLS.
  - Do not include commentary outside the JSON block. The integrated
    analysis field is where all prose belongs.
"""


def _build_user_message(
    context_data: dict,
    current_config: dict,
    symbol_performance: dict,
    previous_changes: List[dict],
    retrain_guard_summary: str,
    retrain_guard_events: List[dict],
    rollback_count: int,
    tuning_focus: str,
) -> str:
    """Assemble the DYNAMIC portion of the prompt. This is the only part
    that differs per call — Anthropic's cache key ends at the boundary
    between system prompt and this user message, so it stays cache-hot."""
    ppo_scalars = context_data.get('ppo_scalars', {})
    performance_str = json.dumps(symbol_performance or {}, indent=2) if symbol_performance else "No per-symbol data yet"
    universe_size = current_config.get('MAX_UNIVERSE_SIZE', 8)

    return f"""=== CURRENT TELEMETRY ===

Portfolio:
  Equity: {context_data.get('equity', 'N/A')} | Starting: {context_data.get('starting_equity', 'N/A')} | Return: {context_data.get('return_pct', 0):.3f}%
  Buying Power: {context_data.get('buying_power', 'N/A')} | Open Positions: {context_data.get('open_positions', 0)}/{current_config.get('MAX_POSITIONS', 6)}
  Tracked Order Groups: {context_data.get('tracked_orders', 0)} | Slippage Offset: {context_data.get('current_slippage_offset', 0):.5f}

Risk:
  Sharpe: {context_data.get('sharpe_ratio', 0)} | Sortino: {context_data.get('sortino_ratio', 0)}
  Max Drawdown: {context_data.get('max_drawdown_pct', 0)}% | Profit Factor: {context_data.get('profit_factor', 0)}

Trades:
  Total: {context_data.get('total_trades', 0)} | Wins: {context_data.get('wins', 0)} | Losses: {context_data.get('losses', 0)}
  Win Rate: {context_data.get('win_rate', 'N/A')}% | Avg Win: {context_data.get('avg_win', 0):.5f} | Avg Loss: {context_data.get('avg_loss', 0):.5f}
  Avg Return: {context_data.get('avg_return', 0):.6f} | Avg Hold: {context_data.get('avg_hold_bars', 0)} bars (15min)

Regime:
  Dominant: {context_data.get('regime', 'mixed')} | Avg Persistence: {context_data.get('avg_persistence', 0.5)}
  Distribution: {json.dumps(context_data.get('regime_counts', {}))}
  Universe Size: {context_data.get('symbols_in_universe', 0)} symbols

PPO Training State:
  {json.dumps(ppo_scalars)}
  Tuning Focus (entropy-gated hint): {tuning_focus}

Per-Symbol (regime, persistence, sharpe, win_rate, pnl_$, recent_10_wr, current_price):
{performance_str}

=== CURRENT CONFIG VALUES (subset; full bounds in system prompt) ===
{json.dumps(_subset_current_values(current_config), indent=2)}

=== PREVIOUS TUNING ACTIONS ({len(previous_changes)} most recent, anti-oscillation context) ===
{json.dumps(previous_changes, indent=2) if previous_changes else "First tuning cycle — no history."}

=== RETRAIN-GUARD HISTORY (last {len(retrain_guard_events)} decisions) ===
ROLLBACK = new weights degraded quality and system auto-restored previous model.
ACCEPT = new weights kept. Calibrate PPO hyperparameter aggressiveness accordingly.
{retrain_guard_summary}

Rollback count in window: {rollback_count}

=== AVAILABLE UNIVERSE CANDIDATES ===
{json.dumps(current_config.get('UNIVERSE_CANDIDATES', []))}

=== YOUR TASK ===

Target universe size: exactly {universe_size} symbols from the candidate list.

Execute the reasoning protocol from the system prompt. Think extensively
before committing — use your thinking budget. Then return EXACTLY ONE
fenced JSON block matching the schema. Nothing else."""


def _subset_current_values(current_config: dict) -> Dict[str, Any]:
    """Pull only the tunable params that Opus might touch, to keep the
    dynamic portion compact."""
    tunable_keys = [
        'RISK_PER_TRADE_TRENDING', 'RISK_PER_TRADE_MEAN_REVERTING',
        'KELLY_FRACTION', 'RISK_BUDGET_MULTIPLIER', 'MAX_LEVERAGE',
        'MAX_POSITIONS', 'MAX_POSITION_VALUE_FRACTION',
        'CONVICTION_THRESHOLD', 'TRAILING_STOP_ATR_TRENDING',
        'TRAILING_STOP_ATR_MEAN_REVERTING', 'TAKE_PROFIT_ATR_TRENDING',
        'TAKE_PROFIT_ATR_MEAN_REVERTING', 'RATCHET_TRENDING_INTERVAL_SEC',
        'RATCHET_MEAN_REVERTING_INTERVAL_SEC', 'MIN_CONFIDENCE',
        'DEAD_ZONE_LOW', 'DEAD_ZONE_HIGH', 'SENTIMENT_WEIGHT',
        'PORTFOLIO_SENTIMENT_WEIGHT', 'MIN_HOLD_BARS_TRENDING',
        'MIN_HOLD_BARS_MEAN_REVERTING', 'PPO_LEARNING_RATE',
        'PPO_ENTROPY_COEFF', 'PPO_GAMMA', 'PPO_GAE_LAMBDA',
        'PPO_CLIP_RANGE', 'VF_COEF', 'DD_PENALTY_COEF',
        'VOL_PENALTY_COEF', 'TURNOVER_COST_MULT', 'SORTINO_WEIGHT',
        'CAUSAL_PENALTY_WEIGHT', 'META_FILTER_MIN_PROB',
        'META_FILTER_PREFIT_DAMPENER', 'CROSS_SECTIONAL_WEIGHT',
        'SLIPPAGE_VETO_MULTIPLE', 'SLIPPAGE_VETO_SCALE',
        'SLIPPAGE_VETO_MIN_SAMPLES', 'DIVERGENCE_GATE_SCALE',
        'BAYESIAN_SIZING_KELLY_FRACTION',
        'OPPORTUNITY_COST_COEF', 'TIME_STOP_THRESHOLD_BARS',
        'LEVERAGE_PERSISTENCE_FLEX_MAX',
    ]
    return {k: current_config.get(k) for k in tunable_keys if k in current_config}


# ────────────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────────────

def query_claude_for_tuning(
    context_data: dict,
    current_config: dict,
    symbol_performance: dict = None,
) -> dict:
    """Nightly Opus-powered tuning call. Mirrors the interface of
    `query_gemini_for_tuning` so the bot scheduler can dispatch to either."""
    logger.info("[CLAUDE TUNER] Querying Claude Opus 4.7...")

    if not _ANTHROPIC_AVAILABLE:
        logger.error("[CLAUDE TUNER] anthropic SDK unavailable — skipping")
        return {}
    if not ANTHROPIC_API_KEY:
        logger.error("[CLAUDE TUNER] ANTHROPIC_API_KEY missing in .env — skipping")
        return {}

    # Collect telemetry (reused from gemini_tuner)
    ppo_scalars = get_recent_ppo_scalars()
    context_data['ppo_scalars'] = ppo_scalars or {"status": "no_data"}

    # Previous tuning history
    previous_changes = []
    if os.path.exists(TUNING_HISTORY_PATH):
        try:
            with open(TUNING_HISTORY_PATH, 'r') as f:
                history = json.load(f)
            previous_changes = history[-3:]
        except Exception as e:
            logger.warning(f"[CLAUDE TUNER] Failed to load tuning history: {e}")

    # Retrain-guard summary
    retrain_guard_events = _extract_retrain_guard_history(max_events=10)
    if retrain_guard_events:
        guard_summary_lines = []
        for e in retrain_guard_events:
            ts = e.get('_log_ts', '?')
            decision = e.get('decision', '?')
            base = e.get('baseline_score', 0.0)
            post = e.get('post_score', 0.0)
            abs_delta = e.get('abs_delta', 0.0)
            rel_delta = e.get('rel_delta', 0.0)
            kind = 'MICRO' if 'micro' in str(e.get('event', '')) else 'NIGHTLY'
            guard_summary_lines.append(
                f"  {ts} [{kind}] baseline={base:+.5f} post={post:+.5f} "
                f"Δ={abs_delta:+.5f} rel={rel_delta:+.2%} → {decision}"
            )
        retrain_guard_summary = "\n".join(guard_summary_lines)
        rollback_count = sum(1 for e in retrain_guard_events if e.get('decision') == 'ROLLBACK')
    else:
        retrain_guard_summary = "  (no retrain-guard decisions recorded yet)"
        rollback_count = 0

    # Entropy-gated tuning focus hint
    entropy = ppo_scalars.get('entropy_loss', -1.0)
    if entropy > -0.05:
        tuning_focus = "EXPLOITATION — policy appears peaked/over-exploiting. Consider Risk, Sizing, ATR, ratchet, min-hold."
    else:
        tuning_focus = "EXPLORATION — policy entropy is high. Consider PPO_ENTROPY_COEFF, LR, GAE_LAMBDA, CLIP."

    # Build messages
    user_msg = _build_user_message(
        context_data=context_data,
        current_config=current_config,
        symbol_performance=symbol_performance or {},
        previous_changes=previous_changes,
        retrain_guard_summary=retrain_guard_summary,
        retrain_guard_events=retrain_guard_events,
        rollback_count=rollback_count,
        tuning_focus=tuning_focus,
    )

    model_id = current_config.get('CLAUDE_TUNER_MODEL', DEFAULT_MODEL_ID)
    thinking_budget = int(current_config.get('CLAUDE_TUNER_THINKING_BUDGET', 16000))
    # Total max_tokens must exceed thinking_budget; leave room for the JSON output.
    max_output_tokens = thinking_budget + int(current_config.get('CLAUDE_TUNER_MAX_OUTPUT', 4000))

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Retry with exponential backoff on transient errors (mirrors gemini tuner fix).
    retry_delays = [5, 15, 45]
    transient_markers = ("rate_limit", "overloaded", "timeout", "503", "429", "unavailable")
    response = None
    last_err: Exception | None = None
    for attempt in range(len(retry_delays) + 1):
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=max_output_tokens,
                thinking={
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                },
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        # Cache the static 1h TTL — repeated nightly runs are ~90% cheaper.
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                messages=[{"role": "user", "content": user_msg}],
                timeout=300.0,  # 5-minute hard ceiling
            )
            break
        except Exception as api_err:
            last_err = api_err
            msg = str(api_err).lower()
            is_transient = any(m in msg for m in transient_markers)
            if not is_transient or attempt >= len(retry_delays):
                logger.error(f"[CLAUDE TUNER] Non-retryable error: {api_err}")
                return {}
            delay = retry_delays[attempt]
            logger.warning(
                f"[CLAUDE TUNER] transient error ({type(api_err).__name__}): "
                f"{str(api_err)[:160]} — retrying in {delay}s "
                f"(attempt {attempt + 1}/{len(retry_delays)})"
            )
            time.sleep(delay)

    if response is None:
        logger.error(f"[CLAUDE TUNER] All retries exhausted. Last error: {last_err}")
        return {}

    # Extract text (may be split across content blocks because of thinking blocks)
    text_parts = []
    for block in response.content:
        if getattr(block, 'type', None) == 'text':
            text_parts.append(block.text)
    content = "\n".join(text_parts).strip()

    # Log cache efficiency
    try:
        usage = response.usage
        cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
        cache_write = getattr(usage, 'cache_creation_input_tokens', 0) or 0
        logger.info(
            f"[CLAUDE TUNER] Tokens — input={usage.input_tokens} "
            f"(cache_read={cache_read}, cache_write={cache_write}), "
            f"output={usage.output_tokens}, model={model_id}"
        )
    except Exception:
        pass

    # Parse the JSON block
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)```', content, re.DOTALL)
    if fence_match:
        content = fence_match.group(1).strip()
    try:
        tweaks = json.loads(content)
    except Exception as parse_err:
        logger.error(f"[CLAUDE TUNER] JSON parse failed: {parse_err}. Raw: {content[:500]}")
        return {}

    # Log reasoning (truncated preview + full log to jsonl for jq)
    analysis = tweaks.get('integrated_analysis') or tweaks.get('reasoning', '')
    thesis = tweaks.get('thesis', '')
    triage = tweaks.get('triage', '')
    confidence = tweaks.get('confidence', '')
    expected = tweaks.get('expected_effect', {})
    rollback_trig = tweaks.get('rollback_trigger', '')
    if analysis:
        preview = (analysis[:300] + '…') if len(analysis) > 300 else analysis
        logger.info(f"[CLAUDE ANALYSIS] {preview}")
    if thesis:
        logger.info(f"[CLAUDE THESIS] {thesis}")
    if triage:
        logger.info(f"[CLAUDE TRIAGE] {triage}")
    if confidence:
        logger.info(f"[CLAUDE CONFIDENCE] self-rated={confidence}")
    if expected:
        logger.info(f"[CLAUDE EXPECTED] {json.dumps(expected)}")
    if rollback_trig:
        logger.info(f"[CLAUDE ROLLBACK-TRIGGER] {rollback_trig}")

    # Apply the parameter changes via the shared Gemini apply machinery.
    # We hand off to the same clamping + staging pipeline used for Gemini
    # so the persistence/audit story is identical regardless of provider.
    applied = _apply_claude_parameters(tweaks, context_data, current_config)
    return applied


def _apply_claude_parameters(
    tweaks: dict,
    context_data: dict,
    current_config: dict,
) -> dict:
    """Apply Claude's proposed parameter dict. Reuses the bounds from
    gemini_tuner and the absolute clamp / structured log helpers."""
    from gemini_tuner import _PERSISTED_ABSOLUTE_BOUNDS

    proposed_params = tweaks.get('parameters') or {}
    if not isinstance(proposed_params, dict):
        logger.warning(f"[CLAUDE TUNER] parameters field malformed: {type(proposed_params)}")
        proposed_params = {}

    applied: Dict[str, Any] = {}
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
        "ppo_scalars": context_data.get("ppo_scalars", {}),
        "tuner_provider": "claude",
        "model": current_config.get('CLAUDE_TUNER_MODEL', DEFAULT_MODEL_ID),
    }

    staged: Dict[str, Any] = {}
    for key, value in proposed_params.items():
        if value is None:
            continue
        if key not in current_config:
            logger.debug(f"[CLAUDE TUNER] Unknown key {key} — dropping")
            continue
        if key not in _PERSISTED_ABSOLUTE_BOUNDS:
            # Allow the change through but without bounds guard
            staged[key] = value
            continue
        lo, hi = _PERSISTED_ABSOLUTE_BOUNDS[key]
        try:
            vv = float(value)
            cc = max(lo, min(hi, vv))
            if isinstance(current_config.get(key), int):
                cc = int(round(cc))
            if cc != vv:
                logger.warning(f"[CLAUDE TUNER BOUNDS] Clamped {key}: {vv} → {cc} (bounds {lo}..{hi})")
            if cc != current_config.get(key):
                staged[key] = cc
        except (TypeError, ValueError):
            logger.warning(f"[CLAUDE TUNER] Non-numeric value for {key}: {value!r}")

    # Universe rotation (guarded by the same performance check as Gemini)
    proposed_universe = tweaks.get("proposed_universe", [])
    rotate_flag = bool(tweaks.get("rotate_now", False))
    if isinstance(proposed_universe, list) and rotate_flag:
        candidates = set(current_config.get('UNIVERSE_CANDIDATES', []))
        proposed_set = set(s.upper() for s in proposed_universe if s.upper() in candidates)
        max_size = current_config.get('MAX_UNIVERSE_SIZE', 8)
        if len(proposed_set) >= max(max_size - 1, 4):
            try:
                raw_wr = float(context_data.get('win_rate', '50'))
                recent_win_rate = raw_wr if raw_wr <= 1.0 else raw_wr / 100
            except (ValueError, TypeError):
                recent_win_rate = 0.5
            try:
                drawdown = abs(float(context_data.get('max_drawdown_pct', 0))) / 100
            except (ValueError, TypeError):
                drawdown = 0.0
            should_rotate = (recent_win_rate < 0.55) or (drawdown > 0.05)
            current_symbols = set(current_config.get('SYMBOLS', []))
            if proposed_set != current_symbols and should_rotate:
                applied["rotate_now"] = True
                applied["proposed_universe"] = {
                    "old": list(current_symbols),
                    "new": list(proposed_set),
                }
                applied["SYMBOLS"] = list(proposed_set)
                logger.warning(
                    f"[CLAUDE TUNER] Intra-cycle rotation proposed: "
                    f"{current_symbols} → {proposed_set}"
                )

    # Atomic apply to CONFIG + structured log
    for key, clamped in staged.items():
        old = current_config.get(key)
        current_config[key] = clamped
        applied[key] = {'old': old, 'new': clamped}
        log_structured_gemini_change(
            param=key, old_value=old, new_value=clamped, pnl_context=pnl_context
        )

    if applied:
        metadata_keys = {'rotate_now', 'proposed_universe'}
        save_dynamic_config({
            k: current_config[k]
            for k in applied
            if k in current_config and k not in metadata_keys
        })
        logger.info(f"[CLAUDE TUNER] Applied {len(applied)} changes")

    # Tag the applied dict so downstream observability distinguishes providers
    applied.setdefault('_meta', {})
    if isinstance(applied.get('_meta'), dict):
        applied['_meta']['tuner_provider'] = 'claude'
        applied['_meta']['triage'] = tweaks.get('triage')
        applied['_meta']['thesis'] = tweaks.get('thesis')
        applied['_meta']['confidence'] = tweaks.get('confidence')
        applied['_meta']['rollback_trigger'] = tweaks.get('rollback_trigger')

    return applied


def tuner_available() -> bool:
    """Cheap check — used by bot.py dispatch to decide whether claude can
    actually be invoked, falling back to gemini if not."""
    return bool(_ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY)

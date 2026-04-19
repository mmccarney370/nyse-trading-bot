---
description: Deep end-to-end audit of the trading bot — find profit leaks, silent bugs, underweighted signals, and bleeding-edge opportunities. Goal is absolute P&L + high Sharpe.
---

# MEGATRON — absolute profit-maximization audit

You are conducting a rigorous, end-to-end audit of the Alpaca/NYSE trading bot at `/home/matthew/nyse_bot/trading_bot`. The single north-star metric is **absolute compounded P&L**, with **Sharpe ratio as a primary constraint** (prefer high Sharpe when it conflicts with raw return; a stable 30% beats 80% with 40% drawdown in real deployment).

Before starting, read `memory/MEMORY.md` and the most recent commits to understand current state:
```bash
python -c "import os; [print(os.path.join(r,f)) for r,_,fs in os.walk(os.path.expanduser('~/.claude/projects/-home-matthew-nyse-bot-trading-bot/memory')) for f in fs if f.endswith('.md')]"
git log --oneline -30
```

## Goals — in strict priority order

1. **Find profit leaks** — anywhere the pipeline is losing edge to fees, slippage, bad timing, noise, dim-mismatch, silent defaults, or premature rejection gates.
2. **Find bugs** — especially silent ones. A component returning a constant default, a dimension mismatch zero-padded, a gate firing on stale state, a multiplier collapsing to 1.0, a weight dropping to 0 without being logged. These are the most dangerous because they don't crash — they just bleed.
3. **Find unused / underweighted signal components** — any feature, model output, or observation dimension that's computed but not consumed, or consumed with wrong weight. Stacking ensemble predictions that get dropped; LLM sentiment blended with 0 weight; causal penalties applied twice; TFT embeddings where most dims are pad; meta-filter rejecting signals but also dampening survivors.
4. **Find novel edge** — architectural upgrades, training recipes, features, data sources, or execution tricks not yet present that can push profitability past the current ceiling. Be genuinely creative but technically grounded.

## Partition the work across parallel Explore agents

Send these five agents in a single message so they run concurrently. Each agent gets a narrow remit and must cite `file:line` for every finding.

### Agent A — Signal blend & gate ordering
Read `strategy/signals.py` (the portfolio path `generate_portfolio_actions` AND any per-symbol sync path), `strategy/portfolio_rebalancer.py`, `strategy/regime.py`, `strategy/cross_sectional.py`, `strategy/cognitive.py`, `strategy/correlation_discount.py`, `strategy/earnings_filter.py`, `strategy/meta_filter.py`, `strategy/adverse_selection.py`, `strategy/liquidity_scaler.py`, `models/causal_signal_manager.py`.

Trace the complete flow from raw features → PPO inference → stacking ensemble blend → sentiment blend → causal penalty → Bayesian sizing → cross-sectional rank → crowding/correlation discount → sentiment level + velocity → regime gate → VIX gate → meta-filter → adverse-selection → slippage veto → divergence gate → earnings blackout → equity-curve scale → final weight + direction.

Look specifically for:
- Components where `weight * value` can silently contribute zero (e.g. `has_sentiment=False` path, or sentiment 8B fallback returning NaN)
- Gates whose ordering causes one to overwrite another (PPO-override vs meta-filter, divergence-gate vs stacking blend)
- Multipliers that are pure penalties with no symmetric boost
- Places where a stale cache value (regime_cache, signal_history, sentiment_cache) is used as "fresh"
- Signs that a factor is applied to both weight AND confidence (double-counting via `gate_mult` + `stacking_meta_blend`)
- Meta-filter Brier score being tracked but never consumed as a confidence dampener
- Alpha attribution layers that sum to <1.0 (entry weight bleed) without being logged

### Agent B — Risk, sizing, and exits
Read `strategy/risk.py`, `strategy/bayesian_sizing.py`, `strategy/portfolio_rebalancer.py`, `strategy/execution_scorecard.py`, `broker/alpaca.py`, `broker/stream.py`, `broker/order_tracker.py`.

Trace every code path from signal → position size: Kelly, CVaR portfolio allocation, Bayesian posterior, notional floor, MAX_POSITION_VALUE_FRACTION, buying-power cap, MAX_TOTAL_RISK_PCT, MAX_SECTOR_CONCENTRATION, Bayesian pause, recovery ladder, slippage veto, R:R rejection, and extended-hours handling.

Then trace every exit path: trailing-stop ratchet (ReplaceOrderRequest PATCH), TP monitor loop, software TP for fractional qty, TIME-STOP dead-trade liquidation, safe_close_position, SL cooldown, circuit breaker, daily-loss threshold, reconcile-on-startup orphan recovery.

Look specifically for:
- Asymmetries where SL is tighter than 1R while TP requires >2R at current WR
- Sizing paths that take a max/min silently (e.g. min(Kelly, 0.08) that ignores high-conviction signals)
- Exits that reset HWM or clear state mid-trade (ratchet replacing before old order cancels)
- Places where current_price or fill_price could be stale (websocket fill vs monitor loop)
- CVaR optimizer failure modes that fall back to uniform allocation (cvxpy infeasible)
- Order-group state machine transitions that can deadlock (pending_exit → open regression)
- Bayesian sizing that caps downside but never upsizes a winning regime

### Agent C — Training, learning loops, and reward shaping
Read `models/trainer.py`, `models/portfolio_env.py`, `models/env.py`, `models/policies.py` (GTrXL), `models/ppo_utils.py`, `models/stacking_ensemble.py`, `models/causal_rl_manager.py`, `models/bot_initializer.py`, `models/features.py`, `gemini_tuner.py`.

Look specifically for:
- Reward terms that push toward do-nothing local minima (DD penalty, vol penalty, turnover cost with no offsetting opportunity cost)
- Training-OOS feature distribution gaps (regime imbalance between train and validate)
- Components trained but never consumed at inference time (aux volatility head, TFT embeddings)
- Stacked regularizers that over-constrain the policy (entropy floor + VF_COEF + aux loss all firing)
- Checkpoints loaded with stale obs_dim, stale symbol universe, or swapped optimizer
- Gemini tuner parameter bounds that drift toward "less risk" over time (monotonic de-aggression)
- Walk-forward OOS acceptance gate rejecting good regimes along with bad (currently floor=-0.25, cap=35%)
- GTrXL train/inference mismatch: segment memory and positional encoding differences
- PPO aux volatility loss effectively a no-op (SB3 rollout buffer lacks required info keys)
- Stacking ensemble label generation off-by-one with PPO action timing

### Agent D — Data ingestion, execution fidelity, microstructure
Read `data/handler.py`, `data/ingestion.py`, `utils/local_llm.py`, `utils/helpers.py`, `strategy/slippage_predictor.py`, `strategy/execution_scorecard.py`, `strategy/alpha_attribution.py`.

Look specifically for:
- Data sources that are imported but never populate their expected fields
- ArcticDB vs Polygon vs Tiingo vs yFinance fallback ordering that silently degrades quality
- Volume/spread/VWAP fields that are zero because the feed doesn't supply them
- Extended-hours bars contaminating regime detection or feature stats
- Feature generation that silently pads with zeros when a symbol has <N bars
- Local LLM (Ollama 8B) sentiment returning NaN or neutral (0.0) and being blended at weight>0 anyway
- Slippage predictor samples that never update because record() isn't called on every fill path
- Alpha attribution gaps (exit_only records) indicating entry-side recorder isn't called in all entry paths
- Missing data that would materially improve signals (e.g. intraday breadth, index futures, VIX term structure, sector ETF flows, options-implied vol, NYSE TICK/TRIN, FOMC/CPI blackout windows, pre-market gap classifier)

### Agent E — Bleeding-edge proposals (no code reading required)
This agent proposes ≥ 3 concrete new capabilities NOT currently in the codebase. Be specific, technically grounded, and honest about implementation cost.

Eligible categories:
- Microstructure signals: orderbook imbalance (Level 2 via Polygon), trade-flow toxicity (VPIN), realized-volatility decomposition, quote-stuffing detection, sweep detection
- Alternative data: short-interest momentum, 13F flows, insider transaction clustering, FTD (failure-to-deliver) data, Fed repo/RRP usage as liquidity proxy, dark-pool print ratio
- RL upgrades: offline RL from historical fills (CQL, IQL), decision-transformer for action prediction, Dreamer-style world-model planning, meta-learning across regimes (MAML), ensemble policy distillation, RLHF on human-labeled good trades
- Execution: adaptive limit placement via Bayesian bandit on offset, TWAP/VWAP slicing for large orders, post-only during low-vol windows, iceberg detection on own orders, PFOF-aware venue selection
- Portfolio construction: entropy-regularized allocation, HRP hierarchical risk parity, black-litterman with Gemini-generated views, regime-conditional Sharpe targeting, tail-risk-budget allocation
- Robustness: adversarial training (synthesize worst-case 15-min windows), distributional shift detection (KS test on features), regime-conditional model ensemble, test-time adaptation
- Novel outputs: predicting 30-min forward-return distribution (not point estimate), next-hour max-drawdown, correlation-regime-change predictor, earnings-reaction direction classifier

For each proposal: what it adds, how it integrates, rough effort (hours/days), expected Sharpe uplift band.

## Deliverable — structured report

Output a single consolidated report. For each finding:

- **Name** — 3-8 word descriptor
- **Severity** — P0 (bleeding profit now) / P1 (high) / P2 (medium) / P3 (low)
- **Evidence** — `file:line` + quoted code snippet
- **Mechanism** — 1-3 sentences explaining how it leaks profit or caps upside
- **Fix** — concrete code change, config flag, or new module. What to write, not just "fix this".
- **Expected Sharpe impact** — your honest band: >+0.5 / +0.1 to +0.5 / marginal / unknown

Group by category: Signal / Risk / Training / Data / Novel. **Put the top 10 findings by expected Sharpe impact at the very top** of the report for immediate action.

## Hard requirements

1. **Cite `file:line`** for every code reference. No hand-waving.
2. **Quote real code** where a finding depends on specific logic.
3. **Be blunt about severity** — don't hedge on P0s.
4. **Propose concrete fixes** — don't stop at "this is wrong", say what to write instead.
5. **Include at least 3 novel proposals** from Agent E that don't exist in the codebase yet.
6. **Don't re-propose already-applied fixes.** The following are already in place: CVaR portfolio allocation, Bayesian sizing, meta-filter, adverse-selection filter, slippage predictor veto (Apr-19 tuned to 2.0× with min-samples gate), PPO–stacking divergence gate, cross-sectional rank, correlation/crowding discount, earnings blackout, equity-curve scale, sentiment blend (8B Ollama primary, 70B retired), HMM regime ensemble with persistence, regime-aligned trailing-stop + TP multipliers, trailing-stop ratchet via PATCH, OrderTracker state machine with startup reconcile, hard-to-borrow DAY-TIF fallback (Apr-19), prefer-whole-shares for trailing-stop compatibility (Apr-19), software TP monitor for fractional qty, TIME-STOP dead-trade liquidation, Gemini tuner with retry/backoff (Apr-19), walk-forward OOS acceptance floor (Apr-19), causal lazy-build background thread, causal cache with hash match, alpha-attribution persistence across restart (Apr-19), execution scorecard daily emit, PPO nightly + micro-retrain, universe rotation Fridays 20:00 ET, extended-hours + fractional shares, per-symbol locks preventing TOCTOU between monitor and stream, asyncio.to_thread wrap around all blocking Alpaca calls, threading.Lock on OrderTracker, ArcticDB tick store, Polygon primary + Tiingo/yFinance fallback, config validation via Pydantic. Acknowledge these if relevant but don't re-propose them.

## What NOT to do

- Don't fix anything in this run — this is a pure audit. The user will pick which findings to implement.
- Don't interrupt a running backtest or training cycle. If the bot is in its trading loop or the trainer is mid-fit, don't stop it. Work entirely in read-mode.
- Don't edit files. Report only.
- Don't recommend disabling features. Recommend fixing, tuning, or replacing.

## Output length guide

- Top-10 summary: ~300 words
- Per-category findings: ~150 words per P0, ~80 per P1, ~40 per P2-P3
- Novel proposals: ~200 words each
- Total target: 2000-3500 words. If longer, truncate the P3s first — never drop P0s or novel proposals.

Start by spawning the five Explore agents in parallel (single message, multiple Agent tool calls), then synthesize their findings into the final report.

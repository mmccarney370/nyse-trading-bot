# NYSE Trading Bot — Technical Whitepaper

**Author:** Matthew McCartney
**Version:** 2.2 (April 2026)
**Status:** Live paper trading

---

## Abstract

This document describes the technical architecture of an autonomous, self-improving equity trading system combining deep reinforcement learning, causal inference, Kelly-fractional Bayesian sizing, Lopez-de-Prado meta-labeling, ensemble regime detection, regime-conditional exits, liquidity-scaled sizing, and LLM-powered hyperparameter optimization. The system trades a configurable universe of US equities on 15-minute bars via the Alpaca API, with 18+ multiplicative signal layers and 100+ auto-tuned parameters. The design emphasizes strict additivity (every new layer defaults to neutral on insufficient data), state persistence across restarts, and explicit observability of every decision point.

---

## 1. Design Philosophy

### 1.1 Multiplicative Layered Gating
Every signal transformation is a multiplier on the target weight — no layer can turn a zero weight into a non-zero one. When uncertainty is high (insufficient training data, model failure, missing inputs), each layer defaults to **1.0 (neutral pass-through)** rather than making confident-but-wrong decisions. This means turning on any new layer is strictly additive: we can ship experimental features without risk of them making the baseline worse.

### 1.2 Observable by Default
Every subsystem logs with a searchable prefix (e.g., `[BAYESIAN-SIZE]`, `[CAUSAL PENALTY APPLIED]`, `[REGIME ENSEMBLE+HMM]`). Parameter changes are logged as structured JSON with full performance context. Internal state (MFE/MAE per position, Bayesian posteriors, slippage samples, drift samples, causal edges) is queryable on disk at any time.

### 1.3 Data-Driven Adaptation
The system maintains dedicated state files for every learned quantity — `meta_filter.pkl`, `bayesian_sizing.pkl`, `slippage_predictor.pkl`, `adverse_selection.pkl`, `earnings_cache.pkl`, `causal_cache_portfolio.pkl`. All round-trip cleanly across restarts and are versioned where schema changes matter (e.g., `regime_cache.json` has `__version__=2`).

### 1.4 Bounded Autonomy
The Gemini self-tuner can modify 100+ parameters but each is clamped by hard bounds validated via Pydantic. Per-cycle step sizes are category-capped (PPO params ±15%, risk params ±20%, reward-shaping ±25%). Every proposal is diffed against history (anti-oscillation).

---

## 2. Signal Pipeline

The portfolio-level signal pipeline transforms market data into position changes through 18 sequential multiplicative layers. Each symbol's target weight `w_s ∈ [-2, +2]` flows through:

```
w_s  ←  PPO(portfolio_obs)                     [base signal]
w_s  ←  w_s · causal_factor(obs, action)       [Layer 1: causal penalty]
w_s  ←  w_s · (1 + W_meta · meta_signed_s)     [Layer 2: stacking ensemble overlay]
w_s  ←  w_s · bps_kelly_multiplier(s)          [Layer 3: Kelly-fractional Bayesian sizing]
w_s  ←  w_s · liquidity_scaler(s, w_s, equity) [Layer 4: liquidity-scaled sizing]
w_s  ←  w_s · cs_multiplier(s)                 [Layer 5: cross-sectional momentum]
w_s  ←  w_s · crowding_discount(s, w_*)        [Layer 6: correlation-aware]
w_s  ←  w_s · (1 + W_sent · sent_s) · vel_factor_s [Layer 7: sentiment level+velocity]
if in_earnings_blackout(s): w_s ← 0             [Layer 8: anti-earnings]
if P_meta_label(s) < θ: w_s ← 0                 [Layer 9: meta-label filter]
if slippage_veto(s): w_s ← w_s · 0.3            [Layer 10: slippage prediction]
if divergence_strong(s): w_s ← w_s · 0.5        [Layer 11: PPO-stacking disagree]
w_s  ←  w_s · ava_multiplier(s)                [Layer 12: adverse selection]
w_s  ←  w_s · regime_gate(s, regime)           [Layer 13: regime direction]
w_s  ←  w_s · vix_gate(VIX, direction)         [Layer 14: volatility regime]
w_s  ←  w_s · spx_gate(SPX<200SMA, direction)  [Layer 15: market breadth]
w_s  ←  w_s · hourly_trend_gate(s, direction)  [Layer 16: 1h alignment]
w_s  ←  w_s · churn_defensive(s)               [Layer 17: adaptive memory]
w_s  ←  w_s · eq_curve_scale                   [Layer 18: equity-curve trader]
risk_budget_today ← CVaR_budget · pacing_scale  [intraday pacing on CVaR budget]
final_w_s ← notional_cap(w_s · risk_budget_today / price_s)

# Exit-side (applied at order submission, not in weight pipeline):
trail_pct_s  ← regime_base_trail · REX_alignment_mult(regime, direction_s)
tp_price_s   ← regime_base_tp    · REX_alignment_mult(regime, direction_s)
```

---

## 3. Subsystems

### 3.1 Ensemble Regime Detector (`strategy/regime.py`)

**Motivation.** The previous HMM-alone detector labeled every symbol `mean_reverting` with persistence 0.95+ across all market conditions — including clear multi-week rallies. HMMs with 2 Gaussian states on (return, rolling_vol, log_vol_change) tend to collapse into high-vs-low-volatility clusters that don't map cleanly to trending vs mean-reverting.

**Design.** Five independent voters, each producing `(regime ∈ {trending_up, trending_down, mean_reverting}, strength ∈ [0,1])`:

1. **Slope voter.** Linear-regression slope over last 50 bars. Trending when `|slope · lookback / mean_price| > τ` (default 2.5%). Sign gives direction.
2. **ADX voter.** Classic Wilder ADX on 14 bars. `ADX ≥ 25` → trending (direction from +DI vs -DI); `ADX ≤ 20` → mean-reverting; middle = neutral.
3. **Autocorrelation voter.** Lag-1 return autocorrelation `ρ`. Strong positive → momentum/trending; strong negative → mean-reverting.
4. **Range-expansion voter.** Short ATR / long ATR ratio. `ratio ≥ 1.3` → trending breakout; `ratio ≤ 0.85` → consolidation.
5. **HMM voter.** Existing ensemble of 4 GaussianHMMs from seeds {42, 123, 456, 789}, weighted by mean persistence.

**Aggregation.**
```
signed = Σᵢ sᵢ · {+1 if up, -1 if down, 0 if MR}
up_strength = Σᵢ sᵢ where regimeᵢ = up
down_strength = Σᵢ sᵢ where regimeᵢ = down
neutral_strength = Σᵢ sᵢ where regimeᵢ = MR
trend_dominance = (up + down) / (up + down + neutral)
direction_consistency = |signed| / (up + down)

if trend_dominance < θ_dom OR direction_consistency < 0.5:
    return mean_reverting, persistence = clip(0.4 + 0.5·(1-dominance), 0.4, 0.95)
else:
    regime = up if signed >= 0 else down
    persistence = clip(0.55 + 0.4 · dominance · consistency, 0.55, 0.95)
```

**Validation.** Live test on a trending tape: under the old HMM-only detector all symbols in the universe were labeled `mean_reverting` with persistence 0.95+ (ceiling-stuck). The 5-voter ensemble correctly differentiated — a majority of symbols labeled `trending_up` with persistence 0.70-0.86 — catching the rally the HMM was missing, with meaningful variation in confidence across symbols.

**Wire-up (`bot.py:_get_all_regimes`).** Previously bypassed `detect_regime` and used a 20-bar slope/volatility heuristic inline. Now delegates to `strategy.regime.detect_regime` on every cache miss, so the ensemble voters actually execute on the live trading path. Cache is versioned (`__version__: 2` in `regime_cache.json`) so stale pre-ensemble entries are discarded on startup.

---

### 3.2 Causal RL Overlay (`models/causal_signal_manager.py`)

**Motivation.** The PPO policy learns `π(action | state)` through reward maximization, but many "profitable" signals are spurious correlations that fail out-of-sample. A causal DAG over (features, action, reward) lets us detect whether the action→reward relationship is genuine — if yes, we can amplify; if no, we dampen.

**Graph discovery.** GES (Greedy Equivalence Search via `pgmpy`) over stacked feature matrix + real (obs, action, reward) from replay buffer. Produces a PDAG (partially directed acyclic graph).

**Fast-path penalty.** Rather than running full DoWhy estimand identification + causal estimation (which takes 30-60 min per symbol on 54-node graphs), we use:
```
has_path(action, reward) in GES graph → binary causal coupling flag
corr(action, reward) on buffer sample → magnitude
factor = 1.0 + (corr · W_causal)  where W_causal = 0.40
factor = clip(factor, 0.5, 1.2)
```
Full DoWhy identification runs asynchronously in a background thread for operator diagnostics but doesn't gate trading decisions.

**Bootstrap.** A converged PPO policy is near-deterministic in inference (`action_std ≈ 0.08` on portfolio rollouts). GES with BIC scoring cannot identify action→reward edges when treatment variance is this low — we observed **0 edges discovered** on the first live run. Fix: during startup bootstrap, inject Gaussian exploration noise:
```
action_bootstrap = action_ppo + N(0, σ=0.4)
```
applied to the env BEFORE `step()` so the reward reflects the noisy action. This provides GES with treatment variance while the mean action still matches production PPO. Result: GES now discovers a non-trivial number of action→reward edges on first run with bootstrap noise (typical: 15-30 edges depending on the specific feature-variance profile of the current universe).

**Column cap for tractability.** GES complexity is approximately `O(p³·n)`. Portfolio observation is 460-dim so raw buffer samples produce 462-column matrices (invariant to sample size `n`). On this width, GES did not complete in 30 minutes of compute. Fix: cap GES input to top `k=58` features by variance, always preserving `action` and `reward` columns. Per-symbol buffers (obs dim 56) are unaffected. With cap, GES completes in 6-7 seconds.

**Staleness & re-bootstrap.** On startup, if the existing buffer's dominant-shape subset has action std < 0.2 (i.e., data is too deterministic for GES), the system clears the buffer + deletes the cached graph, then re-bootstraps with fresh noise-injected rollouts. Prevents cached 0-edge graphs from silently vetoing the penalty layer.

---

### 3.3 Lopez-de-Prado Meta-Labeling Filter (`strategy/meta_filter.py`)

**Motivation.** Per Advances in Financial Machine Learning (Marcos Lopez de Prado, 2018), meta-labeling separates the "primary model" (directional prediction) from the "secondary model" (bet-sizing/filtering). The primary model suggests trades; the secondary model answers "given the primary says go long here, is this actually a good trade?" Published evidence shows this separation consistently raises win rate and Sharpe.

**Features (12 scalar dims):**
`direction`, `|confidence|`, `ppo_strength`, `conviction`, `hour_of_day`, `day_of_week`, `symbol_id`, `regime_flag ∈ {-1, 0, +1}`, `persistence`, `sentiment`, `vix / 20`, `size_rel = notional / equity`

**Model.** LightGBM binary classifier:
- `objective=binary`, `num_leaves=8`, `max_depth=3`, `learning_rate=0.05`
- `min_data_in_leaf = max(3, n/15)` (scales to sample size)
- `num_boost_round=60`, `lambda_l2=0.1`
- Designed for small-N (30-100 closed trades) without over-fitting

**Training.** Nightly at 03:30 ET and at startup. Labels: `1 if realized_return > 0 else 0`.

**Inference (gate).** For each candidate weight with `direction ≠ 0`:
```
P(win) = classifier.predict_proba(features)
if P(win) < θ_min (default 0.33):
    w_s ← 0      (hard reject)
```
Or, optionally, `mode='scale'` for ×0.2 dampening instead of hard reject.

**Threshold tuning.** The appropriate `META_FILTER_MIN_PROB` depends on the model's base rate and mean prediction. A threshold significantly above mean prediction rejects too aggressively (most candidates fall near the mean). The default `0.33` is set just above the observed long-run base win rate and rejects only clearly-below-baseline candidates; Gemini auto-tunes this based on observed rejection rate vs. subsequent live win-rate calibration.

**Pass-through on insufficient data.** If fewer than `META_FILTER_MIN_TRAIN=30` closed trades exist, the classifier is not fit and `should_enter()` returns `True` by default. Strictly additive.

---

### 3.4 Kelly-Fractional Bayesian Per-Symbol Sizing (`strategy/bayesian_sizing.py`)

**Motivation.** Unbalanced per-symbol P&L distributions are common in multi-asset strategies — a handful of winners can carry the book while other names consistently bleed. Flat CVaR sizing treats all symbols identically, losing opportunity to allocate more to proven winners and reduce exposure to losers. A posterior-based sizer solves this by maintaining a per-symbol distribution over win probability.

**Posterior.** Beta(α, β) conjugate update:
```
Prior:      α₀ = 2, β₀ = 3      (equivalent to "5 trades of mild pessimism")
Win:        α ← α + 1
Loss:       β ← β + 1
Mean:       E[P(win)] = α / (α + β)
```

Running tracking of `avg_win` and `avg_loss` via incremental means (one per observation).

**Kelly Criterion (v2.1 — default since April 2026).** Replaces the original EV-based heuristic with the mathematically-optimal Kelly fraction (Kelly 1956, Thorp 1969):
```
b = |avg_win| / |avg_loss|               (win/loss payoff ratio)
f* = (p·b - q) / b                        (raw Kelly fraction)
   = p - q/b
```

For a symbol with p=0.5, avg_win=5%, avg_loss=-4% → b=1.25, f*=0.1 (10% of bankroll full-Kelly).

**Fractional Kelly for safety.** Full Kelly maximizes expected log-wealth but has 50% drawdown expectation at the optimum. **¼-Kelly** is the institutional-standard conservative multiplier (reduces drawdown ~4× while giving up ~44% of return — dominant risk-adjusted tradeoff).

**Mapping Kelly to multiplier.**
```
f_scaled = f* · kelly_fraction            (default kelly_fraction = 0.25)
if f_scaled ≥ 0:
    raw_mult = 1 + (f_scaled / reference_kelly) · (max_mult - 1)
else:
    raw_mult = 1 + (f_scaled / reference_kelly) · (1 - min_mult)
raw_mult  = clip(raw_mult, min_mult, max_mult)
shrinkage = min(1, n / shrinkage_N)
final_mult = 1 + shrinkage · (raw_mult - 1)
final_mult = clip(final_mult, min_mult, max_mult)
```

`reference_kelly = 0.08` means a ¼-Kelly fraction of 0.08 (raw f* = 0.32) triggers the full `max_mult` boost. Shrinkage ensures small-N samples don't dominate sizing.

**Legacy "ev" method.** The original mult = 1 + EV/reference_ev heuristic is retained via `BAYESIAN_SIZING_METHOD='ev'` for rollback. The two methods give different rankings:

| Symbol | n | b=W/L | p | raw f* | Kelly mult (¼) | Legacy EV mult |
|--------|---|-------|---|--------|----------------|-----------------|
| SOFI | 7 | 3.85 | 0.33 | +0.160 | **1.26** | 1.53 |
| TSLA | 8 | 2.95 | 0.31 | +0.073 | 1.14 | **1.60** |
| SMCI | 3 | 1.22 | 0.50 | +0.090 | 1.06 | 1.23 |
| JPM | 10 | 1.55 | 0.33 | -0.098 | 0.82 | 0.59 |
| AAPL | 5 | 1.12 | 0.30 | -0.327 | 0.62 | 0.62 |

Kelly is more conservative on small-N high-payoff symbols (TSLA 1.14 vs 1.60) and less harsh on decent-W/L negative-edge symbols (JPM 0.82 vs 0.59). The Kelly version is defensibly optimal; the EV version was a heuristic.

**Validation example** (illustrative fit from an early snapshot — posterior evolves continuously):

| Symbol | n | E[P(win)] | avg_win | avg_loss | EV | Multiplier |
|--------|---|-----------|---------|----------|-----|------------|
| TSLA | 8 | 0.31 | +2.47% | -0.84% | +0.18% | **1.60×** |
| SOFI | 7 | 0.33 | +5.19% | -1.35% | +0.83% | **1.53×** |
| SMCI | 3 | 0.50 | +5.76% | -4.73% | +0.52% | 1.23× |
| NVDA | 2 | 0.29 | +1.00% | -0.86% | -0.33% | 0.85× |
| AMD | 4 | 0.44 | +0.68% | -1.53% | -0.55% | 0.70× |
| AAPL | 5 | 0.30 | +1.10% | -0.98% | -0.36% | 0.62× |
| JPM | 10 | 0.33 | +1.26% | -0.81% | -0.12% | 0.59× |
| PLTR | 0 | 0.40 (prior) | 1.00% | -0.80% | -0.08% | 1.00× (shrunk) |

This redistribution is exactly the intended behavior: symbols with high win-to-loss ratios (TSLA, SOFI) get boosted; symbols that consistently bleed (JPM, AAPL, AMD) get dampened; new/low-sample symbols stay neutral.

---

### 3.5 Cross-Sectional Momentum Gate (`strategy/cross_sectional.py`)

**Motivation.** A fixed-universe bot often holds yesterday's best performers while today's alpha is in other names. Full daily universe rotation would require PPO retraining (distribution shift). A cheaper alternative: re-rank the existing universe daily and bias toward winners.

**Score.** Per-symbol z-score composite:
```
score_s = 1.0 · z(5d_return_s)
        + 0.6 · z(20d_return_s - 60d_return_s)   [acceleration]
        + 0.4 · z(recent_volume / long_volume)   [volume momentum]
        + 0.8 · z(10d_drawdown_s)                [-drawdown penalty]
```

**Multiplier.**
```
if score ≥ top_tercile(scores): mult = 1.25     (boost)
elif score ≤ bottom_tercile:    mult = 0.50     (dampen)
elif |score| ≤ neutral_band:    mult = 1.00
else: linearly interpolate between 1.0 and [0.50 or 1.25]
```

**Verification (live data).** Today's ranking: `AMD:+1.82×1.25 | SMCI:+1.48×1.25 | NVDA:+0.85×1.25 | SOFI:+0.45×1.11 | AAPL:-0.26×0.97 | TSLA:-0.53×0.50 | JPM:-0.65×0.50 | PLTR:-3.14×0.50`. Matches intuition: strong 5-day performers boosted, underperformers dampened.

---

### 3.6 Correlation-Aware Sizing (`strategy/correlation_discount.py`)

**Motivation.** When AMD + NVDA + SMCI are all long at pairwise correlation 0.85, they act as effectively one 3× bet — not three independent bets. Risk Parity textbooks cover this but most retail bots ignore it.

**Formula.** For each symbol `s` with target weight `w_s`:
```
peers_s = {p ≠ s : sign(w_p) = sign(w_s) AND w_p ≠ 0}
avg_corr_s = mean{corr(s, p) for p in peers_s}  over 60-day daily returns
crowding_s = max(0, avg_corr_s - θ)             where θ = 0.5
discount_s = max(min_factor, 1 - strength · crowding_s)  where strength = 0.5, min_factor = 0.4
w_s ← w_s · discount_s
```

**Example.** At `avg_corr = 0.87`, `crowding = 0.37`, `discount = 1 - 0.5·0.37 = 0.81×`. Each of AMD/NVDA/SMCI independently gets ~0.82× in a tech-cluster long scenario. Floored at `0.4×` to prevent pathological over-suppression.

---

### 3.6b Regime-Conditional Exits (REX, `broker/alpaca.py::_rex_alignment_mults`)

**Motivation.** Today's observed failure mode: the bot used the same exit parameters (TP distance, trailing stop width) regardless of whether the market was trending or mean-reverting. Result: in trending regimes, the trailing stop triggered on normal trend-noise pullbacks, exiting winners early. In mean-reverting regimes, the TP was too far away, expecting continuation that reversed back into losses.

With S2's ensemble regime detector now correctly differentiating `trending_up`, `trending_down`, and `mean_reverting`, REX applies **alignment-aware multipliers** on top of the regime-base ATR multipliers:

| Scenario | TP Multiplier | Trail Multiplier | Rationale |
|---|---|---|---|
| Long in `trending_up` (aligned) | × 1.40 | × 1.25 | Let the trend run; distant TP captures more upside, wider trail survives noise |
| Short in `trending_up` (counter) | × 0.70 | × 0.75 | Counter-trend bet — take quick profits, accept tight stops |
| Short in `trending_down` (aligned) | × 1.40 | × 1.25 | Symmetric to long-in-uptrend |
| Long in `trending_down` (counter) | × 0.70 | × 0.75 | Symmetric to short-in-uptrend |
| Any direction in `mean_reverting` | × 0.85 | × 0.92 | Expect reversal → take profits fast, modest trail tighten |

These multiply onto the regime-base ATR multipliers (which already differentiate trending from MR in a coarser way). REX adds the **direction-alignment** dimension that the base multipliers lack.

**Wire-up.** All four call sites of `_get_trail_percent()` and `_get_tp_price()` pass the `direction` parameter — entry submission, reconcile reattach, orphaned-position reattach, sync path. `REX_ENABLED=False` returns (1.0, 1.0) neutral for rollback.

### 3.6c Liquidity-Scaled Sizing (LIQ, `strategy/liquidity_scaler.py`)

**Motivation.** Every symbol has an Average Daily (dollar) Volume (ADV). When position notional starts to rival ADV, market-impact slippage becomes non-negligible. Institutional rule of thumb: participation should stay below 1% of ADV to keep impact below a few basis points.

At current $30K equity all 8 symbols participate at <1bp of ADV — LIQ is dormant. But as equity grows or the universe picks up thinner names, impact risk escalates quadratically.

**Computation.** Average daily dollar volume over last 20 trading days from 15-min bars:
```
adv_s = mean(close_t · volume_t · 26) over last 20 days   (26 fifteen-min bars/day)
participation_s = |w_s| · equity / adv_s
```

**Scaling.**
```
warn = 0.001 (0.1% of ADV)     — below this: no action
hard = 0.01  (1.0% of ADV)     — above this: floor at min_mult

if participation ≤ warn:         mult = 1.0
elif participation ≥ hard:       mult = min_mult (default 0.3)
else: linear interpolation from 1.0 → min_mult
```

**Extended-hours tightening.** Pre-market (04:00-09:30 ET) and after-hours (16:00-20:00 ET) have 10-20× thinner liquidity than RTH. `LIQUIDITY_EH_FACTOR=5.0` tightens both thresholds by 5×, making the floor kick in at 0.2% of ADV instead of 1.0% during extended hours.

**Stress validation** (at $10M equity, stress scenario):
- SOFI (ADV $507M): 39bp participation → 0.77× mult
- SMCI (ADV $656M): 46bp participation → 0.72× mult
- During extended hours: both hard-floored at 0.30× (39bp > 20bp EH threshold)
- Mid/large-caps (NVDA, TSLA, AAPL at $3B-$12B ADV): unaffected

### 3.7 Asymmetric Trailing Stops (`broker/alpaca.py` + `broker/order_tracker.py`)

**Motivation.** Classic ratchet tightens stops only on profitable positions. But a position that is underwater *and* has never gone green is a clear loser — keeping it on a wide initial ATR-based stop just lets the loss grow. Asymmetric tightening cuts losses faster.

**MFE/MAE tracking.** On every ratchet tick:
```
unrealized_s = (current_price - entry_price) / entry_price · direction
group.max_favorable_pct ← max(group.max_favorable_pct, unrealized_s)
group.max_adverse_pct   ← min(group.max_adverse_pct,   unrealized_s)
group.original_trail_percent ← trail_percent  (captured once on first tick)
```

**Loss-tighten rule.**
```
if unrealized_s ≤ LOSS_THRESHOLD (default -0.007, i.e. -0.7%)
   AND MFE_s ≤ MFE_MAX (default 0.004, i.e. +0.4%)
   AND throttle_elapsed_ok:
       tight_trail = max(0.5%, original_trail · LOSS_FACTOR)    where LOSS_FACTOR = 0.55
       if tight_trail < current_trail:
           submit trailing-stop replace(trail = tight_trail)
```

The `MFE_MAX` gate ensures this doesn't fire on winning positions that pulled back — we only tighten positions that never went meaningfully green.

**Live verification examples.** Observed events in production:
- Losing position at -1.0% unrealized with zero MFE → trail tightens from ~2.4% to ~1.3%
- Position at -0.8% unrealized, MFE still 0 → trail tightens from ~2.65% to ~1.46%

Both positions had never gone green, both got tightened on schedule. Once a position's MFE exceeds `RATCHET_LOSS_TIGHTEN_MFE_MAX`, loss-tighten disables and the normal profit-tier ratchet takes over.

---

### 3.8 Intraday Risk Pacing (`strategy/risk.py::compute_risk_pacing_scale`)

**Motivation.** The existing `DAILY_LOSS_THRESHOLD = -3%` is a binary halt — fine when catastrophic, but provides no graduated response to modest intraday drawdowns. A better design throttles new risk as today's losses accumulate.

**Rules.**
```
daily_pnl% = (current_equity - daily_open_equity) / daily_open_equity

if daily_pnl% ≤ -1.5%:     pacing_scale = 0.2     (tier 2)
elif daily_pnl% ≤ -0.5%:   pacing_scale = 0.5     (tier 1)
else:                      pacing_scale = 1.0     (normal)

consecutive_losses_today = count(losses backward until first win)
if consecutive_losses_today ≥ 3:
    pacing_scale = min(pacing_scale, 0.3)

CVaR_risk_budget_today ← CVaR_risk_budget · pacing_scale
```

**Complementary to existing halt.** At -3% daily the bot already halts entirely via `check_pause_conditions`. Pacing fills in the -0.5% to -3% range with graduated response.

---

### 3.9 Anti-Earnings Filter (`strategy/earnings_filter.py`)

**Motivation.** Earnings announcements produce -5% to -20% single-name overnight gaps with WR historically crushed ~15-20pp on trades held through. The cheapest WR boost is simply not trading into earnings.

**Design.**
- Weekly yfinance sweep populates `(symbol → next_earnings_date)` map
- `is_in_blackout(symbol, pre_days=2, post_days=1)` returns True if `days_until_earnings ∈ [-1, 2]`
- `should_close_before_earnings(symbol, close_pre_days=1)` returns True if `days_until_earnings ∈ [0, 1]`
- Blackout: weight set to zero in gate cascade
- Pre-earnings auto-close: executed at the start of each trading loop iteration via `broker.close_position_safely()`

**Verification.** Apr 16 fetch: `TSLA 2026-04-22 (6 days)`, `SOFI 04-29`, `AAPL 04-30`, `PLTR 05-04`, `AMD/SMCI 05-05`, `NVDA 05-20`, `JPM 07-14`. TSLA auto-close scheduled for Apr 21; blackout window Apr 20-23.

---

### 3.10 Adverse-Selection Detector (`strategy/adverse_selection.py`)

**Motivation.** In market-microstructure literature, adverse selection (AVA) is the central execution-quality metric: when your fill is followed by a systematic move against you, you've been picked off. Every sell-side market maker tracks this to manage toxicity.

**Design.**
- On every entry fill: record `(symbol, side, fill_price, fill_time)`
- Sample current price at T+1, T+5, T+15, T+30 minutes
- Compute signed drift: `drift_s = (price_at_T+N - fill_price) / fill_price · side`
- Maintain rolling window of last 20 fills per symbol at primary offset (T+5 min)
- Toxicity score = mean signed drift (negative = toxic)

**Gate.**
```
if score ≥ threshold (default -0.002 = -20bp):     multiplier = 1.0
else:
    excess = |score - threshold|
    scale = min(1.0, excess / |threshold|)
    multiplier = max(1 - max_penalty, 1 - max_penalty · scale)   where max_penalty = 0.5
```

**Unit tested.** Toxic scenario (5 fills with consistent -0.4% drift) → score -0.0040 → multiplier 0.50. Clean scenario (fills with +0.24% drift) → multiplier 1.00.

---

### 3.11 Slippage Predictor (`strategy/slippage_predictor.py`)

**Motivation.** Every entry has a realized slippage captured in `group.slippage = |fill - limit| / limit`. Historically unused. Before each entry, we can predict expected slippage and skip trades where predicted slippage exceeds expected alpha.

**Design.** Grouped-mean predictor with fallback cascade:
```
(symbol, hour_bucket, size_bucket) → most specific
(symbol, hour_bucket)              → intermediate
(symbol,)                          → symbol-only
(None, hour_bucket, size_bucket)   → hour+size
(None, hour_bucket)                → hour-only
(None, None, None)                 → global mean
median across dataset              → last resort / prior
```

Hour buckets: `open (9-11)`, `mid (11-14)`, `close (14-16)`, `offhr`. Size buckets: `xs (<$1k)`, `s (<$5k)`, `m (<$15k)`, `l (>$15k)`.

**Grouped means** over linear regression: with N<100 per bucket, regression overfits the cross-terms; grouped means are robust and well-calibrated with small N.

**Veto.**
```
predicted_bps = group_mean(sym, hour, size)
threshold_bps = expected_alpha_bps · edge_safety_multiple    default 1.2×
if predicted_bps > threshold_bps:
    w_s ← w_s · SLIPPAGE_VETO_SCALE    default 0.3 (30% of original)
```

**Unit tested.** SMCI opening (30-60bp range) correctly predicted at 42.7bp; SMCI mid-day (8-20bp range) predicted at 13.9bp; JPM all-day (2-8bp range) predicted at 5bp. Veto fires as expected when alpha < 1.2× predicted slip.

---

### 3.12 PPO-Stacking Divergence Gate

**Motivation.** The portfolio PPO and the per-symbol LightGBM stacking ensemble are trained on different objectives over different horizons. When they strongly **disagree** on a symbol's direction, that disagreement is itself a signal. Historically such high-conviction divergences are lower-WR — dampening them is a net positive.

**Rule.**
```
meta_prob_s  = stacking_ensemble.predict(features_s)
meta_signed_s = (meta_prob - 0.5) · 2              ∈ [-1, +1]
ppo_direction = sign(target_weight_s)
disagree = (ppo_direction · meta_signed_s < 0)
           AND |target_weight_s| > MIN_WEIGHT      default 0.03
           AND |meta_signed_s| > MIN_META          default 0.20
if disagree:
    w_s ← w_s · 0.5
```

**Live verification.** Observed: `SOFI DIVERGENCE: PPO_dir=1 vs meta_signed=-0.55 — scaling ×0.5`. PPO wanted a weak long, stacking said strong bearish — dampened.

---

### 3.13 Sentiment Level + Velocity (`strategy/signals.py::get_sentiment_velocity`)

**Motivation.** The legacy sentiment layer used LLM-debate sentiment **level** as a multiplier. But level saturates (most symbols sit at 0.2-0.6 most of the time) and doesn't reflect new information arrival. A stock going from 0.1 → 0.6 has very different implications than one sitting at 0.6 all week.

**Level (existing):** Serialized 3-agent Ollama debate (bull, bear, analyst) over recent headlines; average ∈ [-1, +1]. Cached per `(symbol, date, hour)`.

**Velocity (new):**
```
velocity_s = current_level - mean(past_4_hour_levels)    ∈ [-2, +2]
velocity_factor_s = 1 + W_vel · velocity_s · sign(w_s)
velocity_factor_s = clip(velocity_factor_s, 1 - W_vel, 1 + W_vel)
w_s ← w_s · sentiment_factor_s · velocity_factor_s
```

Direction-aware multiplication: rising sentiment boosts longs and dampens shorts (and vice versa). This matches the "good news → prefer longs" intuition.

---

### 3.14 Gemini Self-Tuner (`gemini_tuner.py`)

**Motivation.** Manually tuning 100+ parameters is intractable. We let Gemini 2.5 Flash propose changes nightly based on full telemetry.

**Prompt structure (simplified):**
1. Portfolio state (equity, return, open positions)
2. Risk metrics (Sharpe, Sortino, drawdown, profit factor)
3. Trade stats (WR, avg win/loss, hold time)
4. Regime distribution
5. Per-symbol performance breakdown
6. PPO training scalars (entropy, value loss)
7. Previous tuning actions (anti-oscillation context)
8. 22 parameter groups with current values + hard bounds

**Output (strict JSON):**
```json
{
  "triage": "CRITICAL | DEGRADED | SUBOPTIMAL | HEALTHY",
  "diagnosis": "One sentence root-cause identification",
  "root_cause": "One sentence on market/system cause",
  "prescription": "2-3 sentences linking diagnosis to param changes",
  "proposed_universe": [...],
  "parameters": {"PARAM_NAME": new_value_or_null, ...}
}
```

**Safety clamping.** Every proposed value flows through:
1. Category-specific step cap (risk ±20%, ATR ±25%, PPO ±15%, new safeguards ±25%)
2. Absolute hard bounds from `ABSOLUTE_BOUNDS` table (Pydantic-validated)
3. Atomic staged update — revert on any validation error

**22 groups, ~60 tunable parameters,** covering: risk budgets, trailing stops, ratchet timing, signal confidence, PPO hyperparameters, reward shaping, regime detection, intraday pacing, asymmetric trailing, meta-filter threshold, cross-sectional weights, anti-earnings windows, sentiment velocity, crowding discount, portfolio meta-blend weight, adverse selection, Bayesian sizing bounds, slippage veto, PPO-stacking divergence.

**Audit trail.** Every change logged as `{"event": "gemini_tuning_batch", "timestamp": ..., "changes_count": N, "parameters_changed": [...], "pnl_context": {...}}` — grep-ready JSON.

---

## 4. State Persistence

All learned state round-trips across restarts via atomic file writes (tempfile + `os.rename`):

| File | Content | Format |
|---|---|---|
| `order_tracker.json` | Active positions + MFE/MAE + original_trail_percent | JSON |
| `regime_cache.json` | Per-symbol regime + persistence + `__version__` | JSON (versioned) |
| `causal_cache_portfolio.pkl` | GES graph + action-reward correlation + model hash | pickle |
| `replay_buffer_portfolio.pkl` | (obs, action, reward) transitions for causal | pickle |
| `meta_filter.pkl` | LightGBM booster + symbol_map + training stats | pickle (schema-versioned) |
| `bayesian_sizing.pkl` | Per-symbol (α, β, avg_win, avg_loss, n) | pickle |
| `slippage_predictor.pkl` | Sample list + global median | pickle |
| `adverse_selection.pkl` | Pending fills + rolled drift deques | pickle |
| `earnings_cache.pkl` | Per-symbol next earnings date + fetch time | pickle |
| `live_signal_history.json` | All historical trades with obs, size, realized_return | JSON |
| `last_entry_times.json` | Min-hold enforcement timestamps | JSON |
| `starting_equity.json` | Session-start equity for drawdown tracking | JSON |
| `dynamic_config.json` | Gemini-tuned parameter deltas | JSON |
| `tuning_history.json` | Last N tuning batches for anti-oscillation | JSON |

---

## 5. Operational Characteristics

### 5.1 Startup sequence (6-7 minutes cold)

1. Logging + Redis + ArcticDB init (~1s)
2. OrderTracker load + position reconciliation (~1s)
3. Broker init + earnings cache load (~1s)
4. Signal generator init (causal managers per symbol + portfolio) (~3s)
5. TFT encoder weight load per symbol (5-7 min on first run; warm-start from `ppo_checkpoints/tft_cache/` on subsequent)
6. Stacking ensemble train per symbol (20 models × 8 symbols, ~20s each) (~3 min on first run)
7. Portfolio PPO model load from `ppo_checkpoints/portfolio/ppo_model.zip`
8. Causal graph rebuild (from cache, ~1s; from scratch 15s with column cap)
9. Meta-filter fit from live_signal_history (~1s)
10. Bayesian sizer fit from history (~instantaneous)
11. Earnings calendar refresh (weekly TTL, typically cache hit)
12. PortfolioEnv construction + causal bootstrap if needed (~30s if bootstrap fires)

### 5.2 Trading loop (~60 seconds per cycle)

1. Get equity + today's daily open → pacing scale (~1ms)
2. Pre-earnings auto-close check for open positions (~500ms API call × N)
3. Fetch data_dict for all symbols (~200ms cache hit, ~2s cache miss)
4. PortfolioEnv.step() + PPO predict (~1s)
5. Stacking meta-blend per symbol (~500ms total for 8 symbols)
6. Causal penalty application (~10ms, no DoWhy)
7. Bayesian sizing (~1ms)
8. Cross-sectional scoring (~50ms)
9. Crowding discount (~10ms)
10. Sentiment level + velocity (~25s first hour, ~10ms cached thereafter)
11. Gate cascade × 16 layers (~50ms)
12. CVaR position sizing with pacing (~300ms, cvxpy)
13. Causal final scaling (~10ms)
14. Per-symbol entry/close order submission (~200ms × N changed positions)

### 5.3 Fill handler (websocket-driven, async)

- Entry fill: measure slippage → record (B2 AVA) → record (ESP) → submit trailing stop + TP
- Exit fill: cancel OCO opposite leg → push reward to causal buffer → update Bayesian posterior → update adaptive memory → clear tracker group

### 5.4 Background threads / tasks

- **Data stream**: `asyncio.create_task(data_stream_task())`
- **Trading loop**: `asyncio.create_task(self.trading_loop())`
- **Monitor loop**: `asyncio.create_task(self.broker.monitor_positions())`
- **Universe rotation**: Friday 8pm ET
- **Gemini tuner + causal refresh + meta-filter refit + earnings refresh**: 03:30 AM ET daily
- **Regime precompute**: 04:00 AM ET daily
- **PPO retrain**: 18:00 ET daily
- **Background regime precompute thread**: on startup, daemon thread

---

## 6. Known Gaps & Future Work

### 6.1 High-priority (pending)
- **A4 — Online PPO fine-tuning.** 50 gradient steps per day at 17:00 ET with LR=1e-5. Addresses the "windows 5-6 rejected every retrain" pattern. Medium risk (catastrophic forgetting), needs checksum-based rollback guard.
- **B1 — Learned exit policy.** Separate small PPO for close-timing decisions. Uses MFE/MAE from S3. Needs ~30 more closed trades with MFE/MAE tagging before training is viable.
- **B3 — Transformer-based entry timer.** 1-min bar classifier to defer entries in adverse microstructure. Requires 1-min bar feed in `data/ingestion.py`.

### 6.2 Research-quality experiments (creative)
- **A2 — Cross-sectional PPO reward.** Reshape reward from portfolio return to `(long_basket − short_basket) − turnover`. Forces pair-trade thinking. Requires full retrain; unknown convergence.
- **Regime-conditional meta-labeler.** Separate LightGBM per regime (trending_up, trending_down, mean_reverting). Better targeting but requires more data per regime.
- **Self-supervised embeddings on TFT encoder.** Cluster historical wins vs losses in the TFT latent space; flag trades whose current embedding is close to past winners.
- **Anti-correlation pair discovery.** Automatically identify pairs with low historical correlation; size long/short legs to be genuinely market-neutral.
- **Newscatcher-augmented sentiment.** Add StockTwits/WSB/AlphaVantage as secondary sources; cross-source disagreement = signal.

### 6.3 Operational
- **News API rate limit (100/day on free tier)** causes sentiment fallback to 0.0 mid-day. Mitigations: upgrade plan, add secondary source, cache longer.
- **Regime cache versioning** currently manual (`__version__=2`). Could be auto-bumped on detect_regime module-hash change.

---

## 7. Acknowledgments & References

**Research foundations:**
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. — meta-labeling framework
- Parisotto, E., et al. (2020). *Stabilizing Transformers for Reinforcement Learning.* ICML. — GTrXL architecture
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms.* — PPO
- Chickering, D.M. (2002). *Optimal Structure Identification With Greedy Search.* JMLR. — GES
- Sharma, A., et al. (2020). *DoWhy: An End-to-End Library for Causal Inference.* — DoWhy
- Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices.* — Ledoit-Wolf shrinkage

**Infrastructure:**
- Alpaca Markets (paper trading API)
- Gemini 2.5 Flash (Google) for nightly hyperparameter tuning
- Ollama (sentiment-70b, llama3.1:8b) for LLM sentiment debate
- stable-baselines3 / sb3_contrib (RecurrentPPO)
- ArcticDB (Man Group) for time-series cache
- pgmpy (GES implementation)
- LightGBM (stacking ensemble + meta-filter)
- cvxpy (CVaR optimization)

---

*Document last revised: 2026-04-16*

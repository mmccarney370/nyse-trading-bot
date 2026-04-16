# NYSE Trading Bot

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Status](https://img.shields.io/badge/Status-Live_Paper_Trading-green)
![RL](https://img.shields.io/badge/RL-GTrXL_Recurrent_PPO-purple)
![Causal](https://img.shields.io/badge/Causal_AI-GES_%2B_DoWhy-orange)
![Meta-Label](https://img.shields.io/badge/Meta--Label-Lopez_de_Prado-9cf)
![Bayesian](https://img.shields.io/badge/Bayesian-Per--Symbol_Sizing-ff69b4)
![Self-Tuning](https://img.shields.io/badge/Self--Tuning-Gemini_2.5_Flash-red)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

> A self-improving algorithmic trading system that thinks about **why** it should trade, sizes bets from **what it's actually learned per symbol**, detects when it's being **picked off**, **refuses** entries that would eat their own alpha, and rewrites **100+ of its own parameters** every night — all while running unsupervised against a live broker.

This isn't a moving-average crossover with extra steps. It's a production-grade stack that fuses **deep reinforcement learning**, **causal inference**, **Bayesian posteriors**, **Lopez-de-Prado meta-labeling**, **ensemble regime detection**, **adverse-selection detection**, and **LLM-powered nightly self-tuning** into a single autonomous trader.

---

## What Makes This Bot Unique

Nine capabilities you **won't find together** in any open-source trading framework. Each one individually takes a research paper to justify; having all sixteen layers stack multiplicatively is what separates signal from noise.

### 🧠 1. It Asks "Is This Actually Causal?" Before Every Trade
Most bots chase correlations. This one runs **GES causal graph discovery** (Greedy Equivalence Search, pgmpy) over 56 features + action + reward, then asks DoWhy whether there's a genuine directed path from `action → reward` for the current market state. Spurious signals get dampened. **23 causal edges discovered** on the current universe, firing on every cycle — you can literally read which features are causally linked to P&L in `causal_cache_portfolio.pkl`.

Every cycle logs `✅ [CAUSAL PENALTY APPLIED in generate_portfolio_actions]` or falls back cleanly to neutral.

### 🎯 2. Lopez-de-Prado Meta-Labeling Filter
After the PPO emits a trade, a **nightly-trained LightGBM classifier** answers a different question: *"given that PPO says go long on AAPL right now, is this actually going to win?"* Probability below `META_FILTER_MIN_PROB` and the trade is **rejected outright**. This is the two-model architecture from *Advances in Financial Machine Learning* (López de Prado, 2018) — the primary model predicts direction, the secondary decides whether to place the bet.

Currently trained on 39 closed trades (Brier score 0.088, well-calibrated). Self-refits nightly at 03:30 ET as the trade log grows.

### 💰 3. Bayesian Per-Symbol Sizing — Not Flat Allocation
Every symbol gets its own **Beta(α, β) posterior on P(win)**, updated after every single closed trade. Winners get scaled up to **1.6×**; bleeders get dampened to **0.4×**. Small-sample shrinkage blends toward 1.0 so two lucky trades don't suddenly triple a position. This directly attacks the "one stock carries the book" failure mode.

Live example right now:
```
TSLA 1.60×  SOFI 1.53×  SMCI 1.23×  NVDA 0.85×  AMD 0.62×  AAPL 0.62×  JPM 0.59×
```
Capital flows automatically to symbols where the posterior shows real edge.

### 📡 4. Five-Voter Ensemble Regime Detection (HMM Isn't Enough)
HMM alone kept labeling every single bar "mean_reverting" at 95%+ persistence — even during clear multi-week rallies. This bot ships an ensemble of **slope + ADX + lag-1 autocorrelation + ATR range-expansion + HMM**, aggregated by direction-consistency voting. The result: `trending_up`, `trending_down`, or `mean_reverting` with a *differentiated* persistence score (0.65–0.90 live, not stuck at ceiling).

Regime flows into *everything* downstream: trailing stop widths, take-profit distances, ratchet cadence, risk budgets, gate multipliers, and the PPO reward function.

### 🎯 5. Cross-Sectional Momentum Gate — Be Where Today's Alpha Is
Every cycle, the bot z-scores all 8 symbols on a composite of (5-day return, 20-vs-60-day acceleration, volume momentum, drawdown severity). Top-tercile scorers get their PPO-emitted weights **boosted 1.25×**; bottom-tercile get **dampened 0.50×**. You see lines like this every 60 seconds:

```
[CS-MOMENTUM] AMD:+1.82×1.25 | SMCI:+1.48×1.25 | NVDA:+0.85×1.25 | PLTR:-3.14×0.50
```

No retrain required — preserves PPO observation shape.

### 🛡️ 6. Adverse-Selection Detection (The Market Maker's Weapon)
Every entry fill is timestamped and price-sampled at T+1, T+5, T+15, T+30 minutes. The bot computes **signed post-fill drift** per symbol and rolls the last 20 fills. When recent fills on a symbol are consistently followed by adverse moves (< -20bp drift), that symbol is **toxic** — it gets dampened up to 50%. This is the exact technique sell-side market makers use to manage information asymmetry. Almost no retail bot tracks it.

### 🚨 7. Slippage Prediction Veto — Refuses to Trade Into Chop
Grouped-mean slippage model over `(symbol, hour-bucket, size-bucket)` learns from real fills. Before every entry, it predicts expected slippage in basis points. If that prediction **exceeds 1.2× expected alpha**, the trade is skipped or dampened. We refuse to pay more in execution cost than we expect to win.

Opening-hour SMCI? Predicted 42bp slippage. JPM mid-day at normal size? Predicted 5bp. The bot *knows* when market conditions make an entry unprofitable before placing it.

### ⚖️ 8. Asymmetric Trailing Stops (Cut Losers, Let Winners Run)
Every position tracks its **Maximum Favorable Excursion (MFE)** and **Maximum Adverse Excursion (MAE)** in real time. The profit-side ratchet tightens as unrealized P&L climbs (classic). But if a position goes underwater **and never went meaningfully green** (MFE ≤ 0.4%), the trail tightens to **55% of its original ATR-based width** — cutting dead losers before they become real ones. Two loss-tightens fired today live: PLTR (trail 2.42% → 1.33%) and SMCI (2.65% → 1.46%).

### 🤖 9. Gemini 2.5 Flash Tunes 100+ Parameters Every Night
At 03:30 AM ET, the bot serializes its full telemetry (Sharpe, Sortino, drawdown, WR, per-symbol P&L, PPO training scalars, regime distribution) and sends it to **Gemini 2.5 Flash** with 19 parameter groups and hard bounds. Gemini proposes changes inside a strict JSON schema. Every value is Pydantic-validated, category-clamped (PPO ±15%, risk ±20%, reward shaping ±25%), anti-oscillation-checked against recent history, and logged as structured JSON.

Every parameter — from `TRAILING_STOP_ATR_TRENDING` to `META_FILTER_MIN_PROB` to `CROWDING_DISCOUNT_STRENGTH` — is eligible for autonomous tuning. The bot genuinely writes its own configuration.

---

## And That's Just the Headline Layers

Below those nine, sixteen more multiplicative gates stack before any order hits the wire:

| Layer | What It Does |
|---|---|
| **PPO-Stacking Divergence** | High-conviction disagreement between the transformer and the LightGBM ensemble → ×0.5 |
| **Crowding Discount** | Correlated same-sign positions (AMD+NVDA+SMCI all long at 0.85 corr) get individually dampened — prevents effective mega-bets |
| **Anti-Earnings Blackout** | Auto-close 1 day before earnings + zero new entries ±2/1 days (TSLA currently scheduled for Apr 21 auto-close) |
| **Intraday Risk Pacing** | Down 0.5% today → CVaR budget × 0.5. Down 1.5% → × 0.2. Three consecutive losses → × 0.3. |
| **Sentiment Level + Velocity** | LLM-debate level (bull/bear/analyst via Ollama 70B) **plus** 4-hour Δsentiment captures information arrival, not just steady state |
| **Stacking Ensemble Overlay** | 160 LightGBM models (20 per symbol, 8-bar forward labels) provide meta-probability overlay |
| **VIX / SPX Breadth Gates** | Longs dampened in bear regimes, shorts in bull regimes |
| **1-Hour Trend Alignment** | 15-min signals must agree with 1-hour trend or get halved |
| **Anti-Churn Gate** | Repeated stops in same direction decay confidence exponentially (0.5ⁿ) — prevents death spiral |
| **Defensive Panic Mode** | 3+ correlated stop-outs in 5 minutes → 3-hour lockout on all new entries |
| **Min Confidence Floor** | Final rejection of anything below the threshold |
| **Equity-Curve Trader** | Treats the bot's own P&L as a time series — sizes down in drawdowns, up in winning streaks |
| **CVaR Portfolio Optimization** | Ledoit-Wolf covariance shrinkage + conviction-weighted allocation |
| **Notional Caps** | Hard safety: never more than 30% of equity in one name |
| **Buying-Power Clamp** | Real-time Alpaca buying-power check |
| **Regime-Scaled ATR Stops** | Trailing stop width changes with detected regime — trending = tight, mean-reverting = wide |

---

## How the 25+ Layers Actually Feel in Logs

Every cycle you see things like this scroll by:

```
[REGIME ENSEMBLE+HMM] SOFI | regime=trending_up | persistence=0.802
[BAYESIAN-SIZE] SOFI:1.53 | TSLA:1.60 | JPM:0.59 | AAPL:0.62
[CS-MOMENTUM] AMD:+1.82×1.25 | PLTR:-3.14×0.50
SOFI sentiment blend: raw=0.433 → factor=1.07 | velocity_factor=0.973
SOFI DIVERGENCE: PPO_dir=1 vs meta_signed=-0.55 — scaling ×0.5
NVDA META-FILTER: P(win)=0.329 < 0.33 — REJECTED
[RATCHET LOSS-TIGHTEN] SMCI unrealized -0.78% (MFE +0.00%) — tightening 2.65% → 1.46%
✅ [CAUSAL PENALTY APPLIED in generate_portfolio_actions] portfolio — causal wrapper used
[PORTFOLIO GATE] JPM: weight -0.1069 → -0.0534 (gates: 1H_TREND(1))
[CVaR REGIME] Using Gemini-tuned risk budget: $535 | regime=mean_reverting
```

Every decision is traceable. Every subsystem has a searchable log prefix. If something feels off, you can `grep` your way to exactly which layer did it.

---

## It Runs Like a Production System

This isn't a notebook experiment. It connects to a **real Alpaca broker**, streams live bars via WebSocket, places real limit orders with trailing stops and take-profits, handles fill events via an async state machine, persists every piece of learned state atomically across restarts, and self-heals from websocket disconnects. 24/7 uptime in paper trading mode.

All state persists across crashes: `meta_filter.pkl`, `bayesian_sizing.pkl`, `slippage_predictor.pkl`, `adverse_selection.pkl`, `earnings_cache.pkl`, `causal_cache_portfolio.pkl`, `replay_buffer_portfolio.pkl`, `order_tracker.json`, `regime_cache.json` (schema-versioned), `live_signal_history.json`. Every learned quantity survives a restart and keeps growing.

---

## Nightly Self-Improvement Cycle

While you're asleep, seven things happen on schedule:

| Time (ET) | What Runs |
|---|---|
| **03:30 AM** | Full telemetry → Gemini 2.5 Flash → parameter adjustments → Pydantic validation → atomic apply + structured audit log |
| **03:30 AM** | GES causal graph rebuild with latest feature data + replay buffer |
| **03:30 AM** | Meta-label classifier retrained on updated live_signal_history |
| **03:30 AM** | yfinance earnings calendar sweep (weekly TTL, uses cache when fresh) |
| **04:00 AM** | Parallel ensemble regime detection refresh across all symbols |
| **06:00 PM** | Online PPO retrain (~75K timesteps incremental update) |
| **Friday 8 PM** | Universe rotation — evaluate 34 candidates, swap losers, full retrain on new roster |

---

## Core Capabilities (Reference)

| Capability | Implementation |
|---|---|
| **Portfolio Optimization** | CVaR with Ledoit-Wolf shrinkage, regime-scaled budgets, intraday pacing, Bayesian per-symbol sizing, buying-power enforcement |
| **Signal Blend** | 8-layer stack: Portfolio PPO → stacking ensemble → causal penalty → Bayesian sizing → cross-sectional momentum → crowding discount → sentiment (level + velocity) → gate cascade |
| **56 Input Features** | Bollinger Bands, RSI, MACD, ATR, CCI, Stochastic, OBV z-score, Chaikin flow, VWAP deviation, volume imbalance, divergence detection, SMA(50/200) + golden cross, hourly-refreshed macro (VIX/yield curve/SPX), TFT encoder embeddings, regime flags |
| **Meta-Labeling Filter** | Lopez-de-Prado-style post-PPO binary classifier (LightGBM); nightly refit |
| **Bayesian Sizing** | Beta(α, β) posterior per symbol; sizing multiplier ∈ [0.4×, 1.6×] with small-sample shrinkage |
| **Ensemble Regime Detection** | 5-voter (slope + ADX + autocorr + range + HMM); direction-consistency aggregation |
| **Cross-Sectional Momentum** | Daily z-score ranking; top tercile × 1.25, bottom × 0.50 |
| **Correlation-Aware Sizing** | 60-day return correlations → same-sign cluster discount |
| **Adverse-Selection Detector** | Rolling 20-fill post-fill drift tracker; toxic symbols dampened up to 50% |
| **Slippage Veto** | Grouped-mean predictor over (symbol, hour, size); veto when predicted > 1.2× alpha |
| **PPO-Stacking Divergence Gate** | Dampens to × 0.5 on high-conviction disagreement |
| **Asymmetric Trailing Stops** | MFE/MAE-gated loss-side tightening (trail → 55% of original ATR width) |
| **Intraday Risk Pacing** | Graduated CVaR throttle: -0.5% → × 0.5, -1.5% → × 0.2, 3 losses → × 0.3 |
| **Anti-Earnings Filter** | yfinance calendar; auto-close 1 day pre-earnings; blackout ±2/1 days |
| **Sentiment: Level + Velocity** | Serialized 3-agent LLM debate (Ollama 70B + 8B fallback); Δsentiment over 4h |
| **Causal RL Overlay** | GES graph + noise-injected bootstrap + column-cap for tractability |
| **Adaptive Exits** | Alpaca native trailing stops + software take-profit + profit-tier ratchet + loss-tighten |
| **Walk-Forward Validation** | 6-window walk-forward threshold optimization with IS/OOS gap monitoring and overfitting rejection |
| **Adaptive Memory** | Real-time anti-churn, cross-asset panic detection, online Bayesian signal-component reweighting |
| **Self-Tuning** | Gemini 2.5 Flash with 19 parameter groups, Pydantic-validated bounds, anti-oscillation history |
| **Dynamic Universe Rotation** | Weekly evaluation of 34 candidate symbols; auto retrain on roster changes |
| **State Persistence** | Atomic writes for every learned quantity; schema-versioned where it matters |
| **Structured Observability** | Searchable log prefixes for every subsystem + TensorBoard for PPO curves |

---

## Architecture

```
trading_bot/
|
|-- bot.py                          # Async event loop + orchestration
|-- config.py                       # 100+ settings, Pydantic-validated
|-- __main__.py                     # Entry point
|-- backtest.py                     # Full backtesting engine
|-- gemini_tuner.py                 # Gemini 2.5 Flash (19 parameter groups)
|
|-- broker/
|   |-- alpaca.py                   # Orders + ratchet + asymmetric loss-tighten + reattach qty-drift fix
|   |-- stream.py                   # WebSocket fills + AVA + slippage + Bayesian posterior update
|   |-- order_tracker.py            # State machine with MFE/MAE per-position tracking
|
|-- strategy/
|   |-- signals.py                  # Signal blend + 16-layer gate cascade
|   |-- regime.py                   # 5-voter ensemble (slope + ADX + autocorr + range + HMM)
|   |-- risk.py                     # CVaR + intraday pacing + Kelly sizing
|   |-- portfolio_rebalancer.py     # Portfolio-level rebalance + causal final scaling
|   |-- universe.py                 # Weekly universe rotation
|   |-- meta_filter.py              # Lopez-de-Prado P(win) classifier
|   |-- bayesian_sizing.py          # Per-symbol Beta posteriors
|   |-- cross_sectional.py          # Daily z-score momentum ranker
|   |-- correlation_discount.py     # Crowding-aware position sizing
|   |-- earnings_filter.py          # Anti-earnings blackout + auto-close
|   |-- slippage_predictor.py       # Grouped-mean slippage predictor
|   |-- adverse_selection.py        # Post-fill toxicity detector
|   |-- cognitive.py                # Equity curve trader + profit velocity + autopsy
|
|-- models/
|   |-- trainer.py                  # PPO/RecurrentPPO training
|   |-- env.py                      # Per-symbol Gym env
|   |-- portfolio_env.py            # Multi-asset portfolio Gym env
|   |-- features.py                 # 56-feature generator (hourly macro refresh)
|   |-- policies.py                 # Custom AuxGTrXL policy (Parisotto 2020 + aux vol head)
|   |-- stacking_ensemble.py        # 20-model LightGBM bootstrap ensemble
|   |-- causal_signal_manager.py    # GES + noise-injected bootstrap + column cap
|   |-- causal_rl_manager.py        # Causal buffer persistence
|   |-- bot_initializer.py          # Startup orchestration
|   |-- ppo_utils.py                # PPO save/load
|
|-- data/
|   |-- handler.py                  # ArcticDB + Redis + Polygon/Alpaca fetch
|   |-- ingestion.py                # Real-time stream + backfill
|
|-- utils/
|   |-- helpers.py                  # Market hours, holiday calendar
|   |-- local_llm.py                # Serialized Ollama multi-agent debate
|   |-- log_setup.py                # Logging configuration
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended for PPO + TFT)
- [Alpaca](https://alpaca.markets/) account (paper trading keys work fine)
- [Redis](https://redis.io/) (caching)
- [Ollama](https://ollama.ai/) with `sentiment-70b` and `llama3.1:8b` (sentiment)
- [Gemini API key](https://ai.google.dev/) (optional, for self-tuning)
- [NewsAPI key](https://newsapi.org/) (optional, for news sentiment)

### Setup

```bash
git clone https://github.com/mmccarney370/nyse-trading-bot.git
cd nyse-trading-bot

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`.env`:
```env
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
GEMINI_API_KEY=your_gemini_key       # optional
NEWS_API_KEY=your_newsapi_key        # optional
```

Ollama models:
```bash
ollama pull llama3.1:8b
ollama pull sentiment-70b   # optional; 8B is used as fallback
```

### Running

```bash
# Live paper trading
python __main__.py

# Backtest
python run_backtest.py --start 2025-01-01 --end 2025-12-31
```

First startup takes **5–6 minutes** (TFT precompute, stacking ensemble training, causal bootstrap, meta-filter fit, earnings calendar refresh). Subsequent starts are faster with cached models and state files.

Logs: `logs/nyse_bot.log` + `nyse_bot.log` (root) | Training curves: `tensorboard --logdir ppo_tensorboard`

---

## Configuration

Everything lives in `config.py` with Pydantic validation. **100+ tunable parameters** across 19 groups — every one of them eligible for autonomous nightly tuning by Gemini inside clamped safety bounds.

| Group | Example Parameters |
|---|---|
| **Risk** | `RISK_PER_TRADE_*`, `MAX_LEVERAGE`, `KELLY_FRACTION`, `MAX_POSITIONS` |
| **Intraday Pacing** | `RISK_PACING_TIER1_LOSS`, `TIER2_LOSS`, `CONSECUTIVE_LOSSES` |
| **Regime Ensemble** | `REGIME_SLOPE_THRESHOLD`, `REGIME_ADX_*`, `REGIME_AUTOCORR_*`, `REGIME_RANGE_*` |
| **Execution** | `TRAILING_STOP_ATR_*`, `TAKE_PROFIT_ATR_*`, `RATCHET_*` |
| **Asymmetric Trailing** | `RATCHET_LOSS_TIGHTEN_THRESHOLD`, `_FACTOR`, `_MFE_MAX` |
| **PPO** | `PPO_LEARNING_RATE`, `PPO_ENTROPY_COEFF`, `GTRXL_*`, `SORTINO_WEIGHT` |
| **Signal Blend** | `PORTFOLIO_META_WEIGHT`, `SENTIMENT_WEIGHT`, `SENTIMENT_VELOCITY_WEIGHT`, `MIN_CONFIDENCE` |
| **Meta-Filter** | `META_FILTER_MIN_PROB`, `META_FILTER_MIN_TRAIN` |
| **Bayesian Sizing** | `BAYESIAN_SIZING_MIN_MULT`, `_MAX_MULT`, `_REFERENCE_EV`, `_SHRINKAGE_N` |
| **Cross-Sectional** | `CROSS_SECTIONAL_WEIGHT`, `_MAX_MULT`, `_MIN_MULT` |
| **Crowding** | `CROWDING_DISCOUNT_THRESHOLD`, `_STRENGTH`, `_MIN_FACTOR` |
| **Earnings** | `EARNINGS_BLACKOUT_PRE_DAYS`, `_POST_DAYS`, `_CLOSE_PRE_DAYS` |
| **Slippage** | `SLIPPAGE_VETO_MULTIPLE`, `_SCALE` |
| **Divergence** | `DIVERGENCE_GATE_SCALE`, `_MIN_WEIGHT`, `_MIN_META` |
| **Adverse Selection** | `ADVERSE_SELECTION_THRESHOLD`, `_MAX_PENALTY` |
| **Causal** | `CAUSAL_PENALTY_WEIGHT`, `CAUSAL_MAX_FEATURES`, `CAUSAL_BOOTSTRAP_*` |
| **Reward Shaping** | `DD_PENALTY_COEF`, `VOL_PENALTY_COEF`, `TURNOVER_COST_MULT`, `SORTINO_WEIGHT` |
| **Universe** | `UNIVERSE_CANDIDATES`, `MAX_UNIVERSE_SIZE` |
| **Tuning** | Gemini schedule, safety bounds |

---

## Monitoring

Every subsystem logs with a searchable ripgrep-friendly prefix:

| Prefix | Subsystem |
|---|---|
| `[CVaR]` / `[RISK PACING]` | Position sizing + intraday throttling |
| `[REGIME ENSEMBLE+HMM]` / `[REGIME CACHE]` | 5-voter regime detection |
| `[CAUSAL BUILD SUCCESS]` / `[CAUSAL PENALTY APPLIED]` / `[CAUSAL BOOTSTRAP]` | Causal graph lifecycle |
| `[TRACKER]` / `[RATCHET]` / `[RATCHET LOSS-TIGHTEN]` | Order state + trailing stops |
| `[META-FILTER]` | Meta-label classifier fit + gate decisions |
| `[BAYESIAN-SIZE]` | Per-symbol posterior + multiplier |
| `[CS-MOMENTUM]` | Cross-sectional daily ranking |
| `[CROWDING-DISCOUNT]` | Correlation-aware sizing |
| `[EARNINGS]` / `[EARNINGS BLACKOUT]` / `[EARNINGS AUTO-CLOSE]` | Anti-earnings events |
| `[ADVERSE-SEL]` / `SLIPPAGE-VETO` | Execution quality gates |
| `[DIVERGENCE]` | PPO-stacking disagreement |
| `[STREAM]` / `[TP HIT]` / `[STOP HIT]` | Fills + OCO logic |
| `[PORTFOLIO GATE]` | Multi-layer confidence cascade |
| `[CHURN]` / `[PANIC DETECT]` / `[DEFENSIVE]` | Adaptive memory state |
| `[GEMINI]` | Nightly self-tuning events |

TensorBoard at `tensorboard --logdir ppo_tensorboard` shows reward curves, clip fraction, explained variance, policy entropy.

`order_tracker.json` holds live position state with MFE/MAE per symbol — watch it in real time.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "Insufficient features for stacking" | Not enough bars | Wait for cache fill (~300 bars) |
| "CVaR fell back to parity" | Sparse overlapping returns | Need 50+ bars per symbol |
| "HMM ensemble failed" | Flat data | Normal — ensemble's 4 other voters still fire |
| `GES discovered 0 edges` | Deterministic PPO | Fixed v2.0 — bootstrap injects `CAUSAL_BOOTSTRAP_NOISE_SIGMA=0.4` |
| Meta-filter rejecting everything | Threshold > model's mean pred | Lower `META_FILTER_MIN_PROB` (default 0.33) |
| BPS mult stuck at 1.0 | < `SHRINKAGE_N` closed trades | Normal — shrinks to neutral on small samples |
| News API `rateLimited` | NewsAPI free tier = 100/day | Sentiment falls back to 0.0; upgrade tier for more |
| `[PANIC DETECT]` triggering | 3+ symbols stopped in 5 min | Working as intended — defensive mode during crashes |
| Alpaca websocket reconnects | Stream blip | Auto-reconnects |
| `insufficient qty available` on reattach | Tracker/Alpaca qty drift | Fixed v2.0 — reattach re-queries live qty |
| TFT features all zero | Cache rebuild needed | Delete `ppo_checkpoints/tft_cache/` |

---

## Documentation

See [**`WHITEPAPER.md`**](./WHITEPAPER.md) for the detailed technical design: mathematical formulations, architectural rationale, algorithmic choices, and research-paper citations behind every subsystem.

---

## Disclaimer

This is a personal project built for learning and paper trading. It is not financial advice. Markets are unpredictable, models overfit, and past performance means nothing about the future. If you choose to run this with real money, that's entirely your decision and your risk. Start with paper trading and stay there until you deeply understand every component.

---

## License

MIT License — Copyright (c) 2026 Matthew McCartney

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

> *"You have power over your mind, not outside events. Realize this, and you will find strength."*
> — Marcus Aurelius

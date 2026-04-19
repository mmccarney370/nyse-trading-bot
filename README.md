# NYSE Trading Bot

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Status](https://img.shields.io/badge/Status-Live_Paper_Trading-green)
![RL](https://img.shields.io/badge/RL-GTrXL_Recurrent_PPO-purple)
![Causal](https://img.shields.io/badge/Causal_AI-GES_%2B_DoWhy-orange)
![Self-Tuning](https://img.shields.io/badge/Self--Tuning-Gemini_2.5_Flash-red)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

A fully autonomous trading system that combines deep reinforcement learning with causal inference, Bayesian statistics, and LLM-powered self-tuning — running unsupervised against a live broker around the clock. It doesn't just predict where prices are going. It asks whether the relationship between its actions and rewards is genuinely causal, sizes each bet by the Kelly-optimal fraction of its per-symbol posterior, adapts its exit strategy to the current market regime, detects when it's being picked off by better-informed counterparties, refuses entries where predicted slippage would eat the alpha, and rewrites its own configuration every night based on what actually worked.

This is not a moving-average crossover with extra steps. Thirty-one upgrades across fourteen strategy modules stack multiplicatively before any order hits the wire.

---

## How It Thinks

At the core is a **GTrXL Transformer-based PPO agent** (Parisotto 2020) that outputs portfolio weights across 8 symbols every 60 seconds. But raw PPO output is just the starting point — it flows through a pipeline that asks progressively harder questions before risking a dollar.

**Is this signal real, or just a correlation?** A GES causal graph (Greedy Equivalence Search, pgmpy) maps the directed relationships between 56 features, the agent's action, and realized reward. If there's no genuine causal path from action to reward for the current market state, the signal gets dampened. The graph is bootstrapped from historical env rollouts with Gaussian exploration noise — because a converged PPO policy is near-deterministic, and GES needs treatment variance to identify edges. Rebuilt nightly.

**Even if it's real, is this a good trade *right now*?** A nightly-trained LightGBM classifier sits after the PPO and asks a different question: "given that the model says go long here, what's the probability this particular trade wins?" This is Lopez de Prado's meta-labeling architecture (*Advances in Financial Machine Learning*, 2018) — separate the directional prediction from the bet-sizing decision. Candidates below threshold get rejected outright. The classifier refits every night as the closed-trade log grows, and its threshold is auto-tuned by Gemini based on observed calibration.

**What regime are we actually in?** The legacy HMM detector labeled every bar "mean_reverting" at 95%+ persistence — even during clear multi-week rallies. It's been replaced by a five-voter ensemble: linear-regression slope, Wilder's ADX, lag-1 return autocorrelation, ATR range-expansion ratio, and HMM as a tiebreaker. Direction-consistency voting produces `trending_up`, `trending_down`, or `mean_reverting` with differentiated persistence (0.65–0.90, not stuck at ceiling). Regime flows into everything downstream — stop widths, take-profit distances, risk budgets, gate multipliers, and the PPO reward function itself.

**What's the sentiment, and is it changing?** A serialized three-agent LLM debate (bull, bear, analyst) via Ollama's 70B model scores recent headlines per symbol. But the raw level (0.2–0.6 most of the time) saturates. What actually matters is the *velocity* — a stock whose sentiment jumped from 0.1 to 0.6 over four hours is a very different signal from one sitting at 0.6 all week. Both level and velocity are applied as direction-aware multipliers, with timeout protection so a hung Ollama inference can never freeze the trading loop.

**Do PPO and the stacking ensemble agree?** 160 LightGBM models (20 per symbol, trained nightly on 8-bar forward labels) provide an independent meta-probability. When the transformer says "go hard long" but the ensemble predicts bearish, that high-conviction disagreement gets dampened to half strength. Agreement passes through.

---

## How It Sizes

Once the signal pipeline decides *what* to trade, a separate stack decides *how much*.

**Kelly-fractional Bayesian sizing.** Every symbol maintains a Beta(α, β) posterior on its win probability, updated after every closed trade. The sizing multiplier isn't a heuristic — it's the ¼-Kelly fraction, mathematically optimal for compound growth (Thorp, 1969). Symbols with strong win/loss ratios get boosted up to 1.6×; symbols that consistently bleed get dampened to 0.4×. Small-sample shrinkage prevents two lucky trades from tripling a position. Capital flows automatically to where the posterior shows real edge.

**Cross-sectional momentum.** Every cycle, the bot z-scores all symbols on a composite of 5-day return, 20-vs-60-day acceleration, volume momentum, and drawdown severity. Top-tercile symbols get boosted 1.25×; bottom-tercile dampened 0.50×. This captures "be where today's alpha is" without requiring a PPO retrain.

**Crowding discount.** When AMD, NVDA, and SMCI are all long at 0.85 pairwise correlation, they're effectively one position — not three independent bets. The bot computes a 60-day return correlation matrix and discounts same-direction positions by their average peer correlation. Prevents a correlated cluster from becoming a single catastrophic bet.

**Liquidity scaling.** Every position is checked against the symbol's average daily dollar volume. Participation above 0.1% of ADV starts getting dampened; above 1% it floors at 0.3×. Extended-hours thresholds are 5× tighter (volume is 10–20× thinner pre/post market). Currently dormant at $30K equity — auto-activates as the account grows.

**CVaR portfolio optimization.** The final allocation uses Conditional Value-at-Risk with Ledoit-Wolf covariance shrinkage, buying-power clamping, and a notional cap of 30% per name. The CVaR budget itself is regime-scaled and Gemini-tuned nightly.

---

## How It Exits

Entry timing gets all the attention, but exit timing is where 40–60% of alpha lives in directional strategies. The bot has four overlapping exit mechanisms.

**Regime-conditional exits.** Now that the ensemble correctly differentiates trending from mean-reverting, exit parameters adapt to alignment. A long position in an uptrend gets a 1.4× wider take-profit and 1.25× wider trailing stop — letting the trend breathe. A counter-trend bet (short in an uptrend) gets tighter everything: take quick profits, accept quick stops. Mean-reverting positions take profits 15% faster than the base.

**Asymmetric trailing stops.** Every position tracks its Maximum Favorable Excursion and Maximum Adverse Excursion in real time. The profit-side ratchet tightens progressively as unrealized P&L climbs. But if a position goes underwater and *never went meaningfully green* (MFE below 0.4%), the trail tightens to 55% of its original width — cutting dead losers before they become real ones.

**TIME-STOP.** Positions held too long without meaningful movement in either direction are "dead trades" — the thesis isn't playing out. After 96 bars (~24 trading hours) with both MFE and MAE below 0.5%, the position is liquidated to free the slot for a better opportunity.

**Software take-profit enforcement.** Alpaca can't hold two closing orders simultaneously, so the take-profit is enforced in software by the monitor loop, with the trailing stop submitted as the native Alpaca order.

---

## How It Protects Itself

The signal pipeline is aggressive by nature — it has to be to find edge. The protection layer is equally aggressive about preventing that edge from being given back.

**Intraday risk pacing.** The CVaR risk budget scales with today's P&L: down 0.5% gets a 50% budget cut, down 1.5% gets 80%, three consecutive losses today floors it at 30%. This is graduated throttling on top of the existing -3% hard daily halt.

**Anti-earnings blackout.** A yfinance earnings calendar (weekly refresh) auto-closes positions one trading day before earnings and blocks new entries in a ±2/1-day window around the event. The single cheapest way to improve win rate — just don't hold through the gap.

**Adverse-selection detection.** Every entry fill is timestamped and price-sampled at T+1, T+5, T+15, and T+30 minutes. The bot computes signed post-fill drift per symbol over a rolling window. When fills are consistently followed by adverse moves, that symbol is toxic — weight gets dampened up to 50%. This is the same technique sell-side market makers use to manage information asymmetry.

**Slippage prediction veto.** A grouped-mean slippage model learns from real fills, bucketed by symbol, hour-of-day, and order size. Before each entry, it predicts expected slippage in basis points. If predicted slippage exceeds 1.2× expected alpha, the trade is skipped. The bot refuses to pay more in execution cost than it expects to earn.

**RETRAIN-GUARD.** Every PPO retrain (nightly 75K timesteps at 18:00 ET) is checkpointed, validated before and after on a deterministic env rollout, and automatically rolled back if the new weights perform worse. On its very first run, it caught a -8012% relative degradation and restored the previous model in under a second. Without this, every downstream layer would have operated on a broken base for 24 hours.

**Sentiment timeout protection.** The Ollama 70B model can hang (GPU OOM, CUDA errors, server crashes). Every individual sentiment debate has a 120-second timeout, and the full sentiment gather has a 300-second timeout. If either fires, the bot falls back to zero sentiment and keeps trading. This prevents a single stuck LLM call from freezing the entire signal pipeline — a failure mode that caused a 5-hour trading halt before the fix was deployed.

---

## How It Improves

The bot doesn't wait for a human to tune it. Nine scheduled tasks run overnight and through the trading day.

**Gemini 2.5 Flash self-tuning** (03:30 AM ET). The bot serializes its full performance profile — Sharpe, Sortino, drawdown, win rate, per-symbol P&L, PPO training scalars, regime distribution, and the history of recent RETRAIN-GUARD decisions — and sends it to Gemini with 25 parameter groups and hard bounds. Gemini writes a Chain-of-Thought reasoning block explaining its analysis, rates its own confidence (which scales step sizes server-side), and proposes changes inside a strict JSON schema. Every value is Pydantic-validated, category-clamped, and anti-oscillation-checked. With retry logic for transient API errors.

**Pre-market micro-retrain** (08:30 AM ET). A lighter PPO fine-tune — 5K timesteps on the last 500 bars at 1e-5 learning rate — adapts the model to overnight news and futures moves before market open. Wrapped in its own RETRAIN-GUARD with stricter thresholds than the nightly run.

**Nightly causal graph rebuild** (03:30 AM ET). GES rediscovers the causal DAG over the latest feature matrix + replay buffer, updating which features genuinely drive rewards.

**Meta-filter refit** (03:30 AM ET). The Lopez-de-Prado classifier retrains on the growing closed-trade log. Brier calibration improves monotonically with sample size.

**Regime precompute** (04:00 AM ET). The five-voter ensemble refreshes across all symbols in parallel.

**EQ-SCORE daily scorecard** (04:30 PM ET). A structured JSON report of what every gate actually did today — per-symbol meta-filter rejects, divergence triggers, slippage vetos, cross-sectional placements, earnings blackouts, entries submitted vs. filled.

**ALPHA-ATTR per-trade attribution** (on every trade close). Records the baseline PPO weight, final post-gate weight, aggregate transformation, MFE/MAE fingerprint, exit reason, and regime at open/close. Answers "which layer pushed this trade out of neutral?" for every single trade.

**Earnings calendar refresh** (weekly via yfinance). Keeps the blackout windows current.

**Universe rotation** (Friday 8 PM ET). Evaluates 34 candidate symbols by liquidity, regime quality, and recent performance. Swaps underperformers and triggers a full retrain on the new roster.

---

## Recent Upgrades (Apr 2026 audit cycle)

A four-agent deep audit on 2026-04-19 surfaced 22 concrete profit leaks across the signal, risk, training, and data layers. The prioritised batch is now live:

**Signal-layer (Agent A)**
- Sentiment blend now applies **after** all gates + equity-curve scale so a gate veto can no longer be silently half-erased by a sentiment tailwind.
- Alpha-attribution `baseline_weights` snapshot moved to just before the gating loop — the aggregate multiplier now reports only the gate+sentiment+eq-scale transformation, keeping per-layer attribution honest.
- Meta-filter now applies a conservative **0.8× pre-fit dampener** during its first ~2 weeks of live trades (previously this window silently returned pass-through for every entry).
- Meta-filter's nightly **Brier score is now consumed** — the `min_prob` threshold tightens when the model is well-calibrated (Brier < 0.30) and loosens otherwise (clipped to [0.35, 0.55]).
- Gate cascade **short-circuits below 1%** cumulative `gate_mult`; skips redundant downstream checks and logs the earliest veto reason instead of the full cascade.
- Equity-curve drawdown scale is now **direction + regime aware**: shorts in a `trending_down` regime (and longs in `trending_up`) are no longer scaled down during bot drawdowns, since they are the best-performing sleeve.
- Causal penalty deferred to the **last multiplier before final renorm** so upstream gates can never read a causal-damped `|weight|` as confidence.

**Risk / exits (Agent B)**
- Ratchet `_ratchet_pending` discard moved to **`finally:`** — a malformed `replace_order` response can no longer strand a symbol's ratchet forever.
- Loss-side tightening has its **own 45 s throttle** (profit ratchet stays 180–540 s); a failing thesis is now cut within one monitor cycle instead of waiting for the profit cooldown.
- Per-cycle **buying-power budget** is now decremented per sized symbol via `reset_bp_budget()` at cycle start — eliminates the 2–3× over-leverage from 5 sequential symbols each claiming full BP.
- CVaR no longer collapses to uniform when a single symbol has short history. It **partitions** into qualified (≥ 100 bars) and insufficient symbols, runs the optimiser on the qualified subset, and gives the insufficient symbols a conviction-weighted 15 % residual slice.
- Bayesian sizer unlocks a **2.0× cap** for "proven winners" (n ≥ 20 closed trades AND p_win ≥ 0.60 AND regime persistence ≥ 0.85) — previously proven high-edge symbols were still capped at 1.6×.
- `MAX_LEVERAGE` **flexes up to +20%** with average regime persistence so the persistence boost is not erased by the uniform gross-exposure rescale.
- MFE / MAE updates **hoisted to the top of `_monitor_one_position`** — TIME-STOP and loss-tighten now see accurate peak excursion even when TP enforcement blocks the ratchet.
- Fractional remainder after a partial exit fill is **actively swept** on the next monitor cycle via a new `_pending_fractional_close` queue (previously left unprotected until the next reconcile).
- Monitor re-reads `tracker.groups.get(sym)` inside the TIME-STOP and PENDING_EXIT branches so it never acts on a state the stream handler has since transitioned.

**Earlier the same day (Apr 19 morning audit):**
- Slippage-veto tuned to `2.0× edge` with a `min_samples=5` gate (was firing ×45 per fill).
- PPO walk-forward OOS acceptance relaxed with a `-0.25` floor + 35 % gap cap (no longer silently rejecting borderline-noise windows alongside genuinely broken ones).
- Hard-to-borrow symbols auto-detected; trailing stops retry with DAY TIF on GTC rejection.
- Entry qty rounds down to whole shares when size ≥ 1 so native trailing stops cover the full position.
- Gemini tuner wrapped in exponential-backoff retry for 503 / timeout errors.
- Causal lazy build now logs explicit completion (`edges=N, elapsed=Ns`).
- Alpha-attribution pending list persisted across restarts (no more orphan `exit_only` records).

Full audit report at [`MEGATRON_AUDIT_2026-04-19.md`](./MEGATRON_AUDIT_2026-04-19.md).

---

## It Runs Like a Production System

This connects to a real Alpaca broker, streams live bars via WebSocket, places real limit orders with trailing stops and take-profits, handles fill events through an async state machine, manages a full order lifecycle (pending entry → open → pending exit → closed), and persists every piece of learned state atomically across restarts. Currently running 24/7 in paper trading mode.

Fourteen state files round-trip across crashes: `order_tracker.json`, `regime_cache.json` (schema-versioned), `meta_filter.pkl`, `bayesian_sizing.pkl`, `slippage_predictor.pkl`, `adverse_selection.pkl`, `earnings_cache.pkl`, `causal_cache_portfolio.pkl`, `replay_buffer_portfolio.pkl`, `live_signal_history.json`, `execution_scorecard_log.jsonl`, `alpha_attribution_log.jsonl`, `dynamic_config.json`, `tuning_history.json`. Every learned quantity survives a restart and keeps growing.

---

## Architecture

```
trading_bot/
|
|-- bot.py                          # Async event loop, 9 scheduled tasks, trading loop
|-- config.py                       # 100+ Pydantic-validated settings
|-- gemini_tuner.py                 # Gemini 2.5 Flash v2.1 (25 groups, CoT reasoning)
|
|-- broker/
|   |-- alpaca.py                   # Orders, ratchet, asymmetric loss-tighten, TIME-STOP
|   |-- stream.py                   # WebSocket fills, AVA recording, slippage + Bayesian updates
|   |-- order_tracker.py            # State machine with MFE/MAE per position
|
|-- strategy/
|   |-- signals.py                  # Signal blend + gate cascade (timeout-protected)
|   |-- regime.py                   # 5-voter ensemble (slope + ADX + autocorr + range + HMM)
|   |-- risk.py                     # CVaR + intraday pacing + Kelly sizing
|   |-- meta_filter.py              # Lopez-de-Prado P(win) classifier
|   |-- bayesian_sizing.py          # Per-symbol Beta posteriors + Kelly fraction
|   |-- cross_sectional.py          # Daily z-score momentum ranking
|   |-- correlation_discount.py     # Crowding-aware position sizing
|   |-- earnings_filter.py          # Anti-earnings blackout + auto-close
|   |-- slippage_predictor.py       # Grouped-mean slippage predictor + veto
|   |-- adverse_selection.py        # Post-fill toxicity detector
|   |-- execution_scorecard.py      # EQ-SCORE daily gate-activity summary
|   |-- alpha_attribution.py        # Per-trade alpha attribution logger
|   |-- cognitive.py                # Equity-curve trader + profit velocity + autopsy
|   |-- portfolio_rebalancer.py     # Portfolio-level rebalance + causal scaling
|   |-- universe.py                 # Weekly universe rotation
|
|-- models/
|   |-- trainer.py                  # PPO training + RETRAIN-GUARD + micro-retrain
|   |-- env.py / portfolio_env.py   # Gym environments (per-symbol + multi-asset)
|   |-- features.py                 # 56-feature generator (hourly macro refresh)
|   |-- policies.py                 # AuxGTrXL policy (gated transformer + aux vol head)
|   |-- stacking_ensemble.py        # 20-model LightGBM bootstrap ensemble
|   |-- causal_signal_manager.py    # GES + noise-injected bootstrap + column cap
|
|-- data/
|   |-- handler.py                  # ArcticDB + Redis + Polygon/Alpaca/yfinance
|   |-- ingestion.py                # Real-time bar stream + historical backfill
|
|-- utils/
|   |-- local_llm.py                # Serialized Ollama debate (timeout-protected)
|   |-- helpers.py                  # Market hours, holidays
|   |-- log_setup.py                # Logging config
```

---

## Getting Started

### System Requirements

The bot runs on anything from a modest desktop to a GPU workstation. Here's what you actually need for each mode:

**GPU Mode (recommended for full capability):**

| Component | Minimum | Recommended |
|---|---|---|
| **CPU** | 4 cores / 8 threads | 8+ cores |
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | NVIDIA with 6 GB VRAM (e.g. RTX 3060) | 8+ GB VRAM (RTX 3070/4070+) |
| **CUDA** | 11.8+ | 12.x |
| **Storage** | 20 GB free | 50 GB+ (ArcticDB + model checkpoints grow over time) |
| **Network** | Stable broadband | Low-latency connection for WebSocket streaming |
| **OS** | Linux, Windows 10/11, macOS | Ubuntu 22.04+ or Windows 11 with WSL2 |

GPU handles PPO training (~5 min nightly), TFT encoder precompute (~6 min on first startup), and Ollama LLM inference. Without GPU, all of these fall back to CPU but take 3–10× longer.

**CPU-Only Mode (everything works, just slower):**

| Component | Minimum | Recommended |
|---|---|---|
| **CPU** | 4 cores / 8 threads | 8+ cores (PPO training is CPU-bound without GPU) |
| **RAM** | 8 GB | 16 GB+ (Ollama 8B model uses ~5 GB) |
| **GPU** | None required | — |
| **Storage** | 20 GB free | 50 GB+ |
| **Network** | Stable broadband | Low-latency connection |
| **OS** | Linux, Windows 10/11, macOS | Any with Python 3.11+ and Docker support |

On CPU-only: nightly PPO retrain takes ~15–30 min instead of ~5 min. TFT precompute takes ~20 min on first startup instead of ~6 min. Ollama sentiment with the 8B model runs fine on CPU (~5–10s per call). Live trading loop is unaffected — signal generation is lightweight.

**Docker mode** handles all dependencies automatically. Without Docker, you'll also need:
- **Python 3.11+** (3.12 recommended)
- **[Redis](https://redis.io/)** — for data caching. On Windows: use [Memurai](https://www.memurai.com/) or Docker.
- **[Ollama](https://ollama.ai/)** — for LLM sentiment. Native Windows/macOS/Linux installers available.
- **[Alpaca](https://alpaca.markets/) account** — paper trading keys are free and work out of the box.

**Optional but recommended:**
- [Gemini API key](https://ai.google.dev/) — enables nightly self-tuning. Free tier works.
- [NewsAPI key](https://newsapi.org/) — enables real news headlines for LLM sentiment. Free tier = 100 requests/day.

### Quick Start (Linux / macOS)

```bash
git clone https://github.com/mmccarney370/nyse-trading-bot.git
cd nyse-trading-bot

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set up Ollama for sentiment analysis
ollama pull llama3.1:8b

# Configure your API keys
cp .env.example .env
# Edit .env with your Alpaca keys (required) and optional keys
```

### Quick Start (Windows)

```powershell
git clone https://github.com/mmccarney370/nyse-trading-bot.git
cd nyse-trading-bot

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Set up Ollama (download from https://ollama.ai — native Windows installer)
ollama pull llama3.1:8b

# Configure your API keys
copy .env.example .env
# Edit .env with your Alpaca keys (required) and optional keys
```

**Windows notes:**
- Redis: install [Memurai](https://www.memurai.com/) (Redis-compatible for Windows) or run Redis in Docker: `docker run -d -p 6379:6379 redis`
- GPU: install CUDA Toolkit from [NVIDIA](https://developer.nvidia.com/cuda-downloads) if using GPU training
- The bot auto-detects Windows and sets the correct asyncio event loop policy

### Quick Start (Docker — easiest, any platform)

Docker bundles everything — Redis, Ollama, Python, CUDA — into one command. Works on Linux, macOS, and Windows with Docker Desktop.

```bash
git clone https://github.com/mmccarney370/nyse-trading-bot.git
cd nyse-trading-bot

cp .env.example .env
# Edit .env with your Alpaca API keys (required)

# GPU mode (requires NVIDIA Container Toolkit):
docker compose up -d

# CPU-only mode (no GPU needed, slower training):
docker compose --profile cpu up -d

# First time: pull the Ollama model for sentiment
docker compose exec ollama ollama pull llama3.1:8b

# Watch the bot run:
docker compose logs -f bot

# Stop:
docker compose down
```

All state (models, caches, logs, order tracker) persists in mounted volumes — survives container rebuilds.

### Running (without Docker)

```bash
python __main__.py                                           # live paper trading
python run_backtest.py --start 2025-01-01 --end 2025-12-31  # backtest
```

First startup takes 5–6 minutes (TFT precompute, stacking training, causal bootstrap, meta-filter fit). Subsequent starts are faster with cached state files. Logs: `logs/nyse_bot.log` | Training curves: `tensorboard --logdir ppo_tensorboard`

---

## Configuration

Everything lives in `config.py` with Pydantic validation — 100+ tunable parameters across 25 groups, all eligible for autonomous nightly tuning by Gemini inside safety-clamped bounds. Key groups: Risk, Intraday Pacing, Regime Ensemble, Execution, Asymmetric Trailing, PPO, Signal Blend, Meta-Filter, Bayesian Sizing, Cross-Sectional, Crowding, Earnings, Slippage, Divergence, Adverse Selection, Causal, Reward Shaping, Universe, Tuning, RETRAIN-GUARD, Micro-Retrain, TIME-STOP, Liquidity, Kelly, REX.

---

## Monitoring

Every subsystem logs with a searchable prefix — `rg '[BAYESIAN-SIZE]'` or `rg '[RETRAIN-GUARD]'` instantly filters to the layer you care about. Key prefixes: `[CVaR]`, `[RISK PACING]`, `[REGIME ENSEMBLE+HMM]`, `[CAUSAL PENALTY APPLIED]`, `[RATCHET LOSS-TIGHTEN]`, `[META-FILTER]`, `[BAYESIAN-SIZE]`, `[CS-MOMENTUM]`, `[CROWDING-DISCOUNT]`, `[EARNINGS BLACKOUT]`, `[ADVERSE-SEL]`, `[SLIPPAGE-VETO]`, `[DIVERGENCE]`, `[TIME-STOP]`, `[RETRAIN-GUARD]`, `[MICRO-RETRAIN]`, `[EQ-SCORE]`, `[ALPHA-ATTR]`, `[GEMINI REASONING]`, `[SENTIMENT TIMEOUT]`.

TensorBoard at `tensorboard --logdir ppo_tensorboard`. Live position state in `order_tracker.json` with MFE/MAE per symbol. Daily gate-activity scorecard in `execution_scorecard_log.jsonl`. Per-trade attribution in `alpha_attribution_log.jsonl`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "CVaR fell back to parity" | Need 50+ overlapping bars per symbol |
| `GES discovered 0 edges` | Fixed — bootstrap injects noise (`CAUSAL_BOOTSTRAP_NOISE_SIGMA=0.4`) |
| Meta-filter rejecting everything | Lower `META_FILTER_MIN_PROB` (default 0.33) |
| BPS multiplier stuck at 1.0 | Normal — shrinkage blends toward neutral with < 8 closed trades |
| News API `rateLimited` | Sentiment falls back to 0.0; upgrade NewsAPI tier for more |
| `[SENTIMENT TIMEOUT]` firing | Ollama 70B hung — sentiment returns 0.0, trading continues normally |
| `[RETRAIN-GUARD] ROLLBACK` | Working as intended — bad retrain caught and reverted |
| `[TIME-STOP]` close failed | Fixed — now uses `close_position_safely` (cancels stop before close) |
| `insufficient qty available` on reattach | Fixed — reattach re-queries live Alpaca qty before submitting |
| Alpaca websocket reconnects | Auto-recovers; check API keys if persistent |

---

## Technical Deep-Dive

See [**`WHITEPAPER.md`**](./WHITEPAPER.md) for mathematical formulations, architectural rationale, algorithm derivations, and research citations behind every subsystem.

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

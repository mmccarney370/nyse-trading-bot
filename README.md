# NYSE Trading Bot

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Status](https://img.shields.io/badge/Status-Live_Paper_Trading-green)
![RL](https://img.shields.io/badge/RL-GTrXL_Recurrent_PPO-purple)
![Causal](https://img.shields.io/badge/Causal_AI-GES_%2B_DoWhy-orange)
![Self-Tuning](https://img.shields.io/badge/Self--Tuning-Gemini_2.5_Flash-red)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

**A fully autonomous, self-improving algorithmic trading system** that combines deep reinforcement learning, causal inference, adaptive memory, and LLM-powered self-tuning to trade US equities around the clock — with zero manual intervention.

---

## Why This Bot Is Different

### It Learns From Its Own Trades
A **GTrXL Transformer-based PPO agent** (not a rules engine, not a simple moving average crossover) makes portfolio allocation decisions across 8 symbols simultaneously. It trains on a realistic multi-asset environment with transaction costs, drawdown penalties, and Sortino-weighted rewards — then continues learning nightly from real market outcomes.

### It Understands *Why*, Not Just *What*
A **causal inference layer** (GES graph discovery + DoWhy) separates genuine market signals from spurious correlations. Before every trade, the bot asks: *"Is the action-reward relationship real, or just a statistical coincidence?"* Trades backed by weak causal evidence get their size reduced automatically.

### It Remembers Its Mistakes
An **adaptive memory system** tracks recent trade outcomes in real-time and adjusts behavior without retraining:
- **Anti-churn**: After a stop-out, confidence for re-entering the same direction decays exponentially (0.5^n per consecutive stop). Prevents the "stop, re-enter, stop, re-enter" death spiral.
- **Cross-asset panic detection**: When 3+ symbols stop out within 5 minutes, the system enters defensive mode and suppresses all new entries for a cooldown period — recognizing that correlated failures are systemic, not individual.
- **Adaptive signal weighting**: Tracks per-component accuracy (ensemble, PPO, sentiment) over a sliding window of closed trades and dynamically reweights the signal blend. Components that have been wrong recently get downweighted; accurate ones get upweighted. Online Bayesian updating of the signal pipeline.

### It Reads Market Direction, Not Just Regime Type
An **ensemble of Hidden Markov Models** classifies each symbol as *trending up*, *trending down*, or *mean-reverting* with a persistence confidence score. The directional regime flows everywhere: it scales risk budgets, adjusts trailing stop widths, gates counter-trend entries (longs in downtrends get 60% confidence suppression), and shapes the PPO reward function.

### It Has Multi-Layer Risk Gates
Seven independent confidence gates stack multiplicatively before any trade executes:

| Gate | What It Does |
|------|-------------|
| **Regime Gate** | Longs in `trending_down` get confidence * 0.4; shorts in `trending_up` likewise |
| **VIX Gate** | VIX > 28 scales down long confidence, boosts short confidence proportionally |
| **SPX Breadth Gate** | S&P 500 below 200-day SMA = bear market; long confidence * 0.3 |
| **1H Trend Gate** | 15-min signal opposing the 1-hour trend gets confidence halved |
| **Anti-Churn Gate** | Recent stop-outs in same direction decay confidence exponentially |
| **Defensive Mode** | Cross-asset panic suppresses all new entries for 3-hour cooldown |
| **MIN_CONFIDENCE** | Final filter: signals below 0.72 confidence are rejected entirely |

A long signal in a downtrend + high VIX + bearish SPX: confidence * 0.4 * 0.85 * 0.3 * 0.5 = ~5% of original. Effectively blocked.

### It Tunes Itself Every Night
At 3:30 AM, the bot sends its complete performance profile — Sharpe, Sortino, drawdown, win rate, per-symbol P&L, regime breakdown — to **Gemini 2.5 Flash**, which proposes parameter adjustments within safety-clamped bounds. Every change is logged as structured JSON.

### It Runs Like a Production System
This isn't a Jupyter notebook experiment. It connects to a **live broker (Alpaca)**, streams real-time bars, places real orders with trailing stops and take-profits, handles websocket fill events, manages a full order lifecycle state machine, persists state across restarts, and self-heals from disconnects. Currently running 24/7 in paper trading mode.

---

## Key Capabilities

| Capability | Implementation |
|---|---|
| **Portfolio Optimization** | CVaR (Conditional Value-at-Risk) with Ledoit-Wolf covariance shrinkage, regime-scaled risk budgets, and real-time buying power enforcement |
| **Signal Generation** | Adaptive multi-layer blend: Recurrent PPO + LightGBM stacking ensemble (20 models, 8-bar forward labels) + causal penalty + Ollama LLM sentiment debate + 7 confidence gates |
| **56 Input Features** | Bollinger Bands, RSI, MACD, ATR, CCI, Stochastic, OBV z-score, Chaikin, VWAP deviation, volume imbalance, divergence detection, SMA(50/200) trend + golden cross, VIX/yield curve macro, TFT encoder embeddings, and more |
| **Directional Regime Detection** | HMM ensemble distinguishes `trending_up`, `trending_down`, and `mean_reverting` with persistence score — not just regime type but direction |
| **Adaptive Exits** | Alpaca native trailing stops with periodic PATCH ratcheting that tightens based on unrealized profit tiers + software take-profit enforcement |
| **Causal Overfitting Brake** | GES-discovered DAG over 56+ variables identifies genuine action-reward pathways; spurious signals get suppressed |
| **Sentiment Analysis** | Serialized multi-agent LLM debate (3 agents: bull, bear, analyst) via Ollama (70B model), with 8b fallback |
| **Walk-Forward Validation** | 6-window walk-forward threshold optimization with IS/OOS gap monitoring and automatic overfitting rejection |
| **Adaptive Memory** | Real-time anti-churn (exponential decay after stops), cross-asset panic detection, and online Bayesian signal weight adjustment |
| **Dynamic Universe Rotation** | Weekly evaluation of 34 candidate symbols by liquidity, regime quality, and recent performance — automatic retraining on rotation |
| **State Persistence** | Atomic file writes for order tracker, regime cache, signal history, replay buffers, and entry timestamps — survives crashes and restarts |
| **Structured Observability** | Every subsystem logs with searchable prefixes (`[CVaR]`, `[CAUSAL]`, `[REGIME]`, `[TRACKER]`, `[RATCHET]`, `[CONF GATE]`, `[1H GATE]`, `[PANIC DETECT]`, `[CHURN]`), plus TensorBoard for PPO training curves |

---

## How It Works

The bot runs a **45-second decision loop** during market hours:

```
Market Data (Alpaca Stream + ArcticDB cache)
     |
     v
Regime Detection (HMM ensemble → trending_up / trending_down / mean_reverting)
     |
     v
Feature Generation (56 technical + macro + SMA trend + TFT features)
     |
     v
Portfolio PPO Inference (GTrXL policy → target weights per symbol)
     |
     v
Signal Blend (adaptive weights from trade accuracy, or static PPO/ensemble/sentiment)
     |
     v
Confidence Gates (regime + VIX + SPX + 1H trend + anti-churn + defensive + min_conf)
     |
     v
CVaR Position Sizing (risk budget + buying power + notional caps)
     |
     v
Order Execution (limit entry → websocket fill → trailing stop + TP)
     |
     v
Position Monitor (ratchet tightening, software TP/SL, reattach)
     |
     v
Adaptive Memory (record outcomes → update churn/panic/weight state)
```

### Overnight Self-Improvement Cycle

| Time (ET) | Task | What Happens |
|---|---|---|
| **3:30 AM** | Gemini Tuning | Sends full performance report to Gemini 2.5 Flash; receives parameter adjustments within safety bounds; applies and logs changes |
| **3:30 AM** | Causal Refresh | Rebuilds GES causal graph with latest feature data + replay buffer transitions |
| **4:00 AM** | Regime Precompute | Parallel HMM fitting across all symbols using ThreadPoolExecutor; updates shared regime cache with directional labels |
| **6:00 PM** | PPO Retrain | Online incremental update of portfolio PPO (75K timesteps) using latest market data |
| **Friday 8 PM** | Universe Rotation | Evaluates 34 candidates, swaps underperformers, retrains all models on new universe |

---

## Architecture

```
trading_bot/
|
|-- bot.py                          # Main async event loop + orchestration
|-- config.py                       # All settings, Pydantic-validated (single source of truth)
|-- __main__.py                     # Entry point
|-- backtest.py                     # Full backtesting engine (portfolio + per-symbol)
|-- gemini_tuner.py                 # Gemini 2.5 Flash self-tuning + structured JSON logging
|
|-- broker/
|   |-- alpaca.py                   # Order execution, trailing stop ratcheting, position sync
|   |-- stream.py                   # WebSocket handler (fill -> OCO -> close -> adaptive memory)
|   |-- order_tracker.py            # Persistent state machine with atomic saves
|
|-- strategy/
|   |-- signals.py                  # Adaptive signal blend + 7-layer confidence gates + memory
|   |-- regime.py                   # Directional HMM ensemble (up/down/MR) + persistence scoring
|   |-- risk.py                     # CVaR optimization, Kelly sizing, buying power clamps
|   |-- portfolio_rebalancer.py     # Portfolio-level rebalance with notional caps
|   |-- universe.py                 # Weekly universe rotation with liquidity/momentum filters
|
|-- models/
|   |-- trainer.py                  # PPO/RecurrentPPO training (startup + nightly online)
|   |-- env.py                      # Per-symbol Gym env (Sortino + causal rewards)
|   |-- portfolio_env.py            # Multi-asset Gym env for portfolio PPO
|   |-- features.py                 # 56-feature generator (technical + SMA + macro + TFT)
|   |-- policies.py                 # Custom AuxGTrXL policy (gated transformer + aux vol head)
|   |-- stacking_ensemble.py        # LightGBM bootstrap ensemble (20 models, 8-bar labels)
|   |-- causal_signal_manager.py    # GES discovery + replay buffer + fast penalty computation
|   |-- causal_rl_manager.py        # Causal buffer persistence + rotation reset
|   |-- bot_initializer.py          # Startup: model loading, cache warming, state restoration
|   |-- ppo_utils.py                # PPO save/load, warmup + constant LR schedule
|
|-- data/
|   |-- handler.py                  # ArcticDB + Redis caching + Polygon/Alpaca/Finnhub fetch
|   |-- ingestion.py                # Real-time bar stream + historical backfill
|
|-- utils/
|   |-- helpers.py                  # Market hours, holiday calendar
|   |-- local_llm.py                # Serialized Ollama multi-agent debate for sentiment
|   |-- log_setup.py                # Logging configuration
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended for PPO training and TFT encoder)
- [Alpaca](https://alpaca.markets/) account (paper trading keys work fine)
- [Redis](https://redis.io/) (for data caching)
- [Ollama](https://ollama.ai/) installed locally (optional, for sentiment analysis)
- [Gemini API key](https://ai.google.dev/) (optional, for self-tuning)
- [News API key](https://newsapi.org/) (optional, for news sentiment)

### Setup

```bash
git clone https://github.com/mmccarney370/nyse-trading-bot.git
cd nyse-trading-bot

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file:

```env
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_secret_key
GEMINI_API_KEY=your_gemini_key          # optional
NEWS_API_KEY=your_newsapi_key           # optional
```

For sentiment analysis:

```bash
ollama pull llama3.1:8b
```

### Running

```bash
# Live paper trading
python __main__.py

# Backtest mode
python run_backtest.py --start 2025-01-01 --end 2025-12-31
```

The bot will initialize data, load or train models, build causal graphs, and start the trading loop. First startup takes ~5 minutes (TFT precompute on GPU + stacking training); subsequent starts are faster with cached models.

Logs: `nyse_bot.log` | Training curves: `tensorboard --logdir ppo_tensorboard`

---

## Configuration

Everything lives in `config.py` with Pydantic validation. Key groups:

| Group | Key Parameters | What They Control |
|---|---|---|
| **Risk** | `RISK_PER_TRADE`, `MAX_LEVERAGE`, `KELLY_FRACTION`, `MAX_POSITIONS` | Capital allocation and risk limits |
| **Regime** | `HMM_ENSEMBLE_SIZE`, `REGIME_SHORT_WEIGHT`, `HURST_TREND_THRESHOLD` | How the bot reads market conditions and direction |
| **Execution** | `TRAILING_STOP_ATR_*`, `TAKE_PROFIT_ATR_*`, `MIN_HOLD_BARS_*`, `RATCHET_*` | Stop placement, profit targets, and trail tightening |
| **PPO** | `PPO_LEARNING_RATE`, `PPO_ENTROPY_COEFF`, `GTRXL_*`, `SORTINO_WEIGHT` | RL training behavior and architecture |
| **Signal** | `PPO_SIGNAL_WEIGHT`, `SENTIMENT_WEIGHT`, `MIN_CONFIDENCE`, `LABEL_HORIZON_BARS` | Signal blend weights, thresholds, and LightGBM label horizon |
| **Causal** | `USE_CAUSAL_RL`, `CAUSAL_PENALTY_WEIGHT` | Causal inference strength and sensitivity |
| **Tuning** | Gemini schedule, safety bounds | Self-tuning schedule and parameter clamps |

The Gemini tuner adjusts most parameters within clamped bounds (typically +/-15-25%). Every change is logged with full performance context as structured JSON.

---

## Monitoring

- **`nyse_bot.log`** — Searchable by subsystem: `[CVaR]`, `[REGIME]`, `[CAUSAL]`, `[TRACKER]`, `[RATCHET]`, `[STREAM]`, `[CONF GATE]`, `[1H GATE]`, `[VIX GATE]`, `[SPX GATE]`, `[REGIME GATE]`, `[CHURN]`, `[PANIC DETECT]`, `[DEFENSIVE]`, `[ADAPTIVE BLEND]`
- **TensorBoard** — `tensorboard --logdir ppo_tensorboard` for reward curves, clip fraction, explained variance, policy entropy
- **Gemini audit trail** — Search `"event": "gemini_change"` for every parameter adjustment with before/after values
- **`order_tracker.json`** — Real-time state of all active positions and their exit orders
- **Monitor heartbeat** — Position status logged every 20 seconds during market hours

---

## Troubleshooting

| Symptom | Cause | Resolution |
|---|---|---|
| "Insufficient features for stacking" | Not enough historical bars | Wait for cache to fill (~300 bars) or increase lookback |
| "CVaR fell back to parity" | Not enough overlapping returns | Need 50+ bars per symbol |
| "HMM ensemble failed" | Data too flat for HMM convergence | Normal; Hurst fallback handles it |
| "Causal buffer insufficient" | Replay buffer needs ~100+ transitions | Fills automatically from live trading |
| `[CONF GATE]` suppressing everything | MIN_CONFIDENCE too high or gates too aggressive | Lower `MIN_CONFIDENCE` in config or tune gate multipliers |
| `[PANIC DETECT]` triggering | 3+ symbols stopped in 5 min | Working as intended; bot goes defensive during market crashes |
| `[CHURN]` messages | Repeated stop-outs on same symbol | Working as intended; prevents re-entry death spiral |
| Stream reconnecting | Alpaca websocket dropped | Auto-reconnects; check API keys if persistent |
| TFT features all zero | Cache needs rebuild | Cleared automatically after 7 days; delete `ppo_checkpoints/tft_cache/` to force |
| LLM NaN warnings | Ollama timeout from concurrent calls | Serialized lock prevents this; increase timeout if still occurring |

---

## Disclaimer

This is a personal project built for learning and paper trading. It is not financial advice. Markets are unpredictable, models overfit, and past performance means nothing about the future. If you choose to run this with real money, that's entirely your decision and your risk. Start with paper trading and stay there until you deeply understand every component.

---

## License

MIT License

Copyright (c) 2026 Matthew McCartney

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

> *"You have power over your mind, not outside events. Realize this, and you will find strength."*
> — Marcus Aurelius

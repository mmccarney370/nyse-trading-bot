# nyse-trading-bot

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Status](https://img.shields.io/badge/Status-Paper_Trading-green)
![RL](https://img.shields.io/badge/RL-Recurrent_PPO-purple)
![Causal](https://img.shields.io/badge/Causal-GES_%2B_DoWhy-orange)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

An autonomous algorithmic trading system for US equities that combines reinforcement learning, causal inference, and adaptive self-tuning to make regime-aware portfolio decisions. It runs continuously during market hours, learns from its own trades, and adjusts its own hyperparameters overnight — no manual intervention required.

This isn't a backtest-only research project. It connects to a live broker (Alpaca), streams real-time bars, places real orders with trailing stops and take-profits, and tracks every position through a full lifecycle state machine. That said, it's currently running in paper trading mode, and you should too until you're confident in what it's doing.

---

## What It Actually Does

The bot wakes up every ~45 seconds during market hours and runs a full decision loop:

1. **Pulls fresh market data** from Alpaca's real-time stream (15-min and 1H bars), with ArcticDB as a local cache so cold starts don't lose history.

2. **Detects the current market regime** using an ensemble of Hidden Markov Models fitted on returns, rolling volatility, and volume changes. Each symbol gets classified as either *trending* or *mean-reverting*, along with a persistence score (how confident the HMM is that the regime will continue). A Hurst exponent fallback kicks in if the HMM can't converge.

3. **Generates trading signals** through a multi-layer stack:
   - A **Recurrent PPO agent** (GTrXL transformer policy via sb3-contrib) trained on a custom multi-asset Gym environment that simulates portfolio allocation with realistic transaction costs, drawdown penalties, and Sortino-weighted rewards.
   - A **LightGBM stacking ensemble** (15 bootstrapped models) for next-bar direction probability.
   - **Causal penalty factors** from a GES-discovered causal graph (via pgmpy) refined with DoWhy statistical refutation — this penalizes actions that the causal model thinks are spurious.
   - Optional **sentiment analysis** from FinBERT + local LLM debate (Ollama).

4. **Sizes positions** using CVaR (Conditional Value-at-Risk) portfolio optimization with Ledoit-Wolf covariance shrinkage, regime-scaled risk budgets, hard notional caps, and real-time buying power checks against the broker.

5. **Executes trades** through Alpaca with bracket-style exit management: every entry gets a trailing stop (native Alpaca auto-trail with periodic PATCH ratcheting to tighten) and a take-profit limit. The websocket stream handles fill events and implements manual OCO (one-cancels-other) logic since Alpaca doesn't support true OCO on equities.

6. **Self-tunes overnight** — at 3:30 AM ET, the bot sends its recent performance metrics (Sharpe, Sortino, drawdown, win rate, profit factor) to Gemini 2.5 Flash, which proposes parameter adjustments within safety-clamped bounds. Changes are logged as structured JSON so you can audit every decision the tuner ever made.

---

## Architecture

```
trading_bot/
|
|-- bot.py                          # Main async event loop + orchestration
|-- config.py                       # All settings, Pydantic-validated
|-- __main__.py                     # Entry point (python -m trading_bot)
|-- backtest.py                     # Full backtesting engine (portfolio + per-symbol modes)
|-- run_backtest.py                 # Standalone backtest runner with CLI args
|-- gemini_tuner.py                 # Gemini API self-tuning + structured logging
|
|-- broker/
|   |-- alpaca.py                   # Order execution, trailing stop ratcheting, position sync
|   |-- stream.py                   # WebSocket trade-update handler (fill -> OCO -> close)
|   |-- order_tracker.py            # Persistent state machine: pending -> open -> exit -> closed
|
|-- strategy/
|   |-- signals.py                  # Signal blending (PPO + ensemble + causal + sentiment)
|   |-- regime.py                   # HMM ensemble + Hurst fallback + divergence detection
|   |-- risk.py                     # CVaR optimization, position sizing, buying power clamps
|   |-- portfolio_rebalancer.py     # Portfolio-level rebalance pipeline
|   |-- universe.py                 # Weekly universe rotation with liquidity filters
|
|-- models/
|   |-- trainer.py                  # PPO/RecurrentPPO training loops (nightly retrain)
|   |-- env.py                      # Per-symbol Gym environment (Sortino + causal rewards)
|   |-- portfolio_env.py            # Multi-asset Gym environment for portfolio PPO
|   |-- features.py                 # 50+ technical features (BB, RSI, MACD, ATR, OBV, etc.)
|   |-- policies.py                 # Custom AuxGTrXL policy (auxiliary volatility head)
|   |-- stacking_ensemble.py        # LightGBM bootstrap ensemble for meta-probabilities
|   |-- causal_signal_manager.py    # GES causal discovery + DoWhy refutation + replay buffer
|   |-- causal_rl_manager.py        # Causal buffer persistence + rotation reset
|   |-- bot_initializer.py          # Startup initialization (model loading, cache warming)
|   |-- ppo_utils.py                # PPO helper utilities
|
|-- data/
|   |-- handler.py                  # ArcticDB local storage + Polygon/Alpaca/Finnhub fetching
|   |-- ingestion.py                # Real-time bar stream + historical backfill
|
|-- utils/
|   |-- helpers.py                  # Market hours, time utilities
|   |-- local_llm.py                # Ollama LLM debate for sentiment
|   |-- log_setup.py                # Logging configuration
|
|-- tests/                          # Unit + integration tests
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- An [Alpaca](https://alpaca.markets/) account (paper trading keys work fine)
- [Ollama](https://ollama.ai/) installed locally (optional, for sentiment analysis)
- A [Gemini API key](https://ai.google.dev/) (optional, for self-tuning)
- A [News API key](https://newsapi.org/) (optional, for causal news features)

### Setup

```bash
git clone https://github.com/mmccarney370/nyse-trading-bot.git
cd nyse-trading-bot

python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
API_KEY=your_alpaca_api_key
API_SECRET=your_alpaca_secret_key
GEMINI_API_KEY=your_gemini_key          # optional
NEWS_API_KEY=your_newsapi_key           # optional
```

If you want sentiment analysis, pull the Ollama models:

```bash
ollama pull llama3.1:8b
```

### Running

```bash
# Live paper trading
python -m trading_bot

# Backtest mode
python run_backtest.py --start 2025-01-01 --end 2025-12-31
```

The bot will:
1. Initialize data (fetch historical bars if cache is empty)
2. Load or train PPO models
3. Build causal graphs (once replay buffer has enough data)
4. Start the trading loop + websocket stream

Logs go to `nyse_bot.log`. TensorBoard logs go to `ppo_tensorboard/`.

---

## How the Pieces Fit Together

### Regime Detection

The regime detector runs an ensemble of 4 Gaussian HMMs with different random seeds, each fitted on a 3D observation space (returns, rolling volatility, log volume change). Each model votes on whether the current state is trending or mean-reverting based on its transition matrix self-probability. The majority vote wins, and the average self-transition probability becomes the "persistence score" — a continuous 0-1 measure of how sticky the current regime is.

This persistence score flows everywhere: it scales the total risk budget, tilts per-symbol weights toward high-conviction names, adjusts min-hold periods, and shapes the PPO reward function. When the HMM can't fit (too little data, degenerate variance), it falls back to a Hurst exponent calculation.

### The PPO Agent

The portfolio-level PPO uses a GTrXL (Gated Transformer-XL) recurrent policy, which gives it memory across timesteps without the vanishing gradient problems of vanilla LSTMs. The observation space is ~50 features per symbol (technical indicators, macro inputs, regime flags, TFT encoder outputs) concatenated with portfolio state (current weights, cash ratio).

The reward function is deliberately complex — it blends raw returns with a Sortino component, volatility penalty, drawdown penalty, turnover cost, causal penalty factor, and regime persistence bonus. The idea is to teach the agent that not all returns are equal: a 1% gain during a trending regime with low drawdown is worth more than a 1% gain during whipsaw conditions.

### Causal Layer

The causal pipeline uses the GES (Greedy Equivalence Search) algorithm from pgmpy to discover a directed acyclic graph over the feature space + action + reward. Each edge is then tested with DoWhy's statistical refutation (random common cause, placebo treatment). Edges that fail refutation are pruned.

The surviving causal graph is used to compute a "penalty factor" for each trade — if the causal model thinks the action-reward relationship is mostly explained by confounders rather than genuine signal, the penalty scales down the position. This acts as a built-in overfitting brake.

### Order Lifecycle

Every trade follows a strict state machine tracked by `OrderTracker`:

```
pending_entry  ->  open  ->  pending_exit  ->  closed
    |                |             |
  (limit order    (trailing     (one exit
   submitted)     stop + TP      fills,
                  attached)      other
                                canceled)
```

The trailing stop uses Alpaca's native auto-trail (set by `trail_percent`), but the bot also runs a periodic ratchet loop that PATCHes the trail tighter as price moves favorably. This means stops get tighter over time even during min-hold periods when no new trades are allowed.

---

## Configuration

Everything lives in `config.py` and is validated by Pydantic on startup. Key groups:

| Group | Examples | What They Control |
|-------|----------|-------------------|
| **Risk** | `RISK_PER_TRADE`, `MAX_LEVERAGE`, `KELLY_FRACTION` | How much capital goes into each position |
| **Regime** | `HMM_ENSEMBLE_SIZE`, `HURST_TREND_THRESHOLD`, `REGIME_SHORT_WEIGHT` | How the bot reads market conditions |
| **Execution** | `TRAILING_STOP_ATR_TRENDING`, `TAKE_PROFIT_ATR_*`, `MIN_HOLD_BARS_*` | Stop placement and holding periods |
| **PPO** | `PPO_LEARNING_RATE`, `PPO_ENT_COEF`, `SORTINO_WEIGHT` | RL training behavior |
| **Tuning** | `GEMINI_TUNING_HOUR`, `TUNING_PCT_BOUND_*` | Self-tuning schedule and safety bounds |

The Gemini tuner can adjust most of these within clamped bounds (typically +/-15-25% of current value). Every change is logged with full context so you can trace why any parameter changed.

---

## Monitoring

- **`nyse_bot.log`** — Main log file. Search for `[CVaR]`, `[REGIME]`, `[CAUSAL]`, `[TRACKER]` prefixes to filter by subsystem.
- **TensorBoard** — `tensorboard --logdir ppo_tensorboard` for training curves, reward progression, and policy entropy.
- **Structured Gemini logs** — Search for `"event": "gemini_change"` in the log to see every parameter adjustment with before/after values and performance context.
- **Order tracker** — `order_tracker.json` has the current state of all active positions.
- **Heartbeat** — The bot logs a heartbeat with portfolio status every 60 seconds during market hours.

---

## Troubleshooting

| Symptom | What's Happening | Fix |
|---------|-----------------|-----|
| "Insufficient features for stacking" | Not enough historical bars loaded | Wait for cache to fill (~300 bars) or increase `lookback_days` |
| "CVaR fell back to parity" | Not enough overlapping returns for optimization | Need 50+ bars per symbol; check data fetch |
| "HMM ensemble failed — falling back to Hurst" | Data too flat or too few bars for HMM | Normal for low-vol periods; Hurst fallback works fine |
| "Causal buffer insufficient" | Replay buffer needs ~1500 transitions | Normal at startup; causal penalties disabled until buffer fills |
| Stream keeps reconnecting | Alpaca websocket dropped | Check API keys, network; bot auto-reconnects with exponential backoff |
| "Failed to check active orders" | Alpaca API rate limit or network blip | Logged as warning; retries next cycle automatically |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_regime.py -v
pytest tests/test_risk.py -v
pytest tests/test_features.py -v
pytest tests/test_backtest_integration.py -v
```

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

> *"The market is a device for transferring money from the impatient to the patient."*
> — Warren Buffett

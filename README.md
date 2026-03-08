# nyse-trading-bot
![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen)
![Mode](https://img.shields.io/badge/Mode-Paper_Trading-blue)
![Tech](https://img.shields.io/badge/Tech-Causal_RL_%2B_PPO-purple)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-Private-red)

**Autonomous Portfolio PPO + Causal RL NYSE Trading Bot**  
*March 2026 Edition*

A fully autonomous, production-grade algorithmic trading system for the NYSE.  
Combines **Recurrent PPO** (portfolio-level with AuxGTrXLPolicy), **Causal Reinforcement Learning** (full stacked feature matrix + symbol_id), dynamic regime detection (HMM ensemble + rolling window + persistence scoring), stacking ensembles, real-time local LLM sentiment (Ollama), and daily Gemini self-tuning.

The bot runs 24/7, makes intelligent regime-aware decisions, **ratchets take-profit and trailing stops even during min-hold**, rotates its universe weekly, and continuously improves via structured Gemini logging.

---

## ✨ Key Features (March 2026)

### Core Intelligence
- **Portfolio-level Recurrent PPO** with AuxGTrXLPolicy + VecNormalize
- **Full Causal RL layer** — multi-symbol stacked feature matrix, GES discovery, DoWhy refutation, replay-buffer counterfactual penalties
- **Dynamic regime detection** — HMM ensemble + rolling window (short/long blend), persistence scoring, recent-return overrides
- **Agent-ready architecture** — ready for future reasoning layer

### Trading & Risk (Major Upgrades)
- **Regime confidence symmetry** — persistence now scales both per-symbol sizing **and** portfolio CVaR risk budget
- **Dynamic position sizing floor** (`REGIME_CONFIDENCE_MIN_SIZE_PCT`) — prevents zero-size positions on marginal regimes
- **TP + Trailing Stop ratcheting active during min-hold** — captures unrealized gains without violating min-hold rules
- CVaR optimization with volatility + drawdown penalties
- Regime-specific min-hold (24 or 48 bars) + breakout boost
- Realistic execution (slippage, commissions, bracket re-attach, buying-power clamps)

### Automation & Self-Improvement
- **Daily Gemini 2.5 Flash tuning at 3:30 AM ET** (55+ parameters)
- **Structured JSON logging** for every Gemini change (per-parameter + batch summary) — full institutional memory
- Daily causal graph refresh + PPO nightly retrain
- Weekly universe rotation with liquidity + diversification filters
- Persistent state (`regime_cache.json`, `last_entry_times.json`, `dynamic_config.json`)

### Observability
- Transparent signal blending (`DEBUG_SIGNAL_BLEND=True`)
- `explain_signal_breakdown()` for full auditability
- TensorBoard support + comprehensive logging (`nyse_bot.log`)
- Live Alpaca heartbeat monitor every 60 seconds

---

## 🛠 Tech Stack
- **Language**: Python 3.11
- **RL**: Stable-Baselines3 + sb3-contrib (RecurrentPPO), PyTorch
- **Causal**: DoWhy, pgmpy (GES), networkx
- **ML**: LightGBM stacking, finBERT sentiment
- **LLM**: Ollama (`sentiment-70b` primary)
- **Data**: ArcticDB + Polygon + Alpaca
- **Broker**: Alpaca-py (paper trading)
- **Tuning**: Gemini 2.5 Flash + structured JSON logging

---

## 📁 Project Structure
```bash
nyse-trading-bot/
├── bot.py                    # Main TradingBot class & event loop
├── config.py                 # Pydantic-validated CONFIG
├── __main__.py               # Entry point
├── requirements.txt
├── .env.example
├── .gitignore
│
├── strategy/
│   ├── signals.py            # SignalGenerator + regime confidence
│   ├── regime.py             # Rolling window + HMM
│   └── risk.py               # Position sizing + CVaR (persistence symmetry)
│
├── broker/
│   └── alpaca.py             # Bracket orders + TP ratcheting during min-hold
│
├── models/
│   ├── trainer.py
│   ├── portfolio_env.py
│   ├── causal_signal_manager.py
│   └── agentic_reasoner.py   # (reserved for future upgrade)
│
├── gemini_tuner.py           # Structured JSON logging + tuning
├── backtest.py
└── utils/local_llm.py
```

---

## 🚀 Quick Start
```bash
git clone https://github.com/mmccarney370/nyse-trading-bot.git
cd nyse-trading-bot

python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
cp .env.example .env            # Add your keys
```

```bash
# Pull Ollama models
ollama pull sentiment-70b
ollama pull llama3.1:8b

# Run the bot
python -m nyse_bot
```

---

## ⚙️ Configuration Highlights (March 2026)
Key new/updated settings in `config.py`:

```python
REGIME_SHORT_LOOKBACK: int = 96
REGIME_SHORT_WEIGHT: float = 0.6
REGIME_CONFIDENCE_MIN_SIZE_PCT: float = 0.3   # New floor for weak regimes
```

All other settings remain in `config.py` (type-safe via Pydantic).

---

## 📊 Logging & Monitoring
- Main log: `nyse_bot.log`
- **Structured Gemini logs**: Search for `"event": "gemini_change"` or `"event": "gemini_tuning_batch"`
- TensorBoard: `tensorboard --logdir ppo_tensorboard`
- Live heartbeat every 60 seconds

---

## 🛠 Troubleshooting
| Issue                              | Solution |
|------------------------------------|--------|
| Gemini structured logs missing     | Restart after 3:30 AM run |
| TP not ratcheting during min-hold  | Confirmed fixed in `alpaca.py` |
| Causal buffer warning              | Normal until ~1500 trades |
| Pipeline GPU warning               | Batch finBERT calls (see signals.py) |

---

## ⚠️ Disclaimer
This bot is for **educational and paper-trading use only**.  
Past performance ≠ future results. Use at your own risk.

---

**Built for autonomy.**  
*Turning markets into code since February 2026*

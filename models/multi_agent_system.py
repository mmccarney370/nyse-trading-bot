# models/multi_agent_system.py
import logging
import numpy as np
import pandas as pd
from config import CONFIG
from utils.local_llm import LocalLLMDebate
from strategy.regime import detect_regime

logger = logging.getLogger(__name__)

class MultiAgentSystem:
    """
    Lightweight hierarchical multi-agent system (no Ray RLlib).
    Regime Agent → Signal Agents (with LLM debate) → Execution Agent.
    """
    def __init__(self, data_ingestion, symbols):
        self.data_ingestion = data_ingestion
        self.symbols = symbols
        self.llm_debate = LocalLLMDebate() if CONFIG.get('USE_AGENT_DEBATE', True) else None

    def get_regime(self, data_dict: dict, timestamp=None) -> str:
        """Regime Agent: decides overall market regime"""
        # Aggregate across symbols for macro view
        closes = []
        for sym, df in data_dict.items():
            if timestamp and timestamp in df.index:
                closes.append(df.loc[timestamp]['close'])
            elif not df.empty:
                closes.append(df['close'].iloc[-1])
        if not closes:
            return 'mean_reverting'
        agg_close = np.mean(closes)
        # Simple but effective regime decision using shared function
        dummy_df = pd.DataFrame({'close': [agg_close] * 100})
        regime_tuple = detect_regime(dummy_df, symbol='portfolio', data_ingestion=self.data_ingestion, lookback=100)
        regime = regime_tuple[0] if isinstance(regime_tuple, (list, tuple)) else regime_tuple
        logger.debug(f"[Regime Agent] Decided regime: {regime}")
        return regime

    def get_signals(self, data_dict: dict, timestamp=None) -> dict:
        """Signal Agents: per-symbol signals with optional LLM debate"""
        signals = {}
        regime = self.get_regime(data_dict, timestamp)

        for sym in self.symbols:
            data = data_dict.get(sym)
            if data is None or data.empty:
                signals[sym] = 0.0
                continue

            # Get base signal from existing PPO or stacking (you already have this logic)
            # For now, placeholder — replace with your actual signal generation
            base_signal = 0.5  # Replace with real PPO output or stacking prob

            if self.llm_debate and CONFIG.get('USE_AGENT_DEBATE', True):
                prompt = f"Debate trading signal for {sym} in {regime} regime. Current base signal: {base_signal:.3f}. "
                prompt += "Respond with a single number from -1.0 (strong short) to 1.0 (strong long)."
                debate_score = self.llm_debate.debate_sentiment([prompt])
                final_signal = base_signal * (debate_score + 1) / 2  # Blend
                logger.debug(f"[Signal Agent {sym}] Base: {base_signal:.3f} → Debate: {debate_score:.3f} → Final: {final_signal:.3f}")
            else:
                final_signal = base_signal

            signals[sym] = final_signal

        return signals

    def execute(self, signals: dict, current_equity: float) -> dict:
        """Execution Agent: converts signals to target weights"""
        total_abs = sum(abs(v) for v in signals.values())
        if total_abs == 0:
            return {sym: 0.0 for sym in signals}

        max_leverage = CONFIG.get('MAX_LEVERAGE', 2.5)
        scale = max_leverage / total_abs if total_abs > max_leverage else 1.0

        target_weights = {sym: signals[sym] * scale for sym in signals}
        return target_weights

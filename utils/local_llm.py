import ollama
import asyncio
import logging
import re # Explicit import for robust number extraction
import numpy as np # P-8 FIX: required for np.isnan to distinguish real neutral (0.0) from failure
from config import CONFIG
logger = logging.getLogger(__name__)
class LocalLLMDebate:
    def __init__(self):
        self.model = CONFIG['LOCAL_LLM_MODEL']
        self.fallback = CONFIG['LOCAL_LLM_FALLBACK']
        # BUG-9 FIX: Read from CONFIG / .env instead of hardcoding localhost
        self.host = CONFIG.get('OLLAMA_HOST', 'http://localhost:11434')
        # P-1 FIX: Proper Ollama Client with host support (ollama.chat() does NOT accept host=)
        self.client = ollama.Client(host=self.host)
        logger.info(f"LocalLLMDebate initialized with Ollama host: {self.host}")
    def _call_ollama(self, model: str, prompt: str) -> float:
        """Synchronous low-level Ollama call — kept exactly as in your original file."""
        try:
            response = self.client.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.2, 'num_predict': 32, 'top_p': 0.9,}
                # host= removed — now handled by self.client
            )
            text = response['message']['content'].strip()
            # Robust number extraction (same as before)
            match = re.search(r'-?\d+\.?\d*', text)
            score = float(match.group(0)) if match else 0.0
            return max(min(score, 1.0), -1.0)
        except Exception as e:
            logger.debug(f"Ollama {model} failed on host {self.host}: {e}")
            return float('nan') # P-8 FIX: Use NaN sentinel for real failures (valid neutral score 0.0 is no longer treated as failure)
    async def debate_sentiment(self, news_texts: list) -> float:
        """Async version — runs every ollama.chat() in a thread pool so it never blocks the trading event loop.
        Called from signals.py / causal_wrapper.py with 'await' (next step)."""
        if not news_texts:
            return 0.0
        headlines = ". ".join(news_texts[:15])
        opinions = []
        for role in ["bull", "bear", "analyst"]:
            prompt = f"You are a {role} trader. Analyze sentiment of these headlines for the stock: {headlines}\nRespond ONLY with a number from -1 (very negative) to 1 (very positive)."
         
            # CRITICAL FIX: Run blocking Ollama call in thread pool — does NOT block async loop
            score = await asyncio.to_thread(self._call_ollama, self.model, prompt)
         
            # P-8 FIX: Only fallback on actual failure (NaN), not on valid neutral score (0.0)
            if np.isnan(score):
                score = await asyncio.to_thread(self._call_ollama, self.fallback, prompt)
         
            opinions.append(score)
            logger.debug(f"Local LLM {role} score: {score:.3f}")
        final = sum(opinions) / len(opinions)
        logger.info(f"Local LLM debate final: {final:.3f} ({len(news_texts)} headlines)")
        return final

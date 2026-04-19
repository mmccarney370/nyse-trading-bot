import ollama
import asyncio
import logging
import re # Explicit import for robust number extraction
import numpy as np # P-8 FIX: required for np.isnan to distinguish real neutral (0.0) from failure
from config import CONFIG
logger = logging.getLogger(__name__)
class LocalLLMDebate:
    # Serialize all Ollama calls — 70B model can only run one inference at a time.
    # Without this, 8 symbols fire concurrently, queue behind the GPU, and timeout.
    # Lazy-init: asyncio.Lock() must be created inside a running event loop.
    _ollama_lock = None

    def __init__(self):
        self.model = CONFIG['LOCAL_LLM_MODEL']
        self.fallback = CONFIG['LOCAL_LLM_FALLBACK']
        self.host = CONFIG.get('OLLAMA_HOST', 'http://localhost:11434')
        self.client = ollama.Client(host=self.host, timeout=180)  # Was 120 — 70B needs more headroom
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
            # Robust number extraction: prefer numbers in [-1, 1] range
            all_numbers = re.findall(r'-?\d+\.?\d*', text)
            if not all_numbers:
                return 0.0
            # Prefer numbers that fall within the expected sentiment range [-1, 1]
            in_range = [n for n in all_numbers if -1.0 <= float(n) <= 1.0]
            if in_range:
                # Take the last in-range number (LLMs typically state the answer last)
                score = float(in_range[-1])
            else:
                # M21 FIX: Log when clamping out-of-range values (e.g. "3 out of 5" → 3 → clamped to 1)
                raw = float(all_numbers[-1])
                # Try to detect "X out of Y" patterns and normalize
                if abs(raw) <= 10:
                    score = raw / 5.0 if abs(raw) > 1.0 else raw  # assume 5-point scale
                else:
                    score = raw / 100.0 if abs(raw) > 1.0 else raw  # assume percentage
                logger.debug(f"[LLM] Out-of-range number {raw} → normalized to {score:.2f}")
            return max(min(score, 1.0), -1.0)
        except Exception as e:
            logger.debug(f"Ollama {model} failed on host {self.host}: {e}")
            return float('nan') # P-8 FIX: Use NaN sentinel for real failures (valid neutral score 0.0 is no longer treated as failure)
    async def debate_sentiment(self, news_texts: list, symbol: str = None) -> float:
        """Async version — serialized via _ollama_lock so 70B model processes one
        debate at a time (prevents timeout from 8 symbols queuing on GPU).
        Each debate still runs 3 agents (bull/bear/analyst) sequentially.

        FIX (Apr 17): Added per-debate timeout. If Ollama hangs (GPU crash, OOM,
        server freeze), the lock is released after timeout so the next symbol can
        proceed. Without this, a single hung inference permanently freezes the
        entire sentiment pipeline and blocks the trading loop."""
        if not news_texts:
            return 0.0
        headlines = " | ".join(news_texts[:15])
        stock_desc = f" for {symbol}" if symbol else ""
        # Serialize across all symbols — 70B can only do one inference at a time
        if LocalLLMDebate._ollama_lock is None:
            LocalLLMDebate._ollama_lock = asyncio.Lock()
        # Per-debate timeout: if one symbol's debate hangs, release lock after N seconds
        # so remaining symbols aren't permanently blocked. 120s = ~40s/role × 3 roles.
        # FIX (Apr 17): without this, a hung Ollama call held _ollama_lock forever,
        # freezing the entire sentiment pipeline and blocking the trading loop for 5+ hours.
        _debate_timeout = 120

        async def _run_debate():
            async with LocalLLMDebate._ollama_lock:
                opinions = []
                for role in ["bull", "bear", "analyst"]:
                    prompt = f"You are a {role} trader. Analyze sentiment of these headlines{stock_desc}: {headlines}\nRespond ONLY with a number from -1 (very negative) to 1 (very positive)."
                    score = await asyncio.to_thread(self._call_ollama, self.model, prompt)
                    if np.isnan(score):
                        score = await asyncio.to_thread(self._call_ollama, self.fallback, prompt)
                    if np.isnan(score):
                        logger.warning(f"Local LLM {role} returned NaN even after fallback — skipping")
                        continue
                    opinions.append(score)
                    logger.debug(f"Local LLM {role} score: {score:.3f}")
                if not opinions:
                    # Apr-19 audit: on complete LLM failure return NaN so the
                    # sentiment blender in signals.py can SKIP blending
                    # entirely. Returning 0.0 previously made "LLM broken"
                    # indistinguishable from "genuinely neutral sentiment"
                    # and let a silent failure corrupt target_weights.
                    logger.warning(f"All LLM opinions were NaN for {symbol} — returning NaN sentinel")
                    return float('nan')
                final = sum(opinions) / len(opinions)
                logger.info(f"Local LLM debate final: {final:.3f} ({len(news_texts)} headlines) [{symbol}]")
                return final

        try:
            return await asyncio.wait_for(_run_debate(), timeout=_debate_timeout)
        except (asyncio.TimeoutError, TimeoutError):
            # Apr-19 audit: NaN sentinel, not 0.0. See note above.
            logger.warning(f"[OLLAMA TIMEOUT] {symbol} debate exceeded {_debate_timeout}s — returning NaN sentinel")
            return float('nan')

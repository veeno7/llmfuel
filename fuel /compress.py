# fuel/compress.py
"""
llmfuel — compress module
Live semantic deduplication of chain-of-thought steps.

Stub interface — DeepSeek will implement the classifier.
Default model: Gemma 3 270M INT4 (~40-70ms/step on CPU)
Pi fallback preset: MiniLM-v2-L6 (<10ms, no GPU needed)
"""

from __future__ import annotations
from typing import Optional

# Preset configs — DeepSeek: fill in model_id and similarity_threshold
PRESETS: dict[str, dict] = {
    "default": {
        "model_id": "google/gemma-3-270m",      # INT4 quantized
        "similarity_threshold": 0.92,
        "max_tokens_per_step": 512,
    },
    "pi": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "similarity_threshold": 0.90,
        "max_tokens_per_step": 256,
    },
}


class CoTDeduper:
    """
    Semantic deduplicator for chain-of-thought reasoning steps.

    Usage:
        deduper = CoTDeduper()                    # Gemma 3 270M INT4
        deduper = CoTDeduper(preset="pi")         # MiniLM-v2-L6 for Raspberry Pi
        compressed = deduper.dedup(steps)

    Args:
        model:   HuggingFace model ID. Overrides preset if set.
        preset:  "default" (Gemma 3 270M) or "pi" (MiniLM-v2-L6).
        verbose: Log which steps were dropped.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        preset: str = "default",
        verbose: bool = False,
    ):
        cfg = PRESETS.get(preset, PRESETS["default"])
        self.model = model or cfg["model_id"]
        self.threshold = cfg["similarity_threshold"]
        self.verbose = verbose
        self._encoder = None   # lazy-loaded on first dedup() call

    def _load_encoder(self):
        # TODO DeepSeek: swap for INT4-quantized Gemma 3 270M loader
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(self.model)

    def dedup(self, steps: list[str]) -> list[str]:
        """
        Remove semantically duplicate CoT steps.

        TODO DeepSeek:
            1. Embed all steps with self._encoder
            2. Compute cosine similarity matrix
            3. Drop steps[i] where max sim to any earlier step >= self.threshold
            4. Integrate with fuel.receipts: record input/output token counts per drop
        """
        if self._encoder is None:
            self._load_encoder()
        # Placeholder: pass-through until DeepSeek wires the classifier
        return steps

    def compression_ratio(self, original: list[str], compressed: list[str]) -> float:
        """Utility: token-count compression ratio (approx by whitespace split)."""
        orig_tokens = sum(len(s.split()) for s in original)
        comp_tokens = sum(len(s.split()) for s in compressed)
        if orig_tokens == 0:
            return 1.0
        return round(comp_tokens / orig_tokens, 4)

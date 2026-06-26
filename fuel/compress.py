"""
fuel/compress.py — compress module
Live semantic deduplication of chain-of-thought steps.

Stub interface — DeepSeek will implement the classifier.
Default model: Gemma 3 270M INT4 (~40-70ms/step on CPU)
Pi fallback preset: MiniLM-v2-L6 (<10ms, no GPU needed)
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .receipts import ReceiptChain

PRESETS: dict[str, dict] = {
    "default": {
        "model_id": "google/gemma-3-270m", # INT4 quantized — future
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
    def __init__(
        self,
        model: Optional[str] = None,
        preset: str = "default",
        verbose: bool = False,
        receipts: Optional["ReceiptChain"] = None,
    ):
        cfg = PRESETS.get(preset, PRESETS["default"])
        self.model = model or cfg["model_id"]
        self.threshold = cfg["similarity_threshold"]
        self.verbose = verbose
        self.receipts = receipts
        self._encoder = None

    def _load_encoder(self):
        # v0.1: use MiniLM for both presets (Gemma 3 270M INT4 coming in v0.2)
        from sentence_transformers import SentenceTransformer
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self._encoder = SentenceTransformer(model_id)

    def dedup(self, steps: list[str]) -> list[str]:
        if not steps:
            return []
        if self._encoder is None:
            self._load_encoder()

        embeddings = self._encoder.encode(
            steps, normalize_embeddings=True, show_progress_bar=False
        )
        kept_idx = [0]
        kept_embs = [embeddings[0]]

        for i in range(1, len(steps)):
            sims = embeddings[i] @ np.stack(kept_embs, axis=1)
            if sims.max() < self.threshold:
                kept_idx.append(i)
                kept_embs.append(embeddings[i])
            elif self.receipts:
                self.receipts.record(
                    action="cot_dedup",
                    input_data=steps[i],
                    output_data="",
                    input_tokens=len(steps[i].split()),
                    output_tokens=0,
                )
                if self.verbose:
                    print(f"[fuel] dropped step {i} (sim={sims.max():.2f})")

        return [steps[i] for i in kept_idx]

    def compression_ratio(self, original: list[str], compressed: list[str]) -> float:
        orig_tokens = sum(len(s.split()) for s in original)
        comp_tokens = sum(len(s.split()) for s in compressed)
        if orig_tokens == 0:
            return 1.0
        return round(comp_tokens / orig_tokens, 4)
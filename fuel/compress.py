"""
fuel/compress.py — compress module
Live semantic deduplication of chain-of-thought steps.

Stub interface — DeepSeek will implement the classifier.
Default model: Gemma 3 270M INT4 (~40-70ms/step on CPU)
Pi fallback preset: MiniLM-v2-L6 (<10ms, no GPU needed)
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .receipts import ReceiptChain

_gemma_model: Optional[Any] = None


def _load_gemma():
    global _gemma_model
    if _gemma_model is not None:
        return _gemma_model

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("llama_cpp is not installed") from exc

    model_path = Path("models/gemma-3-270m-it-int4.gguf")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Gemma GGUF model not found at {model_path}. "
            "Install it in the models/ directory or use preset='pi' instead."
        )

    _gemma_model = Llama(
        model_path=str(model_path),
        embedding=True,
        n_ctx=512,
        verbose=False,
        n_gpu_layers=0,
    )
    return _gemma_model


def _embed_gemma(texts: list[str]) -> np.ndarray:
    model = _load_gemma()
    embs = [model.create_embedding(t)["data"]["embedding"] for t in texts]
    arr = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def _embed_minilm(texts: list[str], model: Any) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

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
        self.preset = preset
        self._encoder: Optional[Any] = None
        self._embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None

    def _load_encoder(self):
        if self.preset == "default":
            try:
                _load_gemma()
                self._embed_fn = _embed_gemma
                return
            except Exception as exc:
                if self.verbose:
                    print(f"[fuel] warning: Gemma fallback to MiniLM: {exc}")
                self.preset = "pi"

        if self.preset == "pi":
            from sentence_transformers import SentenceTransformer

            model_id = "sentence-transformers/all-MiniLM-L6-v2"
            self._encoder = SentenceTransformer(model_id)
            self._embed_fn = lambda texts: _embed_minilm(texts, self._encoder)
            return

        raise RuntimeError(f"Unknown preset: {self.preset}")

    def dedup(self, steps: list[str], return_kept_idx: bool = False):
        if not steps:
            return ([], []) if return_kept_idx else []
        if self._embed_fn is None:
            self._load_encoder()

        embeddings = self._embed_fn(steps)
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

        kept_steps = [steps[i] for i in kept_idx]
        return (kept_steps, kept_idx) if return_kept_idx else kept_steps

    def compression_ratio(self, original: list[str], compressed: list[str]) -> float:
        orig_tokens = sum(len(s.split()) for s in original)
        comp_tokens = sum(len(s.split()) for s in compressed)
        if orig_tokens == 0:
            return 1.0
        return round(comp_tokens / orig_tokens, 4)
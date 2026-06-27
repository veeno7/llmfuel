"""
fuel/compress.py — compress module
Live semantic deduplication of chain-of-thought steps.

Stub interface — DeepSeek will implement the classifier.
Default model: Gemma 3 270M INT4 (~40-70ms/step on CPU)
Pi fallback preset: MiniLM-v2-L6 (<10ms, no GPU needed)
"""

from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING
import numpy as np
from difflib import SequenceMatcher

from .model_utils import ensure_gemma_model

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

    model_candidates = [
        Path("models/gemma-3-270m-it-q4_k_m.gguf"),
        Path("models/gemma-3-270m-it-int4.gguf"),
    ]
    model_path = next((p for p in model_candidates if p.exists()), None)
    if model_path is None:
        try:
            model_path = ensure_gemma_model()
        except Exception as exc:
            raise FileNotFoundError(
                "Gemma GGUF model not found. Install it in the models/ directory or use preset='pi' instead."
            ) from exc

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
    embs = []
    for t in texts:
        payload = model.create_embedding(t)
        data = payload.get("data", [])
        if isinstance(data, list) and data:
            embedding = data[0].get("embedding")
        elif isinstance(data, dict):
            embedding = data.get("embedding")
        else:
            raise ValueError(f"Unexpected Gemma embedding payload: {payload}")

        if isinstance(embedding, list):
            embs.append(np.array(embedding, dtype=np.float32))
        else:
            embs.append(np.array([float(embedding)], dtype=np.float32))

    arr = np.vstack(embs) if embs else np.empty((0, 0), dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def _embed_minilm(texts: list[str], model: Any) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _embed_simple(texts: list[str]) -> np.ndarray:
    tokens = [re.findall(r"\w+", text.lower()) for text in texts]
    vocab = sorted({token for row in tokens for token in row})

    vectors = []
    for row in tokens:
        counts = Counter(row)
        vec = np.array([counts.get(token, 0) for token in vocab], dtype=np.float32)
        norm = np.linalg.norm(vec)
        vectors.append(vec / norm if norm else vec)

    return np.vstack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)

PRESETS: dict[str, dict] = {
    "default": {
        "model_id": "google/gemma-3-270m", # INT4 quantized — future
        "similarity_threshold": 0.86,
        "max_tokens_per_step": 512,
    },
    "pi": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "similarity_threshold": 0.78,
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
        aggressiveness: str = "balanced",
    ):
        cfg = PRESETS.get(preset, PRESETS["default"])
        self.model = model or cfg["model_id"]
        self.threshold = cfg["similarity_threshold"]
        self.verbose = verbose
        self.receipts = receipts
        self.preset = preset
        self.aggressiveness = aggressiveness
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
                    print(f"[fuel] warning: Gemma fallback to lexical dedup: {exc}")
                self.preset = "pi"

        if self.preset == "pi":
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                if self.verbose:
                    print(f"[fuel] warning: MiniLM unavailable, using lexical fallback: {exc}")
                self._embed_fn = _embed_simple
                return

            model_id = "sentence-transformers/all-MiniLM-L6-v2"
            try:
                self._encoder = SentenceTransformer(model_id)
                self._embed_fn = lambda texts: _embed_minilm(texts, self._encoder)
                return
            except Exception as exc:
                if self.verbose:
                    print(f"[fuel] warning: MiniLM load failed, using lexical fallback: {exc}")
                self._embed_fn = _embed_simple
                return

        raise RuntimeError(f"Unknown preset: {self.preset}")

    def dedup(self, steps: list[str], return_kept_idx: bool = False):
        if not steps:
            return ([], []) if return_kept_idx else []
        if self._embed_fn is None:
            self._load_encoder()

        embeddings = np.asarray(self._embed_fn(steps), dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        kept_idx = [0]
        kept_embs = [embeddings[0]]

        for i in range(1, len(steps)):
            current = steps[i].strip().lower()
            previous = steps[kept_idx[-1]].strip().lower()
            current_tokens = set(re.findall(r"\w+", current))
            previous_tokens = set(re.findall(r"\w+", previous))
            overlap = len(current_tokens & previous_tokens) / max(1, len(current_tokens | previous_tokens))
            lexical_ratio = SequenceMatcher(None, current, previous).ratio()
            reasoning_words = {"solve", "add", "calculate", "compute", "check", "reason", "step", "think", "consider", "first", "then", "next", "now", "again", "also"}
            conclusion_words = {"answer", "final", "therefore", "thus", "so", "conclude", "conclusion", "result", "overall", "summary"}
            has_reasoning = bool(current_tokens & reasoning_words) and bool(previous_tokens & reasoning_words)
            has_conclusion = bool(current_tokens & conclusion_words) or bool(previous_tokens & conclusion_words)
            repeated_reasoning = (
                has_reasoning
                and not has_conclusion
                and (overlap >= 0.10 or len(current_tokens & previous_tokens) >= 1 or lexical_ratio >= 0.55)
            )
            semantic_similarity = 0.0
            if embeddings.size and embeddings.shape[0] > i and embeddings.shape[0] > kept_idx[-1]:
                prev_emb = embeddings[kept_idx[-1]]
                curr_emb = embeddings[i]
                if prev_emb.ndim == 1 and curr_emb.ndim == 1:
                    semantic_similarity = float(np.dot(prev_emb, curr_emb))
            semantic_duplicate = semantic_similarity >= self.threshold and not has_conclusion
            short_repetition = len(current_tokens) <= 4 and len(previous_tokens) <= 4 and len(current_tokens & previous_tokens) >= 1
            aggressive = self.aggressiveness in {"aggressive", "max"}
            threshold_overlap = 0.5 if aggressive else 0.4
            threshold_ratio = 0.8 if aggressive else 0.75
            is_duplicate = (
                current == previous
                or current in previous
                or previous in current
                or overlap >= threshold_overlap
                or lexical_ratio >= threshold_ratio
                or repeated_reasoning
                or semantic_duplicate
                or short_repetition
            )
            if is_duplicate:
                if self.receipts:
                    self.receipts.record(
                        action="cot_dedup",
                        input_data=steps[i],
                        output_data="",
                        input_tokens=len(steps[i].split()),
                        output_tokens=0,
                    )
                if self.verbose:
                    reason = "semantic" if semantic_duplicate else "lexical"
                    print(f"[fuel] dropped step {i} ({reason} duplicate, ratio={lexical_ratio:.2f}, sim={semantic_similarity:.2f})")
                continue

            kept_idx.append(i)
            kept_embs.append(embeddings[i])

        kept_steps = [steps[i] for i in kept_idx]
        return (kept_steps, kept_idx) if return_kept_idx else kept_steps

    def compression_ratio(self, original: list[str], compressed: list[str]) -> float:
        orig_tokens = sum(len(s.split()) for s in original)
        comp_tokens = sum(len(s.split()) for s in compressed)
        if orig_tokens == 0:
            return 1.0
        return round(comp_tokens / orig_tokens, 4)
# fuel/adapters/ollama.py
"""
llmfuel — Ollama adapter
Wraps an Ollama client so every streamed CoT response is deduplicated
and receipted automatically.
"""

from __future__ import annotations
import re
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fuel.compress import CoTDeduper
    from fuel.receipts import ReceiptChain

_THINK_TAG = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def wrap_ollama(
    client: Any,
    deduper: "CoTDeduper",
    receipts: Optional["ReceiptChain"] = None,
) -> Any:
    """
    Return a wrapped Ollama client whose .generate() / .chat() calls
    automatically deduplicate CoT steps and log receipts.
    """

    class WrappedClient:
        def __init__(self, client: Any, deduper: "CoTDeduper", receipts: Optional["ReceiptChain"]):
            self._client = client
            self._deduper = deduper
            self._receipts = receipts

        def __getattr__(self, name: str) -> Any:
            return getattr(self._client, name)

        def generate(self, *args: Any, **kwargs: Any) -> Any:
            response = self._client.generate(*args, **kwargs)
            return _process_response(response, self._deduper, self._receipts)

        def chat(self, *args: Any, **kwargs: Any) -> Any:
            response = self._client.chat(*args, **kwargs)
            return _process_response(response, self._deduper, self._receipts)

    def _extract_think_steps(text: str) -> list[str]:
        return [match.group(1).strip() for match in _THINK_TAG.finditer(text)]

    def _rebuild_text(original: str, compressed_steps: list[str], kept_idx: list[int]) -> str:
        kept_iter = iter(compressed_steps)
        idx = [0]

        def replacer(match: re.Match) -> str:
            value = ""
            if idx[0] in kept_idx:
                value = "<think>" + next(kept_iter) + "</think>"
            idx[0] += 1
            return value

        return _THINK_TAG.sub(replacer, original)

    def _process_response(response: Any, deduper: "CoTDeduper", receipts: Optional["ReceiptChain"]):
        if isinstance(response, str):
            text = response
        elif hasattr(response, "text"):
            text = response.text
        elif isinstance(response, dict) and "output" in response:
            text = response.get("output", "")
        else:
            return response

        steps = _extract_think_steps(text)
        if not steps:
            return response

        if receipts is not None and deduper.receipts is None:
            deduper.receipts = receipts

        compressed, kept_idx = deduper.dedup(steps, return_kept_idx=True)
        new_text = _rebuild_text(text, list(compressed), kept_idx)

        if isinstance(response, str):
            return new_text
        if hasattr(response, "text"):
            setattr(response, "text", new_text)
            return response
        if isinstance(response, dict) and "output" in response:
            response["output"] = new_text
            return response

        return response

    return WrappedClient(client, deduper, receipts)

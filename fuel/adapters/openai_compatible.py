from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fuel.compress import CoTDeduper
    from fuel.receipts import ReceiptChain


class OpenAICompatibleAdapter:
    """Thin adapter for OpenAI-compatible APIs that deduplicates repeated reasoning content."""

    def __init__(
        self,
        client: Any,
        deduper: Optional["CoTDeduper"] = None,
        receipts: Optional["ReceiptChain"] = None,
    ):
        self._client = client
        self._deduper = deduper
        self._receipts = receipts

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def chat_completions_create(self, *args: Any, **kwargs: Any) -> Any:
        response = self._client.chat.completions.create(*args, **kwargs)
        return self._postprocess_response(response)

    def completions_create(self, *args: Any, **kwargs: Any) -> Any:
        response = self._client.completions.create(*args, **kwargs)
        return self._postprocess_response(response)

    def _postprocess_response(self, response: Any) -> Any:
        if response is None:
            return response
        if hasattr(response, "choices") and getattr(response, "choices", None):
            for choice in response.choices:
                if hasattr(choice, "message") and getattr(choice.message, "content", None):
                    text = str(choice.message.content)
                    compressed = self._dedup_text(text)
                    choice.message.content = compressed
                elif hasattr(choice, "text") and getattr(choice, "text", None):
                    text = str(choice.text)
                    compressed = self._dedup_text(text)
                    choice.text = compressed
        return response

    def _dedup_text(self, text: str) -> str:
        if not text:
            return text
        deduper = self._deduper or self._get_default_deduper()
        steps = [segment.strip() for segment in text.splitlines() if segment.strip()]
        if len(steps) <= 1:
            return text
        compressed = deduper.dedup(steps)
        return "\n".join(compressed)

    def _get_default_deduper(self) -> "CoTDeduper":
        from fuel import CoTDeduper

        return CoTDeduper()

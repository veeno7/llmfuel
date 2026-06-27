from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fuel.compress import CoTDeduper
    from fuel.receipts import ReceiptChain


def wrap_api_response(
    response: Any,
    deduper: "CoTDeduper",
    receipts: Optional["ReceiptChain"] = None,
) -> Any:
    """Apply CoT deduplication to generic API response objects or dict payloads."""
    if isinstance(response, str):
        text = response
    elif isinstance(response, dict):
        text = response.get("text") or response.get("content") or response.get("output") or ""
    elif hasattr(response, "text"):
        text = getattr(response, "text")
    else:
        return response

    if not text:
        return response

    steps = [segment.strip() for segment in text.split("\n") if segment.strip()]
    if len(steps) <= 1:
        return response

    if receipts is not None and deduper.receipts is None:
        deduper.receipts = receipts

    compressed, kept_idx = deduper.dedup(steps, return_kept_idx=True)
    rebuilt = "\n".join(compressed)

    if isinstance(response, dict):
        response["text"] = rebuilt
        return response
    if hasattr(response, "text"):
        setattr(response, "text", rebuilt)
        return response
    return rebuilt

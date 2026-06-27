import re
from typing import Any, Optional, TYPE_CHECKING

from fuel import CoTDeduper

if TYPE_CHECKING:
    from fuel.receipts import ReceiptChain


class _StreamWrapper:
    def __init__(self, client: Any, deduper: Optional[CoTDeduper] = None):
        self._client = client
        self._deduper = deduper or CoTDeduper()

    def chat(self, model: str, messages: list[dict], stream: bool = True, **kwargs: Any):
        if not stream:
            return self._client.chat(model=model, messages=messages, stream=False, **kwargs)

        full = ""
        for chunk in self._client.chat(model=model, messages=messages, stream=True, **kwargs):
            text = chunk["message"]["content"]
            full += text

            if "</think>" in text:
                blocks = re.findall(r"<think>(.*?)</think>", full, flags=re.S)
                if blocks:
                    steps = [block.strip() for block in blocks if block.strip()]
                    kept_steps, kept_idx = self._deduper.dedup(steps, return_kept_idx=True)
                    kept_by_idx = {}
                    kept_iter = iter(kept_steps)
                    for idx in range(len(steps)):
                        if idx in kept_idx:
                            kept_by_idx[idx] = next(kept_iter)

                    counter = [0]

                    def replacer(match: re.Match) -> str:
                        idx = counter[0]
                        counter[0] += 1
                        if idx in kept_by_idx:
                            return "<think>" + kept_by_idx[idx] + "</think>"
                        return ""

                    reconstructed = re.sub(r"<think>.*?</think>", replacer, full, flags=re.S)
                    yield {"message": {"content": reconstructed}}
                    return

            yield chunk


def wrap_ollama_stream(client: Any, deduper: Optional[CoTDeduper] = None) -> Any:
    return _StreamWrapper(client, deduper)

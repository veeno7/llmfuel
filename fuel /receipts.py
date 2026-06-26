# fuel/receipts.py
"""
llmfuel — receipts module
Local-only, hash-chained audit trail for CoT steps.
Privacy: all content hashed by default; opt-in plaintext via store_plaintext=True.
"""

import hashlib, json, time, uuid
from pathlib import Path
from typing import Any, Optional

RECEIPTS_VERSION = "0.1.0"


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _hash_content(content: Any) -> str:
    # Stable serialization — RFC8785 TODO, json.dumps+sort_keys is safe interim
    return _sha256(json.dumps(content, sort_keys=True, ensure_ascii=False))


class ReceiptChain:
    def __init__(
        self,
        run_id: Optional[str] = None,
        agent: str = "unknown",
        principal: str = "local",
        store_plaintext: bool = False,
        output_path: Optional[Path] = None,
    ):
        self.run_id = run_id or str(uuid.uuid4())
        self.agent = agent
        self.principal = principal
        self.store_plaintext = store_plaintext
        self.output_path = output_path
        self._prev_hash: str = "GENESIS"
        self._step: int = 0
        self._chain: list[dict] = []

    def record(
        self,
        action: str,
        input_data: Any,
        output_data: Any,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> dict:
        self._step += 1
        receipt = {
            "id": str(uuid.uuid4()),
            "ts": int(time.time() * 1000),          # epoch-ms integer
            "agent": self.agent,
            "principal": self.principal,
            "action": action,
            "input_hash": _hash_content(input_data),
            "output_hash": _hash_content(output_data),
            "prev_hash": self._prev_hash,
            "ext": {
                "version": RECEIPTS_VERSION,
                "run_id": self.run_id,
                "step_id": self._step,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                **({"input": input_data, "output": output_data} if self.store_plaintext else {}),
            },
        }
        # Hash-chain: next prev_hash covers the full receipt (not just output)
        self._prev_hash = _sha256(json.dumps(receipt, sort_keys=True))
        self._chain.append(receipt)
        if self.output_path:
            self._flush(receipt)
        return receipt

    def _flush(self, receipt: dict) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(receipt) + "\n")   # JSONL — easy to stream/parse

    def verify_chain(self) -> bool:
        """Re-derive all prev_hashes and confirm integrity."""
        prev = "GENESIS"
        for r in self._chain:
            if r["prev_hash"] != prev:
                return False
            prev = _sha256(json.dumps(r, sort_keys=True))
        return True

    def to_agentreceipts_v1(self, receipt: dict) -> dict:
        """Thin adapter: emit agentreceipts.ai-compatible shape (ISO 8601 ts)."""
        from datetime import datetime, timezone
        return {
            **receipt,
            "ts": datetime.fromtimestamp(receipt["ts"] / 1000, tz=timezone.utc).isoformat(),
        }

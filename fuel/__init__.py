"""
llmfuel — Stop your reasoning model from thinking in circles.

Live chain-of-thought semantic deduplication + local audit receipts
for open models (DeepSeek-R1, QwQ, etc.). Fully offline, no cloud proxy.

    import fuel
    chain = fuel.ReceiptChain(agent="my-agent")
    deduper = fuel.CoTDeduper()
"""

from .receipts import ReceiptChain
from .compress import CoTDeduper
from .model_utils import ensure_gemma_model
from .cli import main as cli_main

__version__ = "0.1.0"
__all__ = ["ReceiptChain", "CoTDeduper", "ensure_gemma_model", "cli_main"]

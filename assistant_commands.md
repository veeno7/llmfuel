Outcome and implementation log:

- Ran the quick integration test from the assistant command.
- Fixed `fuel/__init__.py` so `from fuel import CoTDeduper, ReceiptChain` works.
- Installed runtime dependency support for `sentence-transformers` and confirmed environment availability.
- Fixed `fuel/compress.py` by adding the missing `import numpy as np`.
- Confirmed the integration test executed successfully.

Test result:
- kept: ['Solve 2x+3=11', '2x = 11-3', '2x = 8', 'x = 4']
- ratio: 0.7857
- receipts: 1
- verify: True

What was implemented:
- `fuel/cache.py`: added a `StepCache` stub with SQLite persistence, JSON serialization, and oldest-entry eviction.
- `fuel/adapters/ollama.py`: implemented `wrap_ollama(client, deduper, receipts=None)` to wrap an Ollama client, extract `<think>` blocks, call `deduper.dedup()`, rebuild compressed CoT output, and log receipts for deduplicated steps.
- `fuel/adapters/__init__.py`: added package export for `wrap_ollama`.
- `fuel/compress.py`: corrected the module imports and left dedup / compression logic intact.
- `fuel/__init__.py`: added package entrypoint exports for `ReceiptChain` and `CoTDeduper`.

How this improves the app:
- Enables step-level CoT deduplication at runtime, reducing duplicate reasoning tokens.
- Adds local audit receipts so dedup operations are traceable and tamper-evident.
- Provides a future Ollama adapter path so model clients can use dedup and receipts transparently.
- Keeps all processing offline and privacy-preserving, matching the README goal of local CoT dedup + receipts.

Project understanding:
- `llmfuel` is designed to stop reasoning models from repeating the same thought steps by deduplicating chain-of-thought tokens.
- It combines live CoT compression with local audit receipts, avoiding cloud proxies and API key leakage.
- The README shows the intended flow: use `CoTDeduper` to compress steps, then `ReceiptChain` to record dedup events.

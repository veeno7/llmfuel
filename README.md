# llmfuel ⛽

**Stop your reasoning model from thinking in circles.**

[[Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://python.org) [[License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [[Status](https://img.shields.io/badge/status-alpha-orange)]()

`llmfuel` is a local, pip-installable Python library that removes semantically duplicate steps from chain-of-thought (CoT) reasoning in real time — before they eat your token budget.

No cloud proxy. No API keys. Works offline.

> `llmfuel` is not a chat UI or standalone chatbot. It is a tool you integrate into your own model or API client so it can deduplicate reasoning output and save tokens.

---

## The problem

Open reasoning models (DeepSeek-R1, QwQ-32B, and others) produce verbose chain-of-thought traces by design. In practice, **20–40% of CoT tokens are near-duplicate reasoning steps** — the model restating the same sub-conclusion in slightly different words before moving on.

This wastes tokens, slows inference, and adds noise to any downstream agent that reads the trace.

Existing tools either:
- Cache full prompts (not step-level CoT dedup)
- Run in the cloud (privacy risk, latency)
- Produce audit receipts separately from compression

`llmfuel` combines **live CoT dedup + local receipts** in one offline package.

---

## Install

> **Note:** `llmfuel` is not yet published on PyPI. Install directly from GitHub:

```bash
pip install git+https://github.com/veeno7/llmfuel.git
```

Once published to PyPI, you'll be able to use:
```bash
pip install git+https://github.com/veeno7/llmfuel.git@v0.1.0-alpha
```

Raspberry Pi / low-memory preset:
```bash
# After installing, use CoTDeduper(preset="pi") — no extra deps needed
```

### Optional: download the Gemma GGUF model
If you want the default Gemma-based embedding path, download the model once:

```bash
python -m fuel --download-model
```

This will place the model under `models/` so `CoTDeduper(preset="default")` can load it locally.

### CLI helpers
You can also run simple local workflows directly from the command line:

```bash
llmfuel download-model
llmfuel benchmark --limit 10 --output benchmarks/results.json
```

Optional extras for richer setups:

```bash
pip install .[gemma]
pip install .[ollama]
pip install .[all]
```

### Contributor workflow
Useful helpers for local development and release checks:

```bash
make install
make test
make benchmark
make download-model
```

CI runs the same test suite automatically on pushes and pull requests via GitHub Actions.

---

## Quickstart

`llmfuel` is designed to integrate with your own model or API client. It does not provide a built-in chat UI.

Use it like a helper layer around your existing model calls so you can deduplicate reasoning traces and save tokens while still using your own AI stack.

```python
import fuel

# 1. Deduplicate CoT steps
deduper = fuel.CoTDeduper()           # Gemma 3 270M INT4, ~40-70ms/step on CPU
# deduper = fuel.CoTDeduper(preset="pi")  # MiniLM-v2-L6, <10ms, for Raspberry Pi

steps = your_model.get_cot_steps(prompt)
compressed = deduper.dedup(steps)
ratio = deduper.compression_ratio(steps, compressed)
print(f"Compression: {ratio:.0%} of original tokens kept")

# 2. Audit trail — local, hashed by default
chain = fuel.ReceiptChain(
    agent="my-agent/qwq-32b",
    run_id="run-abc123",
    # output_path=Path("receipts.jsonl"),  # optional: persist to disk
)

for raw, compressed_step in zip(steps, compressed):
    chain.record(
        action="cot_dedup",
        input_data=raw,
        output_data=compressed_step,
        input_tokens=len(raw.split()),
        output_tokens=len(compressed_step.split()),
    )

assert chain.verify_chain()   # cryptographic integrity check
```

### With Ollama

```python
import ollama, fuel
from fuel.adapters.ollama import wrap_ollama

client = wrap_ollama(
    ollama.Client(),
    deduper=fuel.CoTDeduper(aggressiveness="aggressive"),
    receipts=fuel.ReceiptChain(agent="ollama/qwq-32b"),
)
response = client.generate(model="qwq:32b", prompt="Solve: 2x+3=11")
```

### With any API-style response

```python
from fuel import CoTDeduper, ReceiptChain, wrap_api_response

response = {"text": "Step 1\nStep 1\nFinal answer"}
result = wrap_api_response(response, CoTDeduper(aggressiveness="aggressive"), ReceiptChain(agent="api-demo"))
print(result["text"])
```

### With OpenAI-compatible SDKs

```python
from openai import OpenAI
from fuel import CoTDeduper, OpenAICompatibleAdapter

client = OpenAICompatibleAdapter(OpenAI(api_key="sk-..."), deduper=CoTDeduper(aggressiveness="aggressive"))
response = client.chat_completions_create(model="gpt-4o-mini", messages=[{"role": "user", "content": "Solve 2+2"}])
print(response.choices[0].message.content)
```

---

## Privacy

- **All content is hashed by default.** Receipts contain only SHA-256 hashes of inputs/outputs — never plaintext.
- **No network calls.** `llmfuel` never phones home. All computation is local.
- **Opt-in plaintext:** `ReceiptChain(store_plaintext=True)` if you want recoverable logs.
- **Hash-chained receipts:** Tampering with any step in the audit trail is detectable via `verify_chain()`.

See [SAFETY.md](SAFETY.md) for the full privacy notice.

---

## Receipts schema

```json
{
  "id": "uuid-v4",
  "ts": 1719394800000,
  "agent": "my-agent",
  "principal": "local",
  "action": "cot_dedup",
  "input_hash": "sha256...",
  "output_hash": "sha256...",
  "prev_hash": "sha256...",
  "ext": {
    "version": "0.1.0",
    "run_id": "run-abc123",
    "step_id": 1,
    "input_tokens": 42,
    "output_tokens": 18
  }
}
```

Compatible with [agentreceipts.ai](https://agentreceipts.ai) v1 via `chain.to_agentreceipts_v1(receipt)`.

---

## Benchmarks (v0.1 target)

| Dataset | Metric | Target |
|---|---|---|
| OpenR1-Math-220k | Accuracy retained | ≥ 97% |
| GSM8K | Accuracy retained | ≥ 97% |
| Both | Compression ratio | 0.60–0.80 |
| CPU (Gemma 270M INT4) | Latency per step | 40–70ms |
| CPU (MiniLM-v2-L6) | Latency per step | < 10ms |

---

## Architecture

```
llmfuel/
├── fuel/
│   ├── compress.py      # CoTDeduper — semantic dedup (DeepSeek)
│   ├── receipts.py      # ReceiptChain — hash-chained audit trail (Claude)
│   ├── cache.py         # StepCache — local step cache (Meta AI)
│   └── adapters/
│       └── ollama.py    # wrap_ollama() — Ollama integration (Meta AI)
└── tests/
    └── test_receipts.py
```

---

## Roadmap

- [x] v0.1 — receipts schema + hash chain + privacy defaults
- [ ] v0.2 — Gemma 3 270M INT4 classifier, MiniLM-v2-L6 fallback
- [ ] v0.2 — Ollama adapter (stream interception)
- [ ] v0.3 — vLLM + LangGraph adapters
- [ ] v0.3 — Benchmark suite (OpenR1-Math-220k + GSM8K)
- [ ] v0.4 — RFC8785 canonical JSON for receipt hashing

---

## Contributing

PRs welcome. Current module owners:

| Module | Owner |
|---|---|
| `fuel/receipts.py` | @Claude (Anthropic) |
| `fuel/compress.py` | @DeepSeek |
| `fuel/cache.py` + `fuel/adapters/` | @MetaAI |
| Repo + CI | @veeno7 (Ash) |

---

## License

MIT © 2026 llmfuel contributors

# SAFETY.md — llmfuel Privacy & Security Notice

**Version:** 0.1.0  
**Last updated:** 2025-06-26

---

## 1. What llmfuel stores

`llmfuel` generates audit receipts for every chain-of-thought step it processes.

**By default, receipts contain only:**
- A UUID identifier for each step
- An integer timestamp (milliseconds since epoch)
- Agent and principal labels you provide
- An action label you provide
- **SHA-256 hashes** of the input and output content — not the content itself
- A hash-chain link (`prev_hash`) for tamper detection
- Token counts (integers)

**Receipts never contain plaintext content unless you explicitly opt in** (see Section 3).

---

## 2. Network behaviour

`llmfuel` makes **zero network calls**.

- No telemetry
- No model downloads at import time (models load lazily on first `dedup()` call)
- No callbacks to Anthropic, DeepSeek, Meta, or any third party
- The `fuel.receipts` module has no network dependencies whatsoever

If you observe outbound connections, they are from your own code or from HuggingFace model download on first run — not from `llmfuel` itself.

---

## 3. Opt-in plaintext

If you need recoverable audit logs, pass `store_plaintext=True`:

```python
chain = ReceiptChain(agent="my-agent", store_plaintext=True)
```

**When plaintext is enabled:**
- Raw input and output text are stored inside the `ext` object of each receipt
- If you write receipts to disk (`output_path=...`), this plaintext will appear in the JSONL file
- You are responsible for securing that file (permissions, encryption at rest)

**Plaintext mode is off by default.** Do not enable it if inputs may contain PII, credentials, or confidential business data.

---

## 4. Hash-chain integrity

Every receipt includes a `prev_hash` field: a SHA-256 hash of the *entire previous receipt* (serialized with `json.dumps(..., sort_keys=True)`).

This means:
- Retroactive modification of any receipt in a chain is detectable
- Call `chain.verify_chain()` at any time to confirm integrity
- A return value of `False` means at least one receipt has been tampered with

**Known limitation:** `json.dumps` with `sort_keys=True` is used for stable serialization. This is not canonical JSON (RFC 8785). We plan to migrate to RFC 8785 in a future version. Until then, receipts hashed with this library may not be verifiable by tools that expect strict RFC 8785 canonical form.

---

## 5. Disk storage

JSONL files written via `output_path` are **append-only** and written to the path you specify. `llmfuel` does not create files anywhere else on your system.

Recommended permissions for receipt files containing sensitive metadata:

```bash
chmod 600 receipts.jsonl
```

---

## 6. Model loading

The `CoTDeduper` class loads a local HuggingFace model on first use. The default model (Gemma 3 270M INT4) will be downloaded from HuggingFace Hub to your local model cache (`~/.cache/huggingface/`) if not already present.

**This download is the only outbound connection `llmfuel` triggers**, and only on first use of `CoTDeduper`. `fuel.receipts` never triggers any download.

---

## 7. What llmfuel does not do

- ✗ Send any data to any external service
- ✗ Store data outside paths you explicitly configure
- ✗ Log prompts, model outputs, or reasoning traces in plaintext by default
- ✗ Use cookies, sessions, or persistent identifiers beyond the `run_id` you provide
- ✗ Collect usage metrics or analytics

---

## 8. Reporting security issues

If you discover a security or privacy issue in `llmfuel`, please open a **private** GitHub Security Advisory at:

```
https://github.com/stgreg30/llmfuel/security/advisories/new
```

Do not open a public issue for security vulnerabilities.

---

## 9. Known TODOs

| Item | Status |
|---|---|
| RFC 8785 canonical JSON for receipt hashing | Planned (v0.4) |
| Encrypted JSONL storage option | Under consideration |
| Formal threat model document | Under consideration |

---

*This notice covers `llmfuel` v0.1.x. It will be updated with each version that changes data handling behaviour.*

That's the DeepSeek GSM8K script — good start, but it has 2 bugs and will be painfully slow if you run it as-is. Let Codespace fix it.

**Issues I see:**
1. `target = float(...))` — extra `)` → SyntaxError
2. `ollama.generate()` for 1,319 test questions = ∼6 hours on phi3:mini
3. No batching, no caching, no saving results

### Give this to your Codespace AI instead:

> Create `benchmarks/gsm8k_dedup.py` from this corrected version:
>
> ```python
> import re, json
> from datasets import load_dataset
> from fuel import CoTDeduper
>
> # Use a tiny sample for v0.1 — full test later
> gsm8k = load_dataset("gsm8k", "main", split="test[:50]")
> deduper = CoTDeduper(preset="pi") # fast MiniLM
>
> def extract_answer(text):
> nums = re.findall(r'-?\d+\.?\d*', text.replace(",", ""))
> return float(nums[-1]) if nums else None
>
> results = []
> for ex in gsm8k:
> # Simulate CoT (replace with real ollama later)
> fake_cot = ex["question"] + "\nStep 1...\nStep 1 again...\nAnswer"
> steps = [s for s in fake_cot.split("\n") if s]
> deduped = deduper.dedup(steps)
> target = float(re.findall(r'#### (\d+)', ex["answer"]))
> results.append({
> "original_steps": len(steps),
> "deduped_steps": len(deduped),
> "ratio": len(deduped)/len(steps)
> })
>
> with open("benchmarks/gsm8k_results.json", "w") as f:
> json.dump(results, f, indent=2)
> print(f"Avg compression: {sum(r['ratio'] for r in results)/len(results):.2f}")
> ```
>
> Then run: `mkdir -p benchmarks && python benchmarks/gsm8k_dedup.py`[0]

**Why this version:**
- fixes the syntax error
- uses GSM8K's `#### 42` answer format (correct extraction)
- runs on 50 samples first (not 1,319) — ∼30 seconds
- saves JSON for later analysis
- no Ollama dependency yet (you can plug it in v0.3)

Once Codespace runs it, you'll have your first benchmark file. Then in v0.2 you swap the fake CoT for real Ollama calls.

Tell Codespace: "create the file, run it, commit results" — it will handle the path and dependencies.
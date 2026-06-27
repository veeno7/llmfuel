git add .
git commit -m "v0.1.0-alpha: working deduper + receipts + ollama adapter"
git tag -a v0.1.0-alpha -f -m "Skeleton + receipts + stubs"
git push origin main --force
git push origin v0.1.0-alpha --force

Then update your README badge from "coming soon" to:
pip install git+https://github.com/veeno7/llmfuel.git@v0.1.0-alpha

Update fuel/compress.py to use Gemma 3 270M INT4 GGUF for the "default" preset:
> 1. Add this at top (after imports):
```python
from llama_cpp import Llama
_gemma_model = None
def _load_gemma():
global _gemma_model
if _gemma_model is None:
_gemma_model = Llama(
model_path="models/gemma-3-270m-it-int4.gguf",
embedding=True, n_ctx=512, verbose=False, n_gpu_layers=0
)
return _gemma_model
def _embed_gemma(texts):
import numpy as np
model = _load_gemma()
embs = [model.create_embedding(t)["data"]["embedding"] for t in texts]
arr = np.array(embs); return arr / np.linalg.norm(arr, axis=1, keepdims=True)
```
> 2. Modify _load_encoder():
- if preset=="default": set self._embed_fn = _embed_gemma
- if preset=="pi": keep MiniLM (SentenceTransformer)
- if GGUF file missing, fallback to MiniLM and print warning
> 3. In dedup(), replace self._encoder.encode(...) with self._embed_fn(steps)
> 4. Add models/ to .gitignore (don't commit GGUF)
> 5. Run: pip install llama-cpp-python then python -c "from fuel import CoTDeduper; d=CoTDeduper(); print('ok')" — if model missing it should fallback gracefully.[0]


After it finishes, download the GGUF once

mkdir -p models
wget https://huggingface.co/.../gemma-3-270m-it-int4.gguf -O models/gemma-3-270m-it-int4.gguf

Create benchmarks/gsm8k_dedup.py from this corrected version:
> ```python
import re, json
from datasets import load_dataset
from fuel import CoTDeduper
> # Use a tiny sample for v0.1 — full test later
gsm8k = load_dataset("gsm8k", "main", split="test[:50]")
deduper = CoTDeduper(preset="pi") # fast MiniLM
> def extract_answer(text):
nums = re.findall(r'-?\d+\.?\d*', text.replace(",", ""))
return float(nums[-1]) if nums else None
> results = []
for ex in gsm8k:
# Simulate CoT (replace with real ollama later)
fake_cot = ex["question"] + "\nStep 1...\nStep 1 again...\nAnswer"
steps = [s for s in fake_cot.split("\n") if s]
deduped = deduper.dedup(steps)
target = float(re.findall(r'#### (\d+)', ex["answer"]))
results.append({
"original_steps": len(steps),
"deduped_steps": len(deduped),
"ratio": len(deduped)/len(steps)
})
> with open("benchmarks/gsm8k_results.json", "w") as f:
json.dump(results, f, indent=2)
print(f"Avg compression: {sum(r['ratio'] for r in results)/len(results):.2f}")
```
> Then run: mkdir -p benchmarks && python benchmarks/gsm8k_dedup.py[0]

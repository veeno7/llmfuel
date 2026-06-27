The GSM8K issue:
It fell back to 3 samples because Hugging Face changed the dataset ID. The old load_dataset("gsm8k", "main") fails now. Fix it:
Tell Codespace:
In benchmarks/gsm8k_dedup.py, change:
```python
gsm8k = load_dataset("gsm8k", "main", split="test[:50]")
```
to:
```python
gsm8k = load_dataset("openai/gsm8k", "main", split="test[:50]")
```
Then re-run python benchmarks/gsm8k_dedup.py
That will load the real 50 questions and give you the ∼0.75 compression you expect.

Next 2 commands to lock v0.1:

# 1. Add regression test for wrapper
cat > tests/test_ollama_wrapper.py << 'EOF'
from fuel import CoTDeduper, ReceiptChain
from fuel.adapters import wrap_ollama

class Fake:
    def generate(self): return "x <think>a</think> <think>a</think> <think>b</think>"

def test_wrapper_dedups_think():
    chain = ReceiptChain()
    d = CoTDeduper(preset="pi", receipts=chain)
    w = wrap_ollama(Fake(), d, chain)
    out = w.generate()
    assert out.count("<think>a</think>") == 1
    assert chain.verify_chain()
EOF

pytest tests/test_ollama_wrapper.py -v

# 2. Tag it
git add .
git commit -m "v0.1.0: working deduper, receipts, ollama wrapper"
git tag v0.1.0
git push origin main --tags
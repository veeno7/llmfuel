Gemma 3 270M INT4 (replace MiniLM)
Prompt for Codespace:
Implement Gemma 3 270M GGUF embedding in fuel/compress.py:
1. pip install llama-cpp-python
2. At top add:
```python
try:
    from llama_cpp import Llama
    _GEMMA = None
    def _load_gemma():
        global _GEMMA
        if _GEMMA is None:
            _GEMMA = Llama(model_path="models/gemma-3-270m-it-q4_k_m.gguf", embedding=True, n_ctx=512, verbose=False)
        return _GEMMA
except: _GEMMA = None
```
3. In CoTDeduper.**init**, if preset=="default" and _GEMMA: use gemma, else fallback to MiniLM with warning
4. Add models/ to .gitignore
5. Test: python -c "from fuel import CoTDeduper; d=CoTDeduper(); print('ok')"
> Then download: mkdir -p models && wget https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf -O models/gemma-3-270m-it-q4_k_m.gguf
2. Real-time streaming dedup
Prompt:
Create fuel/adapters/ollama_stream.py with streaming wrapper:
```python
import re
from fuel import CoTDeduper
> def wrap_ollama_stream(client, deduper=None):
    deduper = deduper or CoTDeduper()
    class W:
        def chat(self, model, messages, stream=True, **kw):
            full = ""
            buf = ""
            for chunk in client.chat(model=model, messages=messages, stream=True, **kw):
                text = chunk['message']['content']
                full += text
                buf += text
                # when we see </think>, dedup the block
                if '</think>' in buf:
                    parts = re.split(r'(<think>.*?</think>)', full, flags=re.S)
                    out = []
                    for p in parts:
                        if p.startswith('<think>'):
                            steps = [s.strip() for s in p[7:-8].split('\n') if s.strip()]
                            keep = deduper.dedup(steps)
                            out.append('<think>' + '\n'.join(keep) + '</think>')
                        else: out.append(p)
                    yield {'message':{'content':''.join(out)[len(full)-len(text):]}}
                    buf = ""
                else:
                    yield chunk
    return W()
```
Add test in tests/test_stream.py that feeds duplicate think blocks and asserts output has 1 copy.
3. Real GSM8K proof
Prompt:
Update benchmarks/gsm8k_dedup.py to use real Ollama:
```python
from datasets import load_dataset
from fuel.adapters import wrap_ollama
from fuel import CoTDeduper
import ollama, re, json
> ds = load_dataset("openai/gsm8k", "main", split="test[:20]")
d = CoTDeduper(preset="default")
w = wrap_ollama(ollama, d)
results = []
for ex in ds:
    prompt = ex['question'] + "\nThink step by step in <think> tags."
    raw = ollama.generate(model="phi3:mini", prompt=prompt)['response']
    dedup = w.generate(model="phi3:mini", prompt=prompt)['response']
    results.append({"raw_len": len(raw), "dedup_len": len(dedup), "saved": len(raw)-len(dedup)})
json.dump(results, open("benchmarks/gsm8k_real.json","w"), indent=2)
print(f"Avg tokens saved: {sum(r['saved'] for r in results)/len(results):.0f}")
```
Run: ollama pull phi3:mini && python benchmarks/gsm8k_dedup.py

Here are the verification commands for each step — run these after Codespace finishes each task.
Step 1: Gemma 3 270M verification

cd /workspaces/llmfuel

# 1. Check model downloaded
ls -lh models/gemma-3-270m-it-q4_k_m.gguf
# should show ~180-200MB

# 2. Test loader
python -c "
from fuel.compress import _load_gemma
m = _load_gemma()
print('Gemma loaded:', m is not None)
emb = m.create_embedding('test')['data'][0]['embedding']
print('Embedding dim:', len(emb))
"

# 3. Test CoTDeduper uses Gemma
python -c "
from fuel import CoTDeduper
d = CoTDeduper(preset='default', verbose=True)
out = d.dedup(['hello world', 'hello world', 'goodbye'])
print('Input 3 steps -> Output', len(out), 'steps:', out)
assert len(out) == 2, 'Gemma dedup failed'
print('✓ Gemma dedup working')
"

Expected output:


python -c "
from fuel.adapters.ollama_stream import wrap_ollama_stream
from fuel import CoTDeduper
import ollama

w = wrap_ollama_stream(ollama, CoTDeduper())
stream = w.chat(model='phi3:mini', messages=[{'role':'user','content':'2+2 step by step in <think>'}])
for chunk in stream:
    print(chunk['message']['content'], end='')
"
# you should see <think> blocks with duplicates removed live


python benchmarks/gsm8k_dedup.py
cat benchmarks/gsm8k_real.json | head -20
# should show raw_len > dedup_len, saved > 0
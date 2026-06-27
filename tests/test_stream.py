from fuel import CoTDeduper, ReceiptChain
from fuel.adapters import wrap_ollama_stream


class FakeClient:
    def chat(self, model, messages, stream=True, **kwargs):
        yield {"message": {"content": "x <think>a</think> <think>a</think> <think>b</think>"}}


def test_stream_wrapper_dedups_think_blocks():
    chain = ReceiptChain()
    deduper = CoTDeduper(preset="pi", receipts=chain)
    wrapped = wrap_ollama_stream(FakeClient(), deduper=deduper)

    chunks = list(wrapped.chat(model="fake", messages=[{"role": "user", "content": "hi"}]))

    assert len(chunks) == 1
    text = chunks[0]["message"]["content"]
    assert text.count("<think>a</think>") == 1
    assert chain.verify_chain()

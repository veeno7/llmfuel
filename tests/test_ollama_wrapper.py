from fuel import CoTDeduper, ReceiptChain
from fuel.adapters import wrap_ollama

class Fake:
    def generate(self):
        return "x <think>a</think> <think>a</think> <think>b</think>"


def test_wrapper_dedups_think():
    chain = ReceiptChain()
    d = CoTDeduper(preset="pi", receipts=chain)
    w = wrap_ollama(Fake(), d, chain)
    out = w.generate()
    assert out.count("<think>a</think>") == 1
    assert chain.verify_chain()

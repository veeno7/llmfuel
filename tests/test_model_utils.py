import sys
import types
from pathlib import Path

from fuel.model_utils import generate_with_model


def test_generate_with_model_uses_chat_completion(monkeypatch, tmp_path):
    class DummyLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def create_chat_completion(self, **kwargs):
            self.chat_kwargs = kwargs
            return {"choices": [{"message": {"content": "ok"}}]}

    dummy_llm = DummyLLM
    module = types.SimpleNamespace(Llama=dummy_llm)
    monkeypatch.setitem(sys.modules, "llama_cpp", module)

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")

    result = generate_with_model(
        prompt="Explain gravity",
        model_path=str(model_path),
        max_tokens=32,
        n_gpu_layers=4,
        n_ctx=1024,
        temperature=0.2,
        top_p=0.9,
        repeat_penalty=1.1,
        n_threads=6,
        n_batch=128,
    )

    assert result["text"] == "ok"
    assert result["model_path"] == str(model_path)

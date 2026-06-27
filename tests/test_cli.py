from pathlib import Path

from fuel.cli import main
from fuel import wrap_api_response, OpenAICompatibleAdapter


def test_wrap_api_response_dedupes_generic_payload():
    class DummyDeduper:
        def __init__(self):
            self.receipts = None

        def dedup(self, steps, return_kept_idx=False):
            kept = [steps[0], steps[2]]
            idx = [0, 2]
            return (kept, idx) if return_kept_idx else kept

    response = {"text": "First reasoning step\nFirst reasoning step\nFinal answer"}
    result = wrap_api_response(response, DummyDeduper())

    assert result["text"] == "First reasoning step\nFinal answer"


def test_openai_compatible_adapter_dedupes_chat_choices():
    class DummyDeduper:
        def dedup(self, steps):
            return [steps[0], steps[2]]

    class DummyChoiceMessage:
        def __init__(self, content):
            self.content = content

    class DummyChoice:
        def __init__(self, content):
            self.message = DummyChoiceMessage(content)

    class DummyResponse:
        def __init__(self, content):
            self.choices = [DummyChoice(content)]

    class DummyClient:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    return DummyResponse("First reasoning step\nFirst reasoning step\nFinal answer")

    adapter = OpenAICompatibleAdapter(DummyClient(), deduper=DummyDeduper())
    result = adapter.chat_completions_create()

    assert result.choices[0].message.content == "First reasoning step\nFinal answer"


def test_generate_command(monkeypatch, capsys, tmp_path):
    called = {}

    def fake_generate(**kwargs):
        called.update(kwargs)
        return {"text": "Generated response"}

    monkeypatch.setattr("fuel.cli.generate_with_model", fake_generate)

    exit_code = main([
        "generate",
        "--prompt",
        "Explain gravity briefly",
        "--model-path",
        str(tmp_path / "model.gguf"),
        "--max-tokens",
        "64",
        "--n-gpu-layers",
        "8",
        "--ctx-size",
        "2048",
        "--verbose",
    ])

    assert exit_code == 0
    assert called["model_path"] == str(tmp_path / "model.gguf")
    assert called["prompt"] == "Explain gravity briefly"
    assert called["max_tokens"] == 64
    assert called["n_gpu_layers"] == 8
    assert called["n_ctx"] == 2048
    assert "Generated response" in capsys.readouterr().out


def test_download_model_command(monkeypatch, capsys, tmp_path):
    called = {}

    def fake_download(model_path=None, force=False):
        called["model_path"] = model_path
        return Path(model_path or "models/test.gguf")

    monkeypatch.setattr("fuel.cli.ensure_gemma_model", fake_download)

    exit_code = main(["download-model", "--model-path", str(tmp_path / "model.gguf")])

    assert exit_code == 0
    assert called["model_path"] == str(tmp_path / "model.gguf")
    assert "Model ready" in capsys.readouterr().out


def test_benchmark_command(monkeypatch, capsys, tmp_path):
    def fake_run_benchmark(limit=20, output_path=None, preset="default", aggressiveness="balanced"):
        return {
            "results": [{"index": 1, "ratio": 0.5}],
            "avg_compression": 0.5,
            "output_path": output_path or str(tmp_path / "results.json"),
        }

    monkeypatch.setattr("fuel.cli.run_benchmark", fake_run_benchmark)

    exit_code = main(["benchmark", "--limit", "2", "--output", str(tmp_path / "out.json")])

    assert exit_code == 0
    assert "Avg compression" in capsys.readouterr().out
